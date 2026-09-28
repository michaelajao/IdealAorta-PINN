"""Assemble parametric training bundles from the cached CFD data.

For each ``(case, phase)`` snapshot this builds standardized tensors for:
    * velocity supervision (3D streamline points by default; XY/XZ are validation/figure slices
        unless explicitly requested by a diagnostic config),
  * wall supervision (pressure + WSS components + inward normals),
  * interior collocation (a subsample of the velocity points; they are interior
    fluid samples) for the PDE residual,
  * inlet/outlet cross-section points for the soft BCs.

Every point carries the per-case parameter vector ``mu = [d_inlet*, beta, disease, phase]``.
The global :class:`Normalizer` is fit on the TRAIN cases only and reused for any
held-out case (no leakage in leave-one-diameter-out).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import mean_inlet_velocity
from .cache import load_points
from .full_export import SPLIT_TRAIN, SPLIT_VAL, load_full_snapshot, snapshot_time_ms
from .registry import CaseRecord
from .geometry import compute_wall_normals
from .normalize import Normalizer

VELOCITY_KINDS = ("3D",)
SLICE_VELOCITY_KINDS = ("XY", "XZ")
DIASTOLIC_TIME = 2.4


def _phase_value(phase: str) -> float:
    return 1.0 if phase == "systolic" else 0.0


SLICE_PLANE_AXES = {"XY": (0, 1), "XZ": (0, 2)}


def slice_tile_mask(coords: np.ndarray, kind: str, tile_mm: float, parity: int) -> np.ndarray:
    """Checkerboard mask over a slice's in-plane axes: True where the tile parity matches.

    Splits an XY/XZ slice into ``tile_mm`` squares and keeps alternate ones, so a
    diagnostic can supervise on one colour and score on the other. Every held-out
    point then sits within about one tile of supervised data, which is roughly the
    spacing a volumetric export of the 0.8 mm CFD mesh would give everywhere.
    """
    a, b = SLICE_PLANE_AXES[kind]
    t = tile_mm / 1000.0
    ia = np.floor(coords[:, a] / t).astype(np.int64)
    ib = np.floor(coords[:, b] / t).astype(np.int64)
    return ((ia + ib) % 2) == int(parity)


def _load_velocity_points(rec: CaseRecord, phase: str,
                          max_points: int, rng: np.random.Generator,
                          kinds: Sequence[str] = VELOCITY_KINDS,
                          slice_tiles: Optional[Dict] = None,
                          kind_max_points: Optional[Dict[str, int]] = None,
                          ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Pool velocity samples (coords, uvw) from the given kinds for one (case, phase).

    ``slice_tiles`` (``{"tile_mm": 2.0, "parity": 0}``) keeps only one checkerboard
    colour of each XY/XZ slice; ``kind_max_points`` caps each kind before pooling, so
    a sparse source (the 3D streamlines) is not crowded out by a dense slice when the
    pool is subsampled to ``max_points``. Both default to off, as every reported run was.
    """
    coords_parts, vel_parts = [], []
    for kind in kinds:
        df = load_points(rec, kind, phase)
        if df is None or not {"u", "v", "w"}.issubset(df.columns):
            continue
        uvw = df[["u", "v", "w"]].to_numpy(np.float64)
        # Skip a degenerate slice: some CFD plane exports carry no resolved flow
        # (e.g. Case 3 systolic XY/XZ are all-zero). Supervising velocity toward
        # those zeros corrupts the fit, so drop the slice (its 3D export is used).
        if float(np.abs(uvw).max()) < 1e-8:
            print(f"[loaders] skip zero-velocity slice: case {rec.case_id} {phase} {kind}")
            continue
        coords = df[["x", "y", "z"]].to_numpy(np.float64)
        if slice_tiles and kind in SLICE_PLANE_AXES:
            keep = slice_tile_mask(coords, kind, float(slice_tiles["tile_mm"]),
                                   int(slice_tiles.get("parity", 0)))
            coords, uvw = coords[keep], uvw[keep]
        cap = (kind_max_points or {}).get(kind)
        if cap is not None and len(coords) > int(cap):
            sel = rng.choice(len(coords), size=int(cap), replace=False)
            coords, uvw = coords[sel], uvw[sel]
        coords_parts.append(coords)
        vel_parts.append(uvw)
    if not coords_parts:
        return None
    coords = np.vstack(coords_parts)
    vel = np.vstack(vel_parts)
    if len(coords) > max_points:
        sel = rng.choice(len(coords), size=max_points, replace=False)
        coords, vel = coords[sel], vel[sel]
    return coords, vel


def _load_wall_points(rec: CaseRecord, phase: str,
                      max_points: Optional[int] = None,
                      rng: Optional[np.random.Generator] = None
                      ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Wall (coords, pressure, wss components) for one (case, phase).

    ``max_points`` caps the wall cloud per group. This matters more than it looks:
    the trainer concatenates every group's wall points into ONE batch and pushes it
    through nine ``create_graph=True`` autograd calls for the WSS/no-slip term, so
    wall memory scales with (points/group x groups) and is what actually bounds a
    many-group fit -- ``max_velocity_points`` does not touch it. Default ``None``
    keeps the full cloud, which is what every existing run was trained with.
    """
    df = load_points(rec, "WSS", phase)
    if df is None:
        return None
    coords = df[["x", "y", "z"]].to_numpy(np.float64)
    p = df["p"].to_numpy(np.float64).reshape(-1, 1) if "p" in df else np.zeros((len(df), 1))
    wcols = ["wss_x", "wss_y", "wss_z"]
    wss = df[wcols].to_numpy(np.float64) if set(wcols).issubset(df.columns) else np.zeros((len(df), 3))
    if max_points is not None and len(coords) > max_points:
        sel = (rng or np.random.default_rng(0)).choice(len(coords), size=max_points,
                                                       replace=False)
        coords, p, wss = coords[sel], p[sel], wss[sel]
    return coords, p, wss


@dataclass
class GroupTensors:
    """Standardized tensors for one (case, phase) snapshot."""
    case_id: int
    phase: str
    diameter_cm: float
    disease_flag: int
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class Bundle:
    groups: List[GroupTensors]
    normalizer: Normalizer
    n_param: int
    device: str


def _t(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(np.asarray(arr, dtype=np.float32), device=device)


# Parameter vector mu = [d_inlet*, beta, disease_flag, phase]. Healthy cases have
# no defined asymmetry; beta defaults to 1.0 (symmetric) and the disease flag
# already separates them from the diseased family.
N_PARAM = 4


def _beta_value(rec: CaseRecord) -> float:
    return float(rec.beta) if rec.beta is not None else 1.0


def _params_array(norm: Normalizer, diameter_cm: float, beta: float, disease_flag: int,
                  phase: str, n: int) -> np.ndarray:
    mu = np.array([norm.diameter_nd(diameter_cm), float(beta), float(disease_flag),
                   _phase_value(phase)], dtype=np.float64)
    return np.tile(mu, (n, 1))


def fit_normalizer(records: Sequence[CaseRecord], case_ids: Sequence[int],
                   phases: Sequence[str], mu: float, rho: float,
                   max_points: int = 60_000, seed: int = 42,
                   per_phase_velocity_scale: bool = False,
                   velocity_kinds: Sequence[str] = VELOCITY_KINDS,
                   slice_tiles: Optional[Dict] = None,
                   kind_max_points: Optional[Dict[str, int]] = None,
                   full_export: Optional[Dict] = None) -> Normalizer:
    """Fit the global Normalizer on the pooled TRAIN-case data.

    With ``per_phase_velocity_scale`` (S1), U_ref is fit on the systolic speeds and
    a separate diastolic velocity scale is stored, so the near-stagnant phase gets
    O(1) standardized targets while the physics scale stays systolic. With
    ``full_export`` the scales come from the training cubes of the fluid volume and
    the whole wall instead of the streamline and aneurysm-clip exports.
    """
    rng = np.random.default_rng(seed)
    by_id = {r.case_id: r for r in records}
    coords_all, speed_all, diam_all, p_all, wss_all = [], [], [], [], []
    speed_sys, speed_dia = [], []
    for cid in case_ids:
        rec = by_id[cid]
        for phase in phases:
            if full_export:
                snap = _full_snapshot(rec, phase, full_export, rho)
                tr = np.flatnonzero(_train_mask(snap, full_export))
                sel = rng.choice(tr, size=min(max_points, len(tr)), replace=False)
                spd = np.linalg.norm(snap["vol_uvw"][sel], axis=1)
                coords_all.append(snap["vol_xyz"][sel])
                speed_all.append(spd)
                diam_all.append(rec.inlet_diameter_cm)
                (speed_sys if phase == "systolic" else speed_dia).append(spd)
                p_all.append(snap["wall_p"])
                wss_all.append(snap["wall_wss"])
                continue
            vp = _load_velocity_points(rec, phase, max_points, rng, kinds=velocity_kinds,
                                       slice_tiles=slice_tiles,
                                       kind_max_points=kind_max_points)
            if vp is not None:
                coords, vel = vp
                spd = np.linalg.norm(vel, axis=1)
                coords_all.append(coords)
                speed_all.append(spd)
                diam_all.append(rec.inlet_diameter_cm)
                (speed_sys if phase == "systolic" else speed_dia).append(spd)
            wp = _load_wall_points(rec, phase)
            if wp is not None:
                _, p, wss = wp
                p_all.append(p.ravel())
                wss_all.append(wss)
    norm = Normalizer(mu=mu, rho=rho)
    use_phase = per_phase_velocity_scale and speed_sys and speed_dia
    norm.fit(
        coords=np.vstack(coords_all),
        speed=np.concatenate(speed_all),
        diameters_cm=np.array(diam_all),
        pressure=np.concatenate(p_all) if p_all else None,
        wss_components=np.vstack(wss_all) if wss_all else None,
        speed_systolic=np.concatenate(speed_sys) if use_phase else None,
        speed_diastolic=np.concatenate(speed_dia) if use_phase else None,
    )
    return norm


def build_bundle(records: Sequence[CaseRecord],
                 case_ids: Sequence[int],
                 phases: Sequence[str],
                 normalizer: Normalizer,
                 device: str = "cuda",
                 max_velocity_points: int = 40_000,
                 max_wall_points: Optional[int] = None,
                 n_collocation: int = 8_000,
                 wall_normals_method: str = "auto",
                 inlet_n_radial: int = 6,
                 inlet_n_angular: int = 12,
                 velocity_kinds: Sequence[str] = VELOCITY_KINDS,
                 volumetric_collocation: bool = False,
                 slice_tiles: Optional[Dict] = None,
                 kind_max_points: Optional[Dict[str, int]] = None,
                 inlet_face: Optional[Dict] = None,
                 outlet_section: bool = True,
                 full_export: Optional[Dict] = None,
                 seed: int = 42) -> Bundle:
    """Build standardized training tensors for the given cases and phases.

    ``velocity_kinds`` selects which velocity sources to supervise on. The default
    is ``("3D",)`` because XY/XZ exports are slice representations reserved for
    validation and figures. Pass XY/XZ only for legacy/diagnostic ablations.
    ``volumetric_collocation`` (S2): replace the data-plane collocation subsample
    with points rejection-sampled inside the 3D lumen, so the PDE residual
    constrains the off-plane interior (needs wall points for the interior mask).

    Where the soft inlet/outlet conditions sit: by default both are the axial ends
    of the WALL cloud (``detect_inlet_outlet``). The wall export in this dataset is
    an iso-clip of the aneurysm segment only, so those "ends" are the proximal and
    distal necks, not the vessel's inlet and outlet. ``inlet_face`` instead places the
    inlet disk on the true inlet plane in physical coordinates
    (``{"axis": 0, "pos_m": 0.0, "center_m": [0.0, 0.0], "direction": -1}``, radius
    from the case's inlet diameter, velocity sign from ``direction``), which is where
    the CFD's uniform-velocity inlet actually is; ``outlet_section=False`` drops the
    zero-gauge-pressure disk, whose datum otherwise sits on the distal neck.

    ``full_export`` (see :mod:`.full_export`) replaces all of the above with the
    whole-vessel exports: velocity and eddy-viscosity targets from the training cubes
    of the fluid volume, collocation on the volume nodes, the whole wall with normals
    oriented by the volume, the CFD inlet face and the outlet face with its pressure.
    Each group then keeps point pools and the trainer redraws its batch from them
    every ``resample_interval`` epochs (:func:`resample_group`).
    """
    from .geometry import cross_section_points, detect_inlet_outlet, sample_lumen_interior

    rng = np.random.default_rng(seed)
    by_id = {r.case_id: r for r in records}
    groups: List[GroupTensors] = []

    if full_export:
        for cid in case_ids:
            for phase in phases:
                g = _build_full_group(by_id[cid], phase, normalizer, full_export, device, rng)
                resample_group(g, rng, max_velocity_points, n_collocation, max_wall_points)
                groups.append(g)
        return Bundle(groups=groups, normalizer=normalizer, n_param=N_PARAM, device=device)

    for cid in case_ids:
        rec = by_id[cid]
        for phase in phases:
            vp = _load_velocity_points(rec, phase, max_velocity_points, rng,
                                       kinds=velocity_kinds, slice_tiles=slice_tiles,
                                       kind_max_points=kind_max_points)
            wp = _load_wall_points(rec, phase, max_wall_points, rng)
            if vp is None and wp is None:
                continue

            g = GroupTensors(case_id=cid, phase=phase,
                             diameter_cm=rec.inlet_diameter_cm,
                             disease_flag=rec.disease_flag)
            t: Dict[str, torch.Tensor] = g.tensors

            # ---- velocity supervision + collocation ----
            if vp is not None:
                coords, vel = vp
                cs = normalizer.coords_std(coords)
                vnd = normalizer.vel_nd(vel)
                params = _params_array(normalizer, rec.inlet_diameter_cm, _beta_value(rec),
                                       rec.disease_flag, phase, len(cs))
                t["vx"] = _t(cs[:, 0:1], device)
                t["vy"] = _t(cs[:, 1:2], device)
                t["vz"] = _t(cs[:, 2:3], device)
                t["v_params"] = _t(params, device)
                t["u_t"] = _t(vnd[:, 0:1], device)
                t["v_t"] = _t(vnd[:, 1:2], device)
                t["w_t"] = _t(vnd[:, 2:3], device)

                n_coll = min(n_collocation, len(cs))
                csel = rng.choice(len(cs), size=n_coll, replace=False)
                t["cx"] = _t(cs[csel, 0:1], device)
                t["cy"] = _t(cs[csel, 1:2], device)
                t["cz"] = _t(cs[csel, 2:3], device)
                t["c_params"] = _t(params[csel], device)

            # ---- wall supervision (pressure, WSS, no-slip) ----
            if wp is not None:
                wcoords, p, wss = wp
                wcs = normalizer.coords_std(wcoords)
                normals = compute_wall_normals(wcs, method=wall_normals_method)
                wparams = _params_array(normalizer, rec.inlet_diameter_cm, _beta_value(rec),
                                        rec.disease_flag, phase, len(wcs))
                t["wx"] = _t(wcs[:, 0:1], device)
                t["wy"] = _t(wcs[:, 1:2], device)
                t["wz"] = _t(wcs[:, 2:3], device)
                t["w_params"] = _t(wparams, device)
                t["p_t"] = _t(normalizer.pressure_std(p), device)
                t["wss_t"] = _t(normalizer.wss_std_target(wss), device)
                t["normals"] = _t(normals, device)

                # ---- S2: volumetric interior collocation ----
                # Override the data-plane subsample (cx/cy/cz from the velocity block,
                # which lives ~98% on two CFD planes) with points that fill the 3D
                # lumen, so the PDE residual constrains the off-plane interior + bulge
                # core. Falls back to the data subsample if the interior mask is thin.
                if volumetric_collocation:
                    coll = sample_lumen_interior(wcs, normals, n_collocation, rng)
                    if len(coll) >= max(int(0.5 * n_collocation), 100):
                        cparams = _params_array(normalizer, rec.inlet_diameter_cm,
                                                _beta_value(rec), rec.disease_flag, phase, len(coll))
                        t["cx"] = _t(coll[:, 0:1], device)
                        t["cy"] = _t(coll[:, 1:2], device)
                        t["cz"] = _t(coll[:, 2:3], device)
                        t["c_params"] = _t(cparams, device)

                # ---- inlet / outlet BC points (from wall geometry) ----
                io = detect_inlet_outlet(wcs)
                axial = io["axial_dim"]
                g.meta["axial_dim"] = axial
                t_time = (rec.files.get("WSS", {}).get(phase, {}).get("time_s")
                          if phase == "systolic" else DIASTOLIC_TIME)
                t_time = t_time or (2.4 if phase == "diastolic" else 1.8)
                u_inlet = mean_inlet_velocity(rec.inlet_diameter_cm, float(t_time))
                g.meta["u_inlet_nd"] = u_inlet / normalizer.U_ref

                sections = {"inlet": None, "outlet": None} if outlet_section else {"inlet": None}
                if inlet_face:
                    ax_f = int(inlet_face.get("axis", 0))
                    disk = cross_section_points(float(inlet_face["pos_m"]),
                                                inlet_face.get("center_m", [0.0, 0.0]),
                                                rec.inlet_diameter_cm / 200.0, ax_f,
                                                inlet_n_radial, inlet_n_angular)
                    sections["inlet"] = normalizer.coords_std(disk)
                    g.meta["axial_dim"] = ax_f
                    g.meta["u_inlet_nd"] = (float(inlet_face.get("direction", 1))
                                            * u_inlet / normalizer.U_ref)
                for label, fixed in sections.items():
                    pts = fixed if fixed is not None else cross_section_points(
                        io[f"{label}_axial_pos"], io[f"{label}_center"],
                        io[f"{label}_radius"], axial, inlet_n_radial, inlet_n_angular)
                    pr = _params_array(normalizer, rec.inlet_diameter_cm, _beta_value(rec),
                                       rec.disease_flag, phase, len(pts))
                    t[f"{label}_x"] = _t(pts[:, 0:1], device)
                    t[f"{label}_y"] = _t(pts[:, 1:2], device)
                    t[f"{label}_z"] = _t(pts[:, 2:3], device)
                    t[f"{label}_params"] = _t(pr, device)

            groups.append(g)

    return Bundle(groups=groups, normalizer=normalizer, n_param=N_PARAM, device=device)


def load_holdout_velocity(records: Sequence[CaseRecord], case_id: int, phase: str,
                          kind: str, normalizer: Normalizer, device: str = "cuda",
                          max_points: int = 50_000, seed: int = 7
                          ) -> Optional[Dict[str, torch.Tensor]]:
    """Load a held-out velocity slice (e.g. 'XZ') as standardized tensors for validation."""
    rng = np.random.default_rng(seed)
    rec = {r.case_id: r for r in records}[case_id]
    vp = _load_velocity_points(rec, phase, max_points, rng, kinds=(kind,))
    if vp is None:
        return None
    coords, vel = vp
    cs = normalizer.coords_std(coords)
    vnd = normalizer.vel_nd(vel)
    params = _params_array(normalizer, rec.inlet_diameter_cm, _beta_value(rec), rec.disease_flag, phase, len(cs))
    return {
        "x": _t(cs[:, 0:1], device), "y": _t(cs[:, 1:2], device), "z": _t(cs[:, 2:3], device),
        "params": _t(params, device),
        "u_t": _t(vnd[:, 0:1], device), "v_t": _t(vnd[:, 1:2], device), "w_t": _t(vnd[:, 2:3], device),
    }


# ---------------------------------------------------------------------------
# Whole-vessel exports (full_export): point pools, per-epoch-block resampling
# ---------------------------------------------------------------------------
# pool name -> {pool array: group tensor key}; resample_group copies a random subset
# of each pool into the tensor keys the trainer's batched losses already read.
_POOL_KEYS = {
    "vel": {"x": "vx", "y": "vy", "z": "vz", "params": "v_params",
            "u_t": "u_t", "v_t": "v_t", "w_t": "w_t", "nut_t": "nut_t"},
    "coll": {"x": "cx", "y": "cy", "z": "cz", "params": "c_params"},
    "wall": {"x": "wx", "y": "wy", "z": "wz", "params": "w_params",
             "p_t": "p_t", "wss_t": "wss_t", "normals": "normals"},
}


def _full_snapshot(rec: CaseRecord, phase: str, full_export: Dict, rho: float) -> Dict[str, np.ndarray]:
    return load_full_snapshot(rec.case_id, snapshot_time_ms(rec, phase, full_export.get("times_ms", {})),
                              float(full_export.get("block_mm", 2.0)),
                              int(full_export.get("val_every", 5)), float(rho))


def _train_mask(snap: Dict[str, np.ndarray], full_export: Dict) -> np.ndarray:
    """Volume nodes a group may supervise on. In-sample tests train on the even cubes
    only; a leave-one-diameter-out fold scores other cases, so its training cases can
    also use the test cubes (``train_splits: [0, 1]``). The validation cubes stay out."""
    return np.isin(snap["vol_split"], full_export.get("train_splits", [SPLIT_TRAIN]))


def _build_full_group(rec: CaseRecord, phase: str, normalizer: Normalizer, full_export: Dict,
                      device: str, rng: np.random.Generator) -> GroupTensors:
    """One (case, phase) group from the whole-vessel exports, with its point pools.

    Pools: the training cubes of the volume (velocity, and nu_t in the nut network's
    units nu_t / (U_ref L)), every interior node for collocation (positions only, so
    the withheld cubes lend no values), and the whole wall. The inlet face keeps the
    CFD's per-node velocity and the outlet face its pressure, both in full.
    """
    snap = _full_snapshot(rec, phase, full_export, normalizer.rho)
    g = GroupTensors(case_id=rec.case_id, phase=phase, diameter_cm=rec.inlet_diameter_cm,
                     disease_flag=rec.disease_flag)
    g.meta["t_ms"] = snapshot_time_ms(rec, phase, full_export.get("times_ms", {}))
    beta = _beta_value(rec)

    def params(n):
        return _params_array(normalizer, rec.inlet_diameter_cm, beta, rec.disease_flag, phase, n)

    def capped(idx, cap):
        return rng.choice(idx, size=int(cap), replace=False) if len(idx) > int(cap) else idx

    tr = capped(np.flatnonzero(_train_mask(snap, full_export)), full_export.get("velocity_pool", 250_000))
    cs = normalizer.coords_std(snap["vol_xyz"][tr])
    vnd = normalizer.vel_nd(snap["vol_uvw"][tr])
    nut = snap["vol_nut"][tr] / (normalizer.U_ref * normalizer.L)
    ci = capped(np.arange(len(snap["vol_xyz"])), full_export.get("collocation_pool", 300_000))
    ccs = normalizer.coords_std(snap["vol_xyz"][ci])
    wcs = normalizer.coords_std(snap["wall_xyz"])
    pools = {
        "vel": {"x": cs[:, 0:1], "y": cs[:, 1:2], "z": cs[:, 2:3], "params": params(len(cs)),
                "u_t": vnd[:, 0:1], "v_t": vnd[:, 1:2], "w_t": vnd[:, 2:3], "nut_t": nut[:, None]},
        "coll": {"x": ccs[:, 0:1], "y": ccs[:, 1:2], "z": ccs[:, 2:3], "params": params(len(ccs))},
        "wall": {"x": wcs[:, 0:1], "y": wcs[:, 1:2], "z": wcs[:, 2:3], "params": params(len(wcs)),
                 "p_t": normalizer.pressure_std(snap["wall_p"])[:, None],
                 "wss_t": normalizer.wss_std_target(snap["wall_wss"]),
                 "normals": snap["wall_normals"]},
    }
    g.meta["pools"] = {k: {n: _t(a, device) for n, a in d.items()} for k, d in pools.items()}

    t = g.tensors
    ics = normalizer.coords_std(snap["inlet_xyz"])
    ivn = normalizer.vel_nd(snap["inlet_uvw"])
    t["inlet_x"], t["inlet_y"], t["inlet_z"] = (_t(ics[:, i:i + 1], device) for i in range(3))
    t["inlet_params"] = _t(params(len(ics)), device)
    t["inlet_ut"], t["inlet_vt"], t["inlet_wt"] = (_t(ivn[:, i:i + 1], device) for i in range(3))
    g.meta["axial_dim"] = 0
    g.meta["u_inlet_nd"] = float(ivn[:, 0].mean())
    ocs = normalizer.coords_std(snap["outlet_xyz"])
    t["outlet_x"], t["outlet_y"], t["outlet_z"] = (_t(ocs[:, i:i + 1], device) for i in range(3))
    t["outlet_params"] = _t(params(len(ocs)), device)
    t["outlet_p_t"] = _t(normalizer.pressure_std(snap["outlet_p"])[:, None], device)
    print(f"[loaders] full export case {rec.case_id} {phase} t={g.meta['t_ms']} ms: "
          f"{len(cs)} velocity / {len(ccs)} collocation / {len(wcs)} wall pool points, "
          f"{len(ics)} inlet, {len(ocs)} outlet")
    return g


def resample_group(g: GroupTensors, rng: np.random.Generator, n_vel: Optional[int],
                   n_coll: Optional[int], n_wall: Optional[int]) -> None:
    """Draw a fresh batch from each of a full-export group's pools (None = whole pool)."""
    sizes = {"vel": n_vel, "coll": n_coll, "wall": n_wall}
    for name, keys in _POOL_KEYS.items():
        pool = g.meta["pools"][name]
        first = next(iter(pool.values()))
        n = first.shape[0]
        k = n if sizes[name] is None else min(int(sizes[name]), n)
        idx = torch.as_tensor(rng.choice(n, size=k, replace=False), device=first.device)
        for src, dst in keys.items():
            g.tensors[dst] = pool[src].index_select(0, idx)


def load_full_holdout(records: Sequence[CaseRecord], case_ids: Sequence[int],
                      phases: Sequence[str], normalizer: Normalizer, full_export: Dict,
                      device: str = "cuda", max_points: int = 10_000,
                      seed: int = 7) -> List[Dict[str, torch.Tensor]]:
    """Monitoring set per (case, phase): nodes of the withheld validation cubes."""
    rng = np.random.default_rng(seed)
    by_id = {r.case_id: r for r in records}
    out = []
    for cid in case_ids:
        rec = by_id[cid]
        for phase in phases:
            snap = _full_snapshot(rec, phase, full_export, normalizer.rho)
            idx = np.flatnonzero(snap["vol_split"] == SPLIT_VAL)
            if len(idx) > max_points:
                idx = rng.choice(idx, size=max_points, replace=False)
            cs = normalizer.coords_std(snap["vol_xyz"][idx])
            vnd = normalizer.vel_nd(snap["vol_uvw"][idx])
            params = _params_array(normalizer, rec.inlet_diameter_cm, _beta_value(rec),
                                   rec.disease_flag, phase, len(cs))
            out.append({"x": _t(cs[:, 0:1], device), "y": _t(cs[:, 1:2], device),
                        "z": _t(cs[:, 2:3], device), "params": _t(params, device),
                        "u_t": _t(vnd[:, 0:1], device), "v_t": _t(vnd[:, 1:2], device),
                        "w_t": _t(vnd[:, 2:3], device)})
    return out
