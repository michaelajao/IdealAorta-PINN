"""Assemble parametric training bundles from the cached CFD data.

For each ``(case, phase)`` snapshot this builds standardized tensors for:
  * velocity supervision (dense XY/XZ planes + sparse 3D streamline points),
  * wall supervision (pressure + WSS components + inward normals),
  * interior collocation (a subsample of the velocity points; they are interior
    fluid samples) for the PDE residual,
  * inlet/outlet cross-section points for the soft BCs.

Every point carries the per-case parameter vector ``mu = [d_inlet*, disease, phase]``.
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
from .registry import CaseRecord
from .geometry import compute_wall_normals
from .normalize import Normalizer

VELOCITY_KINDS = ("XY", "XZ", "3D")
DIASTOLIC_TIME = 2.4


def _phase_value(phase: str) -> float:
    return 1.0 if phase == "systolic" else 0.0


def _load_velocity_points(rec: CaseRecord, phase: str,
                          max_points: int, rng: np.random.Generator,
                          kinds: Sequence[str] = VELOCITY_KINDS
                          ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Pool velocity samples (coords, uvw) from the given kinds for one (case, phase)."""
    coords_parts, vel_parts = [], []
    for kind in kinds:
        df = load_points(rec, kind, phase)
        if df is None or not {"u", "v", "w"}.issubset(df.columns):
            continue
        coords_parts.append(df[["x", "y", "z"]].to_numpy(np.float64))
        vel_parts.append(df[["u", "v", "w"]].to_numpy(np.float64))
    if not coords_parts:
        return None
    coords = np.vstack(coords_parts)
    vel = np.vstack(vel_parts)
    if len(coords) > max_points:
        sel = rng.choice(len(coords), size=max_points, replace=False)
        coords, vel = coords[sel], vel[sel]
    return coords, vel


def _load_wall_points(rec: CaseRecord, phase: str
                      ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Wall (coords, pressure, wss components) for one (case, phase)."""
    df = load_points(rec, "WSS", phase)
    if df is None:
        return None
    coords = df[["x", "y", "z"]].to_numpy(np.float64)
    p = df["p"].to_numpy(np.float64).reshape(-1, 1) if "p" in df else np.zeros((len(df), 1))
    wcols = ["wss_x", "wss_y", "wss_z"]
    wss = df[wcols].to_numpy(np.float64) if set(wcols).issubset(df.columns) else np.zeros((len(df), 3))
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
                   per_phase_velocity_scale: bool = False) -> Normalizer:
    """Fit the global Normalizer on the pooled TRAIN-case data.

    With ``per_phase_velocity_scale`` (S1), U_ref is fit on the systolic speeds and
    a separate diastolic velocity scale is stored, so the near-stagnant phase gets
    O(1) standardized targets while the physics scale stays systolic.
    """
    rng = np.random.default_rng(seed)
    by_id = {r.case_id: r for r in records}
    coords_all, speed_all, diam_all, p_all, wss_all = [], [], [], [], []
    speed_sys, speed_dia = [], []
    for cid in case_ids:
        rec = by_id[cid]
        for phase in phases:
            vp = _load_velocity_points(rec, phase, max_points, rng)
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
                 n_collocation: int = 8_000,
                 wall_normals_method: str = "auto",
                 inlet_n_radial: int = 6,
                 inlet_n_angular: int = 12,
                 velocity_kinds: Sequence[str] = VELOCITY_KINDS,
                 volumetric_collocation: bool = False,
                 seed: int = 42) -> Bundle:
    """Build standardized training tensors for the given cases and phases.

    ``velocity_kinds`` selects which velocity sources to supervise on; pass e.g.
    ``("XY", "3D")`` to hold out the XZ plane for validation (Stage A de-risk).
    ``volumetric_collocation`` (S2): replace the data-plane collocation subsample
    with points rejection-sampled inside the 3D lumen, so the PDE residual
    constrains the off-plane interior (needs wall points for the interior mask).
    """
    from .geometry import cross_section_points, detect_inlet_outlet, sample_lumen_interior

    rng = np.random.default_rng(seed)
    by_id = {r.case_id: r for r in records}
    groups: List[GroupTensors] = []

    for cid in case_ids:
        rec = by_id[cid]
        for phase in phases:
            vp = _load_velocity_points(rec, phase, max_velocity_points, rng,
                                       kinds=velocity_kinds)
            wp = _load_wall_points(rec, phase)
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

                for label in ("inlet", "outlet"):
                    pts = cross_section_points(io[f"{label}_axial_pos"], io[f"{label}_center"],
                                               io[f"{label}_radius"], axial,
                                               inlet_n_radial, inlet_n_angular)
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
