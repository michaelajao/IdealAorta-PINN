"""One reconstruction problem: a case, a short time window, an observation grid.

Information contract (what a method may use; everything else is hidden):

    geometry      volume node coordinates (collocation, values unused), wall nodes and
                  inward normals, inlet and outlet faces
    boundary      the prescribed inlet plug velocity and the mean outlet pressure at each
                  window time (both are CFD boundary conditions, not measurements)
    observations  velocity at the nodes nearest a randomly offset uniform grid of spacing
                  ``grid_mm`` (a 4D-flow-like voxel sampling), at every window time, at the
                  same locations; node ids are saved
    closure       optional oracle: the CFD eddy viscosity mu_t at the collocation nodes and
                  its least-squares gradient (arms that use it are labelled "oracle mu_t")

Hidden targets: velocity at every unobserved interior node, the aneurysm-zone wall shear,
and the gauge pressure (outlet mean = 0) on the interior sub-cloud > 1 mm from the wall.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..config import RECON_METRICS_DIR
from ..data.full_export import load_block, wall_normals_from_volume
from .numerics import lsq_gradient

MASK_DIR = RECON_METRICS_DIR / "masks"
VALIDATION_FRACTION = 0.10


def grid_observation_ids(X_int: np.ndarray, grid_mm: float, seed: int) -> np.ndarray:
    """Indices (into X_int) of the interior nodes nearest a random-offset uniform grid.

    A grid point counts as inside the lumen when an interior node lies within half a
    grid spacing; duplicates are removed. Density follows the grid, not the mesh, so the
    refined near-wall prism layers are not oversampled.
    """
    h = grid_mm / 1000.0
    rng = np.random.default_rng(seed)
    lo, hi = X_int.min(0), X_int.max(0)
    off = rng.random(3) * h
    axes = [np.arange(lo[d] + off[d], hi[d], h) for d in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d, j = cKDTree(X_int).query(G, distance_upper_bound=0.5 * h)
    return np.unique(j[np.isfinite(d)])


@dataclass
class Snapshot:
    t_ms: int
    uvw: np.ndarray          # (N_int,3) interior velocity (hidden except at obs ids)
    p: np.ndarray            # (N_int,) absolute pressure (hidden)
    mut: np.ndarray          # (N_int,) dynamic eddy viscosity (oracle closure)
    grad_mut: np.ndarray     # (N_int,3)
    wall_wss: np.ndarray     # (N_wall,3) hidden
    inlet_uvw: np.ndarray    # (N_in,3) boundary condition
    outlet_p_mean: float     # boundary condition (gauge datum)


@dataclass
class Problem:
    case: int
    times_ms: List[int]
    target_ms: int
    grid_mm: float
    seed: int
    X: np.ndarray                     # interior node coordinates
    dwall: np.ndarray                 # distance to the nearest wall node
    sac: np.ndarray                   # interior nodes whose nearest wall node is aneurysm zone
    obs_ids: np.ndarray               # observed interior node ids (same at every time)
    wall_xyz: np.ndarray
    wall_normals: np.ndarray
    wall_aneurysm: np.ndarray
    inlet_xyz: np.ndarray
    outlet_xyz: np.ndarray
    snaps: Dict[int, Snapshot] = field(default_factory=dict)

    @property
    def hidden_ids(self) -> np.ndarray:
        m = np.ones(len(self.X), bool)
        m[self.obs_ids] = False
        return np.flatnonzero(m)

    def info(self) -> Dict:
        return {"case": self.case, "times_ms": self.times_ms, "target_ms": self.target_ms,
                "grid_mm": self.grid_mm, "seed": self.seed, "n_interior": int(len(self.X)),
                "n_obs_per_time": int(len(self.obs_ids)),
                "n_obs_total": int(len(self.obs_ids) * len(self.times_ms)),
                "obs_fraction": float(len(self.obs_ids) / len(self.X)),
                "n_wall": int(len(self.wall_xyz)), "n_aneurysm_wall": int(self.wall_aneurysm.sum())}


def validation_split(obs_ids: np.ndarray, seed: int) -> np.ndarray:
    """Boolean mask over ``obs_ids`` holding out 10 % of the observations.

    Every method that selects anything (the neural field's physics weight, the tuned RBF's
    kernel) uses this same split and nothing else; hidden targets are never consulted.
    """
    return np.random.default_rng(1000 + seed).random(len(obs_ids)) < VALIDATION_FRACTION


def build_problem(case: int, times_ms: List[int], target_ms: int, grid_mm: float, seed: int,
                  save_mask: bool = True, min_wall_mm: float = 0.0) -> Problem:
    """Load one (case, window) and draw its observation grid (``seed`` sets the offset).

    ``min_wall_mm`` > 0 drops observations closer than that to the wall (sensitivity to the
    near-wall samples that a voxel of the grid size could not resolve).
    """
    base = load_block(case, "solid", times_ms[0], ["x", "y", "z", "u", "v", "w"])
    Xall = base[["x", "y", "z"]].to_numpy(np.float64)
    inside = np.linalg.norm(base[["u", "v", "w"]].to_numpy(np.float64), axis=1) > 0
    X = Xall[inside]
    walls = [(b, d) for b in ("aneurysm", "wall") if (d := load_block(case, b, times_ms[0])) is not None]
    wdf = pd.concat([d.assign(_an=(b == "aneurysm")) for b, d in walls], ignore_index=True)
    wxyz = wdf[["x", "y", "z"]].to_numpy(np.float64)
    _, first = np.unique(np.round(wxyz, 9), axis=0, return_index=True)
    first = np.sort(first)
    wxyz, wan = wxyz[first], wdf["_an"].to_numpy(bool)[first]
    dwall, jw = cKDTree(wxyz).query(X)
    normals = wall_normals_from_volume(wxyz, X)
    inl = load_block(case, "inlet", times_ms[0])
    moving = np.linalg.norm(inl[["u", "v", "w"]].to_numpy(np.float64), axis=1) > 0
    out = load_block(case, "outlet", times_ms[0])

    obs = grid_observation_ids(X, grid_mm, seed)
    if min_wall_mm > 0:
        obs = obs[dwall[obs] >= min_wall_mm / 1000.0]
    prob = Problem(case=case, times_ms=list(times_ms), target_ms=target_ms, grid_mm=grid_mm, seed=seed,
                   X=X, dwall=dwall, sac=wan[jw], obs_ids=obs, wall_xyz=wxyz, wall_normals=normals,
                   wall_aneurysm=wan, inlet_xyz=inl[["x", "y", "z"]].to_numpy(np.float64)[moving],
                   outlet_xyz=out[["x", "y", "z"]].to_numpy(np.float64))
    for t in times_ms:
        v = load_block(case, "solid", t)
        mut = v["mu_t"].to_numpy(np.float64)
        w = pd.concat([load_block(case, b, t) for b, _ in walls], ignore_index=True).iloc[first]
        il, ol = load_block(case, "inlet", t), load_block(case, "outlet", t)
        prob.snaps[t] = Snapshot(
            t_ms=t, uvw=v[["u", "v", "w"]].to_numpy(np.float64)[inside],
            p=v["p"].to_numpy(np.float64)[inside], mut=mut[inside],
            grad_mut=lsq_gradient(Xall, mut)[inside],
            wall_wss=w[["wss_x", "wss_y", "wss_z"]].to_numpy(np.float64),
            inlet_uvw=il[["u", "v", "w"]].to_numpy(np.float64)[moving],
            outlet_p_mean=float(ol["p"].mean()))
    if save_mask:
        MASK_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"case{case:02d}_t{'-'.join(map(str, times_ms))}_grid{grid_mm:g}_s{seed}"
        stem += f"_mw{min_wall_mm:g}" if min_wall_mm > 0 else ""
        np.save(MASK_DIR / f"{stem}_obs_ids.npy", obs)
        (MASK_DIR / f"{stem}.json").write_text(json.dumps(prob.info(), indent=1))
    return prob
