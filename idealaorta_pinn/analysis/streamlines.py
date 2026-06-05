"""Streamline utilities: CFD reference traces and PINN-field integration.

Two complementary views for the figures:
  * ``cfd_streamline_points`` / ``pinn_speed_on_points`` give a like-for-like
    comparison at identical sample points (robust, no integration needed);
  * ``pinn_streamlines`` integrates streamlines through the PINN-predicted
    velocity field on a structured grid (pyvista), seeded at the inlet.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..data.cache import load_points
from ..data.registry import CaseRecord, cases_by_id
from .predict import TrainedModel, predict_physical


def cfd_streamline_points(records: Sequence[CaseRecord], case_id: int, phase: str
                          ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """CFD 3D streamline points: returns (coords ``(N,3)``, speed ``(N,)``)."""
    rec = cases_by_id(records)[case_id]
    df = load_points(rec, "3D", phase)
    if df is None:
        return None
    coords = df[["x", "y", "z"]].to_numpy(float)
    speed = df["speed"].to_numpy(float) if "speed" in df else \
        np.linalg.norm(df[["u", "v", "w"]].to_numpy(float), axis=1)
    return coords, speed


def pinn_speed_on_points(model: TrainedModel, coords: np.ndarray, rec: CaseRecord,
                         phase: str) -> np.ndarray:
    """PINN-predicted speed at the given physical coordinates."""
    beta = rec.beta if rec.beta is not None else 1.0
    return predict_physical(model, coords, rec.inlet_diameter_cm, rec.disease_flag,
                            phase, beta=beta)["speed"]


def pinn_streamlines(model: TrainedModel, records: Sequence[CaseRecord], case_id: int,
                     phase: str, grid_res: int = 60, n_seed: int = 200,
                     mask_frac: float = 0.06):
    """Integrate streamlines through the PINN velocity field (pyvista).

    Builds a structured grid over the case bounding box, evaluates the PINN
    velocity at every node, and traces from a seed disk near the inlet. Returns
    a ``pyvista.PolyData`` of streamlines (or ``None`` if pyvista is unavailable).
    """
    try:
        import pyvista as pv
    except Exception:
        return None

    pts = cfd_streamline_points(records, case_id, phase)
    if pts is None:
        return None
    coords, _ = pts
    rec = cases_by_id(records)[case_id]

    lo, hi = coords.min(axis=0), coords.max(axis=0)
    xs = np.linspace(lo[0], hi[0], grid_res)
    ys = np.linspace(lo[1], hi[1], max(8, grid_res // 2))
    zs = np.linspace(lo[2], hi[2], max(8, grid_res // 2))
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    beta = rec.beta if rec.beta is not None else 1.0
    pred = predict_physical(model, grid_pts, rec.inlet_diameter_cm, rec.disease_flag,
                            phase, beta=beta)
    vel = np.column_stack([pred["u"], pred["v"], pred["w"]])

    # Restrict integration to the lumen: the PINN field is extrapolated outside
    # the data region, so zero the velocity at grid nodes far from the CFD
    # streamline cloud. Streamlines then terminate at the vessel boundary
    # instead of wandering into nonphysical extrapolated flow.
    from scipy.spatial import cKDTree
    diag = float(np.linalg.norm(hi - lo))
    dist, _ = cKDTree(coords).query(grid_pts)
    outside = dist > mask_frac * diag
    vel[outside] = 0.0
    speed = pred["speed"].copy()
    speed[outside] = 0.0

    grid = pv.StructuredGrid(gx, gy, gz)
    grid["velocity"] = vel
    grid["speed"] = speed

    # Seed disk near the inlet (min-X face), within the local cross-section.
    x_inlet = lo[0] + 0.02 * (hi[0] - lo[0])
    near = coords[coords[:, 0] < x_inlet + 0.05 * (hi[0] - lo[0])]
    if len(near) < 3:
        near = coords
    c = near[:, 1:].mean(axis=0)
    r = np.sqrt(((near[:, 1:] - c) ** 2).sum(axis=1)).mean()
    rng = np.random.default_rng(0)
    ang = rng.uniform(0, 2 * np.pi, n_seed)
    rad = r * np.sqrt(rng.uniform(0, 1, n_seed))
    seed = np.column_stack([np.full(n_seed, x_inlet),
                            c[0] + rad * np.cos(ang), c[1] + rad * np.sin(ang)])
    seed_poly = pv.PolyData(seed)

    return grid.streamlines_from_source(
        seed_poly, vectors="velocity", integration_direction="forward",
        max_time=None, max_steps=2000)
