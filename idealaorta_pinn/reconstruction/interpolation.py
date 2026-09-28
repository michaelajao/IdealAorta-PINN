"""Conventional comparators: interpolate the observations to every interior node.

Each method sees exactly the neural field's information: the observed velocity at both
window times, zero velocity on the wall nodes and the prescribed inlet plug. Each returns
``({t_ms: (N_int, 3) velocity}, meta)`` for ``postprocess.score_field``.

    linear          Delaunay linear interpolation per snapshot (nearest sample outside
                    the hull of the samples)
    rbf             local thin-plate RBF per snapshot, 64 neighbours (the registered
                    comparator; widened only where a neighbourhood is coplanar)
    rbf_tuned       per-snapshot RBF with kernel, neighbours and smoothing chosen on the
                    shared 10 % observation hold-out (``problem.validation_split``)
    rbf_spacetime   one RBF over (x, y, z, c t) through both snapshots, with the time
                    scale c chosen on the same hold-out
"""

from __future__ import annotations

import itertools
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.spatial import cKDTree

from .postprocess import PressureIntegrator, score_field
from .problem import Problem, build_problem, validation_split
from .training import save_run

METHODS = ("linear", "rbf", "rbf_tuned", "rbf_spacetime")
TUNE_KERNELS = ("thin_plate_spline", "cubic", "quintic")
TUNE_NEIGHBORS = (32, 64, 128, 256)
TUNE_SMOOTHING = (0.0, 1e-4, 1e-3, 1e-2)
TIME_SCALES_M_PER_S = (0.0, 0.5, 2.0, 8.0)     # metres of "distance" per second of time

Fields = Dict[int, np.ndarray]


def _sources(P: Problem, ids: np.ndarray, t_ms: int) -> Tuple[np.ndarray, np.ndarray]:
    """Observation nodes + wall (u = 0) + inlet plug at one time, duplicates removed.

    Inlet-face nodes are also volume nodes, so an observation can coincide with one.
    """
    s = P.snaps[t_ms]
    src = np.vstack([P.X[ids], P.wall_xyz, P.inlet_xyz])
    val = np.vstack([s.uvw[ids], np.zeros((len(P.wall_xyz), 3)), s.inlet_uvw])
    _, keep = np.unique(np.round(src, 9), axis=0, return_index=True)
    keep = np.sort(keep)
    return src[keep], val[keep]


def rbf_evaluator(src: np.ndarray, val: np.ndarray, kernel: str = "thin_plate_spline",
                  neighbors: int = 64, smoothing: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
    """Local RBF evaluator. The local systems are solved at evaluation; where a
    neighbourhood is coplanar (e.g. on the inlet face) the linear-polynomial system is
    singular, and only then is the neighbourhood widened (x2, then x4)."""
    def evaluate(x: np.ndarray) -> np.ndarray:
        for n in (neighbors, 2 * neighbors, 4 * neighbors):
            try:
                out = RBFInterpolator(src, val, neighbors=n, kernel=kernel, smoothing=smoothing)(x)
                evaluate.neighbors_used = n
                return out
            except np.linalg.LinAlgError:
                continue
        raise RuntimeError(f"RBF singular for {neighbors}-{4 * neighbors} neighbours")
    evaluate.neighbors_used = neighbors
    return evaluate


def linear_fields(P: Problem) -> Tuple[Fields, Dict]:
    fields = {}
    for t in P.times_ms:
        src, val = _sources(P, P.obs_ids, t)
        U = LinearNDInterpolator(src, val)(P.X)
        bad = ~np.isfinite(U).all(1)
        if bad.any():
            U[bad] = val[cKDTree(src).query(P.X[bad])[1]]
        fields[t] = U
    return fields, {}


def rbf_fields(P: Problem) -> Tuple[Fields, Dict]:
    fields, used = {}, {}
    for t in P.times_ms:
        f = rbf_evaluator(*_sources(P, P.obs_ids, t))
        fields[t] = f(P.X)
        used[str(t)] = f.neighbors_used
    return fields, {"rbf_neighbors": used}


def _holdout_error(P: Problem, obs_val: np.ndarray, predict: Callable[[int, np.ndarray], np.ndarray]) -> float:
    num = den = 0.0
    for t in P.times_ms:
        true = P.snaps[t].uvw[obs_val]
        num += np.sum((predict(t, P.X[obs_val]) - true) ** 2)
        den += np.sum(true ** 2)
    return float(np.sqrt(num / den))


def rbf_tuned_fields(P: Problem) -> Tuple[Fields, Dict]:
    val_mask = validation_split(P.obs_ids, P.seed)
    obs_tr, obs_val = P.obs_ids[~val_mask], P.obs_ids[val_mask]
    best = None
    for kernel, nbr, smooth in itertools.product(TUNE_KERNELS, TUNE_NEIGHBORS, TUNE_SMOOTHING):
        fs = {t: rbf_evaluator(*_sources(P, obs_tr, t), kernel, nbr, smooth) for t in P.times_ms}
        try:
            err = _holdout_error(P, obs_val, lambda t, x: fs[t](x))
        except RuntimeError:
            continue
        if best is None or err < best[0]:
            best = (err, kernel, nbr, smooth)
    err, kernel, nbr, smooth = best
    fields = {t: rbf_evaluator(*_sources(P, P.obs_ids, t), kernel, nbr, smooth)(P.X) for t in P.times_ms}
    return fields, {"val_obs_rel_l2": err, "kernel": kernel, "neighbors": nbr, "smoothing": smooth}


def _spacetime_sources(P: Problem, ids: np.ndarray, c: float, t_centre: float):
    parts = [(_sources(P, ids, t), t) for t in P.times_ms]
    src = np.vstack([np.c_[s, np.full(len(s), c * (t / 1000 - t_centre))] for (s, _), t in parts])
    val = np.vstack([v for (_, v), _ in parts])
    if c == 0.0:     # the same points at both times: average them (steady-in-window limit)
        _, inv = np.unique(np.round(src, 9), axis=0, return_inverse=True)
        inv = inv.ravel()
        cnt = np.bincount(inv)
        src = np.stack([np.bincount(inv, src[:, d]) / cnt for d in range(4)], 1)
        val = np.stack([np.bincount(inv, val[:, d]) / cnt for d in range(3)], 1)
    return src, val


def rbf_spacetime_fields(P: Problem) -> Tuple[Fields, Dict]:
    val_mask = validation_split(P.obs_ids, P.seed)
    obs_tr, obs_val = P.obs_ids[~val_mask], P.obs_ids[val_mask]
    t_centre = 0.5 * sum(P.times_ms) / 1000
    at = lambda c, t, x: np.c_[x, np.full(len(x), c * (t / 1000 - t_centre))]  # noqa: E731
    best = None
    for c in TIME_SCALES_M_PER_S:
        for kernel, nbr in itertools.product(("thin_plate_spline", "cubic"), (64, 128)):
            f = rbf_evaluator(*_spacetime_sources(P, obs_tr, c, t_centre), kernel, nbr)
            try:
                err = _holdout_error(P, obs_val, lambda t, x: f(at(c, t, x)))
            except RuntimeError:
                continue
            if best is None or err < best[0]:
                best = (err, c, kernel, nbr)
    err, c, kernel, nbr = best
    f = rbf_evaluator(*_spacetime_sources(P, P.obs_ids, c, t_centre), kernel, nbr)
    fields = {t: f(at(c, t, P.X)) for t in P.times_ms}
    return fields, {"val_obs_rel_l2": err, "time_scale_m_per_s": c, "kernel": kernel, "neighbors": nbr}


INTERPOLATORS = {"linear": linear_fields, "rbf": rbf_fields,
                 "rbf_tuned": rbf_tuned_fields, "rbf_spacetime": rbf_spacetime_fields}


def comparator_name(method: str, case: int, target: int, grid: float, seed: int, min_wall_mm: float = 0.0) -> str:
    mw = f"_mw{min_wall_mm:g}" if min_wall_mm > 0 else ""
    return f"rev2_{method}_c{case:02d}_t{target}_g{grid:g}_s{seed}{mw}"


def run_comparator(method: str, case: int, times: List[int], target: int, grid: float, seed: int,
                   min_wall_mm: float = 0.0) -> Dict:
    """Reconstruct with one comparator (or ``cfd``, the pipeline floor) and score it."""
    t0 = time.time()
    P = build_problem(case, times, target, grid, seed, min_wall_mm=min_wall_mm)
    if method == "cfd":
        fields, meta = {t: P.snaps[t].uvw for t in P.times_ms}, {}
    else:
        fields, meta = INTERPOLATORS[method](P)
    integ = PressureIntegrator(P.X, P.wall_xyz, P.dwall)
    record = {"method": method, "run_name": comparator_name(method, case, target, grid, seed, min_wall_mm),
              "problem": P.info(), **meta, **score_field(P, integ, fields),
              "seconds": round(time.time() - t0, 1)}
    save_run(record)
    return record
