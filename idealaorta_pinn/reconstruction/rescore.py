"""Rescore saved run records after a change of a post-processing constant (here the blood density).

The velocity field of every run is rebuilt exactly as it was produced, from the saved checkpoint for
a neural field and by re-interpolation with the stored settings for a comparator, and passed through
``postprocess.score_field`` again. Velocity and WSS scores do not depend on the density, so they are
checked against the stored record; only the pressure scores change. The previous pressure scores are
kept in the record under ``rescored``.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Dict

import numpy as np

from . import RHO
from .interpolation import (_sources, _spacetime_sources, linear_fields, rbf_evaluator, rbf_fields)
from .postprocess import PressureIntegrator, pressure_scores, score_field
from .problem import build_problem
from .training import RUNS_DIR, load_field, predict, save_run

CHECK_KEYS = (("velocity", "vel_rel_l2"), ("wss", "wss_mag_rel_l2"))


def _fields(record: Dict, P, device: str):
    """Rebuild ``{t_ms: (N_int, 3)}`` for a record, plus the network pressure head if any."""
    m = record["method"]
    if m == "nf":
        net, scales, _ = load_field(record["run_name"], device)
        full = {t: predict(net, P.X, t, scales, device) for t in P.times_ms}
        return {t: f[:, :3] for t, f in full.items()}, full[P.target_ms][:, 3]
    if m == "cfd":
        return {t: P.snaps[t].uvw for t in P.times_ms}, None
    if m == "linear":
        return linear_fields(P)[0], None
    if m == "rbf":
        return rbf_fields(P)[0], None
    if m == "rbf_tuned":
        k, n, s = record["kernel"], record["neighbors"], record["smoothing"]
        return {t: rbf_evaluator(*_sources(P, P.obs_ids, t), k, n, s)(P.X) for t in P.times_ms}, None
    if m == "rbf_spacetime":
        c, k, n = record["time_scale_m_per_s"], record["kernel"], record["neighbors"]
        t_centre = 0.5 * sum(P.times_ms) / 1000
        f = rbf_evaluator(*_spacetime_sources(P, P.obs_ids, c, t_centre), k, n)
        return {t: f(np.c_[P.X, np.full(len(P.X), c * (t / 1000 - t_centre))]) for t in P.times_ms}, None
    raise ValueError(f"unknown method {m!r}")


def rescore(name: str, device: str = "cuda", tol: float = 1e-6) -> Dict:
    """Rebuild one run's fields and rescore them with the current constants; overwrite its record."""
    record = json.loads((RUNS_DIR / f"{name}.json").read_text())
    pr = record["problem"]
    spec = record.get("spec") or {}
    times = spec.get("times") or pr["times_ms"]
    P = build_problem(pr["case"], list(times), pr["target_ms"], float(pr["grid_mm"]), int(pr["seed"]),
                      save_mask=False, min_wall_mm=float(spec.get("min_wall_mm", 0.0) or record.get("min_wall_mm", 0.0)))
    if int(len(P.obs_ids)) != int(pr["n_obs_per_time"]):
        raise RuntimeError(f"{name}: rebuilt {len(P.obs_ids)} observations, record has {pr['n_obs_per_time']}")
    fields, p_head = _fields(record, P, device)
    integ = PressureIntegrator(P.X, P.wall_xyz, P.dwall)
    new = score_field(P, integ, fields)
    checks = {}
    for sec, key in CHECK_KEYS:
        if sec in record and sec in new and key in record[sec]:
            old_v, new_v = float(record[sec][key]), float(new[sec][key])
            checks[f"{sec}.{key}"] = {"old": old_v, "new": new_v,
                                      "match": abs(old_v - new_v) <= tol * max(1.0, abs(old_v))}
    old_pressure = {"pressure": record.get("pressure")}
    if p_head is not None:
        old_pressure["pressure_own_head"] = record.get("pressure_own_head")
        tgt = P.snaps[P.target_ms]
        ph = p_head[integ.sub]
        record["pressure_own_head"] = pressure_scores(ph - ph[integ.o_idx].mean(), integ.gauge_ref(tgt.p),
                                                      integ, P.sac[integ.sub])
    record.update(new)
    record["rho"] = RHO
    earlier = record.get("rescored", {})
    record["rescored"] = {"date": _dt.date.today().isoformat(), "rho": RHO,
                          "previous": earlier.get("previous", old_pressure),
                          "previous_rho": earlier.get("previous_rho", 1060.0), "checks": checks}
    save_run(record)
    return record
