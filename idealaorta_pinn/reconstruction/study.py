"""Study definitions (``configs/reconstruction/*.yaml``), job expansion and analyses.

A study config names the cases, the time windows, the observation draws
(``[grid_mm, seed]``), the neural-field arms and the interpolation comparators.
``jobs`` expands it into ``scripts/reconstruct.py`` command lines for
``scripts/queue_jobs.sh``; the analyses read the run records in
``report/metrics/reconstruction/runs`` and write study summaries next to them.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..config import CONFIG_DIR, FULL_DIR, RECON_METRICS_DIR, load_yaml
from .postprocess import PRESSURE_FORMS
from .training import RUNS_DIR

SUMMARY_DIR = RECON_METRICS_DIR / "summaries"
NEIGHBOUR_WINDOW_MS = 50


# --------------------------------------------------------------------------- configs
def load_study(name_or_path: str) -> Dict:
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / "reconstruction" / f"{name_or_path}.yaml"
    return load_yaml(path)


def _windows(study: Dict, case: int) -> Iterator[Tuple[str, List[int], int, List]]:
    for wname, w in study["windows"].items():
        times = w.get("case_times", {}).get(case, w["times"])
        yield wname, list(times), w["target"], w.get("draws", study.get("draws"))


def jobs(study: Dict, kind: str) -> List[str]:
    """Command lines (arguments of scripts/reconstruct.py) for one study.

    ``kind``: ``train`` (neural-field arms), ``baseline`` (registered comparators and the
    CFD floor), ``robustness`` (the additional tuned / space-time RBF comparators) or
    ``reference`` (training-free audits of a reference study).
    """
    if kind == "reference":
        lines = []
        for case in study["cases"]:
            for t in study["times"]:
                pressure = "" if case in study.get("pressure_cases", study["cases"]) else " --no-pressure"
                lines.append(f"audit --case {case} --t {t}{pressure}")
                lines.append(f"wss-oracle --case {case} --t {t} --sources {' '.join(study['wss_sources'])}")
        return lines
    lines = []
    for case in study["cases"]:
        for _, times, target, draws in _windows(study, case):
            win = f"--case {case} --times {times[0]} {times[1]} --target {target}"
            for grid, seed in draws:
                draw = f"--grid {grid:g} --seed {seed}"
                if kind == "train":
                    for arm in study["arms"]:
                        for w in arm["wphys"]:
                            lines.append(f"train --arm {arm['arm']} --closure {arm['closure']} --wphys {w:g} "
                                         f"{win} {draw} --init-seed {seed} --steps {study['steps']}")
                elif kind == "baseline":
                    methods = list(study.get("baselines", []))
                    if list(study.get("floor_draw", [])) == [grid, seed]:
                        methods = ["cfd"] + methods
                    lines += [f"baseline --method {m} {win} {draw}" for m in methods]
                elif kind == "robustness":
                    lines += [f"baseline --method {m} {win} {draw}" for m in study.get("robustness_baselines", [])]
                else:
                    raise ValueError(kind)
    return lines


def snapshot_neighbours(case: int, t_ms: int, window_ms: int = NEIGHBOUR_WINDOW_MS) -> Tuple[Optional[int], Optional[int]]:
    """Nearest exported snapshots before and after ``t_ms`` (within ``window_ms``)."""
    times = sorted({int(p.stem.split("_t")[-1]) for p in FULL_DIR.glob(f"case{case:02d}_solid_t*.parquet")})
    before = [t for t in times if t_ms - window_ms <= t < t_ms]
    after = [t for t in times if t_ms < t <= t_ms + window_ms]
    return (before[-1] if before else None), (after[0] if after else None)


# --------------------------------------------------------------------------- records
def load_runs(pattern: str = "*.json", include_sensitivity: bool = False) -> List[Dict]:
    """Run records matching ``pattern``. Sensitivity runs that drop near-wall observations
    (``_mw`` in the name) are a separate analysis and are excluded unless asked for."""
    files = sorted(glob.glob(str(RUNS_DIR / pattern)))
    if not include_sensitivity:
        files = [f for f in files if "_mw" not in Path(f).stem]
    return [json.load(open(f)) for f in files]


def _get(record: Dict, path: str) -> float:
    node = record
    for key in path.split("/"):
        if node is None or key not in node:
            return np.nan
        node = node[key]
    return float(node)


def method_label(r: Dict) -> str:
    if r["method"] != "nf":
        return r["method"]
    s = r["spec"]
    return f"{s['arm']}+{s['closure']} w{s['wphys']:g}"


def _key(r: Dict) -> Tuple[int, int, float]:
    p = r["problem"]
    return p["case"], p["target_ms"], float(p["grid_mm"])


def _write_summary(name: str, payload: Dict) -> Path:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=float))
    return path


# --------------------------------------------------------------------------- analyses
def select_weights(study: Dict) -> Dict:
    """Physics weight per arm with the lowest held-out observation error."""
    runs = [r for r in load_runs("rev2_*.json") if r["method"] == "nf" and r["spec"]["steps"] == study["steps"]
            and _key(r)[0] in study["cases"]]
    table, best = [], {}
    for r in sorted(runs, key=lambda r: (r["spec"]["arm"], r["spec"]["closure"], r["spec"]["wphys"])):
        s = r["spec"]
        table.append({"arm": s["arm"], "closure": s["closure"], "wphys": s["wphys"],
                      "val_obs_rel_l2": r["final_val_obs_rel_l2"]})
        k = f"{s['arm']}+{s['closure']}"
        if k not in best or r["final_val_obs_rel_l2"] < best[k][1]:
            best[k] = (s["wphys"], r["final_val_obs_rel_l2"])
    out = {"table": table, "selected": {k: v[0] for k, v in best.items()}}
    _write_summary(study["name"], out)
    return out


def summarize(study: Dict) -> Dict:
    """Every method per (case, target, grid): velocity, aneurysm WSS and pressure."""
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in load_runs():
        case, target, grid = _key(r)
        if case not in study["cases"] or (r["method"] == "nf" and r["spec"]["steps"] != study["steps"]):
            continue
        cell = rows[f"case{case}_t{target}_g{grid:g}"][method_label(r)]
        cell["vel"].append(_get(r, "velocity/vel_rel_l2"))
        cell["wss"].append(_get(r, "wss/wss_mag_rel_l2"))
        cell["wss_autograd"].append(_get(r, "wss_autograd/wss_mag_rel_l2"))
        cell["p_own"].append(_get(r, "pressure_own_head/p_rel_l2"))
        for form in PRESSURE_FORMS:
            cell[f"p_{form}"].append(_get(r, f"pressure/{form}/p_rel_l2"))
    out = {unit: {m: {k: [float(x) for x in v] for k, v in d.items()} for m, d in by.items()}
           for unit, by in sorted(rows.items())}
    _write_summary(study["name"], out)
    return out


def confirm(study: Dict) -> Dict:
    """Pre-registered hypothesis tests of a confirmatory study (see its ``hypotheses``)."""
    H = study["hypotheses"]
    grid = float(H["primary_grid"])
    by = defaultdict(lambda: defaultdict(list))            # (case, target, grid, label) -> metric -> values
    for r in load_runs():
        case, target, g = _key(r)
        if case not in study["cases"] or (r["method"] == "nf" and r["spec"]["steps"] != study["steps"]):
            continue
        label = r["spec"]["arm"] if r["method"] == "nf" else r["method"]
        for path in ("velocity/vel_rel_l2", "wss/wss_mag_rel_l2", "wss/wss_q99_ratio"):
            by[(case, target, g, label)][path].append(_get(r, path))
        for form in PRESSURE_FORMS:
            by[(case, target, g, label)][f"pressure/{form}/p_rel_l2"].append(_get(r, f"pressure/{form}/p_rel_l2"))
            by[(case, target, g, label)][f"pressure/{form}/dp_err_Pa"].append(
                _get(r, f"pressure/{form}/dp_pred_Pa") - _get(r, f"pressure/{form}/dp_cfd_Pa"))
    mean = lambda k, m: float(np.mean(by[k][m])) if by.get(k) and by[k][m] else np.nan  # noqa: E731
    targets = sorted({w["target"] for w in study["windows"].values()})
    units = [(c, t) for c in study["cases"] for t in targets]
    cand, comp = H["candidate"], H["comparator"]

    def test(spec):
        wins, ratios = 0, []
        for c, t in units:
            a, b = mean((c, t, grid, cand), spec["metric"]), mean((c, t, grid, comp), spec["metric"])
            wins += bool(a < b)
            ratios.append(a / b)
        med = float(np.nanmedian(ratios))
        passed = wins >= spec["min_wins"] and med <= spec["max_median_ratio"]
        return {"wins": int(wins), "of": len(units), "median_ratio": med, "pass": bool(passed),
                "min_wins": spec["min_wins"], "max_median_ratio": spec["max_median_ratio"]}

    g_metric = H["guardrail_velocity"]["metric"]
    per_unit = []
    for c, t in units:
        row = {"case": c, "target_ms": t}
        for label in (cand, "data", comp, "linear", "cfd", "rbf_tuned", "rbf_spacetime"):
            row[label] = {"vel": mean((c, t, grid, label), "velocity/vel_rel_l2"),
                          "wss": mean((c, t, grid, label), "wss/wss_mag_rel_l2"),
                          "p": mean((c, t, grid, label), "pressure/unsteady+mut/p_rel_l2"),
                          "n": len(by.get((c, t, grid, label), {}).get("velocity/vel_rel_l2", []))}
        per_unit.append(row)
    labels = (cand, "data", comp, "linear", "cfd", "rbf_tuned", "rbf_spacetime")
    forms = {f"t{t}_{label}": {form: float(np.nanmedian([mean((c, t, grid, label), f"pressure/{form}/p_rel_l2")
                                                        for c in study["cases"]])) for form in PRESSURE_FORMS}
             | {"median_abs_dp_err_Pa": float(np.nanmedian([abs(mean((c, t, grid, label), "pressure/unsteady+mut/dp_err_Pa"))
                                                            for c in study["cases"]]))}
             for t in targets for label in labels}
    grids = sorted({float(g) for g, _ in study["draws"]})
    ladder = {f"t{t}_{label}_g{g:g}": {m: float(np.nanmedian([mean((c, t, g, label), p) for c in study["cases"]]))
                                       for m, p in (("vel", "velocity/vel_rel_l2"), ("wss", "wss/wss_mag_rel_l2"),
                                                    ("p", "pressure/unsteady+mut/p_rel_l2"))}
              for t in targets for label in (cand, "data", comp, "linear") for g in grids}
    out = {"H1_pressure": test(H["H1_pressure"]), "H2_wss": test(H["H2_wss"]),
           "guardrail_velocity": all(mean((c, t, grid, cand), g_metric) <= mean((c, t, grid, comp), g_metric)
                                     for c, t in units),
           "units": per_unit, "pressure_forms": forms, "ladder": ladder}
    _write_summary(study["name"], out)
    return out
