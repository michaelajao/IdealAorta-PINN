"""Re-score every reported run the way the revision reports it.

The submitted manuscript quoted one velocity number per (case, phase): the XZ-slice
error averaged over the three velocity components (``vel_nrmse_phase``), which is the
vector error divided by sqrt(3) on the easier of the two slices. This script scores
each run on BOTH slices and on the 3D streamline points with the full error vector
(``vel_nrmse_vec``), local percentiles and exceedance fractions, and for the
leave-one-diameter-out folds scores the CFD-interpolation baselines on the very same
points, so every PINN number has its trivial comparator beside it.

Outputs (all regenerable):
    report/metrics/_rescore/rows.json          one row per (run, case, phase, points)
    report/metrics/_rescore/coverage.json      in-sample error vs distance to training data
    report/metrics/_rescore/tiles.json         dense-supervision diagnostic (if trained)
    report/tables/rescore_{insample,lodo,coverage,tiles}.md (+ .csv)

Usage:
    python scripts/report.py rescore                       # everything that has a checkpoint
    python scripts/report.py rescore --runs insample_c1 lodo_2.3
    python scripts/report.py rescore --tiles-only          # just the dense-supervision diagnostic
    python scripts/report.py rescore --figures             # also the revised velocity figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math

import numpy as np

from .metrics import (BASELINE_MODES, PHASES, REPORTED_RUNS,
                                              _json_safe, baseline_velocity_metrics,
                                              baseline_wss_metrics, field_error_stats,
                                              find_checkpoint, pressure_metrics,
                                              velocity_metrics, wss_metrics)
from .predict import load_trained, predict_physical
from ..config import FIGURES_DIR, METRICS_DIR, TABLES_DIR
from ..data.cache import load_points
from ..data.registry import cases_by_id, load_registry

OUT = METRICS_DIR / "_rescore"
KINDS = ("XY", "XZ", "3D")
VEL_KEYS = ("vel_nrmse_vec", "vel_nrmse_comp", "err_p50", "err_p90", "err_p95", "err_p99",
            "err_max", "err_p95_ms", "err_p99_ms", "err_max_ms", "frac_err_gt_10",
            "frac_err_gt_25", "frac_err_gt_50", "nrmse_vec_slow", "nrmse_vec_fast", "frac_slow",
            "speed_q995_cfd", "speed_q995_pinn", "speed_max_cfd", "speed_max_pinn",
            "overshoot_q995", "frac_reversed", "div_rel", "recirc_cfd", "recirc_pinn",
            "recirc_cfd_sac", "recirc_pinn_sac", "phase_u_ref", "n_points")
DIST_BINS_MM = (0.0, 1.0, 2.0, 3.0, 5.0, math.inf)
TILES = {"tile_mm": 2.0, "parity": 1}          # the colour diag_case1_densetiles never saw
DIAG_RUNS = (("baseline (3D streamlines)", "stageA_case1_s12"),
             ("dense (3D + train tiles)", "diag_case1_densetiles"))


def _fmt(x, p=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else str(x)


def _write(rows, stem, title, cols, headers, md_rows=None):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else cols,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    md = [f"# {title}", "", "| " + " | ".join(headers) + " |",
          "|" + "|".join(["---"] * len(headers)) + "|"]
    md += md_rows if md_rows is not None else [
        "| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rows]
    (TABLES_DIR / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def score_runs(labels, device):
    records = load_registry()
    selected = [r for r in REPORTED_RUNS if not labels or r[0] in set(labels)]
    rows, coverage = [], []
    for label, name, cases, regime, _cfg in selected:
        ck = find_checkpoint(name)
        if ck is None:
            print(f"[skip] {label}: no checkpoint for {name}")
            continue
        model = load_trained(ck, device=device)
        train_cases = list(model.config["data"]["train_cases"])
        for cid in cases:
            held = cid not in train_cases
            for ph in PHASES:
                wm = wss_metrics(model, records, cid, ph) or {}
                pm = pressure_metrics(model, records, cid, ph) or {}
                wb = {m: baseline_wss_metrics(records, cid, ph, train_cases, m) or {}
                      for m in BASELINE_MODES} if held else {}
                for kind in KINDS:
                    vm = velocity_metrics(model, records, cid, ph, kind=kind)
                    if vm is None or vm.get("degenerate_slice"):
                        continue
                    row = {"label": label, "run": name, "regime": regime, "case": cid,
                           "phase": ph, "points": kind,
                           "points_role": ("training points" if (kind == "3D" and not held)
                                           else "held out"),
                           **{k: vm.get(k) for k in VEL_KEYS},
                           "wss_rel_l2": wm.get("wss_rel_l2"),
                           "wss_peak_ratio": (wm["wss_peak_pinn"] / wm["wss_peak_cfd"])
                           if wm.get("wss_peak_cfd") else None,
                           "wss_pearson_r": wm.get("wss_pearson_r"),
                           "dp_pearson_r": pm.get("dp_pearson_r")}
                    if held:
                        for m in BASELINE_MODES:
                            bm = baseline_velocity_metrics(records, cid, ph, kind, train_cases, m)
                            row[f"bl_{m}_nrmse_vec"] = bm.get("vel_nrmse_vec") if bm else None
                            row[f"bl_{m}_err_p95"] = bm.get("err_p95") if bm else None
                            row[f"bl_{m}_tag"] = bm.get("neighbours") if bm else None
                            row[f"bl_{m}_wss_rel_l2"] = wb[m].get("wss_rel_l2")
                            row[f"bl_{m}_wss_peak_ratio"] = wb[m].get("wss_peak_ratio")
                    rows.append(row)
                    print(f"{label:14s} c{cid:<2} {ph:9s} {kind:2s} vec={_fmt(row['vel_nrmse_vec'])} "
                          f"comp={_fmt(row['vel_nrmse_comp'])} p95={_fmt(row['err_p95'])} "
                          f">25%={_fmt(row['frac_err_gt_25'])} over={_fmt(row['overshoot_q995'], 2)}"
                          + (f"  | blend={_fmt(row.get('bl_blend_nrmse_vec'))} "
                             f"nearest={_fmt(row.get('bl_nearest_nrmse_vec'))} "
                             f"scaled={_fmt(row.get('bl_nearest_scaled_nrmse_vec'))}" if held else ""))
                if not held:
                    coverage += coverage_rows(model, records, cid, ph, label)
    return rows, coverage


def coverage_rows(model, records, cid, ph, label):
    """In-sample error binned by distance to the nearest 3D streamline training point."""
    from scipy.spatial import cKDTree
    rec = cases_by_id(records)[cid]
    train = load_points(rec, "3D", ph)
    if train is None:
        return []
    tree = cKDTree(train[["x", "y", "z"]].to_numpy(float))
    out = []
    for kind in ("XY", "XZ"):
        df = load_points(rec, kind, ph)
        if df is None:
            continue
        if len(df) > 60_000:
            df = df.sample(60_000, random_state=0)
        coords = df[["x", "y", "z"]].to_numpy(float)
        true = df[["u", "v", "w"]].to_numpy(float)
        q995 = float(np.quantile(np.linalg.norm(true, axis=1), 0.995))
        if q995 < 1e-4:
            continue
        p = predict_physical(model, coords, rec.inlet_diameter_cm, rec.disease_flag, ph,
                             beta=rec.beta if rec.beta is not None else 1.0)
        pred = np.column_stack([p["u"], p["v"], p["w"]])
        dist_mm = tree.query(coords)[0] * 1000.0
        for lo, hi in zip(DIST_BINS_MM[:-1], DIST_BINS_MM[1:]):
            m = (dist_mm >= lo) & (dist_mm < hi)
            if m.sum() < 50:
                continue
            st = field_error_stats(pred[m], true[m], q995)
            out.append({"label": label, "case": cid, "phase": ph, "points": kind,
                        "dist_mm": f"{lo:g}-{hi:g}" if math.isfinite(hi) else f">{lo:g}",
                        "frac_points": float(m.mean()), "vel_nrmse_vec": st["vel_nrmse_vec"],
                        "err_p95": st["err_p95"], "frac_err_gt_25": st["frac_err_gt_25"]})
    return out


def tiles_rows(device):
    """Dense-supervision diagnostic: both Case 1 models on the never-trained tiles."""
    records = load_registry()
    rows = []
    for label, name in DIAG_RUNS:
        ck = find_checkpoint(name)
        # best_model.pt appears while a run trains; final_model.pt only when it ends
        if ck is None or find_checkpoint(name, "final_model.pt") is None:
            print(f"[tiles] {name} has not finished training")
            return []
        model = load_trained(ck, device=device)
        for ph in PHASES:
            for kind in ("XY", "XZ"):
                vm = velocity_metrics(model, records, 1, ph, kind=kind, tiles=TILES)
                rows.append({"model": label, "run": name, "phase": ph, "points": kind,
                             **{k: vm.get(k) for k in VEL_KEYS}})
    return rows


def lodo_table(rows):
    held = [r for r in rows if r["points_role"] == "held out" and r["regime"].startswith("LODO")]
    md = []
    for r in held:
        md.append("| " + " | ".join([
            r["regime"], f"{r['case']}", r["phase"], r["points"],
            _fmt(r["vel_nrmse_vec"]), _fmt(r.get("bl_blend_nrmse_vec")),
            _fmt(r.get("bl_nearest_nrmse_vec")), _fmt(r.get("bl_nearest_scaled_nrmse_vec")),
            _fmt(r["err_p95"]), _fmt(r["wss_rel_l2"]), _fmt(r.get("bl_blend_wss_rel_l2")),
            _fmt(r["wss_peak_ratio"], 2), _fmt(r.get("bl_blend_wss_peak_ratio"), 2)]) + " |")
    _write(held, "rescore_lodo",
           "Leave-one-diameter-out: PINN vs CFD-interpolation baselines on identical points "
           "(vector NRMSE, per-phase 99.5th-percentile speed)",
           [], ["Fold", "Case", "Phase", "Points", "PINN", "Blend", "Nearest",
                "Nearest, Q/A-scaled", "PINN p95", "WSS relL2 PINN", "WSS relL2 blend",
                "Peak WSS PINN", "Peak WSS blend"], md)


def insample_table(rows):
    ins = [r for r in rows if r["regime"] == "in-sample"]
    by = {}
    for r in ins:
        by.setdefault((r["case"], r["phase"]), {})[r["points"]] = r
    md = []
    for (cid, ph), d in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1] != "systolic")):
        xy, xz, d3 = d.get("XY", {}), d.get("XZ", {}), d.get("3D", {})
        md.append("| " + " | ".join([
            f"{cid}", ph, _fmt(xz.get("vel_nrmse_comp")), _fmt(xz.get("vel_nrmse_vec")),
            _fmt(xy.get("vel_nrmse_vec")), _fmt(xy.get("err_p95")), _fmt(xy.get("err_p99_ms"), 2),
            _fmt(xy.get("frac_err_gt_25")), _fmt(xy.get("overshoot_q995"), 2),
            _fmt(xy.get("nrmse_vec_slow")), _fmt(xy.get("div_rel"), 2),
            _fmt(d3.get("vel_nrmse_vec")), _fmt(xy.get("wss_rel_l2")),
            _fmt(xy.get("wss_peak_ratio"), 2)]) + " |")
    _write(ins, "rescore_insample",
           "In-sample reconstruction, re-scored (vector NRMSE unless marked)",
           [], ["Case", "Phase", "XZ per-comp (as submitted)", "XZ", "XY", "XY p95",
                "XY p99 (m/s)", "XY frac > 25%", "XY overshoot", "XY slow-flow",
                "XY div_rel", "3D (training pts)", "WSS relL2", "Peak WSS"], md)


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--runs", nargs="*", default=None, help="REPORTED_RUNS labels (default all).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--tiles-only", action="store_true")
    ap.add_argument("--figures", action="store_true", help="Also draw the revised velocity figures.")


def run(args: argparse.Namespace) -> None:
    import torch
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.tiles_only:
        rows, coverage = score_runs(args.runs, args.device)
        (OUT / "rows.json").write_text(json.dumps(
            [{k: _json_safe(v) for k, v in r.items()} for r in rows], indent=1), encoding="utf-8")
        (OUT / "coverage.json").write_text(json.dumps(
            [{k: _json_safe(v) for k, v in r.items()} for r in coverage], indent=1), encoding="utf-8")
        if rows:
            insample_table(rows)
            lodo_table(rows)
        if coverage:
            _write(coverage, "rescore_coverage",
                   "In-sample error vs distance to the nearest 3D streamline training point",
                   ["label", "phase", "points", "dist_mm", "frac_points", "vel_nrmse_vec",
                    "err_p95", "frac_err_gt_25"],
                   ["Run", "Phase", "Points", "Distance (mm)", "Share of points",
                    "Vector NRMSE", "p95", "Frac > 25%"])

    trows = tiles_rows(args.device)
    if trows:
        (OUT / "tiles.json").write_text(json.dumps(
            [{k: _json_safe(v) for k, v in r.items()} for r in trows], indent=1), encoding="utf-8")
        _write(trows, "rescore_tiles",
               "Dense-supervision diagnostic, Case 1: error on the never-trained 2 mm tiles",
               ["model", "phase", "points", "vel_nrmse_vec", "err_p95", "err_p99_ms",
                "frac_err_gt_25", "nrmse_vec_slow", "nrmse_vec_fast", "overshoot_q995", "div_rel"],
               ["Model", "Phase", "Plane", "Vector NRMSE", "p95", "p99 (m/s)", "Frac > 25%",
                "Slow-flow NRMSE", "Fast-flow NRMSE", "Overshoot", "div_rel"])

    if args.figures:
        from .figures import plane_comparison_pair
        records = load_registry()
        fig_dir = FIGURES_DIR / "_revision"
        for name, cid in (("stageA_case1_s12", 1), ("stageA_case2_insample", 2),
                          ("stageA_case3_insample", 3), ("stageB_richerloo_f16", 4),
                          ("diag_case1_densetiles", 1)):
            ck = find_checkpoint(name)
            if ck is None or find_checkpoint(name, "final_model.pt") is None:
                continue
            model = load_trained(ck, device=args.device)
            for ph in PHASES:
                p = plane_comparison_pair(model, records, cid, ph, out_dir=fig_dir,
                                          tag=name if name.startswith("diag") else "")
                print(f"[figure] {p}")
