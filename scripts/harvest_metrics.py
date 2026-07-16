"""Harvest continuity (div_rel), pressure, WSS, and velocity metrics for every
reported case, straight from the saved checkpoints (inference only, no training).

Emits one JSON blob and a human-readable table so the paper numbers (continuity
residual, pressure agreement, all-fold LODO rows) are all on hand at once.

Each job below names the config that produces its checkpoint, so every row is
reproducible: train the listed configs with ``scripts/run.py`` first, then run
this script. Checkpoint directories are resolved under ``models/``; a run that
has not been trained yet is reported as ``[skip]`` rather than failing the batch.

Usage:
    python scripts/harvest_metrics.py                    # all jobs, CUDA
    python scripts/harvest_metrics.py --device cpu       # no GPU required
    python scripts/harvest_metrics.py --jobs lodo_2.0 lodo_2.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.analysis.metrics import (pressure_metrics, velocity_metrics,  # noqa: E402
                                              wss_metrics)
from idealaorta_pinn.analysis.predict import load_trained  # noqa: E402
from idealaorta_pinn.config import METRICS_DIR, MODELS_DIR  # noqa: E402
from idealaorta_pinn.data.registry import load_registry  # noqa: E402

# (label, experiment name, [cases], regime, config that produces the checkpoint)
# The in-sample rows are the per-case reconstruction floor; the LODO rows are the
# three leave-one-diameter-out folds that together cover every inlet diameter.
JOBS = [
    ("insample_c1", "stageA_case1_s12", [1], "in-sample", "configs/stageA_case1_s12.yaml"),
    ("insample_c2", "stageA_case2_insample", [2], "in-sample", "configs/stageA_case2_insample.yaml"),
    ("insample_c3", "stageA_case3_insample", [3], "in-sample", "configs/stageA_case3_insample.yaml"),
    ("insample_c4", "stageA_case4_insample", [4], "in-sample", "configs/stageA_case4_insample.yaml"),
    ("insample_c7", "stageA_case7_insample", [7], "in-sample", "configs/stageA_case7_insample.yaml"),
    ("lodo_2.0", "stageB_kfold_hold2p0", [1, 2, 3], "LODO-2.0", "configs/stageB_kfold_hold2p0.yaml"),
    ("lodo_2.3", "stageB_richerloo_f16", [4, 5, 6], "LODO-2.3", "configs/stageB_richerloo_f16.yaml"),
    ("lodo_2.6", "stageB_kfold_hold2p6", [7, 8, 9], "LODO-2.6", "configs/stageB_kfold_hold2p6.yaml"),
]
PHASES = ["systolic", "diastolic"]

# Runs may be filed either directly under models/<name>/ (the default output_dir)
# or grouped into a models/<group>/<name>/ subfolder; check both.
_MODEL_SUBDIRS = ("", "_insample", "_ablation", "_seedsweep")


def find_checkpoint(name: str) -> Path | None:
    """Locate ``best_model.pt`` for an experiment, searching the known groupings."""
    for sub in _MODEL_SUBDIRS:
        ck = (MODELS_DIR / sub / name / "best_model.pt") if sub else (MODELS_DIR / name / "best_model.pt")
        if ck.exists():
            return ck
    return None


def _g(d, k):
    return d.get(k) if d else None


def _f(x):
    if x is None:
        return "  --  "
    try:
        if x != x:  # NaN
            return "  nan "
        return f"{x:6.3f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest metrics from trained checkpoints.")
    ap.add_argument("--device", type=str, default="cuda", help="Torch device (cuda/cpu).")
    ap.add_argument("--jobs", type=str, nargs="*", default=None,
                    help="Subset of job labels to harvest (default: all).")
    ap.add_argument("--out", type=Path, default=METRICS_DIR / "harvest_all.json")
    args = ap.parse_args()

    jobs = JOBS if not args.jobs else [j for j in JOBS if j[0] in set(args.jobs)]
    records = load_registry()
    rows = []
    for label, name, cases, regime, cfg_path in jobs:
        ck = find_checkpoint(name)
        if ck is None:
            print(f"[skip] {label}: no checkpoint for '{name}' under {MODELS_DIR}. "
                  f"Train it first:  python scripts/run.py --config {cfg_path}")
            continue
        model = load_trained(ck, device=args.device)
        for cid in cases:
            for ph in PHASES:
                vm = velocity_metrics(model, records, cid, ph, kind="XZ")
                wm = wss_metrics(model, records, cid, ph)
                pm = pressure_metrics(model, records, cid, ph)
                peak_ratio = None
                if wm and wm["wss_peak_cfd"] > 1e-9:
                    peak_ratio = wm["wss_peak_pinn"] / wm["wss_peak_cfd"]
                row = {
                    "label": label, "regime": regime, "case": cid, "phase": ph,
                    "vel_nrmse_phase": _g(vm, "vel_nrmse_phase"),
                    "vel_rel_l2": _g(vm, "vel_rel_l2"),
                    "div_rel": _g(vm, "div_rel"),
                    "div_rms": _g(vm, "div_rms"),
                    "recirc_cfd": _g(vm, "recirc_cfd"), "recirc_pinn": _g(vm, "recirc_pinn"),
                    "wss_nrmse": _g(wm, "wss_nrmse"), "wss_rel_l2": _g(wm, "wss_rel_l2"),
                    "wss_peak_cfd": _g(wm, "wss_peak_cfd"), "wss_peak_pinn": _g(wm, "wss_peak_pinn"),
                    "wss_peak_ratio": peak_ratio,
                    "wss_pearson_r": _g(wm, "wss_pearson_r"),
                    "dp_pearson_r": _g(pm, "dp_pearson_r"), "dp_spearman_r": _g(pm, "dp_spearman_r"),
                    "dp_range_ratio": _g(pm, "dp_range_ratio"),
                }
                rows.append(row)
                print(f"{label:12s} c{cid} {ph:9s} "
                      f"velNRMSE={_f(row['vel_nrmse_phase'])} "
                      f"div_rel={_f(row['div_rel'])} "
                      f"wssNRMSE={_f(row['wss_nrmse'])} "
                      f"peakR={_f(row['wss_peak_ratio'])} "
                      f"dp_r={_f(row['dp_pearson_r'])} dp_rho={_f(row['dp_spearman_r'])} "
                      f"dp_range={_f(row['dp_range_ratio'])}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n[harvest] wrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
