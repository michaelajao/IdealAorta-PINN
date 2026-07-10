"""Harvest continuity (div_rel), pressure, WSS, and velocity metrics for every
reported case, straight from the saved checkpoints (inference only, no training).

Emits one JSON blob and a human-readable table so the paper numbers (continuity
residual, pressure agreement, all-fold LODO rows) are all on hand at once.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.analysis.metrics import (pressure_metrics, velocity_metrics,  # noqa: E402
                                              wss_metrics)
from idealaorta_pinn.analysis.predict import load_trained  # noqa: E402
from idealaorta_pinn.data.registry import load_registry  # noqa: E402

# (label, checkpoint dir, [cases], regime)
JOBS = [
    ("insample_c1", "models/stageA_case1_s12", [1], "in-sample"),
    ("insample_c2", "models/_insample/stageA_case2_insample", [2], "in-sample"),
    ("insample_c3", "models/_insample/stageA_case3_insample", [3], "in-sample"),
    ("insample_c4", "models/_insample/stageA_case4_insample", [4], "in-sample"),
    ("insample_c7", "models/_insample/stageA_case7_insample", [7], "in-sample"),
    ("lodo_2.0", "models/stageB_kfold_hold2p0", [1, 2, 3], "LODO-2.0"),
    ("lodo_2.3", "models/stageB_richerloo_f16", [4, 5, 6], "LODO-2.3"),
    ("lodo_2.6", "models/stageB_kfold_hold2p6", [7, 8, 9], "LODO-2.6"),
]
PHASES = ["systolic", "diastolic"]


def _g(d, k):
    return d.get(k) if d else None


def main() -> None:
    records = load_registry()
    rows = []
    for label, ckpt, cases, regime in JOBS:
        ck = Path(ckpt) / "best_model.pt"
        if not ck.exists():
            print(f"[skip] {label}: no checkpoint at {ck}")
            continue
        model = load_trained(ck, device="cuda")
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
    out = Path("report/metrics/harvest_all.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n[harvest] wrote {out}  ({len(rows)} rows)")


def _f(x):
    if x is None:
        return "  --  "
    try:
        if x != x:  # NaN
            return "  nan "
        return f"{x:6.3f}"
    except (TypeError, ValueError):
        return str(x)


if __name__ == "__main__":
    main()
