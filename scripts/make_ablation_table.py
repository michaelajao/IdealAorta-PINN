"""Build a paper-ready ablation/comparison table from per-experiment metrics.

Reads ``report/metrics/<experiment>/{velocity,wss}.json`` for a set of named
experiments and emits a tidy comparison (Markdown + CSV) into ``report/tables/``.
Pure post-processing — no model or GPU needed; re-run any time metrics change.

Per (experiment, case, phase) it reports the honest per-phase velocity NRMSE,
velocity rel-L2, WSS rel-L2, the WSS peak ratio (PINN/CFD), and the recirculation
fractions (CFD vs PINN, using the M4 noise-robust metric).

Usage:
    python scripts/make_ablation_table.py \
        --rows "baseline:stageA_case1" "S1:stageA_case1_s1" "S1+S2+S4:stageA_case1_s124" \
               "S1+S2:stageA_case1_s12" \
        --cases 1 --title "Stage A in-sample ablation (Case 1)" --out stageA_ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.config import METRICS_DIR, PROJECT_ROOT, TABLES_DIR  # noqa: E402


def _load(experiment: str, kind: str) -> list[dict]:
    p = METRICS_DIR / experiment / f"{kind}.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def _index(rows: list[dict]) -> dict[tuple[int, str], dict]:
    return {(r["case"], r["phase"]): r for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation/comparison table from metrics JSON.")
    ap.add_argument("--rows", nargs="+", required=True,
                    help='"Label:experiment_name" pairs, in display order.')
    ap.add_argument("--cases", type=int, nargs="*", default=None,
                    help="Cases to include (default: all found).")
    ap.add_argument("--phases", type=str, nargs="*", default=["systolic", "diastolic"])
    ap.add_argument("--title", type=str, default="Ablation")
    ap.add_argument("--out", type=str, default="ablation", help="Output basename.")
    args = ap.parse_args()

    pairs = []
    for spec in args.rows:
        label, _, exp = spec.partition(":")
        if not exp:
            ap.error(f"bad --rows entry {spec!r}; expected 'Label:experiment_name'")
        pairs.append((label, exp))

    records = []
    for label, exp in pairs:
        vel = _index(_load(exp, "velocity"))
        wss = _index(_load(exp, "wss"))
        keys = sorted(set(vel) | set(wss))
        for case, phase in keys:
            if args.cases and case not in args.cases:
                continue
            if args.phases and phase not in args.phases:
                continue
            v = vel.get((case, phase), {})
            w = wss.get((case, phase), {})
            peak_cfd, peak_pinn = w.get("wss_peak_cfd"), w.get("wss_peak_pinn")
            peak_ratio = (peak_pinn / peak_cfd) if (peak_cfd and peak_pinn) else None
            records.append({
                "config": label, "experiment": exp, "case": case, "phase": phase,
                "vel_nrmse_phase": v.get("vel_nrmse_phase"),
                "vel_rel_l2": v.get("vel_rel_l2"),
                "wss_rel_l2": w.get("wss_rel_l2"),
                "wss_peak_ratio": peak_ratio,
                "recirc_cfd": v.get("recirc_cfd"),
                "recirc_pinn": v.get("recirc_pinn"),
            })

    if not records:
        print("[ablation] no metrics found for the requested experiments/cases.")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["config", "case", "phase", "vel_nrmse_phase", "vel_rel_l2",
            "wss_rel_l2", "wss_peak_ratio", "recirc_cfd", "recirc_pinn"]

    # CSV
    import csv
    csv_path = TABLES_DIR / f"{args.out}.csv"
    with open(csv_path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wri.writeheader()
        wri.writerows(records)

    # Markdown
    def fmt(x, p=4):
        return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"
    hdr = ["config", "case", "phase", "vel NRMSE", "vel relL2",
           "WSS relL2", "WSS peak PINN/CFD", "recirc CFD", "recirc PINN"]
    lines = [f"# {args.title}", "",
             "| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in records:
        lines.append("| " + " | ".join([
            r["config"], str(r["case"]), r["phase"],
            fmt(r["vel_nrmse_phase"]), fmt(r["vel_rel_l2"]), fmt(r["wss_rel_l2"]),
            fmt(r["wss_peak_ratio"], 2), fmt(r["recirc_cfd"], 3), fmt(r["recirc_pinn"], 3),
        ]) + " |")
    md_path = TABLES_DIR / f"{args.out}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ablation] wrote {md_path.relative_to(PROJECT_ROOT)} and "
          f"{csv_path.relative_to(PROJECT_ROOT)} ({len(records)} rows)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
