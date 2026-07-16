"""Build the leave-one-diameter-out k-fold generalization table (T5).

Reads the three corrected LODO folds' ``report/metrics/<exp>/{velocity,wss}.json``,
labels each held case by symmetry class and each fold as interpolation (the middle
2.3 cm diameter) or extrapolation (2.0 / 2.6 cm, a single bracketing diameter), and
emits per-(case, phase) rows plus a per-fold mean. Primary metrics are per-phase
velocity NRMSE and WSS NRMSE; recirculation is reported CFD/PINN per phase (matched
in-sample, over-predicted held-out). Writes Markdown + CSV + LaTeX into
``report/tables/``. Pure post-processing -- no model/GPU.

Each fold uses its own train-set nondimensionalization (see T2 / make_scales_table),
so only scale-free errors are comparable across folds; never average 2.0/2.3/2.6 into
one "generalization" number (2.3 is interpolation, 2.0/2.6 are extrapolation).

Usage:
    python scripts/make_kfold_table.py --folds \
        2.0:stageB_kfold_hold2p0 2.3:stageB_richerloo_f16 2.6:stageB_kfold_hold2p6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.config import METRICS_DIR, PROJECT_ROOT, TABLES_DIR  # noqa: E402
from idealaorta_pinn.data.registry import cases_by_id, load_registry  # noqa: E402

_SYM = {"axisymmetric": "axisym.", "anterior": "anterior", "posterior": "posterior",
        "healthy": "healthy"}
_PHASE_ORDER = {"systolic": 0, "diastolic": 1}


def _load(exp: str, kind: str) -> dict:
    p = METRICS_DIR / exp / f"{kind}.json"
    return {(r["case"], r["phase"]): r for r in json.loads(p.read_text())} if p.exists() else {}


def _fmt(x, p=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else str(x)


def _mean(xs):
    v = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return sum(v) / len(v) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="LODO k-fold generalization table (T5).")
    ap.add_argument("--folds", nargs="+", required=True,
                    help='"held_cm:experiment" specs, in display order.')
    ap.add_argument("--phases", nargs="*", default=["systolic", "diastolic"])
    ap.add_argument("--out", type=str, default="lodo_kfold")
    ap.add_argument("--title", type=str,
                    default="Leave-one-diameter-out generalization (per-phase NRMSE)")
    args = ap.parse_args()

    by = cases_by_id(load_registry())
    mid = None  # the interpolation fold = the middle held diameter
    held_cm = []
    for spec in args.folds:
        d, _, exp = spec.partition(":")
        if not exp:
            ap.error(f"bad --folds entry {spec!r}; expected 'held_cm:experiment'")
        held_cm.append(float(d))
    mid = sorted(held_cm)[len(held_cm) // 2] if held_cm else None

    rows = []
    for spec in args.folds:
        d_str, _, exp = spec.partition(":")
        d = float(d_str)
        nature = "interp" if (mid is not None and abs(d - mid) < 1e-9 and len(held_cm) >= 3) else "extrap"
        vel, wss = _load(exp, "velocity"), _load(exp, "wss")
        keys = sorted(set(vel) | set(wss), key=lambda k: (k[0], _PHASE_ORDER.get(k[1], 9)))
        fold_vel, fold_wss = [], []
        for case, phase in keys:
            if phase not in args.phases:
                continue
            v, w = vel.get((case, phase), {}), wss.get((case, phase), {})
            pc, pp = w.get("wss_peak_cfd"), w.get("wss_peak_pinn")
            peak = (pp / pc) if (pc and pp) else None
            sym = _SYM.get(getattr(by.get(case), "symmetry", ""), "?")
            row = {
                "held_cm": d, "nature": nature, "case": case, "symmetry": sym, "phase": phase,
                "vel_nrmse": v.get("vel_nrmse_phase"), "wss_nrmse": w.get("wss_nrmse"),
                "peak_ratio": peak,
                "recirc_cfd": v.get("recirc_cfd"), "recirc_pinn": v.get("recirc_pinn"),
                "div_rel": v.get("div_rel"),
            }
            rows.append(row)
            fold_vel.append(row["vel_nrmse"])
            fold_wss.append(row["wss_nrmse"])
        # per-fold mean row
        rows.append({
            "held_cm": d, "nature": nature, "case": "mean", "symmetry": "", "phase": "",
            "vel_nrmse": _mean(fold_vel), "wss_nrmse": _mean(fold_wss),
            "peak_ratio": None, "recirc_cfd": None, "recirc_pinn": None, "div_rel": None,
        })

    if not rows:
        print("[kfold] no metrics found for the requested folds.")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["held_cm", "nature", "case", "symmetry", "phase", "vel_nrmse", "wss_nrmse",
            "peak_ratio", "recirc_cfd", "recirc_pinn", "div_rel"]
    csv_path = TABLES_DIR / f"{args.out}.csv"
    with open(csv_path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wri.writeheader()
        wri.writerows(rows)

    def recirc_cell(r):
        if r["recirc_cfd"] is None:
            return "—"
        return f"{_fmt(r['recirc_cfd'])} / {_fmt(r['recirc_pinn'])}"

    def case_cell(r):
        return "**mean**" if r["case"] == "mean" else f"{r['case']} ({r['symmetry']})"

    hdr = ["Held (cm)", "Nature", "Case (sym)", "Phase", "Vel NRMSE", "WSS NRMSE",
           "Peak P/C", "Recirc CFD/PINN", "div_rel"]
    md = [f"# {args.title}", "",
          "| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        md.append("| " + " | ".join([
            _fmt(r["held_cm"], 1), r["nature"], case_cell(r), r["phase"],
            _fmt(r["vel_nrmse"]), _fmt(r["wss_nrmse"]), _fmt(r["peak_ratio"], 2),
            recirc_cell(r), _fmt(r["div_rel"]),
        ]) + " |")
    md_path = TABLES_DIR / f"{args.out}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [r"\begin{table}[t]", r"  \centering",
           rf"  \caption{{{args.title}. 2.3 cm is interpolation; 2.0/2.6 cm are "
           r"extrapolation (single bracketing diameter). Each fold uses its own "
           r"train-set scales (Table~\ref{tab:scales}); do not average across folds. "
           r"Recirculation is matched in-sample but over-predicted held-out.}}",
           r"  \label{tab:lodo}", r"  \small", r"  \begin{tabular}{llllccccc}",
           r"    \toprule",
           r"    Held & Nature & Case (sym.) & Phase & Vel NRMSE & WSS NRMSE & Peak "
           r"& Recirc C/P & $\mathrm{div}_{\mathrm{rel}}$ \\", r"    \midrule"]
    for r in rows:
        case = r"\textbf{mean}" if r["case"] == "mean" else f"{r['case']} ({r['symmetry']})"
        tex.append("    " + " & ".join([
            _fmt(r["held_cm"], 1), r["nature"], case, r["phase"], _fmt(r["vel_nrmse"]),
            _fmt(r["wss_nrmse"]), _fmt(r["peak_ratio"], 2),
            (f"{_fmt(r['recirc_cfd'])}/{_fmt(r['recirc_pinn'])}" if r["recirc_cfd"] is not None else "—"),
            _fmt(r["div_rel"]),
        ]) + r" \\")
        if r["case"] == "mean":
            tex.append(r"    \midrule")
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (TABLES_DIR / f"{args.out}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    print(f"[kfold] wrote {md_path.relative_to(PROJECT_ROOT)} (+ .csv, .tex), {len(rows)} rows")
    print("\n".join(md))


if __name__ == "__main__":
    main()
