"""Build the dataset / case-inventory table (T1) from the registry.

Reads ``data/registry.json`` (auto-discovered and cross-checked against
``configs/cases.yaml``) and emits, per case: inlet diameter, a reference
non-dimensional diameter, health, symmetry class, the asymmetry ratio beta, the
sac (aneurysm) diameter, and the available cardiac phases. Writes Markdown + CSV +
LaTeX (booktabs) into ``report/tables/``. Pure post-processing -- no model/GPU.

The non-dimensional ``d*`` here uses the WHOLE-dataset mean diameter as the
reference, only so the table is self-contained; the per-fold training scales
(each fold's own ``D_ref``) live in T2 -- see ``make_scales_table.py``.

Usage:
    python scripts/make_dataset_table.py --out dataset_inventory
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.config import PROJECT_ROOT, TABLES_DIR  # noqa: E402
from idealaorta_pinn.data.registry import load_registry  # noqa: E402

_PHASE_ORDER = {"systolic": 0, "diastolic": 1}
_SYM_LABEL = {"axisymmetric": "Axisymmetric", "anterior": "Anterior",
              "posterior": "Posterior", "healthy": "Healthy"}


def _phases(rec) -> str:
    seen: list[str] = []
    for byphase in (getattr(rec, "files", {}) or {}).values():
        for ph in byphase:
            if ph not in seen:
                seen.append(ph)
    seen.sort(key=lambda p: _PHASE_ORDER.get(p, 9))
    return ", ".join(p[0].upper() for p in seen) if seen else "—"


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset/case-inventory table (T1).")
    ap.add_argument("--out", type=str, default="dataset_inventory", help="Output basename.")
    ap.add_argument("--title", type=str, default="Dataset / case inventory (12 idealized cases)")
    args = ap.parse_args()

    records = sorted(load_registry(), key=lambda r: r.case_id)
    if not records:
        print("[dataset] registry is empty; run scripts/prepare.py registry")
        return
    mean_d = sum(r.inlet_diameter_cm for r in records) / len(records)

    rows = []
    for r in records:
        beta = getattr(r, "beta", None)
        diseased = getattr(r, "disease_flag", 0) == 1
        sac = getattr(r, "aneurysm_diameter_cm", None)
        rows.append({
            "case": r.case_id,
            "d_cm": round(r.inlet_diameter_cm, 3),
            "d_star": round(r.inlet_diameter_cm / mean_d, 3),
            "health": getattr(r, "health", "—"),
            "symmetry": _SYM_LABEL.get(getattr(r, "symmetry", ""), getattr(r, "symmetry", "—")),
            "beta": round(beta, 2) if (beta is not None and diseased) else None,
            # The registry stores a nominal sac diameter for every case; it only
            # applies to diseased geometries (healthy controls have no dilation).
            "sac_cm": round(sac, 1) if (sac and diseased) else None,
            "phases": _phases(r),
        })

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["case", "d_cm", "d_star", "health", "symmetry", "beta", "sac_cm", "phases"]

    csv_path = TABLES_DIR / f"{args.out}.csv"
    with open(csv_path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wri.writeheader()
        wri.writerows(rows)

    def cell(x, p=2):
        if x is None:
            return "—"
        return f"{x:.{p}f}" if isinstance(x, float) else str(x)

    hdr = ["Case", "d (cm)", "d*", "Health", "Symmetry", "β", "Sac (cm)", "Phases"]
    md = [f"# {args.title}", "",
          "| " + " | ".join(hdr) + " |",
          "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        md.append("| " + " | ".join([
            str(r["case"]), cell(r["d_cm"]), cell(r["d_star"], 3), r["health"],
            r["symmetry"], cell(r["beta"]), cell(r["sac_cm"], 1), r["phases"],
        ]) + " |")
    md_path = TABLES_DIR / f"{args.out}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    # LaTeX (booktabs)
    tex = [r"\begin{table}[t]", r"  \centering",
           rf"  \caption{{{args.title}. $d^{{*}}=d/\bar d$ with $\bar d$ the dataset-mean "
           r"diameter; per-fold training scales are in Table~\ref{tab:scales}.}}",
           r"  \label{tab:dataset}", r"  \small",
           r"  \begin{tabular}{ccccccc c}", r"    \toprule",
           r"    Case & $d$ (cm) & $d^{*}$ & Health & Symmetry & $\beta$ & Sac (cm) & Phases \\",
           r"    \midrule"]
    for r in rows:
        tex.append("    " + " & ".join([
            str(r["case"]), cell(r["d_cm"]), cell(r["d_star"], 3), r["health"],
            r["symmetry"], cell(r["beta"]), cell(r["sac_cm"], 1), r["phases"],
        ]) + r" \\")
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    tex_path = TABLES_DIR / f"{args.out}.tex"
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")

    print(f"[dataset] wrote {md_path.relative_to(PROJECT_ROOT)}, "
          f"{csv_path.relative_to(PROJECT_ROOT)}, {tex_path.relative_to(PROJECT_ROOT)} "
          f"({len(rows)} cases)")
    print("\n".join(md))


if __name__ == "__main__":
    main()
