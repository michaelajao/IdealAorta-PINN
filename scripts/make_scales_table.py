"""Build the non-dimensionalization-scales table (T2), one row per training set.

Each leave-one-diameter-out fold fits its Normalizer on its own train cases, so
the folds are NOT on a common nondimensionalization. This table makes that
explicit by reading ``models/<exp>/normalizer.json`` for each experiment and
reporting the fitted scales (L, U_ref, diastolic U_ref, D_ref, Re, P_ref,
tau_ref, wss_std). Writes Markdown + CSV + LaTeX (booktabs) into
``report/tables/``. Pure post-processing -- no model/GPU.

Usage:
    python scripts/make_scales_table.py                       # all models with a normalizer
    python scripts/make_scales_table.py --rows \
        "Leave-2.0-out:stageB_kfold_hold2p0" \
        "Leave-2.3-out:stageB_richerloo_f16" \
        "Leave-2.6-out:stageB_kfold_hold2p6"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.config import MODELS_DIR, PROJECT_ROOT, TABLES_DIR  # noqa: E402


def _pairs(args) -> list[tuple[str, str]]:
    if args.rows:
        out = []
        for spec in args.rows:
            label, _, exp = spec.partition(":")
            if not exp:
                raise SystemExit(f"bad --rows entry {spec!r}; expected 'Label:experiment_name'")
            out.append((label, exp))
        return out
    # Default: every model dir that carries a normalizer.json, alphabetically.
    found = sorted(p.parent.name for p in MODELS_DIR.glob("*/normalizer.json"))
    return [(name, name) for name in found]


def main() -> None:
    ap = argparse.ArgumentParser(description="Non-dim scales table (T2).")
    ap.add_argument("--rows", nargs="*", default=None,
                    help='"Label:experiment_name" pairs; default = all models with a normalizer.')
    ap.add_argument("--out", type=str, default="nondim_scales")
    ap.add_argument("--title", type=str, default="Non-dimensionalization scales per training set")
    args = ap.parse_args()

    rows = []
    for label, exp in _pairs(args):
        p = MODELS_DIR / exp / "normalizer.json"
        if not p.exists():
            print(f"[scales] skip {exp}: no normalizer.json")
            continue
        d = json.loads(p.read_text())
        rows.append({
            "trainset": label,
            "L_mm": 1e3 * float(d["L"]),
            "U_ref": float(d["U_ref"]),
            "U_ref_dia": d.get("u_ref_diastolic"),
            "D_ref_cm": float(d["D_ref"]),
            "Re": float(d["Re"]),
            "P_ref": float(d["P_ref"]),
            "tau_ref": float(d["tau_ref"]),
            "wss_std": float(d["wss_std"]),
        })

    if not rows:
        print("[scales] no normalizer.json found under models/.")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["trainset", "L_mm", "U_ref", "U_ref_dia", "D_ref_cm", "Re", "P_ref",
            "tau_ref", "wss_std"]

    csv_path = TABLES_DIR / f"{args.out}.csv"
    with open(csv_path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wri.writeheader()
        wri.writerows(rows)

    def c(x, p=3):
        if x is None:
            return "—"
        return f"{x:.{p}f}" if isinstance(x, float) else str(x)

    hdr = ["Train set", "L (mm)", "U_ref (m/s)", "U_ref^dia (m/s)", "D_ref (cm)",
           "Re", "P_ref (Pa)", "τ_ref (Pa)", "wss_std"]
    md = [f"# {args.title}", "",
          "| " + " | ".join(hdr) + " |",
          "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        md.append("| " + " | ".join([
            r["trainset"], c(r["L_mm"], 2), c(r["U_ref"]), c(r["U_ref_dia"]),
            c(r["D_ref_cm"], 2), c(r["Re"], 0), c(r["P_ref"], 1),
            c(r["tau_ref"], 4), c(r["wss_std"], 3),
        ]) + " |")
    md_path = TABLES_DIR / f"{args.out}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [r"\begin{table}[t]", r"  \centering",
           rf"  \caption{{{args.title}. Each fold's Normalizer is fit on its own "
           r"training cases, so the folds are not on a common nondimensionalization; "
           r"report only scale-free errors across folds.}}",
           r"  \label{tab:scales}", r"  \small",
           r"  \begin{tabular}{lcccccccc}", r"    \toprule",
           r"    Train set & $L$ (mm) & $U_{\mathrm{ref}}$ & $U_{\mathrm{ref}}^{\mathrm{dia}}$ "
           r"& $D_{\mathrm{ref}}$ & $\mathrm{Re}$ & $P_{\mathrm{ref}}$ & $\tau_{\mathrm{ref}}$ "
           r"& $\sigma_\tau$ \\",
           r"    \midrule"]
    for r in rows:
        tex.append("    " + " & ".join([
            r["trainset"].replace("_", r"\_"), c(r["L_mm"], 2), c(r["U_ref"]),
            c(r["U_ref_dia"]), c(r["D_ref_cm"], 2), c(r["Re"], 0), c(r["P_ref"], 1),
            c(r["tau_ref"], 4), c(r["wss_std"], 3),
        ]) + r" \\")
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    tex_path = TABLES_DIR / f"{args.out}.tex"
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")

    print(f"[scales] wrote {md_path.relative_to(PROJECT_ROOT)}, "
          f"{csv_path.relative_to(PROJECT_ROOT)}, {tex_path.relative_to(PROJECT_ROOT)} "
          f"({len(rows)} train sets)")
    print("\n".join(md))


if __name__ == "__main__":
    main()
