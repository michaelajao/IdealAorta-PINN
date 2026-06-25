"""Build the seed-sweep robustness table (T6): mean +/- std (n) across seeds.

For each regime it globs ``report/metrics/<family>_seed*/{velocity,wss}.json`` (the
naming the planned ``run.py --seed`` / ``--name-suffix`` overrides produce) and
aggregates the primary metrics per phase over all (seed, case) samples, reporting
mean +/- std and the sample count n. Per the rigor plan: mean +/- std is primary;
no significance tests at small n. Writes Markdown + CSV + LaTeX into
``report/tables/``. Pure post-processing -- no model/GPU.

Usage:
    python scripts/make_seed_table.py --rows \
        "Leave-2.3-out:stageB_richerloo_f16" \
        "In-sample C1:stageA_case1_s12"
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

# Metric key -> (source file, display label, decimals). Peak ratio is derived.
_METRICS = [
    ("vel_nrmse_phase", "velocity", "Vel NRMSE", 3),
    ("wss_nrmse", "wss", "WSS NRMSE", 3),
    ("wss_peak_ratio", "wss", "WSS peak PINN/CFD", 2),
    ("recirc_pinn", "velocity", "Recirc PINN", 3),
]


def _seed_dirs(family: str) -> list[Path]:
    return sorted(d for d in METRICS_DIR.glob(f"{family}_seed*") if d.is_dir())


def _load(d: Path, kind: str) -> list[dict]:
    p = d / f"{kind}.json"
    return json.loads(p.read_text()) if p.exists() else []


def _mean_std(xs: list[float]) -> tuple[float, float, int]:
    vals = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(vals) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd, n


def _collect(family: str) -> dict[tuple[str, str], list[float]]:
    """{(phase, metric_key): [values over (seed, case)]}; peak ratio derived."""
    acc: dict[tuple[str, str], list[float]] = {}
    for d in _seed_dirs(family):
        vel = {(r["case"], r["phase"]): r for r in _load(d, "velocity")}
        wss = {(r["case"], r["phase"]): r for r in _load(d, "wss")}
        for (case, phase), r in vel.items():
            acc.setdefault((phase, "vel_nrmse_phase"), []).append(r.get("vel_nrmse_phase"))
            acc.setdefault((phase, "recirc_pinn"), []).append(r.get("recirc_pinn"))
        for (case, phase), w in wss.items():
            acc.setdefault((phase, "wss_nrmse"), []).append(w.get("wss_nrmse"))
            pc, pp = w.get("wss_peak_cfd"), w.get("wss_peak_pinn")
            acc.setdefault((phase, "wss_peak_ratio"), []).append(
                (pp / pc) if (pc and pp) else float("nan"))
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed-sweep robustness table (T6).")
    ap.add_argument("--rows", nargs="+", required=True,
                    help='"Regime:family" pairs; globs report/metrics/<family>_seed*/.')
    ap.add_argument("--phases", nargs="*", default=["systolic", "diastolic"])
    ap.add_argument("--out", type=str, default="seed_robustness")
    ap.add_argument("--title", type=str, default="Seed-sweep robustness (mean ± std, n)")
    args = ap.parse_args()

    records = []
    for spec in args.rows:
        regime, _, family = spec.partition(":")
        if not family:
            ap.error(f"bad --rows entry {spec!r}; expected 'Regime:family'")
        dirs = _seed_dirs(family)
        if not dirs:
            print(f"[seed] no report/metrics/{family}_seed*/ dirs yet — skipping '{regime}'")
            continue
        print(f"[seed] {regime}: {len(dirs)} seed dirs ({', '.join(d.name for d in dirs)})")
        acc = _collect(family)
        for phase in args.phases:
            row = {"regime": regime, "phase": phase}
            for key, _src, _lab, _p in _METRICS:
                m, sd, n = _mean_std(acc.get((phase, key), []))
                row[key] = m
                row[f"{key}_sd"] = sd
                row[f"{key}_n"] = n
            records.append(row)

    if not records:
        print("[seed] nothing to aggregate yet — run a seed sweep first "
              "(run.py --seed N produces <family>_seedN/).")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    flat_cols = ["regime", "phase"]
    for key, _s, _l, _p in _METRICS:
        flat_cols += [key, f"{key}_sd", f"{key}_n"]
    csv_path = TABLES_DIR / f"{args.out}.csv"
    with open(csv_path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=flat_cols, extrasaction="ignore")
        wri.writeheader()
        wri.writerows(records)

    def cell(row, key, p):
        m, sd, n = row[key], row[f"{key}_sd"], row[f"{key}_n"]
        if n == 0 or (isinstance(m, float) and math.isnan(m)):
            return "—"
        return f"{m:.{p}f} ± {sd:.{p}f} (n={n})"

    hdr = ["Regime", "Phase"] + [lab for _k, _s, lab, _p in _METRICS]
    md = [f"# {args.title}", "",
          "| " + " | ".join(hdr) + " |",
          "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in records:
        md.append("| " + " | ".join(
            [r["regime"], r["phase"]] + [cell(r, k, p) for k, _s, _l, p in _METRICS]) + " |")
    md_path = TABLES_DIR / f"{args.out}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [r"\begin{table}[t]", r"  \centering",
           rf"  \caption{{{args.title}. Mean $\pm$ std over the seed sweep; "
           r"$n$ counts (seed, case, ...) samples. No significance tests at small $n$.}}",
           r"  \label{tab:seeds}", r"  \small",
           r"  \begin{tabular}{ll" + "c" * len(_METRICS) + "}", r"    \toprule",
           "    " + " & ".join(hdr) + r" \\", r"    \midrule"]
    for r in records:
        tex.append("    " + " & ".join(
            [r["regime"], r["phase"]] + [cell(r, k, p) for k, _s, _l, p in _METRICS]) + r" \\")
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (TABLES_DIR / f"{args.out}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    print(f"[seed] wrote {md_path.relative_to(PROJECT_ROOT)} (+ .csv, .tex), {len(records)} rows")
    print("\n".join(md))


if __name__ == "__main__":
    main()
