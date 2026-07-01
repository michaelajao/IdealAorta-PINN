"""Error-vs-diameter figure (B4): LODO generalization against the in-sample floor.

Plots the per-fold mean per-phase velocity NRMSE and WSS NRMSE at each held inlet
diameter, with the in-sample reconstruction floor as a reference line. The message:
held-out generalization is several times the in-sample floor, 2.3 cm (interpolation)
and 2.0 cm sit lower, and 2.6 cm (extrapolation above the training range) is the
hardest. Each fold uses its own train-set scales, so only the scale-free NRMSE is
comparable across folds. Writes PNG + PDF to report/figures/ and paper/figures/.

Usage:
    python scripts/make_error_vs_diameter.py \
        --folds 2.0:stageB_kfold_hold2p0 2.3:stageB_richerloo_f16 2.6:stageB_kfold_hold2p6 \
        --insample stageA_case1_s12
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.config import FIGURES_DIR, PAPER_FIGURES_DIR, METRICS_DIR, PROJECT_ROOT  # noqa: E402


def _mean(exp: str, kind: str, key: str) -> float:
    p = METRICS_DIR / exp / f"{kind}.json"
    if not p.exists():
        return float("nan")
    vals = [r.get(key) for r in json.loads(p.read_text())]
    vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Error-vs-diameter figure (B4).")
    ap.add_argument("--folds", nargs="+", required=True, help='"held_cm:experiment" specs.')
    ap.add_argument("--insample", type=str, default="stageA_case1_s12",
                    help="Experiment for the in-sample floor line.")
    ap.add_argument("--out", type=str, default="error_vs_diameter")
    args = ap.parse_args()

    diam, vel, wss = [], [], []
    for spec in args.folds:
        d, _, exp = spec.partition(":")
        diam.append(float(d))
        vel.append(_mean(exp, "velocity", "vel_nrmse_phase"))
        wss.append(_mean(exp, "wss", "wss_nrmse"))
    order = sorted(range(len(diam)), key=lambda i: diam[i])
    diam = [diam[i] for i in order]; vel = [vel[i] for i in order]; wss = [wss[i] for i in order]
    mid = sorted(diam)[len(diam) // 2] if diam else None

    floor_vel = _mean(args.insample, "velocity", "vel_nrmse_phase")
    floor_wss = _mean(args.insample, "wss", "wss_nrmse")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, y, floor, label in ((axes[0], vel, floor_vel, "Velocity NRMSE (per-phase)"),
                                (axes[1], wss, floor_wss, "WSS NRMSE (per-phase)")):
        ax.plot(diam, y, "o-", color="#2c6fbb", lw=2, ms=8, label="held-out (fold mean)")
        for d, yy in zip(diam, y):
            nat = "interp" if (mid is not None and abs(d - mid) < 1e-9) else "extrap"
            ax.annotate(f"{yy:.2f}\n({nat})", (d, yy), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8)
        if not math.isnan(floor):
            ax.axhline(floor, ls="--", color="#c0392b", lw=1.5,
                       label=f"in-sample floor ({floor:.2f})")
        ax.set_xlabel("held inlet diameter (cm)")
        ax.set_ylabel(label)
        ax.set_xticks(diam)
        ax.set_ylim(0, max([v for v in y if not math.isnan(v)] + [floor if not math.isnan(floor) else 0]) * 1.25)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_title("Generalization vs. in-sample reconstruction", fontsize=10, loc="left")
    fig.tight_layout()

    for d in (FIGURES_DIR, PAPER_FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
    fig.savefig(PAPER_FIGURES_DIR / f"{args.out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {(FIGURES_DIR / (args.out + '.png')).relative_to(PROJECT_ROOT)} (+ .pdf, paper/.pdf)")
    print(f"      diam={diam}  vel_mean={[round(v,3) for v in vel]}  wss_mean={[round(w,3) for w in wss]}")
    print(f"      in-sample floor: vel={floor_vel:.3f}  wss={floor_wss:.3f}")


if __name__ == "__main__":
    main()
