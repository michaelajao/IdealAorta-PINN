"""Plane-wise (D1-D8) mean speed and k for every case, from the corrected slice workbook.

Run from the repository root:
    python scripts/manuscript/generate_plane_profiles.py

Replaces the original box plots, which carried Mann-Whitney p-values for 9 vs 3 designed
geometries; every case is drawn instead, and no significance test is applied.
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "paper/revision_2026_09/manuscript"


def main() -> None:
    """Regenerate manuscript outputs from retained inputs."""
    (MANUSCRIPT / "figures/generated").mkdir(parents=True, exist_ok=True)
    (MANUSCRIPT / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "report/tables").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/results_on_slices.csv")
    DIAM = {
        c: d
        for d, cs in ((2.0, (1, 2, 3, 10)), (2.3, (4, 5, 6, 11)), (2.6, (7, 8, 9, 12)))
        for c in cs
    }
    COL = {2.0: "#1764ab", 2.3: "#e58b23", 2.6: "#3b8f3b"}
    STYLE = {
        0: ("-", "o", "axisymmetric"),
        1: ("--", "^", "anterior-dominant"),
        2: (":", "v", "posterior-dominant"),
        3: ("-.", "s", "control"),
    }
    planes = [f"D{i}" for i in range(1, 9)]

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), layout="constrained")
    for case in range(1, 13):
        d = df[df.case_id == case].set_index("slice").loc[planes]
        kind = 3 if case >= 10 else (case - 1) % 3
        ls, mk, _name = STYLE[kind]
        kw = {
            "color": COL[DIAM[case]],
            "ls": ls,
            "marker": mk,
            "ms": 4,
            "lw": 1.2 if kind == 3 else 1.4,
            "markerfacecolor": "white" if kind == 3 else COL[DIAM[case]],
        }
        axes[0].plot(range(1, 9), d["avg_vel_mps"], **kw)
        axes[1].plot(range(1, 9), d["tke"] * 1e3, **kw)
    for ax, lab in zip(
        axes,
        (
            "Plane-averaged speed (m s$^{-1}$)",
            "Plane-averaged $k$ ($10^{-3}$ m$^2$ s$^{-2}$)",
        ),
    ):
        ax.set(xticks=range(1, 9), xticklabels=planes, ylabel=lab, xlabel="Plane")
        ax.axvspan(4.5, 5.5, color="0.9", zorder=0)
        ax.grid(alpha=0.2)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=COL[d], lw=2, label=f"$D_{{in}}$ = {d} cm") for d in COL
    ]
    handles += [
        Line2D([], [], color="k", ls=STYLE[k][0], marker=STYLE[k][1], label=STYLE[k][2])
        for k in (0, 1, 2)
    ]
    handles += [
        Line2D(
            [],
            [],
            color="k",
            ls="-.",
            marker="s",
            markerfacecolor="white",
            label="control",
        )
    ]
    fig.legend(
        handles=handles, frameon=False, fontsize=7, loc="outside lower center", ncol=4
    )
    out = MANUSCRIPT / "figures/generated/plane_profiles"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=170)
    print("saved", out)

    plt.close(fig)


if __name__ == "__main__":
    main()
