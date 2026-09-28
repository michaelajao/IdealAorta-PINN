"""Boundary-condition waveforms (from the prescribed equations) and signed x-WSS along three wall lines.

Run from the repository root:
    python scripts/manuscript/generate_bc_xwss.py

The x-WSS lines are taken from the whole-domain rigid-wall exports at the common instant 1.780 s:
wall nodes within 1 mm of the medial plane (z = 0) on the anterior (y > 30 mm) or posterior
(y < 30 mm) side, and within 1 mm of the transverse plane (y = 30 mm) on the z > 0 side,
restricted to the descending branch and Gaussian-weighted along x (0.8 mm width).
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "paper/revision_2026_09/manuscript"
sys.path.insert(0, str(ROOT))
from idealaorta_pinn.data.full_export import load_block

OUT = MANUSCRIPT / "figures/generated"


def main() -> None:
    """Regenerate manuscript outputs from retained inputs."""
    (MANUSCRIPT / "figures/generated").mkdir(parents=True, exist_ok=True)
    (MANUSCRIPT / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "report/tables").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )

    # ---------------- boundary conditions ----------------
    Q0, T, Pmin, Pmax = 2.39e-4, 0.8, 80.0, 120.0
    t = np.linspace(0, T, 1601)
    sy = t <= 0.35
    Q = np.where(sy, Q0 * np.sin(np.pi * t / 0.35) + 0.05 * Q0, 0.05 * Q0)
    P = np.where(sy, Pmin + (Pmax - Pmin) * np.sin(np.pi * t / 0.35), Pmin)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), layout="constrained")
    for ax, y, lab in (
        (axes[0], Q * 1e6, "Inlet flow rate (mL s$^{-1}$)"),
        (axes[1], P, "Outlet pressure (mmHg)"),
    ):
        ax.plot(t, y, color="#1764ab", lw=1.6)
        for a, b, name in (
            (0.175, 0.180, "systolic window"),
            (0.790, 0.800, "diastolic window"),
        ):
            ax.axvspan(a - 0.004, b + 0.004, color="#e58b23", alpha=0.35, lw=0)
        ax.set(xlabel="Time in cycle, $t-nT$ (s)", ylabel=lab, xlim=(0, T))
        ax.grid(alpha=0.2)
    axes[0].annotate(
        "peak $1.05\\,Q_0$ = 251 mL s$^{-1}$",
        xy=(0.175, 1.05 * Q0 * 1e6),
        xytext=(0.36, 215),
        fontsize=7.5,
        arrowprops={"arrowstyle": "->", "lw": 0.6},
    )
    axes[0].annotate(
        "$0.05\\,Q_0$",
        xy=(0.6, 0.05 * Q0 * 1e6),
        xytext=(0.55, 45),
        fontsize=7.5,
        arrowprops={"arrowstyle": "->", "lw": 0.6},
    )
    axes[1].text(0.183, 81.5, "analysed\nwindows", fontsize=7, color="#b35900")
    fig.savefig(OUT / "bc_profiles.pdf")
    fig.savefig(OUT / "bc_profiles.png", dpi=170)
    plt.close(fig)

    # ---------------- signed x-WSS lines at 1.780 s ----------------
    TOL, Y0 = 1.0e-3, 0.030

    # case colours follow the original manuscript's line plots
    CASE_COL = {
        1: "#ED7D31",
        2: "#2E9BD6",
        3: "#70AD47",
        4: "#1F4E79",
        5: "#548235",
        6: "#B04FB5",
        7: "#FFC000",
        8: "#9E480E",
        9: "#E0001B",
    }
    edges = np.arange(-0.010, 0.060 + 0.5e-3, 0.5e-3)
    centres = 0.5 * (edges[1:] + edges[:-1])

    def line(w, sel, sigma=0.8e-3):
        """Gaussian kernel average of tau_w,x along x (no gaps where the line crosses sparse wall nodes)."""
        x, tx = w[sel, 0], w[sel, 3]
        K = np.exp(-0.5 * ((centres[:, None] - x[None, :]) / sigma) ** 2)
        wsum = K.sum(1)
        out = (K @ tx) / np.maximum(wsum, 1e-12)
        out[wsum < 0.5] = np.nan
        return out

    fig, axes = plt.subplots(
        3, 1, figsize=(6.4, 7.4), sharex=True, layout="constrained"
    )
    titles = (
        "(a) anterior wall, medial plane",
        "(b) posterior wall, medial plane",
        "(c) wall in the transverse plane",
    )
    for case in range(1, 13):
        w = np.vstack(
            [
                b[["x", "y", "z", "wss_x"]].to_numpy(float)
                for b in (
                    load_block(case, "wall", 1780),
                    load_block(case, "aneurysm", 1780),
                )
                if b is not None
            ]
        )
        desc = (w[:, 0] > 0.001) | (w[:, 1] > 0.0155)
        sels = (
            desc & (np.abs(w[:, 2]) < TOL) & (w[:, 1] > Y0),
            desc & (np.abs(w[:, 2]) < TOL) & (w[:, 1] < Y0),
            desc & (np.abs(w[:, 1] - Y0) < TOL) & (w[:, 2] > 0),
        )
        ctrl = case >= 10
        kw = (
            {"color": "0.65", "lw": 0.8, "ls": "--", "zorder": 1}
            if ctrl
            else {"color": CASE_COL[case], "lw": 1.3, "zorder": 2}
        )
        for ax, sel in zip(axes, sels):
            ax.plot(
                centres * 1e3,
                line(w, sel),
                label=None if ctrl else f"Case {case}",
                **kw,
            )
    for ax, ttl in zip(axes, titles):
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(ttl, fontsize=9, loc="left")
        ax.set_ylabel(r"$\tau_{w,x}$ (Pa)")
        ax.set_xlim(-10, 60)
        ax.grid(alpha=0.25)
    axes[2].set_xlabel("x (mm)")
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], color="0.65", lw=0.8, ls="--"))
    l.append("controls (Cases 10-12)")
    fig.legend(h, l, loc="outside upper center", ncol=5, frameon=False, fontsize=7.5)
    fig.savefig(OUT / "xwss_lines.pdf")
    fig.savefig(OUT / "xwss_lines.png", dpi=170)
    plt.close(fig)
    print("saved")


if __name__ == "__main__":
    main()
