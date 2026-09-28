"""WSS magnitude on the dilation segment of all twelve cases at 1.780 s, one colour scale.

Run from the repository root:
    python scripts/manuscript/generate_wss_montage.py

The wall is unwrapped about the vessel axis of the descending branch (y = 30 mm, z = 0):
horizontal axis x, vertical axis the angle from the anterior (+y) direction, so the anterior
wall is at 0 deg and the posterior wall at +-180 deg. Same segment definition as
generate_cfd_summary.py. Diastole (2.400 s) is drawn as a second figure.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "paper/revision_2026_09/manuscript"
sys.path.insert(0, str(ROOT))
from idealaorta_pinn.data.full_export import load_block

ROWS = {2.0: (1, 2, 3, 10), 2.3: (4, 5, 6, 11), 2.6: (7, 8, 9, 12)}
COLS = ("axisymmetric", "anterior-dominant", "posterior-dominant", "control")
AXI = {2.0: 1, 2.3: 4, 2.6: 7}


def main() -> None:
    """Regenerate manuscript outputs from retained inputs."""
    (MANUSCRIPT / "figures/generated").mkdir(parents=True, exist_ok=True)
    (MANUSCRIPT / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "report/tables").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )
    for t, vmax, tag in ((1780, 20.0, "systole"), (2400, 1.5, "diastole")):
        fig, axes = plt.subplots(
            3, 4, figsize=(6.6, 4.6), sharey=True, layout="constrained"
        )
        for r, (d, cases) in enumerate(ROWS.items()):
            a = load_block(AXI[d], "aneurysm", 1780)
            x0, x1 = a["x"].min() - 0.01, a["x"].max() + 0.01
            for c_i, case in enumerate(cases):
                w = np.vstack(
                    [
                        b[["x", "y", "z", "wss_x", "wss_y", "wss_z"]].to_numpy(float)
                        for b in (
                            load_block(case, "wall", t),
                            load_block(case, "aneurysm", t),
                        )
                        if b is not None
                    ]
                )
                keep = (
                    (w[:, 0] >= x0)
                    & (w[:, 0] <= x1)
                    & ((w[:, 0] > 0.001) | (w[:, 1] > 0.0155))
                )
                w = w[keep]
                theta = np.degrees(np.arctan2(w[:, 2], w[:, 1] - 0.030))
                ax = axes[r, c_i]
                im = ax.scatter(
                    w[:, 0] * 1e3,
                    theta,
                    c=np.linalg.norm(w[:, 3:], axis=1),
                    s=0.25,
                    cmap="turbo",
                    vmin=0,
                    vmax=vmax,
                    rasterized=True,
                )
                ax.set(
                    xlim=(x0 * 1e3, x1 * 1e3),
                    ylim=(-180, 180),
                    yticks=[-180, -90, 0, 90, 180],
                )
                ax.set_title(
                    f"Case {case}" + (f"\n{COLS[c_i]}" if r == 0 else ""), fontsize=7.5
                )
                if r == 2:
                    ax.set_xlabel("x (mm)")
                if c_i == 0:
                    ax.set_ylabel(f"$D_{{in}}$ = {d} cm\nangle (deg)")
        cb = fig.colorbar(
            im, ax=axes, orientation="horizontal", shrink=0.5, pad=0.02, extend="max"
        )
        cb.set_label(f"WSS magnitude (Pa) at {t / 1000:.3f} s")
        out = MANUSCRIPT / "figures/generated" / f"wss_montage_{tag}"
        fig.savefig(out.with_suffix(".pdf"))
        fig.savefig(out.with_suffix(".png"), dpi=170)
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    main()
