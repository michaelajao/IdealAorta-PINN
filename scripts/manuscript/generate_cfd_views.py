"""Streamlines (medial and transverse planes) and anterior/posterior WSS views for all twelve cases
at the common instant 1.780 s, from the whole-domain rigid-wall exports, on shared colour scales.

Run from the repository root:
    python scripts/manuscript/generate_cfd_views.py

These replace the CFD-Post images of the original manuscript, which were exported at case-specific
times (1.775-1.815 s) with separate colour scales.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "paper/revision_2026_09/manuscript"
sys.path.insert(0, str(ROOT))
from idealaorta_pinn.data.full_export import load_block

OUT = MANUSCRIPT / "figures/generated"
T = 1780
Y0 = 0.030
ORDER = [
    (1, 4, 7),
    (2, 5, 8),
    (3, 6, 9),
    (10, 11, 12),
]  # rows: morphology, cols: diameter
ROWLAB = ["axisymmetric", "anterior-dominant", "posterior-dominant", "control"]
DIAM = {1: 2.0, 4: 2.3, 7: 2.6}
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "pdf.fonttype": 42})


def plane_field(case, plane, h=0.35e-3, slab=0.45e-3):
    v = load_block(case, "solid", T, ["x", "y", "z", "u", "v", "w"])
    X = v[["x", "y", "z"]].to_numpy(float)
    U = v[["u", "v", "w"]].to_numpy(float)
    if plane == "xy":
        m = np.abs(X[:, 2]) < slab
        P, Q = X[m][:, [0, 1]], U[m][:, [0, 1]]
    else:  # transverse plane through the descending-branch axis
        m = np.abs(X[:, 1] - Y0) < slab
        P, Q = X[m][:, [0, 2]], U[m][:, [0, 2]]
    g0 = np.arange(P[:, 0].min(), P[:, 0].max(), h)
    g1 = np.arange(P[:, 1].min(), P[:, 1].max(), h)
    G0, G1 = np.meshgrid(g0, g1)
    d, _ = cKDTree(P).query(np.c_[G0.ravel(), G1.ravel()])
    inside = (d < 0.9e-3).reshape(G0.shape)
    gu = griddata(P, Q[:, 0], (G0, G1), method="linear")
    gv = griddata(P, Q[:, 1], (G0, G1), method="linear")
    gu[~inside] = np.nan
    gv[~inside] = np.nan
    return g0, g1, gu, gv, inside


def _stream_panel(ax, case, plane, norm):
    g0, g1, gu, gv, inside = plane_field(case, plane)
    sp = np.hypot(gu, gv)
    ax.contourf(
        g0 * 1e3, g1 * 1e3, inside.astype(float), levels=[0.5, 1.5], colors=["#f4f4f4"]
    )
    ax.contour(
        g0 * 1e3,
        g1 * 1e3,
        inside.astype(float),
        levels=[0.5],
        colors="0.3",
        linewidths=0.6,
    )
    strm = ax.streamplot(
        g0 * 1e3,
        g1 * 1e3,
        np.nan_to_num(gu),
        np.nan_to_num(gv),
        color=np.nan_to_num(sp),
        cmap="turbo",
        norm=norm,
        density=2.2,
        linewidth=0.55,
        arrowsize=0.5,
        broken_streamlines=False,
    )
    ax.set_aspect("equal")
    if plane == "xy":
        ax.set_xlim(-45, 75)
    else:
        ax.set_xlim(-15, 62)
    ax.tick_params(labelsize=7, length=2)
    return strm


def streamlines_by_diameter(vmax=1.6):
    norm = Normalize(0, vmax)
    for c, dlab in enumerate(("2.0", "2.3", "2.6")):
        fig, axes = plt.subplots(
            4,
            2,
            figsize=(6.5, 7.4),
            layout="constrained",
            gridspec_kw={"width_ratios": [1.15, 1.0]},
        )
        for r, row in enumerate(ORDER):
            case = row[c]
            strm = _stream_panel(axes[r, 0], case, "xy", norm)
            _stream_panel(axes[r, 1], case, "xz", norm)
            axes[r, 0].set_ylabel(f"Case {case}, {ROWLAB[r]}\ny (mm)", fontsize=8)
            axes[r, 1].set_ylabel("z (mm)", fontsize=8)
            if r == 0:
                axes[r, 0].set_title("medial plane ($xy$)", fontsize=8.5)
                axes[r, 1].set_title("transverse plane ($y=30$ mm)", fontsize=8.5)
            if r == 3:
                for ax in axes[r]:
                    ax.set_xlabel("x (mm)", fontsize=8)
        cb = fig.colorbar(
            strm.lines,
            ax=axes,
            orientation="horizontal",
            shrink=0.6,
            pad=0.01,
            extend="max",
        )
        cb.set_label(
            f"speed (m s$^{{-1}}$), systolic phase, inlet diameter {dlab} cm", fontsize=8
        )
        cb.ax.tick_params(labelsize=7.5)
        tag = dlab.replace(".", "p")
        fig.savefig(OUT / f"streamlines_{tag}.pdf", dpi=300)
        fig.savefig(OUT / f"streamlines_{tag}.png", dpi=170)
        plt.close(fig)
        print("saved streamlines", dlab)


def wss_views(fname, vmax=20.0):
    """Anterior (seen from +y) and posterior (seen from -y) views, one figure per inlet diameter."""
    norm = Normalize(0, vmax)
    for c, dlab in enumerate(("2.0", "2.3", "2.6")):
        fig, axes = plt.subplots(4, 2, figsize=(6.4, 6.9), layout="constrained")
        a = load_block({0: 1, 1: 4, 2: 7}[c], "aneurysm", T)
        x0, x1 = a["x"].min() - 0.006, a["x"].max() + 0.006
        for r, row in enumerate(ORDER):
            case = row[c]
            w = np.vstack(
                [
                    b[["x", "y", "z", "wss_x", "wss_y", "wss_z"]].to_numpy(float)
                    for b in (
                        load_block(case, "wall", T),
                        load_block(case, "aneurysm", T),
                    )
                    if b is not None
                ]
            )
            w = w[
                (w[:, 0] >= x0)
                & (w[:, 0] <= x1)
                & ((w[:, 0] > 0.001) | (w[:, 1] > 0.0155))
            ]
            mag = np.linalg.norm(w[:, 3:], axis=1)
            for k, (side, sgn) in enumerate(
                (("anterior view", 1), ("posterior view", -1))
            ):
                ax = axes[r, k]
                sel = sgn * (w[:, 1] - Y0) > 0
                o = np.argsort(sgn * w[sel, 1])
                sc = ax.scatter(
                    w[sel, 0][o] * 1e3,
                    w[sel, 2][o] * 1e3,
                    c=mag[sel][o],
                    s=2.2,
                    cmap="turbo",
                    norm=norm,
                    rasterized=True,
                )
                ax.set_aspect("equal")
                ax.tick_params(labelsize=7, length=2)
                if r == 0:
                    ax.set_title(side, fontsize=8.5)
                if r == 3:
                    ax.set_xlabel("x (mm)", fontsize=8)
                if k == 0:
                    ax.set_ylabel(f"Case {case}, {ROWLAB[r]}\nz (mm)", fontsize=8)
        cb = fig.colorbar(
            sc, ax=axes, orientation="horizontal", shrink=0.6, pad=0.01, extend="max"
        )
        cb.set_label(
            f"WSS magnitude (Pa) at 1.780 s, inlet diameter {dlab} cm", fontsize=8
        )
        cb.ax.tick_params(labelsize=7.5)
        fig.savefig(OUT / f"{fname}_{dlab.replace('.', 'p')}.pdf", dpi=300)
        fig.savefig(OUT / f"{fname}_{dlab.replace('.', 'p')}.png", dpi=170)
        plt.close(fig)
        print("saved", fname, dlab)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["sl", "wss"]
    if set(which) - {"sl", "wss"}:
        raise SystemExit("Expected sl and/or wss")
    if "sl" in which:
        streamlines_by_diameter()
    if "wss" in which:
        wss_views("wss_views")
