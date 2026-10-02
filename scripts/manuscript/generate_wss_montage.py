"""WSS magnitude (systole, diastole) and TAWSS on the dilation segment of all twelve cases.

Run from the repository root:
    python scripts/manuscript/generate_wss_montage.py

The wall is unwrapped about the vessel axis of the descending branch (y = 30 mm, z = 0):
horizontal axis x, vertical axis the angle from the anterior (+y) direction, so the anterior
wall is at 0 deg and the posterior wall at +-180 deg. Same segment definition as
generate_cfd_summary.py. The instantaneous WSS comes from the whole-domain exports at 1.780 s
(systolic phase) and 2.400 s (diastolic phase); TAWSS comes from the solver's time-averaged
wall shear exported at the same wall nodes (data/raw/TAWSS 30.09.2026), drawn on a log scale.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "paper/revision_2026_09/manuscript"
TAWSS_DIR = ROOT / "data/raw/TAWSS 30.09.2026"
sys.path.insert(0, str(ROOT))
from idealaorta_pinn.data.full_export import load_block

ROWS = {2.0: (1, 2, 3, 10), 2.3: (4, 5, 6, 11), 2.6: (7, 8, 9, 12)}
COLS = ("axisymmetric", "anterior-dominant", "posterior-dominant", "control")
AXI = {2.0: 1, 2.3: 4, 2.6: 7}


def read_tawss(case: int) -> np.ndarray:
    """(N, 4) array of x, y, z (m) and TAWSS (Pa) from a CFD-Post wall export, all zones stacked."""
    path = next(p for p in TAWSS_DIR.glob("*.csv") if p.stem.lower() == f"case {case}")
    rows, in_data = [], False
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("["):
            in_data = line == "[Data]"
        elif in_data and line and (line[0].isdigit() or line[0] == "-"):
            rows.append([float(v) for v in line.split(",")])
    return np.asarray(rows)


def segment(w: np.ndarray, d: float) -> np.ndarray:
    """Rows of ``w`` on the dilation segment of inlet diameter ``d`` (descending branch only)."""
    a = load_block(AXI[d], "aneurysm", 1780)
    x0, x1 = a["x"].min() - 0.01, a["x"].max() + 0.01
    keep = (w[:, 0] >= x0) & (w[:, 0] <= x1) & ((w[:, 0] > 0.001) | (w[:, 1] > 0.0155))
    return w[keep]


def montage(values, norm, label: str, out: str, extend: str) -> None:
    """Three diameters x four morphologies, unwrapped wall, one colour scale."""
    fig, axes = plt.subplots(3, 4, figsize=(6.6, 4.6), sharey=True, layout="constrained")
    for r, (d, cases) in enumerate(ROWS.items()):
        a = load_block(AXI[d], "aneurysm", 1780)
        x0, x1 = a["x"].min() - 0.01, a["x"].max() + 0.01
        for c_i, case in enumerate(cases):
            w = segment(values(case), d)
            theta = np.degrees(np.arctan2(w[:, 2], w[:, 1] - 0.030))
            ax = axes[r, c_i]
            im = ax.scatter(w[:, 0] * 1e3, theta, c=w[:, 3], s=0.25, cmap="turbo", norm=norm, rasterized=True)
            ax.set(xlim=(x0 * 1e3, x1 * 1e3), ylim=(-180, 180), yticks=[-180, -90, 0, 90, 180])
            ax.set_title(f"Case {case}" + (f"\n{COLS[c_i]}" if r == 0 else ""), fontsize=7.5)
            if r == 2:
                ax.set_xlabel("x (mm)")
            if c_i == 0:
                ax.set_ylabel(f"$D_{{in}}$ = {d} cm\nangle (deg)")
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", shrink=0.5, pad=0.02, extend=extend)
    if isinstance(norm, LogNorm):
        ticks = [0.1, 0.3, 1, 3, 10]
        cb.set_ticks(ticks, labels=[f"{v:g}" for v in ticks])
        cb.minorticks_off()
    cb.set_label(label)
    path = MANUSCRIPT / "figures/generated" / out
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=170)
    plt.close(fig)
    print("saved", path)


def wss_values(t: int):
    def load(case: int) -> np.ndarray:
        w = np.vstack([b[["x", "y", "z", "wss_x", "wss_y", "wss_z"]].to_numpy(float)
                       for b in (load_block(case, "wall", t), load_block(case, "aneurysm", t)) if b is not None])
        return np.column_stack([w[:, :3], np.linalg.norm(w[:, 3:], axis=1)])
    return load


def tawss_table() -> None:
    """TAWSS percentiles over the dilation segment and sac-body medians by wall side (tables/tawss_summary.tex).

    The sac body is the middle third of the segment; anterior and posterior are within 60 deg of
    0 and 180 deg.
    """
    rows = []
    morph = dict(zip((1, 2, 3, 10), COLS)) | dict(zip((4, 5, 6, 11), COLS)) | dict(zip((7, 8, 9, 12), COLS))
    for d, cases in ROWS.items():
        a = load_block(AXI[d], "aneurysm", 1780)
        x0, x1 = a["x"].min(), a["x"].max()
        for case in cases:
            w = read_tawss(case)
            w = w[(w[:, 0] >= x0) & (w[:, 0] <= x1) & ((w[:, 0] > 0.001) | (w[:, 1] > 0.0155))]
            t, th = w[:, 3], np.degrees(np.arctan2(w[:, 2], w[:, 1] - 0.030))
            body = (w[:, 0] > x0 + (x1 - x0) / 3) & (w[:, 0] < x1 - (x1 - x0) / 3)
            ant, post = t[body & (np.abs(th) < 60)], t[body & (np.abs(th) > 120)]
            rows.append(f"{case} & {d:.1f} & {morph[case]} & {np.percentile(t, 5):.2f} & {np.median(t):.2f} & "
                        f"{np.percentile(t, 95):.2f} & {np.median(ant):.2f} & {np.median(post):.2f} \\\\")
    head = [
        "% Generated by scripts/manuscript/generate_wss_montage.py from data/raw/TAWSS 30.09.2026.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{TAWSS (Pa) on the dilation segment. Percentiles are over the whole segment; the sac-body "
        r"medians use its middle third, within $60^\circ$ of the anterior or posterior direction.}",
        r"\label{tab:tawss}",
        r"\begin{tabular}{cclccccc}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{Segment} & \multicolumn{2}{c}{Sac body, median} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
        r"Case & $D_{\rm in}$ (cm) & Morphology & 5th pct. & Median & 95th pct. & Anterior & Posterior \\",
        r"\midrule",
    ]
    tail = [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (MANUSCRIPT / "tables/tawss_summary.tex").write_text("\n".join(head + rows + tail), encoding="utf-8")
    print("\n".join(rows))


def main() -> None:
    """Regenerate the WSS and TAWSS montages."""
    (MANUSCRIPT / "figures/generated").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
    montage(wss_values(1780), plt.Normalize(0, 20.0), "WSS magnitude (Pa), systolic phase",
            "wss_montage_systole", "max")
    montage(wss_values(2400), plt.Normalize(0, 1.5), "WSS magnitude (Pa), diastolic phase",
            "wss_montage_diastole", "max")
    montage(read_tawss, LogNorm(0.1, 12.0), "TAWSS (Pa), log scale", "tawss_montage", "both")
    tawss_table()


if __name__ == "__main__":
    main()
