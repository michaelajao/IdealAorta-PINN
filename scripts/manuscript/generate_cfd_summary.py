"""Common-time CFD descriptors for all twelve cases from the whole-domain rigid-wall exports.

Run from the repository root:
    python scripts/manuscript/generate_cfd_summary.py

Every case is evaluated at 1.780 s (systole) and 2.400 s (diastole). The dilation segment is
the axial window x in [x_min, x_max] of the aneurysm wall zone (descending branch only; the
inlet leg, which overlaps that x range near x = 0 at y < 15.5 mm, is excluded) of the axisymmetric case with
the same inlet diameter, so controls are measured over the same stretch of vessel. Values are
node-weighted because the exports carry no cell volumes or face areas.
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "paper/revision_2026_09/manuscript"
sys.path.insert(0, str(ROOT))
from idealaorta_pinn.data.full_export import load_block

RHO, MU, Q0, T = 1060.0, 0.0035, 2.39e-4, 0.8
DIAM = {
    c: d
    for d, cs in ((2.0, (1, 2, 3, 10)), (2.3, (4, 5, 6, 11)), (2.6, (7, 8, 9, 12)))
    for c in cs
}
BETA = {1: 1.00, 2: 0.48, 3: 2.08, 4: 1.00, 5: 0.48, 6: 2.08, 7: 1.00, 8: 0.48, 9: 2.08}
AXI = {2.0: 1, 2.3: 4, 2.6: 7}


def main() -> None:
    """Regenerate manuscript outputs from retained inputs."""
    (MANUSCRIPT / "figures/generated").mkdir(parents=True, exist_ok=True)
    (MANUSCRIPT / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "report/tables").mkdir(parents=True, exist_ok=True)
    window = {}
    for d, c in AXI.items():
        a = load_block(c, "aneurysm", 1780)
        window[d] = (float(a["x"].min()), float(a["x"].max()))

    rows = []
    for case in range(1, 13):
        d = DIAM[case]
        x0, x1 = window[d]
        row = {
            "case": case,
            "inlet_cm": d,
            "beta": BETA.get(case, ""),
            "segment_x_mm": f"{x0 * 1e3:.1f}-{x1 * 1e3:.1f}",
        }
        for t, tag in ((1780, "sys"), (2400, "dia")):
            v = load_block(case, "solid", t)
            X = v[["x", "y", "z"]].to_numpy(float)
            U = v[["u", "v", "w"]].to_numpy(float)
            speed = np.linalg.norm(U, axis=1)
            seg = (
                (X[:, 0] >= x0)
                & (X[:, 0] <= x1)
                & ((X[:, 0] > 0.001) | (X[:, 1] > 0.0155))
                & (speed > 0)
            )
            walls = [load_block(case, b, t) for b in ("wall", "aneurysm")]
            w = np.vstack(
                [
                    b[["x", "y", "wss_x", "wss_y", "wss_z"]].to_numpy(float)
                    for b in walls
                    if b is not None
                ]
            )
            wseg = (
                (w[:, 0] >= x0)
                & (w[:, 0] <= x1)
                & ((w[:, 0] > 0.001) | (w[:, 1] > 0.0155))
            )
            wss = np.linalg.norm(w[wseg, 2:], axis=1)
            pin = float(load_block(case, "inlet", t)["p"].mean())
            pout = float(load_block(case, "outlet", t)["p"].mean())
            row |= {
                f"dp_{tag}_Pa": pin - pout,
                f"seg_speed_mean_{tag}": float(speed[seg].mean()),
                f"seg_reverse_frac_{tag}": float((U[seg, 0] < 0).mean()),
                f"seg_mut_ratio_{tag}": float(
                    v["mu_t"].to_numpy(float)[seg].mean() / MU
                ),
                f"seg_k_mean_{tag}": float(v["k"].to_numpy(float)[seg].mean())
                if "k" in v
                else np.nan,
                f"wss_median_{tag}": float(np.median(wss)),
                f"wss_q95_{tag}": float(np.quantile(wss, 0.95)),
                f"wss_q99_{tag}": float(np.quantile(wss, 0.99)),
                "n_wall_seg": int(wseg.sum()),
            }
        rows.append(row)
        print(
            case,
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
            flush=True,
        )

    out = ROOT / "report/tables/eacfm_cfd_summary.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # Nondimensional groups of the prescribed inflow.
    groups = []
    for d in (2.0, 2.3, 2.6):
        D = d / 100
        A = np.pi * D**2 / 4
        Upk, Ubase = 1.05 * Q0 / A, 0.05 * Q0 / A
        groups.append(
            (
                d,
                Upk,
                RHO * Upk * D / MU,
                RHO * Ubase * D / MU,
                D / 2 * np.sqrt(2 * np.pi / T * RHO / MU),
            )
        )
        print(
            f"D={d} cm U_peak={Upk:.3f} Re_peak={groups[-1][2]:.0f} Re_base={groups[-1][3]:.0f} Wo={groups[-1][4]:.1f}"
        )

    label = lambda r: "Control" if r["beta"] == "" else f"{r['beta']:.2f}"
    body = []
    for r in rows:
        body.append(
            " & ".join(
                [
                    str(r["case"]),
                    f"{r['inlet_cm']:.1f}",
                    label(r),
                    f"{r['dp_sys_Pa']:.0f}",
                    f"{r['dp_dia_Pa']:.1f}",
                    f"{r['wss_median_sys']:.2f}",
                    f"{r['wss_q95_sys']:.1f}",
                    f"{r['wss_median_dia']:.3f}",
                    f"{r['wss_q95_dia']:.2f}",
                    f"{r['seg_speed_mean_sys']:.3f}",
                    f"{100 * r['seg_reverse_frac_sys']:.0f}",
                    f"{100 * r['seg_reverse_frac_dia']:.0f}",
                    f"{r['seg_mut_ratio_sys']:.1f}",
                ]
            )
            + r" \\"
        )
    tex = (
        r"""% Generated by generate_cfd_summary.py from data/processed/full (rigid-wall CFD); CSV: report/tables/eacfm_cfd_summary.csv
\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{3.2pt}
\caption{Common-time descriptors of the rigid-wall CFD at systole (1.780~s) and diastole (2.400~s). $\Delta p$ is the difference between the mean static pressure on the inlet and outlet faces. Wall shear, mean speed, the reversed-flow fraction (nodes with $u_x<0$, the local downstream direction in the straight segment) and the mean eddy-to-molecular viscosity ratio are evaluated over the same axial segment for every case with a given inlet diameter, namely the extent of the dilation of the axisymmetric case, so the controls are measured over the corresponding stretch of undilated vessel. All values are node-weighted descriptive summaries of single designed geometries.}
\label{tab:cfd_summary}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ccc rr cc cc c cc c}
\toprule
 & & & \multicolumn{2}{c}{$\Delta p$ (Pa)} & \multicolumn{2}{c}{Systolic $\tau_w$ (Pa)} & \multicolumn{2}{c}{Diastolic $\tau_w$ (Pa)} & Mean speed & \multicolumn{2}{c}{Reversed flow (\%)} & $\overline{\mu_t}/\mu$ \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){11-12}
Case & $D_{\rm in}$ (cm) & $\beta$ & Sys. & Dia. & Median & 95th pct. & Median & 95th pct. & sys. (m~s$^{-1}$) & Sys. & Dia. & sys. \\
\midrule
"""
        + "\n".join(body)
        + r"""
\bottomrule
\end{tabular}%
}
\end{table}
"""
    )
    (MANUSCRIPT / "tables/cfd_summary.tex").write_text(tex, encoding="utf-8")
    g = "\n".join(
        f"{d:.1f} & {u:.3f} & {re:.0f} & {rb:.0f} & {wo:.1f} \\\\"
        for d, u, re, rb, wo in groups
    )
    (MANUSCRIPT / "tables/flow_groups.tex").write_text(
        r"""% Generated by generate_cfd_summary.py from the prescribed waveform and fluid properties.
\begin{table}[htbp]
\centering
\small
\caption{Inflow scales implied by the prescribed waveform ($\rho=1060$~kg~m$^{-3}$, $\mu=3.5\times10^{-3}$~Pa~s, $T=0.8$~s). Velocities are cross-sectional means at the inlet; $\mathrm{Re}=\rho \bar U D_{\rm in}/\mu$ and $\mathrm{Wo}=(D_{\rm in}/2)\sqrt{2\pi\rho/(\mu T)}$.}
\label{tab:flow_groups}
\begin{tabular}{ccccc}
\toprule
$D_{\rm in}$ (cm) & Peak $\bar U$ (m~s$^{-1}$) & Peak Re & Diastolic-plateau Re & Wo \\
\midrule
"""
        + g
        + r"""
\bottomrule
\end{tabular}
\end{table}
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
