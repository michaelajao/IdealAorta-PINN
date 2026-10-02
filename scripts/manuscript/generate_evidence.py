"""Rebuild manuscript evidence from saved records; no training or record mutation.

Run from the repository root:
    python scripts/manuscript/generate_evidence.py
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "paper/revision_2026_09/manuscript"
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "report/metrics/reconstruction/runs"
OUT = MANUSCRIPT / "figures/generated"
TABLES = MANUSCRIPT / "tables"


def main() -> None:
    """Regenerate manuscript outputs from retained inputs."""
    (MANUSCRIPT / "figures/generated").mkdir(parents=True, exist_ok=True)
    (MANUSCRIPT / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "report/tables").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    # near-wall sensitivity runs (_mw) are a separate analysis, not extra draws
    records = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json")) if "_mw" not in p.stem]

    def label(r):
        return r.get("spec", {}).get("arm", r["method"])

    def get(r, path):
        for k in path.split("/"):
            r = r[k]
        return float(r)

    def selected(method, case, time, grid=2.5):
        return [
            r
            for r in records
            if label(r) == method
            and r["problem"]["case"] == case
            and r["problem"]["target_ms"] == time
            and r["problem"]["grid_mm"] == grid
            and (
                r["method"] != "nf"
                or (
                    r["spec"]["steps"] == 20000
                    and r["spec"]["closure"] == "oracle"
                    and r["spec"]["wphys"] == (0 if method == "data" else 0.01)
                )
            )
        ]

    P = "pressure/unsteady+mut/p_rel_l2"
    W = "wss/wss_mag_rel_l2"
    V = "velocity/vel_rel_l2"

    def vals(m, c, t, path, grid=2.5):
        return np.array([get(r, path) for r in selected(m, c, t, grid)])

    def median(m, t, path, grid=2.5):
        return float(np.median([vals(m, c, t, path, grid).mean() for c in range(2, 9)]))

    def table(name, caption, ident, columns, header, rows):
        text = "\\begin{table}[htbp]\n\\centering\n\\small\n"
        text += "\\caption{" + caption + "}\n\\label{" + ident + "}\n"
        text += "\\begin{tabular}{" + columns + "}\n\\toprule\n"
        text += header + " \\\\\n\\midrule\n"
        text += "\n".join(" & ".join(row) + " \\\\" for row in rows)
        text += "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        (TABLES / name).write_text(text, encoding="utf-8")

    # Check complete primary and secondary coverage and recover all draw-level scores.
    flat = []
    for c in range(2, 9):
        for t in (1780, 2400):
            for g, n in ((1.5, 1), (2.5, 3), (4.0, 1)):
                for m in ("cont", "data", "rbf", "linear"):
                    rr = selected(m, c, t, g)
                    if len(rr) != n:
                        raise ValueError(
                            f"Expected {n} records for {(m, c, t, g)}, found {len(rr)}"
                        )
                    for r in rr:
                        q = r["pressure"]["unsteady+mut"]
                        flat.append(
                            {
                                "case": c,
                                "time_ms": t,
                                "grid_mm": g,
                                "method": m,
                                "seed": r["problem"]["seed"],
                                "pressure": get(r, P),
                                "wss": get(r, W),
                                "velocity": get(r, V),
                                "velocity_p95_ms": get(r, "velocity/vel_p95_ms"),
                                "wss_q99_ratio": get(r, "wss/wss_q99_ratio"),
                                "abs_pressure_drop_error_Pa": abs(
                                    q["dp_pred_Pa"] - q["dp_cfd_Pa"]
                                ),
                                "source": r["run_name"],
                            }
                        )
    with (ROOT / "report/tables/eacfm_reconstruction_all_draws.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=flat[0])
        writer.writeheader()
        writer.writerows(flat)

    summary = []
    for t, phase in ((1780, "Systole"), (2400, "Diastole")):
        for path, endpoint in ((V, "Velocity"), (P, "Pressure"), (W, "Wall shear")):
            summary.append(
                [phase, endpoint]
                + [
                    f"{median(m, t, path):.3f}"
                    for m in ("cont", "data", "rbf", "linear", "cfd")
                ]
            )
    table(
        "reconstruction_summary.tex",
        "Median of seven case means at 2.5~mm; each reconstructed case mean uses three paired draws. Entries are node-weighted relative L2 errors. Pressure is conditional on oracle CFD eddy viscosity. Full CFD denotes the same post-processing applied to complete velocity; its error is a reference, not a strict lower bound.",
        "tab:recon_summary",
        "llccccc",
        "Phase & Quantity & Continuity & Data-only & RBF & Linear & Full CFD",
        summary,
    )

    pair_wins = {}
    for path, name, filename in (
        (P, "pressure", "case_pressure.tex"),
        (W, "wall-shear magnitude", "case_shear.tex"),
    ):
        rows = []
        wins = 0
        ratio_of_means = []
        adverse = []
        for c in range(2, 9):
            for t, phase in ((1780, "Sys."), (2400, "Dia.")):
                a = {r["problem"]["seed"]: get(r, path) for r in selected("cont", c, t)}
                b = {r["problem"]["seed"]: get(r, path) for r in selected("rbf", c, t)}
                ratios = np.array([a[k] / b[k] for k in sorted(a)])
                wins += int((ratios < 1).sum())
                ratio_of_means.append(
                    np.mean(list(a.values())) / np.mean(list(b.values()))
                )
                adverse += [(c, t, k, a[k] / b[k]) for k in a if a[k] >= b[k]]
                cells = [
                    f"{vals(m, c, t, path).mean():.3f} ({vals(m, c, t, path).std(ddof=1):.3f})"
                    for m in ("cont", "data", "rbf")
                ]
                rows.append(
                    [str(c), phase]
                    + cells
                    + [f"{ratios.min():.2f}--{ratios.max():.2f}"]
                )
        table(
            filename,
            "Case-wise "
            + name
            + " relative L2 at 2.5~mm, given as the mean (sample standard deviation) over three paired observation/initialization draws. The last column is the range of draw-wise continuity/RBF error ratios, and values above one favor RBF. The spread describes these draws, not patient-population uncertainty.",
            "tab:case_" + ("pressure" if path == P else "shear"),
            "llcccc",
            "Case & Phase & Continuity & Data-only & RBF & Ratio range",
            rows,
        )
        pair_wins[name] = {
            "wins": wins,
            "total": 42,
            "median_ratio": float(np.median(ratio_of_means)),
            "adverse": adverse,
        }

    local = []
    for t, phase in ((1780, "Systole"), (2400, "Diastole")):
        for m, name in [("cont", "Continuity"), ("data", "Data-only"), ("rbf", "RBF")]:
            dp = np.median(
                [
                    np.mean(
                        [
                            abs(
                                r["pressure"]["unsteady+mut"]["dp_pred_Pa"]
                                - r["pressure"]["unsteady+mut"]["dp_cfd_Pa"]
                            )
                            for r in selected(m, c, t)
                        ]
                    )
                    for c in range(2, 9)
                ]
            )
            local.append(
                [
                    phase,
                    name,
                    f"{median(m, t, 'velocity/vel_p95_ms'):.4f}",
                    f"{median(m, t, 'velocity/vel_rel_l2_sac'):.3f}",
                    f"{median(m, t, 'wss/wss_q99_ratio'):.3f}",
                    f"{dp:.2f}",
                ]
            )
    table(
        "local_metrics.tex",
        "Additional error measures at 2.5~mm, reported as medians of seven case means. Velocity p95 is the 95th percentile of pointwise vector-error magnitude; sac velocity is relative L2. The shear ratio compares predicted and CFD 99th percentiles. Pressure-drop error is averaged in absolute value within each case before taking the median.",
        "tab:local_metrics",
        "llcccc",
        "Phase & Method & Velocity p95 & Sac velocity & Shear q99 ratio & Drop error \\\\ & & (m/s) & rel. L2 & & (Pa)",
        local,
    )

    pilot = []
    for c in (1, 9):
        for t, phase in ((1780, "Sys."), (2400, "Dia.")):
            mean = lambda m, p, c=c, t=t: vals(m, c, t, p).mean()
            pilot.append(
                [
                    str(c),
                    phase,
                    f"{mean('cont', P):.3f}",
                    f"{mean('steady', 'pressure_own_head/p_rel_l2'):.3f}",
                    f"{mean('unsteady', 'pressure_own_head/p_rel_l2'):.3f}",
                    f"{mean('cont', W):.3f}",
                    f"{mean('unsteady', 'wss_autograd/wss_mag_rel_l2'):.3f}",
                ]
            )
    table(
        "pilot_endpoints.tex",
        "Pilot endpoints, comparing the pressure and WSS of the continuity field passed through the shared post-processing with the momentum PINNs' own pressure output and automatic-differentiation WSS. Values are relative L2 errors averaged over two draws at systole and one at diastole. The two routes differ by design, so the table compares the end-to-end PINN with the staged reconstruction.",
        "tab:pilot",
        "llccccc",
        r"Case & Phase & \shortstack{Cont. pressure\\(post-proc.)} & \shortstack{Steady head\\(direct)} & "
        r"\shortstack{Unsteady head\\(direct)} & \shortstack{Cont. shear\\(post-proc.)} & "
        r"\shortstack{Unsteady shear\\(autograd)}",
        pilot,
    )

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4), layout="constrained")
    for row, t in enumerate((1780, 2400)):
        for col, (path, title) in enumerate(
            ((P, "Pressure"), (W, "Wall-shear magnitude"))
        ):
            ax = axes[row, col]
            for m, name, color, marker in [
                ("cont", "Continuity", "#1764ab", "o"),
                ("data", "Data-only", "#e58b23", "s"),
                ("rbf", "RBF", "#666666", "^"),
            ]:
                ax.plot(
                    [1.5, 2.5, 4],
                    [median(m, t, path, g) for g in (1.5, 2.5, 4)],
                    label=name,
                    color=color,
                    marker=marker,
                )
            ax.set(
                xlabel="Observation spacing (mm)",
                ylabel="Relative L2 error",
                title=("Systole: " if row == 0 else "Diastole: ") + title,
                xticks=[1.5, 2.5, 4],
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.savefig(OUT / "observation_budget.pdf")
    fig.savefig(OUT / "observation_budget.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.8), layout="constrained")
    forms = ["steady+mut", "unsteady+mut", "steady+lam", "unsteady+lam"]
    for ax, t, phase in zip(axes, [1780, 2400], ["Systole", "Diastole"]):
        for i, (m, name, color) in enumerate(
            [
                ("cont", "Continuity", "#1764ab"),
                ("data", "Data-only", "#e58b23"),
                ("rbf", "RBF", "#777777"),
                ("cfd", "Full CFD", "#6a8f61"),
            ]
        ):
            ax.bar(
                np.arange(4) + (i - 1.5) * 0.19,
                [median(m, t, "pressure/" + f + "/p_rel_l2") for f in forms],
                width=0.19,
                label=name,
                color=color,
            )
        ax.set(
            xticks=np.arange(4),
            xticklabels=[
                "Steady\nCFD eddy viscosity",
                "Unsteady\nCFD eddy viscosity",
                "Steady\nblood viscosity only",
                "Unsteady\nblood viscosity only",
            ],
            ylabel="Pressure relative L2",
            title=phase,
        )
        ax.grid(axis="y", alpha=0.2)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        frameon=False,
        ncol=4,
        fontsize=8,
        loc="outside upper center",
    )
    fig.savefig(OUT / "pressure_information.pdf")
    fig.savefig(OUT / "pressure_information.png", dpi=170)
    plt.close(fig)

    # Spatial illustration from retained checkpoints, evaluated only at hidden CFD nodes.
    import torch

    from idealaorta_pinn.data.full_export import load_block
    from idealaorta_pinn.reconstruction.training import load_field, predict

    torch.set_num_threads(4)
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.4), layout="constrained")
    spatial = []
    for row, (case, t) in enumerate([(4, 1780), (8, 2400)]):
        name = f"rev2_cont_oracle_c{case:02d}_t{t}_g2.5_s0_i0_w0.01"
        net, scales, spec = load_field(name)
        base = load_block(case, "solid", spec["times"][0])
        target = load_block(case, "solid", t)
        interior = np.linalg.norm(base[["u", "v", "w"]].to_numpy(), axis=1) > 0
        xyz = target[["x", "y", "z"]].to_numpy()[interior]
        truth = target[["u", "v", "w"]].to_numpy()[interior]
        obs = np.load(
            ROOT
            / "report/metrics/reconstruction/masks"
            / f"case{case:02d}_t{'-'.join(map(str, spec['times']))}_grid2.5_s0_obs_ids.npy"
        )
        hidden = np.ones(len(xyz), bool)
        hidden[obs] = False
        z0 = 0.5 * (xyz[:, 2].min() + xyz[:, 2].max())
        ids = np.flatnonzero(hidden & (np.abs(xyz[:, 2] - z0) <= 0.00035))
        pred = predict(net, xyz[ids], t, scales, torch.device("cpu"))[:, :3]
        speed = np.linalg.norm(truth[ids], axis=1)
        pspeed = np.linalg.norm(pred, axis=1)
        error = np.linalg.norm(pred - truth[ids], axis=1)
        vmax = max(speed.max(), pspeed.max())
        for col, (values, title, cmap, limit) in enumerate(
            [
                (speed, "CFD speed", "viridis", vmax),
                (pspeed, "Continuity-field speed", "viridis", vmax),
                (error, "Vector error magnitude", "magma", error.max()),
            ]
        ):
            ax = axes[row, col]
            im = ax.scatter(
                xyz[ids, 0] * 1000,
                xyz[ids, 1] * 1000,
                c=values,
                s=0.3,
                cmap=cmap,
                vmin=0,
                vmax=limit,
                rasterized=True,
            )
            ax.set_aspect("equal")
            ax.set(
                xlabel="x (mm)",
                ylabel="y (mm)",
                title=f"Case {case}, "
                + ("systole" if t == 1780 else "diastole")
                + "\n"
                + title,
            )
            fig.colorbar(
                im, ax=ax, orientation="horizontal", pad=0.06, label="m/s", shrink=0.9
            )
        spatial.append(
            {
                "case": case,
                "time_ms": t,
                "seed": 0,
                "n_plane_hidden": len(ids),
                "slab_halfwidth_mm": 0.35,
                "vector_error_max_ms": float(error.max()),
                "vector_error_p95_ms": float(np.quantile(error, 0.95)),
            }
        )
    fig.savefig(OUT / "hidden_velocity_examples.pdf")
    fig.savefig(OUT / "hidden_velocity_examples.png", dpi=180)
    plt.close(fig)

    audit = {
        "record_count": len(records),
        "methods": dict(Counter(r["method"] for r in records)),
        "paired_draws": pair_wins,
        "spatial_examples": spatial,
        "robustness_coverage": [
            {k: r["problem"][k] for k in ("case", "target_ms", "grid_mm", "seed")}
            | {"method": r["method"]}
            for r in records
            if r["method"] in ("rbf_tuned", "rbf_spacetime")
        ],
    }
    (ROOT / "report/tables/eacfm_evidence_audit.json").write_text(
        json.dumps(audit, indent=2)
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
