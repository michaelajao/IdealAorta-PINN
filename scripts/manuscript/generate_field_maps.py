"""Spatial pressure and aneurysm-wall shear maps: CFD, continuity field and registered RBF.

Run from the repository root:
    python scripts/manuscript/generate_field_maps.py [case target_ms]

Uses the saved checkpoint and observation mask of draw 0 at 2.5 mm and recomputes both
reconstructions through the shared post-processing (unsteady + oracle mu_t pressure,
h = 0.25 mm wall shear). Nothing is trained and no run record is written; the relative
errors printed here must match the saved records for the same draw.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "paper/revision_2026_09/manuscript"
sys.path.insert(0, str(ROOT))
from idealaorta_pinn.reconstruction.interpolation import rbf_fields
from idealaorta_pinn.reconstruction.postprocess import (
    PressureIntegrator,
    wall_interpolant,
    wss_newton2,
)
from idealaorta_pinn.reconstruction.problem import build_problem
from idealaorta_pinn.reconstruction.training import load_field, predict


def main(case: int = 4, target: int = 1780) -> None:
    """Regenerate manuscript outputs from retained inputs."""
    (MANUSCRIPT / "figures/generated").mkdir(parents=True, exist_ok=True)
    (MANUSCRIPT / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "report/tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    name = f"rev2_cont_oracle_c{case:02d}_t{target}_g2.5_s0_i0_w0.01"
    net, scales, spec = load_field(name)
    P = build_problem(case, spec["times"], target, 2.5, 0, save_mask=False)
    saved = np.load(
        ROOT
        / "report/metrics/reconstruction/masks"
        / f"case{case:02d}_t{'-'.join(map(str, spec['times']))}_grid2.5_s0_obs_ids.npy"
    )
    if not np.array_equal(saved, P.obs_ids):
        raise ValueError("Observation mask differs from the saved draw")
    other = next(t for t in P.times_ms if t != target)
    tgt = P.snaps[target]

    fields = {
        "Continuity field": {
            t: predict(net, P.X, t, scales, torch.device("cpu"))[:, :3]
            for t in P.times_ms
        }
    }
    fields["RBF"] = rbf_fields(P)[0]
    integ = PressureIntegrator(P.X, P.wall_xyz, P.dwall)
    p_ref = integ.gauge_ref(tgt.p)
    an = P.wall_aneurysm
    w_ref = np.linalg.norm(tgt.wall_wss[an], axis=1)

    res, summary = {}, {"case": case, "target_ms": target, "seed": 0}
    for k, F in fields.items():
        dUdt = (F[target] - F[other]) / ((target - other) / 1000)
        p = integ.pressure(F[target], dUdt, tgt.mut, "unsteady+mut")
        W = wss_newton2(
            wall_interpolant(P.X, F[target], P.wall_xyz),
            P.wall_xyz,
            P.wall_normals,
            0.25,
        )
        w = np.linalg.norm(W[an], axis=1)
        res[k] = (p, w)
        summary[k] = {
            "p_rel_l2": float(np.linalg.norm(p - p_ref) / np.linalg.norm(p_ref)),
            "wss_mag_rel_l2": float(np.linalg.norm(w - w_ref) / np.linalg.norm(w_ref)),
            "p_abs_err_p95_Pa": float(np.quantile(np.abs(p - p_ref), 0.95)),
            "wss_abs_err_p95_Pa": float(np.quantile(np.abs(w - w_ref), 0.95)),
        }
    print(json.dumps(summary, indent=1))
    np.savez_compressed(
        ROOT / f"tmp/field_maps_c{case:02d}_t{target}.npz",
        p_ref=p_ref,
        w_ref=w_ref,
        **{f"p_{i}": res[k][0] for i, k in enumerate(res)},
        **{f"w_{i}": res[k][1] for i, k in enumerate(res)},
    )
    (ROOT / f"report/tables/eacfm_field_maps_c{case:02d}_t{target}.json").write_text(
        json.dumps(summary, indent=1)
    )

    # Mid-plane slab (|z - z_mid| < 0.35 mm) of the pressure sub-cloud; wall unwrapped about the sac axis.
    Xs = P.X[integ.sub]
    z0 = 0.5 * (Xs[:, 2].min() + Xs[:, 2].max())
    sl = np.abs(Xs[:, 2] - z0) < 0.35e-3
    Wx = P.wall_xyz[an]
    theta = np.degrees(
        np.arctan2(Wx[:, 2], Wx[:, 1] - 0.030)
    )  # 0 deg = anterior (+y), +-180 = posterior

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 5.0), layout="constrained")
    # Error scale: 99th percentile of |error| over both methods on the slab, so isolated
    # near-wall outliers do not wash out the maps (they saturate instead).
    emax = np.quantile(
        np.concatenate([np.abs(res[k][0] - p_ref)[sl] for k in res]), 0.99
    )
    panels = [
        (
            p_ref,
            "CFD gauge pressure (Pa)",
            "viridis",
            (p_ref[sl].min(), p_ref[sl].max()),
        )
    ]
    panels += [
        (res[k][0] - p_ref, f"{k} error (Pa)", "PuOr_r", (-emax, emax)) for k in res
    ]
    for ax, (v, title, cmap, lim) in zip(axes[0], panels):
        im = ax.scatter(
            Xs[sl, 0] * 1e3,
            Xs[sl, 1] * 1e3,
            c=v[sl],
            s=0.4,
            cmap=cmap,
            vmin=lim[0],
            vmax=lim[1],
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=title)
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.9)
    wmax = np.quantile(w_ref, 0.995)
    for ax, (v, title) in zip(
        axes[1],
        [(w_ref, "CFD $|\\tau_w|$ (Pa)")]
        + [(res[k][1], f"{k} $|\\tau_w|$ (Pa)") for k in res],
    ):
        im = ax.scatter(
            Wx[:, 0] * 1e3,
            theta,
            c=v,
            s=0.4,
            cmap="viridis",
            vmin=0,
            vmax=wmax,
            rasterized=True,
        )
        ax.set(
            xlabel="x (mm)",
            ylabel="angle from anterior (deg)",
            title=title,
            ylim=(-180, 180),
            yticks=[-180, -90, 0, 90, 180],
        )
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.9)
    out = MANUSCRIPT / "figures/generated" / f"field_maps_c{case:02d}_t{target}"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=170)

    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=int, nargs="?", default=4, choices=range(1, 10))
    parser.add_argument(
        "target_ms", type=int, nargs="?", default=1780, choices=(1780, 2400)
    )
    args = parser.parse_args()
    main(args.case, args.target_ms)
