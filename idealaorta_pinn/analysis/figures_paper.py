"""Additional paper figures: WSS maps and the parametric WSS sweep.

These complement ``figures.plane_comparison`` (velocity planes) and
``figures.convergence_curves``. They reuse the inference helpers in
``analysis.metrics`` so the WSS is computed exactly as the trained model defines
it. Saved as static PNGs under ``report/figures/<experiment>/`` (the caller
passes ``out_dir``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import FIGURES_DIR
from ..data.cache import load_points
from ..data.geometry import compute_wall_normals
from ..data.registry import CaseRecord, cases_by_id
from .metrics import predict_wss_physical
from .predict import TrainedModel


_KIND_AXES = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}


def _three_panel(ca, cb, cfd, pinn, err, titles, cmaps, vmaxes,
                 unit, out_path: Path, axis_labels=("", ""), diverging=False, vmin=0.0):
    """CFD | Surrogate | Absolute-error scatter maps on a projected plane.

    Clean publication panels (named columns, no exposed [min,max] debug ranges,
    no baked-in suptitle -- the LaTeX caption supplies case/phase/disease state)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, vals, title, cmap, vm in zip(axes, (cfd, pinn, err), titles, cmaps, vmaxes):
        lo = -vm if diverging else vmin
        sc = ax.scatter(ca, cb, c=vals, s=4, cmap=cmap, vmin=lo, vmax=vm)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(axis_labels[0], fontsize=11)
        ax.set_ylabel(axis_labels[1], fontsize=11)
        ax.tick_params(labelsize=9)
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.8, label=unit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def wss_map(model: TrainedModel, records: Sequence[CaseRecord], case_id: int, phase: str,
            kind: str = "XZ", max_points: int = 20_000, out_dir: Path = FIGURES_DIR) -> Path:
    """CFD | Surrogate | error wall-shear-stress magnitude (Pa), wall points
    projected onto the ``kind`` plane (``XY`` looks down the vessel; ``XZ`` is the
    long-axis side view)."""
    rec = cases_by_id(records)[case_id]
    df = load_points(rec, "WSS", phase)
    if df is None or "wss" not in df.columns:
        raise ValueError(f"no WSS data for case {case_id} {phase}")
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    coords = df[["x", "y", "z"]].to_numpy(float)
    cfd = df["wss"].to_numpy(float)
    normals = compute_wall_normals(coords)
    beta = rec.beta if rec.beta is not None else 1.0
    pinn = predict_wss_physical(model, coords, normals, rec.inlet_diameter_cm,
                                rec.disease_flag, phase, beta=beta)["wss_magnitude"]
    err = np.abs(pinn - cfd)
    a, b = _KIND_AXES[kind]
    vm = float(np.percentile(cfd, 99)) or 1e-9
    return _three_panel(
        coords[:, a], coords[:, b], cfd, pinn, err,
        titles=("CFD WSS", "PINN WSS", "Absolute error"),
        cmaps=("inferno", "inferno", "magma"),
        vmaxes=(vm, vm, float(np.percentile(err, 99)) or 1e-9),
        unit="Pa", axis_labels=("xyz"[a] + " (m)", "xyz"[b] + " (m)"),
        out_path=(Path(out_dir) / f"case{case_id:02d}_{phase}_{kind}_wss_map.png"))


# axisymmetric / anterior / posterior diseased case ids by inlet diameter (cm)
_SYMMETRY_SERIES = {"axisymmetric": {2.0: 1, 2.3: 4, 2.6: 7},
                    "anterior": {2.0: 2, 2.3: 5, 2.6: 8},
                    "posterior": {2.0: 3, 2.3: 6, 2.6: 9}}


def mu_sweep(model: TrainedModel, records: Sequence[CaseRecord], symmetry: str,
             phase: str, out_dir: Path = FIGURES_DIR, n: int = 25) -> Path:
    """Parametric response: on a fixed geometry (the held $2.3$ cm case of the
    given symmetry series), sweep the inlet-diameter parameter $d^{*}$ over
    [2.0, 2.6] cm and plot the predicted peak WSS, against the per-diameter CFD
    peak WSS of the matching-symmetry cases. The geometry is held fixed, so the
    curve isolates the network's response to $d^{*}$; the per-geometry markers
    (distinct meshes) give the physical trend for context."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = _SYMMETRY_SERIES[symmetry]
    geom_rec = cases_by_id(records)[series[2.3]]            # fixed 2.3 cm geometry
    wdf = load_points(geom_rec, "WSS", phase)
    coords = wdf[["x", "y", "z"]].to_numpy(float)
    normals = compute_wall_normals(coords)
    beta = geom_rec.beta if geom_rec.beta is not None else 1.0

    ds = np.linspace(2.0, 2.6, n)
    pred_peak = []
    for d in ds:
        w = predict_wss_physical(model, coords, normals, float(d), geom_rec.disease_flag,
                                 phase, beta=beta)["wss_magnitude"]
        pred_peak.append(float(np.percentile(w, 99)))

    # CFD truth and the surrogate evaluated on EACH REAL geometry (matching d*),
    # so the surrogate-vs-CFD comparison is like-for-like per diameter -- unlike the
    # fixed-geometry sweep curve, whose 2.0/2.6 values sit on the 2.3 cm mesh.
    cfd_d, cfd_peak, pinn_d, pinn_real_peak = [], [], [], []
    for d_cm, cid in series.items():
        rec = cases_by_id(records)[cid]
        cdf = load_points(rec, "WSS", phase)
        if cdf is None or "wss" not in cdf.columns:
            continue
        cfd_d.append(d_cm)
        cfd_peak.append(float(np.percentile(cdf["wss"].to_numpy(float), 99)))
        rc = cdf[["x", "y", "z"]].to_numpy(float)
        rn = compute_wall_normals(rc)
        rbeta = rec.beta if rec.beta is not None else 1.0
        w = predict_wss_physical(model, rc, rn, float(d_cm), rec.disease_flag,
                                 phase, beta=rbeta)["wss_magnitude"]
        pinn_d.append(d_cm)
        pinn_real_peak.append(float(np.percentile(w, 99)))

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    ax.plot(ds, pred_peak, "-", color="#2c6e9c", lw=2,
            label="PINN (fixed 2.3 cm geometry, sweep $d^{*}$)")
    ax.scatter(pinn_d, pinn_real_peak, marker="X", s=90, color="#c0392b", zorder=6,
               label="PINN (each real geometry)")
    ax.scatter(cfd_d, cfd_peak, color="k", zorder=5, label="CFD peak WSS (per geometry)")
    ax.axvline(2.3, ls=":", color="gray", lw=1)
    ax.text(2.305, ax.get_ylim()[0], " held-out", color="gray", fontsize=9, va="bottom")
    ax.set_xlabel("inlet diameter $d^{*}$ (cm)", fontsize=11)
    ax.set_ylabel("peak WSS (Pa, 99th pct)", fontsize=11)
    ax.set_title(f"Parametric WSS response — {symmetry}, {phase}", fontsize=12)
    ax.legend(fontsize=9)
    out = Path(out_dir) / f"mu_sweep_{symmetry}_{phase}_wss.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
