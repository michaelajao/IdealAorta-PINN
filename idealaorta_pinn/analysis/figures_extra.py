"""Additional publication figures (inference only; CPU-friendly).

  * ``wall_surface_3d``  -- 3D wall point cloud coloured by WSS or pressure,
    CFD vs surrogate, on a shared scale (the data exports WSS and pressure only
    on the wall, so these are the 3D pressure/WSS views the dataset supports).
  * ``agreement_scatter`` -- CFD-vs-surrogate calibration: speed and WSS density
    scatters with the identity line, slope and R^2, plus a Bland--Altman panel.
  * ``mu_sweep`` -- the parametric claim: on a fixed geometry, sweep the inlet
    diameter parameter and show the predicted peak WSS varies smoothly and tracks
    the per-diameter CFD values.
  * ``error_summary_bars`` -- systolic vs diastolic NRMSE / WSS / recirculation
    QC card for one case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from ..config import FIGURES_DIR
from ..data.cache import load_points
from ..data.geometry import compute_wall_normals
from ..data.registry import CaseRecord, cases_by_id
from .metrics import predict_wss_physical, recirculation_fraction, velocity_metrics, wss_metrics
from .predict import TrainedModel, predict_physical

# axisymmetric / anterior / posterior diseased case ids by inlet diameter (cm)
_SYMMETRY_SERIES = {"axisymmetric": {2.0: 1, 2.3: 4, 2.6: 7},
                    "anterior": {2.0: 2, 2.3: 5, 2.6: 8},
                    "posterior": {2.0: 3, 2.3: 6, 2.6: 9}}


def _wall_fields(model: TrainedModel, rec: CaseRecord, phase: str):
    df = load_points(rec, "WSS", phase)
    if df is None or "wss" not in df.columns:
        return None
    coords = df[["x", "y", "z"]].to_numpy(float)
    normals = compute_wall_normals(coords)
    beta = rec.beta if rec.beta is not None else 1.0
    cfd_wss = df["wss"].to_numpy(float)
    cfd_p = df["p"].to_numpy(float) if "p" in df.columns else None
    pinn_wss = predict_wss_physical(model, coords, normals, rec.inlet_diameter_cm,
                                    rec.disease_flag, phase, beta=beta)["wss_magnitude"]
    pinn_p = predict_physical(model, coords, rec.inlet_diameter_cm, rec.disease_flag,
                              phase, beta=beta)["p"]
    return coords, cfd_wss, pinn_wss, cfd_p, pinn_p


def wall_surface_3d(model: TrainedModel, records: Sequence[CaseRecord], case_id: int,
                    phase: str, field: str = "wss", out_dir: Path = FIGURES_DIR,
                    max_points: int = 20_000) -> Path:
    """CFD vs surrogate on the 3D wall point cloud, coloured by ``field`` in
    {``wss``, ``pressure``}. Saved as a static PNG (kaleido)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .figures import _equal_aspect_ranges

    rec = cases_by_id(records)[case_id]
    res = _wall_fields(model, rec, phase)
    if res is None:
        raise ValueError(f"no wall data for case {case_id} {phase}")
    coords, cfd_wss, pinn_wss, cfd_p, pinn_p = res
    if field == "pressure" and cfd_p is None:
        raise ValueError("no wall pressure in the export")
    if len(coords) > max_points:
        idx = np.random.default_rng(0).choice(len(coords), max_points, replace=False)
        coords = coords[idx]
        cfd_wss, pinn_wss = cfd_wss[idx], pinn_wss[idx]
        if cfd_p is not None:
            cfd_p, pinn_p = cfd_p[idx], pinn_p[idx]

    if field == "wss":
        cfd, pinn, unit, cscale = cfd_wss, pinn_wss, "WSS (Pa)", "Inferno"
        cmin = 0.0
        cmax = float(np.percentile(cfd, 99)) or 1e-9
    else:  # pressure: signed -> symmetric diverging scale
        cfd, pinn, unit, cscale = cfd_p, pinn_p, "pressure (Pa)", "RdBu_r"
        cmax = float(np.percentile(np.abs(cfd), 99)) or 1e-9
        cmin = -cmax

    rng = _equal_aspect_ranges(coords)
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=(f"CFD | {unit.split()[0]}", f"Surrogate | {unit.split()[0]}"),
                        horizontal_spacing=0.02)
    for col, vals, cbx in ((1, cfd, 0.46), (2, pinn, 1.01)):
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers",
            marker=dict(size=2, color=vals, colorscale=cscale, cmin=cmin, cmax=cmax,
                        opacity=0.9, colorbar=dict(title=unit, x=cbx, len=0.75)),
            showlegend=False, hoverinfo="skip"), row=1, col=col)
    for i in (1, 2):
        scene = "scene" if i == 1 else f"scene{i}"
        fig.layout[scene].update(
            xaxis=dict(title="x (m)", range=list(rng["x"])),
            yaxis=dict(title="y (m)", range=list(rng["y"])),
            zaxis=dict(title="z (m)", range=list(rng["z"])),
            aspectmode="cube", camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)))
    fig.update_layout(template="plotly_white",
                      title=f"Wall {unit.split()[0]}: CFD vs surrogate — Case {case_id} "
                            f"({rec.inlet_diameter_cm} cm, {rec.health}), {phase}",
                      margin=dict(l=5, r=5, t=55, b=5))
    out = Path(out_dir) / f"case{case_id:02d}_{phase}_wall3d_{field}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out), width=1400, height=600, scale=2)
    return out


def _ident_panel(ax, cfd, pinn, label, unit):
    from scipy import stats
    m = np.isfinite(cfd) & np.isfinite(pinn)
    cfd, pinn = cfd[m], pinn[m]
    hb = ax.hexbin(cfd, pinn, gridsize=45, cmap="viridis", mincnt=1, bins="log")
    lim = float(max(np.percentile(cfd, 99.5), np.percentile(pinn, 99.5)))
    ax.plot([0, lim], [0, lim], "r--", lw=1.2, label="identity")
    sl, ic, r, *_ = stats.linregress(cfd, pinn)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel(f"CFD {label} ({unit})", fontsize=11)
    ax.set_ylabel(f"Surrogate {label} ({unit})", fontsize=11)
    ax.set_title(f"{label}: slope={sl:.2f}, $R^2$={r**2:.3f}", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    return hb


def agreement_scatter(model: TrainedModel, records: Sequence[CaseRecord], case_id: int,
                      phase: str, kind: str = "XZ", out_dir: Path = FIGURES_DIR,
                      max_points: int = 40_000) -> Path:
    """Speed and WSS calibration (identity line, slope, R^2) + WSS Bland--Altman."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = cases_by_id(records)[case_id]
    beta = rec.beta if rec.beta is not None else 1.0
    df = load_points(rec, kind, phase)
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    coords = df[["x", "y", "z"]].to_numpy(float)
    cfd_speed = (df["speed"].to_numpy(float) if "speed" in df
                 else np.linalg.norm(df[["u", "v", "w"]].to_numpy(float), axis=1))
    pinn_speed = predict_physical(model, coords, rec.inlet_diameter_cm, rec.disease_flag,
                                  phase, beta=beta)["speed"]
    wdf = load_points(rec, "WSS", phase)
    wc = wdf[["x", "y", "z"]].to_numpy(float)
    cfd_wss = wdf["wss"].to_numpy(float)
    pinn_wss = predict_wss_physical(model, wc, compute_wall_normals(wc), rec.inlet_diameter_cm,
                                    rec.disease_flag, phase, beta=beta)["wss_magnitude"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    _ident_panel(axes[0], cfd_speed, pinn_speed, "speed", "m/s")
    _ident_panel(axes[1], cfd_wss, pinn_wss, "WSS", "Pa")
    # Bland--Altman for WSS
    mean = 0.5 * (cfd_wss + pinn_wss)
    diff = pinn_wss - cfd_wss
    bias, sd = float(np.mean(diff)), float(np.std(diff))
    axes[2].scatter(mean, diff, s=3, alpha=0.25, color="#2c6e9c")
    for y, ls, lab in ((bias, "-", f"bias={bias:.2f}"),
                       (bias + 1.96 * sd, "--", f"+1.96σ={bias+1.96*sd:.2f}"),
                       (bias - 1.96 * sd, "--", f"-1.96σ={bias-1.96*sd:.2f}")):
        axes[2].axhline(y, ls=ls, color="k", lw=1)
        axes[2].text(axes[2].get_xlim()[1], y, " " + lab, va="center", fontsize=9)
    axes[2].set_xlabel("mean WSS (Pa)", fontsize=11)
    axes[2].set_ylabel("surrogate − CFD (Pa)", fontsize=11)
    axes[2].set_title("WSS Bland–Altman", fontsize=12)
    fig.suptitle(f"CFD vs surrogate agreement — Case {case_id} "
                 f"({rec.inlet_diameter_cm} cm), {phase}", fontsize=13)
    out = Path(out_dir) / f"case{case_id:02d}_{phase}_agreement.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def mu_sweep(model: TrainedModel, records: Sequence[CaseRecord], symmetry: str,
             phase: str, out_dir: Path = FIGURES_DIR, n: int = 25) -> Path:
    """Parametric response: on a fixed geometry (the held $2.3$ cm case of the
    given symmetry series), sweep the inlet-diameter parameter $d^{*}$ over
    [2.0, 2.6] cm and plot the predicted peak WSS, against the per-diameter CFD
    peak WSS of the matching-symmetry cases. The geometry is held fixed, so the
    curve isolates the network's response to $d^{*}$; the CFD markers (distinct
    geometries) give the physical trend for context."""
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

    cfd_d, cfd_peak = [], []
    for d_cm, cid in series.items():
        cdf = load_points(cases_by_id(records)[cid], "WSS", phase)
        if cdf is not None and "wss" in cdf.columns:
            cfd_d.append(d_cm)
            cfd_peak.append(float(np.percentile(cdf["wss"].to_numpy(float), 99)))

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    ax.plot(ds, pred_peak, "-", color="#2c6e9c", lw=2,
            label="surrogate (fixed 2.3 cm geometry, sweep $d^{*}$)")
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


def error_summary_bars(metrics: Dict[str, Dict], case_id: int,
                       out_dir: Path = FIGURES_DIR) -> Path:
    """Grouped systolic/diastolic bars (velocity NRMSE, WSS rel-L2, recirc abs
    error) from a {phase: metrics} dict, as an at-a-glance QC card."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = [p for p in ("systolic", "diastolic") if p in metrics]
    labels = ["vel NRMSE", "WSS rel-$L_2$", "|recirc err|"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    w = 0.35
    x = np.arange(len(labels))
    for i, ph in enumerate(phases):
        m = metrics[ph]
        vals = [m.get("vel_nrmse_phase", np.nan), m.get("wss_rel_l2", np.nan),
                abs(m.get("recirc_cfd", 0) - m.get("recirc_pinn", 0))]
        ax.bar(x + (i - 0.5) * w, vals, w, label=ph)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("error", fontsize=11)
    ax.set_title(f"Case {case_id} reconstruction QC", fontsize=12)
    ax.legend(fontsize=9)
    out = Path(out_dir) / f"case{case_id:02d}_error_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def physics_residual_map(model: TrainedModel, records: Sequence[CaseRecord], case_id: int,
                         phase: str, kind: str = "XZ", out_dir: Path = FIGURES_DIR,
                         max_points: int = 12_000) -> Path:
    """Continuity and momentum residual magnitude of the predicted field on a
    plane (standardized units). Shows the field satisfies the PDE away from the
    jet and where the residual concentrates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from ..pinn.physics import compute_residuals
    from .figures import _plane_axes
    from .predict import _phase_value

    rec = cases_by_id(records)[case_id]
    df = load_points(rec, kind, phase)
    if df is None:
        raise ValueError(f"no {kind} data for case {case_id} {phase}")
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    coords = df[["x", "y", "z"]].to_numpy(float)
    norm = model.normalizer
    cs = norm.coords_std(coords).astype("float32")
    beta = rec.beta if rec.beta is not None else 1.0
    mu = np.array([norm.diameter_nd(rec.inlet_diameter_cm), float(beta),
                   float(rec.disease_flag), _phase_value(phase)], dtype="float32")
    dev = model.device
    x = torch.tensor(cs[:, 0:1], device=dev, requires_grad=True)
    y = torch.tensor(cs[:, 1:2], device=dev, requires_grad=True)
    z = torch.tensor(cs[:, 2:3], device=dev, requires_grad=True)
    params = torch.tensor(np.tile(mu, (len(cs), 1)), device=dev)
    r = compute_residuals(model.networks, x, y, z, params, model.Re)
    cont = np.abs(r["res_cont"].detach().cpu().numpy().ravel())
    mom = np.sqrt((r["res_x"] ** 2 + r["res_y"] ** 2
                   + r["res_z"] ** 2).detach().cpu().numpy()).ravel()

    a, b = _plane_axes(coords)
    ca, cb = coords[:, a], coords[:, b]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, vals, title in ((axes[0], cont, "$|\\nabla\\!\\cdot\\!\\mathbf{u}|$ (continuity)"),
                            (axes[1], mom, "momentum residual $|\\mathcal{R}_{\\mathrm{mom}}|$")):
        vm = float(np.percentile(vals, 99)) or 1e-9
        sc = ax.scatter(ca, cb, c=vals, s=4, cmap="magma", vmin=0, vmax=vm)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("xyz"[a] + " (m)", fontsize=11)
        ax.set_ylabel("xyz"[b] + " (m)", fontsize=11)
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="standardized residual")
    fig.suptitle(f"PDE residual on the {kind} plane — Case {case_id} "
                 f"({rec.inlet_diameter_cm} cm), {phase}", fontsize=13)
    out = Path(out_dir) / f"case{case_id:02d}_{phase}_{kind}_residual.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
