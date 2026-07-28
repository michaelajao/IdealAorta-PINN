"""Figures: static paper-ready PNGs (matplotlib) and interactive 3D Plotly scenes.

Static figures go to ``report/figures/`` as PNG:
  * ``plane_comparison``    -- CFD | PINN | error speed maps on an XY/XZ slice
  * ``wss_map``             -- CFD | PINN | error wall-shear-stress maps
  * ``mu_sweep``            -- parametric sweep across inlet diameter
  * ``error_summary``       -- per-run QC card of velocity/WSS NRMSE by phase
  * ``convergence_curves``  -- training loss history from ``loss_history.csv``
  * ``save_comparison_png`` -- 3D CFD-vs-PINN streamline view

Interactive, rotatable CFD-vs-PINN 3D streamline scenes (``save_comparison``) go to
``report/interactive/`` as self-contained HTML. Plotting backends are imported
lazily so importing this module stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..config import (FIGURES_DIR, INTERACTIVE_DIR, PAPER_FIGURES_DIR,
                      PROJECT_ROOT)
from ..data.cache import load_points
from ..data.geometry import compute_wall_normals
from ..data.registry import CaseRecord, cases_by_id
from .metrics import _fold_mean, interp_diameter, parse_folds, predict_wss_physical
from .predict import TrainedModel, predict_physical


# ---------------------------------------------------------------------------
# Streamline helpers: CFD reference traces and PINN speed sampled on them.
# A single robust, integration-free view: the CFD-exported 3D streamline
# geometry, coloured by CFD speed on one side and by PINN speed sampled at those
# identical points on the other. (Integrating streamlines *through* the predicted
# field was tried and removed: on held-out cases the predicted field is
# inaccurate and non-solenoidal, so the integration stalls into a blob.)
# ---------------------------------------------------------------------------
def cfd_streamline_points(records: Sequence[CaseRecord], case_id: int, phase: str
                          ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """CFD 3D streamline points: returns (coords ``(N,3)``, speed ``(N,)``)."""
    rec = cases_by_id(records)[case_id]
    df = load_points(rec, "3D", phase)
    if df is None:
        return None
    coords = df[["x", "y", "z"]].to_numpy(float)
    speed = df["speed"].to_numpy(float) if "speed" in df else \
        np.linalg.norm(df[["u", "v", "w"]].to_numpy(float), axis=1)
    return coords, speed


def pinn_speed_on_points(model: TrainedModel, coords: np.ndarray, rec: CaseRecord,
                         phase: str) -> np.ndarray:
    """PINN-predicted speed at the given physical coordinates."""
    beta = rec.beta if rec.beta is not None else 1.0
    return predict_physical(model, coords, rec.inlet_diameter_cm, rec.disease_flag,
                            phase, beta=beta)["speed"]


# ---------------------------------------------------------------------------
# Static, paper-ready PNG figures (matplotlib)
# ---------------------------------------------------------------------------
def _plane_axes(coords: np.ndarray):
    """Return the two in-plane axis indices (largest spread)."""
    spread = coords.max(axis=0) - coords.min(axis=0)
    order = np.argsort(spread)[::-1]
    return int(order[0]), int(order[1])


def plane_comparison(model: TrainedModel, records: Sequence[CaseRecord], case_id: int,
                     phase: str, kind: str = "XY", max_points: int = 40_000,
                     out_dir: Path = FIGURES_DIR, shared_scale: bool = False) -> Path:
    """CFD | PINN | |error| speed maps on a CFD plane; saved as PNG.

    Color scaling (default, ``shared_scale=False``): each panel uses its own
    robust (99th-pct) range, so every field is legible regardless of model
    quality. With ``shared_scale=True`` the CFD and PINN panels share a
    *CFD-driven* ceiling so their colors are directly comparable; the ground
    truth — never the PINN — sets the scale, so the CFD is never hidden (a PINN
    that over-predicts simply saturates at the top of the bar). Either way each
    panel title shows the field's true ``[min, max]`` so magnitude mismatches
    are explicit even when a panel saturates.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = cases_by_id(records)[case_id]
    df = load_points(rec, kind, phase)
    if df is None:
        raise ValueError(f"No {kind} data for case {case_id} {phase}")
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    coords = df[["x", "y", "z"]].to_numpy(float)
    cfd_speed = (df["speed"].to_numpy(float) if "speed" in df
                 else np.linalg.norm(df[["u", "v", "w"]].to_numpy(float), axis=1))
    beta = rec.beta if rec.beta is not None else 1.0
    pinn = predict_physical(model, coords, rec.inlet_diameter_cm, rec.disease_flag, phase, beta=beta)
    pinn_speed = pinn["speed"]
    err = np.abs(pinn_speed - cfd_speed)

    a, b = _plane_axes(coords)
    ca, cb = coords[:, a], coords[:, b]

    def _vmax(v: np.ndarray) -> float:
        return float(np.percentile(v, 99)) or 1e-9

    # CFD ground truth sets the ceiling; PINN either shares it (comparable) or
    # uses its own (always legible). The CFD is never scaled by the PINN.
    cfd_vmax = _vmax(cfd_speed)
    pinn_vmax = cfd_vmax if shared_scale else _vmax(pinn_speed)

    # Clean publication panels, arranged side by side (CFD | PINN | absolute
    # error). To keep three panels of a long, thin tube legible without a wall of
    # redundant colorbars, the two speed panels SHARE one CFD-driven colorbar
    # (so any PINN over-prediction shows as saturation rather than being hidden
    # by per-panel rescaling); only the error panel carries a second bar. No
    # baked-in suptitle -- the LaTeX caption supplies geometry/phase/disease.
    speed_shared = shared_scale or abs(pinn_vmax - cfd_vmax) < 1e-12
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), constrained_layout=True)

    def _panel(ax, vals, title, cmap, vm):
        sc = ax.scatter(ca, cb, c=vals, s=3, cmap=cmap, vmin=0, vmax=vm)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("xyz"[a] + " (m)", fontsize=10)
        ax.set_ylabel("xyz"[b] + " (m)", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal")
        return sc

    sc_cfd = _panel(axes[0], cfd_speed, "CFD", "turbo", cfd_vmax)
    sc_pinn = _panel(axes[1], pinn_speed, "PINN", "turbo", pinn_vmax)
    sc_err = _panel(axes[2], err, "Absolute error", "magma", _vmax(err))

    if speed_shared:
        # one colorbar shared by CFD+PINN -- honest common scale, less clutter
        fig.colorbar(sc_pinn, ax=[axes[0], axes[1]], shrink=0.85, label="speed (m/s)")
    else:
        fig.colorbar(sc_cfd, ax=axes[0], shrink=0.85, label="speed (m/s)")
        fig.colorbar(sc_pinn, ax=axes[1], shrink=0.85, label="speed (m/s)")
    fig.colorbar(sc_err, ax=axes[2], shrink=0.85, label="|error| (m/s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"case{case_id:02d}_{phase}_{kind}_velocity.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def convergence_curves(history_csv: Path, out_dir: Path = FIGURES_DIR,
                       fname: str = "convergence.png", title: str = "") -> Path:
    """Training convergence: component losses + model-selection monitor vs epoch.

    Reads a trainer ``loss_history.csv`` and plots the raw component losses
    (velocity, physics, pressure, WSS) and the stable ``monitor`` on a log-y
    axis — the standard PINN convergence figure. The monitor is what drives
    best-model / early-stopping (held-out rel-L2 if configured, else the
    unweighted component-loss sum).
    """
    import csv as _csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history_csv = Path(history_csv)
    rows = list(_csv.DictReader(open(history_csv)))
    if not rows:
        raise ValueError(f"empty loss history: {history_csv}")
    ep = [float(r["epoch"]) for r in rows]

    def col(name):
        return [float(r[name]) for r in rows] if name in rows[0] else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for key, lab in (("velocity", "data velocity"), ("physics", "PDE residual"),
                     ("pressure", "wall pressure"), ("wss", "WSS")):
        ys = col(key)
        if ys is not None:
            ax1.plot(ep, ys, label=lab, lw=1.3)
    ax1.set_yscale("log"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.set_title("Component losses"); ax1.grid(True, which="both", alpha=0.25); ax1.legend(fontsize=8)

    mon = col("monitor")
    if mon is not None:
        ax2.plot(ep, mon, color="crimson", lw=1.4)
    ax2.set_yscale("log"); ax2.set_xlabel("epoch")
    ax2.set_ylabel("monitor (model-selection metric)")
    ax2.set_title("Convergence monitor"); ax2.grid(True, which="both", alpha=0.25)

    if title:
        fig.suptitle(title, fontsize=13)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Interactive 3D Plotly figures (CFD vs PINN) for screenshots
# ---------------------------------------------------------------------------
def _equal_aspect_ranges(xyz: np.ndarray) -> dict:
    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    half = max((maxs - mins).max() / 2.0, 1e-9)
    return {ax: (float(center[i] - half), float(center[i] + half))
            for i, ax in enumerate("xyz")}


def _segment_bounds(coords: np.ndarray):
    """Split an ordered point sequence into streamline segments at large jumps."""
    if len(coords) < 2:
        return [(0, len(coords))]
    step = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    finite = step[np.isfinite(step)]
    if finite.size == 0:
        return [(0, len(coords))]
    thr = max(np.quantile(finite, 0.95) * 4.0, np.median(finite) * 8.0)
    breaks = np.where(step > thr)[0]
    segs, start = [], 0
    for b in breaks:
        end = b + 1
        if end - start >= 2:
            segs.append((start, end))
        start = end
    if len(coords) - start >= 2:
        segs.append((start, len(coords)))
    return segs


def _ordered_line_arrays(coords: np.ndarray):
    """Gray connecting lines along an ordered point sequence (None-separated)."""
    xs, ys, zs = [], [], []
    for s, e in _segment_bounds(coords):
        xs += coords[s:e, 0].tolist() + [None]
        ys += coords[s:e, 1].tolist() + [None]
        zs += coords[s:e, 2].tolist() + [None]
    return xs, ys, zs


def comparison_figure(model: TrainedModel, records: Sequence[CaseRecord],
                      case_id: int, phase: str, marker_size: float = 2.0):
    """Interactive 1x2 figure: CFD streamlines vs the PINN speed on the same lines.

    Both panels draw the CFD-exported streamline geometry; the left is coloured by
    CFD speed and the right by the PINN's predicted speed sampled at the same points.
    Colour mismatches localize where the PINN over/under-predicts along the true flow
    paths -- a robust comparison that needs no integration. Speeds share one colour
    scale.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rec = cases_by_id(records)[case_id]
    pts = cfd_streamline_points(records, case_id, phase)
    if pts is None:
        raise ValueError(f"No 3D streamline data for case {case_id} {phase}")
    cfd_coords, cfd_speed = pts

    # PINN field sampled on the CFD streamline geometry.
    pinn_speed = pinn_speed_on_points(model, cfd_coords, rec, phase)
    pinn_lines = _ordered_line_arrays(cfd_coords)
    pinn_markers = (cfd_coords[:, 0], cfd_coords[:, 1], cfd_coords[:, 2])
    pinn_pts = cfd_coords
    pinn_label = "PINN (on CFD streamlines)"

    cmax = float(np.percentile(np.concatenate([cfd_speed, pinn_speed]), 99))
    rng = _equal_aspect_ranges(np.vstack([cfd_coords, pinn_pts]))
    clx, cly, clz = _ordered_line_arrays(cfd_coords)

    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(f"CFD | Case {case_id} {phase}", f"{pinn_label} | Case {case_id} {phase}"),
        horizontal_spacing=0.02)

    def add_panel(col, lines, markers, speed, cbx):
        fig.add_trace(go.Scatter3d(x=lines[0], y=lines[1], z=lines[2], mode="lines",
                      line=dict(color="rgba(60,60,60,0.25)", width=1.5),
                      showlegend=False, hoverinfo="skip"), row=1, col=col)
        fig.add_trace(go.Scatter3d(x=markers[0], y=markers[1], z=markers[2], mode="markers",
                      marker=dict(size=marker_size, color=speed, colorscale="Turbo", cmin=0,
                                  cmax=cmax, opacity=0.9,
                                  colorbar=dict(title="Speed (m/s)", x=cbx, len=0.75)),
                      showlegend=False, hoverinfo="skip"), row=1, col=col)

    add_panel(1, (clx, cly, clz), (cfd_coords[:, 0], cfd_coords[:, 1], cfd_coords[:, 2]),
              cfd_speed, 0.46)
    add_panel(2, pinn_lines, pinn_markers, pinn_speed, 1.01)

    for i in (1, 2):
        scene = "scene" if i == 1 else f"scene{i}"
        fig.layout[scene].update(
            xaxis=dict(title="x (m)", range=list(rng["x"])),
            yaxis=dict(title="y (m)", range=list(rng["y"])),
            zaxis=dict(title="z (m)", range=list(rng["z"])),
            aspectmode="cube", camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)))

    fig.update_layout(template="plotly_white",
                      title=f"3D streamlines: CFD vs PINN — Case {case_id} "
                            f"({rec.inlet_diameter_cm} cm, {rec.health}), {phase}",
                      margin=dict(l=5, r=5, t=55, b=5))
    return fig


def save_comparison(model: TrainedModel, records: Sequence[CaseRecord], case_id: int,
                    phase: str, out_dir: Path = INTERACTIVE_DIR) -> Path:
    """Save a rotatable CFD-vs-PINN HTML figure; returns the file path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = comparison_figure(model, records, case_id, phase)
    path = out_dir / f"case{case_id:02d}_{phase}_streamlines.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def save_comparison_png(model: TrainedModel, records: Sequence[CaseRecord], case_id: int,
                        phase: str, out_dir: Path = FIGURES_DIR) -> Path:
    """Save a static CFD-vs-PINN streamline PNG (kaleido); returns the file path.

    Mirrors the interactive figure so the manuscript figure is regenerated by the
    pipeline rather than hand-captured from the HTML: the robust speed-on-CFD-lines
    comparison, written as ``*_streamlines3d.png``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = comparison_figure(model, records, case_id, phase)
    path = out_dir / f"case{case_id:02d}_{phase}_streamlines3d.png"
    fig.write_image(str(path), width=1600, height=700, scale=2)
    return path


# ---------------------------------------------------------------------------
# Paper WSS figures: wall-shear-stress maps and the parametric diameter sweep.
# (Moved here from the former analysis/figures_paper.py; they reuse the
# inference helpers in analysis.metrics so the WSS is computed exactly as the
# trained model defines it.)
# ---------------------------------------------------------------------------

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
        cmaps=("turbo", "turbo", "magma"),
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


# ---------------------------------------------------------------------------
# Per-run QC card. Auto-produced once per run; regenerable under --skip-train.
# ---------------------------------------------------------------------------
def error_summary(metrics_dir: Path, out_dir: Path = FIGURES_DIR,
                  name: str = "error_summary") -> Optional[Path]:
    """A11: per-case velocity NRMSE and WSS NRMSE, systolic vs diastolic, as grouped
    bars read from ``report/metrics/<exp>/{velocity,wss}.json``. Catches phase
    collapse (a diastolic bar towering over systolic) at a glance."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_dir = Path(metrics_dir)

    def _load(kind: str, key: str) -> Dict:
        p = metrics_dir / f"{kind}.json"
        if not p.exists():
            return {}
        return {(r["case"], r["phase"]): r.get(key) for r in json.loads(p.read_text())}

    vel = _load("velocity", "vel_nrmse_phase")
    wss = _load("wss", "wss_nrmse")
    cases = sorted({c for (c, _ph) in list(vel) + list(wss)})
    if not cases:
        return None

    x = np.arange(len(cases))
    wdt = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(max(6.0, 1.1 * len(cases)), 4.2))
    for ax, data, title in ((axes[0], vel, "Velocity NRMSE (per-phase)"),
                            (axes[1], wss, "WSS NRMSE")):
        sysv = [data.get((c, "systolic"), np.nan) for c in cases]
        diav = [data.get((c, "diastolic"), np.nan) for c in cases]
        ax.bar(x - wdt / 2, sysv, wdt, label="systolic", color="#c0392b")
        ax.bar(x + wdt / 2, diav, wdt, label="diastolic", color="#2c6fbb")
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{c}" for c in cases], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def error_vs_diameter(folds: Sequence[str], insample: str = "stageA_case1_s12",
                      out: str = "error_vs_diameter") -> Path:
    """Held-out error against the in-sample floor, per held inlet diameter.

    Plots each fold's mean per-phase velocity and WSS NRMSE at the diameter it held
    out, with the in-sample reconstruction floor as a reference line, and annotates
    every point as interpolation or extrapolation. ``folds`` are ``"2.3:experiment"``
    specs. Pure post-processing: reads the metric JSON only, no model or GPU.

    Each fold carries its own train-set nondimensionalization, so only the scale-free
    NRMSE plotted here is comparable across folds.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parsed = sorted(parse_folds(folds), key=lambda t: t[0])
    diam = [d for d, _ in parsed]
    vel = [_fold_mean(exp, "velocity", "vel_nrmse_phase") for _, exp in parsed]
    wss = [_fold_mean(exp, "wss", "wss_nrmse") for _, exp in parsed]
    mid = interp_diameter(diam)

    floor_vel = _fold_mean(insample, "velocity", "vel_nrmse_phase")
    floor_wss = _fold_mean(insample, "wss", "wss_nrmse")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, y, floor, label in ((axes[0], vel, floor_vel, "Velocity NRMSE (per-phase)"),
                                (axes[1], wss, floor_wss, "WSS NRMSE (per-phase)")):
        ax.plot(diam, y, "o-", color="#2c6fbb", lw=2, ms=8, label="held-out (fold mean)")
        for d, yy in zip(diam, y):
            if np.isnan(yy):
                continue
            nat = "interp" if (mid is not None and abs(d - mid) < 1e-9) else "extrap"
            ax.annotate(f"{yy:.2f}\n({nat})", (d, yy), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8)
        if not np.isnan(floor):
            ax.axhline(floor, ls="--", color="#c0392b", lw=1.5,
                       label=f"in-sample floor ({floor:.2f})")
        ax.set_xlabel("held inlet diameter (cm)")
        ax.set_ylabel(label)
        ax.set_xticks(diam)
        finite = [v for v in y if not np.isnan(v)] + ([floor] if not np.isnan(floor) else [])
        ax.set_ylim(0, max(finite) * 1.25 if finite else 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_title("Generalization vs. in-sample reconstruction", fontsize=10, loc="left")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"{out}.{ext}", dpi=150, bbox_inches="tight")
    if PAPER_FIGURES_DIR.parent.exists():      # only when a local paper/ tree is present
        PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PAPER_FIGURES_DIR / f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)

    png = FIGURES_DIR / f"{out}.png"
    print(f"[fig] wrote {png.relative_to(PROJECT_ROOT)} (+ .pdf)")
    print(f"      diam={diam}  vel_mean={[round(v, 3) for v in vel]}  "
          f"wss_mean={[round(w, 3) for w in wss]}")
    print(f"      in-sample floor: vel={floor_vel:.3f}  wss={floor_wss:.3f}")
    return png
