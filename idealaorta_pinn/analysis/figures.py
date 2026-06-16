"""Figures: static paper-ready PNGs (matplotlib) and interactive 3D Plotly scenes.

Static figures (CFD | PINN | error speed maps on a plane, and a parametric
error-vs-diameter summary) go to ``report/figures``; interactive, rotatable
CFD-vs-PINN 3D streamline scenes (for screenshots into the paper) go to
``report/interactive`` as self-contained HTML. Plotting backends are imported
lazily so importing this module stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..config import FIGURES_DIR, INTERACTIVE_DIR
from ..data.cache import load_points
from ..data.registry import CaseRecord, cases_by_id
from .predict import TrainedModel, predict_physical
from .streamlines import cfd_streamline_points, pinn_speed_on_points, pinn_streamlines


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

    # Clean publication panels: named columns (CFD / Surrogate / Absolute error),
    # no exposed [min,max] debug ranges in titles, and no baked-in suptitle -- the
    # LaTeX figure caption supplies case geometry, phase, and disease state.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, vals, title, cmap, vm in (
        (axes[0], cfd_speed, "CFD", "turbo", cfd_vmax),
        (axes[1], pinn_speed, "Surrogate", "turbo", pinn_vmax),
        (axes[2], err, "Absolute error", "magma", _vmax(err)),
    ):
        sc = ax.scatter(ca, cb, c=vals, s=3, cmap=cmap, vmin=0, vmax=vm)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("xyz"[a] + " (m)", fontsize=11)
        ax.set_ylabel("xyz"[b] + " (m)", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="speed (m/s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"case{case_id:02d}_{phase}_{kind}_velocity.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# NOTE: the old `error_vs_diameter` line plot was removed — with a single
# validated diameter it degenerated to one point, and it mixed the misleading
# rel-L2 metric. A better parametric summary (one trained model evaluated across
# its in-sample + held-out diameters, held-out point highlighted, using
# U_ref-normalized nrmse) will be added once multi-diameter validation results
# from a single model are available (Stage C / a full leave-one-out sweep).


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


def _pyvista_line_arrays(poly):
    """Extract (line coords with breaks) and (marker coords, speed) from a pyvista PolyData."""
    pts = np.asarray(poly.points)
    if "speed" in poly.point_data:
        speed = np.asarray(poly["speed"])
    elif "velocity" in poly.point_data:
        speed = np.linalg.norm(np.asarray(poly["velocity"]), axis=1)
    else:
        speed = np.zeros(len(pts))
    conn = np.asarray(poly.lines)
    lx, ly, lz, mx, my, mz, ms = [], [], [], [], [], [], []
    i = 0
    while i < len(conn):
        n = int(conn[i])
        ids = conn[i + 1:i + 1 + n]
        i += n + 1
        seg = pts[ids]
        lx += seg[:, 0].tolist() + [None]
        ly += seg[:, 1].tolist() + [None]
        lz += seg[:, 2].tolist() + [None]
        mx += seg[:, 0].tolist()
        my += seg[:, 1].tolist()
        mz += seg[:, 2].tolist()
        ms += speed[ids].tolist()
    return (lx, ly, lz), (np.array(mx), np.array(my), np.array(mz), np.array(ms))


def comparison_figure(model: TrainedModel, records: Sequence[CaseRecord],
                      case_id: int, phase: str, marker_size: float = 2.0,
                      grid_res: int = 60, n_seed: int = 240, traced: bool = False):
    """Interactive 1x2 figure: CFD streamlines vs the surrogate on the same lines.

    Both panels draw the CFD-exported streamline geometry; the left is coloured by
    CFD speed and the right by the surrogate's predicted speed *sampled at the same
    points* (``traced=False``, default). Colour mismatches localize where the
    surrogate over/under-predicts along the true flow paths -- a robust comparison
    that needs no integration. ``traced=True`` instead integrates streamlines
    through the PINN field (pyvista, masked to the lumen); that path is fragile
    because integrating an extrapolated/near-wall field can terminate the lines
    early, so it is not the default. Speeds share one colour scale.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rec = cases_by_id(records)[case_id]
    pts = cfd_streamline_points(records, case_id, phase)
    if pts is None:
        raise ValueError(f"No 3D streamline data for case {case_id} {phase}")
    cfd_coords, cfd_speed = pts

    if traced:
        poly = pinn_streamlines(model, records, case_id, phase, grid_res=grid_res, n_seed=n_seed)
        if poly is None or poly.n_points == 0 or np.asarray(poly.lines).size == 0:
            raise RuntimeError(
                f"PINN streamline tracing produced no streamlines for case {case_id} {phase}. "
                "Check pyvista availability and the trained field, or increase n_seed / grid_res.")
        (plx, ply, plz), (pmx, pmy, pmz, pinn_speed) = _pyvista_line_arrays(poly)
        pinn_lines = (plx, ply, plz)
        pinn_markers = (pmx, pmy, pmz)
        pinn_pts = np.column_stack([pmx, pmy, pmz])
        pinn_label = "Surrogate (traced)"
    else:
        # Robust: the surrogate field sampled on the CFD streamline geometry.
        pinn_speed = pinn_speed_on_points(model, cfd_coords, rec, phase)
        clx2, cly2, clz2 = _ordered_line_arrays(cfd_coords)
        pinn_lines = (clx2, cly2, clz2)
        pinn_markers = (cfd_coords[:, 0], cfd_coords[:, 1], cfd_coords[:, 2])
        pinn_pts = cfd_coords
        pinn_label = "Surrogate (on CFD streamlines)"

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
