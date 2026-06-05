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
from .streamlines import cfd_streamline_points, pinn_streamlines


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
                     out_dir: Path = FIGURES_DIR) -> Path:
    """CFD | PINN | |error| speed maps on a CFD plane; saved as PNG."""
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
    vmax = float(np.percentile(np.concatenate([cfd_speed, pinn_speed]), 99))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, vals, title, cmap, vm in (
        (axes[0], cfd_speed, "CFD", "turbo", vmax),
        (axes[1], pinn_speed, "PINN", "turbo", vmax),
        (axes[2], err, "|error|", "magma", float(np.percentile(err, 99) + 1e-9)),
    ):
        sc = ax.scatter(ca, cb, c=vals, s=3, cmap=cmap, vmin=0, vmax=vm)
        ax.set_title(title)
        ax.set_xlabel("xyz"[a] + " (m)")
        ax.set_ylabel("xyz"[b] + " (m)")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="m/s")
    fig.suptitle(f"Case {case_id} ({rec.inlet_diameter_cm} cm, {rec.health}) — "
                 f"{kind} plane, {phase}", fontsize=13)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"case{case_id:02d}_{phase}_{kind}_velocity.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def error_vs_diameter(rows: List[Dict], metric: str = "vel_rel_l2",
                      out_dir: Path = FIGURES_DIR, fname: str = "error_vs_diameter.png") -> Path:
    """Line plot of a validation metric vs inlet diameter (parametric summary)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_phase: Dict[str, List] = {}
    for r in rows:
        if metric in r and "diameter_cm" in r:
            by_phase.setdefault(r.get("phase", "all"), []).append((r["diameter_cm"], r[metric]))

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    for phase, pts in by_phase.items():
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, "o-", label=phase)
    ax.set_xlabel("Inlet diameter (cm)")
    ax.set_ylabel(metric)
    ax.set_title("Surrogate error vs inlet diameter")
    ax.grid(True, alpha=0.3)
    ax.legend()
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
                      grid_res: int = 60, n_seed: int = 240):
    """Interactive 1x2 figure: CFD-traced streamlines vs PINN-traced streamlines.

    Left = the CFD-exported streamlines; right = streamlines integrated through
    the PINN velocity field (seeded identically at the inlet, masked to the
    lumen). Both are drawn as faint trajectory lines with points coloured by
    speed on a shared scale. Raises if PINN tracing yields no streamlines.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rec = cases_by_id(records)[case_id]
    pts = cfd_streamline_points(records, case_id, phase)
    if pts is None:
        raise ValueError(f"No 3D streamline data for case {case_id} {phase}")
    cfd_coords, cfd_speed = pts

    poly = pinn_streamlines(model, records, case_id, phase, grid_res=grid_res, n_seed=n_seed)
    if poly is None or poly.n_points == 0 or np.asarray(poly.lines).size == 0:
        raise RuntimeError(
            f"PINN streamline tracing produced no streamlines for case {case_id} {phase}. "
            "Check pyvista availability and the trained field, or increase n_seed / grid_res.")
    (plx, ply, plz), (pmx, pmy, pmz, pinn_speed) = _pyvista_line_arrays(poly)
    pinn_pts = np.column_stack([pmx, pmy, pmz])
    pinn_label = "PINN (traced)"

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
    add_panel(2, (plx, ply, plz), (pmx, pmy, pmz), pinn_speed, 1.01)

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
