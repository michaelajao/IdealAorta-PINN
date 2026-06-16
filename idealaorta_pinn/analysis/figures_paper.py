"""Additional paper figures: WSS maps, wall pressure, and axial velocity profiles.

These complement ``figures.plane_comparison`` (velocity planes) and
``figures.convergence_curves``. All reuse the inference helpers in
``analysis.predict`` / ``analysis.metrics`` so the WSS/pressure are computed
exactly as the trained model defines them. Saved as static PNGs under
``report/figures/<experiment>/`` (the caller passes ``out_dir``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..config import FIGURES_DIR, PROJECT_ROOT, TABLES_DIR
from ..data.cache import load_points
from ..data.geometry import compute_wall_normals
from ..data.registry import CaseRecord, cases_by_id
from .figures import _plane_axes
from .metrics import predict_wss_physical
from .predict import TrainedModel, predict_physical

SLICES_CSV = PROJECT_ROOT / "data" / "results_on_slices.csv"


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
        titles=("CFD WSS", "Surrogate WSS", "Absolute error"),
        cmaps=("inferno", "inferno", "magma"),
        vmaxes=(vm, vm, float(np.percentile(err, 99)) or 1e-9),
        unit="Pa", axis_labels=("xyz"[a] + " (m)", "xyz"[b] + " (m)"),
        out_path=(Path(out_dir) / f"case{case_id:02d}_{phase}_{kind}_wss_map.png"))


def wall_pressure_map(model: TrainedModel, records: Sequence[CaseRecord], case_id: int,
                      phase: str, kind: str = "XZ", max_points: int = 20_000,
                      out_dir: Path = FIGURES_DIR) -> Path:
    """CFD | Surrogate | error wall pressure, wall points projected onto ``kind``.

    Pressure is a gauge field (defined up to a constant), so both fields are
    mean-subtracted to compare *spatial structure* on a symmetric diverging scale.
    """
    rec = cases_by_id(records)[case_id]
    df = load_points(rec, "WSS", phase)
    if df is None or "p" not in df.columns:
        raise ValueError(f"no wall pressure for case {case_id} {phase}")
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    coords = df[["x", "y", "z"]].to_numpy(float)
    cfd = df["p"].to_numpy(float)
    beta = rec.beta if rec.beta is not None else 1.0
    pinn = predict_physical(model, coords, rec.inlet_diameter_cm, rec.disease_flag,
                            phase, beta=beta)["p"]
    cfd0, pinn0 = cfd - cfd.mean(), pinn - pinn.mean()       # gauge-align
    err = np.abs(pinn0 - cfd0)
    a, b = _KIND_AXES[kind]
    vm = float(np.percentile(np.abs(cfd0), 99)) or 1e-9
    return _three_panel(
        coords[:, a], coords[:, b], cfd0, pinn0, err,
        titles=("CFD pressure", "Surrogate pressure", "Absolute error"),
        cmaps=("RdBu_r", "RdBu_r", "magma"),
        vmaxes=(vm, vm, float(np.percentile(err, 99)) or 1e-9),
        unit="Pa", diverging=True, axis_labels=("xyz"[a] + " (m)", "xyz"[b] + " (m)"),
        out_path=(Path(out_dir) / f"case{case_id:02d}_{phase}_{kind}_pressure_map.png"))


def velocity_profile(model: TrainedModel, records: Sequence[CaseRecord], case_id: int,
                     phase: str, kind: str = "XZ", out_dir: Path = FIGURES_DIR) -> Path:
    """Transverse speed profile across the aneurysm bulge: CFD vs PINN.

    Locates the bulge as the axial station with the widest lumen, takes a thin
    slab there, and plots speed vs the transverse coordinate for CFD and PINN.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = cases_by_id(records)[case_id]
    df = load_points(rec, kind, phase)
    if df is None or "speed" not in df.columns:
        raise ValueError(f"no {kind} data for case {case_id} {phase}")
    coords = df[["x", "y", "z"]].to_numpy(float)
    speed = df["speed"].to_numpy(float)
    ax_ax = 0                                   # axial axis is x (configs/constants.yaml)
    tr = 1 if _plane_axes(coords)[0] == ax_ax else _plane_axes(coords)[0]
    # find the axial station with the widest transverse extent (the bulge)
    xa = coords[:, ax_ax]
    edges = np.linspace(xa.min(), xa.max(), 25)
    widths = [(np.ptp(coords[(xa >= edges[i]) & (xa < edges[i + 1]), tr])
               if np.any((xa >= edges[i]) & (xa < edges[i + 1])) else 0.0)
              for i in range(len(edges) - 1)]
    ib = int(np.argmax(widths))
    x0 = 0.5 * (edges[ib] + edges[ib + 1])
    tol = (xa.max() - xa.min()) / 40.0
    sel = np.abs(xa - x0) < tol
    if sel.sum() < 20:
        sel = np.abs(xa - x0) < 2 * tol
    tc = coords[sel, tr]
    cfd = speed[sel]
    beta = rec.beta if rec.beta is not None else 1.0
    pinn = predict_physical(model, coords[sel], rec.inlet_diameter_cm, rec.disease_flag,
                            phase, beta=beta)["speed"]
    order = np.argsort(tc)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.scatter(tc[order], cfd[order], s=10, c="k", alpha=0.5, label="CFD")
    ax.scatter(tc[order], pinn[order], s=10, c="crimson", alpha=0.5, label="PINN")
    ax.set_xlabel(f"{'xyz'[tr]} (m)  [transverse]")
    ax.set_ylabel("speed (m/s)")
    ax.set_title(f"Case {case_id} ({rec.inlet_diameter_cm} cm, {rec.health}) — "
                 f"bulge profile @ {'xyz'[ax_ax]}={x0:.3f} m, {phase}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"case{case_id:02d}_{phase}_bulge_profile.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _axial_speed_profile(model: TrainedModel, rec: CaseRecord, phase: str,
                         kind: str = "XZ", n_bins: int = 8):
    """Return axial-bin CFD/PINN mean speed profiles for one case.

    Uses CFD point data directly (``XZ`` with ``XY`` fallback), bins points along
    x into equal-count bins, and computes mean speed per bin for CFD and PINN.
    """
    df = load_points(rec, kind, phase)
    if df is None:
        df = load_points(rec, "XY", phase)
    if df is None or "speed" not in df.columns:
        return None

    coords = df[["x", "y", "z"]].to_numpy(float)
    xa = coords[:, 0]
    cfd_speed = df["speed"].to_numpy(float)

    beta = rec.beta if rec.beta is not None else 1.0
    pred = predict_physical(model, coords, rec.inlet_diameter_cm, rec.disease_flag, phase, beta=beta)
    pinn_speed = pred["speed"]

    edges = np.quantile(xa, np.linspace(0, 1, n_bins + 1))
    idx = np.arange(1, n_bins + 1)
    cfd_bin = np.full(n_bins, np.nan)
    pinn_bin = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (xa >= edges[i]) & (xa <= edges[i + 1]) if i == n_bins - 1 else ((xa >= edges[i]) & (xa < edges[i + 1]))
        if np.any(m):
            cfd_bin[i] = float(np.mean(cfd_speed[m]))
            pinn_bin[i] = float(np.mean(pinn_speed[m]))

    return idx, cfd_bin, pinn_bin


def axial_velocity_profile(model: TrainedModel, records: Sequence[CaseRecord], case_id: int,
                           phase: str = "systolic", kind: str = "XZ",
                           out_dir: Path = FIGURES_DIR) -> Path:
    """Clean axial velocity profile (CFD vs PINN) for one case."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = cases_by_id(records)[case_id]
    prof = _axial_speed_profile(model, rec, phase=phase, kind=kind)
    if prof is None:
        raise ValueError(f"no valid CFD speed data for case {case_id} {phase}")
    dn, cfd_v, pinn_v = prof

    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.plot(dn, cfd_v, "o-", color="black", lw=1.6, ms=4.5, label="CFD")
    ax.plot(dn, pinn_v, "s--", color="crimson", lw=1.6, ms=4.2, label="PINN")
    ax.set_xlabel("axial bin (inlet -> outlet)")
    ax.set_ylabel("mean speed (m/s)")
    ax.set_title(f"Case {case_id} ({rec.inlet_diameter_cm} cm, {rec.health}) — axial velocity profile, {phase}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"case{case_id:02d}_{phase}_axial_velocity_profile.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def slice_reference_table(csv_path: Path = SLICES_CSV, out_dir: Path = TABLES_DIR,
                          filename: str = "results_on_slices_reference_table.csv") -> Path:
    """Write a clean, sorted reference table from ``data/results_on_slices.csv``."""
    sl = pd.read_csv(csv_path)
    out = sl.copy()
    out["slice_n"] = out["slice"].astype(str).str.extract(r"D(\d+)").astype(float)
    out = out.sort_values(["case_id", "slice_n", "slice"], kind="stable")
    cols = [
        "case_id", "slice", "slice_label", "area_cm2",
        "avg_vel_mps", "avg_pressure_pa", "tke", "tef",
    ]
    out = out[[c for c in cols if c in out.columns]]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    out.to_csv(path, index=False)
    return path
