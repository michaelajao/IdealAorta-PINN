"""Additional paper figures: WSS maps and the slice reference table.

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
import pandas as pd

from ..config import FIGURES_DIR, PROJECT_ROOT, TABLES_DIR
from ..data.cache import load_points
from ..data.geometry import compute_wall_normals
from ..data.registry import CaseRecord, cases_by_id
from .metrics import predict_wss_physical
from .predict import TrainedModel

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
        titles=("CFD WSS", "PINN WSS", "Absolute error"),
        cmaps=("inferno", "inferno", "magma"),
        vmaxes=(vm, vm, float(np.percentile(err, 99)) or 1e-9),
        unit="Pa", axis_labels=("xyz"[a] + " (m)", "xyz"[b] + " (m)"),
        out_path=(Path(out_dir) / f"case{case_id:02d}_{phase}_{kind}_wss_map.png"))


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
