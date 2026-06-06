"""Standalone CFD-only visualization (no PINN) on honest, per-field color scales.

Plots what the CFD ground truth actually contains for one case:
  * velocity speed on XY / XZ planes (systolic + diastolic)
  * velocity components u, v, w (XY plane)
  * wall shear stress magnitude + wall pressure

Each panel gets its OWN robust (1-99 pct) color range, so a field is never
squashed by another field's scale. Saves PNGs to report/figures/cfd_check/.

Usage:
    python scripts/plot_cfd_check.py            # case 1
    python scripts/plot_cfd_check.py 4          # case 4
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "report" / "figures" / "cfd_check"
OUT.mkdir(parents=True, exist_ok=True)

MAX_PTS = 60_000  # downsample big planes for plotting speed


def load(case: int, kind: str, phase: str) -> pd.DataFrame | None:
    f = PROC / f"case{case:02d}_{kind}_{phase}.parquet"
    if not f.exists():
        print(f"  [missing] {f.name}")
        return None
    df = pd.read_parquet(f)
    if len(df) > MAX_PTS:
        df = df.sample(MAX_PTS, random_state=0)
    return df


def plane_xy(kind: str):
    """In-plane axis labels for a given export kind."""
    return ("x", "y") if kind in ("XY", "3D") else ("x", "z")


def panel(ax, df, ax0, ax1, field, title, cmap, diverging=False):
    vals = df[field].to_numpy(float)
    if diverging:
        m = float(np.percentile(np.abs(vals), 99)) or 1e-9
        vmin, vmax = -m, m
    else:
        vmin = float(np.percentile(vals, 1))
        vmax = float(np.percentile(vals, 99)) or 1e-9
    sc = ax.scatter(df[ax0], df[ax1], c=vals, s=3, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{title}\n[{vals.min():.3g}, {vals.max():.3g}]", fontsize=9)
    ax.set_xlabel(f"{ax0} (m)"); ax.set_ylabel(f"{ax1} (m)")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, shrink=0.8)


def fig_speed(case: int):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    specs = [("XY", "systolic"), ("XY", "diastolic"),
             ("XZ", "systolic"), ("XZ", "diastolic")]
    for ax, (kind, phase) in zip(axes.ravel(), specs):
        df = load(case, kind, phase)
        if df is None:
            ax.set_visible(False); continue
        a0, a1 = plane_xy(kind)
        panel(ax, df, a0, a1, "speed", f"{kind} {phase} — speed (m/s)", "turbo")
    fig.suptitle(f"Case {case}: CFD velocity speed (each panel own scale)", fontsize=13)
    p = OUT / f"cfd_case{case:02d}_speed.png"
    fig.savefig(p, dpi=200); plt.close(fig); print("  wrote", p.name); return p


def fig_components(case: int):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for row, phase in enumerate(("systolic", "diastolic")):
        df = load(case, "XY", phase)
        for col, comp in enumerate(("u", "v", "w")):
            ax = axes[row, col]
            if df is None:
                ax.set_visible(False); continue
            panel(ax, df, "x", "y", comp, f"XY {phase} — {comp} (m/s)",
                  "RdBu_r", diverging=True)
    fig.suptitle(f"Case {case}: CFD velocity components (XY plane)", fontsize=13)
    p = OUT / f"cfd_case{case:02d}_components.png"
    fig.savefig(p, dpi=200); plt.close(fig); print("  wrote", p.name); return p


def fig_wall(case: int):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    specs = [("wss", "systolic", "turbo", False), ("wss", "diastolic", "turbo", False),
             ("p", "systolic", "viridis", False), ("p", "diastolic", "viridis", False)]
    for ax, (field, phase, cmap, div) in zip(axes.ravel(), specs):
        df = load(case, "WSS", phase)
        if df is None or field not in df.columns:
            ax.set_visible(False); continue
        unit = "Pa"
        panel(ax, df, "x", "y", field, f"wall {field} {phase} ({unit})", cmap, div)
    fig.suptitle(f"Case {case}: CFD wall shear stress + wall pressure", fontsize=13)
    p = OUT / f"cfd_case{case:02d}_wall.png"
    fig.savefig(p, dpi=200); plt.close(fig); print("  wrote", p.name); return p


def main():
    case = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Plotting CFD for case {case} -> {OUT}")
    fig_speed(case)
    fig_components(case)
    fig_wall(case)
    print("done.")


if __name__ == "__main__":
    main()
