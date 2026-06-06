"""Demonstrate WHY the CFD panel in case01_diastolic_XY_velocity.png is one flat color.

Same CFD data (case 1, XY, diastolic), plotted three times with different color
ceilings (vmax):
  1. vmax = own 99pct (~0.12 m/s)  -> the true, rich CFD field
  2. vmax = 1.6 (systolic peak)    -> faint
  3. vmax = 12 (shared CFD|PINN scale, as in the original figure) -> one flat color

The data never changes; only the colorbar ceiling does. That is the entire
reason the CFD panel looked empty.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures" / "cfd_check"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(ROOT / "data" / "processed" / "case01_XY_diastolic.parquet")
df = df.sample(40_000, random_state=0)  # match the original figure's sampling
x, y, spd = df["x"].to_numpy(), df["y"].to_numpy(), df["speed"].to_numpy()
own = float(np.percentile(spd, 99))

settings = [
    (own, f"own diastolic scale\nvmax={own:.3f} m/s  (true field)"),
    (1.6, "systolic-peak scale\nvmax=1.6 m/s"),
    (12.0, "shared CFD|PINN scale\nvmax=12 m/s  (as in your figure)"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for ax, (vmax, title) in zip(axes, settings):
    sc = ax.scatter(x, y, c=spd, s=3, cmap="turbo", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, shrink=0.8, label="m/s")
fig.suptitle("SAME CFD data (case 1, XY, diastolic) — only the colorbar ceiling changes",
             fontsize=13)
p = OUT / "cfd_case01_diastolic_scale_demo.png"
fig.savefig(p, dpi=200); plt.close(fig)
print("wrote", p)
print(f"CFD diastolic speed: min={spd.min():.4f}  max={spd.max():.4f}  "
      f"99pct={own:.4f} m/s")
