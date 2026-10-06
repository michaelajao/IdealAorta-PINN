"""Every figure of the study, in one place.

All plotting settings are in the configuration block below (fonts, sizes, colors, markers,
color scales, labels), so the look of every figure is changed here. Each figure is saved as
a 300 dpi PNG under ``report/figures/``.

Study results (``main.py report <study>``, from the run records in ``report/runs``)
    <study>_ratios          candidate / comparator error ratio per case-phase unit
    <study>_spacing         median pressure and WSS error against observation spacing
    <study>_pressure_forms  median pressure error in the four post-processing settings

CFD fields (``main.py figures cfd``, from the CFD exports in ``data/``)
    bc_profiles             prescribed inlet flow rate and outlet pressure over one cycle
    streamlines_<d>         medial- and transverse-plane streamlines, one figure per inlet diameter
    wss_systole, wss_diastole, tawss, mps
                            WSS, TAWSS and maximum principal stress on the unwrapped dilation segment
    xwss_lines              signed axial wall shear along three wall lines
    plane_profiles          plane-averaged speed and k on planes D1-D8

Reconstruction maps (``main.py figures maps``, from trained fields in ``models/``)
    hidden_velocity         CFD and continuity-field speed at hidden nodes of the mid-plane
    field_maps_c<case>_t<ms>  pressure error and aneurysm-zone WSS, continuity field and RBF
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from .config import FIGURES_DIR, MASK_DIR, MPS_DIR, SLICES_CSV, TAWSS_DIR

# =========================================================================== configuration
DPI = 300
RC_PARAMS = {"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False}
RC_PARAMS_MAPS = {"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                  "xtick.labelsize": 7, "ytick.labelsize": 7}
FIGSIZE = {
    "ratios": (6.6, 4.4), "spacing": (6.6, 4.4), "pressure_forms": (6.6, 4.8),
    "bc_profiles": (6.6, 2.5), "streamlines": (6.5, 7.4), "montage": (6.6, 4.6),
    "xwss_lines": (6.4, 7.4), "plane_profiles": (6.6, 2.9), "hidden_velocity": (6.8, 4.4),
    "field_maps": (6.8, 5.0),
}

# methods: legend label and color
METHODS = {
    "cont": ("Continuity field", "#1764ab"),
    "data": ("Data-only field", "#e58b23"),
    "rbf": ("RBF", "#666666"),
    "linear": ("Linear", "#b07aa1"),
    "cfd": ("Complete CFD velocity", "#6a8f61"),
}
MARKERS = {"cont": "o", "data": "s", "rbf": "^", "linear": "v"}

# geometries: rows of the montages are inlet diameters, columns the sac shapes
CASES_BY_DIAMETER = {2.0: (1, 2, 3, 10), 2.3: (4, 5, 6, 11), 2.6: (7, 8, 9, 12)}
SHAPES = ("axisymmetric", "anterior-dominant", "posterior-dominant", "control")
AXISYMMETRIC = {2.0: 1, 2.3: 4, 2.6: 7}           # defines each diameter's dilation segment
DIAMETER_COLOR = {2.0: "#1764ab", 2.3: "#e58b23", 2.6: "#3b8f3b"}
SHAPE_MARKER = {"axisymmetric": "o", "anterior-dominant": "^", "posterior-dominant": "v", "control": "s"}
SHAPE_LINE = {"axisymmetric": "-", "anterior-dominant": "--", "posterior-dominant": ":", "control": "-."}
CASE_LINE_COLOR = {1: "#ED7D31", 2: "#2E9BD6", 3: "#70AD47", 4: "#1F4E79", 5: "#548235",
                   6: "#B04FB5", 7: "#FFC000", 8: "#9E480E", 9: "#E0001B"}
CONTROL_LINE = {"color": "0.65", "lw": 0.8, "ls": "--", "zorder": 1}

# instants, geometry and color scales
T_SYS, T_DIA = 1780, 2400
PHASE = {T_SYS: "Systole", T_DIA: "Diastole"}
AXIS_Y = 0.030                 # y (m) of the descending-branch axis; the medial plane is z = 0
SEGMENT_PAD = 0.010            # m added to each side of the aneurysm zone for the montages
COLOR_SCALE = {"speed": (0, 1.6), "wss_systole": (0, 20.0), "wss_diastole": (0, 1.5),
               "tawss": (0.1, 12.0), "mps": (40, 110)}
CMAP_FIELD, CMAP_ERROR = "turbo", "PuOr_r"
TIE_COLOR, LIMIT_COLOR, MEDIAN_COLOR, DRAW_COLOR = "black", "#b22222", "#555555", "#9a9a9a"
PRESSURE_FORMS = (
    ("steady+mut", "Steady\nCFD eddy viscosity"),
    ("unsteady+mut", "Unsteady\nCFD eddy viscosity"),
    ("steady+lam", "Steady\nblood viscosity only"),
    ("unsteady+lam", "Unsteady\nblood viscosity only"),
)
# prescribed boundary conditions (one cycle of period T)
WAVEFORM = {"Q0": 2.39e-4, "T": 0.8, "systole": 0.35, "base": 0.05, "P_min": 80.0, "P_max": 120.0}


# =========================================================================== helpers
def _pyplot(rc: Dict = RC_PARAMS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    plt.rcParams.update(rc)
    return plt


def _save(fig, name: str) -> Path:
    import matplotlib.pyplot as plt
    path = FIGURES_DIR / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def case_style(case: int):
    """(inlet diameter in cm, sac shape) of Case 1-12."""
    for d, cases in CASES_BY_DIAMETER.items():
        if case in cases:
            return d, SHAPES[cases.index(case)]
    raise ValueError(case)


def wall_nodes(case: int, t_ms: int, columns: Iterable[str]) -> np.ndarray:
    """Wall and aneurysm-zone nodes of one snapshot stacked into one array."""
    from .data import load_block
    cols = list(columns)
    return np.vstack([b[cols].to_numpy(float) for b in (load_block(case, "wall", t_ms),
                                                       load_block(case, "aneurysm", t_ms)) if b is not None])


def segment_bounds(diameter: float):
    """x range (m) of the aneurysm zone of the axisymmetric case with this inlet diameter."""
    from .data import load_block
    a = load_block(AXISYMMETRIC[diameter], "aneurysm", T_SYS)
    return float(a["x"].min()), float(a["x"].max())


def descending(w: np.ndarray) -> np.ndarray:
    """Mask of the descending branch (excludes the inlet limb of the arch)."""
    return (w[:, 0] > 0.001) | (w[:, 1] > 0.0155)


def angle_from_anterior(w: np.ndarray) -> np.ndarray:
    """Angle (deg) about the descending-branch axis: 0 = anterior (+y), +-180 = posterior."""
    return np.degrees(np.arctan2(w[:, 2], w[:, 1] - AXIS_Y))


def read_tawss(case: int) -> np.ndarray:
    """(N, 4) x, y, z (m) and TAWSS (Pa) from a CFD-Post wall export, all zones stacked."""
    path = next(p for p in TAWSS_DIR.glob("*.csv") if p.stem.lower() == f"case {case}")
    rows, in_data = [], False
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("["):
            in_data = line == "[Data]"
        elif in_data and line and (line[0].isdigit() or line[0] == "-"):
            rows.append([float(v) for v in line.split(",")])
    return np.asarray(rows)


def read_mps(case: int, inner_mm: float = 0.4) -> np.ndarray:
    """(N, 4) x, y, z (m) and maximum principal stress (kPa) on the inner wall surface."""
    import pandas as pd
    from scipy.spatial import cKDTree
    d = pd.read_csv(MPS_DIR / f"Case {case}.xls", sep="\t")
    X = d.iloc[:, 1:4].to_numpy(float)
    s = d.iloc[:, 4].to_numpy(float) / 1e3
    inner = cKDTree(wall_nodes(case, T_SYS, ("x", "y", "z"))).query(X)[0] < inner_mm * 1e-3
    return np.column_stack([X[inner], s[inner]])


# =========================================================================== study results
def _draw_ratios(study: Dict) -> Dict:
    """(case, target) -> metric -> candidate/comparator ratio of each paired draw."""
    from .study import _get, _key, load_runs

    H = study["hypotheses"]
    grid, cand, comp = float(H["primary_grid"]), H["candidate"], H["comparator"]
    paths = {"p": "pressure/unsteady+mut/p_rel_l2", "wss": "wss/wss_mag_rel_l2"}
    vals: Dict = {}
    for r in load_runs():
        case, target, g = _key(r)
        if case not in study["cases"] or g != grid or (r["method"] == "nf" and r["spec"]["steps"] != study["steps"]):
            continue
        label = r["spec"]["arm"] if r["method"] == "nf" else r["method"]
        if label in (cand, comp):
            for key, path in paths.items():
                vals.setdefault((case, target, key, int(r["problem"]["seed"])), {})[label] = _get(r, path)
    out: Dict = {}
    for (case, target, key, _seed), d in vals.items():
        if cand in d and comp in d:
            out.setdefault((case, target), {}).setdefault(key, []).append(d[cand] / d[comp])
    return out


def ratio_figure(study: Dict, result: Dict) -> Path:
    from matplotlib.lines import Line2D

    plt = _pyplot()
    H = study["hypotheses"]
    cand, comp = H["candidate"], H["comparator"]
    draws = _draw_ratios(study)
    units = [(u["case"], u["target_ms"]) for u in result["units"]]
    ypos, y = {}, 0.0
    for t in sorted({t for _, t in units}):
        for c in sorted(c for c, tt in units if tt == t):
            ypos[(c, t)] = y
            y += 1
        y += 0.8
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["ratios"], sharey=True, layout="constrained")
    for ax, (title, key, limit) in zip(axes, (("Pressure", "p", H["H1_pressure"]["max_median_ratio"]),
                                              ("Aneurysm-zone WSS", "wss", H["H2_wss"]["max_median_ratio"]))):
        ratios = []
        for u in result["units"]:
            c, t = u["case"], u["target_ms"]
            r = u[cand][key] / u[comp][key]
            ratios.append(r)
            d = draws.get((c, t), {}).get(key, [])
            ax.scatter(d, [ypos[(c, t)]] * len(d), s=9, color=DRAW_COLOR, lw=0, zorder=2)
            diameter, shape = case_style(c)
            ax.scatter(r, ypos[(c, t)], s=38, marker=SHAPE_MARKER[shape], color=DIAMETER_COLOR[diameter],
                       edgecolor="black", lw=0.5, zorder=3)
        med = float(np.median(ratios))
        ax.axvline(1, color=TIE_COLOR, lw=0.9)
        ax.axvline(limit, color=LIMIT_COLOR, ls="--", lw=0.9)
        ax.axvline(med, color=MEDIAN_COLOR, ls=":", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlim(0.12, 2.2)
        ax.set_xticks([0.15, 0.25, 0.5, 1, 2], ["0.15", "0.25", "0.5", "1", "2"])
        ax.set_xlabel(f"Error ratio, {METHODS[cand][0].lower()} / {METHODS[comp][0]}")
        ax.set_title(f"{title}\n{sum(r < 1 for r in ratios)}/{len(ratios)} units below 1, "
                     f"median ratio {med:.2f}", fontsize=RC_PARAMS["font.size"])
        ax.text(limit * 0.97, -1.1, f"limit {limit}", color=LIMIT_COLOR, fontsize=7, ha="right", va="center")
        ax.set_ylim(y - 0.5, -1.7)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(list(ypos.values()), [f"Case {c}, {PHASE.get(t, t)}" for c, t in ypos])
    handles = [Line2D([], [], marker="s", ls="", color=col, label=f"{d:.1f} cm inlet") for d, col in DIAMETER_COLOR.items()]
    handles += [Line2D([], [], marker=SHAPE_MARKER[s], ls="", color="white", markeredgecolor="black", label=s)
                for s in SHAPES[:3]]
    handles += [Line2D([], [], marker="o", ls="", color=DRAW_COLOR, ms=3.5, label="single draw"),
                Line2D([], [], color=MEDIAN_COLOR, ls=":", label="median of units")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=7.5)
    return _save(fig, f"{study['name']}_ratios")


def spacing_figure(study: Dict, result: Dict) -> Path:
    plt = _pyplot()
    ladder = result["ladder"]
    grids = sorted({float(g) for g, _ in study["draws"]})
    targets = sorted({int(k.split("_")[0][1:]) for k in ladder})
    labels = [m for m in ("cont", "data", "rbf") if f"t{targets[0]}_{m}_g{grids[0]:g}" in ladder]
    fig, axes = plt.subplots(len(targets), 2, figsize=FIGSIZE["spacing"], layout="constrained", squeeze=False)
    for row, t in enumerate(targets):
        for col, (key, name) in enumerate((("p", "Pressure"), ("wss", "Aneurysm-zone WSS"))):
            ax = axes[row, col]
            for m in labels:
                ax.plot(grids, [ladder[f"t{t}_{m}_g{g:g}"][key] for g in grids], label=METHODS[m][0],
                        color=METHODS[m][1], marker=MARKERS[m])
            ax.set(xlabel="Observation spacing (mm)", ylabel="Relative L2 error",
                   title=f"{PHASE.get(t, t)}: {name}", xticks=grids)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    return _save(fig, f"{study['name']}_spacing")


def pressure_forms_figure(study: Dict, result: Dict) -> Path:
    plt = _pyplot()
    forms = result["pressure_forms"]
    targets = sorted({int(k.split("_")[0][1:]) for k in forms})
    labels = [m for m in ("cont", "data", "rbf", "cfd") if f"t{targets[0]}_{m}" in forms]
    width = 0.8 / len(labels)
    fig, axes = plt.subplots(len(targets), 1, figsize=FIGSIZE["pressure_forms"], layout="constrained", squeeze=False)
    for ax, t in zip(axes[:, 0], targets):
        for i, m in enumerate(labels):
            ax.bar(np.arange(len(PRESSURE_FORMS)) + (i - (len(labels) - 1) / 2) * width,
                   [forms[f"t{t}_{m}"][f] for f, _ in PRESSURE_FORMS], width=width,
                   label=METHODS[m][0], color=METHODS[m][1])
        ax.set(xticks=np.arange(len(PRESSURE_FORMS)), xticklabels=[name for _, name in PRESSURE_FORMS],
               ylabel="Pressure relative L2", title=PHASE.get(t, t))
        ax.grid(axis="y", alpha=0.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, ncol=len(labels), fontsize=8,
               loc="outside upper center")
    return _save(fig, f"{study['name']}_pressure_forms")


def plot_study(study: Dict, result: Dict) -> Dict[str, Path]:
    """The result figures of one confirmatory study."""
    return {"ratios": ratio_figure(study, result), "spacing": spacing_figure(study, result),
            "pressure_forms": pressure_forms_figure(study, result)}


# =========================================================================== CFD fields
def bc_profiles() -> Path:
    plt = _pyplot()
    W = WAVEFORM
    t = np.linspace(0, W["T"], 1601)
    sy = t <= W["systole"]
    Q = np.where(sy, W["Q0"] * np.sin(np.pi * t / W["systole"]) + W["base"] * W["Q0"], W["base"] * W["Q0"])
    P = np.where(sy, W["P_min"] + (W["P_max"] - W["P_min"]) * np.sin(np.pi * t / W["systole"]), W["P_min"])
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["bc_profiles"], layout="constrained")
    for ax, y, lab in ((axes[0], Q * 1e6, "Inlet flow rate (mL s$^{-1}$)"), (axes[1], P, "Outlet pressure (mmHg)")):
        ax.plot(t, y, color=METHODS["cont"][1], lw=1.6)
        for a, b in ((0.175, 0.180), (0.790, 0.800)):         # analyzed systolic and diastolic windows
            ax.axvspan(a - 0.004, b + 0.004, color=METHODS["data"][1], alpha=0.35, lw=0)
        ax.set(xlabel="Time in cycle, $t-nT$ (s)", ylabel=lab, xlim=(0, W["T"]))
        ax.grid(alpha=0.2)
    peak = (1 + W["base"]) * W["Q0"] * 1e6
    axes[0].annotate(f"peak {1 + W['base']:.2f}$\\,Q_0$ = {peak:.0f} mL s$^{{-1}}$",
                     xy=(W["systole"] / 2, peak), xytext=(0.36, 215), fontsize=7.5,
                     arrowprops={"arrowstyle": "->", "lw": 0.6})
    axes[0].annotate(f"{W['base']:.2f}$\\,Q_0$", xy=(0.6, W["base"] * W["Q0"] * 1e6), xytext=(0.55, 45),
                     fontsize=7.5, arrowprops={"arrowstyle": "->", "lw": 0.6})
    axes[1].text(0.183, 81.5, "analyzed\nwindows", fontsize=7, color="#b35900")
    return _save(fig, "bc_profiles")


def _plane_field(case: int, plane: str, h: float = 0.35e-3, slab: float = 0.45e-3):
    """Velocity in a thin slab, interpolated to a regular grid (nan outside the lumen)."""
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    from .data import load_block
    v = load_block(case, "solid", T_SYS, ["x", "y", "z", "u", "v", "w"])
    X, U = v[["x", "y", "z"]].to_numpy(float), v[["u", "v", "w"]].to_numpy(float)
    if plane == "xy":
        m = np.abs(X[:, 2]) < slab
        P, Q = X[m][:, [0, 1]], U[m][:, [0, 1]]
    else:                                                      # transverse plane through the axis
        m = np.abs(X[:, 1] - AXIS_Y) < slab
        P, Q = X[m][:, [0, 2]], U[m][:, [0, 2]]
    g0 = np.arange(P[:, 0].min(), P[:, 0].max(), h)
    g1 = np.arange(P[:, 1].min(), P[:, 1].max(), h)
    G0, G1 = np.meshgrid(g0, g1)
    inside = (cKDTree(P).query(np.c_[G0.ravel(), G1.ravel()])[0] < 0.9e-3).reshape(G0.shape)
    gu = griddata(P, Q[:, 0], (G0, G1), method="linear")
    gv = griddata(P, Q[:, 1], (G0, G1), method="linear")
    gu[~inside] = np.nan
    gv[~inside] = np.nan
    return g0, g1, gu, gv, inside


def streamlines() -> Dict[str, Path]:
    plt = _pyplot(RC_PARAMS_MAPS)
    from matplotlib.colors import Normalize
    norm = Normalize(*COLOR_SCALE["speed"])
    out = {}
    for d, cases in CASES_BY_DIAMETER.items():
        fig, axes = plt.subplots(4, 2, figsize=FIGSIZE["streamlines"], layout="constrained",
                                 gridspec_kw={"width_ratios": [1.15, 1.0]})
        for r, case in enumerate(cases):
            for k, plane in enumerate(("xy", "xz")):
                g0, g1, gu, gv, inside = _plane_field(case, plane)
                ax = axes[r, k]
                ax.contourf(g0 * 1e3, g1 * 1e3, inside.astype(float), levels=[0.5, 1.5], colors=["#f4f4f4"])
                ax.contour(g0 * 1e3, g1 * 1e3, inside.astype(float), levels=[0.5], colors="0.3", linewidths=0.6)
                strm = ax.streamplot(g0 * 1e3, g1 * 1e3, np.nan_to_num(gu), np.nan_to_num(gv),
                                     color=np.nan_to_num(np.hypot(gu, gv)), cmap=CMAP_FIELD, norm=norm,
                                     density=2.2, linewidth=0.55, arrowsize=0.5, broken_streamlines=False)
                ax.set_aspect("equal")
                ax.set_xlim(*((-45, 75) if plane == "xy" else (-15, 62)))
                ax.tick_params(length=2)
            axes[r, 0].set_ylabel(f"Case {case}, {SHAPES[r]}\ny (mm)")
            axes[r, 1].set_ylabel("z (mm)")
        axes[0, 0].set_title("medial plane ($xy$)")
        axes[0, 1].set_title(f"transverse plane ($y={AXIS_Y * 1e3:.0f}$ mm)")
        for ax in axes[-1]:
            ax.set_xlabel("x (mm)")
        cb = fig.colorbar(strm.lines, ax=axes, orientation="horizontal", shrink=0.6, pad=0.01, extend="max")
        cb.set_label(f"speed (m s$^{{-1}}$), systolic phase, inlet diameter {d} cm")
        out[f"streamlines_{d}"] = _save(fig, f"streamlines_{str(d).replace('.', 'p')}")
    return out


def _montage(values, norm, label: str, name: str, extend: str, size: float = 0.25) -> Path:
    """Three inlet diameters x four sac shapes, the dilation segment unwrapped about the axis."""
    plt = _pyplot(RC_PARAMS_MAPS)
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(3, 4, figsize=FIGSIZE["montage"], sharey=True, layout="constrained")
    for r, (d, cases) in enumerate(CASES_BY_DIAMETER.items()):
        x0, x1 = segment_bounds(d)
        x0, x1 = x0 - SEGMENT_PAD, x1 + SEGMENT_PAD
        for c_i, case in enumerate(cases):
            w = values(case)
            w = w[(w[:, 0] >= x0) & (w[:, 0] <= x1) & descending(w)]
            ax = axes[r, c_i]
            im = ax.scatter(w[:, 0] * 1e3, angle_from_anterior(w), c=w[:, 3], s=size, cmap=CMAP_FIELD,
                            norm=norm, rasterized=True)
            ax.set(xlim=(x0 * 1e3, x1 * 1e3), ylim=(-180, 180), yticks=[-180, -90, 0, 90, 180])
            ax.set_title(f"Case {case}" + (f"\n{SHAPES[c_i]}" if r == 0 else ""))
            if r == 2:
                ax.set_xlabel("x (mm)")
            if c_i == 0:
                ax.set_ylabel(f"$D_{{in}}$ = {d} cm\nangle (deg)")
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", shrink=0.5, pad=0.02, extend=extend)
    if isinstance(norm, LogNorm):
        ticks = [0.1, 0.3, 1, 3, 10]
        cb.set_ticks(ticks, labels=[f"{v:g}" for v in ticks])
        cb.minorticks_off()
    cb.set_label(label)
    return _save(fig, name)


def wss_montages() -> Dict[str, Path]:
    from matplotlib.colors import LogNorm, Normalize

    def wss(t):
        def load(case):
            w = wall_nodes(case, t, ("x", "y", "z", "wss_x", "wss_y", "wss_z"))
            return np.column_stack([w[:, :3], np.linalg.norm(w[:, 3:], axis=1)])
        return load

    return {
        "wss_systole": _montage(wss(T_SYS), Normalize(*COLOR_SCALE["wss_systole"]),
                                "WSS magnitude (Pa), systolic phase", "wss_systole", "max"),
        "wss_diastole": _montage(wss(T_DIA), Normalize(*COLOR_SCALE["wss_diastole"]),
                                 "WSS magnitude (Pa), diastolic phase", "wss_diastole", "max"),
        "tawss": _montage(read_tawss, LogNorm(*COLOR_SCALE["tawss"]), "TAWSS (Pa), log scale", "tawss", "both"),
    }


def mps_montage() -> Path:
    from matplotlib.colors import Normalize
    return _montage(read_mps, Normalize(*COLOR_SCALE["mps"]),
                    "maximum principal stress on the inner wall surface (kPa)", "mps", "both", size=0.3)


def xwss_lines(tol: float = 1.0e-3, sigma: float = 0.8e-3) -> Path:
    """Signed tau_w,x along the anterior and posterior medial wall lines and the transverse line,
    Gaussian-weighted along x so the lines have no gaps where wall nodes are sparse."""
    plt = _pyplot()
    from matplotlib.lines import Line2D
    edges = np.arange(-0.010, 0.060 + 0.5e-3, 0.5e-3)
    centres = 0.5 * (edges[1:] + edges[:-1])

    def line(w, sel):
        K = np.exp(-0.5 * ((centres[:, None] - w[sel, 0][None, :]) / sigma) ** 2)
        wsum = K.sum(1)
        out = (K @ w[sel, 3]) / np.maximum(wsum, 1e-12)
        out[wsum < 0.5] = np.nan
        return out

    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE["xwss_lines"], sharex=True, layout="constrained")
    for case in range(1, 13):
        w = wall_nodes(case, T_SYS, ("x", "y", "z", "wss_x"))
        desc = descending(w)
        sels = (desc & (np.abs(w[:, 2]) < tol) & (w[:, 1] > AXIS_Y),
                desc & (np.abs(w[:, 2]) < tol) & (w[:, 1] < AXIS_Y),
                desc & (np.abs(w[:, 1] - AXIS_Y) < tol) & (w[:, 2] > 0))
        kw = CONTROL_LINE if case >= 10 else {"color": CASE_LINE_COLOR[case], "lw": 1.3, "zorder": 2}
        for ax, sel in zip(axes, sels):
            ax.plot(centres * 1e3, line(w, sel), label=None if case >= 10 else f"Case {case}", **kw)
    for ax, title in zip(axes, ("(a) anterior wall, medial plane", "(b) posterior wall, medial plane",
                                "(c) wall in the transverse plane")):
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel(r"$\tau_{w,x}$ (Pa)")
        ax.set_xlim(-10, 60)
        ax.grid(alpha=0.25)
    axes[2].set_xlabel("x (mm)")
    h, labels = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], **{k: v for k, v in CONTROL_LINE.items() if k != "zorder"}))
    labels.append("controls (Cases 10-12)")
    fig.legend(h, labels, loc="outside upper center", ncol=5, frameon=False, fontsize=7.5)
    return _save(fig, "xwss_lines")


def plane_profiles() -> Path:
    import pandas as pd
    from matplotlib.lines import Line2D
    plt = _pyplot()
    df = pd.read_csv(SLICES_CSV)
    planes = [f"D{i}" for i in range(1, 9)]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["plane_profiles"], layout="constrained")
    for case in range(1, 13):
        d = df[df.case_id == case].set_index("slice").loc[planes]
        diameter, shape = case_style(case)
        col = DIAMETER_COLOR[diameter]
        kw = {"color": col, "ls": SHAPE_LINE[shape], "marker": SHAPE_MARKER[shape], "ms": 4,
              "lw": 1.2 if shape == "control" else 1.4, "markerfacecolor": "white" if shape == "control" else col}
        axes[0].plot(range(1, 9), d["avg_vel_mps"], **kw)
        axes[1].plot(range(1, 9), d["tke"] * 1e3, **kw)
    for ax, lab in zip(axes, ("Plane-averaged speed (m s$^{-1}$)", "Plane-averaged $k$ ($10^{-3}$ m$^2$ s$^{-2}$)")):
        ax.set(xticks=range(1, 9), xticklabels=planes, ylabel=lab, xlabel="Plane")
        ax.axvspan(4.5, 5.5, color="0.9", zorder=0)
        ax.grid(alpha=0.2)
    handles = [Line2D([], [], color=c, lw=2, label=f"$D_{{in}}$ = {d} cm") for d, c in DIAMETER_COLOR.items()]
    handles += [Line2D([], [], color="k", ls=SHAPE_LINE[s], marker=SHAPE_MARKER[s],
                       markerfacecolor="white" if s == "control" else "k", label=s) for s in SHAPES]
    fig.legend(handles=handles, frameon=False, fontsize=7, loc="outside lower center", ncol=4)
    return _save(fig, "plane_profiles")


def plot_cfd() -> Dict[str, Path]:
    """Every CFD-field figure (needs the CFD exports in data/)."""
    out = {"bc_profiles": bc_profiles()}
    out.update(streamlines())
    out.update(wss_montages())
    out["mps"] = mps_montage()
    out["xwss_lines"] = xwss_lines()
    out["plane_profiles"] = plane_profiles()
    return out


# =========================================================================== reconstruction maps
def _load_mask(case: int, times, grid: float = 2.5, seed: int = 0) -> np.ndarray:
    return np.load(MASK_DIR / f"case{case:02d}_t{'-'.join(map(str, times))}_grid{grid:g}_s{seed}_obs_ids.npy")


def hidden_velocity(examples=((4, T_SYS), (8, T_DIA))) -> Path:
    """CFD and continuity-field speed and their vector error at hidden nodes of the mid-plane slab."""
    import torch

    from .data import load_block
    from .training import load_field, predict
    plt = _pyplot()
    fig, axes = plt.subplots(len(examples), 3, figsize=FIGSIZE["hidden_velocity"], layout="constrained",
                             squeeze=False)
    for row, (case, t) in enumerate(examples):
        net, scales, spec = load_field(f"cont_oracle_c{case:02d}_t{t}_g2.5_s0_i0_w0.01")
        base = load_block(case, "solid", spec["times"][0])
        target = load_block(case, "solid", t)
        interior = np.linalg.norm(base[["u", "v", "w"]].to_numpy(), axis=1) > 0
        xyz = target[["x", "y", "z"]].to_numpy()[interior]
        truth = target[["u", "v", "w"]].to_numpy()[interior]
        hidden = np.ones(len(xyz), bool)
        hidden[_load_mask(case, spec["times"])] = False
        z0 = 0.5 * (xyz[:, 2].min() + xyz[:, 2].max())
        ids = np.flatnonzero(hidden & (np.abs(xyz[:, 2] - z0) <= 0.35e-3))
        pred = predict(net, xyz[ids], t, scales, torch.device("cpu"))[:, :3]
        speed, pspeed = np.linalg.norm(truth[ids], axis=1), np.linalg.norm(pred, axis=1)
        error = np.linalg.norm(pred - truth[ids], axis=1)
        vmax = max(speed.max(), pspeed.max())
        for col, (values, title, cmap, limit) in enumerate([
                (speed, "CFD speed", "viridis", vmax),
                (pspeed, f"{METHODS['cont'][0]} speed", "viridis", vmax),
                (error, "Vector error magnitude", "magma", error.max())]):
            ax = axes[row, col]
            im = ax.scatter(xyz[ids, 0] * 1e3, xyz[ids, 1] * 1e3, c=values, s=0.3, cmap=cmap, vmin=0,
                            vmax=limit, rasterized=True)
            ax.set_aspect("equal")
            ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"Case {case}, {PHASE[t].lower()}\n{title}")
            fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.06, label="m s$^{-1}$", shrink=0.9)
    return _save(fig, "hidden_velocity")


def field_maps(case: int = 4, target: int = T_SYS) -> Path:
    """Mid-plane pressure error and unwrapped aneurysm-zone WSS of the continuity field and RBF
    (draw 0 at 2.5 mm), recomputed through the shared post-processing."""
    import torch

    from .interpolation import rbf_fields
    from .postprocess import PressureIntegrator, wall_interpolant, wss_newton2
    from .problem import build_problem
    from .training import load_field, predict
    plt = _pyplot({**RC_PARAMS, "font.size": 8})
    net, scales, spec = load_field(f"cont_oracle_c{case:02d}_t{target}_g2.5_s0_i0_w0.01")
    P = build_problem(case, spec["times"], target, 2.5, 0, save_mask=False)
    if not np.array_equal(_load_mask(case, spec["times"]), P.obs_ids):
        raise ValueError("observation mask differs from the saved draw")
    other = next(t for t in P.times_ms if t != target)
    tgt = P.snaps[target]
    fields = {METHODS["cont"][0]: {t: predict(net, P.X, t, scales, torch.device("cpu"))[:, :3] for t in P.times_ms},
              METHODS["rbf"][0]: rbf_fields(P)[0]}
    integ = PressureIntegrator(P.X, P.wall_xyz, P.dwall)
    p_ref = integ.gauge_ref(tgt.p)
    an = P.wall_aneurysm
    w_ref = np.linalg.norm(tgt.wall_wss[an], axis=1)
    res = {}
    for k, F in fields.items():
        dUdt = (F[target] - F[other]) / ((target - other) / 1000)
        p = integ.pressure(F[target], dUdt, tgt.mut, "unsteady+mut")
        W = wss_newton2(wall_interpolant(P.X, F[target], P.wall_xyz), P.wall_xyz, P.wall_normals, 0.25)
        res[k] = (p, np.linalg.norm(W[an], axis=1))

    Xs = P.X[integ.sub]
    sl = np.abs(Xs[:, 2] - 0.5 * (Xs[:, 2].min() + Xs[:, 2].max())) < 0.35e-3
    Wx = P.wall_xyz[an]
    theta = angle_from_anterior(Wx)
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE["field_maps"], layout="constrained")
    # error scale: 99th percentile of |error| over both methods, so isolated outliers saturate
    emax = np.quantile(np.concatenate([np.abs(res[k][0] - p_ref)[sl] for k in res]), 0.99)
    panels = [(p_ref, "CFD gauge pressure (Pa)", "viridis", (p_ref[sl].min(), p_ref[sl].max()))]
    panels += [(res[k][0] - p_ref, f"{k} error (Pa)", CMAP_ERROR, (-emax, emax)) for k in res]
    for ax, (v, title, cmap, lim) in zip(axes[0], panels):
        im = ax.scatter(Xs[sl, 0] * 1e3, Xs[sl, 1] * 1e3, c=v[sl], s=0.4, cmap=cmap, vmin=lim[0], vmax=lim[1],
                        rasterized=True)
        ax.set_aspect("equal")
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=title)
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.9)
    wmax = np.quantile(w_ref, 0.995)
    for ax, (v, title) in zip(axes[1], [(w_ref, "CFD $|\\tau_w|$ (Pa)")]
                              + [(res[k][1], f"{k} $|\\tau_w|$ (Pa)") for k in res]):
        im = ax.scatter(Wx[:, 0] * 1e3, theta, c=v, s=0.4, cmap="viridis", vmin=0, vmax=wmax, rasterized=True)
        ax.set(xlabel="x (mm)", ylabel="angle from anterior (deg)", title=title, ylim=(-180, 180),
               yticks=[-180, -90, 0, 90, 180])
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.9)
    return _save(fig, f"field_maps_c{case:02d}_t{target}")


def plot_maps() -> Dict[str, Path]:
    """Reconstruction maps from the trained fields (needs models/ and the CFD exports)."""
    return {"hidden_velocity": hidden_velocity(),
            "field_maps_c04_t1780": field_maps(4, T_SYS),
            "field_maps_c07_t2400": field_maps(7, T_DIA)}
