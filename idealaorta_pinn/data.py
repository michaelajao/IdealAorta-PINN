"""Whole-domain CFD exports: parsing, the parquet cache and the snapshot loaders.

The exports (``data/raw/full_2026-09/Case N/<time>.csv``) hold one ANSYS CFD-Post CSV
per (case, time) with five ``[Name]`` / ``[Data]`` blocks. ``convert_all``
(``main.py prepare``) splits them once into
``data/processed/full/caseNN_<block>_t<ms>.parquet``:

    solid      every node of the fluid domain (the block keeps the geometry body's
               default name) with u, v, w, p and the eddy viscosity mu_t
    wall,      the lumen wall, split into two named zones that share a seam ring;
    aneurysm   healthy cases have no aneurysm zone
    inlet      the x = 0 face (uniform plug in -x)
    outlet     the x = 0.135 m face
"""

from __future__ import annotations

import io
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import FULL_DIR, FULL_RAW_DIR, RHO

SPLIT_TRAIN, SPLIT_TEST, SPLIT_VAL = 0, 1, 2

# Raw filenames are kept verbatim; their times are interpreted here. Case 1's "1.755.csv"
# carries the waveform-peak inlet velocity (0.7988 m/s, as in every other 2.0 cm case at
# 1.775 s), so it is the 1.775 s snapshot. Case 2's "1.755.csv" matches the waveform at
# 1.755 s and is kept as such.
TIME_RELABEL_MS = {(1, "1.755"): 1775}

# Canonical column names: (canonical, lowercase prefix of the CFD-Post column). More
# specific prefixes ("velocity u") come before generic ones ("velocity [").
_COLUMN_RULES: List[tuple] = [
    ("u", "velocity u"), ("v", "velocity v"), ("w", "velocity w"), ("speed", "velocity ["),
    ("wss_x", "wall shear x"), ("wss_y", "wall shear y"), ("wss_z", "wall shear z"),
    ("wss", "wall shear ["), ("p", "pressure"), ("x", "x ["), ("y", "y ["), ("z", "z ["),
    # CFD-Post names and the Fluent names it passes through
    ("mu_t", "eddy viscosity"), ("mu_t", "turbulent viscosity"),
    ("k", "turbulence kinetic energy"), ("k", "turbulent kinetic energy"),
    ("omega", "turbulence eddy frequency"), ("omega", "specific dissipation rate"),
    ("gamma", "intermittency"), ("wall_dist", "wall distance"),
]


# --------------------------------------------------------------------------- parsing
def _canonical_name(raw: str) -> Optional[str]:
    key = raw.strip().lower()
    return next((canon for canon, prefix in _COLUMN_RULES if key.startswith(prefix)), None)


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename CFD-Post columns to canonical names; drop unrecognized and repeated ones."""
    seen, keep, names = set(), [], []
    for pos, col in enumerate(df.columns):
        canon = _canonical_name(str(col))
        if canon is None or canon in seen:
            continue
        seen.add(canon)
        keep.append(pos)
        names.append(canon)
    out = df.iloc[:, keep].copy()
    out.columns = names
    return out


def read_cfdpost_blocks(path: str | Path) -> Dict[str, pd.DataFrame]:
    """Parse a multi-block CFD-Post export into ``{block name: DataFrame}``.

    Surface-only fields such as wall shear are ``null`` at interior nodes and become NaN;
    a column that is null throughout a block is removed, and the velocity magnitude is
    dropped (it is recomputable from u, v, w). Coordinates are float64, fields float32.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    blocks: Dict[str, pd.DataFrame] = {}
    for chunk in text.split("[Name]")[1:]:
        name, _, rest = chunk.strip().partition("\n")
        _, _, table = rest.partition("[Data]")
        df = pd.read_csv(io.StringIO(table.strip()), na_values=["null"], skipinitialspace=True)
        df = canonicalize_columns(df).drop(columns=["speed"], errors="ignore")
        df = df.dropna(axis=1, how="all")
        for col in df.columns:
            df[col] = df[col].astype(np.float64 if col in ("x", "y", "z") else np.float32)
        blocks[name.strip()] = df
    return blocks


# --------------------------------------------------------------------------- parquet cache
def raw_time_ms(case_id: int, stem: str) -> int:
    """Snapshot time in ms for a raw export filename stem such as ``"1.78"`` or ``"2.40"``."""
    return TIME_RELABEL_MS.get((case_id, stem), int(round(1000 * float(stem))))


def raw_exports(root: Path = FULL_RAW_DIR) -> List[tuple]:
    """``(case_id, t_ms, path)`` for every raw whole-domain export under ``root``."""
    out = []
    for folder in sorted(root.glob("Case *")):
        m = re.fullmatch(r"Case (\d+)", folder.name)
        if not m:
            continue
        case_id = int(m.group(1))
        out += [(case_id, raw_time_ms(case_id, csv.stem), csv) for csv in sorted(folder.glob("*.csv"))]
    return out


def convert_export(case_id: int, t_ms: int, path: Path, overwrite: bool = False) -> List[Path]:
    """Split one raw export into per-block parquet files; returns the files written."""
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    targets = {b: FULL_DIR / f"case{case_id:02d}_{b}_t{t_ms}.parquet"
               for b in ("solid", "wall", "aneurysm", "inlet", "outlet")}
    if not overwrite and all(p.exists() for b, p in targets.items() if b != "aneurysm"):
        return []
    written = []
    for block, df in read_cfdpost_blocks(path).items():
        dest = FULL_DIR / f"case{case_id:02d}_{block}_t{t_ms}.parquet"
        if overwrite or not dest.exists():
            df.to_parquet(dest, index=False)
            written.append(dest)
    return written


def convert_all(cases: Optional[List[int]] = None, overwrite: bool = False) -> List[Path]:
    """Convert every raw whole-domain export (optionally only some cases)."""
    written = []
    for case_id, t_ms, path in raw_exports():
        if not cases or case_id in cases:
            written += convert_export(case_id, t_ms, path, overwrite=overwrite)
    return written


# --------------------------------------------------------------------------- loaders
def load_block(case_id: int, block: str, t_ms: int, columns=None) -> Optional[pd.DataFrame]:
    path = FULL_DIR / f"case{case_id:02d}_{block}_t{t_ms}.parquet"
    return pd.read_parquet(path, columns=columns) if path.exists() else None


def block_split(coords: np.ndarray, block_mm: float, val_every: int = 5) -> np.ndarray:
    """3D checkerboard split: SPLIT_TRAIN, SPLIT_TEST or SPLIT_VAL per point.

    Odd cubes are withheld; the withheld cubes whose index hash is 0 mod ``val_every``
    become the monitoring set, the rest the test set.
    """
    ijk = np.floor(np.asarray(coords, float) / (block_mm / 1000.0)).astype(np.int64)
    held = (ijk.sum(axis=1) % 2) == 1
    val = held & (((ijk[:, 0] + 2 * ijk[:, 1] + 3 * ijk[:, 2]) % int(val_every)) == 0)
    split = np.full(len(ijk), SPLIT_TRAIN, dtype=np.int8)
    split[held] = SPLIT_TEST
    split[val] = SPLIT_VAL
    return split


def wall_normals_from_volume(wall: np.ndarray, interior: np.ndarray,
                             k: int = 16, k_orient: int = 8) -> np.ndarray:
    """Unit wall normals pointing into the lumen.

    The direction is the local-PCA surface normal; the sign comes from the nearest
    interior fluid nodes (the U-shaped vessel's centroid lies outside the lumen, so
    flipping toward it would fail).
    """
    wall = np.asarray(wall, float)
    _, nb = cKDTree(wall).query(wall, k=min(k, len(wall)))
    P = wall[nb] - wall[nb].mean(axis=1, keepdims=True)
    _, vec = np.linalg.eigh(np.einsum("nki,nkj->nij", P, P))
    n = vec[:, :, 0]
    _, j = cKDTree(interior).query(wall, k=k_orient)
    s = np.sign(np.einsum("ij,ij->i", n, interior[j].mean(axis=1) - wall))
    n = n * np.where(s == 0, 1.0, s)[:, None]
    return n / np.linalg.norm(n, axis=1, keepdims=True)


@lru_cache(maxsize=32)
def load_full_snapshot(case_id: int, t_ms: int, block_mm: float = 2.0, val_every: int = 5,
                       rho: float = RHO) -> Dict[str, np.ndarray]:
    """All arrays for one (case, time), in SI units.

    ``vol_*`` are the interior fluid nodes (the zero-velocity wall nodes of the volume
    block are dropped; the wall arrays carry them), ``vol_nut`` is mu_t / rho and
    ``vol_split`` the checkerboard label. The wall is the union of the wall zones,
    de-duplicated along the shared seam. Cached: treat the arrays as read-only.
    """
    vol = load_block(case_id, "solid", t_ms, ["x", "y", "z", "u", "v", "w", "p", "mu_t"])
    if vol is None:
        raise FileNotFoundError(f"no export for case {case_id} at {t_ms} ms in {FULL_DIR}")
    xyz = vol[["x", "y", "z"]].to_numpy(np.float64)
    uvw = vol[["u", "v", "w"]].to_numpy(np.float64)
    inside = np.linalg.norm(uvw, axis=1) > 0

    parts = [d.assign(_aneurysm=(b == "aneurysm")) for b in ("aneurysm", "wall")
             if (d := load_block(case_id, b, t_ms)) is not None]
    wall = pd.concat(parts, ignore_index=True)
    wxyz = wall[["x", "y", "z"]].to_numpy(np.float64)
    _, first = np.unique(np.round(wxyz, 9), axis=0, return_index=True)   # seam ring -> aneurysm
    wall = wall.iloc[np.sort(first)]
    wxyz = wall[["x", "y", "z"]].to_numpy(np.float64)
    wall_aneurysm = wall["_aneurysm"].to_numpy(bool)
    _, nearest_wall = cKDTree(wxyz).query(xyz[inside])

    inlet = load_block(case_id, "inlet", t_ms)
    outlet = load_block(case_id, "outlet", t_ms)
    iuvw = inlet[["u", "v", "w"]].to_numpy(np.float64)
    moving = np.linalg.norm(iuvw, axis=1) > 0          # drop the no-slip rim of the inlet disk
    return {
        "vol_xyz": xyz[inside], "vol_uvw": uvw[inside],
        "vol_p": vol["p"].to_numpy(np.float64)[inside],
        "vol_nut": vol["mu_t"].to_numpy(np.float64)[inside] / rho,
        "vol_split": block_split(xyz[inside], block_mm, val_every),
        "wall_xyz": wxyz, "wall_p": wall["p"].to_numpy(np.float64),
        "wall_wss": wall[["wss_x", "wss_y", "wss_z"]].to_numpy(np.float64),
        "wall_normals": wall_normals_from_volume(wxyz, xyz[inside]),
        "inlet_xyz": inlet[["x", "y", "z"]].to_numpy(np.float64)[moving],
        "inlet_uvw": iuvw[moving],
        "outlet_xyz": outlet[["x", "y", "z"]].to_numpy(np.float64),
        "outlet_p": outlet["p"].to_numpy(np.float64),
        # the aneurysm wall zone, and the volume nodes whose nearest wall node is on it
        # (the sac); both all-False for healthy cases
        "wall_aneurysm": wall_aneurysm,
        "vol_sac": wall_aneurysm[nearest_wall],
    }
