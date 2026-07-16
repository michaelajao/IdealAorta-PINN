"""Robust parser for ANSYS CFD-Post CSV exports.

CFD-Post is ANSYS's unified post-processor; it reads both CFX and Fluent solver
results, so this export format is solver-agnostic (the present dataset was
post-processed in CFD-Post regardless of the underlying solver).

CFD-Post export format
----------------------
A short metadata block, then a line ``[Data]``, then a single header row of
comma-separated column names, then comma-separated numeric rows::

    [Name]
    D Streamlines

    [Data]
    X [ m ], Y [ m ], Z [ m ], Velocity [ m s^-1 ], Velocity u [ m s^-1 ], ...
    0.0, -7.05e-04, 3.73e-04, 7.98e-01, -7.98e-01, 0.0, 0.0
    ...

This module ports the proven parser from the project's original
``generate_diseased_streamline_plots.py`` and adds:
  * canonical column renaming (``X [ m ]`` -> ``x`` etc.),
  * data-kind detection (3D / XY / XZ / WSS) from the filename,
  * phase + systolic-timestamp parsing that tolerates the filename typos present
    in the dataset (``Sysstolic``, ``Syastolic``, ``Distolic``, ...).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical column names
# ---------------------------------------------------------------------------
# Each canonical name maps to a list of lowercase prefixes that identify the
# raw CFX column (matched after stripping/lower-casing). Order matters: more
# specific names (e.g. "velocity u") must be checked before generic ones
# (e.g. "velocity").
_COLUMN_RULES: List[tuple[str, str]] = [
    ("u", "velocity u"),
    ("v", "velocity v"),
    ("w", "velocity w"),
    ("speed", "velocity ["),      # "Velocity [ m s^-1 ]" -> magnitude
    ("wss_x", "wall shear x"),
    ("wss_y", "wall shear y"),
    ("wss_z", "wall shear z"),
    ("wss", "wall shear ["),      # "Wall Shear [ Pa ]" -> magnitude
    ("p", "pressure"),
    ("x", "x ["),
    ("y", "y ["),
    ("z", "z ["),
]


def _canonical_name(raw: str) -> Optional[str]:
    key = raw.strip().lower()
    for canon, prefix in _COLUMN_RULES:
        if key.startswith(prefix):
            return canon
    return None


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw CFD-Post columns to canonical names; drop unrecognized/duplicate columns.

    Streamline exports can repeat the coordinate columns
    (``X,Y,Z,Velocity,u,v,w,X,Y,Z``); we keep the FIRST column mapping to each
    canonical name and select positionally, so duplicate raw labels never produce
    duplicate ``x``/``y``/``z`` outputs (which would break ``df[["x","y","z"]]``).
    """
    seen: set[str] = set()
    keep_pos: List[int] = []
    names: List[str] = []
    for pos, col in enumerate(df.columns):
        canon = _canonical_name(str(col))
        if canon is None or canon in seen:
            continue
        seen.add(canon)
        keep_pos.append(pos)
        names.append(canon)
    out = df.iloc[:, keep_pos].copy()
    out.columns = names
    return out


def read_cfdpost_csv(path: str | Path, canonical: bool = True) -> pd.DataFrame:
    """Parse a CFD-Post-exported CSV into a numeric DataFrame.

    Args:
        path: CSV file path.
        canonical: if True, rename columns to canonical names (``x``, ``y``,
            ``z``, ``u``, ``v``, ``w``, ``speed``, ``p``, ``wss``, ``wss_x`` ...).

    Returns:
        DataFrame of float columns with NaN rows dropped.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    data_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "[Data]"), None)
    if data_idx is None or data_idx + 1 >= len(lines):
        raise ValueError(f"Could not find a [Data] section in {path}")

    header = [c.strip() for c in lines[data_idx + 1].split(",")]
    n_cols = len(header)

    rows: List[List[str]] = []
    for raw in lines[data_idx + 2:]:
        if not raw.strip():
            continue
        # A new [Name]/[Data] block would mark the end of a homogeneous table.
        if raw.lstrip().startswith("["):
            break
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != n_cols:
            continue
        rows.append(parts)

    # Build positionally: a streamline export can repeat column NAMES
    # (X,Y,Z,...,X,Y,Z), and per-label ``to_numeric`` would then receive a
    # DataFrame (duplicate label) and raise. Integer columns sidestep that; the
    # raw header names are reattached afterwards.
    df = pd.DataFrame(rows)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="any").reset_index(drop=True)
    df.columns = header
    if not canonical:
        return df
    # Streamline exports emit each point twice; drop exact-duplicate rows so a
    # validation/figure slice is not silently up-weighted by the duplication.
    return canonicalize_columns(df).drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Filename interpretation
# ---------------------------------------------------------------------------
_DATA_KIND_PATTERNS: List[tuple[str, str]] = [
    ("3D", r"3\s*d\s*velocity\s*streamlines"),
    ("XY", r"xy\s*plane"),
    ("XZ", r"xz\s*plane"),
    ("WSS", r"wss\s*and\s*pressure"),
]


def detect_data_kind(name: str) -> str:
    """Classify a CFD-Post export by its filename: '3D', 'XY', 'XZ', 'WSS' or 'unknown'."""
    low = name.lower()
    for kind, pat in _DATA_KIND_PATTERNS:
        if re.search(pat, low):
            return kind
    return "unknown"


def normalize_phase(name: str) -> str:
    """Map a filename to 'systolic' / 'diastolic' / 'unknown', tolerating typos.

    The dataset contains spellings such as ``Sysstolic``, ``Syastolic``,
    ``Distolic``, ``Diatolic``, ``Diaatolic``.
    """
    low = name.lower()
    # Diastolic must be checked first: it starts with 'd' (no 'd' in systolic).
    # ``d\w*tolic`` catches diastolic / distolic / diatolic / diaatolic / dstolic.
    if re.search(r"d\w*tolic", low):
        return "diastolic"
    # ``s\w*tolic`` catches systolic / sysstolic / syastolic / sustolic.
    if re.search(r"s\w*tolic", low):
        return "systolic"
    return "unknown"


def parse_time_seconds(name: str) -> Optional[float]:
    """Extract the snapshot time in seconds from a filename (e.g. '1.778s' -> 1.778)."""
    m = re.search(r"(\d+\.\d+)\s*s\b", name.lower())
    return float(m.group(1)) if m else None
