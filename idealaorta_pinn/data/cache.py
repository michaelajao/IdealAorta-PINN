"""Parse-once cache of the CFX CSV exports.

The raw CSVs (especially the ~270k-row plane files) are slow to re-parse. This
module converts each registry file into a compact parquet table of canonical
columns, and additionally writes a decimated ``.npz`` for the large plane files
(used by the interactive/figure code so browsers stay responsive).

Layout under ``data/processed/``::

    case01_3D_systolic.parquet
    case01_XY_diastolic.parquet
    case01_XY_diastolic.dec.npz        (decimated companion for plane files)
    ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR
from .cfdpost import read_cfdpost_csv
from .registry import CaseRecord

# Plane files are dense; decimate when caching for visualization.
_PLANE_KINDS = {"XY", "XZ"}
_DECIMATE_TARGET = 40_000  # approx number of points kept in the decimated companion


def _stem(case_id: int, kind: str, phase: str) -> str:
    return f"case{case_id:02d}_{kind}_{phase}"


def parquet_path(case_id: int, kind: str, phase: str) -> Path:
    return PROCESSED_DIR / f"{_stem(case_id, kind, phase)}.parquet"


def decimated_path(case_id: int, kind: str, phase: str) -> Path:
    return PROCESSED_DIR / f"{_stem(case_id, kind, phase)}.dec.npz"


def cache_one(record: CaseRecord, kind: str, phase: str,
              overwrite: bool = False) -> Optional[Path]:
    """Parse one (case, kind, phase) CFX file into parquet (+ decimated npz)."""
    src = record.get_file(kind, phase)
    if src is None:
        return None

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pq = parquet_path(record.case_id, kind, phase)
    if pq.exists() and not overwrite:
        return pq

    df = read_cfdpost_csv(src, canonical=True)
    df.to_parquet(pq, index=False)

    if kind in _PLANE_KINDS:
        stride = max(1, len(df) // _DECIMATE_TARGET)
        dec = df.iloc[::stride].reset_index(drop=True)
        np.savez_compressed(
            decimated_path(record.case_id, kind, phase),
            **{c: dec[c].to_numpy(dtype=np.float32) for c in dec.columns},
        )
    return pq


def build_cache(records: Iterable[CaseRecord], overwrite: bool = False) -> List[Path]:
    """Cache every file referenced by the given case records."""
    written: List[Path] = []
    for rec in records:
        for kind, phases in rec.files.items():
            for phase in phases:
                p = cache_one(rec, kind, phase, overwrite=overwrite)
                if p is not None:
                    written.append(p)
    return written


def load_points(record: CaseRecord, kind: str, phase: str,
                decimated: bool = False) -> Optional[pd.DataFrame]:
    """Load cached points for a (case, kind, phase); builds the cache if missing."""
    if decimated:
        dp = decimated_path(record.case_id, kind, phase)
        if dp.exists():
            with np.load(dp) as z:
                return pd.DataFrame({k: z[k] for k in z.files})
        # fall through to full parquet if no decimated companion
    pq = parquet_path(record.case_id, kind, phase)
    if not pq.exists():
        if cache_one(record, kind, phase) is None:
            return None
    return pd.read_parquet(pq)
