"""Streamline utilities: CFD reference traces and PINN speed sampled on them.

The figures use a single robust, integration-free view: the CFD-exported 3D
streamline geometry, coloured by CFD speed on one side and by the PINN speed
sampled at those identical points on the other. (Integrating streamlines *through*
the predicted field was tried and removed: on held-out cases the predicted field
is inaccurate and non-solenoidal, so the integration stalls into a blob rather than
tracing the lumen.)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..data.cache import load_points
from ..data.registry import CaseRecord, cases_by_id
from .predict import TrainedModel, predict_physical


def cfd_streamline_points(records: Sequence[CaseRecord], case_id: int, phase: str
                          ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """CFD 3D streamline points: returns (coords ``(N,3)``, speed ``(N,)``)."""
    rec = cases_by_id(records)[case_id]
    df = load_points(rec, "3D", phase)
    if df is None:
        return None
    coords = df[["x", "y", "z"]].to_numpy(float)
    speed = df["speed"].to_numpy(float) if "speed" in df else \
        np.linalg.norm(df[["u", "v", "w"]].to_numpy(float), axis=1)
    return coords, speed


def pinn_speed_on_points(model: TrainedModel, coords: np.ndarray, rec: CaseRecord,
                         phase: str) -> np.ndarray:
    """PINN-predicted speed at the given physical coordinates."""
    beta = rec.beta if rec.beta is not None else 1.0
    return predict_physical(model, coords, rec.inlet_diameter_cm, rec.disease_flag,
                            phase, beta=beta)["speed"]
