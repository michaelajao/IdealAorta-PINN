"""Global non-dimensionalization fit on training cases only.

Single length scale (see pinn/physics.py for the residual derivation):

    x_s = (x - x_mean) / L          coordinates -> ~[-1, 1]
    u_s = u / U_ref                 velocity (naturally O(1) with U_ref ~ peak speed)
    p_s = (p - p_mean) / (rho U_ref^2)   gauge pressure (P_ref = rho U_ref^2; IP-PINN Eq.17)
    tau_s = tau / tau_ref           WSS, tau_ref = mu U_ref / L  (Newtonian: tau_s = strain_s)
    d_s = D / D_ref                 parameter: non-dimensional inlet diameter
    Re  = rho U_ref L / mu

WSS targets are additionally divided by ``wss_std`` (a scalar) in the data loss
to bring the steep near-wall values to O(1); this does not affect the momentum
residual (WSS does not appear in it).

The scales are fit on the TRAINING cases only, then applied to held-out cases,
so leave-one-diameter-out validation has no information leakage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Normalizer:
    mu: float = 0.0035
    rho: float = 1060.0
    x_mean: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    L: float = 1.0
    U_ref: float = 1.0
    D_ref: float = 1.0
    p_mean: float = 0.0
    wss_std: float = 1.0

    # ---- derived ----
    @property
    def P_ref(self) -> float:
        return self.rho * self.U_ref ** 2

    @property
    def tau_ref(self) -> float:
        return self.mu * self.U_ref / self.L

    @property
    def Re(self) -> float:
        return self.rho * self.U_ref * self.L / self.mu

    # ---- fit ----
    def fit(self,
            coords: np.ndarray,
            speed: np.ndarray,
            diameters_cm: np.ndarray,
            pressure: Optional[np.ndarray] = None,
            wss_components: Optional[np.ndarray] = None,
            u_quantile: float = 0.995) -> "Normalizer":
        """Fit scales from pooled TRAINING data arrays.

        Args:
            coords: (N,3) physical coordinates (m).
            speed: (N,) velocity magnitudes (m/s).
            diameters_cm: (N,) or (k,) inlet diameters present in training (cm).
            pressure: (M,) wall pressures (Pa), optional.
            wss_components: (M,3) wall-shear components (Pa), optional.
            u_quantile: robust upper quantile for U_ref (avoids outliers).
        """
        coords = np.asarray(coords, dtype=np.float64)
        self.x_mean = coords.mean(axis=0).tolist()
        extent = float((coords.max(axis=0) - coords.min(axis=0)).max())
        self.L = max(0.5 * extent, 1e-6)

        spd = np.asarray(speed, dtype=np.float64)
        self.U_ref = max(float(np.quantile(spd, u_quantile)), 1e-3)

        self.D_ref = float(np.mean(np.asarray(diameters_cm, dtype=np.float64)))

        if pressure is not None and len(pressure):
            self.p_mean = float(np.mean(np.asarray(pressure, dtype=np.float64)))

        if wss_components is not None and len(wss_components):
            tau_s = np.asarray(wss_components, dtype=np.float64) / self.tau_ref
            self.wss_std = max(float(np.std(tau_s)), 1e-6)
        return self

    # ---- transforms ----
    def coords_std(self, coords: np.ndarray) -> np.ndarray:
        return (np.asarray(coords, dtype=np.float64) - np.asarray(self.x_mean)) / self.L

    def vel_nd(self, uvw: np.ndarray) -> np.ndarray:
        return np.asarray(uvw, dtype=np.float64) / self.U_ref

    def pressure_std(self, p: np.ndarray) -> np.ndarray:
        return (np.asarray(p, dtype=np.float64) - self.p_mean) / self.P_ref

    def wss_std_target(self, wss_components: np.ndarray) -> np.ndarray:
        """Standardized WSS target = (tau / tau_ref) / wss_std."""
        return (np.asarray(wss_components, dtype=np.float64) / self.tau_ref) / self.wss_std

    def diameter_nd(self, d_cm: float) -> float:
        return float(d_cm) / self.D_ref

    # ---- de-normalization (for reporting in physical units) ----
    def vel_to_physical(self, uvw_nd: np.ndarray) -> np.ndarray:
        return np.asarray(uvw_nd) * self.U_ref

    def pressure_to_physical(self, p_std: np.ndarray) -> np.ndarray:
        return np.asarray(p_std) * self.P_ref + self.p_mean

    def wss_to_physical(self, wss_std_val: np.ndarray) -> np.ndarray:
        return np.asarray(wss_std_val) * self.wss_std * self.tau_ref

    # ---- persistence ----
    def to_dict(self) -> Dict:
        d = asdict(self)
        d.update({"P_ref": self.P_ref, "tau_ref": self.tau_ref, "Re": self.Re})
        return d

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> "Normalizer":
        """Rebuild from a ``to_dict()`` payload (derived fields are recomputed)."""
        return cls(mu=d["mu"], rho=d["rho"], x_mean=d["x_mean"], L=d["L"],
                   U_ref=d["U_ref"], D_ref=d["D_ref"], p_mean=d["p_mean"],
                   wss_std=d["wss_std"])

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
