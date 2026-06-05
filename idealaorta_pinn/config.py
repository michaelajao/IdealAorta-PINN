"""Central path and configuration management for IdealAorta-PINN.

Everything that needs a filesystem path goes through here so that no other
module hardcodes folder names. The raw CFD case folders live under
``data/raw/`` after migration; nothing downstream depends on their (messy)
original names because discovery is done by the case registry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

# Repository root = parent of the package directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
REGISTRY_PATH: Path = DATA_DIR / "registry.json"

# Trained models live in a top-level models/ folder (one subfolder per experiment).
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Deliverables live in a top-level report/ folder: figures, metrics, interactive.
REPORT_DIR: Path = PROJECT_ROOT / "report"
FIGURES_DIR: Path = REPORT_DIR / "figures"          # paper-ready PNG figures
METRICS_DIR: Path = REPORT_DIR / "metrics"          # CSV/JSON + human-readable .txt
INTERACTIVE_DIR: Path = REPORT_DIR / "interactive"  # rotatable 3D Plotly HTML

# Manuscript: LaTeX fragments, bibliography, and a copy of the figures used.
PAPER_DIR: Path = PROJECT_ROOT / "paper"
PAPER_FIGURES_DIR: Path = PAPER_DIR / "figures"

# The original spreadsheet of slice-averaged CFD results (independent validation).
# Migrated into data/raw/ alongside the case folders.
RESULTS_SLICES_XLSX: Path = RAW_DIR / "Results on Slices.xlsx"

# Default external location of the raw CFD exports prior to migration.
# `scripts/00_migrate_data.py` moves the 12 ``Case *`` folders from here into
# ``data/raw/``. Override on the command line with ``--source`` if it differs.
DEFAULT_DATA_SOURCE: Path = Path(
    r"c:\Users\ajaoo\Downloads\Idealised Geometry Results\Idealised Geometry Results"
)


def ensure_output_dirs() -> None:
    """Create the standard model/report/paper directories if they do not exist."""
    for d in (MODELS_DIR, REPORT_DIR, FIGURES_DIR, METRICS_DIR, INTERACTIVE_DIR,
              PAPER_DIR, PAPER_FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_constants() -> Dict[str, Any]:
    """Load and cache ``configs/constants.yaml``."""
    return load_yaml(CONFIG_DIR / "constants.yaml")


@lru_cache(maxsize=1)
def load_cases() -> Dict[str, Any]:
    """Load and cache ``configs/cases.yaml`` (canonical case metadata)."""
    return load_yaml(CONFIG_DIR / "cases.yaml")


@dataclass(frozen=True)
class FluidProperties:
    """Resolved fluid properties (SI units)."""

    rho: float
    mu: float

    @property
    def nu(self) -> float:
        """Kinematic viscosity (m^2/s)."""
        return self.mu / self.rho


def fluid_properties() -> FluidProperties:
    """Return the configured blood fluid properties."""
    fluid = load_constants()["fluid"]
    return FluidProperties(rho=float(fluid["rho"]), mu=float(fluid["mu"]))


# ---------------------------------------------------------------------------
# Pulsatile inlet waveform and boundary-condition helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Waveform:
    """Pulsatile inlet flow-rate waveform Q(t) from the manuscript.

    Q(t) = Q_max * sin(pi (t - 0.5 n T) / w) + Q_base  during the systolic pulse,
         = Q_base                                       during diastole,
    with Q_base = Q_base_fraction * Q_max and w = systolic_pulse_width.
    """

    Q_max: float
    Q_base_fraction: float
    T: float
    pulse_width: float

    @property
    def Q_base(self) -> float:
        return self.Q_base_fraction * self.Q_max

    def flow_rate(self, t: float) -> float:
        """Instantaneous flow rate Q(t) (m^3/s) for absolute time ``t`` (s)."""
        n = math.floor(t / (0.5 * self.T))
        t_local = t - 0.5 * n * self.T
        if 0.0 <= t_local <= self.pulse_width:
            return self.Q_max * math.sin(math.pi * t_local / self.pulse_width) + self.Q_base
        return self.Q_base


def waveform() -> Waveform:
    w = load_constants()["waveform"]
    return Waveform(Q_max=float(w["Q_max"]), Q_base_fraction=float(w["Q_base_fraction"]),
                    T=float(w["T"]), pulse_width=float(w["systolic_pulse_width"]))


def mean_inlet_velocity(diameter_cm: float, t: float) -> float:
    """Cross-section mean inlet velocity (m/s) for an inlet diameter at time ``t``.

    U = Q(t) / A with A = pi (D/2)^2; sets the inlet velocity BC per (diameter, phase).
    """
    d_m = diameter_cm / 100.0
    area = math.pi * (d_m / 2.0) ** 2
    return waveform().flow_rate(t) / area


def outlet_pressure_pa(p_mmHg: float | None = None) -> float:
    """Outlet static pressure in Pa (defaults to the configured mid-cycle value)."""
    c = load_constants()
    if p_mmHg is None:
        p_mmHg = c["pressure"]["P_outlet_default_mmHg"]
    return float(p_mmHg) * float(c["pressure"]["mmHg_to_Pa"])
