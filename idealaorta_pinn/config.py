"""Central path and configuration management for IdealAorta-PINN.

Everything that needs a filesystem path goes through here so that no other
module hardcodes folder names. The whole-domain CFD exports live under
``data/raw/full_2026-09/`` and are converted to the parquet cache under
``data/processed/full/`` by ``scripts/prepare.py full``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

# Repository root = parent of the package directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
# Whole-domain exports of the rigid-wall CFD runs (fluid volume, whole wall, inlet,
# outlet) at five instants per case.
FULL_RAW_DIR: Path = RAW_DIR / "full_2026-09"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FULL_DIR: Path = PROCESSED_DIR / "full"            # parquet cache of the full_2026-09 exports

# Trained neural fields live in a top-level models/ folder (one subfolder per run).
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Deliverables live in a top-level report/ folder, namespaced by output kind.
REPORT_DIR: Path = PROJECT_ROOT / "report"
METRICS_DIR: Path = REPORT_DIR / "metrics"
# Sparse-reconstruction study (idealaorta_pinn.reconstruction): per-run JSONs, oracles,
# baselines, observation masks and study summaries, one subfolder per kind.
RECON_METRICS_DIR: Path = METRICS_DIR / "reconstruction"


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_constants() -> Dict[str, Any]:
    """Load and cache ``configs/constants.yaml``."""
    return load_yaml(CONFIG_DIR / "constants.yaml")
