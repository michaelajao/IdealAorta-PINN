"""Paths and physical constants.

Every filesystem path of the project is defined here, so no other module hardcodes a
folder name.

Inputs
    data/raw/full_2026-09/Case <n>/<time>.csv   whole-domain CFD exports (on request)
    data/processed/full/                        their per-block parquet (``main.py prepare``)
    data/raw/TAWSS 30.09.2026/, data/raw/MPS/,  further exports for the CFD figures only
    data/results_on_slices.csv

Outputs
    models/<run>/          trained neural fields (checkpoint and training history)
    report/runs/           one JSON record per reconstruction: method, problem and scores
    report/masks/          observed node indices of each observation draw
    report/checks/         training-free checks of the CFD fields (``audit_*``, ``wss_*``)
                           and diagnostics of trained momentum PINNs (``diag_*``)
    report/tables/         study summaries (JSON) and the CSV tables of ``main.py report``
    report/figures/        figures of ``main.py report``
    report/logs/           one log per queued job and ``done.txt`` (``main.py run-jobs``)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"

DATA_DIR: Path = PROJECT_ROOT / "data"
FULL_RAW_DIR: Path = DATA_DIR / "raw" / "full_2026-09"
FULL_DIR: Path = DATA_DIR / "processed" / "full"
# further CFD and structural exports, used only by the CFD figures (plots.py)
TAWSS_DIR: Path = DATA_DIR / "raw" / "TAWSS 30.09.2026"     # time-averaged WSS at the wall nodes
MPS_DIR: Path = DATA_DIR / "raw" / "MPS"                    # maximum principal stress, ANSYS Mechanical
SLICES_CSV: Path = DATA_DIR / "results_on_slices.csv"       # plane-averaged speed and k on planes D1-D8

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORT_DIR: Path = PROJECT_ROOT / "report"
RUNS_DIR: Path = REPORT_DIR / "runs"
MASK_DIR: Path = REPORT_DIR / "masks"
CHECKS_DIR: Path = REPORT_DIR / "checks"
TABLES_DIR: Path = REPORT_DIR / "tables"
FIGURES_DIR: Path = REPORT_DIR / "figures"
LOG_DIR: Path = REPORT_DIR / "logs"


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_constants() -> Dict[str, Any]:
    """Load and cache ``configs/constants.yaml``."""
    return load_yaml(CONFIG_DIR / "constants.yaml")


RHO: float = float(load_constants()["fluid"]["rho"])   # blood density (kg/m^3)
MU: float = float(load_constants()["fluid"]["mu"])     # molecular dynamic viscosity (Pa s)
