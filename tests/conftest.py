"""Pytest fixtures and path bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.config import RAW_DIR  # noqa: E402


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    if not RAW_DIR.exists() or not any(RAW_DIR.glob("Case *")):
        pytest.skip("data/raw not populated; run scripts/00_migrate_data.py --apply")
    return RAW_DIR
