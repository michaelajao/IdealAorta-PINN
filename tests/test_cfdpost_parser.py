"""Tests for the CFX parser and filename interpretation."""

from __future__ import annotations

from idealaorta_pinn.data.cfx import (
    detect_data_kind,
    normalize_phase,
    parse_time_seconds,
    read_cfx_export_csv,
)
from idealaorta_pinn.data.registry import discover_cases


def test_phase_typo_normalization():
    assert normalize_phase("3D Velocity Streamlines Systolic 1.778s.csv") == "systolic"
    assert normalize_phase("C4, 3D Velocity Streamlines Sysstolic 1.792s.csv") == "systolic"
    assert normalize_phase("WSS at Syastolic Phase 1.775s.csv") == "systolic"
    assert normalize_phase("XY Plane Velocity Streamlines Diastolic 2.4s.csv") == "diastolic"
    assert normalize_phase("C9 Distolic 2.4s.csv") == "diastolic"


def test_data_kind_detection():
    assert detect_data_kind("3D Velocity Streamlines Systolic 1.778s.csv") == "3D"
    assert detect_data_kind("C5, XY Plane Velocity Streamlines Diastolic 2.4s.csv") == "XY"
    assert detect_data_kind("XZ Plane Velocity Streamlines Systolic 1.8s.csv") == "XZ"
    assert detect_data_kind("WSS and Pressure on Aneurysm at Systolic Phase 1.778s.csv") == "WSS"


def test_time_parsing():
    assert parse_time_seconds("... Systolic 1.778s.csv") == 1.778
    assert parse_time_seconds("... Diastolic 2.4s.csv") == 2.4
    assert parse_time_seconds("no time here.csv") is None


def test_parse_real_case1_3d(raw_dir):
    records = discover_cases(raw_dir)
    case1 = next(r for r in records if r.case_id == 1)
    path = case1.get_file("3D", "systolic")
    assert path is not None and path.exists()
    df = read_cfx_export_csv(path, canonical=True)
    assert {"x", "y", "z", "u", "v", "w", "speed"}.issubset(df.columns)
    assert len(df) > 1000
    # velocity magnitude should be physically plausible (m/s)
    assert df["speed"].abs().max() < 10.0
