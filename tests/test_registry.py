"""Tests for case discovery / registry."""

from __future__ import annotations

from idealaorta_pinn.data.registry import cases_by_id, discover_cases


def test_discovers_twelve_cases(raw_dir):
    records = discover_cases(raw_dir)
    assert len(records) == 12
    assert sorted(r.case_id for r in records) == list(range(1, 13))


def test_case_attributes(raw_dir):
    by_id = cases_by_id(discover_cases(raw_dir))

    c1 = by_id[1]
    assert c1.inlet_diameter_cm == 2.0
    assert c1.health == "diseased"
    assert c1.symmetry == "axisymmetric"
    assert c1.beta == 1.0

    c2 = by_id[2]  # "Upper" -> anterior, beta = 2.08
    assert c2.symmetry == "anterior" and c2.beta == 2.08

    c3 = by_id[3]  # "Lower" -> posterior, beta = 0.48
    assert c3.symmetry == "posterior" and c3.beta == 0.48

    c10 = by_id[10]
    assert c10.health == "healthy" and c10.inlet_diameter_cm == 2.0 and c10.beta is None


def test_phase_typos_are_resolved(raw_dir):
    """The 'Sysstolic'/'Syastolic' typos must be classified as systolic, so the
    systolic snapshots that the naive parser misses are correctly discovered."""
    by_id = cases_by_id(discover_cases(raw_dir))
    # Case 4 stores its systolic 3D as 'C4, ... Sysstolic 1.792s.csv'.
    avail4 = by_id[4].available()
    assert {"systolic", "diastolic"} <= set(avail4.get("3D", []))
    # Healthy case 10 stores systolic WSS via the 'Syastolic' typo.
    assert "systolic" in by_id[10].available().get("WSS", [])
    # Every case has both phases for the dense planes.
    for cid in range(1, 13):
        assert set(by_id[cid].available().get("XY", [])) >= {"systolic", "diastolic"}
