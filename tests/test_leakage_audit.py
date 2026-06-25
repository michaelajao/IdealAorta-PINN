"""Leave-one-diameter-out leakage audit.

The actual leakage vector is the case selection passed to the data layer, not the
Normalizer maths: a held-out case must influence neither the fitted scales nor the
supervision tensors, even when it is present in the ``records`` registry. This is
exactly the bug class behind the mixed-supervision checkpoints (a held case leaking
in through the wrong case list). These tests run on CPU with small point budgets so
they stay off the GPU and finish quickly; they skip when the cached data is absent.
"""

from __future__ import annotations

import pytest

from idealaorta_pinn.data.cache import load_points
from idealaorta_pinn.data.loaders import build_bundle, fit_normalizer
from idealaorta_pinn.data.registry import cases_by_id, load_registry

# Diseased leave-2.3-out split: train on 2.0 cm, hold a 2.3 cm case.
TRAIN = [1]      # 2.0 cm axisymmetric
HELD = [4]       # 2.3 cm axisymmetric (held out)
MU, RHO = 0.0035, 1060.0


@pytest.fixture(scope="module")
def records(raw_dir):
    recs = load_registry()
    if not recs:
        pytest.skip("registry empty")
    by = cases_by_id(recs)
    needed = set(TRAIN) | set(HELD)
    if not needed.issubset(by):
        pytest.skip("split case ids not in registry")
    # Probe the processed cache; skip if the parquet exports are not built.
    if load_points(by[TRAIN[0]], "3D", "systolic") is None:
        pytest.skip("processed cache not built")
    return recs


def test_normalizer_identical_with_held_case_in_registry(records):
    """Adding the held case to ``records`` (but not ``case_ids``) must not move scales."""
    by = cases_by_id(records)
    train_only = [by[c] for c in TRAIN]
    train_plus_held = [by[c] for c in TRAIN + HELD]
    phases = ["systolic", "diastolic"]

    n_clean = fit_normalizer(train_only, TRAIN, phases, MU, RHO)
    n_contaminated = fit_normalizer(train_plus_held, TRAIN, phases, MU, RHO)

    # Bit-identical: the extra held record is never iterated, so the fit is unchanged.
    assert n_clean.to_dict() == n_contaminated.to_dict()


def test_build_bundle_emits_no_held_case_supervision(records):
    """No supervision group may belong to a case outside ``case_ids``."""
    by = cases_by_id(records)
    train_plus_held = [by[c] for c in TRAIN + HELD]
    norm = fit_normalizer([by[c] for c in TRAIN], TRAIN, ["systolic"], MU, RHO)

    bundle = build_bundle(train_plus_held, TRAIN, ["systolic"], norm,
                          device="cpu", max_velocity_points=2000, n_collocation=200)

    group_cases = {g.case_id for g in bundle.groups}
    assert group_cases, "bundle produced no groups (data missing?)"
    assert group_cases.issubset(set(TRAIN))            # only train cases supervised
    assert group_cases.isdisjoint(set(HELD))           # held case never appears
    for g in bundle.groups:
        assert g.case_id in TRAIN
