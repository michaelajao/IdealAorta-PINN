"""Checks on study plumbing that silently corrupted results once, and on data conversion."""

import pytest

from idealaorta_pinn.config import FULL_DIR, FULL_RAW_DIR
from idealaorta_pinn.data.full_export import raw_time_ms
from idealaorta_pinn.reconstruction.reference import time_derivative
from idealaorta_pinn.reconstruction.study import jobs, load_study

needs_cache = pytest.mark.skipif(not (FULL_DIR / "case01_solid_t1780.parquet").exists(),
                                 reason="whole-domain parquet cache not present")


def test_raw_time_labels():
    assert raw_time_ms(1, "1.755") == 1775          # mislabelled peak-systole file
    assert raw_time_ms(2, "1.755") == 1755          # genuine 1.755 s snapshot
    assert raw_time_ms(6, "1.78") == 1780 and raw_time_ms(7, "2.40") == 2400


@needs_cache
def test_time_derivative_rejects_misordered_snapshots():
    # a "later" snapshot that precedes t flips the sign of du/dt; it must be refused
    with pytest.raises(ValueError):
        time_derivative(1, 1780, t_before=1775, t_after=1778)
    with pytest.raises(ValueError):
        time_derivative(1, 1780, t_before=1788, t_after=None)


def test_study_matrices_match_the_registered_runs():
    assert len(jobs(load_study("pilot_select"), "train")) == 13
    assert len(jobs(load_study("pilot_main"), "train")) == 30
    assert len(jobs(load_study("confirm"), "train")) == 140
    assert len(jobs(load_study("confirm"), "baseline")) == 7 * 2 * 5 * 2 + 7 * 2    # + the CFD floor
    case2 = [j for j in jobs(load_study("confirm"), "train") if "--case 2 " in j and "--target 1780" in j]
    assert case2 and all("--times 1780 1788" in j for j in case2)     # Case 2 has no 1.775 s export


@pytest.mark.skipif(not (FULL_RAW_DIR / "Case 10" / "2.4.csv").exists(), reason="raw exports not present")
@needs_cache
def test_converter_reproduces_cache():
    import numpy as np
    import pandas as pd

    from idealaorta_pinn.data.cfdpost import read_cfdpost_blocks
    for block, df in read_cfdpost_blocks(FULL_RAW_DIR / "Case 10" / "2.4.csv").items():
        ref = pd.read_parquet(FULL_DIR / f"case10_{block}_t2400.parquet")
        assert list(df.columns) == list(ref.columns) and list(df.dtypes) == list(ref.dtypes)
        np.testing.assert_array_equal(df.to_numpy(), ref.to_numpy())
