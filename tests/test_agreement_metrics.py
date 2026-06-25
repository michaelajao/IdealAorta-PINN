"""Unit tests for the agreement metric helpers (_pearson, _ccc, _bland_altman).

Pure-numpy, so these run without ``data/raw`` or a GPU. They pin the property
that distinguishes CCC from Pearson: a perfectly correlated but scaled/offset
prediction (the jet-core over-prediction signature) keeps r=1 but drops CCC.
"""

from __future__ import annotations

import numpy as np
import pytest

from idealaorta_pinn.analysis.metrics import _bland_altman, _ccc, _pearson

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# _pearson
# --------------------------------------------------------------------------- #
def test_pearson_perfect_positive():
    x = np.linspace(0.0, 1.0, 50)
    assert _pearson(2.0 * x + 3.0, x) == pytest.approx(1.0, abs=1e-9)


def test_pearson_perfect_negative():
    x = np.linspace(0.0, 1.0, 50)
    assert _pearson(-x, x) == pytest.approx(-1.0, abs=1e-9)


def test_pearson_constant_input_is_nan():
    x = np.linspace(0.0, 1.0, 10)
    assert np.isnan(_pearson(np.ones_like(x), x))


def test_pearson_too_few_points_is_nan():
    assert np.isnan(_pearson(np.array([1.0]), np.array([1.0])))


# --------------------------------------------------------------------------- #
# _ccc  (agreement, not just correlation)
# --------------------------------------------------------------------------- #
def test_ccc_identical_is_one():
    x = RNG.normal(size=200)
    assert _ccc(x, x) == pytest.approx(1.0, abs=1e-9)


def test_ccc_penalizes_scale_that_pearson_misses():
    # Perfectly correlated but 2x over-predicted: r stays 1, CCC must drop below it.
    true = np.linspace(1.0, 5.0, 100)
    pred = 2.0 * true
    assert _pearson(pred, true) == pytest.approx(1.0, abs=1e-9)
    assert _ccc(pred, true) < 0.95


def test_ccc_penalizes_pure_offset():
    true = RNG.normal(size=200)
    assert _ccc(true + 1.0, true) < 1.0


# --------------------------------------------------------------------------- #
# _bland_altman
# --------------------------------------------------------------------------- #
def test_bland_altman_constant_offset():
    true = np.linspace(1.0, 9.0, 100)        # mean 5.0
    ba = _bland_altman(true + 2.0, true)
    assert ba["ba_bias"] == pytest.approx(2.0, abs=1e-9)
    assert ba["ba_sd"] == pytest.approx(0.0, abs=1e-9)
    assert ba["ba_loa_low"] == pytest.approx(2.0, abs=1e-9)
    assert ba["ba_loa_high"] == pytest.approx(2.0, abs=1e-9)
    assert ba["ba_bias_pct"] == pytest.approx(40.0, abs=1e-9)   # 2.0 / 5.0


def test_bland_altman_zero_bias_symmetric_noise():
    true = np.full(1000, 5.0)
    noise = RNG.normal(scale=0.5, size=1000)
    ba = _bland_altman(true + noise, true)
    assert ba["ba_bias"] == pytest.approx(0.0, abs=0.1)
    assert ba["ba_loa_high"] > ba["ba_bias"] > ba["ba_loa_low"]
