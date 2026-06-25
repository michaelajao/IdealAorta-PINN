"""Unit tests for the swirl / recirculation metrics and sac-geometry helpers.

Pure-numpy synthetic geometry, so these run without ``data/raw``.
"""

from __future__ import annotations

import numpy as np
import pytest

from idealaorta_pinn.analysis.metrics import (recirculation_fraction,
                                              secondary_flow_fraction)
from idealaorta_pinn.data.geometry import lumen_inside_mask, sac_axial_band

AXIS = np.array([1.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# secondary_flow_fraction
# --------------------------------------------------------------------------- #
def test_secondary_flow_pure_axial_is_zero():
    v = np.tile(AXIS, (100, 1)) * np.linspace(0.5, 2.0, 100)[:, None]
    assert secondary_flow_fraction(v, AXIS) == pytest.approx(0.0, abs=1e-12)


def test_secondary_flow_pure_transverse_is_one():
    v = np.tile([0.0, 1.0, 0.0], (100, 1)) * np.linspace(0.5, 2.0, 100)[:, None]
    assert secondary_flow_fraction(v, AXIS) == pytest.approx(1.0, abs=1e-12)


def test_secondary_flow_45_degrees_is_sqrt_half():
    v = np.tile([1.0, 1.0, 0.0], (50, 1))
    assert secondary_flow_fraction(v, AXIS) == pytest.approx(1.0 / np.sqrt(2), abs=1e-9)


def test_secondary_flow_speed_floor_excludes_small_vectors():
    # 90 axial fast + 10 transverse slow: the floor drops the slow ones -> ~0.
    fast = np.tile(AXIS, (90, 1)) * 1.0
    slow = np.tile([0.0, 1.0, 0.0], (10, 1)) * 1e-3
    v = np.vstack([fast, slow])
    assert secondary_flow_fraction(v, AXIS, speed_floor=0.1) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# recirculation_fraction
# --------------------------------------------------------------------------- #
def test_recirc_all_aligned_is_zero():
    v = np.tile(AXIS, (100, 1))
    assert recirculation_fraction(v, AXIS) == 0.0


def test_recirc_all_reversed_is_one():
    v = np.tile(-AXIS, (100, 1))
    assert recirculation_fraction(v, AXIS) == 1.0


def test_recirc_half_reversed_is_half():
    v = np.vstack([np.tile(AXIS, (50, 1)), np.tile(-AXIS, (50, 1))])
    assert recirculation_fraction(v, AXIS) == pytest.approx(0.5)


def test_recirc_speed_floor_excludes_noise():
    flow = np.tile(AXIS, (80, 1))                      # clean forward flow
    noise = np.tile(-AXIS, (20, 1)) * 1e-4             # tiny reversed noise
    v = np.vstack([flow, noise])
    assert recirculation_fraction(v, AXIS, speed_floor=0.1) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# geometry helpers on a synthetic cylinder + bulge
# --------------------------------------------------------------------------- #
def _cylinder_wall(R=1.0, L=10.0, n_axial=40, n_theta=40):
    x = np.linspace(0.0, L, n_axial)
    th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    X, TH = np.meshgrid(x, th, indexing="ij")
    coords = np.column_stack([X.ravel(), (R * np.cos(TH)).ravel(), (R * np.sin(TH)).ravel()])
    # inward normals point toward the axis
    normals = np.column_stack([np.zeros(coords.shape[0]),
                               (-np.cos(TH)).ravel(), (-np.sin(TH)).ravel()])
    return coords, normals


def _bulged_wall(R=1.0, B=1.5, w=1.0, L=12.0, n_axial=60, n_theta=40):
    x = np.linspace(0.0, L, n_axial)
    th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    X, TH = np.meshgrid(x, th, indexing="ij")
    r = R + B * np.exp(-(((X - L / 2.0) / w) ** 2))
    return np.column_stack([X.ravel(), (r * np.cos(TH)).ravel(), (r * np.sin(TH)).ravel()])


def test_lumen_mask_inside_vs_outside():
    coords, normals = _cylinder_wall(R=1.0, L=10.0)
    query = np.array([[5.0, 0.0, 0.0],     # on the axis -> inside
                      [5.0, 0.5, 0.0],     # within the lumen -> inside
                      [5.0, 2.0, 0.0],     # outside the wall -> outside
                      [5.0, 0.0, 3.0]])    # outside the wall -> outside
    mask = lumen_inside_mask(query, coords, normals)
    assert mask.tolist() == [True, True, False, False]


def test_sac_axial_band_centers_on_bulge():
    coords = _bulged_wall(R=1.0, B=1.5, w=1.0, L=12.0)
    band = sac_axial_band(coords)
    assert band["axial_dim"] == 0                      # x has the largest span
    assert band["x_lo"] < 6.0 < band["x_hi"]           # brackets the mid-vessel bulge
    assert band["x_center"] == pytest.approx(6.0, abs=1.0)
    assert band["radius"] > 1.5                         # wider than the R=1.0 baseline


def test_sac_band_excludes_inlet_outlet_ends():
    coords = _bulged_wall(L=12.0)
    band = sac_axial_band(coords)
    # the band must sit in the interior, not at the inlet/outlet faces
    assert band["x_lo"] > 0.15 * 12.0
    assert band["x_hi"] < 0.85 * 12.0
