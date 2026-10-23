import unittest
import numpy as np
from audit_external_axes import rank_axis_maps


class AxisCalibrationTests(unittest.TestCase):
    def test_recovers_signed_component_permutation_from_solenoidal_affine_ensemble(self):
        # Every matrix is the exact gradient of a different trace-free affine
        # field. No spatial finite differences are used to manufacture truth.
        rng=np.random.default_rng(244);physical=rng.normal(size=(64,3,3))
        trace=np.trace(physical,axis1=1,axis2=2)
        physical[:,2,2]-=trace
        mapping=np.array([[0,0,-1],[1,0,0],[0,-1,0]])
        stored=physical@mapping
        ranking=rank_axis_maps(stored)
        expected=-mapping  # Global sign convention fixes the first mapped sign positive.
        np.testing.assert_array_equal(ranking[0]['component_to_grid_axes'],expected)
        self.assertLess(ranking[0]['divergence_over_gradient_Frobenius'],1e-14)
        self.assertGreater(ranking[1]['divergence_over_gradient_Frobenius'],.1)

    def test_axis_mapping_is_not_inferred_from_zero_or_nonfinite_gradients(self):
        for invalid in (np.zeros((10,3,3)),np.full((10,3,3),np.nan),np.zeros((0,3,3))):
            with self.assertRaises(ValueError):rank_axis_maps(invalid)
