"""Analytical verification only; these tests do not establish CFD performance."""
import unittest
import numpy as np
from operators import (voxel_quadrature, voxel_average, paired_gaussian_noise,
                       GlobalDivergenceFreeRBF)


class MeasurementTests(unittest.TestCase):
    def test_physical_box_averages(self):
        centres = np.array([[0.2, 0.3, 0.4], [-0.2, 0.1, 0.8]])
        widths = np.array([0.03, 0.05, 0.07])
        def polynomial(p, t):
            x, y, z = p.T
            return np.stack((x*x + y*z + t, y*y + x*z - t, z*z + x*y), axis=1)
        result = voxel_average(polynomial, centres, widths, [0, 0.1], order=2, chunk_voxels=1)
        for i, t in enumerate([0, 0.1]):
            np.testing.assert_allclose(result[i], polynomial(centres, t) + widths ** 2 / 12, atol=1e-14)

    def test_affine_field_and_geometry_rejection(self):
        centres = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        affine = lambda x, t: 2*x + np.array([t, 1, 2])
        value = voxel_average(affine, centres, 0.02, [0.2], order=3)
        np.testing.assert_allclose(value[0], affine(centres, 0.2), atol=1e-14)
        with self.assertRaises(ValueError):
            voxel_average(affine, centres, 0.02, [0.2], box_is_inside=lambda c, w: np.array([True, False]))

    def test_noise_identity_reproducibility_and_immutability(self):
        clean = np.arange(60, dtype=float).reshape(2, 10, 3)
        original = clean.copy()
        zero = paired_gaussian_noise(clean, [1.775, 1.780], 0, 123)
        np.testing.assert_array_equal(zero.observed, clean)
        first = paired_gaussian_noise(clean, [1.775, 1.780], 0.02, 123)
        repeat = paired_gaussian_noise(clean, [1.775, 1.780], 0.02, 123)
        np.testing.assert_array_equal(first.observed, repeat.observed)
        np.testing.assert_array_equal(clean, original)
        first.observed[0, 0, 0] = -99
        np.testing.assert_array_equal(clean, original)

    def test_temporal_noise_amplification(self):
        clean = np.zeros((2, 100000, 3))
        sigma, dt = 0.02, 0.005
        for rho in [0, 0.75, 1]:
            result = paired_gaussian_noise(clean, [0, dt], sigma, 987, rho)
            expected = np.sqrt(2 * (1-rho)) * sigma / dt
            np.testing.assert_allclose(result.metadata['expected_acceleration_noise_std_m_s2'], expected)
            np.testing.assert_allclose(result.metadata['realized_acceleration_noise_rms_m_s2'], expected, rtol=0.015, atol=1e-12)
        anticorrelated = paired_gaussian_noise(clean[:, :20], [0, dt], sigma, 9, -1)
        np.testing.assert_array_equal(anticorrelated.noise[1], -anticorrelated.noise[0])

    def test_invalid_inputs(self):
        clean = np.zeros((2, 4, 3))
        with self.assertRaises(ValueError):
            paired_gaussian_noise(clean, [0, 0], .02, 0)
        with self.assertRaises(ValueError):
            paired_gaussian_noise(clean, [0, 1], -.02, 0)
        with self.assertRaises(ValueError):
            voxel_quadrature(np.zeros((1, 3)), 0)


class SolenoidalTests(unittest.TestCase):
    def test_divergence_with_independent_finite_differences(self):
        rng = np.random.default_rng(1234)
        model = GlobalDivergenceFreeRBF(rng.uniform(-1, 1, (7, 3)), rng.normal(size=(7, 3)), .7)
        query = rng.uniform(-1, 1, (60, 3))
        def divergence(step):
            result = np.zeros(len(query))
            for axis in range(3):
                offset = np.eye(3)[axis] * step
                result += (model.predict(query + offset)[:, axis] - model.predict(query - offset)[:, axis]) / (2 * step)
            return result
        coarse = np.linalg.norm(divergence(1e-3))
        fine = np.linalg.norm(divergence(1e-4))
        self.assertLess(fine, coarse / 50)
        self.assertLess(fine / np.sqrt(len(query)), 2e-7)
        np.testing.assert_allclose(model.predict(query, 1), model.predict(query, 256), atol=1e-14)

    def test_recover_global_field_from_voxel_averages(self):
        rng = np.random.default_rng(22)
        centres = rng.uniform(-.4, .4, (4, 3))
        truth = GlobalDivergenceFreeRBF(centres, rng.normal(size=(4, 3)), .55)
        measurement_centres = rng.uniform(-.8, .8, (35, 3))
        points, weights = voxel_quadrature(measurement_centres, .25, order=3)
        observed = np.einsum('nqc,q->nc', truth.predict(points.reshape(-1, 3)).reshape(points.shape), weights)
        fit = GlobalDivergenceFreeRBF.fit_observations(points, weights, observed, centres, .55, ridge=0)
        query = rng.uniform(-.7, .7, (25, 3))
        np.testing.assert_allclose(fit.predict(query), truth.predict(query), atol=2e-12)
        point_fit = GlobalDivergenceFreeRBF.fit_observations(measurement_centres[:, None], [1.], observed, centres, .55, ridge=0)
        # Deliberately treating averages as point values produces measurable bias.
        self.assertGreater(np.linalg.norm(point_fit.predict(query) - truth.predict(query)), 1e-3)

    def test_dense_memory_guard(self):
        with self.assertRaises(ValueError):
            GlobalDivergenceFreeRBF.fit_observations(np.zeros((5, 1, 3)), [1], np.zeros((5, 3)), np.zeros((5, 3)), 1, max_matrix_elements=1)

    def test_augmented_ridge_memory_guard(self):
        # The observation matrix alone fits (90 entries), but the ridge block
        # would add 900 entries; guard the augmented matrix before allocation.
        with self.assertRaises(ValueError):
            GlobalDivergenceFreeRBF.fit_observations(np.zeros((1, 1, 3)), [1], np.zeros((1, 3)), np.zeros((10, 3)), 1, ridge=.01, max_matrix_elements=100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
