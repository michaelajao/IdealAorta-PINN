"""Isolated kinematic measurement operators; no historical study imports or writes.

Coordinates/voxel widths: metres. Time: seconds. Velocity/noise: metres/second.
These are controlled synthetic operators, not a phase-contrast MRI simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np


def _xyz(value, name):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 3 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite (N, 3)")
    return out


def voxel_quadrature(centres, width_m, order=3):
    """Tensor Gauss-Legendre nodes and normalized weights for full box voxels.

    A scalar width gives isotropic voxels; a (3,) width allows anisotropy. This
    integrates physical volume, not the nonuniform density of CFD mesh nodes.
    """
    centres = _xyz(centres, "centres")
    width = np.broadcast_to(np.asarray(width_m, dtype=float), (3,))
    if not np.isfinite(width).all() or np.any(width <= 0):
        raise ValueError("voxel widths must be finite and positive")
    if not isinstance(order, int) or order < 1:
        raise ValueError("quadrature order must be a positive integer")
    xi, wi = np.polynomial.legendre.leggauss(order)
    ix = np.asarray(list(itertools.product(range(order), repeat=3)))
    offset = xi[ix] * width / 2
    weights = wi[ix].prod(axis=1) / 8
    return centres[:, None, :] + offset[None, :, :], weights


def voxel_average(evaluate_velocity, centres, width_m, times_s, order=3,
                  box_is_inside=None, chunk_voxels=256):
    """Evaluate a frozen continuous velocity oracle over box voxels.

    ``evaluate_velocity(points_m, time_s)`` returns an (N, 3) array. The optional
    geometry-only ``box_is_inside(centres, width_m)`` must certify that each full
    box lies inside the lumen; checking only quadrature points is insufficient.
    Boundary-crossing boxes are rejected, never silently renormalized. If no
    geometry callback is provided, full boxes are assumed (analytic tests only).
    The quadrature operation must also be applied to each fitted model in its
    observation loss; averages must not be misrepresented as point samples.
    """
    centres = _xyz(centres, "centres")
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or len(times) == 0:
        raise ValueError("times must be a nonempty finite vector")
    if not isinstance(chunk_voxels, int) or chunk_voxels < 1:
        raise ValueError("chunk_voxels must be a positive integer")
    if box_is_inside is not None:
        valid = np.asarray(box_is_inside(centres.copy(), width_m), dtype=bool)
        if valid.shape != (len(centres),) or not valid.all():
            raise ValueError("all full voxel boxes must be certified inside the lumen")
    result = np.empty((len(times), len(centres), 3), dtype=float)
    for start in range(0, len(centres), chunk_voxels):
        points, weights = voxel_quadrature(centres[start:start + chunk_voxels], width_m, order)
        for j, t in enumerate(times):
            val = _xyz(evaluate_velocity(points.reshape(-1, 3), float(t)), "oracle velocity")
            if len(val) != points.shape[0] * points.shape[1]:
                raise ValueError("oracle returned the wrong number of velocities")
            result[j, start:start + len(points)] = np.einsum("nqc,q->nc", val.reshape(points.shape), weights)
    return result


@dataclass(frozen=True)
class NoiseResult:
    observed: np.ndarray
    noise: np.ndarray
    metadata: dict


def paired_gaussian_noise(clean_pair, times_s, sigma_m_s, seed, correlation=0.0):
    """Add reproducible componentwise Gaussian noise to two observation frames.

    The component standard deviation is supplied in m/s, not scaled by local
    velocity; low-speed diastolic values therefore do not receive weaker noise.
    Noise seed is independent of the geometry/grid and network-initialization
    seeds. Cross-time correlation is explicit; spatial/component independence
    is an experimental assumption. The clean input is never modified.
    """
    clean = np.asarray(clean_pair, dtype=float)
    times = np.asarray(times_s, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma_m_s, dtype=float), (3,))
    if clean.ndim != 3 or clean.shape[0] != 2 or clean.shape[2] != 3 or not np.isfinite(clean).all():
        raise ValueError("clean_pair must be finite (2, N, 3)")
    if times.shape != (2,) or not np.isfinite(times).all() or times[0] == times[1]:
        raise ValueError("two distinct finite times are required")
    if not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise ValueError("noise standard deviations must be finite and nonnegative")
    if not np.isfinite(correlation) or not -1 <= correlation <= 1:
        raise ValueError("correlation must lie in [-1, 1]")
    rng = np.random.default_rng(seed)
    z0, zi = rng.standard_normal(clean.shape)
    z1 = correlation * z0 + np.sqrt(1 - correlation ** 2) * zi
    noise = np.stack((z0, z1)) * sigma
    dt = float(times[1] - times[0])
    difference_noise = (noise[1] - noise[0]) / dt
    metadata = {
        "model": "additive Gaussian; spatially and componentwise independent",
        "seed": int(seed), "generator": "NumPy default_rng/PCG64",
        "sigma_m_s": sigma.tolist(), "temporal_correlation": float(correlation),
        "delta_t_s": dt,
        "expected_acceleration_noise_std_m_s2": (np.sqrt(2 * (1 - correlation)) * sigma / abs(dt)).tolist(),
        "realized_acceleration_noise_rms_m_s2": np.sqrt(np.mean(difference_noise ** 2, axis=0)).tolist(),
    }
    return NoiseResult(clean.copy() + noise, noise, metadata)


def divergence_free_gaussian_kernel(query, centres, length_scale_m):
    """Global 3-D matrix kernel ell^2 (Hessian - Laplacian I) exp(-r^2/2ell^2).

    Each column is analytically solenoidal. The returned shape is (N, M, 3, 3).
    No nearest-neighbour switching or partition blending is performed.
    """
    query, centres = _xyz(query, "query"), _xyz(centres, "centres")
    ell = float(length_scale_m)
    if not np.isfinite(ell) or ell <= 0:
        raise ValueError("length_scale_m must be finite and positive")
    r = (query[:, None, :] - centres[None, :, :]) / ell
    r2 = np.sum(r * r, axis=-1)
    outer = r[..., :, None] * r[..., None, :]
    return np.exp(-r2 / 2)[..., None, None] * (outer + (2 - r2)[..., None, None] * np.eye(3))


@dataclass
class GlobalDivergenceFreeRBF:
    """Small global prototype, not yet a validated aneurysm-scale comparator.

    Reduced centres control dense matrix size. Hyperparameters/centres must be
    selected with geometry and held-out observations only. Boundary conditions
    must be included explicitly by a future study adapter. Ridge regularization
    changes the coefficients but preserves the analytic zero-divergence basis.
    """
    centres: np.ndarray
    coefficients: np.ndarray
    length_scale_m: float

    @classmethod
    def fit_observations(cls, support_points, weights, observed_velocity,
                         basis_centres, length_scale_m, ridge=1e-8,
                         max_matrix_elements=5_000_000):
        """Fit point or voxel-average data with the same forward operator.

        support_points: (N, Q, 3); weights: (Q,) summing to one. Point data use
        Q=1 and weight=1. Full voxel averages use ``voxel_quadrature``. This dense
        prototype raises before creating an oversized design matrix.
        """
        points = np.asarray(support_points, dtype=float)
        w = np.asarray(weights, dtype=float)
        observed = _xyz(observed_velocity, "observed_velocity")
        centres = _xyz(basis_centres, "basis_centres")
        if points.ndim != 3 or points.shape[2] != 3 or points.shape[0] != len(observed) or not np.isfinite(points).all():
            raise ValueError("support_points must be finite (N, Q, 3)")
        if w.shape != (points.shape[1],) or not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1):
            raise ValueError("nonnegative quadrature weights must sum to one")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        if len(observed) == 0 or len(centres) == 0:
            raise ValueError("at least one observation and basis centre are required")
        matrix_elements = 9 * len(observed) * len(centres)
        if ridge:
            matrix_elements += 9 * len(centres) ** 2
        if matrix_elements > max_matrix_elements:
            raise ValueError("dense prototype matrix exceeds the configured size limit")
        averaged = np.zeros((len(observed), len(centres), 3, 3))
        for q, weight in enumerate(w):
            averaged += weight * divergence_free_gaussian_kernel(points[:, q], centres, length_scale_m)
        design = averaged.transpose(0, 2, 1, 3).reshape(3 * len(observed), 3 * len(centres))
        target = observed.reshape(-1)
        if ridge:
            design = np.vstack((design, np.sqrt(ridge) * np.eye(3 * len(centres))))
            target = np.concatenate((target, np.zeros(3 * len(centres))))
        coef = np.linalg.lstsq(design, target, rcond=None)[0].reshape(-1, 3)
        return cls(centres.copy(), coef, float(length_scale_m))

    def predict(self, points, chunk_size=256):
        points = _xyz(points, "points")
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        out = np.empty_like(points)
        for start in range(0, len(points), chunk_size):
            kernel = divergence_free_gaussian_kernel(points[start:start + chunk_size], self.centres, self.length_scale_m)
            out[start:start + chunk_size] = np.einsum("nmij,mj->ni", kernel, self.coefficients)
        return out
