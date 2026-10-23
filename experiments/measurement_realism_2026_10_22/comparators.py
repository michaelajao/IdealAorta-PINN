"""Bounded global scalar/vector RBF comparators with operator-aware fitting."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from operators import divergence_free_gaussian_kernel


def geometry_centres(points, count, seed=17, candidate_limit=12000):
    """Farthest-point coverage on a geometry-only, reproducible candidate pool."""
    x = np.asarray(points, float)
    rng = np.random.default_rng(seed)
    pool = x[rng.choice(len(x), min(candidate_limit, len(x)), replace=False)]
    count = min(int(count), len(pool))
    if count < 1:
        raise ValueError("at least one centre required")
    chosen = np.empty(count, int)
    chosen[0] = int(np.argmin(np.sum((pool - pool.mean(0)) ** 2, axis=1)))
    distance = np.full(len(pool), np.inf)
    for i in range(count):
        distance = np.minimum(distance, np.sum((pool - pool[chosen[i]]) ** 2, axis=1))
        if i + 1 < count:
            chosen[i + 1] = int(np.argmax(distance))
    return pool[chosen].copy()


@dataclass
class ReducedRBF:
    centres: np.ndarray
    ell: float
    vector: bool
    origin: np.ndarray
    length: float
    coefficients: np.ndarray | None = None

    @property
    def n_columns(self):
        return 3 * len(self.centres) + 11 if self.vector else len(self.centres) + 4

    def design(self, support, weights, max_bytes=1_500_000_000):
        p = np.asarray(support, float)
        w = np.asarray(weights, float)
        n = len(p)
        rows = n * 3 if self.vector else n
        if rows * self.n_columns * 8 > max_bytes:
            raise MemoryError("RBF design exceeds the explicit byte budget")
        if self.vector:
            k = np.zeros((n, len(self.centres), 3, 3))
            for q, weight in enumerate(w):
                for start in range(0, n, 128):
                    k[start:start + 128] += weight * divergence_free_gaussian_kernel(
                        p[start:start + 128, q], self.centres, self.ell)
            result = np.zeros((3 * n, self.n_columns))
            result[:, :3 * len(self.centres)] = k.transpose(0, 2, 1, 3).reshape(3 * n, -1)
            # Constants and a trace-free affine basis are also exactly solenoidal.
            a = np.zeros((n, 3, 11))
            a[:, :, :3] = np.eye(3)
            mean = (np.einsum("nqc,q->nc", p, w) - self.origin) / self.length
            col = 3
            for i in range(3):
                for j in range(3):
                    if i != j:
                        a[:, i, col] = mean[:, j]
                        col += 1
            a[:, 0, 9], a[:, 2, 9] = mean[:, 0], -mean[:, 2]
            a[:, 1, 10], a[:, 2, 10] = mean[:, 1], -mean[:, 2]
            result[:, 3 * len(self.centres):] = a.reshape(3 * n, 11)
            return result
        result = np.zeros((n, self.n_columns))
        for q, weight in enumerate(w):
            for start in range(0, n, 256):
                d = (p[start:start + 256, q, None] - self.centres[None]) / self.ell
                result[start:start + 256, :len(self.centres)] += weight * np.exp(-.5 * np.sum(d*d, axis=2))
        result[:, -4] = 1
        result[:, -3:] = (np.einsum("nqc,q->nc", p, w) - self.origin) / self.length
        return result

    def predict(self, points, chunk=512):
        return self.observe(np.asarray(points)[:, None], np.ones(1), chunk)

    def observe(self, support, weights, chunk=512):
        if self.coefficients is None:
            raise ValueError("model has not been fitted")
        out = []
        for start in range(0, len(support), chunk):
            value = self.design(support[start:start + chunk], weights) @ self.coefficients
            out.append(value.reshape(-1, 3))
        return np.concatenate(out)


def fit_rbf_pair(observations, centres, boundary_xyz, boundary_values, inlet_xyz,
                 inlet_values, vector, length_scales, ridges, max_iter=1000, tol=1e-7):
    """Select on the SAME noisy observation hold-out; retain it for every final fit.

    Each snapshot is fitted independently. The spatial kernel is global; there
    is no discontinuous nearest-neighbour switching. All candidate failures and
    LSQR stop codes are returned, rather than silently discarded.
    """
    from scipy.sparse.linalg import lsqr
    obs = observations
    fit, val = ~obs.validation, obs.validation
    origin = centres.mean(0)
    length = max(float(np.ptp(centres, axis=0).max()), 1e-8)
    scale = max(float(np.quantile(np.linalg.norm(obs.values[:, fit], axis=2), .995)), 1e-6)
    trials, winner, best = [], None, np.inf
    for ell in length_scales:
        template = ReducedRBF(centres, float(ell), vector, origin, length)
        pieces = [template.design(obs.support_m[fit], obs.weights),
                  template.design(boundary_xyz[:, None], np.ones(1)),
                  template.design(inlet_xyz[:, None], np.ones(1))]
        # Match the relative data:wall:inlet loss weights of the neural arm.
        factors = [1/np.sqrt(fit.sum()), 1/np.sqrt(len(boundary_xyz)),
                   np.sqrt(2/len(inlet_xyz))]
        matrix = np.concatenate([a*f for a, f in zip(pieces, factors)])
        del pieces
        for ridge in ridges:
            started = time.monotonic()
            models, stops, residuals, validation_error = [], [], [], []
            for frame in range(2):
                targets = [obs.values[frame, fit], boundary_values[frame], inlet_values[frame]]
                targets = [t.reshape(-1) if vector else t for t in targets]
                y = np.concatenate([v*f/scale for v, f in zip(targets, factors)])
                if vector:
                    solved = [lsqr(matrix, y, damp=np.sqrt(ridge), atol=tol, btol=tol, iter_lim=max_iter)]
                    coefficient = solved[0][0]
                else:
                    solved = [lsqr(matrix, y[:, c], damp=np.sqrt(ridge), atol=tol, btol=tol,
                                   iter_lim=max_iter) for c in range(3)]
                    coefficient = np.stack([s[0] for s in solved], axis=1)
                model = ReducedRBF(centres, float(ell), vector, origin, length, coefficient * scale)
                pred = model.observe(obs.support_m[val], obs.weights)
                validation_error.append(float(np.sum((pred-obs.values[frame, val])**2)))
                models.append(model)
                stops.append([{"istop":int(s[1]), "iterations":int(s[2]), "normr":float(s[3]),
                               "condition_estimate":float(s[6])} for s in solved])
            error = float(np.sqrt(sum(validation_error)/np.sum(obs.values[:, val]**2)))
            converged = all(s["istop"] in (0,1,2,4,5) for frame in stops for s in frame)
            trials.append({"ell_m":float(ell), "ridge":float(ridge), "val_observed_rel_l2":error,
                           "lsqr":stops, "converged":converged,
                           "seconds":time.monotonic()-started})
            # Unconverged fits remain in the audit but cannot win selection.
            if converged and np.isfinite(error) and error < best:
                best, winner = error, models
    if winner is None:
        raise RuntimeError(f"No converged RBF candidate: {trials}")
    return winner, {"trials":trials, "best_val_observed_rel_l2":best,
                    "basis_centres":len(centres), "global_solenoidal_basis":bool(vector),
                    "selection_uses_hidden_targets":False, "fit_policy":obs.metadata["fitting_policy"]}
