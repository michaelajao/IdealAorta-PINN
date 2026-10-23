"""Observed values and fitting indices, kept separate from clean CFD targets."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np

from operators import paired_gaussian_noise, voxel_quadrature


def array_hash(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def common_validation_mask(n, seed, fraction=0.1):
    """An exact-count location split, shared across times and methods."""
    if n < 3 or not 0 < fraction < 1:
        raise ValueError("at least three observations and a fractional split required")
    mask = np.zeros(n, dtype=bool)
    k = min(n - 1, max(1, int(round(n * fraction))))
    mask[np.random.default_rng(seed).permutation(n)[:k]] = True
    return mask


def matched_location_subsets(anchor_ids, distance_m, validation, threshold_m, seed):
    """Match both train and validation counts; random removal uses no field values."""
    ids = np.asarray(anchor_ids, dtype=np.int64)
    distance = np.asarray(distance_m, dtype=float)
    val = np.asarray(validation, dtype=bool)
    if ids.ndim != 1 or distance.shape != ids.shape or val.shape != ids.shape:
        raise ValueError("ids, distances and split must have the same vector shape")
    keep = distance >= threshold_m
    rng = np.random.default_rng(seed)
    random_keep = np.zeros(len(ids), bool)
    for group in (False, True):
        pool = np.flatnonzero(val == group)
        count = int(np.sum(keep & (val == group)))
        random_keep[rng.choice(pool, count, replace=False)] = True
    if not keep.any() or not (keep & val).any() or not (keep & ~val).any():
        raise ValueError("near-wall exclusion left an empty fitting/validation stratum")
    return {"all_points": np.arange(len(ids)),
            "exclude_near_wall": np.flatnonzero(keep),
            "matched_random_removal": np.flatnonzero(random_keep)}


@dataclass(frozen=True)
class Observations:
    times_ms: tuple
    support_m: np.ndarray       # (N,Q,3), Q=1 for points
    weights: np.ndarray         # (Q,), same physical observation operator for all fits
    clean: np.ndarray           # (2,N,3), observation oracle, never a scoring target
    values: np.ndarray          # (2,N,3), noisy fitting data
    validation: np.ndarray      # (N,), locations held out at BOTH times
    anchor_ids: np.ndarray
    metadata: dict

    @classmethod
    def make(cls, times_ms, support_m, weights, clean, validation, anchor_ids,
             sigma_m_s, noise_seed, correlation=0.0, metadata=None, realized_noise=None):
        support = np.array(support_m, dtype=float, copy=True)
        w = np.array(weights, dtype=float, copy=True)
        clean_copy = np.array(clean, dtype=float, copy=True)
        val = np.array(validation, dtype=bool, copy=True)
        ids = np.array(anchor_ids, dtype=np.int64, copy=True)
        n = len(ids)
        if support.ndim != 3 or support.shape[0] != n or support.shape[2] != 3:
            raise ValueError("support must be (N,Q,3)")
        if w.shape != (support.shape[1],) or np.any(w < 0) or not np.isclose(w.sum(), 1):
            raise ValueError("invalid averaging weights")
        if clean_copy.shape != (2, n, 3) or val.shape != (n,) or not val.any() or val.all():
            raise ValueError("invalid clean values or nonempty shared split required")
        if not np.isfinite(support).all() or not np.isfinite(w).all():
            raise ValueError("nonfinite observation geometry")
        if len(times_ms) != 2 or times_ms[0] >= times_ms[1]:
            raise ValueError("times must be two increasing milliseconds")
        noise = paired_gaussian_noise(clean_copy, np.asarray(times_ms) / 1000,
                                      sigma_m_s, noise_seed, correlation)
        values = noise.observed
        if realized_noise is not None:
            realized = np.asarray(realized_noise, dtype=float)
            if realized.shape != clean_copy.shape or not np.isfinite(realized).all():
                raise ValueError("paired realized noise must match clean observations")
            values = clean_copy + realized
        meta = dict(metadata or {})
        meta.update(noise.metadata)
        if realized_noise is not None:
            meta['noise_pairing'] = 'exact realization subset from the base point observations'
            dt = (times_ms[1]-times_ms[0])/1000
            meta['realized_acceleration_noise_rms_m_s2'] = np.sqrt(np.mean(((realized[1]-realized[0])/dt)**2,axis=0)).tolist()
        meta.update({"observation_sha256": array_hash(support, w, values, val, ids),
                     "fitting_policy": "shared hold-out retained; no all-observation refit for any method",
                     "n_fit_per_time": int((~val).sum()), "n_val_per_time": int(val.sum())})
        for a in (support, w, clean_copy, values, val, ids):
            a.flags.writeable = False
        return cls(tuple(times_ms), support, w, clean_copy, values, val, ids, meta)

    def subset(self, rows, label):
        rows = np.asarray(rows, dtype=np.int64)
        # Subset the EXISTING realization: removal arms cannot receive fresh noise.
        arrays = [a.copy() for a in (self.support_m[rows], self.weights, self.clean[:, rows],
                                     self.values[:, rows], self.validation[rows], self.anchor_ids[rows])]
        if not arrays[4].any() or arrays[4].all():
            raise ValueError("subset has empty fitting/validation stratum")
        meta = dict(self.metadata, removal_arm=label,
                    n_fit_per_time=int((~arrays[4]).sum()), n_val_per_time=int(arrays[4].sum()))
        meta["observation_sha256"] = array_hash(arrays[0], arrays[1], arrays[3], arrays[4], arrays[5])
        for a in arrays:
            a.flags.writeable = False
        return Observations(self.times_ms, *arrays, meta)


def torch_observe(net, support, weights, time_value):
    """Differentiable physical average; time_value is already network-scaled."""
    import torch
    n, q, _ = support.shape
    t = torch.full((n * q, 1), float(time_value), dtype=support.dtype, device=support.device)
    velocity = net(support.reshape(-1, 3), t)[:, :3].reshape(n, q, 3)
    return (velocity * weights[None, :, None]).sum(dim=1)
