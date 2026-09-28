"""Least-squares node gradients on the unstructured CFD point cloud.

Every derivative of a *nodal* field in this package (the CFD fields themselves, or a
reconstruction sampled at the CFD nodes) goes through these operators, so references,
baselines and neural fields are differentiated identically.

The stencil is the ``k`` nearest neighbours with inverse-square-distance weights. With
``k = 40`` the median divergence of the CFD velocity is about 1 % of |grad u| in the bulk;
``k = 20`` leaves degenerate one-sided stencils at the prism-to-tetrahedra transition
that dominate RMS statistics (reference audit, revision plan 2026-09-25).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

DEFAULT_K = 40


def lsq_operator(X: np.ndarray, k: int = DEFAULT_K):
    """Per-node gradient weights: ``grad f_i = sum_j C[i, :, j] (f[nb[i, j]] - f_i)``.

    Returns:
        ``(nb, C)`` with neighbour indices ``(N, k)`` and weights ``(N, 3, k)``.
    """
    _, nb = cKDTree(X).query(X, k=k + 1)
    nb = nb[:, 1:]
    A = X[nb] - X[:, None, :]                                    # (N, k, 3)
    w = 1.0 / np.maximum(np.einsum("nkd,nkd->nk", A, A), 1e-18)
    AtW = np.transpose(A * w[..., None], (0, 2, 1))              # (N, 3, k)
    return nb, np.linalg.solve(AtW @ A, AtW)


def stencil_condition(X: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """Condition number of each node's weighted normal matrix (stencil quality check)."""
    A = X[nb] - X[:, None, :]
    w = 1.0 / np.maximum(np.einsum("nkd,nkd->nk", A, A), 1e-18)
    return np.linalg.cond(np.transpose(A * w[..., None], (0, 2, 1)) @ A)


def grad(nb: np.ndarray, C: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Gradient of nodal field(s): ``(N,) -> (N, 3)`` or ``(N, m) -> (N, m, 3)``."""
    if f.ndim == 1:
        return np.einsum("ndk,nk->nd", C, f[nb] - f[:, None])
    return np.einsum("ndk,nkm->nmd", C, f[nb] - f[:, None, :])


def lsq_gradient(X: np.ndarray, f: np.ndarray, k: int = DEFAULT_K) -> np.ndarray:
    """Gradient of one scalar nodal field (builds the operator; use grad() to reuse it)."""
    nb, C = lsq_operator(X, k)
    return grad(nb, C, f)


def sparse_grad_matrix(nb: np.ndarray, C: np.ndarray) -> sp.csr_matrix:
    """``G`` of shape ``(3N, N)`` with ``(G f)[3 i + d] = d f / d x_d`` at node ``i``."""
    N, k = nb.shape
    rows = np.repeat(np.arange(3 * N), k + 1)
    cols = np.repeat(np.concatenate([nb, np.arange(N)[:, None]], 1)[:, None, :], 3, 1).reshape(-1)
    vals = np.concatenate([C, -C.sum(2, keepdims=True)], 2).reshape(-1)
    return sp.csr_matrix((vals, (rows, cols)), shape=(3 * N, N))
