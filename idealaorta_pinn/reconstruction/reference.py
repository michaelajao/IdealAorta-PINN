"""Training-free audits of the CFD fields: which equations and targets are identifiable.

    momentum_budget   do the exported fields satisfy the momentum balance the models are
                      asked to enforce? Residual of rho du/dt + rho (u.grad)u + grad p
                      + (2/3) rho grad k - div[(mu + mu_t)(grad u + grad u^T)] with the time
                      term, the k term or mu_t dropped, by region; plus the Newtonian wall
                      shear of the complete nodal velocity against the exported WSS
    pressure_oracle   pressure integrated from the complete CFD velocity on the interior
                      sub-cloud, per momentum form; a floor row integrates the CFD's own
                      grad p and must reproduce it (0.5-0.9 % in every case)
    wss_oracle        wall shear from the tangential velocity at h and 2h inside the wall,
                      by a Newtonian (first and second order) and a Spalding law-of-the-wall
                      estimate, from the complete field or from a subset of nodes

du/dt is a finite difference of snapshots on the same mesh (node identity is exact):
central and non-uniform when snapshots exist on both sides, otherwise one-sided. It is an
approximation, not the instantaneous derivative. All statistics weight nodes equally (the
exports carry no cell volumes); near-wall prism layers are reported separately.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

from ..data.full_export import SPLIT_TRAIN, load_block, load_full_snapshot, wall_normals_from_volume
from . import MU, RHO
from .numerics import DEFAULT_K, grad, lsq_operator, sparse_grad_matrix, stencil_condition

NU = MU / RHO
KAPPA, B_LOG = 0.41, 5.2                         # log-law constants for Spalding's law
WSS_H_MM = (0.1, 0.25, 0.5, 1.0, 1.5)
MOMENTUM_FORMS = ("unsteady+k+mu_t", "unsteady+mu_t (no k)", "steady+k+mu_t",
                  "steady+mu_t (no k)", "unsteady+k laminar", "steady laminar (no k)")


# --------------------------------------------------------------------------- helpers
def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(a ** 2, axis=-1)))) if a.ndim > 1 else float(np.sqrt(np.mean(a ** 2)))


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-30))


def _velocity(case: int, t_ms: int) -> np.ndarray:
    return load_block(case, "solid", t_ms, ["u", "v", "w"]).to_numpy(np.float64)


def time_derivative(case: int, t_ms: int, t_before: Optional[int], t_after: Optional[int]):
    """Finite-difference du/dt at ``t_ms`` from same-mesh snapshots, and a label.

    ``t_before`` must precede and ``t_after`` follow ``t_ms``; passing a snapshot on the
    wrong side would silently flip the sign of the derivative, so it is rejected.
    """
    if t_before is not None and t_before >= t_ms:
        raise ValueError(f"t_before={t_before} ms is not before t={t_ms} ms")
    if t_after is not None and t_after <= t_ms:
        raise ValueError(f"t_after={t_after} ms is not after t={t_ms} ms")
    U = _velocity(case, t_ms)
    if t_before and t_after:
        Um, Up = _velocity(case, t_before), _velocity(case, t_after)
        h1, h2 = (t_ms - t_before) / 1000, (t_after - t_ms) / 1000
        dudt = (-h2 / (h1 * (h1 + h2))) * Um + ((h2 - h1) / (h1 * h2)) * U + (h1 / (h2 * (h1 + h2))) * Up
        return dudt, f"central non-uniform ({t_before},{t_ms},{t_after}) ms"
    other = t_before or t_after
    if other:
        return (U - _velocity(case, other)) / ((t_ms - other) / 1000), f"one-sided ({other},{t_ms}) ms"
    return np.zeros_like(U), "none"


class _Fields:
    """CFD volume fields at one instant and the momentum-balance terms built from them."""

    def __init__(self, case: int, t_ms: int, t_before: Optional[int], t_after: Optional[int], k: int):
        v = load_block(case, "solid", t_ms)
        self.X = v[["x", "y", "z"]].to_numpy(np.float64)
        self.U = v[["u", "v", "w"]].to_numpy(np.float64)
        self.P = v["p"].to_numpy(np.float64)
        self.K = v["k"].to_numpy(np.float64)
        self.MUT = v["mu_t"].to_numpy(np.float64)
        self.is_wall = np.linalg.norm(self.U, axis=1) == 0
        dudt, self.dudt_note = time_derivative(case, t_ms, t_before, t_after)
        self.nb, self.C = lsq_operator(self.X, k)
        self.gU = grad(self.nb, self.C, self.U)             # [n, i, d] = d u_i / d x_d
        self.gP = grad(self.nb, self.C, self.P)
        self.S = self.gU + np.transpose(self.gU, (0, 2, 1))
        self.conv = RHO * np.einsum("nd,nid->ni", self.U, self.gU)
        self.unst = RHO * dudt
        self.kterm = (2.0 / 3.0) * RHO * grad(self.nb, self.C, self.K)
        self.dsig_t = self._div_sigma(MU + self.MUT)
        self.dsig_l = self._div_sigma(np.full(len(self.X), MU))
        # sac: interior nodes whose nearest wall node lies on the aneurysm zone
        self.walls = [b for b in (load_block(case, "aneurysm", t_ms), load_block(case, "wall", t_ms)) if b is not None]
        self.dwall, _ = cKDTree(self.X[self.is_wall]).query(self.X)
        self.sac = np.zeros(len(self.X), bool)
        if len(self.walls) == 2:
            allw = np.vstack([w[["x", "y", "z"]].to_numpy() for w in self.walls])
            lab = np.r_[np.ones(len(self.walls[0]), bool), np.zeros(len(self.walls[1]), bool)]
            self.sac = lab[cKDTree(allw).query(self.X)[1]]

    def _div_sigma(self, mu_eff: np.ndarray) -> np.ndarray:
        sig = (mu_eff[:, None, None] * self.S).reshape(len(self.X), 9)
        return np.einsum("nijj->ni", grad(self.nb, self.C, sig).reshape(len(self.X), 3, 3, 3))

    def residuals(self) -> Dict[str, np.ndarray]:
        u, c, g, k, dt, dl = self.unst, self.conv, self.gP, self.kterm, self.dsig_t, self.dsig_l
        return {"unsteady+k+mu_t": u + c + g + k - dt, "unsteady+mu_t (no k)": u + c + g - dt,
                "steady+k+mu_t": c + g + k - dt, "steady+mu_t (no k)": c + g - dt,
                "unsteady+k laminar": u + c + g + k - dl, "steady laminar (no k)": c + g - dl}

    def implied_grad_p(self) -> Dict[str, np.ndarray]:
        u, c, k, dt, dl = self.unst, self.conv, self.kterm, self.dsig_t, self.dsig_l
        return {"floor: grad of CFD p": self.gP,
                "unsteady+k+mu_t": -(u + c + k - dt), "unsteady+mu_t (no k)": -(u + c - dt),
                "steady+mu_t (no k)": -(c - dt), "steady+k+mu_t": -(c + k - dt),
                "steady laminar (no k)": -(c - dl)}


# --------------------------------------------------------------------------- momentum budget
def momentum_budget(case: int, t_ms: int, t_before: Optional[int] = None, t_after: Optional[int] = None,
                    k: int = DEFAULT_K, _fields: Optional[_Fields] = None) -> Dict:
    """Residual of each momentum form, relative to the convective term, by region."""
    F = _fields or _Fields(case, t_ms, t_before, t_after, k)
    interior = ~F.is_wall
    regions = {"bulk(>0.5mm)": interior & (F.dwall > 5e-4),
               "bulk(>1.5mm)": interior & (F.dwall > 1.5e-3),
               "bulk_sac": interior & (F.dwall > 5e-4) & F.sac,
               "bulk_outside_sac": interior & (F.dwall > 5e-4) & ~F.sac,
               "near_wall(<0.5mm)": interior & (F.dwall <= 5e-4)}
    terms = {"unsteady": F.unst, "grad_p": F.gP, "k_term": F.kterm,
             "div_sigma_turb": F.dsig_t, "div_sigma_lam": F.dsig_l}
    variants = F.residuals()
    budget = {}
    for rn, m in regions.items():
        ref = _rms(F.conv[m])
        conv_med = np.median(np.linalg.norm(F.conv[m], axis=1))
        div = np.einsum("nii->n", F.gU[m])
        gnorm = np.sqrt(np.sum(F.gU[m] ** 2, axis=(1, 2)))
        budget[rn] = {
            "n_nodes": int(m.sum()),
            "rms_conv_Pa_per_m": ref,
            "term_rms_over_conv": {tn: _rms(t[m]) / ref for tn, t in terms.items()},
            "residual_rms_over_conv": {vn: _rms(r[m]) / ref for vn, r in variants.items()},
            # robust: the RMS is dominated by a few degenerate stencils at the
            # prism-to-tetrahedra transition, so medians are the primary statistic
            "residual_median_over_conv_median": {
                vn: float(np.median(np.linalg.norm(r[m], axis=1)) / conv_med) for vn, r in variants.items()},
            "term_median_over_conv_median": {
                tn: float(np.median(np.linalg.norm(t[m], axis=1)) / conv_med) for tn, t in terms.items()},
            "div_u_over_gradu_median": float(np.median(np.abs(div) / np.maximum(gnorm, 1e-12))),
            "div_u_rms_over_gradu_rms": float(np.sqrt(np.mean(div ** 2)) / np.sqrt(np.mean(gnorm ** 2))),
        }
    return {"case": case, "t_ms": t_ms, "dudt": F.dudt_note, "k_neighbors": k, "weighting": "per node",
            "n_nodes": int(len(F.X)), "budget": budget, "wss_oracle": _wss_from_full_gradient(F)}


def _wss_from_full_gradient(F: _Fields) -> Dict:
    """Newtonian wall shear from the LSQ gradient of the complete nodal velocity."""
    wall_df = pd.concat(F.walls, ignore_index=True)
    wxyz = wall_df[["x", "y", "z"]].to_numpy(np.float64)
    _, first = np.unique(np.round(wxyz, 9), axis=0, return_index=True)
    wall_df = wall_df.iloc[np.sort(first)]
    wxyz = wall_df[["x", "y", "z"]].to_numpy(np.float64)
    wss_cfd = wall_df[["wss_x", "wss_y", "wss_z"]].to_numpy(np.float64)
    nrm = wall_normals_from_volume(wxyz, F.X[~F.is_wall])
    tr = np.einsum("nij,nj->ni", F.S[cKDTree(F.X).query(wxyz)[1]], nrm)
    wss_or = MU * (tr - np.sum(tr * nrm, 1, keepdims=True) * nrm)
    dfirst, _ = cKDTree(F.X[~F.is_wall]).query(wxyz)       # first off-wall node distance
    yplus = dfirst * np.sqrt(np.linalg.norm(wss_cfd, axis=1) / RHO) / NU
    is_an = np.zeros(len(wxyz), bool)
    if len(F.walls) == 2:
        is_an = cKDTree(F.walls[0][["x", "y", "z"]].to_numpy()).query(wxyz)[0] < 1e-9
    mt, mo = np.linalg.norm(wss_cfd, axis=1), np.linalg.norm(wss_or, axis=1)
    cond = stencil_condition(F.X, F.nb)
    return {"vec_rel_l2": _rel(wss_or - wss_cfd, wss_cfd), "mag_rel_l2": _rel(mo - mt, mt),
            "q99_ratio": float(np.quantile(mo, .99) / np.quantile(mt, .99)),
            "max_ratio": float(mo.max() / mt.max()),
            "aneurysm_mag_rel_l2": _rel(mo[is_an] - mt[is_an], mt[is_an]) if is_an.any() else None,
            "yplus_first_node_p50_p99_max": [float(np.quantile(yplus, q)) for q in (.5, .99)] + [float(yplus.max())],
            "first_node_dist_mm_p50": float(np.median(dfirst) * 1e3),
            "lsq_cond_p50_p99": [float(np.quantile(cond, .5)), float(np.quantile(cond, .99))]}


# --------------------------------------------------------------------------- pressure oracle
def pressure_oracle(case: int, t_ms: int, t_before: Optional[int] = None, t_after: Optional[int] = None,
                    forms: Optional[Sequence[str]] = None, min_wall_mm: float = 1.0, k: int = DEFAULT_K,
                    lsqr_iter: int = 6000, _fields: Optional[_Fields] = None) -> Dict:
    """Pressure integrated from the complete CFD velocity, per momentum form.

    The integration runs on the interior sub-cloud more than ``min_wall_mm`` from the wall
    with its own stencils (whole-domain integration is dominated by the unreliable
    prism-layer viscous terms). Gauge: mean over sub-cloud nodes with x > 0.130 m.
    """
    F = _fields or _Fields(case, t_ms, t_before, t_after, k)
    sub = (~F.is_wall) & (F.dwall > min_wall_mm / 1000)
    Xs = F.X[sub]
    nb2, C2 = lsq_operator(Xs, k)
    G = sparse_grad_matrix(nb2, C2)
    o_idx = np.flatnonzero(Xs[:, 0] > 0.130)
    i_idx = np.flatnonzero(Xs[:, 0] < 0.005)
    p_ref = F.P[sub] - F.P[sub][o_idx].mean()
    evalm = {"sub": np.ones(len(Xs), bool), "sub_sac": F.sac[sub], "sub_outside_sac": ~F.sac[sub]}
    gauge = sp.csr_matrix((np.full(len(o_idx), 1e3 / len(o_idx)), (np.zeros(len(o_idx), int), o_idx)),
                          shape=(1, len(Xs)))
    A = sp.vstack([G, gauge]).tocsr()
    out = {"dp_cfd_Pa": float(p_ref[i_idx].mean() - p_ref[o_idx].mean()), "n_unknowns": int(len(Xs)),
           "row_min_wall_mm": min_wall_mm, "dudt": F.dudt_note}
    for name, f in F.implied_grad_p().items():
        if forms and name not in forms and not name.startswith("floor"):
            continue
        sol = lsqr(A, np.r_[f[sub].reshape(-1), 0.0], atol=1e-10, btol=1e-10, iter_lim=lsqr_iter)
        p = sol[0] - sol[0][o_idx].mean()
        out[name] = {**{f"p_rel_l2_{en}": _rel(p[em] - p_ref[em], p_ref[em]) for en, em in evalm.items() if em.any()},
                     "p_rms_err_Pa": float(np.sqrt(np.mean((p - p_ref) ** 2))),
                     "dp_Pa": float(p[i_idx].mean() - p[o_idx].mean()), "lsqr_iters": int(sol[2])}
    return out


def audit(case: int, t_ms: int, t_before: Optional[int] = None, t_after: Optional[int] = None,
          with_pressure: bool = True, forms: Optional[Sequence[str]] = None, k: int = DEFAULT_K) -> Dict:
    """Momentum budget and (optionally) pressure oracle from one set of gradients."""
    F = _Fields(case, t_ms, t_before, t_after, k)
    out = momentum_budget(case, t_ms, k=k, _fields=F)
    if with_pressure:
        out["pressure_oracle"] = pressure_oracle(case, t_ms, forms=forms, k=k, _fields=F)
    return out


# --------------------------------------------------------------------------- WSS oracle
def spalding_yplus(up: np.ndarray) -> np.ndarray:
    """Spalding's law of the wall, y+ as a function of u+."""
    ku = KAPPA * up
    return up + np.exp(-KAPPA * B_LOG) * (np.exp(ku) - 1 - ku - ku ** 2 / 2 - ku ** 3 / 6)


def spalding_utau(U: np.ndarray, y: np.ndarray, iters: int = 80) -> np.ndarray:
    """Friction velocity from speed ``U`` at wall distance ``y`` (vectorized bisection)."""
    lo, hi = np.full_like(U, 1e-5), np.full_like(U, 5.0)
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        f = spalding_yplus(np.minimum(U / mid, 200.0)) - y * mid / NU     # decreasing in u_tau
        lo, hi = np.where(f > 0, mid, lo), np.where(f > 0, hi, mid)
    return np.sqrt(lo * hi)


def _wss_region_scores(Wp: np.ndarray, Wt: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict:
    out = {}
    for tag, m in masks.items():
        if not m.any():
            continue
        mp, mt = np.linalg.norm(Wp[m], axis=1), np.linalg.norm(Wt[m], axis=1)
        out[tag] = {"vec_rel_l2": _rel(Wp[m] - Wt[m], Wt[m]), "mag_rel_l2": _rel(mp - mt, mt),
                    "pearson_mag": float(np.corrcoef(mp, mt)[0, 1]),
                    "q99_ratio": float(np.quantile(mp, .99) / np.quantile(mt, .99)),
                    "mean_ratio": float(mp.mean() / mt.mean())}
    return out


def wss_oracle(case: int, t_ms: int, sources: Sequence[str] = ("full",), frac: float = 0.05,
               seed: int = 0) -> Dict:
    """Wall-shear estimators from velocity sampled at h and 2h along the inward normal.

    ``sources``: ``full`` (every CFD volume node, an oracle), ``train`` (the even 2 mm
    checkerboard cubes of the earlier velocity-only runs) or ``frac`` (a uniform random
    fraction of nodes). Reported per h; picking the best h is an oracle choice.
    """
    s = load_full_snapshot(case, t_ms)
    X, T, W, n, Wt = s["vol_xyz"], s["vol_uvw"], s["wall_xyz"], s["wall_normals"], s["wall_wss"]
    masks = {"wall": np.ones(len(W), bool), "aneurysm": s["wall_aneurysm"]}
    tang = lambda U: U - np.sum(U * n, 1, keepdims=True) * n  # noqa: E731
    rows = []
    for src in sources:
        keep = {"full": np.ones(len(X), bool), "train": s["vol_split"] == SPLIT_TRAIN,
                "frac": np.random.default_rng(seed).random(len(X)) < frac}[src]
        f = LinearNDInterpolator(np.vstack([X[keep], W]), np.vstack([T[keep], np.zeros((len(W), 3))]))
        for h_mm in WSS_H_MM:
            h = h_mm / 1000
            u1, u2 = tang(f(W + h * n)), tang(f(W + 2 * h * n))
            ok = np.isfinite(u1).all(1) & np.isfinite(u2).all(1)
            U1 = np.linalg.norm(u1, axis=1)
            ut = spalding_utau(np.where(ok, U1, 0.0), np.full(len(W), h))
            estimates = {"newton1": MU * u1 / h, "newton2": MU * (4 * u1 - u2) / (2 * h),
                         "spalding": RHO * ut[:, None] ** 2 * u1 / np.maximum(U1, 1e-12)[:, None]}
            for name, E in estimates.items():
                E = np.where(ok[:, None], E, 0.0)
                rows.append({"source": src, "n_source": int(keep.sum()), "h_mm": h_mm, "estimator": name,
                             "frac_undefined": float(1 - ok.mean()), **_wss_region_scores(E, Wt, masks)})
    return {"case": case, "t_ms": t_ms, "n_wall": int(len(W)), "rows": rows}
