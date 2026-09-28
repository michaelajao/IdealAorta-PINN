"""Velocity -> pressure and wall shear, identically for every method, plus the scores.

Any reconstructed velocity field sampled at the CFD nodes (interior nodes; the wall
nodes carry u = 0) is turned into

    pressure   least-squares integration of the momentum balance on the interior
               sub-cloud more than ``min_wall_mm`` from the wall (node stencils are
               unreliable in the prism layers); gauge = mean over sub-cloud nodes with
               x > 0.130 m
    wall shear mu (4 u_t(h) - u_t(2h)) / (2h) along the inward normal (Newtonian, second
               order) at a pre-registered distance h, from a linear interpolant

The momentum form is the one the CFD fields satisfy (``reference.momentum_budget``):
rho du/dt + rho (u.grad)u + grad p - div[(mu + mu_t)(grad u + grad u^T)] = 0, with no
(2/3) rho k term. Integrating the CFD's own grad p reproduces its pressure to 0.5-0.9 %.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse.linalg import lsqr

from . import MU, RHO
from .numerics import DEFAULT_K, grad, lsq_operator, sparse_grad_matrix

OUTLET_GAUGE_X = 0.130      # m: sub-cloud nodes beyond this define the pressure gauge
INLET_REGION_X = 0.005      # m: sub-cloud nodes before this define the inlet pressure

# The four information settings of the pressure step: time term and closure on/off.
PRESSURE_FORMS = ("unsteady+mut", "steady+mut", "unsteady+lam", "steady+lam")


class PressureIntegrator:
    """Stencils for one geometry (interior + wall nodes) and the sub-cloud integration."""

    def __init__(self, X_int: np.ndarray, wall_xyz: np.ndarray, dwall: np.ndarray,
                 min_wall_mm: float = 1.0, k: int = DEFAULT_K):
        self.n_int = len(X_int)
        self.Xall = np.vstack([X_int, wall_xyz])
        self.nb, self.C = lsq_operator(self.Xall, k)
        self.sub = dwall > min_wall_mm / 1000.0
        Xs = X_int[self.sub]
        nb2, C2 = lsq_operator(Xs, k)
        G = sparse_grad_matrix(nb2, C2)
        self.o_idx = np.flatnonzero(Xs[:, 0] > OUTLET_GAUGE_X)
        self.i_idx = np.flatnonzero(Xs[:, 0] < INLET_REGION_X)
        gauge = sp.csr_matrix((np.full(len(self.o_idx), 1e3 / len(self.o_idx)),
                               (np.zeros(len(self.o_idx), int), self.o_idx)), shape=(1, len(Xs)))
        self.A = sp.vstack([G, gauge]).tocsr()

    def momentum_rhs(self, U: np.ndarray, dUdt: Optional[np.ndarray],
                     mut: Optional[np.ndarray]) -> np.ndarray:
        """grad p implied by interior velocity ``U`` (N_int, 3); wall nodes get u = 0.

        ``dUdt`` None drops the time term (steady form); ``mut`` None drops the eddy
        viscosity (laminar form).
        """
        Ua = np.vstack([U, np.zeros((len(self.Xall) - self.n_int, 3))])
        gU = grad(self.nb, self.C, Ua)
        S = gU + np.transpose(gU, (0, 2, 1))
        mu_eff = np.full(len(Ua), MU)
        if mut is not None:
            mu_eff[: self.n_int] += mut
        sig = (mu_eff[:, None, None] * S).reshape(len(Ua), 9)
        dsig = np.einsum("nijj->ni", grad(self.nb, self.C, sig).reshape(len(Ua), 3, 3, 3))
        f = -RHO * np.einsum("nd,nid->ni", Ua, gU) + dsig
        if dUdt is not None:
            f[: self.n_int] -= RHO * dUdt
        return f[: self.n_int]

    def integrate(self, f_int: np.ndarray, iters: int = 6000) -> np.ndarray:
        """Gauge pressure on the sub-cloud from a grad-p field on the interior nodes."""
        rhs = np.r_[f_int[self.sub].reshape(-1), 0.0]
        p = lsqr(self.A, rhs, atol=1e-10, btol=1e-10, iter_lim=iters)[0]
        return p - p[self.o_idx].mean()

    def pressure(self, U: np.ndarray, dUdt: Optional[np.ndarray], mut: Optional[np.ndarray],
                 form: str) -> np.ndarray:
        """Sub-cloud gauge pressure for one of :data:`PRESSURE_FORMS`."""
        unsteady, closure = form.split("+")
        return self.integrate(self.momentum_rhs(U, dUdt if unsteady == "unsteady" else None,
                                                mut if closure == "mut" else None))

    def gauge_ref(self, p_int: np.ndarray) -> np.ndarray:
        """Reference (CFD) pressure on the sub-cloud with the same gauge."""
        ps = p_int[self.sub]
        return ps - ps[self.o_idx].mean()


def pressure_scores(p_pred_sub: np.ndarray, p_ref_sub: np.ndarray, integ: PressureIntegrator,
                    sac_sub: np.ndarray) -> Dict[str, float]:
    """Relative L2, RMS (Pa) and inlet-outlet drop of a sub-cloud gauge pressure."""
    def rel(m):
        return float(np.linalg.norm(p_pred_sub[m] - p_ref_sub[m]) / np.linalg.norm(p_ref_sub[m]))

    def drop(p):
        return float(p[integ.i_idx].mean() - p[integ.o_idx].mean())

    out = {"p_rel_l2": rel(np.ones(len(p_ref_sub), bool)),
           "p_rms_Pa": float(np.sqrt(np.mean((p_pred_sub - p_ref_sub) ** 2))),
           "dp_pred_Pa": drop(p_pred_sub), "dp_cfd_Pa": drop(p_ref_sub)}
    if sac_sub.any():
        out["p_rel_l2_sac"] = rel(sac_sub)
    return out


def wall_interpolant(X_int: np.ndarray, U: np.ndarray, wall_xyz: np.ndarray) -> LinearNDInterpolator:
    """Linear interpolant of a nodal velocity field with u = 0 on the wall nodes."""
    return LinearNDInterpolator(np.vstack([X_int, wall_xyz]), np.vstack([U, np.zeros((len(wall_xyz), 3))]))


def wss_newton2(interp: LinearNDInterpolator, wall_xyz: np.ndarray, normals: np.ndarray,
                h_mm: float = 0.25) -> np.ndarray:
    """Second-order one-sided Newtonian wall shear from u_t at h and 2h inside the wall."""
    h = h_mm / 1000.0
    tang = lambda U: U - np.sum(U * normals, 1, keepdims=True) * normals  # noqa: E731
    u1, u2 = tang(interp(wall_xyz + h * normals)), tang(interp(wall_xyz + 2 * h * normals))
    est = MU * (4 * u1 - u2) / (2 * h)
    return np.where(np.isfinite(est).all(1)[:, None], est, 0.0)


def wss_scores(Wp: np.ndarray, Wt: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Magnitude and vector relative L2, 99th-percentile ratio and Pearson r on ``mask``."""
    mp, mt = np.linalg.norm(Wp[mask], axis=1), np.linalg.norm(Wt[mask], axis=1)
    return {"wss_mag_rel_l2": float(np.linalg.norm(mp - mt) / np.linalg.norm(mt)),
            "wss_vec_rel_l2": float(np.linalg.norm(Wp[mask] - Wt[mask]) / np.linalg.norm(Wt[mask])),
            "wss_q99_ratio": float(np.quantile(mp, .99) / np.quantile(mt, .99)),
            "wss_pearson": float(np.corrcoef(mp, mt)[0, 1])}


def velocity_scores(Up: np.ndarray, Ut: np.ndarray, U_ref: float, sac: np.ndarray) -> Dict[str, float]:
    """Vector relative L2, RMS in m/s and over ``U_ref``, 95th-percentile error, sac L2."""
    e = np.linalg.norm(Up - Ut, axis=1)
    out = {"vel_rel_l2": float(np.linalg.norm(Up - Ut) / np.linalg.norm(Ut)),
           "vel_rms_ms": float(np.sqrt(np.mean(e ** 2))),
           "vel_rms_over_Uref": float(np.sqrt(np.mean(e ** 2)) / U_ref),
           "vel_p95_ms": float(np.quantile(e, .95))}
    if sac.any():
        out["vel_rel_l2_sac"] = float(np.linalg.norm(Up[sac] - Ut[sac]) / np.linalg.norm(Ut[sac]))
    return out


def score_field(P, integ: PressureIntegrator, fields: Dict[int, np.ndarray],
                h_pre_mm: float = 0.25, h_sweep=(0.1, 0.25, 0.5)) -> Dict:
    """Score one method's nodal velocity at both window times on every hidden target.

    ``fields`` maps window time (ms) to interior velocity (N_int, 3). The pressure step is
    evaluated in all four :data:`PRESSURE_FORMS`; wall shear at the pre-registered
    ``h_pre_mm`` (primary) and over ``h_sweep`` (an oracle choice, labelled as such).
    """
    tgt = P.snaps[P.target_ms]
    t_other = [t for t in P.times_ms if t != P.target_ms][0]
    U = fields[P.target_ms]
    dUdt = (fields[P.target_ms] - fields[t_other]) / ((P.target_ms - t_other) / 1000)
    hid = P.hidden_ids
    U_ref = float(np.quantile(np.linalg.norm(tgt.uvw, axis=1), 0.995))
    out = {"velocity": velocity_scores(U[hid], tgt.uvw[hid], U_ref, P.sac[hid])}
    an = P.wall_aneurysm
    if an.any():
        interp = wall_interpolant(P.X, U, P.wall_xyz)
        out["wss"] = {**wss_scores(wss_newton2(interp, P.wall_xyz, P.wall_normals, h_pre_mm), tgt.wall_wss, an),
                      "h_mm": h_pre_mm}
        out["wss_h_sweep_oracle"] = {str(h): wss_scores(wss_newton2(interp, P.wall_xyz, P.wall_normals, h),
                                                        tgt.wall_wss, an)["wss_mag_rel_l2"] for h in h_sweep}
    p_ref = integ.gauge_ref(tgt.p)
    sac_sub = P.sac[integ.sub]
    out["pressure"] = {form: pressure_scores(integ.pressure(U, dUdt, tgt.mut, form), p_ref, integ, sac_sub)
                       for form in PRESSURE_FORMS}
    return out
