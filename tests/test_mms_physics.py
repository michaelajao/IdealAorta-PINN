"""Manufactured-solution (MMS) verification of the RANS-mean residual operator.

The existing test_physics_residual.py checks only continuity (a divergence-free
field gives ~0, a divergent one does not). This file verifies the FULL momentum
operator — convection + pressure gradient + (variable-viscosity) viscous term,
including the eddy-viscosity-gradient strain cross-terms — against a closed-form
analytic residual. If a sign is flipped, a term is dropped, or the nut-gradient
cross-terms are missing, these tests fail.

Method: feed an analytic Taylor-Green field (+ analytic pressure and a chosen
nut field) into compute_residuals (which differentiates by autograd), and compare
to the residual assembled from the analytic derivatives written out by hand. Done
in float64 so the autograd second derivatives are accurate to ~1e-10 and the
tolerance can be tight. No network is involved.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from idealaorta_pinn.pinn.physics import compute_residuals

PI = math.pi


def _coords(n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Spread points over a couple of periods so sin/cos structure is exercised.
    x = (2.0 * torch.rand(n, 1, generator=g, dtype=torch.float64) - 1.0).requires_grad_(True)
    y = (2.0 * torch.rand(n, 1, generator=g, dtype=torch.float64) - 1.0).requires_grad_(True)
    z = (2.0 * torch.rand(n, 1, generator=g, dtype=torch.float64) - 1.0).requires_grad_(True)
    params = torch.zeros(n, 0, dtype=torch.float64)
    return x, y, z, params


# --- analytic Taylor-Green field (z-invariant, divergence-free) ----------------
# u =  sin(pi x) cos(pi y),  v = -cos(pi x) sin(pi y),  w = 0
def _tg_fields(nut_field):
    def u(nin):
        return torch.sin(PI * nin[:, 0:1]) * torch.cos(PI * nin[:, 1:2])

    def v(nin):
        return -torch.cos(PI * nin[:, 0:1]) * torch.sin(PI * nin[:, 1:2])

    def w(nin):
        return 0.0 * (nin[:, 0:1] + nin[:, 1:2] + nin[:, 2:3])

    def p(nin):
        return torch.cos(PI * nin[:, 0:1]) * torch.cos(PI * nin[:, 1:2])

    return {"u": u, "v": v, "w": w, "p": p, "nut": nut_field}


def _analytic_pieces(xn, yn):
    """Closed-form fields and derivatives at numpy coords (z-invariant)."""
    sx, cx = np.sin(PI * xn), np.cos(PI * xn)
    sy, cy = np.sin(PI * yn), np.cos(PI * yn)
    u, v = sx * cy, -cx * sy
    # first derivatives
    u_x, u_y = PI * cx * cy, -PI * sx * sy
    v_x, v_y = PI * sx * sy, -PI * cx * cy
    p_x, p_y = -PI * sx * cy, -PI * cx * sy
    # laplacians (z-invariant -> only xx + yy; each second deriv brings -pi^2)
    lap_u, lap_v = -2.0 * PI**2 * u, -2.0 * PI**2 * v
    return dict(u=u, v=v, u_x=u_x, u_y=u_y, v_x=v_x, v_y=v_y,
                p_x=p_x, p_y=p_y, lap_u=lap_u, lap_v=lap_v)


def test_continuity_zero_for_taylor_green():
    x, y, z, params = _coords()
    r = compute_residuals(_tg_fields(lambda nin: 0.0 * nin[:, 0:1]),
                          x, y, z, params, Re=137.0)
    assert r["res_cont"].abs().max().item() < 1e-8


def test_momentum_constant_viscosity_matches_closed_form():
    Re, C = 137.0, 0.03                       # constant eddy viscosity
    x, y, z, params = _coords()
    fields = _tg_fields(lambda nin: C + 0.0 * nin[:, 0:1])
    r = compute_residuals(fields, x, y, z, params, Re=Re)

    xn, yn = x.detach().numpy(), y.detach().numpy()
    a = _analytic_pieces(xn, yn)
    nu_eff = 1.0 / Re + C
    conv_x = a["u"] * a["u_x"] + a["v"] * a["u_y"]      # w, *_z = 0
    conv_y = a["u"] * a["v_x"] + a["v"] * a["v_y"]
    res_x = conv_x + a["p_x"] - nu_eff * a["lap_u"]
    res_y = conv_y + a["p_y"] - nu_eff * a["lap_v"]

    assert np.allclose(r["res_x"].detach().numpy(), res_x, atol=1e-7, rtol=1e-6)
    assert np.allclose(r["res_y"].detach().numpy(), res_y, atol=1e-7, rtol=1e-6)
    assert np.abs(r["res_z"].detach().numpy()).max() < 1e-8   # w-momentum trivial


def test_nut_gradient_cross_terms_are_present_and_correct():
    """Variable eddy viscosity nut = D*sin(pi x). The viscous term gains
    grad(nut).(2 S_ij) cross-terms; with this nut only nut_x is nonzero, so the
    x-momentum gains nut_x*2*u_x and nu_eff itself becomes spatially varying.
    A code path that dropped these terms (or used 1/Re instead of 1/Re+nut here)
    would fail this test."""
    Re, D = 90.0, 0.05
    x, y, z, params = _coords()
    fields = _tg_fields(lambda nin: D * torch.sin(PI * nin[:, 0:1]))
    r = compute_residuals(fields, x, y, z, params, Re=Re)

    xn, yn = x.detach().numpy(), y.detach().numpy()
    a = _analytic_pieces(xn, yn)
    nut = D * np.sin(PI * xn)
    nut_x = D * PI * np.cos(PI * xn)                  # nut_y = nut_z = 0
    nu_eff = 1.0 / Re + nut

    conv_x = a["u"] * a["u_x"] + a["v"] * a["u_y"]
    conv_y = a["u"] * a["v_x"] + a["v"] * a["v_y"]
    # viscous with variable nu_eff + grad(nut).strain cross-terms
    visc_x = nu_eff * a["lap_u"] + nut_x * 2.0 * a["u_x"]
    visc_y = nu_eff * a["lap_v"] + nut_x * (a["v_x"] + a["u_y"])   # nut_x*(v_x+u_y)
    res_x = conv_x + a["p_x"] - visc_x
    res_y = conv_y + a["p_y"] - visc_y

    assert np.allclose(r["res_x"].detach().numpy(), res_x, atol=1e-7, rtol=1e-6)
    assert np.allclose(r["res_y"].detach().numpy(), res_y, atol=1e-7, rtol=1e-6)
