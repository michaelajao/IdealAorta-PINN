"""Derivative checks for ``idealaorta_pinn.reconstruction.fields``.

1. The Ethier-Steinman exact unsteady 3D Navier-Stokes solution, written in physical units
   and mapped through the study's scalings (L, U, tau, rho), makes the non-dimensional
   residuals vanish: this checks the chain rules for x, t and p and the viscous term.
2. Dropping the time term from the same solution leaves a clear residual (the check can fail).
3. The expanded variable-viscosity term equals div[nu (grad u + grad u^T)] computed directly
   by autograd of the stress, on a smooth non-solenoidal field.
4. The vectorized ``residuals_func`` equals the reference ``residuals`` on a network.
"""

import pytest
import torch

from idealaorta_pinn.reconstruction.fields import FieldNet, _g, residuals, residuals_func

RHO, NU = 1060.0, 3.3e-3            # kinematic viscosity O(1e-3) so the viscous term matters
L, U, TAU, TC = 0.05, 0.8, 0.0025, 0.01
XC = (0.02, 0.03, -0.01)
A, D = 3.0 * torch.pi / 4 / 0.05, 3.0 * torch.pi / 2 / 0.05     # wavenumbers (1/m)


@pytest.fixture(autouse=True)
def float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(previous)


def ethier_steinman(x, t):
    """Exact solution (Ethier & Steinman 1994): velocity (m/s) and pressure (Pa)."""
    X, Y, Z = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    e = torch.exp(-NU * D ** 2 * t)
    u = -A * (torch.exp(A * X) * torch.sin(A * Y + D * Z) + torch.exp(A * Z) * torch.cos(A * X + D * Y)) * e
    v = -A * (torch.exp(A * Y) * torch.sin(A * Z + D * X) + torch.exp(A * X) * torch.cos(A * Y + D * Z)) * e
    w = -A * (torch.exp(A * Z) * torch.sin(A * X + D * Y) + torch.exp(A * Y) * torch.cos(A * Z + D * X)) * e
    p = -A ** 2 / 2 * (torch.exp(2 * A * X) + torch.exp(2 * A * Y) + torch.exp(2 * A * Z)
                       + 2 * torch.sin(A * X + D * Y) * torch.cos(A * Z + D * X) * torch.exp(A * (Y + Z))
                       + 2 * torch.sin(A * Y + D * Z) * torch.cos(A * X + D * Y) * torch.exp(A * (Z + X))
                       + 2 * torch.sin(A * Z + D * X) * torch.cos(A * Y + D * Z) * torch.exp(A * (X + Y))) * e ** 2
    return torch.cat([u, v, w], 1), RHO * p


def scaled_solution(xs, tin):
    uvw, p = ethier_steinman(torch.tensor(XC) + L * xs, TC + TAU * tin)
    return torch.cat([uvw / U, p / (RHO * U ** 2)], 1)


def _points(n=512):
    xs = (torch.rand(n, 3) * 0.4 - 0.2).requires_grad_(True)
    tin = (torch.rand(n, 1) * 2 - 1).requires_grad_(True)
    return xs, tin


def _ratios(steady):
    xs, tin = _points()
    nu = torch.full((len(xs), 1), NU / (U * L))
    r = residuals(scaled_solution, xs, tin, nu, torch.zeros(len(xs), 3), time_coef=L / (U * TAU), steady=steady)
    ref = scaled_solution(xs, tin)
    J = torch.stack([_g(ref[:, i:i + 1], xs) for i in range(3)], 1)
    conv = torch.einsum("nj,nij->ni", ref[:, :3], J).abs().mean()
    return (r["mom"].abs().mean() / conv).item(), (r["cont"].abs().mean() / J.abs().mean()).item()


def test_exact_solution_satisfies_unsteady_residual():
    mom, cont = _ratios(steady=False)
    assert mom < 1e-8 and cont < 1e-10


def test_dropping_time_term_is_detected():
    mom, _ = _ratios(steady=True)
    assert mom > 1e-3


def test_variable_viscosity_expansion_matches_stress_divergence():
    xs, tin = _points()

    def field(a, b):
        x, y, z = a[:, 0:1], a[:, 1:2], a[:, 2:3]
        return torch.cat([torch.sin(2 * x + y) * b, x * y * z + z ** 2, torch.cos(x * z) + y ** 3 * b, x * 0], 1)

    nu = 0.1 + 0.05 * torch.sin(3 * xs[:, 0:1]) * torch.cos(xs[:, 2:3])
    r = residuals(field, xs, tin, nu, _g(nu, xs), time_coef=0.0, steady=True)
    out = field(xs, tin)
    J = torch.stack([_g(out[:, i:i + 1], xs) for i in range(3)], 1)
    sig = nu[:, :, None] * (J + J.transpose(1, 2))
    div_sig = torch.stack([sum(_g(sig[:, i, j:j + 1], xs)[:, j] for j in range(3)) for i in range(3)], 1)
    direct = torch.einsum("nj,nij->ni", out[:, :3], J) - div_sig          # p = 0
    assert ((r["mom"] - direct).abs().max() / direct.abs().max()).item() < 1e-10


def test_vectorized_residuals_match_reference():
    net = FieldNet(width=64, depth=3, n_freq=16, seed=1).double()
    xs, tin = _points(256)
    nu, gnu = torch.rand(256, 1) * 0.01, torch.randn(256, 3) * 0.01
    a = residuals(net, xs, tin, nu, gnu, 7.0, steady=False)
    b = residuals_func(net, xs, tin, nu, gnu, 7.0, steady=False)
    for k in ("mom", "cont"):
        assert ((a[k] - b[k]).abs().max() / a[k].abs().max()).item() < 1e-10
