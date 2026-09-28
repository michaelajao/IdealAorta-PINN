"""Space-time neural field for one short window, and its RANS-mean residuals.

Scaling (single length scale, as in ``idealaorta_pinn.pinn.physics``):

    x_s = (x - x_c) / L          t_in = (t - t_c) / tau      (window mapped to [-1, 1])
    u_s = u / U                  p_s = (p - p_out(t)) / (rho U^2)
    nu_s = (mu + mu_t) / (rho U L)

Dividing the momentum balance by U^2 / L gives

    (L / (U tau)) du_s/dt_in + (u_s . grad_s) u_s + grad_s p_s
        - div_s[ nu_s (grad_s u_s + grad_s u_s^T) ] = 0,

with the viscous term expanded without assuming div u = 0:

    div[nu (grad u + grad u^T)]_i = nu (lap u_i + d_i div u) + d_j nu (d_j u_i + d_i u_j).

No (2/3) rho k term: the exported pressure satisfies the balance without it (reference
budget, all 12 cases). ``nu_s`` and its gradient are inputs (oracle closure) or the
molecular value alone (laminar arm); the steady arm drops the time term.

``residuals`` (repeated autograd) is the readable reference implementation;
``residuals_func`` (vectorized forward-over-reverse, several times faster) is what
training uses. Both are checked against an exact Navier-Stokes solution in
``tests/test_reconstruction_fields.py``.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


class FieldNet(nn.Module):
    """(x_s, t_in) -> (u_s, v_s, w_s, p_s); Gaussian Fourier features on space."""

    def __init__(self, width: int = 256, depth: int = 6, n_freq: int = 64, sigma: float = 2.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(3, n_freq, generator=g) * sigma)
        layers, d_in = [], 2 * n_freq + 4
        for i in range(depth):
            layers += [nn.Linear(d_in if i == 0 else width, width), nn.SiLU()]
        layers += [nn.Linear(width, 4)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, xs: torch.Tensor, tin: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * xs @ self.B
        return self.mlp(torch.cat([torch.sin(proj), torch.cos(proj), xs, tin], dim=1))


def _g(out: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True, retain_graph=True)[0]


def residuals(fn, xs: torch.Tensor, tin: torch.Tensor, nu_s: torch.Tensor, gnu_s: torch.Tensor,
              time_coef: float, steady: bool, momentum: bool = True) -> Dict[str, torch.Tensor]:
    """Continuity and (optionally) momentum residuals at collocation points.

    ``fn(xs, tin)`` returns (N,4); xs and tin must require grad. ``nu_s`` (N,1) and
    ``gnu_s`` (N,3) are the non-dimensional effective viscosity and its x_s-gradient.
    """
    out = fn(xs, tin)
    u, p = out[:, :3], out[:, 3:4]
    J = torch.stack([_g(u[:, i:i + 1], xs) for i in range(3)], dim=1)       # J[n,i,j] = d u_i / d x_j
    div = J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]
    res = {"cont": div}
    if not momentum:
        return res
    gp = _g(p, xs)
    lap = torch.stack([sum(_g(J[:, i, j:j + 1], xs)[:, j] for j in range(3)) for i in range(3)], dim=1)
    gdiv = _g(div.unsqueeze(1), xs)
    S = J + J.transpose(1, 2)
    visc = nu_s * (lap + gdiv) + torch.einsum("nj,nij->ni", gnu_s, S)
    conv = torch.einsum("nj,nij->ni", u, J)
    mom = conv + gp - visc
    if not steady:
        dudt = torch.cat([_g(u[:, i:i + 1], tin) for i in range(3)], dim=1)
        mom = mom + time_coef * dudt
    res["mom"] = mom
    return res


def wall_shear_s(fn, xs: torch.Tensor, tin: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Tangential part of (grad u + grad u^T) n in scaled units; times mu U / L gives Pa."""
    out = fn(xs, tin)
    J = torch.stack([_g(out[:, i:i + 1], xs) for i in range(3)], dim=1)
    t = torch.einsum("nij,nj->ni", J + J.transpose(1, 2), normals)
    return t - (t * normals).sum(1, keepdim=True) * normals


def residuals_func(net: nn.Module, xs: torch.Tensor, tin: torch.Tensor, nu_s: torch.Tensor,
                   gnu_s: torch.Tensor, time_coef: float, steady: bool,
                   momentum: bool = True) -> Dict[str, torch.Tensor]:
    """Same as :func:`residuals`, via vectorized forward-over-reverse derivatives.

    Only for ``nn.Module`` fields taking (xs, tin); equal to :func:`residuals` to round-off
    (``tests/test_reconstruction_fields.py``) and several times faster.
    """
    from torch.func import functional_call, hessian, jacrev, vmap

    params = {k: v for k, v in net.named_parameters()}
    buffers = {k: v for k, v in net.named_buffers()}

    def f(z):                                   # z = (x, y, z, t) -> (u, v, w, p)
        return functional_call(net, (params, buffers), (z[None, :3], z[None, 3:]))[0]

    z = torch.cat([xs, tin], 1)
    out = net(xs, tin)
    Jz = vmap(jacrev(f))(z)                     # (N, 4 out, 4 in)
    J = Jz[:, :3, :3]
    div = J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]
    res = {"cont": div}
    if not momentum:
        return res
    H = vmap(hessian(lambda q: f(q)[:3]))(z)[:, :, :3, :3]      # (N, 3, 3, 3): d2 u_i / dx_j dx_k
    lap = H.diagonal(dim1=2, dim2=3).sum(-1)                    # (N,3)
    gdiv = H[:, 0, 0, :] + H[:, 1, 1, :] + H[:, 2, 2, :]         # d_k div u  (N,3)
    S = J + J.transpose(1, 2)
    visc = nu_s * (lap + gdiv) + torch.einsum("nj,nij->ni", gnu_s, S)
    mom = torch.einsum("nj,nij->ni", out[:, :3], J) + Jz[:, 3, :3] - visc
    if not steady:
        mom = mom + time_coef * Jz[:, :3, 3]
    res["mom"] = mom
    return res
