"""RANS-mean Navier-Stokes physics residuals (steady, quasi-steady per phase).

Nondimensionalization (single length scale — see dataset/normalize.py)
----------------------------------------------------------------------
Physical coordinates map to standardized coordinates by ``x = x_mean + L * x_s``
with ``x_s`` ~ O(1). Velocity ``u = U_ref * u_s`` and gauge pressure
``p = P_ref * p_s`` with ``P_ref = rho * U_ref^2``. Substituting into the steady
incompressible RANS-mean momentum equation and dividing by ``U_ref^2 / L`` gives
a fully dimensionless residual in which every derivative is taken w.r.t. the
SAME standardized coordinate ``x_s``:

    (u_s . grad_s) u_s + grad_s p_s
        = div_s[ (1/Re + nu_t_s) (grad_s u_s + grad_s u_s^T) ]

with ``Re = rho * U_ref * L / mu`` and ``nu_t_s = nu_t / (U_ref * L)`` the
non-dimensional eddy viscosity output by the nut network. For incompressible
flow the viscous term expands to

    (1/Re + nu_t_s) * laplacian_s(u_{s,i})
        + grad_s(nu_t_s) . (grad_s u_s + grad_s u_s^T)_i .

Note the second term contracts the eddy-viscosity gradient with row ``i`` of the
FULL tensor ``(grad_s u_s + grad_s u_s^T)`` — that is ``2 S_i``, twice the
conventional strain-rate tensor ``S = (grad u + grad u^T) / 2``. The molecular
part contributes no gradient term because ``1/Re`` is constant.

The eddy-viscosity gradient is tracked (the nut field is smooth and learnable);
this matches the SST k-omega-transition mean field the CFD produced without
re-solving the turbulence transport equations.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Tuple

import torch
import torch.nn as nn

FieldFn = Callable[[torch.Tensor], torch.Tensor]


def _grad(out: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """First derivative d(out)/d(inp); robust to constant fields.

    A field that is constant (or linear, after one derivative) yields an output
    with no grad_fn; its derivative is identically zero. Real networks are
    nonlinear so this guard never triggers in training, but it keeps analytic
    checks and degenerate slices well-defined.
    """
    if not out.requires_grad:
        return torch.zeros_like(inp)
    g = torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out),
                            create_graph=True, retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(inp) if g is None else g


def compute_residuals(fields: Mapping[str, FieldFn],
                      x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                      params: torch.Tensor,
                      Re: float) -> Dict[str, torch.Tensor]:
    """Compute momentum + continuity residuals at the given collocation points.

    Args:
        fields: callables for 'u','v','w','p' (and optionally 'nut'), each taking
            the concatenated input ``[x, y, z, params]`` and returning ``(N,1)``.
        x, y, z: standardized coordinates ``(N,1)``, MUST require grad.
        params: per-point conditioning vector ``(N, n_param)`` (no grad needed).
        Re: Reynolds number based on the single length scale L.

    Returns:
        Dict with residual tensors ('res_x','res_y','res_z','res_cont') and the
        raw field values ('u','v','w','p','nut') for monitoring.
    """
    net_in = torch.cat([x, y, z, params], dim=1)
    u = fields["u"](net_in).view(-1, 1)
    v = fields["v"](net_in).view(-1, 1)
    w = fields["w"](net_in).view(-1, 1)
    p = fields["p"](net_in).view(-1, 1)

    has_nut = "nut" in fields
    nut = fields["nut"](net_in).view(-1, 1) if has_nut else torch.zeros_like(u)

    # First derivatives
    u_x, u_y, u_z = _grad(u, x), _grad(u, y), _grad(u, z)
    v_x, v_y, v_z = _grad(v, x), _grad(v, y), _grad(v, z)
    w_x, w_y, w_z = _grad(w, x), _grad(w, y), _grad(w, z)
    p_x, p_y, p_z = _grad(p, x), _grad(p, y), _grad(p, z)

    # Second derivatives (Laplacian components)
    u_xx, u_yy, u_zz = _grad(u_x, x), _grad(u_y, y), _grad(u_z, z)
    v_xx, v_yy, v_zz = _grad(v_x, x), _grad(v_y, y), _grad(v_z, z)
    w_xx, w_yy, w_zz = _grad(w_x, x), _grad(w_y, y), _grad(w_z, z)

    if has_nut:
        nut_x, nut_y, nut_z = _grad(nut, x), _grad(nut, y), _grad(nut, z)
    else:
        nut_x = nut_y = nut_z = torch.zeros_like(u)

    nu_eff = (1.0 / Re) + nut  # (N,1)

    lap_u = u_xx + u_yy + u_zz
    lap_v = v_xx + v_yy + v_zz
    lap_w = w_xx + w_yy + w_zz

    # grad(nu_t) . (strain rows): 2 S_ij contributions
    visc_x = nu_eff * lap_u + (nut_x * 2.0 * u_x + nut_y * (u_y + v_x) + nut_z * (u_z + w_x))
    visc_y = nu_eff * lap_v + (nut_x * (v_x + u_y) + nut_y * 2.0 * v_y + nut_z * (v_z + w_y))
    visc_z = nu_eff * lap_w + (nut_x * (w_x + u_z) + nut_y * (w_y + v_z) + nut_z * 2.0 * w_z)

    res_x = u * u_x + v * u_y + w * u_z + p_x - visc_x
    res_y = u * v_x + v * v_y + w * v_z + p_y - visc_y
    res_z = u * w_x + v * w_y + w * w_z + p_z - visc_z
    res_cont = u_x + v_y + w_z

    return {
        "res_x": res_x, "res_y": res_y, "res_z": res_z, "res_cont": res_cont,
        "u": u, "v": v, "w": w, "p": p, "nut": nut,
    }


def compute_physics_loss(networks: Mapping[str, nn.Module],
                         x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                         params: torch.Tensor,
                         Re: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    """MSE of momentum + continuity residuals at interior collocation points."""
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)

    r = compute_residuals(networks, x, y, z, params, Re)

    mse = nn.MSELoss()
    zero = torch.zeros_like(r["res_x"])
    loss = (mse(r["res_x"], zero) + mse(r["res_y"], zero)
            + mse(r["res_z"], zero) + mse(r["res_cont"], torch.zeros_like(r["res_cont"])))

    stats = {
        "res_x": r["res_x"].detach().abs().mean().item(),
        "res_y": r["res_y"].detach().abs().mean().item(),
        "res_z": r["res_z"].detach().abs().mean().item(),
        "res_cont": r["res_cont"].detach().abs().mean().item(),
        "nut_mean": r["nut"].detach().mean().item(),
        "nut_max": r["nut"].detach().max().item(),
    }
    return loss, stats
