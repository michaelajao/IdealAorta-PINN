"""Data-fidelity losses: velocity, pressure, and autodiff wall shear stress.

All quantities are in the standardized non-dimensional units defined by
``dataset.normalize.Normalizer`` (network outputs u_s, v_s, w_s, p_s). WSS is
computed from the velocity gradient at the wall (Arzani et al., 2021): in the
single-length-scale nondimensionalization the Newtonian non-dimensional wall
stress equals the standardized strain rate, ``tau_s = grad_s u_s + grad_s u_s^T``.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn

_MSE = nn.MSELoss()


def _net_in(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
            params: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, y, z, params], dim=1)


def data_velocity_loss(networks: Mapping[str, nn.Module],
                       x, y, z, params,
                       u_t, v_t, w_t,
                       relative: bool = False, eps: float = 1e-8,
                       phase_gain: float = 1.0) -> torch.Tensor:
    """Velocity data loss (standardized) at data points.

    S1 (preferred): with ``phase_gain = s = U_ref_phase/U_ref`` the absolute MSE is
    divided by ``s**2``. The velocity nets already emit ``u_s = q*s`` (q ~ O(1)), so
    MSE(u_s, u_t) ~ s**2; dividing by s**2 recovers the O(1) error in q, giving both
    phases an EQUAL gradient budget. s=1 for systole -> systole unchanged.

    Legacy: ``relative=True`` divides by the group's mean-square target speed
    (over-corrects; superseded by per-phase scaling + phase_gain).
    """
    net_in = _net_in(x, y, z, params)
    u = networks["u"](net_in).view(-1, 1)
    v = networks["v"](net_in).view(-1, 1)
    w = networks["w"](net_in).view(-1, 1)
    if relative:
        num = ((u - u_t) ** 2 + (v - v_t) ** 2 + (w - w_t) ** 2).mean()
        den = (u_t ** 2 + v_t ** 2 + w_t ** 2).mean() + eps
        return num / den
    mse = _MSE(u, u_t) + _MSE(v, v_t) + _MSE(w, w_t)
    if phase_gain != 1.0:
        mse = mse / (phase_gain ** 2)
    return mse


def data_pressure_loss(net_p: nn.Module, x, y, z, params, p_t) -> torch.Tensor:
    """MSE between predicted and CFD wall pressure (standardized)."""
    p = net_p(_net_in(x, y, z, params)).view(-1, 1)
    return _MSE(p, p_t)


def wss_loss(networks: Mapping[str, nn.Module],
             x, y, z, params,
             wss_t: torch.Tensor, normals: torch.Tensor,
             wss_std: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Autodiff WSS loss at wall points.

    Args:
        wss_t: standardized target WSS components ``(N,3)`` (i.e. tau/tau_ref/wss_std).
        normals: inward unit wall normals ``(N,3)``.
        wss_std: scalar WSS standardization factor.

    Returns:
        (loss, predicted standardized WSS components ``(N,3)``).
    """
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)
    net_in = _net_in(x, y, z, params)

    u = networks["u"](net_in).view(-1, 1)
    v = networks["v"](net_in).view(-1, 1)
    w = networks["w"](net_in).view(-1, 1)
    one = torch.ones_like(u)

    def grad(o, i):
        return torch.autograd.grad(o, i, grad_outputs=one, create_graph=True, retain_graph=True)[0]

    u_x, u_y, u_z = grad(u, x), grad(u, y), grad(u, z)
    v_x, v_y, v_z = grad(v, x), grad(v, y), grad(v, z)
    w_x, w_y, w_z = grad(w, x), grad(w, y), grad(w, z)

    # Non-dimensional Newtonian stress tensor tau_s_ij = du_i/dx_j + du_j/dx_i
    txx, tyy, tzz = 2.0 * u_x, 2.0 * v_y, 2.0 * w_z
    txy, txz, tyz = (u_y + v_x), (u_z + w_x), (v_z + w_y)

    nx, ny, nz = normals[:, 0:1], normals[:, 1:2], normals[:, 2:3]
    tx = txx * nx + txy * ny + txz * nz
    ty = txy * nx + tyy * ny + tyz * nz
    tz = txz * nx + tyz * ny + tzz * nz
    t_dot_n = tx * nx + ty * ny + tz * nz

    wss_x = (tx - t_dot_n * nx) / wss_std
    wss_y = (ty - t_dot_n * ny) / wss_std
    wss_z = (tz - t_dot_n * nz) / wss_std
    pred = torch.cat([wss_x, wss_y, wss_z], dim=1)

    loss = _MSE(pred, wss_t)
    return loss, pred


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Relative L2 error ||pred - true|| / ||true||."""
    return (torch.norm(pred - true) / (torch.norm(true) + 1e-12)).item()


def vector_metrics(pred: torch.Tensor, true: torch.Tensor) -> Dict[str, float]:
    """Common error metrics between two ``(N, d)`` tensors."""
    diff = pred - true
    pf, tf = pred.flatten(), true.flatten()
    vx, vy = pf - pf.mean(), tf - tf.mean()
    corr = (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-12)
    return {
        "relative_l2": relative_l2(pred, true),
        "mae": diff.abs().mean().item(),
        "rmse": torch.sqrt((diff ** 2).mean()).item(),
        "correlation": corr.item(),
    }


# ---------------------------------------------------------------------------
# Boundary-condition losses (soft): no-slip wall, inlet velocity, outlet pressure
# ---------------------------------------------------------------------------
def noslip_loss(networks: Mapping[str, nn.Module], x, y, z, params) -> torch.Tensor:
    """u = v = w = 0 at wall points."""
    net_in = _net_in(x, y, z, params)
    u = networks["u"](net_in)
    v = networks["v"](net_in)
    w = networks["w"](net_in)
    z0 = torch.zeros_like(u)
    return _MSE(u, z0) + _MSE(v, z0) + _MSE(w, z0)


def inlet_velocity_loss(networks: Mapping[str, nn.Module], x, y, z, params,
                        u_inlet_nd: float, axial_dim: int = 0) -> torch.Tensor:
    """Uniform axial inlet velocity (non-dimensional); transverse components 0."""
    net_in = _net_in(x, y, z, params)
    comps = [networks["u"](net_in), networks["v"](net_in), networks["w"](net_in)]
    target_axial = torch.full_like(comps[axial_dim], u_inlet_nd)
    loss = _MSE(comps[axial_dim], target_axial)
    for d in range(3):
        if d != axial_dim:
            loss = loss + _MSE(comps[d], torch.zeros_like(comps[d]))
    return loss


def outlet_pressure_loss(net_p: nn.Module, x, y, z, params) -> torch.Tensor:
    """Zero gauge pressure at the outlet (p_s = 0)."""
    p = net_p(_net_in(x, y, z, params))
    return _MSE(p, torch.zeros_like(p))
