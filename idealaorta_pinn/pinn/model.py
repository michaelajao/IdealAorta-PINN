"""Parametric PINN field networks (P2INN-style parameter conditioning).

Follows the group's PINN "house style" (TAA-aneurysm): one decoupled scalar
subnetwork per output (u, v, w, p) plus a smaller turbulent-viscosity network
(nut), each built from a Fourier-feature coordinate encoder and Swish residual
blocks.

Parameter conditioning follows P2INN (Cho et al., 2024) and IP-PINN
(Kalajahi-Arzani, 2025): rather than treating the conditioning parameters
``mu = [d_inlet*, beta, disease_flag, phase]`` "merely as a coordinate", they are
passed through a small dedicated *parameter encoder* and concatenated with the
Fourier-encoded coordinates before the manifold network::

    f_Theta(x; mu) = g_manifold( [ FourierEncode(x) ; g_param(mu) ] )

Raw concatenation (no parameter encoder) is available as an ablation via
``use_param_encoder=False``. Only the spatial coordinates are Fourier-encoded;
the parameters index a geometry family, not an oscillatory field.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish activation f(x) = x * sigmoid(x) — smooth higher derivatives."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    """Residual block: x -> Linear -> Swish -> Linear -> (+x).

    Deliberately no activation *after* the skip-add: the physics residuals take
    second derivatives of this network, so keeping the skip path linear preserves
    gradient flow for the Laplacian.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(dim, dim), Swish(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(x)


class FourierFeatures(nn.Module):
    """Random Fourier-feature encoding of spatial coordinates (Tancik 2020).

    Maps a ``(N, n_spatial)`` coordinate tensor to ``[sin(2pi B x), cos(2pi B x)]``.
    """

    def __init__(self, n_spatial: int, num_frequencies: int = 16, scale=1.0):
        super().__init__()
        self.n_spatial = n_spatial
        # S4: ``scale`` may be a scalar (isotropic, standard Tancik) OR a per-axis
        # vector of length n_spatial (anisotropic). A per-axis scale gives each
        # spatial axis its own Fourier bandwidth — used to match the encoding's
        # resolution to an anisotropic standardized domain (long thin vessel), so
        # the transverse shear layer / recirculation are not under-resolved.
        s = torch.as_tensor(scale, dtype=torch.float32)
        if s.ndim == 0:
            s = s.expand(n_spatial)
        B = torch.randn(n_spatial, num_frequencies) * s.view(n_spatial, 1)
        self.register_buffer("B", B)
        self.out_features = 2 * num_frequencies

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * coords @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def _mlp(in_dim: int, hidden: List[int],
         activation: Optional[Callable[[], nn.Module]] = None) -> nn.Sequential:
    """Plain ``Linear -> activation`` stack (no output layer; callers append theirs).

    ``activation`` is a *factory* (e.g. ``Swish`` or ``lambda: nn.LeakyReLU(0.2)``),
    not an instance: each layer needs its own module, and taking a factory keeps any
    configured hyperparameters instead of silently reconstructing a bare default.
    """
    make_act = activation or Swish
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), make_act()]
        prev = h
    return nn.Sequential(*layers)


class FieldNet(nn.Module):
    """Scalar parametric field network: ``[x, y, z, mu] -> scalar``."""

    def __init__(self,
                 n_spatial: int = 3,
                 n_param: int = 4,
                 hidden_dim: int = 128,
                 num_layers: int = 6,
                 num_frequencies: int = 16,
                 fourier_scale: float = 1.0,
                 use_fourier: bool = True,
                 use_param_encoder: bool = True,
                 param_encoder_dims: Optional[List[int]] = None,
                 coord_encoder_dims: Optional[List[int]] = None,
                 vel_phase_gain: Optional[float] = None):
        super().__init__()
        self.n_spatial = n_spatial
        self.n_param = n_param
        self.use_fourier = use_fourier
        self.use_param_encoder = use_param_encoder and n_param > 0
        # S1: per-phase output gain (velocity nets only). The raw output is trained
        # to O(1) for BOTH phases; multiplying by 1.0 for systole (phase param==1)
        # and by this diastolic gain for diastole (phase==0) yields the single-scale
        # standardized velocity u_s that physics/WSS/BC consume. None -> no gain.
        self.vel_phase_gain = vel_phase_gain

        # Coordinate encoding g_theta_c: Fourier features, optionally followed by
        # an FC coordinate encoder (literal P2INN g_theta_c).
        if use_fourier:
            self.fourier: Optional[FourierFeatures] = FourierFeatures(
                n_spatial, num_frequencies, fourier_scale)
            coord_dim = self.fourier.out_features
        else:
            self.fourier = None
            coord_dim = n_spatial
        if coord_encoder_dims:
            self.coord_encoder: Optional[nn.Sequential] = _mlp(coord_dim, coord_encoder_dims)
            coord_dim = coord_encoder_dims[-1]
        else:
            self.coord_encoder = None

        # Parameter encoding g_theta_p (P2INN-style); identity (raw concat) if disabled.
        if self.use_param_encoder:
            dims = param_encoder_dims or [32, 32]
            self.param_encoder: Optional[nn.Sequential] = _mlp(n_param, dims)
            param_dim = dims[-1]
        else:
            self.param_encoder = None
            param_dim = n_param

        self.encoder = nn.Sequential(nn.Linear(coord_dim + param_dim, hidden_dim), Swish())
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_layers)])
        self.decoder = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coords = x[:, : self.n_spatial]
        params = x[:, self.n_spatial:]

        c = self.fourier(coords) if self.fourier is not None else coords
        if self.coord_encoder is not None:
            c = self.coord_encoder(c)
        if self.param_encoder is not None:
            pp = self.param_encoder(params)
        else:
            pp = params
        h = torch.cat([c, pp], dim=-1)
        h = self.encoder(h)
        h = self.residual_blocks(h)
        out = self.decoder(h)
        if self.vel_phase_gain is not None:
            phase = params[:, -1:]                       # 1.0 systole, 0.0 diastole
            gain = phase + (1.0 - phase) * self.vel_phase_gain
            out = out * gain
        return out


class NutNet(FieldNet):
    """Turbulent-viscosity network with softplus positivity and a hard floor.

    Output = softplus(raw + SHIFT) + nu_t_min. The +SHIFT keeps the softplus in
    its high-gradient regime at init, avoiding the vanishing-gradient trap that
    collapses nu_t toward zero (mirrors the TAA Net2_nut design).
    """

    SOFTPLUS_SHIFT = 2.0

    def __init__(self, *args, initial_nut: float = 0.05, nu_t_min: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.nu_t_min = nu_t_min
        # C3: forward is softplus(raw+SHIFT)+nu_t_min, so to make the init output
        # equal `initial_nut` we must subtract the floor before inverting softplus
        # (the previous code ignored +nu_t_min, biasing init high by the floor).
        nut_above_floor = max(initial_nut - nu_t_min, 1e-8)
        target_raw = math.log(math.expm1(nut_above_floor))
        nn.init.constant_(self.decoder.bias, target_raw - self.SOFTPLUS_SHIFT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = super().forward(x)
        return torch.nn.functional.softplus(raw + self.SOFTPLUS_SHIFT) + self.nu_t_min


def create_networks(n_param: int = 4,
                    hidden_dim: int = 128,
                    num_layers: int = 6,
                    num_frequencies: int = 16,
                    fourier_scale: float = 1.0,
                    use_fourier: bool = True,
                    use_param_encoder: bool = True,
                    param_encoder_dims: Optional[List[int]] = None,
                    coord_encoder_dims: Optional[List[int]] = None,
                    nut_hidden_dim: int = 64,
                    nut_num_layers: int = 4,
                    nu_t_min: float = 1e-3,
                    initial_nut: float = 0.05,
                    vel_phase_gain: Optional[float] = None,
                    device: str = "cuda") -> Dict[str, nn.Module]:
    """Create the five parametric field networks (u, v, w, p, nut).

    ``vel_phase_gain`` (S1): per-phase diastolic gain applied to the u/v/w nets
    only (p and nut are unscaled). None -> single-scale (legacy) behavior.
    """
    common = dict(n_spatial=3, n_param=n_param, num_frequencies=num_frequencies,
                  fourier_scale=fourier_scale, use_fourier=use_fourier,
                  use_param_encoder=use_param_encoder, param_encoder_dims=param_encoder_dims,
                  coord_encoder_dims=coord_encoder_dims)
    nets = {
        "u": FieldNet(hidden_dim=hidden_dim, num_layers=num_layers, vel_phase_gain=vel_phase_gain, **common),
        "v": FieldNet(hidden_dim=hidden_dim, num_layers=num_layers, vel_phase_gain=vel_phase_gain, **common),
        "w": FieldNet(hidden_dim=hidden_dim, num_layers=num_layers, vel_phase_gain=vel_phase_gain, **common),
        "p": FieldNet(hidden_dim=hidden_dim, num_layers=num_layers, **common),
        "nut": NutNet(hidden_dim=nut_hidden_dim, num_layers=nut_num_layers,
                      nu_t_min=nu_t_min, initial_nut=initial_nut, **common),
    }
    return {k: net.to(device) for k, net in nets.items()}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
