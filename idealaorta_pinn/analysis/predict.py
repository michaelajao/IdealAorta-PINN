"""Load a trained surrogate and evaluate fields in physical units.

Reconstructs the networks from a checkpoint's config, restores weights and the
fitted :class:`Normalizer`, and provides batched prediction of velocity and
pressure (in SI units) at arbitrary physical coordinates and parameter values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from ..data.normalize import Normalizer
from ..pinn.model import create_networks


@dataclass
class TrainedModel:
    networks: Dict[str, torch.nn.Module]
    normalizer: Normalizer
    Re: float
    config: Dict
    device: str


def load_trained(checkpoint_path: str | Path, device: str = "cuda") -> TrainedModel:
    """Load networks + normalizer + scales from a training checkpoint."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    m = cfg["model"]
    nut = m.get("nut", {})
    # Rebuild the normalizer first so the per-phase velocity gain (S1) can be
    # restored into the velocity nets — otherwise predictions are mis-scaled.
    norm = Normalizer.from_dict(ckpt["normalizer"])
    vel_phase_gain = (norm.vel_phase_gain("diastolic")
                      if norm.u_ref_diastolic is not None else None)
    networks = create_networks(
        n_param=int(ckpt.get("n_param", 4)),
        hidden_dim=int(m.get("hidden_dim", 128)),
        num_layers=int(m.get("num_layers", 6)),
        num_frequencies=int(m.get("num_frequencies", 16)),
        fourier_scale=float(m.get("fourier_scale", 1.0)),
        use_fourier=bool(m.get("use_fourier", True)),
        use_param_encoder=bool(m.get("use_param_encoder", True)),
        param_encoder_dims=m.get("param_encoder_dims"),
        coord_encoder_dims=m.get("coord_encoder_dims"),
        nut_hidden_dim=int(nut.get("hidden_dim", 64)),
        nut_num_layers=int(nut.get("num_layers", 4)),
        nu_t_min=float(nut.get("nu_t_min", 1e-3)),
        initial_nut=float(nut.get("initial_nut", 0.05)),
        vel_phase_gain=vel_phase_gain,
        device=device,
    )
    for k, net in networks.items():
        net.load_state_dict(ckpt["networks"][k])
        net.eval()
    return TrainedModel(networks=networks, normalizer=norm,
                        Re=float(ckpt.get("Re", norm.Re)), config=cfg, device=device)


def _phase_value(phase: str) -> float:
    return 1.0 if phase == "systolic" else 0.0


@torch.no_grad()
def predict_physical(model: TrainedModel, coords_phys: np.ndarray,
                     diameter_cm: float, disease_flag: int, phase: str,
                     beta: float = 1.0, batch: int = 100_000) -> Dict[str, np.ndarray]:
    """Predict (u, v, w, p, speed) in SI units at physical coordinates ``(N,3)``.

    The parameter vector is ``[d_inlet*, beta, disease_flag, phase]`` to match
    training; ``beta`` defaults to 1.0 (symmetric / healthy).
    """
    norm = model.normalizer
    cs = norm.coords_std(coords_phys).astype(np.float32)
    mu = np.array([norm.diameter_nd(diameter_cm), float(beta), float(disease_flag),
                   _phase_value(phase)], dtype=np.float32)
    params = np.tile(mu, (len(cs), 1))

    us, vs, ws, ps = [], [], [], []
    for i in range(0, len(cs), batch):
        net_in = torch.tensor(np.hstack([cs[i:i + batch], params[i:i + batch]]),
                              device=model.device)
        us.append(model.networks["u"](net_in).cpu().numpy())
        vs.append(model.networks["v"](net_in).cpu().numpy())
        ws.append(model.networks["w"](net_in).cpu().numpy())
        ps.append(model.networks["p"](net_in).cpu().numpy())

    u = norm.vel_to_physical(np.vstack(us)).ravel()
    v = norm.vel_to_physical(np.vstack(vs)).ravel()
    w = norm.vel_to_physical(np.vstack(ws)).ravel()
    p = norm.pressure_to_physical(np.vstack(ps)).ravel()
    speed = np.sqrt(u * u + v * v + w * w)
    return {"u": u, "v": v, "w": w, "p": p, "speed": speed}
