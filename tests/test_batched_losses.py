"""Correctness guard for the batched loss path.

Trainer._component_losses (the hot path) concatenates every (case,phase) group's
points and reduces with a vectorized segment-mean. It must be mathematically
identical to the readable per-group reference _component_losses_reference. This
test builds a real 2-case / 2-phase bundle (S1 + S2 on, so per-phase pg2 and
volumetric collocation are exercised) on CPU and asserts every one of the seven
loss components agrees to floating-point tolerance, plus that _physics_only
matches the physics term. Skips if the processed dataset is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.config import PROCESSED_DIR, PROJECT_ROOT  # noqa: E402
from idealaorta_pinn.pinn.train import COMPONENTS, Trainer  # noqa: E402


def _have_data() -> bool:
    reg = PROJECT_ROOT / "data" / "registry.json"
    return reg.exists() and PROCESSED_DIR.exists() and any(PROCESSED_DIR.glob("*.parquet"))


def _tiny_cfg(tmp: Path) -> dict:
    return {
        "experiment": {"name": "_test_batched"},
        "random_seed": 0,
        "device": "cpu",
        "output_dir": str(tmp),
        "loss_balance": {"per_phase_velocity_scale": True},   # S1 -> pg2 varies by phase
        "data": {"train_cases": [1, 7], "phases": ["systolic", "diastolic"],
                 "velocity_kinds": ["XZ"]},
        "model": {"hidden_dim": 32, "num_layers": 2, "num_frequencies": 8,
                  "use_param_encoder": True, "param_encoder_dims": [8, 8],
                  "nut": {"hidden_dim": 16, "num_layers": 2, "initial_nut": 0.02}},
        "physics": {"mu": 0.0035, "rho": 1060.0, "n_collocation": 200},
        "loss_weights": {c: 1.0 for c in COMPONENTS},
        "adaptive_weights": {"enabled": False},
        "training": {"epochs": 10, "lr": 1e-4},
        "loaders": {"max_velocity_points": 1500, "volumetric_collocation": True,  # S2
                    "inlet_n_radial": 4, "inlet_n_angular": 6},
    }


@pytest.mark.skipif(not _have_data(), reason="processed dataset not available")
def test_batched_losses_match_reference(tmp_path):
    trainer = Trainer(_tiny_cfg(tmp_path))
    ref = trainer._component_losses_reference()
    new = trainer._component_losses()

    assert set(ref) == set(new) == set(COMPONENTS)
    for c in COMPONENTS:
        assert torch.allclose(new[c], ref[c], atol=1e-6, rtol=1e-5), (
            f"{c}: batched={float(new[c]):.6e} reference={float(ref[c]):.6e}")
    # Every component must be a real, finite, differentiable scalar (not a dropped 0).
    for c in COMPONENTS:
        assert torch.isfinite(new[c]) and new[c].requires_grad


@pytest.mark.skipif(not _have_data(), reason="processed dataset not available")
def test_physics_only_matches_physics_component(tmp_path):
    trainer = Trainer(_tiny_cfg(tmp_path))
    ref_phys = trainer._component_losses_reference()["physics"]
    only = trainer._physics_only()
    assert torch.allclose(only, ref_phys, atol=1e-6, rtol=1e-5), (
        f"physics_only={float(only):.6e} reference={float(ref_phys):.6e}")
