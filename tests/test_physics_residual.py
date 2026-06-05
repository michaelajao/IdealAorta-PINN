"""Tests for the autograd physics residuals.

Validates the wiring with analytic fields (no networks): a divergence-free field
must give ~zero continuity residual; a divergent field must not.
"""

from __future__ import annotations

import torch

from idealaorta_pinn.pinn.physics import compute_residuals


def _coords(n=256):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, 1, generator=g, requires_grad=True)
    y = torch.rand(n, 1, generator=g, requires_grad=True)
    z = torch.rand(n, 1, generator=g, requires_grad=True)
    params = torch.zeros(n, 0)  # no parameters needed for the analytic test
    return x, y, z, params


# The ``0 * other_coords`` terms keep every field structurally dependent on
# x, y and z so that all autograd cross-derivatives are defined (real networks
# always use every input; only these toy fields would otherwise be degenerate).
def _dep(nin):
    """Zero contribution that still ties a field to all three coordinates."""
    return 0.0 * (nin[:, 0:1] + nin[:, 1:2] + nin[:, 2:3])


def test_divergence_free_field_has_zero_continuity():
    x, y, z, params = _coords()
    # u=y, v=z, w=x  -> div = u_x + v_y + w_z = 0
    fields = {
        "u": lambda nin: nin[:, 1:2] + _dep(nin),
        "v": lambda nin: nin[:, 2:3] + _dep(nin),
        "w": lambda nin: nin[:, 0:1] + _dep(nin),
        "p": lambda nin: _dep(nin),
    }
    r = compute_residuals(fields, x, y, z, params, Re=100.0)
    assert r["res_cont"].abs().mean().item() < 1e-5


def test_divergent_field_has_nonzero_continuity():
    x, y, z, params = _coords()
    # u=x, v=0, w=0 -> div = 1
    fields = {
        "u": lambda nin: nin[:, 0:1] + _dep(nin),
        "v": lambda nin: _dep(nin),
        "w": lambda nin: _dep(nin),
        "p": lambda nin: _dep(nin),
    }
    r = compute_residuals(fields, x, y, z, params, Re=100.0)
    assert abs(r["res_cont"].abs().mean().item() - 1.0) < 1e-4
