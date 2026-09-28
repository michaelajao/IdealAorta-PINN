"""Sparse-observation reconstruction of hidden hemodynamic fields (the revision study).

Given velocity at grid-sampled nodes in a short time window, recover the velocity
elsewhere, the gauge pressure and the aneurysm-wall shear, and ask what information
each target needs. Everything runs on the whole-domain exports
(``data/processed/full``, see ``idealaorta_pinn.data.full_export``).

Modules
-------
numerics        least-squares node gradients and the sparse gradient operator
problem         one reconstruction problem: case, window, observation grid, hidden targets
fields          space-time neural field and its RANS-mean residuals (autograd / torch.func)
postprocess     velocity -> pressure (momentum integration) and wall shear, for every method
interpolation   conventional comparators: linear, local RBF, validation-tuned and space-time RBF
reference       training-free audits on the CFD fields: momentum budget, pressure and WSS oracles
training        fit one neural-field arm and score it on the hidden targets
study           experiment matrices from configs/reconstruction/*.yaml, selection and analyses

The command-line front end is ``scripts/reconstruct.py``. Run names keep the ``rev2_``
experiment prefix of the revision study (like ``stageA_`` / ``stageB_`` of the original).
"""

from ..config import load_constants

_FLUID = load_constants()["fluid"]
RHO: float = float(_FLUID["rho"])      # blood density (kg/m^3)
MU: float = float(_FLUID["mu"])        # molecular dynamic viscosity (Pa s)
