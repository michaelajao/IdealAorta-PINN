#!/bin/bash
# One-time setup of the project environment on Sulis (run on a login node, which has
# internet access; compute nodes may not).
#
#   bash scripts/sulis/setup.sh            # installs Miniforge (if absent) + env "idealaorta"
#
# A self-contained conda env is used rather than the PIP-PyTorch module so the stack
# matches brosnan's and does not depend on the module tree's toolchain pins.
set -euo pipefail
cd "$(dirname "$0")/../.."

CONDA_DIR=${CONDA_DIR:-$HOME/miniforge3}
ENV=${ENV:-idealaorta}

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  curl -fsSL -o /tmp/miniforge_$USER.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge_$USER.sh -b -p "$CONDA_DIR"
  rm -f /tmp/miniforge_$USER.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

conda env list | grep -q "^$ENV " || conda create -y -n "$ENV" python=3.11
conda activate "$ENV"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
echo "Environment ready: source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV"
