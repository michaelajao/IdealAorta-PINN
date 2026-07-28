#!/bin/bash
# Regenerate the rotatable 3D Plotly HTML for every reported run.
#
# The training queues all passed --no-interactive, so only the one run that was
# launched by hand (stageA_case1_s12) ever got its HTML. This is inference-only:
# --skip-train reuses each checkpoint (and, unlike a real run, does NOT archive the
# existing outputs), --no-figures leaves the static PNGs untouched.
#
#   scripts/regen_interactive.sh                 # all runs
#   scripts/regen_interactive.sh stageB_kfold_hold2p0   # a subset
set -u
cd "$(dirname "$0")/.." || exit 1

GPU=${GPU:-1}
PY=${PY:-~/miniconda3/envs/deep_tf/bin/python}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ALL=(stageA_case1_s12
     stageA_case2_insample  stageA_case3_insample  stageA_case4_insample
     stageA_case5_insample  stageA_case6_insample  stageA_case7_insample
     stageA_case8_insample  stageA_case9_insample  stageA_case10_insample
     stageA_case11_insample stageA_case12_insample
     stageB_kfold_hold2p0 stageB_richerloo_f16 stageB_kfold_hold2p6
     stageB_richerloo_f16_rich)
CFGS=("$@"); [ $# -gt 0 ] || CFGS=("${ALL[@]}")

fail=0
for cfg in "${CFGS[@]}"; do
  echo ""
  echo "########## [$(date +%F' '%H:%M:%S)] INTERACTIVE $cfg ##########"
  $PY scripts/run.py --config "configs/$cfg.yaml" --skip-train --no-figures \
      --device cuda
  rc=$?          # capture IMMEDIATELY: any command substitution would clobber $?
  if [ $rc -eq 0 ]; then
    echo "########## [$(date +%F' '%H:%M:%S)] OK $cfg ##########"
  else
    echo "########## [$(date +%F' '%H:%M:%S)] FAILED $cfg exit=$rc ##########"
    fail=1
  fi
done
echo ""
echo "########## INTERACTIVE REGEN COMPLETE $(date) ##########"
exit $fail
