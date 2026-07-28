#!/bin/bash
# Sequential training driver with a GPU memory guard before every run.
#
#   scripts/queue.sh stageA_case5_insample stageA_case6_insample
#   NEED_MIB=42000 scripts/queue.sh stageB_richerloo_f16_rich
#   GPU=0 scripts/queue.sh stageB_kfold_hold2p0
#
# Each argument is a config basename under configs/ (no .yaml). A job waits until the
# GPU actually has NEED_MIB free rather than racing an in-flight run into an OOM.
# Measured peaks: in-sample ~12 GB, LODO (trimmed) ~23 GB, the rich LODO arm ~40 GB.
set -u
cd "$(dirname "$0")/.." || exit 1

GPU=${GPU:-1}
NEED_MIB=${NEED_MIB:-26000}
PY=${PY:-~/miniconda3/envs/deep_tf/bin/python}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ $# -gt 0 ] || { echo "usage: [GPU=n] [NEED_MIB=n] $0 <config> [<config> ...]" >&2; exit 2; }

free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU"; }

fail=0
for cfg in "$@"; do
  echo ""
  echo "########## [$(date +%F' '%H:%M:%S)] QUEUED $cfg (needs ${NEED_MIB} MiB) ##########"
  while [ "$(free_mib)" -lt "$NEED_MIB" ]; do
    echo "  [$(date +%H:%M:%S)] waiting: need ${NEED_MIB} MiB, have $(free_mib) MiB free"
    sleep 300
  done
  echo "########## [$(date +%F' '%H:%M:%S)] START $cfg (free=$(free_mib) MiB) ##########"
  $PY scripts/run.py --config "configs/$cfg.yaml" --device cuda --no-interactive
  rc=$?          # capture IMMEDIATELY: any command substitution would clobber $?
  if [ $rc -eq 0 ]; then
    echo "########## [$(date +%F' '%H:%M:%S)] OK $cfg ##########"
  else
    echo "########## [$(date +%F' '%H:%M:%S)] FAILED $cfg exit=$rc -- see report/logs/$cfg/run.log ##########"
    fail=1
  fi
  sleep 20
done
echo ""
echo "########## QUEUE COMPLETE $(date) ##########"
exit $fail
