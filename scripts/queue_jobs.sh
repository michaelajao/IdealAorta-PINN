#!/bin/bash
# Run a file of scripts/reconstruct.py job lines with bounded concurrency.
#
#   python scripts/reconstruct.py jobs confirm --kind train > jobs_train.txt
#   scripts/queue_jobs.sh jobs_train.txt 6 gpu            # GPUs from $GPUS (default "0 1"), round-robin
#   scripts/queue_jobs.sh jobs_baseline.txt 3 cpu         # CPU-only (interpolation, audits)
#
# Each line is the argument list of scripts/reconstruct.py (e.g. "train --arm cont ...").
# One log per job under report/logs/reconstruction/, and one line per finished job,
# with its real exit code, in report/logs/reconstruction/done.txt.
set -u
cd "$(dirname "$0")/.." || exit 1

JOBS=${1:?usage: $0 JOBFILE [NPAR] [gpu|cpu]}
NPAR=${2:-4}
MODE=${3:-gpu}
PY=${PY:-python}
read -r -a GPU_LIST <<< "${GPUS:-0 1}"
LOG_DIR=report/logs/reconstruction
mkdir -p "$LOG_DIR"

i=0
while read -r line; do
  [ -z "$line" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do sleep 20; done
  if [ "$MODE" = gpu ]; then
    dev=${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}
  else
    dev=""
  fi
  i=$((i + 1))
  tag=$(echo "$line" | tr ' ' '_' | tr -d '-')
  (
    CUDA_VISIBLE_DEVICES=$dev OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} \
      $PY -u -W ignore scripts/reconstruct.py $line > "$LOG_DIR/$tag.log" 2>&1
    rc=$?          # capture immediately: a command substitution would clobber $?
    echo "$(date -u +%FT%TZ) exit=$rc $line" >> "$LOG_DIR/done.txt"
  ) &
  sleep 5
done < "$JOBS"
wait
echo "$(date -u +%FT%TZ) QUEUE FINISHED $JOBS" >> "$LOG_DIR/done.txt"
