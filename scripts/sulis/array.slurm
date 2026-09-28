#!/bin/bash
# Slurm job array over a file of scripts/reconstruct.py job lines (the Sulis counterpart
# of scripts/queue_jobs.sh): array task i runs line i of JOBFILE.
#
#   python scripts/reconstruct.py jobs confirm --kind train > jobs_train.txt
#   mkdir -p report/logs/reconstruction/slurm        # Slurm does not create the --output dir
#   sbatch --array=1-$(grep -c . jobs_train.txt)%40 scripts/sulis/array.slurm jobs_train.txt
#
# CPU-only work (interpolation baselines, audits) goes to the CPU partition instead:
#   sbatch --partition=compute --gres=none --account=su003 --cpus-per-task=8 \
#          --array=1-N scripts/sulis/array.slurm jobs_baseline.txt
#
# One log per task under report/logs/reconstruction/, and one line per finished task,
# with its real exit code, in report/logs/reconstruction/done.txt.
#SBATCH --job-name=recon
#SBATCH --account=su003-csmm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=7500
#SBATCH --time=04:00:00
#SBATCH --output=report/logs/reconstruction/slurm/%A_%a.out
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1

JOBS=${1:?usage: sbatch --array=1-N $0 JOBFILE}
line=$(grep . "$JOBS" | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -z "$line" ] && { echo "no line ${SLURM_ARRAY_TASK_ID} in $JOBS"; exit 1; }

source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate "${ENV:-idealaorta}"

LOG_DIR=report/logs/reconstruction
tag=$(echo "$line" | tr ' ' '_' | tr -d '-')
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \
  srun python -u -W ignore scripts/reconstruct.py $line > "$LOG_DIR/$tag.log" 2>&1
rc=$?          # capture immediately: a command substitution would clobber $?
echo "$(date -u +%FT%TZ) exit=$rc job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $line" >> "$LOG_DIR/done.txt"
exit $rc
