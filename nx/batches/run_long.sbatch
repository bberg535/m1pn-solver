#!/bin/bash -l
#SBATCH --job-name=pn-mc-long
#SBATCH --constraint=nx
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=tmp/slurm_logs/pn-mc-long-%A_%a.log
set -Eeuo pipefail
TASK_ROOT=${PN_AFC_ROOT:?Use nx/scripts/submit_long_tasks.py to select this checkout}
exec bash "$TASK_ROOT/nx/scripts/run_task_nx.sh" "$@"
