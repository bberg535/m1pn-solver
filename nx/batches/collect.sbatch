#!/bin/bash -l
#SBATCH --job-name=pn-mc-collect
#SBATCH --constraint=nx
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=tmp/slurm_logs/pn-mc-collect-%j.log
set -Eeuo pipefail
TASK_ROOT=${PN_AFC_ROOT:?Use nx/scripts/submit_nx.sh to select this checkout}
exec bash "$TASK_ROOT/nx/scripts/collect_nx.sh" "$@"
