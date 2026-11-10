#!/bin/bash -l
#SBATCH --job-name=ew-cmp
#SBATCH --constraint=nx
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
W=/home/warehouse15/bberg/phd/03_projects/M1PN_MassCorrection_20261104/.claude/worktrees/element-weno
CAMPAIGN=${1:?campaign directory}
source "$W/nx/scripts/nx_common.sh"
nx_setup
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CAMPAIGN/tasks.tsv")
NAME=${LINE%%$'\t'*}
ARGS=${LINE#*$'\t'}
OUT=$CAMPAIGN/runs/$NAME
if [[ -f $OUT/summary.json ]]; then echo "skip $NAME"; exit 0; fi
rm -rf "$OUT"
mkdir -p "$CAMPAIGN/runs"
echo "$NAME: $ARGS"
# shellcheck disable=SC2086
"$W/tmp/build_nx/m1pn_fast" $ARGS --threads "$SLURM_CPUS_PER_TASK" --output "$OUT"
rm -rf "$TMPDIR"
echo TASK_DONE
