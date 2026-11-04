#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/nx_common.sh"
TASK_PROFILE=${1:?Usage: prepare_nx.sh production|smoke NEW_CAMPAIGN}
TASK_CAMPAIGN=${2:?Usage: prepare_nx.sh production|smoke NEW_CAMPAIGN}
shift 2
nx_inside "$TASK_CAMPAIGN" || nx_fail 'Campaign must remain in nx/.'
[[ ! -e "$TASK_CAMPAIGN" ]] || nx_fail 'Campaign already exists; preserve it and choose a new name.'
nx_setup
nx_build "$TASK_NX/tmp/builds/nx_$SLURM_JOB_ID"
exec "$TASK_PYTHON" "$TASK_NX/scripts/study.py" prepare --exe "$TASK_BUILD/m1pn_mc_study" \
  --profile "$TASK_PROFILE" --output "$TASK_CAMPAIGN" --threads "$OMP_NUM_THREADS" "$@"
