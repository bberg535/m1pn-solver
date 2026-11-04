#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/nx_common.sh"
TASK_CAMPAIGN=${1:?Usage: run_task_nx.sh CAMPAIGN}
nx_inside "$TASK_CAMPAIGN" || nx_fail 'Campaign must remain in nx/.'
nx_setup
TASK_THREADS=$("$TASK_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["threads"])' "$TASK_CAMPAIGN/manifest.json")
[[ "$TASK_THREADS" == "$OMP_NUM_THREADS" ]] || nx_fail "Prepared for $TASK_THREADS threads, allocated $OMP_NUM_THREADS."
exec "$TASK_PYTHON" "$TASK_NX/scripts/study.py" run --campaign "$TASK_CAMPAIGN" \
  --task "${SLURM_ARRAY_TASK_ID:?This wrapper requires an array task}" --launcher srun
