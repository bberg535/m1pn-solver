#!/usr/bin/env bash
set -euo pipefail
TASK_NX=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
TASK_PYTHON=${PN_AFC_PYTHON:-python3}
TASK_EXE=${1:?Usage: smoke_local.sh /path/to/m1pn_mc_study NEW_OUTPUT}
TASK_OUTPUT=${2:?Usage: smoke_local.sh /path/to/m1pn_mc_study NEW_OUTPUT}
mkdir -p "$TASK_NX/tmp/test_tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$TASK_NX/tmp/test_tmp"
"$TASK_PYTHON" "$TASK_NX/scripts/study.py" prepare --exe "$TASK_EXE" \
  --profile smoke --output "$TASK_OUTPUT" --threads 1
TASK_COUNT=$("$TASK_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$TASK_OUTPUT/tasks.json")
for ((TASK_INDEX=0; TASK_INDEX<TASK_COUNT; ++TASK_INDEX)); do
  "$TASK_PYTHON" "$TASK_NX/scripts/study.py" run --campaign "$TASK_OUTPUT" \
    --task "$TASK_INDEX" --launcher direct --timeout 120
done
"$TASK_PYTHON" "$TASK_NX/scripts/study.py" collect --campaign "$TASK_OUTPUT"
