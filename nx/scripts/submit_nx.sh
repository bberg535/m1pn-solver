#!/usr/bin/env bash
# This is the only submission helper. --dry-run never calls sbatch.
set -Eeuo pipefail
TASK_NX=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
TASK_ROOT=$(cd "$TASK_NX/.." && pwd -P)
TASK_PYTHON=${PN_AFC_PYTHON:-python3}
TASK_DRY=0
if [[ "${1:-}" == --dry-run ]]; then TASK_DRY=1; shift; fi
TASK_STAGE=${1:?Usage: submit_nx.sh [--dry-run] prepare PROFILE CAMPAIGN | run CAMPAIGN [CONCURRENCY [INDICES]] | collect CAMPAIGN [ARRAY_JOB_ID]}
shift
TASK_EXTRA=()
case "$TASK_STAGE" in
  prepare)
    TASK_PROFILE=${1:?Missing production|smoke}; TASK_CAMPAIGN=${2:?Missing new campaign}; shift 2
    [[ "$TASK_PROFILE" == production || "$TASK_PROFILE" == smoke ]] || { echo 'Unknown profile' >&2; exit 2; }
    TASK_ARGS=("$TASK_PROFILE" "$TASK_CAMPAIGN" "$@")
    [[ ! -e "$TASK_CAMPAIGN" ]] || { echo 'Campaign exists; choose a new output directory.' >&2; exit 2; }
    ;;
  run)
    TASK_CAMPAIGN=${1:?Missing prepared campaign}; TASK_PARALLEL=${2:-2}; TASK_INDICES=${3:-}
    [[ "$TASK_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo 'Concurrency must be positive.' >&2; exit 2; }
    TASK_COUNT=$("$TASK_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$TASK_CAMPAIGN/tasks.json")
    (( TASK_COUNT > 0 )) || { echo 'No tasks.' >&2; exit 2; }
    [[ -f "$TASK_CAMPAIGN/manifest.json" ]] || { echo 'Preparation is incomplete.' >&2; exit 2; }
    TASK_MAX=$("$TASK_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan"]["execution"]["max_concurrent_tasks"])' "$TASK_CAMPAIGN/manifest.json")
    (( TASK_PARALLEL <= TASK_MAX )) || { echo "Frozen plan permits at most $TASK_MAX concurrent tasks." >&2; exit 2; }
    TASK_THREADS=$("$TASK_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["threads"])' "$TASK_CAMPAIGN/manifest.json")
    TASK_EXTRA+=("--cpus-per-task=$TASK_THREADS")
    TASK_INDICES=${TASK_INDICES:-"0-$((TASK_COUNT-1))"}
    "$TASK_PYTHON" -c 'import re,sys; s=sys.argv[1]; n=int(sys.argv[2]); assert re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*",s),"Invalid array indices"; bounds=[list(map(int,p.split("-"))) for p in s.split(",")]; assert all(0<=b[0]<=b[-1]<n for b in bounds),"Array index out of range"' "$TASK_INDICES" "$TASK_COUNT"
    TASK_EXTRA+=("--array=$TASK_INDICES%$TASK_PARALLEL")
    TASK_ARGS=("$TASK_CAMPAIGN")
    ;;
  collect)
    TASK_CAMPAIGN=${1:?Missing prepared campaign}; TASK_ARRAY_JOB=${2:-}
    [[ -f "$TASK_CAMPAIGN/manifest.json" ]] || { echo 'Preparation is incomplete.' >&2; exit 2; }
    if [[ -n "$TASK_ARRAY_JOB" ]]; then
      [[ "$TASK_ARRAY_JOB" =~ ^[0-9]+$ ]] || { echo 'Expected numeric array job ID.' >&2; exit 2; }
      TASK_EXTRA+=("--dependency=afterany:$TASK_ARRAY_JOB")
    fi
    TASK_ARGS=("$TASK_CAMPAIGN")
    ;;
  *) echo "Unknown stage: $TASK_STAGE" >&2; exit 2 ;;
esac
TASK_CAMPAIGN=$("$TASK_PYTHON" -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]).resolve(); p.relative_to(pathlib.Path(sys.argv[2]).resolve()); print(p)' "$TASK_CAMPAIGN" "$TASK_NX")
# Use an absolute campaign path inside the compute job, independent of cwd.
if [[ "$TASK_STAGE" == prepare ]]; then TASK_ARGS[1]=$TASK_CAMPAIGN; else TASK_ARGS[0]=$TASK_CAMPAIGN; fi
TASK_LOGS="$TASK_NX/tmp/slurm_logs"
TASK_COMMAND=(sbatch --parsable "--chdir=$TASK_NX" "--export=ALL,PN_AFC_ROOT=$TASK_ROOT"
              "--output=$TASK_LOGS/pn-mc-$TASK_STAGE-%A_%a.log" "${TASK_EXTRA[@]}"
              "$TASK_NX/batches/$TASK_STAGE.sbatch" "${TASK_ARGS[@]}")
if (( TASK_DRY )); then printf '%q ' "${TASK_COMMAND[@]}"; printf '\n'; exit 0; fi
command -v sbatch >/dev/null || { echo 'sbatch is not available; use --dry-run locally.' >&2; exit 2; }
mkdir -p "$TASK_LOGS"
printf '%q ' "${TASK_COMMAND[@]}" >&2; printf '\n' >&2
TASK_JOB=$("${TASK_COMMAND[@]}")
printf '%s\n' "${TASK_JOB%%;*}"
