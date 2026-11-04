#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/nx_common.sh"
nx_setup
nx_build "${1:-$TASK_NX/tmp/builds/nx_$SLURM_JOB_ID}"
