#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TASK_CONFIG=${MFEM_CONFIG:-/home/warehouse15/bberg/spack/opt/spack/linux-zen3/mfem-4.8.0-ttr6kagekrts4je7mczrxuvlxw5564se/share/mfem/config.mk}
bash "$TASK_ROOT/scripts/build_existing.sh" "$TASK_CONFIG" "${1:-$TASK_ROOT/tmp/build}"
