#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ctest --test-dir "${1:-$TASK_ROOT/tmp/build}" --output-on-failure
