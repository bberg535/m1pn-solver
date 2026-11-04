#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$TASK_ROOT/tmp/build/m1pn" --mcl-mode lazy --geometry-cache on "$@"
