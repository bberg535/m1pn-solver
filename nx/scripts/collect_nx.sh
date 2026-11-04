#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/nx_common.sh"
TASK_CAMPAIGN=${1:?Usage: collect_nx.sh CAMPAIGN}
nx_inside "$TASK_CAMPAIGN" || nx_fail 'Campaign must remain in nx/.'
nx_setup
exec "$TASK_PYTHON" "$TASK_NX/scripts/study.py" collect --campaign "$TASK_CAMPAIGN"
