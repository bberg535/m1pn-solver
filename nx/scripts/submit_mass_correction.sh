#!/usr/bin/env bash
# Standard B only; never reuse a prepared biharmonic campaign.
set -Eeuo pipefail
TASK_NX=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ "${1:-}" == --help || $# == 0 ]]; then
  cat <<'USAGE'
Usage: bash submit_mass_correction.sh [--dry-run] prepare production|smoke NEW_CAMPAIGN
       bash submit_mass_correction.sh [--dry-run] run CAMPAIGN [CONCURRENCY [INDICES]]
       bash submit_mass_correction.sh [--dry-run] collect CAMPAIGN [ARRAY_JOB_ID]
Method: R2, m1pn_mass_correction (B), lumped mass, PN viscosity scale 1, SSPRK3.
For the complete dependent production chain: python3 scripts/start_mass_correction.py
USAGE
  exit 0
fi
TASK_ARGS=("$@")
TASK_INDEX=0
if [[ "${TASK_ARGS[0]}" == --dry-run ]]; then TASK_INDEX=1; fi
TASK_STAGE=${TASK_ARGS[$TASK_INDEX]:-}
case "$TASK_STAGE" in
  prepare) TASK_PLAN="$TASK_NX/study_plan.json" ;;
  run|collect)
    TASK_CAMPAIGN=${TASK_ARGS[$((TASK_INDEX+1))]:?Missing prepared campaign}
    TASK_PLAN="$TASK_CAMPAIGN/manifest.json"
    ;;
  *) exec bash "$TASK_NX/scripts/submit_nx.sh" "${TASK_ARGS[@]}" ;;
esac
"${PN_AFC_PYTHON:-python3}" - "$TASK_PLAN" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
method = data.get("plan", data)["method"]
expected = {"model": "r2", "stabilization": "m1pn_mass_correction",
            "mass_mode": "lumped", "mass_corrections": 0, "integrator": "ssprk3",
            "pn_viscosity_scale": 1.0, "biharmonic_gamma": 0.0, "kv_gamma": 0.0}
if any(method.get(key) != value for key, value in expected.items()):
    raise SystemExit("Expected standard B with lumped mass, SSPRK3, PN viscosity 1 and zero biharmonic/KV coefficients.")
PY
exec bash "$TASK_NX/scripts/submit_nx.sh" "${TASK_ARGS[@]}"
