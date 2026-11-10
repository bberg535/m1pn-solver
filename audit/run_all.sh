#!/usr/bin/env bash
# Reproduces every number of audit/REPORT.md. Needs the main build's libm1pn_core.a (see build.sh, MAINBUILD).
# One core, deterministic. Total runtime on one core: about 15 minutes.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash "$HERE/build.sh" ops runs probe probe2 energy_random
R="$HERE/results"; B="$HERE/build"
"$B/audit_ops" all                        > "$R/ops.jsonl"
"$B/audit_energy_random"                  > "$R/energy_random.jsonl"
"$B/audit_probe"                          > "$R/probe.jsonl"
"$B/audit_probe2"                         > "$R/probe2.jsonl"
"$B/audit_runs" small                     > "$R/small.jsonl"
"$B/audit_runs" crossing 64 1 7           > "$R/crossing64_p1_P7.jsonl"
T6_PN=3 "$B/audit_runs" T6                > "$R/T6_pn3.jsonl"
"$B/audit_runs" T7                        > "$R/T7.jsonl"
python3 "$HERE/make_summary.py"
