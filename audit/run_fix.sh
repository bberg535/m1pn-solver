#!/usr/bin/env bash
# Reproduces every number of docs/BOUNDARY_FIXES.md (about 25 minutes on one core).
# Legacy mode (AUDIT_FIX=legacy) = all fix flags off = commit 8efe517 behaviour, bit for bit.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R="$HERE/results/fix"; B="$HERE/build"
mkdir -p "$R"
bash "$HERE/build.sh" runs ops c6 energy_random c1_probe tests/element_weno
bash "$HERE/build_prod_tests.sh" tests/element_weno tests/element_weno_fixes
# repository tests on unpatched production code
"$B/prod_element_weno"       > "$R/existing_tests_new.txt"
"$B/prod_element_weno_fixes" > "$R/fix_tests.txt"
# C1 high precision
"$B/audit_c1_probe" > "$HERE/results/c1_probe.jsonl"
python3 "$HERE/c1_highprecision.py" "$HERE/results/c1_probe.jsonl" > "$HERE/results/c1_highprecision.jsonl"
# C6
"$B/audit_c6" > "$R/c6.jsonl"
# T1/T2/T3/T8 and crossing 64^2, P7, p=1, T=0.32 (legacy / new / symmetric residual)
for mode in legacy new; do
  AUDIT_FIX=$mode "$B/audit_runs" small > "$R/small_$mode.jsonl"
  AUDIT_DUMP="$R/dens_$mode" AUDIT_FIX=$mode "$B/audit_runs" crossing 64 1 7 > "$R/crossing64_$mode.jsonl"
done
AUDIT_DUMP="$R/dens_sym" AUDIT_FIX=sym=1 "$B/audit_runs" crossing 64 1 7 > "$R/crossing64_sym.jsonl"
# operator-level audit (A, B, T4/T5, sensor scaling) with the fixes
AUDIT_FIX=new "$B/audit_ops" all > "$R/ops_new.jsonl"
AUDIT_FIX=new "$B/audit_energy_random" > "$R/energy_random_new.jsonl"
python3 "$HERE/compare_fix.py"
