#!/usr/bin/env bash
#SBATCH --job-name=tardis-fcs3d
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=460G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_fcs3d_20261121/slurm-%j.log
# perf/init3d2: (1) bitwise G3Q P15 adaptive after the parallel nodal gathers; (2) CPU ctor phases on 312^3 P3 (the GPU
# part runs out of memory there); (3) 3D FCS eval kernel variants M1PN_FCS3D_LB = 0/1/2 x M1PN_FCS3D_SORT = 0/1:
# G3Q P7 uniform full runs (result vs LB0/SORT0) and 168^3 P3 one step (FCS build time).
set -uo pipefail
REPO=/home/user/bberg/phd/03_projects/M1PN_Solver/.claude/worktrees/init3d2
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_fcs3d_20261121
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-fcs3d start $(date) on $(hostname) ==="
GPU=1 BUILD_JOBS=32 bash "$REPO/scripts/build.sh" "$CAMPAIGN/build_gpu" > "$CAMPAIGN/build_gpu.log" 2>&1 || {
    echo "BUILD FAILED"; tail -n 40 "$CAMPAIGN/build_gpu.log"; exit 1; }
BIN="$CAMPAIGN/build_gpu/m1pn_fast"
(cd "$CAMPAIGN/build_gpu" && ctest --output-on-failure) > "$CAMPAIGN/ctest.log" 2>&1
CRC=$?; echo "ctest rc=$CRC ($(date +%T))"
[ $CRC -eq 0 ] || { echo "CTEST FAILED"; tail -n 40 "$CAMPAIGN/ctest.log"; exit 1; }
bash "$REPO/tests/gpu_validate_adaptive.sh" "$BIN" "$CAMPAIGN/gpu_validate_adaptive" > "$CAMPAIGN/gpu_validate_adaptive.log" 2>&1
echo "validate rc=$? $(tail -n 1 "$CAMPAIGN/gpu_validate_adaptive.log")"
COMMON="--threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 400000 --gpu"
L3="--case lattice-3d --inflow vacuum --dimension 3 --degree 1 --background 1e-10 --mirror-reduction off --skip-fields --time 3.2 --fcs analytic --break-units on --split-breaks on"
G3Q="--elements 7 --cells-per-unit 4,8,16,16,16,8,4"
M168="--elements 7 --cells-per-unit 12,24,32,32,32,24,12"
AD3="--adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8"
run() {
    local name=$1; shift
    rm -rf "$RUNS/$name"; local t0=$(date +%s)
    env M1PN_CTOR_TIMING=1 "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    echo "[done] $name rc=$? $(( $(date +%s) - t0 ))s $(grep -h 'gpu ctor: reaction and FCS' "$RUNS/$name.log" | awk '{print "fcs", $5, "s"}')"
}
result() {
    python3 -c '
import json, sys
s = json.load(open(sys.argv[1])); L = s.get("lattice_fcs", {})
print(sys.argv[1].split("/")[-2], repr(s["final_density_mass"]), repr(s["absorption_rate_final"]), "table", round(L.get("table_seconds", 0), 2),
      "unc", repr(L.get("uncollided_mass_final")), "init", round(s["initialization_seconds"], 1))
' "$RUNS/$1/summary.json"
}
run g3q_p15_ad "$BIN" $COMMON $L3 $G3Q --pn-order 15 $AD3
result g3q_p15_ad; echo "  ref 0.2633599056799755 0.3650589396907601"
run m312_p3 "$BIN" $COMMON $L3 --elements 7 --cells-per-unit 24,48,56,56,56,48,24 --pn-order 3 --max-steps 1
grep -E "^ctor|^init: scheme" "$RUNS/m312_p3.log" | awk '{v=$(NF-1); if (v+0>0.3) print}'
for lb in 0 1 2; do for so in 0 1; do
    run g3q_p7_lb${lb}_s${so} M1PN_FCS3D_LB=$lb M1PN_FCS3D_SORT=$so "$BIN" $COMMON $L3 $G3Q --pn-order 7
    result g3q_p7_lb${lb}_s${so}
done; done
for lb in 0 1 2; do for so in 0 1; do
    run m168_p3_lb${lb}_s${so} M1PN_FCS3D_LB=$lb M1PN_FCS3D_SORT=$so "$BIN" $COMMON $L3 $M168 --pn-order 3 --max-steps 1
done; done
echo "=== tardis-fcs3d finished $(date) ==="
