#!/usr/bin/env bash
#SBATCH --job-name=tardis-gpumem3d3
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=460G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_gpumem3d3_20261121/slurm-%j.log
# perf/init3d2: lazy datum buffer db, per-point FCS arrays freed after the nodal build. Bitwise G3Q P15/P35
# adaptive; G3H P21 adaptive (OOM before, job 164641) and 216^3 P3 memory (one step).
set -uo pipefail
REPO=/home/user/bberg/phd/03_projects/M1PN_Solver/.claude/worktrees/init3d2
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_gpumem3d3_20261121
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-gpumem3d start $(date) on $(hostname) ==="
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
run_gpu() {
    local name=$1; shift
    rm -rf "$RUNS/$name"; local t0=$(date +%s)
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "$RUNS/$name.gpumem" 2>/dev/null &
    local SMI=$!
    M1PN_CTOR_TIMING=1 "$BIN" $COMMON "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    local rc=$?
    kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
    echo "[done] $name rc=$rc $(( $(date +%s) - t0 ))s gpu_peak $(sort -n "$RUNS/$name.gpumem" | tail -n 1) MiB"
    grep -E "^gpu (create|ctor)|out of memory|coefficients" "$RUNS/$name.log"
    grep -o '"coefficients":[0-9]*' "$RUNS/$name/config.json" 2>/dev/null | head -1
}
G3Q="--elements 7 --cells-per-unit 4,8,16,16,16,8,4"
AD3="--adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8"
check() {
    python3 -c '
import json, sys
s = json.load(open(sys.argv[1]))
ok = (repr(s["final_density_mass"]), repr(s["absorption_rate_final"])) == (sys.argv[2], sys.argv[3])
print(sys.argv[1].split("/")[-2], "bitwise", ok, "init", round(s["initialization_seconds"], 1),
      "radius", round(s["gpu_timing"]["spectral_radius"], 2), "total", round(s["total_seconds"], 1))
' "$RUNS/$1/summary.json" "$2" "$3"
}
run_gpu g3q_p15_ad $L3 $G3Q --pn-order 15 $AD3
check g3q_p15_ad 0.2633599056799755 0.3650589396907601
run_gpu g3q_p35_ad $L3 $G3Q --pn-order 35 $AD3
check g3q_p35_ad 0.2634203950569552 0.36494916485137036
G3H="--elements 7 --cells-per-unit 8,16,24,24,24,16,8"
run_gpu g3h_p21_ad $L3 $G3H --pn-order 21 $AD3
check g3h_p21_ad x x
run_gpu m216_p3 $L3 --elements 7 --cells-per-unit 16,32,40,40,40,32,16 --pn-order 3 --max-steps 1
echo "=== tardis-gpumem3d3 finished $(date) ==="
