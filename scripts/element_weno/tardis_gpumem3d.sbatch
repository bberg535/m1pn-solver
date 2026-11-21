#!/usr/bin/env bash
#SBATCH --job-name=tardis-gpumem3d
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=460G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_gpumem3d_20261121/slurm-%j.log
# Device memory per GPU-constructor phase (M1PN_CTOR_TIMING) on graded 3D meshes, one step: 168^3 P3/P7, 216^3 P3.
set -uo pipefail
REPO=/home/user/bberg/phd/03_projects/M1PN_Solver/.claude/worktrees/init3d2
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_gpumem3d_20261121
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-gpumem3d start $(date) on $(hostname) ==="
GPU=1 BUILD_JOBS=32 bash "$REPO/scripts/build.sh" "$CAMPAIGN/build_gpu" > "$CAMPAIGN/build_gpu.log" 2>&1 || {
    echo "BUILD FAILED"; tail -n 40 "$CAMPAIGN/build_gpu.log"; exit 1; }
BIN="$CAMPAIGN/build_gpu/m1pn_fast"
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
run_gpu m168_p3 $L3 --elements 7 --cells-per-unit 12,24,32,32,32,24,12 --pn-order 3 --max-steps 1
run_gpu m168_p7 $L3 --elements 7 --cells-per-unit 12,24,32,32,32,24,12 --pn-order 7 --max-steps 1
run_gpu m216_p3 $L3 --elements 7 --cells-per-unit 16,32,40,40,40,32,16 --pn-order 3 --max-steps 1
echo "=== tardis-gpumem3d finished $(date) ==="
