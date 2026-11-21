#!/usr/bin/env bash
#SBATCH --job-name=tardis-g3hp21
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=460G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_g3h_p21_20261121/slurm-%j.log
# experiment/element-weno after the perf/init3d2 merge: G3H P21 uniform and adaptive (paper step 4), FCS, broken lines,
# split, T = 3.2; G3Q P35 adaptive for the current init/total time.
set -uo pipefail
REPO=/home/user/bberg/phd/03_projects/M1PN_Solver
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_g3h_p21_20261121
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-g3hp21 start $(date) on $(hostname) ==="
git -C "$REPO" rev-parse --short HEAD
GPU=1 BUILD_JOBS=32 bash "$REPO/scripts/build.sh" "$CAMPAIGN/build_gpu" > "$CAMPAIGN/build_gpu.log" 2>&1 || {
    echo "BUILD FAILED"; tail -n 40 "$CAMPAIGN/build_gpu.log"; exit 1; }
BIN="$CAMPAIGN/build_gpu/m1pn_fast"
COMMON="--threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 400000 --gpu"
L3="--case lattice-3d --inflow vacuum --dimension 3 --degree 1 --background 1e-10 --mirror-reduction off --skip-fields --time 3.2 --fcs analytic --break-units on --split-breaks on"
G3Q="--elements 7 --cells-per-unit 4,8,16,16,16,8,4"
G3H="--elements 7 --cells-per-unit 8,16,24,24,24,16,8"
AD3="--adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8"
run() {
    local name=$1; shift
    rm -rf "$RUNS/$name"; local t0=$(date +%s)
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$RUNS/$name.gpumem" 2>/dev/null &
    local SMI=$!
    "$BIN" $COMMON "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    local rc=$?
    kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
    echo "[done] $name rc=$rc $(( $(date +%s) - t0 ))s gpu_peak $(sort -n "$RUNS/$name.gpumem" | tail -n 1) MiB"
    python3 -c '
import json, sys
s = json.load(open(sys.argv[1] + "/summary.json")); ad = s.get("adaptive_pn") or {}
print("   init %.1f solver %.1f total %.1f frac %.3f meanN %s mass %r rate %r rss %.1fGB" % (s["initialization_seconds"], s["solver_seconds"],
      s["total_seconds"], ad.get("mean_coeff_fraction", 1), ad.get("mean_order"), s["final_density_mass"], s["absorption_rate_final"], s["peak_rss_kib_rank0"]/1e6))
' "$RUNS/$name"
}
run g3h_p21_ad  $L3 $G3H --pn-order 21 $AD3
run g3h_p21_uni $L3 $G3H --pn-order 21
run g3q_p35_ad  $L3 $G3Q --pn-order 35 $AD3
run g3q_p21_ad  $L3 $G3Q --pn-order 21 $AD3
echo "=== tardis-g3hp21 finished $(date) ==="
