#!/usr/bin/env bash
#SBATCH --job-name=tardis-lowN
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_lowN_20261120/slurm-%j.log
# Point 4 (lower N through FCS + filter): graded lattice meshes with broken lines, split stage, mirror and FCS as in
# block A, now with the constant angular filter (sigma_f T = 16.6, as tab:lattice), P21 / P35, uniform and adaptive.
# Binary of tardis_perf4_20261120 (perf/solver-202611). Restartable.
set -uo pipefail
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_lowN_20261120
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"
BIN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_perf4_20261120/build_gpu/m1pn_fast
[ -x "$BIN" ] || { echo "binary missing"; exit 1; }
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-lowN start $(date) ==="
COMMON="--threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 65536 --gpu"
run_gpu() {
    local name=$1; shift
    if [ -f "$RUNS/$name/summary.json" ]; then echo "[skip] $name"; return 0; fi
    rm -rf "$RUNS/$name"; local t0=$(date +%s)
    "$BIN" $COMMON "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    echo "[done] $name rc=$? $(( $(date +%s) - t0 ))s $(date +%T)"
}
LAT_ADAPT="--case lattice-paul --inflow vacuum --dimension 2 --elements 7 --degree 1 --background 1e-10 --fcs analytic --break-units on --split-breaks on --mirror-reduction auto --time 3.2"
FILT="--filter constant --filter-strength 5.2"
GBH="--cells-per-unit 16,32,64,64,64,32,16"
GB="--cells-per-unit 32,64,128,128,128,64,32"
AD="--adaptive-pn on --adaptive-tile 8"
for m in GB GBH; do
    mesh=${!m}; lm=${m,,}
    run_gpu ${lm}_p21_filt          $LAT_ADAPT $mesh --pn-order 21 $FILT
    run_gpu ${lm}_p21_filt_ad1e4    $LAT_ADAPT $mesh --pn-order 21 $FILT $AD --adaptive-tol 1e-4
    run_gpu ${lm}_p21_filt_f01      $LAT_ADAPT $mesh --pn-order 21 $FILT $AD --adaptive-tol 3e-4 --adaptive-floor 0.1
    run_gpu ${lm}_p35_filt          $LAT_ADAPT $mesh --pn-order 35 $FILT
    run_gpu ${lm}_p35_filt_ad1e4    $LAT_ADAPT $mesh --pn-order 35 $FILT $AD --adaptive-tol 1e-4
    run_gpu ${lm}_p15_filt          $LAT_ADAPT $mesh --pn-order 15 $FILT
done
echo "=== tardis-lowN finished $(date) ==="
