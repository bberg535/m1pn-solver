#!/usr/bin/env bash
#SBATCH --job-name=tardis-blockA-filt
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_blockA_filt_20261120/slurm-%j.log
# Block A (tab:lattice:dynamicpn) with the merged perf/solver-202611 code, now with the constant angular filter
# (sigma_f T = 16.6, as tab:lattice) in runs/ (dyn_*, read by tardis_paper_report.py), and the same configurations
# without filter in runs_nofilt/ for comparison. Restartable.
set -uo pipefail
REPO=/home/user/bberg/phd/03_projects/M1PN_Solver
CAMPAIGN=$REPO/nx/tmp/tardis_blockA_filt_20261120
mkdir -p "$CAMPAIGN/runs" "$CAMPAIGN/runs_nofilt"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-blockA-filt start $(date) on $(hostname) ==="
git -C "$REPO" rev-parse --short HEAD
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU=1 BUILD_JOBS=32 bash "$REPO/scripts/build.sh" "$CAMPAIGN/build_gpu" > "$CAMPAIGN/build_gpu.log" 2>&1 || {
    echo "BUILD FAILED"; tail -n 40 "$CAMPAIGN/build_gpu.log"; exit 1; }
BIN="$CAMPAIGN/build_gpu/m1pn_fast"
sha256sum "$BIN" > "$CAMPAIGN/binary.sha256"
(cd "$CAMPAIGN/build_gpu" && ctest --output-on-failure) > "$CAMPAIGN/ctest.log" 2>&1
CRC=$?; echo "ctest rc=$CRC ($(date +%T))"
[ $CRC -eq 0 ] || { echo "CTEST FAILED"; tail -n 40 "$CAMPAIGN/ctest.log"; exit 1; }
bash "$REPO/tests/gpu_validate_adaptive.sh" "$BIN" "$CAMPAIGN/gpu_validate_adaptive" > "$CAMPAIGN/gpu_validate_adaptive.log" 2>&1
VRC=$?; tail -n 1 "$CAMPAIGN/gpu_validate_adaptive.log"
[ $VRC -eq 0 ] || { echo "ADAPTIVE GPU VALIDATION FAILED (rc=$VRC)"; exit 1; }
RUNS=$CAMPAIGN/runs
EXTRA=""
run_gpu() {
    local name=$1; shift
    if [ -f "$RUNS/$name/summary.json" ]; then echo "[skip] $name"; return 0; fi
    rm -rf "$RUNS/$name"; local t0=$(date +%s)
    "$BIN" --threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 65536 --gpu \
        "$@" $EXTRA --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    echo "[done] $(basename $RUNS)/$name rc=$? $(( $(date +%s) - t0 ))s $(date +%T)"
}
blockA() {
LAT_ADAPT="--case lattice-paul --inflow vacuum --dimension 2 --elements 7 --degree 1 --background 1e-10 --fcs analytic --break-units on --split-breaks on --mirror-reduction auto --time 3.2"
GBQ="--cells-per-unit 8,16,32,32,32,16,8"
GBH="--cells-per-unit 16,32,64,64,64,32,16"
GB="--cells-per-unit 32,64,128,128,128,64,32"

# GBQ (224^2 central)
run_gpu dyn_gbq_p21_uni       $LAT_ADAPT $GBQ --pn-order 21
run_gpu dyn_gbq_p21_t16       $LAT_ADAPT $GBQ --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 16
run_gpu dyn_gbq_p21_t8        $LAT_ADAPT $GBQ --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8
run_gpu dyn_gbq_p21_t8_f01    $LAT_ADAPT $GBQ --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8 --adaptive-floor 0.1
run_gpu dyn_gbq_p35_uni       $LAT_ADAPT $GBQ --pn-order 35
run_gpu dyn_gbq_p35_t16       $LAT_ADAPT $GBQ --pn-order 35 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 16
run_gpu dyn_gbq_p35_t8        $LAT_ADAPT $GBQ --pn-order 35 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8
run_gpu dyn_gbq_p35_1e4_t8    $LAT_ADAPT $GBQ --pn-order 35 --adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 8
run_gpu dyn_gbq_p35_t8_f01    $LAT_ADAPT $GBQ --pn-order 35 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8 --adaptive-floor 0.1

# GBH (448^2 central)
run_gpu dyn_gbh_p21_uni       $LAT_ADAPT $GBH --pn-order 21
run_gpu dyn_gbh_p21_t16       $LAT_ADAPT $GBH --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 16
run_gpu dyn_gbh_p21_t8        $LAT_ADAPT $GBH --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8
run_gpu dyn_gbh_p21_t8_f01    $LAT_ADAPT $GBH --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8 --adaptive-floor 0.1
run_gpu dyn_gbh_p35_uni       $LAT_ADAPT $GBH --pn-order 35
run_gpu dyn_gbh_p35_t16       $LAT_ADAPT $GBH --pn-order 35 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 16
run_gpu dyn_gbh_p35_t8        $LAT_ADAPT $GBH --pn-order 35 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8
run_gpu dyn_gbh_p35_1e4_t8    $LAT_ADAPT $GBH --pn-order 35 --adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 8
run_gpu dyn_gbh_p35_1e4_t16   $LAT_ADAPT $GBH --pn-order 35 --adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 16
run_gpu dyn_gbh_p35_t8_f01    $LAT_ADAPT $GBH --pn-order 35 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8 --adaptive-floor 0.1

# GB (896^2 central; also provides the "GB, broken lines, split stage" rows of Table 8)
run_gpu dyn_gb_p21_uni        $LAT_ADAPT $GB --pn-order 21
run_gpu dyn_gb_p21_t8         $LAT_ADAPT $GB --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8
run_gpu dyn_gb_p21_t8_f01     $LAT_ADAPT $GB --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8 --adaptive-floor 0.1
run_gpu dyn_gb_p35_uni        $LAT_ADAPT $GB --pn-order 35
run_gpu dyn_gb_p35_1e4_t16    $LAT_ADAPT $GB --pn-order 35 --adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 16
run_gpu dyn_gb_p35_1e4_t8     $LAT_ADAPT $GB --pn-order 35 --adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 8
run_gpu dyn_gb_p35_t8_f01     $LAT_ADAPT $GB --pn-order 35 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8 --adaptive-floor 0.1

}
echo "=== with filter $(date +%T) ==="
RUNS=$CAMPAIGN/runs; EXTRA="--filter constant --filter-strength 5.2"; blockA
echo "=== without filter $(date +%T) ==="
RUNS=$CAMPAIGN/runs_nofilt; EXTRA=""; blockA
python3 "$REPO/scripts/element_weno/tardis_paper_report.py" "$CAMPAIGN" > "$CAMPAIGN/REPORT.md" 2>&1
echo "=== tardis-blockA-filt finished $(date) ==="
