#!/usr/bin/env bash
#SBATCH --job-name=tardis-3d-graded
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_3d_graded_20261120/slurm-%j.log

# lattice-3d on graded tensor meshes (A100), with the 2D improvements: FCS, broken material lines + split stage,
# dynamic adaptive PN. Uniform references (U56/U112 with FCS, same options) come from tardis_paper_20261120 block D/D2;
# U168 with FCS does not fit the A100 (FCS time table ~98 GB), the graded meshes reach that centre resolution:
#   G3Q 4,8,16,16,16,8,4      72^3 = 0.37 M elements, centre as U112  (table ~8 GB)
#   G3H 8,16,24,24,24,16,8   120^3 = 1.73 M elements, centre as U168  (table ~36 GB)
#   G3B 6,12,32,32,32,12,6   132^3 = 2.30 M elements, centre as U224  (table ~48 GB)
# Restartable: any run with an existing summary.json is skipped.
set -uo pipefail

REPO=/home/user/bberg/phd/03_projects/M1PN_Solver
CAMPAIGN=$REPO/nx/tmp/tardis_3d_graded_20261120
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1

echo "=== tardis-3d-graded start $(date) on $(hostname) ==="
git -C "$REPO" rev-parse --short HEAD
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

find "$REPO/src" -type f -exec touch {} +
GPU=1 BUILD_JOBS=32 bash "$REPO/scripts/build.sh" "$CAMPAIGN/build_gpu" > "$CAMPAIGN/build_gpu.log" 2>&1 || {
    echo "BUILD FAILED"; tail -n 40 "$CAMPAIGN/build_gpu.log"; exit 1; }
BIN="$CAMPAIGN/build_gpu/m1pn_fast"
sha256sum "$BIN" > "$CAMPAIGN/binary.sha256"
(cd "$CAMPAIGN/build_gpu" && ctest --output-on-failure) > "$CAMPAIGN/ctest.log" 2>&1
echo "ctest rc=$? ($(date +%T))"
bash "$REPO/tests/gpu_validate_adaptive.sh" "$BIN" "$CAMPAIGN/gpu_validate_adaptive" > "$CAMPAIGN/gpu_validate_adaptive.log" 2>&1
VRC=$?
tail -n 1 "$CAMPAIGN/gpu_validate_adaptive.log"
[ $VRC -eq 0 ] || { echo "ADAPTIVE GPU VALIDATION FAILED (rc=$VRC)"; exit 1; }

run_gpu() {
    local name=$1; shift
    if [ -f "$RUNS/$name/summary.json" ]; then echo "[skip] $name"; return 0; fi
    rm -rf "$RUNS/$name"
    local t0=$(date +%s)
    "$BIN" --threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low \
        --memory-mib 65536 --gpu "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    local rc=$?
    echo "[done] $name rc=$rc $(( $(date +%s) - t0 ))s $(date +%T)"
    return 0
}

L3="--case lattice-3d --inflow vacuum --dimension 3 --degree 1 --time 3.2 --background 1e-10 --mirror-reduction off --skip-fields --elements 7 --fcs analytic"
BRK3="--break-units on --split-breaks on"
AD3="--adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8"
G3Q="--cells-per-unit 4,8,16,16,16,8,4"
G3H="--cells-per-unit 8,16,24,24,24,16,8"
G3B="--cells-per-unit 6,12,32,32,32,12,6"

echo "=== G3Q $(date +%T) ==="
run_gpu l3d_g3q_p7_fcs          $L3 $G3Q --pn-order 7
run_gpu l3d_g3q_p7_fcs_brk      $L3 $G3Q --pn-order 7 $BRK3
run_gpu l3d_g3q_p7_fcs_brk_ad   $L3 $G3Q --pn-order 7 $BRK3 $AD3
run_gpu l3d_g3q_p15_fcs         $L3 $G3Q --pn-order 15
run_gpu l3d_g3q_p15_fcs_brk_ad  $L3 $G3Q --pn-order 15 $BRK3 $AD3
echo "=== G3H $(date +%T) ==="
run_gpu l3d_g3h_p7_fcs          $L3 $G3H --pn-order 7
run_gpu l3d_g3h_p7_fcs_brk      $L3 $G3H --pn-order 7 $BRK3
run_gpu l3d_g3h_p7_fcs_brk_ad   $L3 $G3H --pn-order 7 $BRK3 $AD3
echo "=== G3B $(date +%T) ==="
run_gpu l3d_g3b_p7_fcs          $L3 $G3B --pn-order 7
run_gpu l3d_g3b_p7_fcs_brk_ad   $L3 $G3B --pn-order 7 $BRK3 $AD3
run_gpu l3d_g3b_p7_fcs_brk      $L3 $G3B --pn-order 7 $BRK3

echo "=== Report $(date +%T) ==="
python3 "$REPO/scripts/element_weno/tardis_paper_report.py" "$CAMPAIGN" > "$CAMPAIGN/REPORT.md" 2>&1
sed -n '/Block D/,/Block F/p' "$CAMPAIGN/REPORT.md"
echo "=== tardis-3d-graded finished $(date) ==="
