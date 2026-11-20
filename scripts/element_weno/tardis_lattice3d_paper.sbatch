#!/usr/bin/env bash
#SBATCH --job-name=tardis-l3d-paper
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_l3d_paper_20261120/slurm-%j.log
# tab:lattice3d on the merged standard (experiment/element-weno with perf/solver-202611 and feature/fcs3d-table):
# uniform U56/U112/U168 and graded G3H/G3B, P7 (and P15), without/with FCS, broken lines + split, filter. A100.
# Restartable: runs with summary.json are skipped. Report via tardis_paper_report.py (block D lists l3d_*).
set -uo pipefail
REPO=/home/user/bberg/phd/03_projects/M1PN_Solver
CAMPAIGN=$REPO/nx/tmp/tardis_l3d_paper_20261120
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-l3d-paper start $(date) on $(hostname) ==="
git -C "$REPO" rev-parse --short HEAD
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
run_gpu() {
    local name=$1; shift
    if [ -f "$RUNS/$name/summary.json" ]; then echo "[skip] $name"; return 0; fi
    rm -rf "$RUNS/$name"; local t0=$(date +%s)
    "$BIN" --threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 65536 --gpu \
        "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    echo "[done] $name rc=$? $(( $(date +%s) - t0 ))s $(date +%T)"
}
L3="--case lattice-3d --inflow vacuum --dimension 3 --degree 1 --time 3.2 --background 1e-10 --mirror-reduction off --skip-fields"
BRK3="--break-units on --split-breaks on"
FILT="--filter constant --filter-strength 5.2"
G3H="--elements 7 --cells-per-unit 8,16,24,24,24,16,8"
G3B="--elements 7 --cells-per-unit 6,12,32,32,32,12,6"
for n in 56 112 168; do
    run_gpu l3d_p7_${n}_nofcs      $L3 --elements $n --pn-order 7
    run_gpu l3d_p7_${n}_fcs        $L3 --elements $n --pn-order 7 --fcs analytic
    run_gpu l3d_p7_${n}_fcs_brk    $L3 --elements $n --pn-order 7 --fcs analytic $BRK3
done
run_gpu l3d_p7_112_fcs_filter      $L3 --elements 112 --pn-order 7 --fcs analytic $FILT
run_gpu l3d_p7_168_fcs_brk_filter  $L3 --elements 168 --pn-order 7 --fcs analytic $BRK3 $FILT
run_gpu l3d_g3h_p7_fcs             $L3 $G3H --pn-order 7 --fcs analytic
run_gpu l3d_g3h_p7_fcs_brk         $L3 $G3H --pn-order 7 --fcs analytic $BRK3
run_gpu l3d_g3b_p7_fcs             $L3 $G3B --pn-order 7 --fcs analytic
run_gpu l3d_g3b_p7_fcs_brk         $L3 $G3B --pn-order 7 --fcs analytic $BRK3
run_gpu l3d_p15_112_fcs            $L3 --elements 112 --pn-order 15 --fcs analytic
run_gpu l3d_g3h_p15_fcs_brk        $L3 $G3H --pn-order 15 --fcs analytic $BRK3
run_gpu l3d_p15_168_fcs            $L3 --elements 168 --pn-order 15 --fcs analytic
python3 "$REPO/scripts/element_weno/tardis_paper_report.py" "$CAMPAIGN" > "$CAMPAIGN/REPORT.md" 2>&1
echo "=== tardis-l3d-paper finished $(date) ==="
