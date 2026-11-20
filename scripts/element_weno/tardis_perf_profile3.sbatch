#!/usr/bin/env bash
#SBATCH --job-name=tardis-perf3
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_perf3_20261120/slurm-%j.log
# perf/solver-202611 on the A100: fork-join overlap of the adaptive kernels (M1PN_GPU_OVERLAP=1/0), faster setup
# (sparse Boundary, CUDA context on a helper thread). Configurations as tardis_paper_20261120 block A (references:
# dyn_* there, same options). Builds from the worktree. Restartable.
set -uo pipefail
SRC=/home/user/bberg/phd/03_projects/M1PN_Solver/.claude/worktrees/perf-solver
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_perf3_20261120
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-perf3 start $(date) on $(hostname) ==="
git -C "$SRC" rev-parse --short HEAD
GPU=1 BUILD_JOBS=32 bash "$SRC/scripts/build.sh" "$CAMPAIGN/build_gpu" > "$CAMPAIGN/build_gpu.log" 2>&1 || {
    echo "BUILD FAILED"; tail -n 40 "$CAMPAIGN/build_gpu.log"; exit 1; }
BIN="$CAMPAIGN/build_gpu/m1pn_fast"
(cd "$CAMPAIGN/build_gpu" && ctest --output-on-failure) > "$CAMPAIGN/ctest.log" 2>&1
CRC=$?; echo "ctest rc=$CRC ($(date +%T))"
[ $CRC -eq 0 ] || { echo "CTEST FAILED"; tail -n 40 "$CAMPAIGN/ctest.log"; exit 1; }
bash "$SRC/tests/gpu_validate_adaptive.sh" "$BIN" "$CAMPAIGN/gpu_validate_adaptive" > "$CAMPAIGN/gpu_validate_adaptive.log" 2>&1
VRC=$?; tail -n 1 "$CAMPAIGN/gpu_validate_adaptive.log"
[ $VRC -eq 0 ] || { echo "ADAPTIVE GPU VALIDATION FAILED (rc=$VRC)"; exit 1; }
COMMON="--threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 65536 --gpu"
run_gpu() {
    local name=$1; shift
    if [ -f "$RUNS/$name/summary.json" ]; then echo "[skip] $name"; return 0; fi
    rm -rf "$RUNS/$name"; local t0=$(date +%s)
    "$BIN" $COMMON "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    echo "[done] $name rc=$? $(( $(date +%s) - t0 ))s $(date +%T)"
}
LAT_ADAPT="--case lattice-paul --inflow vacuum --dimension 2 --elements 7 --degree 1 --background 1e-10 --fcs analytic --break-units on --split-breaks on --mirror-reduction auto --time 3.2"
GBQ="--cells-per-unit 8,16,32,32,32,16,8"
GBH="--cells-per-unit 16,32,64,64,64,32,16"
GB="--cells-per-unit 32,64,128,128,128,64,32"
for m in GBQ GBH GB; do
    mesh=${!m}; lm=${m,,}
    run_gpu dyn_${lm}_p35_1e4_t8        $LAT_ADAPT $mesh --pn-order 35 --adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 8
    M1PN_GPU_OVERLAP=0 run_gpu dyn_${lm}_p35_1e4_t8_serial $LAT_ADAPT $mesh --pn-order 35 --adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 8
    run_gpu dyn_${lm}_p35_t8_f01        $LAT_ADAPT $mesh --pn-order 35 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8 --adaptive-floor 0.1
    run_gpu dyn_${lm}_p35_uni           $LAT_ADAPT $mesh --pn-order 35
done
python3 - "$RUNS" /home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_paper_20261120/runs <<'PY'
import json, os, sys
runs, ref = sys.argv[1], sys.argv[2]
for r in sorted(os.listdir(runs)):
    f = os.path.join(runs, r, 'summary.json')
    if not os.path.isfile(f): continue
    d = json.load(open(f)); g = d.get('gpu_timing', {})
    base = r.replace('_serial', '')
    fr = os.path.join(ref, base, 'summary.json')
    old = json.load(open(fr)) if os.path.isfile(fr) else None
    o = f"old total {old['total_seconds']:7.1f} solver {old['solver_seconds']:7.2f} init {old['initialization_seconds']:5.1f} mass {old['final_density_mass']!r}" if old else ""
    print(f"{r:28s} total {d['total_seconds']:7.1f} solver {d['solver_seconds']:7.2f} init {d['initialization_seconds']:5.1f} wait {g.get('wait',0):6.2f} mass {d['final_density_mass']!r} | {o}")
PY
echo "=== tardis-perf3 finished $(date) ==="
