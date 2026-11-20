#!/usr/bin/env bash
#SBATCH --job-name=tardis-perf
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_perf_20261120/slurm-%j.log

# Solver performance on the A100 (perf/solver-202611): (1) ghost traces as DGEMM against the per-warp kernel on the
# paper runs (references: tardis_paper_20261120 block A), (2) host vs device ghost on the uniform 896^2 lattice,
# (3) nsys kernel profiles of adaptive P35 on GB and GBH (GB gains only 2x from adaptivity, GBH 11x).
# Builds from the worktree of the branch. Restartable: runs with summary.json are skipped.
set -uo pipefail
SRC=/home/user/bberg/phd/03_projects/M1PN_Solver/.claude/worktrees/perf-solver
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_perf_20261120
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS" "$CAMPAIGN/nsys"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1

echo "=== tardis-perf start $(date) on $(hostname) ==="
git -C "$SRC" rev-parse --short HEAD
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU=1 BUILD_JOBS=32 bash "$SRC/scripts/build.sh" "$CAMPAIGN/build_gpu" > "$CAMPAIGN/build_gpu.log" 2>&1 || {
    echo "BUILD FAILED"; tail -n 40 "$CAMPAIGN/build_gpu.log"; exit 1; }
BIN="$CAMPAIGN/build_gpu/m1pn_fast"
(cd "$CAMPAIGN/build_gpu" && ctest --output-on-failure) > "$CAMPAIGN/ctest.log" 2>&1
CRC=$?; echo "ctest rc=$CRC ($(date +%T))"
[ $CRC -eq 0 ] || { echo "CTEST FAILED"; tail -n 40 "$CAMPAIGN/ctest.log"; exit 1; }

COMMON="--threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 65536 --gpu"
run_gpu() {
    local name=$1; shift
    if [ -f "$RUNS/$name/summary.json" ]; then echo "[skip] $name"; return 0; fi
    rm -rf "$RUNS/$name"
    local t0=$(date +%s)
    "$BIN" $COMMON "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    echo "[done] $name rc=$? $(( $(date +%s) - t0 ))s $(date +%T)"
}
LAT_ADAPT="--case lattice-paul --inflow vacuum --dimension 2 --elements 7 --degree 1 --background 1e-10 --fcs analytic --break-units on --split-breaks on --mirror-reduction auto --time 3.2"
GBQ="--cells-per-unit 8,16,32,32,32,16,8"
GBH="--cells-per-unit 16,32,64,64,64,32,16"
GB="--cells-per-unit 32,64,128,128,128,64,32"
LA="--case lattice-paul --inflow vacuum --dimension 2 --degree 1 --background 1e-10 --mirror-reduction off --fcs analytic --elements 896 --pn-order 21"

echo "=== (1) ghost DGEMM on the uniform P35 runs $(date +%T) ==="
run_gpu gbq_p35_uni          $LAT_ADAPT $GBQ --pn-order 35
run_gpu gbh_p35_uni          $LAT_ADAPT $GBH --pn-order 35
M1PN_GPU_GHOST_GEMM=0 run_gpu gbh_p35_uni_nogemm $LAT_ADAPT $GBH --pn-order 35
run_gpu gb_p35_uni           $LAT_ADAPT $GB --pn-order 35
run_gpu gb_p21_uni           $LAT_ADAPT $GB --pn-order 21
echo "=== (2) host vs device ghost, uniform 896^2 P21 (t = 0.8) $(date +%T) ==="
M1PN_GPU_GHOST_WHY=1 run_gpu u896_p21_t08_auto $LA --time 0.8
M1PN_GPU_GHOST=0 run_gpu u896_p21_t08_host $LA --time 0.8
M1PN_GPU_GHOST=1 run_gpu u896_p21_t08_dev  $LA --time 0.8
M1PN_GPU_GHOST=1 M1PN_GPU_GHOST_GEMM=0 run_gpu u896_p21_t08_dev_nogemm $LA --time 0.8
echo "=== (3) nsys: adaptive P35 on GB and GBH (t = 0.8) $(date +%T) ==="
NSYS=$(command -v nsys || ls -d /usr/local/cuda/bin/nsys 2>/dev/null | head -1)
AD="--adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 8"
for m in GB GBH; do
    mesh=${!m}
    name=prof_${m,,}_p35_ad_t08
    run_gpu ${name}_plain $LAT_ADAPT $mesh --pn-order 35 $AD --time 0.8
    if [ -n "$NSYS" ] && [ ! -f "$CAMPAIGN/nsys/$name.txt" ]; then
        rm -rf "$RUNS/${name}_nsys"
        "$NSYS" profile -t cuda,osrt -s none --cpuctxsw=none -o "$CAMPAIGN/nsys/$name" -f true \
            "$BIN" $COMMON $LAT_ADAPT $mesh --pn-order 35 $AD --time 0.8 --output "$RUNS/${name}_nsys" \
            > "$RUNS/${name}_nsys.log" 2>&1
        "$NSYS" stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum -f column "$CAMPAIGN/nsys/$name.nsys-rep" \
            > "$CAMPAIGN/nsys/$name.txt" 2>&1
        echo "[nsys] $name $(date +%T)"
    fi
done
[ -z "$NSYS" ] && echo "nsys not found"

echo "=== Summary $(date +%T) ==="
python3 - "$RUNS" <<'PY'
import json, os, sys
runs = sys.argv[1]
for r in sorted(os.listdir(runs)):
    f = os.path.join(runs, r, 'summary.json')
    if not os.path.isfile(f):
        continue
    d = json.load(open(f)); g = d.get('gpu_timing', {}); a = d.get('adaptive_pn', {})
    print(f"{r:28s} solver {d['solver_seconds']:8.2f} ghost {g.get('host_ghost', 0):7.2f} wait {g.get('wait', 0):7.2f} "
          f"sync {g.get('sync', 0):6.2f} steps {d['steps']} mass {d['final_density_mass']!r} active {a.get('mean_coeff_fraction', 1):.3f}")
PY
echo "=== tardis-perf finished $(date) ==="
