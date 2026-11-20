#!/usr/bin/env bash
#SBATCH --job-name=tardis-perf2
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_perf_20261120/slurm2-%j.log
# Follow-up of tardis_perf_profile.sbatch with its binary: device ghost (DGEMM) forced on GB (ratio < 0.1 picks the
# host), and full-length nsys kernel profiles of GB P35 adaptive (1e-4, tile 8) and uniform with device ghost.
set -uo pipefail
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_perf_20261120
RUNS=$CAMPAIGN/runs
BIN=$CAMPAIGN/build_gpu/m1pn_fast
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-perf2 start $(date) ==="
COMMON="--threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 65536 --gpu"
run_gpu() {
    local name=$1; shift
    if [ -f "$RUNS/$name/summary.json" ]; then echo "[skip] $name"; return 0; fi
    rm -rf "$RUNS/$name"; local t0=$(date +%s)
    "$BIN" $COMMON "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    echo "[done] $name rc=$? $(( $(date +%s) - t0 ))s $(date +%T)"
}
LAT_ADAPT="--case lattice-paul --inflow vacuum --dimension 2 --elements 7 --degree 1 --background 1e-10 --fcs analytic --break-units on --split-breaks on --mirror-reduction auto --time 3.2"
GB="--cells-per-unit 32,64,128,128,128,64,32"
GBH="--cells-per-unit 16,32,64,64,64,32,16"
AD="--adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 8"
M1PN_GPU_GHOST=1 M1PN_GPU_GHOST_WHY=1 run_gpu gb_p35_uni_dev $LAT_ADAPT $GB --pn-order 35
M1PN_GPU_GHOST=1 run_gpu gb_p21_uni_dev $LAT_ADAPT $GB --pn-order 21
NSYS=$(command -v nsys || ls -d /usr/local/cuda/bin/nsys 2>/dev/null | head -1)
prof() {
    local name=$1; shift
    [ -f "$CAMPAIGN/nsys/$name.txt" ] && return 0
    rm -rf "$RUNS/${name}_nsys"
    "$NSYS" profile -t cuda,osrt -s none --cpuctxsw=none -o "$CAMPAIGN/nsys/$name" -f true \
        "$BIN" $COMMON "$@" --output "$RUNS/${name}_nsys" > "$RUNS/${name}_nsys.log" 2>&1
    "$NSYS" stats -r cuda_gpu_kern_sum,cuda_api_sum -f column "$CAMPAIGN/nsys/$name.nsys-rep" > "$CAMPAIGN/nsys/$name.txt" 2>&1
    echo "[nsys] $name $(date +%T)"
}
prof prof_gb_p35_ad_full $LAT_ADAPT $GB --pn-order 35 $AD
M1PN_GPU_GHOST=1 prof prof_gb_p35_uni_dev_full $LAT_ADAPT $GB --pn-order 35
prof prof_gbh_p35_ad_full $LAT_ADAPT $GBH --pn-order 35 $AD
python3 - "$RUNS" <<'PY'
import json, os, sys
runs = sys.argv[1]
for r in sorted(os.listdir(runs)):
    f = os.path.join(runs, r, 'summary.json')
    if not os.path.isfile(f): continue
    d = json.load(open(f)); g = d.get('gpu_timing', {}); a = d.get('adaptive_pn', {})
    print(f"{r:28s} solver {d['solver_seconds']:8.2f} ghost {g.get('host_ghost', 0):7.2f} wait {g.get('wait', 0):7.2f} sync {g.get('sync', 0):6.2f} radius {g.get('spectral_radius',0):5.2f} steps {d['steps']} mass {d['final_density_mass']!r} active {a.get('mean_coeff_fraction', 1):.3f}")
PY
echo "=== tardis-perf2 finished $(date) ==="
