#!/usr/bin/env bash
#SBATCH --job-name=tardis-hipn3d
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=460G
#SBATCH --time=08:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_highpn3d_20261121/slurm-%j.log
# High PN orders in 3D (experiment/element-weno f48ef45): lattice-3d, FCS, broken lines + split, T = 3.2,
# G3Q P21/P35 and G3H P21, uniform and adaptive (tol 3e-4, tile 8). Samples GPU memory with nvidia-smi.
set -uo pipefail
REPO=/home/user/bberg/phd/03_projects/M1PN_Solver
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_highpn3d_20261121
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-hipn3d start $(date) on $(hostname) ==="
git -C "$REPO" rev-parse --short HEAD
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2
GPU=1 BUILD_JOBS=32 bash "$REPO/scripts/build.sh" "$CAMPAIGN/build_gpu" > "$CAMPAIGN/build_gpu.log" 2>&1 || {
    echo "BUILD FAILED"; tail -n 40 "$CAMPAIGN/build_gpu.log"; exit 1; }
BIN="$CAMPAIGN/build_gpu/m1pn_fast"
COMMON="--threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 400000 --gpu"
L3="--case lattice-3d --inflow vacuum --dimension 3 --degree 1 --background 1e-10 --mirror-reduction off --skip-fields --time 3.2 --fcs analytic --break-units on --split-breaks on"
G3Q="--elements 7 --cells-per-unit 4,8,16,16,16,8,4"
G3H="--elements 7 --cells-per-unit 8,16,24,24,24,16,8"
AD3="--adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8"
run_gpu() {
    local name=$1; shift
    rm -rf "$RUNS/$name"; local t0=$(date +%s)
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$RUNS/$name.gpumem" 2>/dev/null &
    local SMI=$!
    M1PN_SPLIT_TIMING=1 "$BIN" $COMMON "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    local rc=$?
    kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
    echo "[done] $name rc=$rc $(( $(date +%s) - t0 ))s gpu_peak $(sort -n "$RUNS/$name.gpumem" | tail -n 1) MiB $(date +%T)"
    [ $rc -eq 0 ] || tail -n 5 "$RUNS/$name.log"
}
run_gpu g3q_p21_uni $L3 $G3Q --pn-order 21
run_gpu g3q_p21_ad  $L3 $G3Q --pn-order 21 $AD3
run_gpu g3q_p35_ad  $L3 $G3Q --pn-order 35 $AD3
run_gpu g3q_p35_uni $L3 $G3Q --pn-order 35
run_gpu g3h_p21_ad  $L3 $G3H --pn-order 21 $AD3
run_gpu g3h_p21_uni $L3 $G3H --pn-order 21
python3 - "$RUNS" <<'PY'
import json, os, sys
R = sys.argv[1]
for n in sorted(os.listdir(R)):
    f = os.path.join(R, n, 'summary.json')
    if os.path.isfile(f):
        s = json.load(open(f)); ad = s.get('adaptive_pn') or {}
        print(f"{n:14s} init {s['initialization_seconds']:6.1f} rss {s['peak_rss_kib_rank0']/1e6:6.1f}GB total {s['total_seconds']:7.1f} solver {s['solver_seconds']:7.1f} mass {s['final_density_mass']!r} "
              f"rate {s['absorption_rate_final']!r} frac {ad.get('mean_coeff_fraction', 1):.3f} meanN {ad.get('mean_order', float('nan'))}")
PY
echo "=== tardis-hipn3d finished $(date) ==="
