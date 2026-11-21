#!/usr/bin/env bash
#SBATCH --job-name=tardis-ad3d-k
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_ad3d_kernel_20261121/slurm-%j.log
# perf/adaptive3d: structured 3D adaptive stabilization kernel (M1PN_GPU_AD3=1) against the connectivity kernel (=0).
# Lattice-3d, FCS, broken lines + split, adaptive tol 3e-4 tile 8, T = 3.2: G3Q P15/P7, G3H P7; nsys of G3Q P15.
set -uo pipefail
REPO=/home/user/bberg/phd/03_projects/M1PN_Solver/.claude/worktrees/adaptive3d
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_ad3d_kernel_20261121
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-ad3d-k start $(date) on $(hostname) ==="
git -C "$REPO" rev-parse --short HEAD
GPU=1 BUILD_JOBS=32 bash "$REPO/scripts/build.sh" "$CAMPAIGN/build_gpu" > "$CAMPAIGN/build_gpu.log" 2>&1 || {
    echo "BUILD FAILED"; tail -n 40 "$CAMPAIGN/build_gpu.log"; exit 1; }
BIN="$CAMPAIGN/build_gpu/m1pn_fast"
(cd "$CAMPAIGN/build_gpu" && ctest --output-on-failure) > "$CAMPAIGN/ctest.log" 2>&1
CRC=$?; echo "ctest rc=$CRC ($(date +%T))"
[ $CRC -eq 0 ] || { echo "CTEST FAILED"; tail -n 40 "$CAMPAIGN/ctest.log"; exit 1; }
bash "$REPO/tests/gpu_validate_adaptive.sh" "$BIN" "$CAMPAIGN/gpu_validate_adaptive" > "$CAMPAIGN/gpu_validate_adaptive.log" 2>&1
echo "validate rc=$? $(tail -n 1 "$CAMPAIGN/gpu_validate_adaptive.log")"
COMMON="--threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 65536 --gpu"
L3="--case lattice-3d --inflow vacuum --dimension 3 --degree 1 --background 1e-10 --mirror-reduction off --skip-fields --time 3.2 --fcs analytic --break-units on --split-breaks on"
G3Q="--elements 7 --cells-per-unit 4,8,16,16,16,8,4"
G3H="--elements 7 --cells-per-unit 8,16,24,24,24,16,8"
AD3="--adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8"
run_gpu() {
    local s=$1 name=$2; shift 2
    rm -rf "$RUNS/$name"; local t0=$(date +%s)
    M1PN_GPU_AD3=$s "$BIN" $COMMON "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    echo "[done] $name rc=$? $(( $(date +%s) - t0 ))s $(date +%T)"
}
for s in 1 0; do
    run_gpu $s g3q_p15_ad_s$s $L3 $G3Q --pn-order 15 $AD3
    run_gpu $s g3q_p7_ad_s$s  $L3 $G3Q --pn-order 7 $AD3
    run_gpu $s g3h_p7_ad_s$s  $L3 $G3H --pn-order 7 $AD3
done
NSYS=$(command -v nsys || ls -d /usr/local/cuda/bin/nsys 2>/dev/null | head -1)
M1PN_GPU_AD3=1 "$NSYS" profile -t cuda -s none --cpuctxsw=none -o "$CAMPAIGN/p15_ad" -f true \
    "$BIN" $COMMON $L3 $G3Q --pn-order 15 $AD3 --output "$RUNS/nsys_p15_ad" > "$RUNS/nsys_p15_ad.log" 2>&1
"$NSYS" stats -r cuda_gpu_kern_sum -f column "$CAMPAIGN/p15_ad.nsys-rep" > "$CAMPAIGN/p15_ad.txt" 2>&1
python3 - "$RUNS" <<'PY'
import json, os, sys
R = sys.argv[1]
for n in sorted(os.listdir(R)):
    f = os.path.join(R, n, 'summary.json')
    if os.path.isfile(f):
        s = json.load(open(f)); ad = s.get('adaptive_pn') or {}
        print(f"{n:18s} total {s['total_seconds']:7.1f} solver {s['solver_seconds']:7.1f} mass {s['final_density_mass']!r} "
              f"rate {s['absorption_rate_final']!r} frac {ad.get('mean_coeff_fraction', 1):.3f}")
PY
echo "=== tardis-ad3d-k finished $(date) ==="
