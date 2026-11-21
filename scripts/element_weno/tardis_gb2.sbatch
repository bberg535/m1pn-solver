#!/usr/bin/env bash
#SBATCH --job-name=tardis-gb2
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=05:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_gb2_20261121/slurm-%j.log
# 2D lattice, finer graded meshes everywhere: GBY = 48,96,192,192,192,96,48 and GB2 = 64,128,256,256,256,128,64 (twice GB),
# P35 uniform with the Block A setup (FCS, broken lines, split, mirror auto, constant filter 5.2), deviations against
# our MC reference (after job 164702: GBX refined only the central intervals).
set -uo pipefail
REPO=/home/user/bberg/phd/03_projects/M1PN_Solver
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_gb2_20261121
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-gb2 start $(date) on $(hostname) ==="
git -C "$REPO" rev-parse --short HEAD
BIN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_g3h_p21_20261121/build_gpu/m1pn_fast
[ -x "$BIN" ] || { GPU=1 BUILD_JOBS=32 bash "$REPO/scripts/build.sh" "$CAMPAIGN/build_gpu" > "$CAMPAIGN/build_gpu.log" 2>&1 || exit 1; BIN="$CAMPAIGN/build_gpu/m1pn_fast"; }
echo "binary $BIN"
COMMON="--threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 400000 --gpu"
LAT="--case lattice-paul --inflow vacuum --dimension 2 --elements 7 --degree 1 --background 1e-10 --fcs analytic --break-units on --split-breaks on --mirror-reduction auto --time 3.2 --filter constant --filter-strength 5.2"
GB="--cells-per-unit 32,64,128,128,128,64,32"
GBX="--cells-per-unit 32,64,192,192,192,64,32"
AD="--adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 8"
run() {
    local name=$1; shift
    rm -rf "$RUNS/$name"; local t0=$(date +%s)
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$RUNS/$name.gpumem" 2>/dev/null &
    local SMI=$!
    "$BIN" $COMMON $LAT "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    local rc=$?
    kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
    echo "[done] $name rc=$rc $(( $(date +%s) - t0 ))s gpu_peak $(sort -n "$RUNS/$name.gpumem" | tail -n 1) MiB $(date +%T)"
    [ $rc -eq 0 ] || { tail -n 2 "$RUNS/$name.log"; return 1; }
    python3 -c '
import json, sys
s = json.load(open(sys.argv[1] + "/summary.json")); L = s["lattice_fcs"]; ad = s.get("adaptive_pn") or {}
M, A, R = 2.080890, 1.119098, 0.731555
d = lambda x, r: 100 * (x / r - 1)
print("   dM %+.4f dA %+.4f dR %+.4f | init %.1f solver %.1f total %.1f | active %.3f meanN %s" % (
    d(L["total_mass_final"], M), d(L["total_absorbed"], A), d(L["total_absorption_rate_final"], R),
    s["initialization_seconds"], s["solver_seconds"], s["total_seconds"], ad.get("mean_coeff_fraction", 1), ad.get("mean_order")))
' "$RUNS/$name"
}
GBY="--cells-per-unit 48,96,192,192,192,96,48"
GB2="--cells-per-unit 64,128,256,256,256,128,64"
run gby_p35_uni $GBY --pn-order 35
run gb2_p35_uni $GB2 --pn-order 35
echo "=== tardis-gb2 finished $(date) ==="
