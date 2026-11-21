#!/usr/bin/env bash
#SBATCH --job-name=tardis-init3d
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=460G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_init3d_20261121/slurm-%j.log
# perf/init3d: x-independent steady inflow data once per side. Bitwise check against jobs 164640 (G3Q P15) and
# 164641 (G3Q P35 adaptive), init timing with M1PN_CTOR_TIMING.
set -uo pipefail
REPO=/home/user/bberg/phd/03_projects/M1PN_Solver/.claude/worktrees/init3d
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_init3d_20261121
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-init3d start $(date) on $(hostname) ==="
git -C "$REPO" rev-parse --short HEAD; git -C "$REPO" status --short | head
GPU=1 BUILD_JOBS=32 bash "$REPO/scripts/build.sh" "$CAMPAIGN/build_gpu" > "$CAMPAIGN/build_gpu.log" 2>&1 || {
    echo "BUILD FAILED"; tail -n 40 "$CAMPAIGN/build_gpu.log"; exit 1; }
BIN="$CAMPAIGN/build_gpu/m1pn_fast"
(cd "$CAMPAIGN/build_gpu" && ctest --output-on-failure) > "$CAMPAIGN/ctest.log" 2>&1
CRC=$?; echo "ctest rc=$CRC ($(date +%T))"
[ $CRC -eq 0 ] || { echo "CTEST FAILED"; tail -n 40 "$CAMPAIGN/ctest.log"; exit 1; }
bash "$REPO/tests/gpu_validate_adaptive.sh" "$BIN" "$CAMPAIGN/gpu_validate_adaptive" > "$CAMPAIGN/gpu_validate_adaptive.log" 2>&1
echo "validate rc=$? $(tail -n 1 "$CAMPAIGN/gpu_validate_adaptive.log")"
COMMON="--threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 400000 --gpu"
L3="--case lattice-3d --inflow vacuum --dimension 3 --degree 1 --background 1e-10 --mirror-reduction off --skip-fields --time 3.2 --fcs analytic --break-units on --split-breaks on"
G3Q="--elements 7 --cells-per-unit 4,8,16,16,16,8,4"
AD3="--adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8"
run_gpu() {
    local name=$1; shift
    rm -rf "$RUNS/$name"; local t0=$(date +%s)
    M1PN_CTOR_TIMING=1 "$BIN" $COMMON "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    echo "[done] $name rc=$? $(( $(date +%s) - t0 ))s $(date +%T)"
}
run_gpu g3q_p15_uni $L3 $G3Q --pn-order 15
run_gpu g3q_p15_ad  $L3 $G3Q --pn-order 15 $AD3
run_gpu g3q_p35_ad  $L3 $G3Q --pn-order 35 $AD3
for f in "$RUNS"/*.log; do echo "--- $(basename $f)"; grep -E "^(ctor: tail|init):" "$f"; done
python3 - "$RUNS" <<'PY'
import json, os, sys
R = sys.argv[1]
ref = {'g3q_p15_uni': (0.26339701964034246, 0.3649935306268435), 'g3q_p15_ad': (0.2633599056799755, 0.3650589396907601),
       'g3q_p35_ad': (0.2634203950569552, 0.36494916485137036)}
for n in sorted(os.listdir(R)):
    f = os.path.join(R, n, 'summary.json')
    if os.path.isfile(f):
        s = json.load(open(f)); m, r = s['final_density_mass'], s['absorption_rate_final']
        print(f"{n:12s} init {s['initialization_seconds']:6.1f} rss {s['peak_rss_kib_rank0']/1e6:6.1f}GB total {s['total_seconds']:7.1f} solver {s['solver_seconds']:7.1f} "
              f"mass {m!r} rate {r!r} bitwise {ref.get(n) == (m, r)}")
PY
echo "=== tardis-init3d finished $(date) ==="
