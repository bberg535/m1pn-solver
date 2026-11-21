#!/usr/bin/env bash
#SBATCH --job-name=tardis-ad3d
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_adaptive3d_20261121/slurm-%j.log
# Dynamic adaptive PN in 3D on the merged standard (paper step 4): graded G3Q/G3H with FCS, broken lines + split,
# uniform PN against adaptive (tol 3e-4, tile 8), P7 and P15. A100.
set -uo pipefail
REPO=/home/user/bberg/phd/03_projects/M1PN_Solver
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_adaptive3d_20261121
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-stab3d start $(date) on $(hostname) ==="
git -C "$REPO" rev-parse --short HEAD
GPU=1 BUILD_JOBS=32 bash "$REPO/scripts/build.sh" "$CAMPAIGN/build_gpu" > "$CAMPAIGN/build_gpu.log" 2>&1 || {
    echo "BUILD FAILED"; tail -n 40 "$CAMPAIGN/build_gpu.log"; exit 1; }
BIN="$CAMPAIGN/build_gpu/m1pn_fast"
(cd "$CAMPAIGN/build_gpu" && ctest --output-on-failure) > "$CAMPAIGN/ctest.log" 2>&1
CRC=$?; echo "ctest rc=$CRC ($(date +%T))"
[ $CRC -eq 0 ] || { echo "CTEST FAILED"; tail -n 40 "$CAMPAIGN/ctest.log"; exit 1; }
COMMON="--threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 65536 --gpu"
L3="--case lattice-3d --inflow vacuum --dimension 3 --degree 1 --background 1e-10 --mirror-reduction off --skip-fields"
BRK3="--break-units on --split-breaks on"
G3H="--elements 7 --cells-per-unit 8,16,24,24,24,16,8"
run_gpu() {
    local s=$1 name=$2; shift 2
    rm -rf "$RUNS/$name"; local t0=$(date +%s)
    M1PN_GPU_STRIP3=$s timeout 6000 "$BIN" $COMMON "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    echo "[done] $name rc=$? $(( $(date +%s) - t0 ))s $(date +%T)"
}
G3Q="--elements 7 --cells-per-unit 4,8,16,16,16,8,4"
AD3="--adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8"
run_gpu 1 g3q_p7_brk     $L3 --time 3.2 $G3Q --pn-order 7 --fcs analytic $BRK3
run_gpu 1 g3q_p7_brk_ad  $L3 --time 3.2 $G3Q --pn-order 7 --fcs analytic $BRK3 $AD3
run_gpu 1 g3q_p15_brk    $L3 --time 3.2 $G3Q --pn-order 15 --fcs analytic $BRK3
run_gpu 1 g3q_p15_brk_ad $L3 --time 3.2 $G3Q --pn-order 15 --fcs analytic $BRK3 $AD3
run_gpu 1 g3h_p7_brk_ad  $L3 --time 3.2 $G3H --pn-order 7 --fcs analytic $BRK3 $AD3
python3 - "$RUNS" <<'PY'
import json, os, sys
R = sys.argv[1]
ref = dict(mass=0.702146, abs=2.497854, rate=0.993430)
for n in sorted(os.listdir(R)):
    f = os.path.join(R, n, 'summary.json')
    if not os.path.isfile(f):
        continue
    s = json.load(open(f))
    m, a, r = s.get('final_density_mass'), s.get('absorbed'), s.get('absorption_rate_final')
    dev = lambda x, y: f"{100 * (x / y - 1):+.4f}" if isinstance(x, (int, float)) else "-"
    print(f"{n:22s} total {s.get('total_seconds', 0):8.1f} solver {s.get('solver_seconds', 0):8.1f} "
          f"mass {m!r} dM {dev(m, ref['mass'])} dA {dev(a, ref['abs'])} dR {dev(r, ref['rate'])} rss {s.get('peak_rss_kib_rank0')}")
PY
echo "=== tardis-stab3d finished $(date) ==="
