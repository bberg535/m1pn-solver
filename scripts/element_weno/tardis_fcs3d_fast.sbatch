#!/usr/bin/env bash
#SBATCH --job-name=tardis-fcs3d-fast
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_fcs3d_fast_20261120/slurm-%j.log

# Fast 3D lattice FCS (feature/fcs3d-table: polar rule, nodal FP32 table, adaptive 1/8-point quadrature, auto batch)
# on the A100. References with the Cartesian point table: tardis_paper_20261120 (U56/U112 P7 FCS) and
# tardis_3d_graded_20261120 (G3H/G3B P7 FCS). U168 with FCS did not fit before (point table ~98 GB).
# Builds from the worktree of the branch. Restartable: runs with summary.json are skipped.
set -uo pipefail

SRC=/home/user/bberg/phd/03_projects/M1PN_Solver/.claude/worktrees/fcs3d-table
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_fcs3d_fast_20261120
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1

echo "=== tardis-fcs3d-fast start $(date) on $(hostname) ==="
git -C "$SRC" rev-parse --short HEAD
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

GPU=1 BUILD_JOBS=32 bash "$SRC/scripts/build.sh" "$CAMPAIGN/build_gpu" > "$CAMPAIGN/build_gpu.log" 2>&1 || {
    echo "BUILD FAILED"; tail -n 40 "$CAMPAIGN/build_gpu.log"; exit 1; }
BIN="$CAMPAIGN/build_gpu/m1pn_fast"
sha256sum "$BIN" > "$CAMPAIGN/binary.sha256"
(cd "$CAMPAIGN/build_gpu" && ctest --output-on-failure) > "$CAMPAIGN/ctest.log" 2>&1
CRC=$?
echo "ctest rc=$CRC ($(date +%T))"
[ $CRC -eq 0 ] || { echo "CTEST FAILED"; tail -n 40 "$CAMPAIGN/ctest.log"; exit 1; }

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

L3="--case lattice-3d --inflow vacuum --dimension 3 --degree 1 --time 3.2 --background 1e-10 --mirror-reduction off --skip-fields --fcs analytic"
BRK3="--break-units on --split-breaks on"
G3H="--elements 7 --cells-per-unit 8,16,24,24,24,16,8"
G3B="--elements 7 --cells-per-unit 6,12,32,32,32,12,6"

run_gpu fast_p7_56_fcs        $L3 --elements 56  --pn-order 7
run_gpu fast_p7_112_fcs       $L3 --elements 112 --pn-order 7
M1PN_FCS3D_ADAPTIVE_QUAD=0 run_gpu fast_p7_112_fcs_q8 $L3 --elements 112 --pn-order 7
run_gpu fast_g3h_p7_fcs       $L3 $G3H --pn-order 7
run_gpu fast_g3b_p7_fcs       $L3 $G3B --pn-order 7
run_gpu fast_p7_168_fcs       $L3 --elements 168 --pn-order 7
run_gpu fast_p7_168_fcs_brk   $L3 --elements 168 --pn-order 7 $BRK3

echo "=== Summary $(date +%T) ==="
python3 - "$RUNS" <<'PY'
import json, os, sys
mc = dict(mass=0.702146, absorbed=2.497854, rate=0.993430)
runs = sys.argv[1]
for r in sorted(os.listdir(runs)):
    f = os.path.join(runs, r, 'summary.json')
    if not os.path.isfile(f):
        continue
    d = json.load(open(f)); l = d.get('lattice_fcs', {})
    print(f"{r:22s} total {d['total_seconds']:8.1f} s table {l.get('table_seconds', 0):8.1f} s "
          f"ev {l.get('evaluated_points', '-')} | mass {100*(l['total_mass_final']/mc['mass']-1):+.3f}% "
          f"abs {100*(l['total_absorbed']/mc['absorbed']-1):+.3f}% rate {100*(l['total_absorption_rate_final']/mc['rate']-1):+.3f}%")
PY
echo "=== tardis-fcs3d-fast finished $(date) ==="
