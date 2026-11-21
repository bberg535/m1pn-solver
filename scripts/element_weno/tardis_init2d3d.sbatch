#!/usr/bin/env bash
#SBATCH --job-name=tardis-init2d3d
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=460G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_init2d3d_20261121/slurm-%j.log
# perf/init3d2 final (128-register FCS eval kernel by default, parallel nodal gathers): bitwise and init times against
# experiment/element-weno (base) for 2D lattice GBQ/GBH P35 (uniform, adaptive) and 3D G3Q P7/P15, G3H P21.
set -uo pipefail
NEW=/home/user/bberg/phd/03_projects/M1PN_Solver/.claude/worktrees/init3d2
BASE=/home/user/bberg/phd/03_projects/M1PN_Solver
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_init2d3d_20261121
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-init2d3d start $(date) on $(hostname) ==="
echo "new $(git -C $NEW rev-parse --short HEAD) (+ worktree changes), base $(git -C $BASE rev-parse --short HEAD) ($(git -C $BASE branch --show-current))"
for v in new base; do
    R=$NEW; [ $v = base ] && R=$BASE
    GPU=1 BUILD_JOBS=32 bash "$R/scripts/build.sh" "$CAMPAIGN/build_$v" > "$CAMPAIGN/build_$v.log" 2>&1 || {
        echo "BUILD $v FAILED"; tail -n 40 "$CAMPAIGN/build_$v.log"; exit 1; }
done
(cd "$CAMPAIGN/build_new" && ctest --output-on-failure) > "$CAMPAIGN/ctest.log" 2>&1
CRC=$?; echo "ctest rc=$CRC ($(date +%T))"
[ $CRC -eq 0 ] || { echo "CTEST FAILED"; tail -n 40 "$CAMPAIGN/ctest.log"; exit 1; }
bash "$NEW/tests/gpu_validate_adaptive.sh" "$CAMPAIGN/build_new/m1pn_fast" "$CAMPAIGN/gpu_validate_adaptive" > "$CAMPAIGN/gpu_validate_adaptive.log" 2>&1
echo "validate rc=$? $(tail -n 1 "$CAMPAIGN/gpu_validate_adaptive.log")"
COMMON="--threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 400000 --gpu"
L2="--case lattice-paul --inflow vacuum --dimension 2 --elements 7 --degree 1 --background 1e-10 --fcs analytic --break-units on --split-breaks on --mirror-reduction auto --time 3.2"
GBQ="--cells-per-unit 8,16,32,32,32,16,8"
GBH="--cells-per-unit 16,32,64,64,64,32,16"
L3="--case lattice-3d --inflow vacuum --dimension 3 --degree 1 --background 1e-10 --mirror-reduction off --skip-fields --time 3.2 --fcs analytic --break-units on --split-breaks on"
G3Q="--elements 7 --cells-per-unit 4,8,16,16,16,8,4"
G3H="--elements 7 --cells-per-unit 8,16,24,24,24,16,8"
AD2="--adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8"
AD3="--adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8"
both() {
    local name=$1; shift
    for v in base new; do
        rm -rf "$RUNS/${name}_$v"
        M1PN_CTOR_TIMING=1 "$CAMPAIGN/build_$v/m1pn_fast" $COMMON "$@" --output "$RUNS/${name}_$v" > "$RUNS/${name}_$v.log" 2>&1
        [ $? -eq 0 ] || echo "  $name $v FAILED: $(tail -n 1 "$RUNS/${name}_$v.log")"
    done
    python3 -c '
import json, sys
a = json.load(open(sys.argv[1] + "_base/summary.json")); b = json.load(open(sys.argv[1] + "_new/summary.json"))
key = lambda s: (s["final_density_mass"], s["absorption_rate_final"])
f = lambda s: "init %6.1f solver %7.1f total %7.1f" % (s["initialization_seconds"], s["solver_seconds"], s["total_seconds"])
print("%-14s bitwise %-5s | base %s | new %s" % (sys.argv[1].split("/")[-1], key(a) == key(b), f(a), f(b)))
' "$RUNS/$name"
}
both gbq_p35_uni $L2 $GBQ --pn-order 35
both gbq_p35_ad  $L2 $GBQ --pn-order 35 $AD2
both gbh_p35_uni $L2 $GBH --pn-order 35
both gbh_p35_ad  $L2 $GBH --pn-order 35 $AD2
both g3q_p7_uni  $L3 $G3Q --pn-order 7
both g3q_p15_ad  $L3 $G3Q --pn-order 15 $AD3
both g3q_p35_ad  $L3 $G3Q --pn-order 35 $AD3
both g3h_p21_ad  $L3 $G3H --pn-order 21 $AD3
echo "=== tardis-init2d3d finished $(date) ==="
