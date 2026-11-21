#!/usr/bin/env bash
#SBATCH --job-name=tardis-init3d2
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=460G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_init3d2_20261121/slurm-%j.log
# perf/init3d2: (1) spectral-radius start vectors drawn during the GPU constructor (bitwise check G3Q P15 adaptive
# against 164652); (2) ncu profile of the 3D FCS kernel (G3Q P7, one batch); (3) init breakdown on graded meshes
# 168^3, 216^3, 264^3, 312^3 (P3, one step) and 168^3 P7 (one step).
set -uo pipefail
REPO=/home/user/bberg/phd/03_projects/M1PN_Solver/.claude/worktrees/init3d2
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_init3d2_20261121
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-init3d2 start $(date) on $(hostname) ==="
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
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$RUNS/$name.gpumem" 2>/dev/null &
    local SMI=$!
    M1PN_CTOR_TIMING=1 /usr/bin/time -f "maxrss_kib %M" "$BIN" $COMMON "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    local rc=$?
    kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
    echo "[done] $name rc=$rc $(( $(date +%s) - t0 ))s gpu_peak $(sort -n "$RUNS/$name.gpumem" | tail -n 1) MiB $(date +%T)"
}
# (1) bitwise
run_gpu g3q_p15_ad $L3 $G3Q --pn-order 15 $AD3
python3 - "$RUNS/g3q_p15_ad/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print("bitwise", (s['final_density_mass'], s['absorption_rate_final']) == (0.2633599056799755, 0.3650589396907601),
      "init", round(s['initialization_seconds'], 1), "radius", s['gpu_timing']['spectral_radius'])
PY
# (2) ncu: one mid batch of the FCS eval kernel (65536 points per batch)
NCU=/sfw/cuda/12.6/bin/ncu
M1PN_FCS3D_BATCH=65536 timeout 3600 $NCU --kernel-name k_lattice3d_eval_batch --launch-skip 12 --launch-count 1 --set full \
    --import-source yes -f -o "$CAMPAIGN/fcs_ncu" "$BIN" $COMMON $L3 $G3Q --pn-order 7 --max-steps 1 --output "$RUNS/ncu_run" \
    > "$CAMPAIGN/ncu.log" 2>&1
echo "ncu rc=$? $(tail -n 2 "$CAMPAIGN/ncu.log" | tr '\n' ' ')"
[ -f "$CAMPAIGN/fcs_ncu.ncu-rep" ] && $NCU --import "$CAMPAIGN/fcs_ncu.ncu-rep" --page details > "$CAMPAIGN/fcs_ncu_details.txt" 2>&1
# (3) finer meshes, init only
run_gpu m168_p7 $L3 --elements 7 --cells-per-unit 12,24,32,32,32,24,12 --pn-order 7 --max-steps 1
for spec in "168 12,24,32,32,32,24,12" "216 16,32,40,40,40,32,16" "264 20,40,48,48,48,40,20" "312 24,48,56,56,56,48,24"; do
    set -- $spec
    run_gpu m$1_p3 $L3 --elements 7 --cells-per-unit $2 --pn-order 3 --max-steps 1
done
for f in "$RUNS"/m*.log; do echo "--- $(basename $f)"; grep -E "^(ctor|gpu ctor|init):" "$f" | awk '{v=$(NF-1); if (v+0>0.5) print}'; grep -E "maxrss_kib|out of memory" "$f"; done
echo "=== tardis-init3d2 finished $(date) ==="
