#!/usr/bin/env bash
#SBATCH --job-name=tardis-perf3d
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_perf3d_20261120/slurm-%j.log
# 3D solver profile on the A100 with the binary of tardis_l3d_paper_20261120 (merged standard): nsys kernel summary
# of lattice-3d U112 P7 with FCS (T = 3.2) and ncu (speed of light, occupancy, instruction mix) of the main kernels.
set -uo pipefail
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_perf3d_20261120
BIN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_l3d_paper_20261120/build_gpu/m1pn_fast
mkdir -p "$CAMPAIGN/runs"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-perf3d start $(date) ==="
[ -x "$BIN" ] || { echo "binary missing"; exit 1; }
NSYS=$(command -v nsys || ls -d /usr/local/cuda/bin/nsys 2>/dev/null | head -1)
NCU=$(command -v ncu || ls -d /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute*/ncu 2>/dev/null | head -1)
echo "nsys=$NSYS ncu=$NCU"
COMMON="--threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 65536 --gpu"
L3="--case lattice-3d --inflow vacuum --dimension 3 --degree 1 --background 1e-10 --mirror-reduction off --skip-fields"
"$NSYS" profile -t cuda,osrt -s none --cpuctxsw=none -o "$CAMPAIGN/u112_p7_fcs" -f true \
    "$BIN" $COMMON $L3 --elements 112 --pn-order 7 --fcs analytic --time 3.2 --output "$CAMPAIGN/runs/u112_nsys" \
    > "$CAMPAIGN/runs/u112_nsys.log" 2>&1
"$NSYS" stats -r cuda_gpu_kern_sum,cuda_api_sum -f column "$CAMPAIGN/u112_p7_fcs.nsys-rep" > "$CAMPAIGN/u112_p7_fcs.txt" 2>&1
echo "[nsys] done $(date +%T)"
"$NCU" --kernel-name regex:"k_stab_element|k_gather_stab|k_limiter_group|k_line_solve|k_ghost|k_weno|k_combine|k_react_rows" \
    --launch-skip 20 --launch-count 12 --section SpeedOfLight --section Occupancy --section LaunchStats \
    --section WarpStateStats --section ComputeWorkloadAnalysis \
    --metrics smsp__inst_executed.sum,sm__sass_thread_inst_executed_op_integer_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,dram__bytes.sum \
    "$BIN" $COMMON $L3 --elements 112 --pn-order 7 --time 0.1 --output "$CAMPAIGN/runs/u112_ncu" \
    > "$CAMPAIGN/u112_ncu.txt" 2>&1
echo "[ncu] rc=$? $(date +%T)"
echo "=== tardis-perf3d finished $(date) ==="
