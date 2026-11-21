#!/usr/bin/env bash
#SBATCH --job-name=tardis-perf-ad3d
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_perf_ad3d_20261121/slurm-%j.log
# Kernel profile of dynamic adaptive PN in 3D (merged standard, binary of tardis_adaptive3d_20261121): G3Q P15 with
# FCS, broken lines + split, uniform against adaptive (tol 3e-4, tile 8), T = 3.2. nsys kernel/API summaries.
set -uo pipefail
CAMPAIGN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_perf_ad3d_20261121
BIN=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_adaptive3d_20261121/build_gpu/m1pn_fast
mkdir -p "$CAMPAIGN/runs"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1
echo "=== tardis-perf-ad3d start $(date) ==="
NSYS=$(command -v nsys || ls -d /usr/local/cuda/bin/nsys 2>/dev/null | head -1)
COMMON="--threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low --memory-mib 65536 --gpu"
L3="--case lattice-3d --inflow vacuum --dimension 3 --degree 1 --background 1e-10 --mirror-reduction off --skip-fields --time 3.2"
G3Q="--elements 7 --cells-per-unit 4,8,16,16,16,8,4 --fcs analytic --break-units on --split-breaks on"
AD3="--adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8"
for v in uni ad; do
    X=""; [ $v = ad ] && X="$AD3"
    "$NSYS" profile -t cuda,osrt -s none --cpuctxsw=none -o "$CAMPAIGN/p15_$v" -f true \
        "$BIN" $COMMON $L3 $G3Q --pn-order 15 $X --output "$CAMPAIGN/runs/p15_$v" > "$CAMPAIGN/runs/p15_$v.log" 2>&1
    "$NSYS" stats -r cuda_gpu_kern_sum,cuda_api_sum,osrt_sum -f column "$CAMPAIGN/p15_$v.nsys-rep" > "$CAMPAIGN/p15_$v.txt" 2>&1
    echo "[nsys] $v done $(date +%T)"
done
echo "=== tardis-perf-ad3d finished $(date) ==="
