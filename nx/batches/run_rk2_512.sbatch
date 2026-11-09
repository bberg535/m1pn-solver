#!/bin/bash -l
#SBATCH --job-name=pn-mc-rk2
#SBATCH --constraint=nx
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_MassCorrection_20261104/nx/tmp/slurm_logs/rk2-512-%j.log
set -Eeuo pipefail
ROOT=/home/user/bberg/phd/03_projects/M1PN_MassCorrection_20261104
BIN=$ROOT/nx/tmp/builds/rk2_512/m1pn_mc_study
CASE=${1:?case}
ELEMENTS=${2:?elements}
TIME=${3:?time}
BACKGROUND=${4:?background}
INFLOW=${5:?inflow}
TAG=${6:?tag}
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT=$ROOT/nx/tmp/rk2_512_20261106/$TAG
[[ ! -e $OUT ]]
exec "$BIN" --case "$CASE" --inflow "$INFLOW" --dimension 2 --mesh-type hex \
  --elements "$ELEMENTS" --pn-order 11 --time "$TIME" --cfl 1 \
  --model r2 --mass-mode lumped --stabilization m1pn_mass_correction \
  --kv-gamma 0 --biharmonic-gamma 0 --pn-viscosity-scale 1 \
  --time-integrator ssprk2 --max-steps 100000 --memory-mib 49152 \
  --pulse-width 0.07 --background "$BACKGROUND" --threads 16 --output "$OUT"
