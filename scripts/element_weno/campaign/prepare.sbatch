#!/bin/bash -l
#SBATCH --job-name=ew-prep
#SBATCH --constraint=nx
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
W=${M1PN_ROOT:-/home/warehouse15/bberg/phd/03_projects/M1PN_Solver}
source "$W/nx/scripts/nx_common.sh"
nx_setup
B=${M1PN_BUILD:-$W/tmp/build_nx}
"$TASK_CMAKE" -S "$W" -B "$B" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$TASK_CXX" -DMPI_CXX_COMPILER="$TASK_CXX" -DMFEM_CONFIG="$TASK_MFEM_CONFIG" \
  -DENABLE_MPI=ON -DENABLE_OPENMP=ON -DBUILD_TESTING=ON \
  -DCMAKE_CXX_STANDARD_LIBRARIES="$TASK_ZLIB_PREFIX/lib/libz.so.1" \
  -DCMAKE_TRY_COMPILE_PLATFORM_VARIABLES=CMAKE_CXX_STANDARD_LIBRARIES
"$TASK_CMAKE" --build "$B" --target m1pn_fast m1pn_element_weno m1pn_element_weno_tests --parallel "$SLURM_CPUS_PER_TASK"
OMP_NUM_THREADS=4 "$B/m1pn_element_weno_tests" | tail -1
OMP_NUM_THREADS=2 "$TASK_PYTHON" "$W/tests/fast_cli.py" "$B/m1pn_fast" "$B/m1pn_element_weno" | tail -1
sha256sum "$B/m1pn_fast" > "$B/m1pn_fast.sha256"
rm -rf "$TMPDIR"
echo PREPARE_DONE
