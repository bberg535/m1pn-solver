#!/usr/bin/env bash
# ctest of a build plus the MPI unit test (bitwise serial vs distributed line solves).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD=${1:-$ROOT/tmp/build}
ctest --test-dir "$BUILD" --output-on-failure -j "${TEST_JOBS:-8}"
MPIRUN=${MPIRUN:-/home/warehouse15/bberg/spack/opt/spack/linux-zen3/openmpi-5.0.10-kfdcvl5tog3tsi5rusglshnfcauemfjq/bin/mpirun}  # the MPI of scripts/build.sh (UCX OpenMPI)
OMP_NUM_THREADS=4 "$MPIRUN" -np 4 --oversubscribe "$BUILD/m1pn_element_weno_mpi_tests"
