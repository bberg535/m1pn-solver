#!/usr/bin/env bash
# Build the element-WENO solver (no MFEM): m1pn_fast, m1pn_element_weno, tests.
#   bash scripts/build.sh [BUILD_DIR]            (default tmp/build; on NX inside a Slurm job, the login host has 1 core)
#   GPU=1 bash scripts/build.sh tmp/build_gpu    (CUDA time loop, nvcc /sfw/cuda/12.6; build on tardis)
#   MPI_CXX=/path/to/mpic++ bash scripts/build.sh  (another MPI; default: UCX OpenMPI of the spack env m1pn-ucx)
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD=${1:-$ROOT/tmp/build}
CXX=${MPI_CXX:-/home/warehouse15/bberg/spack/opt/spack/linux-zen3/openmpi-5.0.10-kfdcvl5tog3tsi5rusglshnfcauemfjq/bin/mpic++}
EXTRA=()
if [[ "${GPU:-0}" == "1" ]]; then
  EXTRA+=(-DM1PN_ELEMENT_GPU=ON -DCMAKE_CUDA_COMPILER=${NVCC:-/sfw/cuda/12.6/bin/nvcc})
fi
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" -DENABLE_MPI=ON "${EXTRA[@]}"
cmake --build "$BUILD" --parallel "${BUILD_JOBS:-32}"
