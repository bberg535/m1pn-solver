#!/usr/bin/env bash
set -euo pipefail
TASK_FILTER=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TASK_PARENT=$(cd "$TASK_FILTER/.." && pwd)
# Read the established configuration, never install or switch MFEM variants.
TASK_CONFIG=${MFEM_CONFIG:-$(sed -n 's/^MFEM_CONFIG:FILEPATH=//p' "$TASK_PARENT/tmp/opt_20261109/build_opt/CMakeCache.txt")}
test -f "$TASK_CONFIG"
TASK_CXX=$(make --no-print-directory -s -f "$TASK_PARENT/scripts/mfem-flags.mk" "MFEM_CONFIG=$TASK_CONFIG" compiler)
TASK_MPI=$(make --no-print-directory -s -f "$TASK_PARENT/scripts/mfem-flags.mk" "MFEM_CONFIG=$TASK_CONFIG" mpi)
TASK_MPI_FLAG=OFF
if [[ "$TASK_MPI" == YES ]]; then TASK_MPI_FLAG=ON; fi
TASK_BUILD=${1:-"$TASK_FILTER/tmp/build"}
cmake -S "$TASK_FILTER" -B "$TASK_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$TASK_CXX" -DMFEM_CONFIG="$TASK_CONFIG" \
  -DENABLE_MPI="$TASK_MPI_FLAG" -DENABLE_OPENMP=ON -DBUILD_TESTING=ON
cmake --build "$TASK_BUILD" --parallel "${BUILD_JOBS:-2}" --target m1pn_nn_filter m1pn_unfiltered nn_filter_checks
