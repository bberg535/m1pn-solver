#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TASK_CONFIG=${1:?Usage: build_existing.sh /existing/share/mfem/config.mk [build-directory]}
TASK_BUILD=${2:-"$TASK_ROOT/tmp/build"}
test -f "$TASK_CONFIG"
TASK_CXX=$(make --no-print-directory -s -f "$TASK_ROOT/scripts/mfem-flags.mk" "MFEM_CONFIG=$TASK_CONFIG" compiler)
TASK_MPI=$(make --no-print-directory -s -f "$TASK_ROOT/scripts/mfem-flags.mk" "MFEM_CONFIG=$TASK_CONFIG" mpi)
# Preserve an existing compiler spelling: OpenMPI's C++ aliases otherwise
# cause CMake to reset the cache and discard MFEM_CONFIG on the first retry.
if [[ -f "$TASK_BUILD/CMakeCache.txt" ]]; then
  TASK_CACHED_CXX=$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "$TASK_BUILD/CMakeCache.txt")
  if [[ -n "$TASK_CACHED_CXX" && "$TASK_CACHED_CXX" != "$TASK_CXX" ]]; then
    case "$(basename "$TASK_CACHED_CXX"):$(basename "$TASK_CXX")" in
      mpicxx:mpic++|mpic++:mpicxx)
        if [[ "$(readlink -f "$TASK_CACHED_CXX")" != "$(readlink -f "$TASK_CXX")" ]]; then
          echo "Compiler stack differs; choose a new build directory." >&2
          exit 1
        fi
        TASK_CXX=$TASK_CACHED_CXX
        ;;
      *) echo "Compiler differs from existing cache; choose a new build directory." >&2; exit 1 ;;
    esac
  fi
fi
TASK_MPI_FLAG=OFF
if [[ "$TASK_MPI" == YES ]]; then TASK_MPI_FLAG=ON; fi
cmake -S "$TASK_ROOT" -B "$TASK_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$TASK_CXX" -DMFEM_CONFIG="$TASK_CONFIG" \
  -DENABLE_MPI="$TASK_MPI_FLAG" -DENABLE_OPENMP=ON -DBUILD_TESTING=ON
cmake --build "$TASK_BUILD" --parallel "${BUILD_JOBS:-2}"
