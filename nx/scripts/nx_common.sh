#!/usr/bin/env bash
# Shared by compute-node scripts; only activates already installed packages.
set -Eeuo pipefail
TASK_NX=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
TASK_ROOT=$(cd "$TASK_NX/.." && pwd -P)
nx_fail() { printf 'NX PN-MC: %s\n' "$*" >&2; exit 2; }
nx_inside() {
  python3 -c 'import pathlib,sys; pathlib.Path(sys.argv[1]).resolve().relative_to(pathlib.Path(sys.argv[2]).resolve())' "$1" "$TASK_NX"
}
nx_setup() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || nx_fail 'Use the prepared sbatch wrappers; build/run only in an nx allocation.'
  [[ "$(uname -m)" == x86_64 ]] || nx_fail 'Requires the documented x86-64 nx stack.'
  [[ -n "${SLURM_CPUS_PER_TASK:-}" ]] || nx_fail 'SLURM_CPUS_PER_TASK is missing.'
  if [[ -n "${PN_AFC_ENV_FILE:-}" ]]; then
    source "$PN_AFC_ENV_FILE"
  elif type module >/dev/null 2>&1; then
    module load binutils/2.43 gcc/13.2.0
  fi
  if ! type spack >/dev/null 2>&1; then
    source "${SPACK_SETUP:-$HOME/spack/share/spack/setup-env.sh}"
  fi
  [[ -z "${SPACK_ENV:-}" ]] || spack env deactivate
  TASK_ARCH=$(spack arch -t)
  case "$TASK_ARCH" in zen3|zen4) ;; *) nx_fail "Existing Zen3 MFEM requires nx; detected $TASK_ARCH." ;; esac
  TASK_HASHES=(ttr6kagekrts4je7mczrxuvlxw5564se 3d4h4vvfp5y7e6rb7ucihocfcp6mzeot y3kaeguuqknrs5uhcky2pxvf7aeqajy7 aoizezxokv75owtz4rhsahitto32hzyh foqygcae7osj27gypt74brlxoh2ik6lk)
  TASK_PREFIXES=()
  for TASK_HASH in "${TASK_HASHES[@]}"; do
    TASK_PREFIXES+=("$(spack location -i "/$TASK_HASH")")
    [[ -d "${TASK_PREFIXES[-1]}" ]] || nx_fail "Installed package missing: $TASK_HASH"
    spack load "/$TASK_HASH"
  done
  TASK_MFEM_CONFIG="${TASK_PREFIXES[0]}/share/mfem/config.mk"
  TASK_MPI_PREFIX=${TASK_PREFIXES[2]}
  TASK_ZLIB_PREFIX=${TASK_PREFIXES[4]}
  [[ -f "$TASK_MFEM_CONFIG" && -x "$TASK_MPI_PREFIX/bin/mpicxx" ]] || nx_fail 'MFEM config or installed MPI compiler missing.'
  [[ $(make --no-print-directory -s -f "$TASK_ROOT/scripts/mfem-flags.mk" "MFEM_CONFIG=$TASK_MFEM_CONFIG" mpi) == YES ]] || nx_fail 'Selected MFEM must be MPI-enabled.'
  TASK_CXX=$(make --no-print-directory -s -f "$TASK_ROOT/scripts/mfem-flags.mk" "MFEM_CONFIG=$TASK_MFEM_CONFIG" compiler)
  [[ "$(readlink -f "$TASK_CXX")" == "$(readlink -f "$TASK_MPI_PREFIX/bin/mpicxx")" ]] || nx_fail 'MFEM compiler differs from the installed MPI stack.'
  TASK_CMAKE=''
  for TASK_CANDIDATE in "${PN_AFC_CMAKE:-}" /usr/bin/cmake /usr/local/bin/cmake "$(command -v cmake || true)"; do
    [[ -n "$TASK_CANDIDATE" && -x "$TASK_CANDIDATE" ]] || continue
    TASK_BANNER=$("$TASK_CANDIDATE" --version 2>/dev/null) || continue
    if [[ "$TASK_BANNER" =~ ^cmake[[:space:]]version[[:space:]]([0-9]+)\.([0-9]+) ]] &&
       (( BASH_REMATCH[1] > 3 || (BASH_REMATCH[1] == 3 && BASH_REMATCH[2] >= 20) )); then
      TASK_CMAKE=$TASK_CANDIDATE; break
    fi
    [[ -z "${PN_AFC_CMAKE:-}" ]] || nx_fail 'Requested CMake is not runnable or too old.'
  done
  if [[ -z "$TASK_CMAKE" ]] && type module >/dev/null 2>&1; then
    module load cmake/3.28.3
    TASK_CMAKE=$(command -v cmake)
    "$TASK_CMAKE" --version
  fi
  [[ -n "$TASK_CMAKE" ]] || nx_fail 'No runnable CMake >=3.20; set PN_AFC_CMAKE.'
  TASK_PYTHON=${PN_AFC_PYTHON:-python3}
  "$TASK_PYTHON" -c 'import numpy; print("NumPy ready")'
  export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=FALSE
  export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SLURM_MPI_TYPE=none
  export TMPDIR="$TASK_NX/tmp/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-prepare}"
  export MPLCONFIGDIR="$TMPDIR/mpl" MPLBACKEND=Agg
  mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
  printf 'Job=%s Host=%s Arch=%s Threads=%s Root=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$TASK_ARCH" "$OMP_NUM_THREADS" "$TASK_ROOT"
  "$TASK_CXX" --version
}
nx_build() {
  TASK_BUILD=${1:-"$TASK_NX/tmp/builds/nx_$SLURM_JOB_ID"}
  nx_inside "$TASK_BUILD" || nx_fail 'Build directory must remain in nx/.'
  "$TASK_CMAKE" -S "$TASK_NX" -B "$TASK_BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$TASK_CXX" -DMPI_CXX_COMPILER="$TASK_CXX" \
    -DMFEM_CONFIG="$TASK_MFEM_CONFIG" -DENABLE_MPI=ON -DENABLE_OPENMP=ON -DBUILD_TESTING=ON \
    -DCMAKE_CXX_STANDARD_LIBRARIES="$TASK_ZLIB_PREFIX/lib/libz.so.1" \
    -DCMAKE_TRY_COMPILE_PLATFORM_VARIABLES=CMAKE_CXX_STANDARD_LIBRARIES
  "$TASK_CMAKE" --build "$TASK_BUILD" --parallel "${BUILD_JOBS:-$SLURM_CPUS_PER_TASK}"
  TASK_CTEST="$(dirname "$TASK_CMAKE")/ctest"
  [[ -x "$TASK_CTEST" ]] || nx_fail 'Matching CTest executable missing.'
  "$TASK_CTEST" --test-dir "$TASK_BUILD" --output-on-failure --output-log "$TASK_BUILD/ctest_nx.log"
}
