#!/usr/bin/env bash
# Builds the audit binaries out-of-tree (audit/build). Uses the existing MFEM-free parts only:
# the patched copy of element_weno.cpp plus the unmodified libm1pn_core.a objects of the main build.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
MAINBUILD=${MAINBUILD:-/home/warehouse15/bberg/phd/03_projects/M1PN_Solver/tmp/build}
SPACK=/home/warehouse15/bberg/spack/opt/spack/linux-zen3
CXX=$SPACK/openmpi-5.0.10-y3kaeguuqknrs5uhcky2pxvf7aeqajy7/bin/mpic++
FLAGS="-O3 -DNDEBUG -std=gnu++17 -DEIGEN_DONT_PARALLELIZE -DM1PN_MPI -I$ROOT/src -isystem /usr/include/eigen3 \
 -isystem$SPACK/mfem-4.8.0-ttr6kagekrts4je7mczrxuvlxw5564se/include -isystem$SPACK/hypre-2.33.0-3d4h4vvfp5y7e6rb7ucihocfcp6mzeot/include \
 -fno-fast-math -fno-math-errno -fno-trapping-math -march=x86-64-v3 -mtune=znver3 -ffp-contract=fast -fopenmp"
LIBS="$MAINBUILD/libm1pn_core.a -L$SPACK/mfem-4.8.0-ttr6kagekrts4je7mczrxuvlxw5564se/lib -lmfem \
 -L$SPACK/hypre-2.33.0-3d4h4vvfp5y7e6rb7ucihocfcp6mzeot/lib -lHYPRE -L$SPACK/openblas-0.3.34-mvst5nm6tuocshgzljzcdwyopd52sblj/lib -lopenblas \
 -L$SPACK/metis-5.1.0-aoizezxokv75owtz4rhsahitto32hzyh/lib -lmetis -lrt -L$SPACK/zlib-ng-2.3.3-foqygcae7osj27gypt74brlxoh2ik6lk/lib -lz \
 -Wl,-rpath,$SPACK/hypre-2.33.0-3d4h4vvfp5y7e6rb7ucihocfcp6mzeot/lib -Wl,-rpath,$SPACK/openblas-0.3.34-mvst5nm6tuocshgzljzcdwyopd52sblj/lib \
 -Wl,-rpath,$SPACK/metis-5.1.0-aoizezxokv75owtz4rhsahitto32hzyh/lib -Wl,-rpath,$SPACK/zlib-ng-2.3.3-foqygcae7osj27gypt74brlxoh2ik6lk/lib"
mkdir -p "$HERE/build"
python3 "$HERE/make_audit_source.py" "$ROOT/src/element_weno.cpp" "$HERE/build/element_weno_audit.cpp"
$CXX $FLAGS -c "$HERE/build/element_weno_audit.cpp" -o "$HERE/build/element_weno_audit.o"
for t in "$@"; do
  if [[ "$t" == tests/* ]]; then   # existing repository test, linked against the (patched copy of the) current element_weno.cpp
    $CXX $FLAGS "$ROOT/$t.cpp" "$HERE/build/element_weno_audit.o" -o "$HERE/build/$(basename "$t")" $LIBS
  else
    $CXX $FLAGS "$HERE/audit_$t.cpp" "$HERE/build/element_weno_audit.o" -o "$HERE/build/audit_$t" $LIBS
  fi
done
