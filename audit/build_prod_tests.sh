#!/usr/bin/env bash
# Builds repository tests against the UNPATCHED current src/element_weno.cpp (production code, no audit hooks).
# usage: build_prod_tests.sh tests/element_weno [tests/other ...]; binaries in audit/build/prod_<name>
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
# reuse flags/libs of build.sh
eval "$(sed -n '/^MAINBUILD=/,/^mkdir -p/p' "$HERE/build.sh" | sed '$d' | sed "s#\$HERE#$HERE#g; s#\$ROOT#$ROOT#g")"
mkdir -p "$HERE/build"
$CXX $FLAGS -c "$ROOT/src/element_weno.cpp" -o "$HERE/build/element_weno_prod.o"
for t in "$@"; do
  $CXX $FLAGS "$ROOT/$t.cpp" "$HERE/build/element_weno_prod.o" -o "$HERE/build/prod_$(basename "$t")" $LIBS
done
