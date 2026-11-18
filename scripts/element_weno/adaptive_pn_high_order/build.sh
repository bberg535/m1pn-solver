#!/usr/bin/env bash
set -eu
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "$source_dir/../../.."
study=tmp/adaptive_pn_high_order
eigen_include=${EIGEN3_INCLUDE_DIR:-/usr/include/eigen3}
compiler=${CXX:-g++}
common=(-std=c++17 -O2 -DEIGEN_DONT_PARALLELIZE -I src -isystem "$eigen_include")
mkdir -p "$study/objects"
object_dir="$study/objects"
"$compiler" "${common[@]}" -c src/adaptive_pn.cpp -o "$object_dir/adaptive_pn.o"
for unit in angular boundary element_weno positivity_projection separable_trace element_weno_mpi; do
    "$compiler" "${common[@]}" -DNDEBUG -fopenmp -c "src/$unit.cpp" -o "$object_dir/$unit.o"
done
shared=("$object_dir/adaptive_pn.o" "$object_dir/angular.o")
"$compiler" "${common[@]}" "$source_dir/physics.cpp" "${shared[@]}" -o "$study/physics_timing"
cp "$study/physics_timing" "$study/physics"
"$compiler" "${common[@]}" "$source_dir/stability.cpp" "${shared[@]}" -o "$study/stability"
"$compiler" "${common[@]}" -fopenmp "$source_dir/native.cpp" "$object_dir"/*.o -o "$study/native"
