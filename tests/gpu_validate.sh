#!/usr/bin/env bash
# GPU validation of the element-WENO path: CPU vs --gpu with the SAME m1pn_fast binary.
#
# Submit on tardis (one A100):
#   sbatch tests/gpu_validate.sh [M1PN_FAST] [OUTDIR]
# or run interactively on a GPU node:
#   bash tests/gpu_validate.sh [M1PN_FAST] [OUTDIR]
#
# Defaults: M1PN_FAST = tmp/build_gpu/m1pn_fast (built with -DM1PN_ELEMENT_GPU=ON),
#           OUTDIR    = tmp/gpu_validate/<job id or timestamp>.
# Optional env: GPU_VALIDATE_TIMING=1 additionally runs lattice FCS 224^2, P15, T=3.2 on the GPU (timing only).
#
# Cases and tolerances: docs/ELEMENT_WENO_GPU.md, section "Validierung des Systemstands".
# Density (final.csv) max-rel and L2-rel deviation must stay below the tolerance, step counts must match.
# Mirror cases (--mirror-reduction auto, half/quarter/octant domain): CPU+mirror vs GPU+mirror, and GPU+mirror vs
# GPU on the full domain with the same fixed dt (the mirror reduction is exact up to rounding).
# Reference values (16.11.2026): 3e-16 .. 3e-12, lattice FCS 1.7e-10 (FCS table built on the device).
# Exit status: number of failed cases (0 = all pass).
#
#SBATCH --job-name=m1pn-gpu-validate
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=gpu_validate-%j.log
set -u

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then ROOT=$SLURM_SUBMIT_DIR; else ROOT=$(cd "$(dirname "$0")/.." && pwd); fi
CMP="$ROOT/tests/gpu_validate_cmp.py"
B=${1:-$ROOT/tmp/build_gpu/m1pn_fast}
OUT=${2:-$ROOT/tmp/gpu_validate/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
THREADS=${SLURM_CPUS_PER_TASK:-32}
export OMP_NUM_THREADS=$THREADS OMP_PROC_BIND=close

[[ -x "$B" ]] || { echo "m1pn_fast not found: $B"; exit 99; }
[[ -f "$CMP" ]] || { echo "compare script not found: $CMP"; exit 99; }
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
mkdir -p "$OUT"
echo "binary: $B"; echo "output: $OUT"; echo "git: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)"

EW0="--stabilization element-weno --time-integrator ssprk43 --cfl 0.2 --model r2 --mesh-type hex
    --threads $THREADS --angular-reduction auto --sample-moments low"
EW="$EW0 --dt 0 --mirror-reduction off"
LAT="--case lattice-paul --dimension 2 --inflow vacuum --background 1e-10 --degree 1"
FAILED=0
FAILED_CASES=""

# pair TAG TOL [--fcs-integrals|-] SOLVER ARGS...
pair() {
  local tag=$1 tol=$2 extra=$3; shift 3
  [[ "$extra" == "-" ]] && extra=""
  rm -rf "$OUT/${tag}_cpu" "$OUT/${tag}_gpu"
  echo "$tag"
  # shellcheck disable=SC2086
  if ! "$B" $EW "$@" --output "$OUT/${tag}_cpu" > "$OUT/${tag}_cpu.log" 2>&1; then
    echo "  FAIL: cpu run failed"; tail -3 "$OUT/${tag}_cpu.log"; FAILED=$((FAILED+1)); FAILED_CASES+=" $tag"; return
  fi
  # shellcheck disable=SC2086
  if ! "$B" $EW "$@" --gpu --output "$OUT/${tag}_gpu" > "$OUT/${tag}_gpu.log" 2>&1; then
    echo "  FAIL: gpu run failed"; tail -3 "$OUT/${tag}_gpu.log"; FAILED=$((FAILED+1)); FAILED_CASES+=" $tag"; return
  fi
  # shellcheck disable=SC2086
  python3 "$CMP" "$OUT/${tag}_cpu" "$OUT/${tag}_gpu" --max-rel "$tol" $extra || { FAILED=$((FAILED+1)); FAILED_CASES+=" $tag"; }
}

# Tolerance: 1e-8 (project rule for admissible deviations). Measured values are 4-8 orders of magnitude below.
pair crossing_q1_64_P7  1e-8 - --case crossing-study --inflow crossing-study-transport --dimension 2 --elements 64 \
     --degree 1 --pn-order 7 --time 0.05 --pulse-width .07 --background .001
pair boundary_q2_32_P7  1e-8 - --case smooth --inflow smooth-transport --dimension 2 --elements 32 --degree 2 \
     --pn-order 7 --time .15
pair linefcs_q1_128_P7  1e-8 - --case line-source --dimension 2 --inflow vacuum --pulse-width .01 --background .0009 \
     --sigma-s 3 --time 0.02 --elements 128 --degree 1 --pn-order 7 --fcs analytic
pair lattice_56_P7      1e-8 - $LAT --elements 56 --pn-order 7 --time 0.2
pair latticefcs_56_P7   1e-8 --fcs-integrals $LAT --elements 56 --pn-order 7 --time 0.2 --fcs analytic
pair lattice_56_P35     1e-8 - $LAT --elements 56 --pn-order 35 --time 0.2
pair lattice3d_q2_14_P3 1e-8 - --case lattice-3d --dimension 3 --inflow vacuum --background 1e-10 --elements 14 \
     --degree 2 --pn-order 3 --time 0.1
# lattice-3d FCS: time table built on the device (lattice3d_row), CPU builds it on the host.
pair latticefcs3d_q1_28_P3 1e-8 --fcs-integrals --case lattice-3d --dimension 3 --inflow vacuum --background 1e-10 \
     --elements 28 --degree 1 --pn-order 3 --time 0.5 --fcs analytic

# Graded meshes (--cells-per-unit, docs/GRADED_MESH.md): size classes on the device, strip kernel (Q1) and the per
# node stabilization (Q2), class pair tables in the limiter, WENO extensions per width ratio, scaled boundary faces.
GR="--cells-per-unit 6,8,12,16,12,8,6"
pair graded_lattice_q1_P7     1e-8 - $LAT --elements 56 --pn-order 7 --time 0.2 $GR
pair graded_latticefcs_q1_P7  1e-8 --fcs-integrals $LAT --elements 56 --pn-order 7 --time 0.2 --fcs analytic $GR
pair graded_lattice_q2_P7     1e-8 - --case lattice-paul --dimension 2 --inflow vacuum --background 1e-10 --degree 2 \
     --elements 28 --pn-order 7 --time 0.2 --cells-per-unit 3,4,6,8,6,4,3
pair graded_linefcs_q1_P7     1e-8 - --case line-source --dimension 2 --inflow vacuum --pulse-width .01 \
     --background .0009 --sigma-s 3 --time 0.02 --elements 128 --degree 1 --pn-order 7 --fcs analytic \
     --cells-per-unit 16,32,64,32,16

# Periodic meshes (cyclic line solve on the device): smooth transport with exact periodic data.
PER="--case smooth --periodic --pn-order 7 --time .1"
pair periodic_1d_q3_32       1e-8 - $PER --dimension 1 --elements 32 --degree 3
pair periodic_2d_q1_64       1e-8 - $PER --dimension 2 --elements 64 --degree 1
pair periodic_2d_q2_32       1e-8 - $PER --dimension 2 --elements 32 --degree 2
pair periodic_2d_q4_16       1e-8 - $PER --dimension 2 --elements 16 --degree 4
pair periodic_3d_q2_8        1e-8 - --case smooth --periodic --pn-order 3 --time .05 --dimension 3 --elements 8 --degree 2

# mirror_case TAG TOL ARGS...: three runs with one fixed dt (taken from the CPU mirror run with --dt 0).
mirror_case() {
  local tag=$1 tol=$2; shift 2
  rm -rf "$OUT/${tag}_cpu" "$OUT/${tag}_gpu" "$OUT/${tag}_full"
  echo "$tag"
  # shellcheck disable=SC2086
  if ! "$B" $EW0 --dt 0 --mirror-reduction auto "$@" --output "$OUT/${tag}_probe" > "$OUT/${tag}_probe.log" 2>&1; then
    echo "  FAIL: cpu probe failed"; tail -3 "$OUT/${tag}_probe.log"; FAILED=$((FAILED+1)); FAILED_CASES+=" $tag"; return
  fi
  local dt mirrored
  dt=$(python3 -c "import json,sys;print(repr(json.load(open(sys.argv[1]))['dt_selected']))" "$OUT/${tag}_probe/config.json")
  mirrored=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['mirror_reduction'])" "$OUT/${tag}_probe/config.json")
  rm -rf "$OUT/${tag}_probe"
  if [[ "$mirrored" != "True" ]]; then
    echo "  FAIL: mirror reduction not active"; FAILED=$((FAILED+1)); FAILED_CASES+=" $tag"; return
  fi
  local run ok=1
  for run in "cpu:auto:" "gpu:auto:--gpu" "full:off:--gpu"; do
    IFS=: read -r name mode flag <<< "$run"
    # shellcheck disable=SC2086
    if ! "$B" $EW0 --dt "$dt" --mirror-reduction "$mode" "$@" $flag --output "$OUT/${tag}_$name" \
         > "$OUT/${tag}_$name.log" 2>&1; then
      echo "  FAIL: $name run failed"; tail -3 "$OUT/${tag}_$name.log"; ok=0
    fi
  done
  if [[ $ok -eq 1 ]]; then
    echo "  CPU+mirror vs GPU+mirror (dt $dt):"
    python3 "$CMP" "$OUT/${tag}_cpu" "$OUT/${tag}_gpu" --max-rel "$tol" || ok=0
    echo "  GPU+mirror vs GPU full domain:"
    python3 "$CMP" "$OUT/${tag}_full" "$OUT/${tag}_gpu" --max-rel "$tol" || ok=0
  fi
  [[ $ok -eq 1 ]] || { FAILED=$((FAILED+1)); FAILED_CASES+=" $tag"; }
}
LS="--case line-source --dimension 2 --inflow vacuum --pulse-width .01 --background .0009 --sigma-s 3 --time 0.02"
mirror_case mirror_line_q1_128_P7     1e-8 $LS --elements 128 --degree 1 --pn-order 7
mirror_case mirror_linefcs_q1_128_P7  1e-8 $LS --elements 128 --degree 1 --pn-order 7 --fcs analytic
mirror_case mirror_line_q2_64_P7      1e-8 $LS --elements 64 --degree 2 --pn-order 7
mirror_case mirror_lattice_56_P7      1e-8 $LAT --elements 56 --pn-order 7 --time 0.2
mirror_case mirror_latticefcs_56_P7   1e-8 $LAT --elements 56 --pn-order 7 --time 0.2 --fcs analytic
# lattice-3d is deliberately asymmetric (no mirror); the 3D smooth pulse with vacuum inflow mirrors in y and z.
mirror_case mirror_smooth3d_q2_12_P3  1e-8 --case smooth --inflow vacuum --dimension 3 --elements 12 --degree 2 \
     --pn-order 3 --sigma-s 1 --time 0.1

if [[ "${GPU_VALIDATE_TIMING:-0}" == "1" ]]; then
  echo "timing: lattice FCS 224^2 P15 T=3.2 on the GPU (reference 16.11.2026: table 1.5 s, solver 21.6 s, mass 2.072102)"
  rm -rf "$OUT/latticefcs_224_P15_gpu"
  # shellcheck disable=SC2086
  if "$B" $EW $LAT --elements 224 --pn-order 15 --time 3.2 --fcs analytic --gpu --output "$OUT/latticefcs_224_P15_gpu" \
       > "$OUT/latticefcs_224_P15_gpu.log" 2>&1; then
    python3 -c "import json,sys;d=json.load(open(sys.argv[1]));l=d['lattice_fcs'];print('  table %.1f s solver %.1f s total %.1f s mass %.6f absorbed %.6f rate %.6f'%(l['table_seconds'],d['solver_seconds'],d['total_seconds'],l['total_mass_final'],l['total_absorbed'],l['total_absorption_rate_final']))" "$OUT/latticefcs_224_P15_gpu/summary.json"
  else
    echo "  timing run failed"; tail -3 "$OUT/latticefcs_224_P15_gpu.log"
  fi
fi

if [[ $FAILED -eq 0 ]]; then echo "GPU validation: all cases PASS"; else echo "GPU validation: $FAILED FAILED:$FAILED_CASES"; fi
exit $FAILED
