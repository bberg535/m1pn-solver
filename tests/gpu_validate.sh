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

EW="--stabilization element-weno --time-integrator ssprk43 --cfl 0.2 --dt 0 --model r2 --mesh-type hex
    --threads $THREADS --mirror-reduction off --angular-reduction auto --sample-moments low"
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
