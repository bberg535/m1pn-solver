#!/usr/bin/env bash
# GPU validation of ADAPTIVE PN (tile-based, --adaptive-pn on): CPU vs --gpu with the same m1pn_fast binary and the
# same dt. Not covered by tests/gpu_validate.sh (no adaptive case there). Covers 2D lattice and 3D lattice-3d.
#
# Submit on tardis (one A100), only when no other GPU job is running:
#   sbatch tests/gpu_validate_adaptive.sh [M1PN_FAST] [OUTDIR]
#
# Exit status: number of failed cases (0 = all pass). Tolerance 1e-8 as in gpu_validate.sh.
#SBATCH --job-name=m1pn-gpu-adapt
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=gpu_validate_adaptive-%j.log
set -u

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then ROOT=$SLURM_SUBMIT_DIR; else ROOT=$(cd "$(dirname "$0")/.." && pwd); fi
CMP="$ROOT/tests/gpu_validate_cmp.py"
B=${1:-$ROOT/tmp/build_gpu/m1pn_fast}
OUT=${2:-$ROOT/tmp/gpu_validate_adaptive/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
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
AD="--adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 4"
FAILED=0
FAILED_CASES=""

# pair TAG TOL ARGS...  (CPU run, then GPU run with the CPU dt, then compare densities)
pair() {
  local tag=$1 tol=$2; shift 2
  rm -rf "$OUT/${tag}_cpu" "$OUT/${tag}_gpu"
  echo "$tag"
  # shellcheck disable=SC2086
  if ! "$B" $EW "$@" --output "$OUT/${tag}_cpu" > "$OUT/${tag}_cpu.log" 2>&1; then
    echo "  FAIL: cpu run failed"; tail -3 "$OUT/${tag}_cpu.log"; FAILED=$((FAILED+1)); FAILED_CASES+=" $tag"; return
  fi
  local dt_cpu
  dt_cpu=$(python3 -c "import json,sys; print(repr(json.load(open(sys.argv[1]))['dt_selected']))" "$OUT/${tag}_cpu/config.json")
  # shellcheck disable=SC2086
  if ! "$B" $EW "$@" --dt "$dt_cpu" --gpu --output "$OUT/${tag}_gpu" > "$OUT/${tag}_gpu.log" 2>&1; then
    echo "  FAIL: gpu run failed"; tail -3 "$OUT/${tag}_gpu.log"; FAILED=$((FAILED+1)); FAILED_CASES+=" $tag"; return
  fi
  # shellcheck disable=SC2086
  python3 "$CMP" "$OUT/${tag}_cpu" "$OUT/${tag}_gpu" --max-rel "$tol" || { FAILED=$((FAILED+1)); FAILED_CASES+=" $tag"; }
}

LAT="--case lattice-paul --dimension 2 --inflow vacuum --background 1e-10 --degree 1 --time 0.2"
GR="--elements 7 --cells-per-unit 2,4,8,8,8,4,2"
L3="--case lattice-3d --dimension 3 --inflow vacuum --background 1e-10 --mirror-reduction off"
LS="--case line-source --dimension 2 --inflow vacuum --pulse-width .01 --background .0009 --sigma-s 3 --time 0.02"
CR="--case crossing-study --inflow crossing-study-transport --dimension 2 --pulse-width .07 --background .001 --time 0.05"

# both TAG ARGS...: the same configuration uniform (TAG_uni) and with adaptive PN (TAG_adapt)
both() { local tag=$1; shift; pair "${tag}_uni" 1e-8 "$@"; pair "${tag}_adapt" 1e-8 "$@" $AD; }

# 2D lattice: features of 18./19.11. added one by one (graded mesh, broken lines, split stage, FCS, mirror)
both lat_56_P7            $LAT --elements 56 --pn-order 7
both lat_graded_P7        $LAT $GR --pn-order 7
both lat_graded_brk_P7    $LAT $GR --pn-order 7 --break-units on
both lat_graded_split_P7  $LAT $GR --pn-order 7 --break-units on --split-breaks on
both lat_full_P7          $LAT $GR --pn-order 7 --break-units on --split-breaks on --fcs analytic --mirror-reduction auto
both lat_full_P15         $LAT $GR --pn-order 15 --break-units on --split-breaks on --fcs analytic --mirror-reduction auto
pair lat_full_P15_floor 1e-8 $LAT $GR --pn-order 15 --break-units on --split-breaks on --fcs analytic \
     --mirror-reduction auto $AD --adaptive-floor 0.1
# other 2D cases
both line_fcs_mirror_P7   $LS --elements 128 --pn-order 7 --degree 1 --fcs analytic --mirror-reduction auto
both cross_64_P7          $CR --elements 64 --pn-order 7 --degree 1
both cross_q2_32_P7       $CR --elements 32 --pn-order 7 --degree 2
# 3D lattice
both l3d_14_P3            $L3 --elements 14 --degree 1 --pn-order 3 --time 0.1
both l3d_14_P7            $L3 --elements 14 --degree 1 --pn-order 7 --time 0.1
both l3d_q2_14_P3         $L3 --elements 14 --degree 2 --pn-order 3 --time 0.1
both l3d_fcs_28_P3        $L3 --elements 28 --degree 1 --pn-order 3 --time 0.5 --fcs analytic
# graded meshes are 1D/2D only (--cells-per-unit); 3D: broken lines + split stage + FCS on the uniform mesh
both l3d_brk_split_fcs_P3 $L3 --elements 14 --degree 1 --pn-order 3 --time 0.1 --break-units on --split-breaks on \
     --fcs analytic

if [[ $FAILED -eq 0 ]]; then echo "GPU adaptive validation: all cases PASS"; else echo "GPU adaptive validation: $FAILED FAILED:$FAILED_CASES"; fi
exit $FAILED
