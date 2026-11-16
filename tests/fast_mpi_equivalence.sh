#!/usr/bin/env bash
# m1pn_fast element WENO with MPI against one rank: initial/final fields (and 3D slices) bitwise, summary sums up
# to rounding (1e-12). With --dt 0 and an active high-order bound (Q2+) dt may differ in the last bits between rank
# counts (power iteration norms summed per rank): then fields within 1e-8.
#
#   sbatch tests/fast_mpi_equivalence.sh [M1PN_FAST] [OUTDIR]       (one NX node, 2 sockets)
#
# Cases: crossing 2D Q1/Q2, lattice-paul with and without FCS, line source with FCS, lattice-3d with and without FCS,
# shadow-weih. One rank (32 threads), 2 ranks (one per socket, 32 threads), 4 ranks (16 threads). Mirror reduction off
# (MPI does not combine with mirror sides). Exit status: number of failed comparisons.
#
#SBATCH --job-name=fast-mpi-eq
#SBATCH --partition=long
#SBATCH --constraint=nx
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=fast_mpi_equivalence-%j.log
set -u
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then ROOT=$SLURM_SUBMIT_DIR; else ROOT=$(cd "$(dirname "$0")/.." && pwd); fi
B=${1:-$ROOT/tmp/build_cpu/m1pn_fast}
OUT=${2:-$ROOT/tmp/fast_mpi_equivalence/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
mkdir -p "$OUT"
echo "binary: $B"; echo "output: $OUT"; echo "git: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)"
export OMP_PLACES=cores OMP_PROC_BIND=close
MPIRUN=${MPIRUN:-/home/warehouse15/bberg/spack/opt/spack/linux-zen3/openmpi-5.0.10-kfdcvl5tog3tsi5rusglshnfcauemfjq/bin/mpirun}  # must match the MPI of the binary (scripts/build.sh: UCX OpenMPI)
COMMON="--stabilization element-weno --mirror-reduction off"
FAILED=0

run_case() { # tag args...
  local tag=$1; shift
  echo "$tag"
  rm -rf "$OUT/${tag}_r1" "$OUT/${tag}_r2" "$OUT/${tag}_r4"
  OMP_NUM_THREADS=32 "$B" $COMMON "$@" --threads 32 --output "$OUT/${tag}_r1" > "$OUT/${tag}_r1.log" 2>&1 \
    || { echo "  FAIL: 1 rank"; tail -3 "$OUT/${tag}_r1.log"; FAILED=$((FAILED+1)); return; }
  "$MPIRUN" -np 2 --map-by ppr:1:package:PE=32 --bind-to core -x OMP_NUM_THREADS=32 -x OMP_PLACES -x OMP_PROC_BIND \
    "$B" $COMMON "$@" --threads 32 --output "$OUT/${tag}_r2" > "$OUT/${tag}_r2.log" 2>&1 \
    || { echo "  FAIL: 2 ranks"; tail -3 "$OUT/${tag}_r2.log"; FAILED=$((FAILED+1)); return; }
  "$MPIRUN" -np 4 --map-by ppr:2:package:PE=16 --bind-to core -x OMP_NUM_THREADS=16 -x OMP_PLACES -x OMP_PROC_BIND \
    "$B" $COMMON "$@" --threads 16 --output "$OUT/${tag}_r4" > "$OUT/${tag}_r4.log" 2>&1 \
    || { echo "  FAIL: 4 ranks"; tail -3 "$OUT/${tag}_r4.log"; FAILED=$((FAILED+1)); return; }
  python3 - "$OUT/${tag}" <<'EOF' || FAILED=$((FAILED+1))
import json, sys, filecmp, os
base = sys.argv[1]
def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, p + k + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool): out[p + k] = v
    return out
keys = ["final_density_mass", "initial_density_mass", "ledger.injected", "ledger.absorbed", "ledger.boundary_outflow",
        "ledger.nudge_mass", "absorption_rate_final", "uncollided_mass_final", "minimum_coefficient_density",
        "minimum_cone_margin", "lattice_fcs.total_mass_final", "lattice_fcs.total_absorbed",
        "lattice_fcs.total_absorption_rate_final", "lattice_fcs.uncollided_mass_final", "lattice_fcs.table_points"]
s1 = flat(json.load(open(base + "_r1/summary.json")))
ok = True
for r in ("r2", "r4"):
    s = flat(json.load(open(base + f"_{r}/summary.json")))
    worst, which = 0.0, ""
    for k in keys:
        if k not in s1: continue
        a, b = s1[k], s[k]
        d = abs(a - b) / max(abs(a), 1e-300) if a != 0 else abs(b)
        if d > worst: worst, which = d, k
    steps = s1["steps"] == s["steps"]
    slices = all(filecmp.cmp(base + "_r1/" + f, base + f"_{r}/" + f, shallow=False)
                 for f in ("initial.csv", "final.csv", "slices.csv") if os.path.exists(base + "_r1/" + f))
    # With --dt 0 the high-order step comes from a power iteration whose norms are summed per rank: dt may differ in
    # the last bits when that bound is active (Q2+). Then fields within 1e-8, sums within 1e-6; otherwise bitwise.
    c1, c = json.load(open(base + "_r1/config.json")), json.load(open(base + f"_{r}/config.json"))
    same_dt = c1["dt_selected"] == c["dt_selected"]
    if same_dt:
        good = worst <= 1e-12 and steps and slices
        fields = "bitwise" if slices else "DIFFER"
    else:
        import csv
        dens = lambda q: [float(x["u0"]) for x in csv.DictReader(open(q))]
        a, b = dens(base + "_r1/final.csv"), dens(base + f"_{r}/final.csv")
        dev = max(abs(x - y) for x, y in zip(a, b)) / max(abs(x) for x in a)
        good = worst <= 1e-6 and steps and dev <= 1e-8
        fields = f"dt differs by {abs(c1['dt_selected'] - c['dt_selected']) / c1['dt_selected']:.0e}, density {dev:.1e}"
    ok &= good
    print(f"  {r}: summary max rel {worst:.1e} ({which}) | steps {'=' if steps else '!='} | "
          f"fields {fields} | solver {s1['solver_seconds']:.1f} -> {s['solver_seconds']:.1f} s"
          + ("" if good else "  FAIL"))
sys.exit(0 if ok else 1)
EOF
}
LAT="--case lattice-paul --dimension 2 --inflow vacuum --background 1e-10 --degree 1"
L3D="--case lattice-3d --dimension 3 --inflow vacuum --background 1e-10 --degree 1"
run_case crossing_q1_128_P7 --case crossing-study --inflow crossing-study-transport --dimension 2 --elements 128 \
    --degree 1 --pn-order 7 --time 0.1
run_case crossing_q2_64_P7 --case crossing-study --inflow crossing-study-transport --dimension 2 --elements 64 \
    --degree 2 --pn-order 7 --time 0.1
run_case crossing_q2_64_P7_fixeddt --case crossing-study --inflow crossing-study-transport --dimension 2 \
    --elements 64 --degree 2 --pn-order 7 --time 0.1 --dt 0.002
run_case lattice_112_P7 $LAT --elements 112 --pn-order 7 --time 1.0
run_case latticefcs_112_P7 $LAT --elements 112 --pn-order 7 --time 1.0 --fcs analytic
run_case linefcs_q1_128_P7 --case line-source --dimension 2 --inflow vacuum --pulse-width .01 --background .0009 \
    --sigma-s 3 --time 0.2 --elements 128 --degree 1 --pn-order 7 --fcs analytic
run_case lattice3d_28_P3 $L3D --elements 28 --pn-order 3 --time 1.0
run_case lattice3dfcs_28_P3 $L3D --elements 28 --pn-order 3 --time 1.0 --fcs analytic
run_case shadow_8_P3 --case shadow-weih --dimension 3 --inflow shadow-beam --elements 8 --degree 1 --pn-order 3 \
    --time 0.3
if [[ $FAILED -eq 0 ]]; then echo "MPI equivalence: all cases PASS"; else echo "MPI equivalence: $FAILED FAILED"; fi
exit $FAILED
