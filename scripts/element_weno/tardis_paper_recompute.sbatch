#!/usr/bin/env bash
#SBATCH --job-name=tardis-paper
#SBATCH --partition=med
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/home/user/bberg/phd/03_projects/M1PN_Solver/nx/tmp/tardis_paper_20261120/slurm-%j.log

# Recomputes all tardis (NVIDIA A100) paper benchmarks on the current standard
# (experiment/element-weno: weno_q=3 default, exact adaptive-PN submatrix mass solve,
# 4x faster GPU row-wise FCS table, split-breaks, graded meshes) and the adaptive-PN GPU fix of 20.11.2026
# (fix/gpu-adaptive-consistency: GPU adaptive PN equals the CPU; validated in-job before the runs).
#
# Restartable: any run with an existing summary.json is skipped.
set -uo pipefail

REPO=/home/user/bberg/phd/03_projects/M1PN_Solver
CAMPAIGN=$REPO/nx/tmp/tardis_paper_20261120
RUNS=$CAMPAIGN/runs
mkdir -p "$RUNS"

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1

echo "=== tardis-paper start $(date) on $(hostname) ==="
git -C "$REPO" rev-parse --short HEAD
git -C "$REPO" status -sb
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. Fresh GPU build inside the job (touch sources for NFS timestamps)
find "$REPO/src" -type f -exec touch {} +
GPU=1 BUILD_JOBS=32 bash "$REPO/scripts/build.sh" "$CAMPAIGN/build_gpu" > "$CAMPAIGN/build_gpu.log" 2>&1 || {
    echo "BUILD FAILED; tail of build_gpu.log:"
    tail -n 40 "$CAMPAIGN/build_gpu.log"
    exit 1
}
BIN="$CAMPAIGN/build_gpu/m1pn_fast"
sha256sum "$BIN" > "$CAMPAIGN/binary.sha256"
echo "Built $BIN ($(cat "$CAMPAIGN/binary.sha256"))"

# 2. CTest on the tardis node
(cd "$CAMPAIGN/build_gpu" && ctest --output-on-failure) > "$CAMPAIGN/ctest.log" 2>&1
echo "ctest rc=$? ($(date +%T))"

# 3. CPU vs GPU validation of adaptive PN with all features (31 cases); abort if any fails.
bash "$REPO/tests/gpu_validate_adaptive.sh" "$BIN" "$CAMPAIGN/gpu_validate_adaptive" > "$CAMPAIGN/gpu_validate_adaptive.log" 2>&1
VRC=$?
tail -n 1 "$CAMPAIGN/gpu_validate_adaptive.log"
if [ $VRC -ne 0 ]; then
    echo "ADAPTIVE GPU VALIDATION FAILED (rc=$VRC); see gpu_validate_adaptive.log"
    exit 1
fi

run_gpu() {
    local name=$1; shift
    if [ -f "$RUNS/$name/summary.json" ]; then
        echo "[skip] $name"
        return 0
    fi
    rm -rf "$RUNS/$name"
    local t0=$(date +%s)
    "$BIN" --threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low \
        --memory-mib 65536 --gpu "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    echo "[done] $name rc=$rc ${dt}s $(date +%T)"
    return 0
}

run_cpu() {
    local name=$1; shift
    if [ -f "$RUNS/$name/summary.json" ]; then
        echo "[skip] $name"
        return 0
    fi
    rm -rf "$RUNS/$name"
    local t0=$(date +%s)
    "$BIN" --threads 32 --cfl 1.0 --dt 0 --angular-reduction auto --sample-moments low \
        --memory-mib 65536 "$@" --output "$RUNS/$name" > "$RUNS/$name.log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    echo "[done] $name rc=$rc ${dt}s $(date +%T)"
    return 0
}

# ==============================================================================
# BLOCK A: Table 9 — Dynamic tile-based adaptive PN on GBQ, GBH, GB (A100)
#          + 3 generalization checks (line source, crossing, lattice-3d)
# ==============================================================================
echo "=== BLOCK A: Dynamic adaptive PN (Table 9) $(date +%T) ==="
LAT_ADAPT="--case lattice-paul --inflow vacuum --dimension 2 --elements 7 --degree 1 --background 1e-10 --fcs analytic --break-units on --split-breaks on --mirror-reduction auto --time 3.2"
GBQ="--cells-per-unit 8,16,32,32,32,16,8"
GBH="--cells-per-unit 16,32,64,64,64,32,16"
GB="--cells-per-unit 32,64,128,128,128,64,32"

# GBQ (224^2 central)
run_gpu dyn_gbq_p21_uni       $LAT_ADAPT $GBQ --pn-order 21
run_gpu dyn_gbq_p21_t16       $LAT_ADAPT $GBQ --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 16
run_gpu dyn_gbq_p21_t8        $LAT_ADAPT $GBQ --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8
run_gpu dyn_gbq_p21_t8_f01    $LAT_ADAPT $GBQ --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8 --adaptive-floor 0.1
run_gpu dyn_gbq_p35_uni       $LAT_ADAPT $GBQ --pn-order 35
run_gpu dyn_gbq_p35_t16       $LAT_ADAPT $GBQ --pn-order 35 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 16
run_gpu dyn_gbq_p35_t8        $LAT_ADAPT $GBQ --pn-order 35 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8
run_gpu dyn_gbq_p35_1e4_t8    $LAT_ADAPT $GBQ --pn-order 35 --adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 8
run_gpu dyn_gbq_p35_t8_f01    $LAT_ADAPT $GBQ --pn-order 35 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8 --adaptive-floor 0.1

# GBH (448^2 central)
run_gpu dyn_gbh_p21_uni       $LAT_ADAPT $GBH --pn-order 21
run_gpu dyn_gbh_p21_t16       $LAT_ADAPT $GBH --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 16
run_gpu dyn_gbh_p21_t8        $LAT_ADAPT $GBH --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8
run_gpu dyn_gbh_p21_t8_f01    $LAT_ADAPT $GBH --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8 --adaptive-floor 0.1
run_gpu dyn_gbh_p35_uni       $LAT_ADAPT $GBH --pn-order 35
run_gpu dyn_gbh_p35_t16       $LAT_ADAPT $GBH --pn-order 35 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 16
run_gpu dyn_gbh_p35_t8        $LAT_ADAPT $GBH --pn-order 35 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8
run_gpu dyn_gbh_p35_1e4_t8    $LAT_ADAPT $GBH --pn-order 35 --adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 8
run_gpu dyn_gbh_p35_1e4_t16   $LAT_ADAPT $GBH --pn-order 35 --adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 16
run_gpu dyn_gbh_p35_t8_f01    $LAT_ADAPT $GBH --pn-order 35 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8 --adaptive-floor 0.1

# GB (896^2 central; also provides the "GB, broken lines, split stage" rows of Table 8)
run_gpu dyn_gb_p21_uni        $LAT_ADAPT $GB --pn-order 21
run_gpu dyn_gb_p21_t8         $LAT_ADAPT $GB --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8
run_gpu dyn_gb_p21_t8_f01     $LAT_ADAPT $GB --pn-order 21 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8 --adaptive-floor 0.1
run_gpu dyn_gb_p35_uni        $LAT_ADAPT $GB --pn-order 35
run_gpu dyn_gb_p35_1e4_t16    $LAT_ADAPT $GB --pn-order 35 --adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 16
run_gpu dyn_gb_p35_1e4_t8     $LAT_ADAPT $GB --pn-order 35 --adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 8
run_gpu dyn_gb_p35_t8_f01     $LAT_ADAPT $GB --pn-order 35 --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8 --adaptive-floor 0.1

# Generalization checks for dynamic adaptive PN
run_gpu gen_line_p21_uni      --case line-source --inflow vacuum --dimension 2 --elements 128 --degree 1 --pn-order 21 --time 0.35 --sigma-s 3 --pulse-width .01 --background .0009 --fcs analytic --mirror-reduction auto
run_gpu gen_line_p21_adapt    --case line-source --inflow vacuum --dimension 2 --elements 128 --degree 1 --pn-order 21 --time 0.35 --sigma-s 3 --pulse-width .01 --background .0009 --fcs analytic --mirror-reduction auto --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8
run_gpu gen_cross_p21_uni     --case crossing-study --inflow crossing-study-transport --dimension 2 --elements 64 --degree 1 --pn-order 21 --time 0.8 --pulse-width .07 --background .001 --mirror-reduction off
run_gpu gen_cross_p21_adapt   --case crossing-study --inflow crossing-study-transport --dimension 2 --elements 64 --degree 1 --pn-order 21 --time 0.8 --pulse-width .07 --background .001 --mirror-reduction off --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8
run_gpu gen_l3d_p7_uni        --case lattice-3d --inflow vacuum --dimension 3 --elements 14 --degree 1 --pn-order 7 --time 1.0 --background 1e-10 --mirror-reduction off
run_gpu gen_l3d_p7_adapt      --case lattice-3d --inflow vacuum --dimension 3 --elements 14 --degree 1 --pn-order 7 --time 1.0 --background 1e-10 --mirror-reduction off --adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 4

# ==============================================================================
# BLOCK B: Table 8 — Graded mesh GB, broken lines, split stage (mirror-x, FCS, no filter)
# ==============================================================================
echo "=== BLOCK B: Graded mesh GB & broken lines (Table 8) $(date +%T) ==="
LAT_T8="--case lattice-paul --inflow vacuum --dimension 2 --degree 1 --background 1e-10 --fcs analytic --mirror-reduction auto --time 3.2"
for P in 21 35; do
    run_gpu t8_u896_p${P}         $LAT_T8 --elements 896 --pn-order $P
    run_gpu t8_gb_p${P}           $LAT_T8 --elements 7 $GB --pn-order $P
    run_gpu t8_gb_broken_p${P}    $LAT_T8 --elements 7 $GB --pn-order $P --break-units on
    # split-stage rows are already computed as dyn_gb_p21_uni and dyn_gb_p35_uni in Block A
done

# ==============================================================================
# BLOCK C: Crossing beams & Ablation (Table 12)
# ==============================================================================
echo "=== BLOCK C: Crossing beams & Ablation $(date +%T) ==="
CR="--case crossing-study --inflow crossing-study-transport --dimension 2 --time 0.32 --pulse-width .07 --background .001 --mirror-reduction off"
for P in 7 15; do
    run_gpu cross_q1_512_p${P}    $CR --degree 1 --elements 512 --pn-order $P
    run_gpu cross_q2_256_p${P}    $CR --degree 2 --elements 256 --pn-order $P
done
run_gpu abl_full         $CR --degree 2 --elements 256 --pn-order 7 --time-integrator ssprk43
run_gpu abl_ssprk3       $CR --degree 2 --elements 256 --pn-order 7 --time-integrator ssprk3
run_gpu abl_weno_02      $CR --degree 2 --elements 256 --pn-order 7 --time-integrator ssprk43 --weno-linear 0.2
run_gpu abl_no_limit     $CR --degree 2 --elements 256 --pn-order 7 --time-integrator ssprk43 --element-limit off

# ==============================================================================
# BLOCK D: Table 10 — 3D lattice A100 runs
# ==============================================================================
echo "=== BLOCK D: 3D Lattice A100 (Table 10) $(date +%T) ==="
L3="--case lattice-3d --inflow vacuum --dimension 3 --degree 1 --time 3.2 --background 1e-10 --mirror-reduction off --skip-fields"
run_gpu l3d_p7_56_nofcs           $L3 --elements 56  --pn-order 7
run_gpu l3d_p7_112_nofcs          $L3 --elements 112 --pn-order 7
run_gpu l3d_p7_168_nofcs          $L3 --elements 168 --pn-order 7
run_gpu l3d_p7_56_fcs             $L3 --elements 56  --pn-order 7 --fcs analytic
run_gpu l3d_p7_112_fcs            $L3 --elements 112 --pn-order 7 --fcs analytic
run_gpu l3d_p7_112_fcs_filter     $L3 --elements 112 --pn-order 7 --fcs analytic --filter constant --filter-strength 5.2

# ==============================================================================
# BLOCK D2: 3D lattice with the 2D improvements that exist in 3D (FCS + broken material lines + split stage,
#           dynamic adaptive PN). Graded meshes in 3D follow in a separate job once implemented.
# ==============================================================================
echo "=== BLOCK D2: 3D Lattice, broken lines / split / adaptive PN $(date +%T) ==="
BRK3="--break-units on --split-breaks on"
AD3="--adaptive-pn on --adaptive-tol 3e-4 --adaptive-tile 8"
for E in 56 112 168; do
    run_gpu l3d_p7_${E}_fcs_brk        $L3 --elements $E --pn-order 7 --fcs analytic $BRK3
    run_gpu l3d_p7_${E}_fcs_ad         $L3 --elements $E --pn-order 7 --fcs analytic $AD3
    run_gpu l3d_p7_${E}_fcs_brk_ad     $L3 --elements $E --pn-order 7 --fcs analytic $BRK3 $AD3
done
run_gpu l3d_p7_168_fcs                $L3 --elements 168 --pn-order 7 --fcs analytic
for E in 56 112; do
    run_gpu l3d_p15_${E}_fcs           $L3 --elements $E --pn-order 15 --fcs analytic
    run_gpu l3d_p15_${E}_fcs_ad        $L3 --elements $E --pn-order 15 --fcs analytic $AD3
    run_gpu l3d_p15_${E}_fcs_brk_ad    $L3 --elements $E --pn-order 15 --fcs analytic $BRK3 $AD3
done

# ==============================================================================
# BLOCK E: Table 3 — Line source 512^2 & 1024^2 (GPU)
# ==============================================================================
echo "=== BLOCK E: Line source (Table 3) $(date +%T) ==="
LS="--case line-source --inflow vacuum --dimension 2 --degree 1 --time 0.33333333333333333 --sigma-s 3 --pulse-width .01 --background .0009 --mirror-reduction off"
for P in 3 7 11; do
    run_gpu ls_512_p${P}_nofcs        $LS --elements 512 --pn-order $P
    run_gpu ls_512_p${P}_nofcs_f33    $LS --elements 512 --pn-order $P --filter constant --filter-strength 99.0
done
for P in 7 15 21 35 45 63; do
    run_gpu ls_512_p${P}_fcs          $LS --elements 512 --pn-order $P --fcs analytic
    run_gpu ls_512_p${P}_fcs_filt     $LS --elements 512 --pn-order $P --fcs analytic --filter constant --filter-strength 49.56
done
for P in 7 15 21 35; do
    run_gpu ls_1024_p${P}_fcs         $LS --elements 1024 --pn-order $P --fcs analytic
    run_gpu ls_1024_p${P}_fcs_filt    $LS --elements 1024 --pn-order $P --fcs analytic --filter constant --filter-strength 49.56
done
run_gpu ls_96_p7_fcs                  $LS --elements 96 --pn-order 7 --fcs analytic

# ==============================================================================
# BLOCK F: Table 4 & 5 — 2D Uniform Lattice (A100, mirror-off as in paper Table 4)
# ==============================================================================
echo "=== BLOCK F: 2D Uniform Lattice (Tables 4 & 5) $(date +%T) ==="
LA="--case lattice-paul --inflow vacuum --dimension 2 --degree 1 --background 1e-10 --time 3.2 --mirror-reduction off"
# Without FCS (Table 4 + Table 5 StaRMAP comparison)
run_gpu lat_224_p7_nofcs          $LA --elements 224 --pn-order 7
run_gpu lat_448_p7_nofcs          $LA --elements 448 --pn-order 7
run_gpu lat_896_p7_nofcs          $LA --elements 896 --pn-order 7
run_gpu lat_448_p15_nofcs         $LA --elements 448 --pn-order 15
run_gpu lat_896_p15_nofcs         $LA --elements 896 --pn-order 15
run_gpu lat_448_p21_nofcs         $LA --elements 448 --pn-order 21
run_gpu lat_896_p21_nofcs         $LA --elements 896 --pn-order 21
# With FCS and FCS + filter (sigma_f * T = 5.2 * 3.2 = 16.64)
run_gpu lat_896_p7_fcs            $LA --elements 896  --pn-order 7  --fcs analytic
run_gpu lat_896_p7_fcs_filt       $LA --elements 896  --pn-order 7  --fcs analytic --filter constant --filter-strength 5.2
run_gpu lat_896_p11_fcs           $LA --elements 896  --pn-order 11 --fcs analytic
run_gpu lat_896_p11_fcs_filt      $LA --elements 896  --pn-order 11 --fcs analytic --filter constant --filter-strength 5.2
run_gpu lat_896_p21_fcs           $LA --elements 896  --pn-order 21 --fcs analytic
run_gpu lat_896_p21_fcs_filt      $LA --elements 896  --pn-order 21 --fcs analytic --filter constant --filter-strength 5.2
run_gpu lat_896_p35_fcs           $LA --elements 896  --pn-order 35 --fcs analytic
run_gpu lat_896_p35_fcs_filt      $LA --elements 896  --pn-order 35 --fcs analytic --filter constant --filter-strength 5.2
run_gpu lat_896_p45_fcs_filt      $LA --elements 896  --pn-order 45 --fcs analytic --filter constant --filter-strength 5.2
run_gpu lat_1344_p21_fcs          $LA --elements 1344 --pn-order 21 --fcs analytic
run_gpu lat_1344_p21_fcs_filt     $LA --elements 1344 --pn-order 21 --fcs analytic --filter constant --filter-strength 5.2

# ==============================================================================
# REPORT GENERATION
# ==============================================================================
echo "=== Generating REPORT.md $(date +%T) ==="
python3 "$REPO/scripts/element_weno/tardis_paper_report.py" "$CAMPAIGN" > "$CAMPAIGN/REPORT.md" 2>&1
cat "$CAMPAIGN/REPORT.md"
echo "=== tardis-paper finished $(date) ==="
