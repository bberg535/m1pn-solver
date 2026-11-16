#!/bin/bash
#SBATCH --partition=med --constraint=ampere --gres=gpu:1 --cpus-per-task=32 --mem=200G --time=04:00:00 --job-name=b4_ab
#SBATCH --output=tmp/block4_ab/slurm-%j.out
set -uo pipefail
D=$PWD/tmp/block4_ab
touch src/element_weno.cpp src/element_weno_gpu.cpp src/element_weno_gpu.cu
GPU=1 BUILD_JOBS=32 bash scripts/build.sh tmp/build_gpu >$D/build.log 2>&1 || { tail -40 $D/build.log; exit 1; }
B=$D/m1pn_fast; cp tmp/build_gpu/m1pn_fast $B; sha256sum $B >$D/binary.sha256
echo "== gpu_validate"
bash tests/gpu_validate.sh $B $D/validate 2>&1 | tail -25
export OMP_NUM_THREADS=32 OMP_PROC_BIND=close
run() { local name=$1; shift; $B "$@" --gpu --output $D/$name >$D/$name.log 2>&1 || echo "FAILED $name"; }
L3="--case lattice-3d --dimension 3 --inflow vacuum --background 1e-10 --elements 56 --degree 1 --time 1"
L2="--case lattice-paul --dimension 2 --inflow vacuum --background 1e-10 --degree 1 --pn-order 35 --time 1 --elements 224"
for P in 7 15; do
  run l3d_P${P}_A $L3 --pn-order $P
  M1PN_GPU_BOUNDARY_HIGH=0 run l3d_P${P}_hostA $L3 --pn-order $P
done
run l2d_P35_A $L2
M1PN_GPU_BOUNDARY_HIGH=0 run l2d_P35_hostA $L2
python3 - <<'PY'
import json,csv,os
D="tmp/block4_ab"; P="tmp/block4_profile"
def summ(p):
    s=json.load(open(p+"/summary.json")); g=s["gpu_timing"]
    return s, f"solver {s['solver_seconds']:7.1f}  A {g['host_boundary_high']:6.1f}  B {g['host_boundary']:6.1f}  sync {g['sync']:5.1f}  wait {g['wait']:5.1f}"
def dens(p):
    r=list(csv.DictReader(open(p+"/final.csv"))); return [float(x["u0"]) for x in r]
for base,new,hostA in [("l3d_P7","l3d_P7_A","l3d_P7_hostA"),("l3d_P15","l3d_P15_A","l3d_P15_hostA"),("l2d_P35_E224","l2d_P35_A","l2d_P35_hostA")]:
    for lab,p in [("baseline",f"{P}/{base}"),("B only",f"{D}/{hostA}"),("A+B",f"{D}/{new}")]:
        if os.path.exists(p+"/summary.json"):
            s,t=summ(p); print(f"{base:14} {lab:9} steps {s['steps']}  {t}  mass {s['final_density_mass']:.15g}")
    a,b=dens(f"{D}/{new}"),dens(f"{D}/{hostA}")
    print(f"{'':14} A device vs host: max|du0|/max {max(abs(x-y) for x,y in zip(a,b))/max(abs(x) for x in a):.1e}")
PY
echo done
