#!/bin/bash
#SBATCH --partition=med --constraint=ampere --gres=gpu:1 --cpus-per-task=32 --time=03:00:00 --job-name=block4_prof
#SBATCH --output=tmp/block4_profile/slurm-%j.out
set -euo pipefail
D=tmp/block4_profile
GPU=1 BUILD_JOBS=32 bash scripts/build.sh tmp/build_gpu >$D/build.log 2>&1
B=$D/m1pn_fast; cp tmp/build_gpu/m1pn_fast $B; sha256sum $B >$D/binary.sha256
nvidia-smi --query-gpu=name,driver_version --format=csv >$D/gpu.txt; lscpu | grep -E "Model name|^CPU\(s\)" >>$D/gpu.txt
run() { local name=$1; shift; echo "== $name"; /usr/bin/time -v $B "$@" --gpu --output $D/$name >$D/$name.log 2>&1 || echo "FAILED $name"; }
L3="--case lattice-3d --dimension 3 --inflow vacuum --background 1e-10 --elements 56 --degree 1 --time 1"
run l3d_P7      $L3 --pn-order 7
run l3d_P7_fcs  $L3 --pn-order 7 --fcs analytic
run l3d_P15     $L3 --pn-order 15
run l3d_P15_fcs $L3 --pn-order 15 --fcs analytic
L2="--case lattice-paul --dimension 2 --inflow vacuum --background 1e-10 --degree 1 --pn-order 35 --time 1"
run l2d_P35_E224 $L2 --elements 224
run l2d_P35_E448 $L2 --elements 448
# Nsight timeline of a short 3D run (if available)
NSYS=$(ls /sfw/cuda/12.6/bin/nsys /sfw/cuda/12.6/nsight-systems*/bin/nsys 2>/dev/null | head -1 || true)
if [[ -n "$NSYS" ]]; then
  $NSYS profile -o $D/l3d_P15_nsys --force-overwrite true -t cuda,nvtx,osrt $B $L3 --pn-order 15 --time .1 --gpu --output $D/l3d_P15_nsys_run >$D/nsys.log 2>&1 || echo "nsys failed"
  $NSYS stats --report cuda_gpu_kern_sum,cuda_api_sum $D/l3d_P15_nsys.nsys-rep >$D/nsys_stats.txt 2>&1 || true
fi
python3 - <<'PY'
import json,glob,os
for d in sorted(glob.glob("tmp/block4_profile/l*/summary.json")):
    s=json.load(open(d)); g=s.get("gpu_timing",{})
    print(os.path.basename(os.path.dirname(d)), "steps",s["steps"],"solver %.1f"%s["solver_seconds"],
          " ".join(f"{k} {v:.1f}" for k,v in g.items() if isinstance(v,(int,float))))
PY
echo done
