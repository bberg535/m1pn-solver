#!/usr/bin/env python3
"""Independent 1D isotropic kinetic pulse density after boundary shutdown."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from run_efficiency import ROOT, compare, physical, read_field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    cpu = min(os.sched_getaffinity(0)); results = []
    (out/"plan.json").write_text(json.dumps(dict(time=.16, dt=.0001, duration=.06, elements=32, pn_order=3,
        modes=["direct","defect","pcg"], implicit_rtol=1e-12, implicit_atol=1e-14, max_iterations=300,
        binary_sha256=hashlib.sha256((ROOT/"tmp/build/m1pn").read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+"\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for mode in ("direct", "defect", "pcg"):
        target = out / mode
        cmd = ["taskset", "-c", str(cpu), str(ROOT/"tmp/build/m1pn"), "--dimension", "1", "--elements", "32",
               "--pn-order", "3", "--case", "vacuum", "--inflow", "pulse", "--m1-exterior", "isotropic",
               "--exterior-density", "1", "--pulse-duration", ".06", "--stabilization", "biharmonic-corrected",
               "--biharmonic-gamma", ".004", "--time-integrator", "imex-euler", "--implicit-solver", mode,
               "--implicit-rtol", "1e-12", "--implicit-atol", "1e-14", "--implicit-max-iterations", "300",
               "--mcl-mode", "lazy", "--geometry-cache", "on", "--time", ".16", "--dt", ".0001",
               "--checkpoint-stride", "1", "--threads", "1", "--output", str(target)]
        print("RUN pulse " + mode, flush=True)
        start = time.perf_counter()
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed = time.perf_counter()-start
        target.mkdir(exist_ok=True)
        (target/"stdout.log").write_text(proc.stdout); (target/"stderr.log").write_text(proc.stderr)
        row = dict(mode=mode, command=cmd, returncode=proc.returncode, process_seconds=elapsed)
        if proc.returncode:
            row.update(status="failed", error=proc.stderr)
        else:
            summary = json.loads((target/"summary.json").read_text())
            x, u = read_field(target/"final.csv"); rho = physical(u)[:, 0]
            t = summary["time"]; duration = .06
            truth = .5*np.maximum(0, np.minimum(1, x[:, 0]/(t-duration))-x[:, 0]/t)
            weights = np.loadtxt(target/"nodal_mass.csv", delimiter=",", skiprows=1, usecols=1)
            error = rho-truth
            row.update(status="completed", rho_L1=float(np.dot(weights, abs(error))),
                       rho_L2=float(np.sqrt(np.dot(weights, error**2))), rho_Linf=float(np.max(abs(error))),
                       rho_relative_L1=float(np.dot(weights, abs(error))/np.dot(weights, abs(truth))), **summary)
            try:
                row.update(compare(out/"direct", target))
            except AssertionError:
                row.update(field_gate_pass=False, gate_error="Checkpoint time grids or state shapes differ from direct reference")
        results.append(row)
        (out/"results.json").write_text(json.dumps(results, indent=2)+"\n")
    if results[0]["status"] == "completed":
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            x, _ = read_field(out/"direct/final.csv")
            xs = np.linspace(0, 1, 1001); t=.16
            ax.plot(xs, .5*np.maximum(0, np.minimum(1, xs/(t-.06))-xs/t), "k-", label="Exact kinetic density")
            for row in results:
                if row["status"] == "completed":
                    xx, u = read_field(out/row["mode"]/"final.csv")
                    order = np.argsort(xx[:, 0]); ax.plot(xx[order, 0], physical(u)[order, 0], "o--", ms=3, label=row["mode"])
            ax.set(xlabel="x", ylabel="Density", xlim=(0, .4)); ax.legend(); fig.tight_layout()
            fig.savefig(out/"density.pdf"); fig.savefig(out/"density.png", dpi=160); plt.close(fig)
        except ImportError:
            pass
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__": main()
