#!/usr/bin/env python3
"""Local DLRA campaign (docs/DLRA.md): full-rank element WENO against DLRA at high P_N.

Usage: campaign.py BIN OUTDIR [--only PREFIX] [--threads 6] [--list]

Families (all with --sample-moments low; one run at a time, OMP_NUM_THREADS = threads):
  LS   GHS line source with first-collision source (docs/FCS.md), Q1/128 (129^2 nodes), T = 1/3.
       full: mirror reduction auto (quarter domain, production) and off (same domain as DLRA).
  LAT  Lattice (Paul (44)-(47)), Q1/112, T = 3.2, SSPRK(4,3), cfl 0.2 (settings of the ew_hetero_paper campaign).
  CR   Crossing study (pure transport, kinetic reference), Q2/64, T = 0.32.
Each run directory gets summary.json etc.; evaluation: line_source_error.py, crossing_error.py, lattice_analysis.py.
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

LS = ("--case line-source --dimension 2 --pulse-width .01 --background 9e-4 --sigma-s 3 --time 0.3333333333333333 "
      "--stabilization element-weno --fcs analytic --degree 1 --elements 128")
LAT = ("--case lattice-paul --dimension 2 --inflow vacuum --stabilization element-weno --degree 1 --elements 112 "
       "--background 1e-10 --time 3.2 --time-integrator ssprk43 --cfl 0.2")
CR = "--case crossing-study --dimension 2 --stabilization element-weno --degree 2 --elements 64 --time 0.32"


def tasks():
    out = []
    for n in (7, 11, 15, 21, 31, 41):
        out.append((f"LS_full_mauto_P{n}", f"{LS} --pn-order {n} --mirror-reduction auto"))
    for n in (15, 21, 31, 41):
        out.append((f"LS_full_moff_P{n}", f"{LS} --pn-order {n} --mirror-reduction off"))
    # Fixed rank, increasing N (Peng, McClarren, Frank, JCP 421 (2020): more basis functions at fixed rank).
    for n in (15, 21, 31, 41):
        for r in (16, 32, 64):
            out.append((f"LS_dlra_r{r}_P{n}", f"{LS} --pn-order {n} --dlra on --dlra-fixed-rank on --dlra-rank {r}"))
    # Rank-adaptive, tolerance per unit time (theta = T ||a|| dt).
    for n in (15,):
        for tol in ("1",):
            out.append((f"LS_dlra_rate{tol}_P{n}",
                        f"{LS} --pn-order {n} --dlra on --dlra-tolerance-mode rate --dlra-tolerance {tol}"))
    for n in (3, 7, 11, 15, 21):
        out.append((f"LAT_full_mauto_P{n}", f"{LAT} --pn-order {n} --mirror-reduction auto"))
    for n in (11, 21):
        out.append((f"LAT_full_moff_P{n}", f"{LAT} --pn-order {n} --mirror-reduction off"))
    for n in (11, 21, 31):
        for r in (8, 16, 32):
            out.append((f"LAT_dlra_r{r}_P{n}", f"{LAT} --pn-order {n} --dlra on --dlra-fixed-rank on --dlra-rank {r}"))
        out.append((f"LAT_dlra_rate1_P{n}", f"{LAT} --pn-order {n} --dlra on --dlra-tolerance-mode rate --dlra-tolerance 1"))
    for n in (7, 11, 15, 21, 31):
        out.append((f"CR_full_P{n}", f"{CR} --pn-order {n}"))
    for n in (15, 31):
        for r in (16, 32, 64):
            out.append((f"CR_dlra_r{r}_P{n}", f"{CR} --pn-order {n} --dlra on --dlra-fixed-rank on --dlra-rank {r}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin")
    ap.add_argument("out")
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks() if t[0].startswith(a.only)]
    with open(out / "tasks.tsv", "w") as f:
        for name, args in tasks():
            f.write(f"{name}\t{args}\n")
    if a.list:
        for name, args in todo:
            print(name, args)
        return
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads))
    for name, args in todo:
        run = out / "runs" / name
        if (run / "summary.json").exists():
            continue
        start = time.time()
        cmd = [a.bin, *args.split(), "--sample-moments", "low", "--output", str(run)]
        with open(out / "runs" / f"{name}.log", "w") as log:
            code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{name}: exit {code}, {time.time() - start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
