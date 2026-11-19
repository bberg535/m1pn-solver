#!/usr/bin/env python3
"""Compare static and dynamic-AMR M1PN lattice runs against the MC field (timings, AMR columns, QoIs, ledger; docs/DYNAMIC_AMR.md)."""
import json
import math
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

MC = dict(mass=2.080890, absorbed=1.119098, rate=0.731555)


def load_mc(mc_path):
    d = np.loadtxt(mc_path, delimiter=",", skiprows=1)
    n = int(d[:, 0].max()) + 1
    ref = np.zeros((n, n))
    ref[d[:, 1].astype(int), d[:, 0].astype(int)] = d[:, 4]
    g, w = np.polynomial.legendre.leggauss(4)
    g, w = (g + 1) / 2, w / 2
    h = 7.0 / n
    c = (np.arange(n)[:, None] + g[None, :]).ravel() * h
    Y, X = np.meshgrid(c, c, indexing="ij")
    pts = np.stack([Y.ravel(), X.ravel()], axis=1)
    return n, ref, ref > 1e-4, pts, w


def field_err(run, n, ref, mask, pts, w):
    fd = np.loadtxt(run / "final.csv", delimiter=",", skiprows=1, usecols=(1, 2, 4))
    xs, ys = np.unique(fd[:, 0]), np.unique(fd[:, 1])
    o = np.lexsort((fd[:, 0], fd[:, 1]))
    Z = (math.sqrt(4 * math.pi) * fd[o, 2]).reshape(len(ys), len(xs))
    f = RegularGridInterpolator((ys, xs), Z, method="linear")
    A = np.einsum("iajb,a,b->ij", f(pts).reshape(n, 4, n, 4), w, w)
    rel = np.abs(A - ref)[mask] / ref[mask]
    return np.abs(A - ref).sum() / ref.sum(), np.median(rel), np.quantile(rel, 0.95)


def main():
    mc_path = Path(sys.argv[1])
    runs = [Path(p) for p in sys.argv[2:] if (Path(p) / "summary.json").exists()]
    n, ref, mask, pts, w = load_mc(mc_path)
    print(f"{'run':24s} {'total':>7s} {'solve':>7s} {'init':>6s} {'amr':>6s} {'rem':>4s} {'nodes':>7s} {'steps':>5s} "
          f"{'frac':>5s} | {'mass':>7s} {'abs':>7s} {'rate':>7s} % | {'L1':>7s} {'med':>7s} {'95%':>7s} | {'bal':>8s}")
    for run in runs:
        s = json.load(open(run / "summary.json"))
        fcs = s.get("lattice_fcs")
        ap = s.get("adaptive_pn", {})
        gt = s.get("gpu_timing", {})
        if fcs:
            dev = [100 * (fcs[k] / MC[m] - 1) for k, m in
                   (("total_mass_final", "mass"), ("total_absorbed", "absorbed"), ("total_absorption_rate_final", "rate"))]
        else:
            dev = [0.0, 0.0, 0.0]
        l1, med, q95 = field_err(run, n, ref, mask, pts, w)
        am = s.get("amr", {})
        c = json.load(open(run / "config.json"))
        nodes = am.get("mean_nodes") if am.get("enabled") else c["coefficients"]
        amr_s = sum(am.get("seconds", {}).values()) if am.get("enabled") else 0
        print(f"{run.name:24s} {s['total_seconds']:6.1f}s {s['solver_seconds']:6.1f}s {s['initialization_seconds']:5.1f}s "
              f"{amr_s:5.1f}s {am.get('remeshes', 0):4d} {nodes:7.0f} {s['steps']:5d} "
              f"{ap.get('mean_coeff_fraction', 1.0):5.3f} | "
              f"{dev[0]:+7.3f} {dev[1]:+7.3f} {dev[2]:+7.3f}   | {l1:7.1e} {med:7.1e} {q95:7.1e} | {s['ledger']['balance_relative']:8.1e}")


if __name__ == "__main__":
    main()
