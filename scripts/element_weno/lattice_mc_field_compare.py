#!/usr/bin/env python3
"""lattice-paul runs (uniform or graded tensor mesh, Q1 samples in final.csv) against the Monte-Carlo reference:
mass, total absorption and absorption rate (summary.json, lattice_fcs totals with FCS) and the density field as
cell averages on the 224^2 MC grid (bilinear interpolation of the Q1 samples, 4 x 4 Gauss points per MC cell).

Usage: lattice_mc_field_compare.py MC_FIELD_CSV RUN_DIR [RUN_DIR ...]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

MC = dict(mass=2.080890, absorbed=1.119098, rate=0.731555)


def mc_field(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    n = int(d[:, 0].max()) + 1
    ref = np.zeros((n, n))
    ref[d[:, 1].astype(int), d[:, 0].astype(int)] = d[:, 4]
    return n, ref


def cell_averages(run, n):
    d = np.loadtxt(Path(run) / "final.csv", delimiter=",", skiprows=1, usecols=(1, 2, 4))
    xs, ys = np.unique(d[:, 0]), np.unique(d[:, 1])
    o = np.lexsort((d[:, 0], d[:, 1]))
    Z = (math.sqrt(4 * math.pi) * d[o, 2]).reshape(len(ys), len(xs))
    f = RegularGridInterpolator((ys, xs), Z, method="linear")
    g, w = np.polynomial.legendre.leggauss(4)
    g, w = (g + 1) / 2, w / 2
    h = 7.0 / n
    c = (np.arange(n)[:, None] + g[None, :]).ravel() * h  # n*4 points per axis
    Y, X = np.meshgrid(c, c, indexing="ij")
    V = f(np.stack([Y.ravel(), X.ravel()], axis=1)).reshape(n, 4, n, 4)
    return np.einsum("iajb,a,b->ij", V, w, w)


def main():
    n, ref = mc_field(sys.argv[1])
    mask = ref > 1e-4
    print(f"{'run':28s} {'s':>7s} {'elements':>9s} | {'mass':>7s} {'absorb':>7s} {'rate':>7s} % | "
          f"{'L1':>7s} {'median':>7s} {'95%':>7s}")
    for run in sys.argv[2:]:
        s = json.load(open(Path(run) / "summary.json"))
        p = json.load(open(Path(run) / "progress.json"))
        fcs = s.get("lattice_fcs")
        mass = fcs["total_mass_final"] if fcs else s["final_density_mass"]
        absorbed = fcs["total_absorbed"] if fcs else s["ledger"]["absorbed"]
        rate = fcs["total_absorption_rate_final"] if fcs else s["absorption_rate_final"]
        dev = [100 * (v / MC[k] - 1) for k, v in (("mass", mass), ("absorbed", absorbed), ("rate", rate))]
        A = cell_averages(run, n)
        rel = np.abs(A - ref)[mask] / ref[mask]
        print(f"{Path(run).name:28s} {p['wall_seconds']:7.0f} {s.get('cells', 0):9d} | "
              f"{dev[0]:+7.3f} {dev[1]:+7.3f} {dev[2]:+7.3f}   | {np.abs(A - ref).sum() / ref.sum():7.1e} "
              f"{np.median(rel):7.1e} {np.quantile(rel, .95):7.1e}")


if __name__ == "__main__":
    main()
