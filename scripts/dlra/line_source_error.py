#!/usr/bin/env python3
"""Relative L1/L2 density error of GHS line-source runs (element WENO on [0,1]^2, sigma_s = 3, T = 1/3) against the
Ganapol reference, with the same metric as scripts/line_source/analyze.py (lumped nodal masses, exact rescaling to
the physical GH problem: r = 3 r', rho = rho' / 9).

Usage: line_source_error.py REFERENCE_GH.csv RUN_DIR [RUN_DIR ...]   (prints one JSON line per run)
Also writes RUN_DIR/error.json. With --pair FULL DLRA it additionally reports the relative L1 distance of the two
density fields (same sample grid).
"""
import json, math, sys
import numpy as np


def load(run):
    with open(run + "/final.csv") as fh:
        head = fh.readline().strip().split(",")
    ix, iy, iu = head.index("x"), head.index("y"), head.index("u0")
    d = np.loadtxt(run + "/final.csv", delimiter=",", skiprows=1, usecols=(ix, iy, iu))
    m = np.loadtxt(run + "/nodal_mass.csv", delimiter=",", skiprows=1, usecols=1)
    return d[:, 0], d[:, 1], math.sqrt(4 * math.pi) * d[:, 2], m


def main():
    args = sys.argv[1:]
    pair = None
    if args and args[0] == "--pair":
        pair = args[1:3]
        args = args[3:]
    ref = np.loadtxt(args[0], delimiter=",", skiprows=1)
    rr, phi = ref[:, 0], ref[:, 1]
    for run in args[1:]:
        x, y, rho, m = load(run)
        x, y, rho, m = .5 + 3 * (x - .5), .5 + 3 * (y - .5), rho / 9, 9 * m
        R = np.interp(np.hypot(x - .5, y - .5), rr, phi)
        e = rho - R
        s = json.load(open(run + "/summary.json"))
        out = dict(run=run, L1=float(np.sum(m * abs(e)) / np.sum(m * abs(R))),
                   L2=float(math.sqrt(np.sum(m * e * e) / np.sum(m * R * R))),
                   rho_min=float(rho.min()), steps=s["steps"], solver_seconds=s["solver_seconds"],
                   total_seconds=s["total_seconds"], peak_rss_mib=s["peak_rss_kib_rank0"] / 1024.,
                   balance_relative=s["ledger"]["balance_relative"])
        if "dlra" in s and s["dlra"].get("enabled"):
            out.update(rank_final=s["dlra"]["final_rank"], rank_max=s["dlra"]["max_rank"],
                       rank_mean=s["dlra"]["mean_rank"], low_rank_moments=s["dlra"]["low_rank_moments"])
        json.dump(out, open(run + "/error.json", "w"))
        print(json.dumps(out))
    if pair:
        _, _, a, m = load(pair[0])
        _, _, b, _ = load(pair[1])
        print(json.dumps(dict(pair=pair, relative_L1_distance=float(np.sum(m * abs(a - b)) / np.sum(m * abs(a))))))


if __name__ == "__main__":
    main()
