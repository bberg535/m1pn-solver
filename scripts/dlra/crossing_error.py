#!/usr/bin/env python3
"""Relative L1/L2 density error of crossing-study runs (two Gaussian pulses, pure transport, T = final time) against
the exact kinetic density rho = bg + p(x - T, y; .3, .5) + p(x, y - T; .5, .3) (scripts/element_weno/crossing.py),
with lumped nodal masses (nodal_mass.csv). Prints one JSON line per run and writes RUN_DIR/error.json.

Usage: crossing_error.py RUN_DIR [RUN_DIR ...]
"""
import json, math, sys
import numpy as np


def main():
    for run in sys.argv[1:]:
        cfg = json.load(open(run + "/config.json"))
        s = json.load(open(run + "/summary.json"))
        T, w, bg = s["time"], cfg.get("pulse_width", .07), cfg["background"]
        with open(run + "/final.csv") as fh:
            head = fh.readline().strip().split(",")
        ix, iy, iu = head.index("x"), head.index("y"), head.index("u0")
        d = np.loadtxt(run + "/final.csv", delimiter=",", skiprows=1, usecols=(ix, iy, iu))
        m = np.loadtxt(run + "/nodal_mass.csv", delimiter=",", skiprows=1, usecols=1)
        x, y, rho = d[:, 0], d[:, 1], math.sqrt(4 * math.pi) * d[:, 2]
        p = lambda cx, cy: np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w * w))
        R = bg + p(.3 + T, .5) + p(.5, .3 + T)
        e = rho - R
        out = dict(run=run, order=cfg["pn_order"], L1=float(np.sum(m * abs(e)) / np.sum(m * abs(R))),
                   L2=float(math.sqrt(np.sum(m * e * e) / np.sum(m * R * R))), rho_max=float(rho.max()),
                   rho_min=float(rho.min()), steps=s["steps"], solver_seconds=s["solver_seconds"],
                   peak_rss_mib=s["peak_rss_kib_rank0"] / 1024., balance_relative=s["ledger"]["balance_relative"])
        if s.get("dlra", {}).get("enabled"):
            out.update(rank_final=s["dlra"]["final_rank"], rank_max=s["dlra"]["max_rank"],
                       rank_mean=s["dlra"]["mean_rank"], low_rank_moments=s["dlra"]["low_rank_moments"])
        json.dump(out, open(run + "/error.json", "w"))
        print(json.dumps(out))


if __name__ == "__main__":
    main()
