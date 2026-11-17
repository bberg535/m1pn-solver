#!/usr/bin/env python3
"""Kobayashi benchmarks: m1pn_fast detectors.csv against the reference (Kyoto analytic for case i, Monte Carlo for
case ii) and the Ardra P21 results, all as tabulated in Brown, Chang, Hanebutte, UCRL-JC-135163 (Physor 2000),
Tables III-VI (3 digits). Line C of problem 1 lies at y = 55 (the table heading says y = 95; its first value equals
line A at y = 55).

Usage: compare.py CASE RUN_DIR [RUN_DIR ...]      CASE = 1i | 1ii | 2i | 2ii
       compare.py --summary CSV RUN_DIR [...]      one row per run: max/median |rel| per line and overall
"""
import csv
import json
import statistics
import sys
from pathlib import Path

S = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
# reference and Ardra P21 per case and line, in detector order (lines A, B, C as in src/uncollided_box3d.hpp)
REF = {
    "1i": {
        "A": [5.96e0, 1.37e0, 5.01e-1, 2.52e-1, 1.50e-1, 5.95e-2, 1.53e-2, 4.18e-3, 1.19e-3, 3.47e-4],
        "B": [5.96e0, 4.71e-1, 1.70e-1, 8.68e-2, 5.25e-2, 1.33e-2, 1.46e-3, 1.75e-4, 2.25e-5, 3.01e-6],
        "C": [5.95e-2, 5.50e-2, 4.81e-2, 3.97e-2, 3.16e-2, 2.35e-2, 5.84e-3, 1.57e-3, 4.53e-4, 1.37e-4],
    },
    "1ii": {
        "A": [8.29e0, 1.87e0, 7.13e-1, 3.86e-1, 2.54e-1, 1.37e-1, 4.66e-2, 1.58e-2, 5.49e-3, 1.83e-3],
        "B": [8.29e0, 6.64e-1, 2.69e-1, 1.56e-1, 1.05e-1, 3.03e-2, 4.13e-3, 5.65e-4, 7.87e-5, 9.64e-6],
        "C": [1.37e-1, 1.28e-1, 1.13e-1, 9.60e-2, 7.75e-2, 5.66e-2, 1.89e-2, 6.40e-3, 2.33e-3, 7.94e-4],
    },
    "2i": {
        "A": [5.96e0, 1.37e0, 5.01e-1, 2.52e-1, 1.50e-1, 9.92e-2, 7.02e-2, 5.22e-2, 4.03e-2, 3.21e-2],
        "B": [3.21e-2, 1.71e-3, 1.41e-4, 3.27e-5, 1.09e-5, 4.14e-6],
    },
    "2ii": {
        "A": [8.61e0, 2.16e0, 8.94e-1, 4.78e-1, 2.89e-1, 1.89e-1, 1.31e-1, 9.49e-2, 7.10e-2, 5.46e-2],
        "B": [5.46e-2, 6.63e-3, 1.27e-3, 4.19e-4, 1.74e-4, 5.92e-5],
    },
}
ARDRA_P21 = {
    "1i": {
        "A": [5.77e0, 1.63e0, 4.46e-1, 2.92e-1, 1.60e-1, 6.69e-2, 1.71e-2, 3.33e-3, 7.44e-4, 3.22e-5],
        "B": [5.77e0, 4.67e-1, 1.67e-1, 8.25e-2, 5.57e-2, 1.21e-2, 1.49e-3, 1.80e-4, 2.33e-5, 3.15e-6],
        "C": [6.69e-2, 6.47e-2, 5.01e-2, 3.46e-2, 2.95e-2, 2.44e-2, 5.27e-3, 1.32e-3, 4.66e-4, 1.07e-4],
    },
    "1ii": {
        "A": [7.94e0, 2.18e0, 6.45e-1, 4.30e-1, 2.62e-1, 1.46e-1, 4.84e-2, 1.54e-2, 5.08e-3, 1.24e-3],
        "B": [7.94e0, 6.43e-1, 2.58e-1, 1.51e-1, 1.06e-1, 2.89e-2, 4.18e-3, 5.99e-4, 8.91e-5, 1.09e-5],
        "C": [1.46e-1, 1.38e-1, 1.15e-1, 9.18e-2, 7.67e-2, 5.74e-2, 1.89e-2, 6.68e-3, 2.28e-3, 7.24e-4],
    },
    "2i": {
        "A": [5.96e0, 1.32e0, 4.96e-1, 2.58e-1, 1.48e-1, 9.94e-2, 7.18e-2, 4.72e-2, 3.46e-2, 2.86e-2],
        "B": [2.86e-2, 1.85e-3, 1.80e-6, 4.66e-5, 1.21e-5, 1.97e-5],
    },
    "2ii": {
        "A": [8.61e0, 2.13e0, 8.87e-1, 4.77e-1, 2.93e-1, 2.06e-1, 1.48e-1, 7.93e-2, 4.92e-2, 3.66e-2],
        "B": [3.66e-2, 7.65e-3, 1.20e-3, 3.52e-4, 1.56e-4, 5.08e-5],
    },
}


def coords(case, line, k):
    s = S[k]
    if line == "A":
        return (5, s, 5)
    if case.startswith("1"):
        return (s, s, s) if line == "B" else (s, 55, 5)
    return (s, 95, 5)


def load(run):
    rows = {}
    with open(Path(run) / "detectors.csv") as f:
        for r in csv.DictReader(f):
            rows[(r["line"], float(r["x"]), float(r["y"]), float(r["z"]))] = float(r["rho"])
    return rows


def errors(case, values):
    out = {}
    for line, ref in REF[case].items():
        out[line] = [values[(line, *map(float, coords(case, line, k)))] / v - 1 for k, v in enumerate(ref)]
    return out


def stats(errs):
    allv = [abs(e) for es in errs.values() for e in es]
    return max(allv), statistics.median(allv)


def describe(run):
    c = json.load(open(Path(run) / "config.json"))
    p = json.load(open(Path(run) / "progress.json"))
    return c, p


def table(case, runs):
    ardra = errors(case, {(l, *map(float, coords(case, l, k))): v for l, vs in ARDRA_P21[case].items()
                          for k, v in enumerate(vs)})
    labels = ["Ardra P21"] + [Path(r).name for r in runs]
    errs = [ardra] + [errors(case, load(r)) for r in runs]
    for line in REF[case]:
        print(f"\n{case} line {line}" + "".join(f" | {l[:22]:>22s}" for l in labels))
        for k, ref in enumerate(REF[case][line]):
            x = coords(case, line, k)
            print(f"  {str(x):>14s} {ref:8.2e}" + "".join(f" | {100 * e[line][k]:+21.1f}%" for e in errs))
    print("\nmax / median |rel| over all detectors:")
    for l, e in zip(labels, errs):
        m, med = stats(e)
        print(f"  {l:40s} {100 * m:7.1f} % {100 * med:7.2f} %")


def summary(out_csv, runs):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "case", "pn", "cell_size", "fcs", "filter", "filter_strength", "time", "steps", "wall_seconds",
                    "max_rel", "median_rel"] + [f"max_rel_{l}" for l in "ABC"])
        for r in runs:
            c, p = describe(r)
            case = c["material"].split(",")[0].replace("kobayashi-", "") if "material" in c else "?"
            e = errors(case, load(r))
            m, med = stats(e)
            w.writerow([Path(r).name, case, c.get("pn_order"), c.get("cell_size"), c.get("fcs"), c.get("filter_mode"),
                        c.get("filter_constant"), p["time"], p["steps"], round(p["wall_seconds"], 1), m, med] +
                       [max(abs(v) for v in e[l]) if l in e else "" for l in "ABC"])


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2], sys.argv[3:])
    else:
        table(sys.argv[1], sys.argv[2:])
