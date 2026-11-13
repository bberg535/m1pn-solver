#!/usr/bin/env python3
"""Compare two element-WENO runs: max |a - b| / max |a| per field of final.csv, plus summary scalars.

Usage: compare_runs.py REF_DIR NEW_DIR [--tolerance 1e-6]
Exit code 1 if any field exceeds the tolerance.
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def load(path):
    with open(path / "final.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref", type=Path)
    p.add_argument("new", type=Path)
    p.add_argument("--tolerance", type=float, default=1e-6)
    a = p.parse_args()
    ref, new = load(a.ref), load(a.new)
    worst = 0.0
    report = {}
    for key in ("density", "jx", "jy", "jz"):
        r, n = ref[key], new[key]
        scale = max(max(abs(v) for v in ref["density"]), 1e-300)  # |J| <= rho: one physical scale
        diff = max(abs(x - y) for x, y in zip(r, n))
        report[key] = diff / scale
        if key == "density" or scale > 1e-300:
            worst = max(worst, diff / scale)
    sr = json.loads((a.ref / "summary.json").read_text())
    sn = json.loads((a.new / "summary.json").read_text())
    for key in ("final_mass", "min_coefficient_margin", "kinetic_l2_error"):
        if key in sr and key in sn:
            report[key + "_rel"] = abs(sr[key] - sn[key]) / max(abs(sr[key]), 1e-300)
    report["wall_ref"] = sr["wall_seconds"]
    report["wall_new"] = sn["wall_seconds"]
    report["boundary_ref"] = sr["seconds"]["boundary"]
    report["boundary_new"] = sn["seconds"]["boundary"]
    report["boundary_speedup"] = sr["seconds"]["boundary"] / max(sn["seconds"]["boundary"], 1e-300)
    report["max_field_rel_diff"] = worst
    report["within_tolerance"] = worst <= a.tolerance
    print(json.dumps(report, indent=1))
    return 0 if worst <= a.tolerance else 1


if __name__ == "__main__":
    sys.exit(main())
