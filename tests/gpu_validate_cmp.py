#!/usr/bin/env python3
"""Compare a CPU and a --gpu run of m1pn_fast (same binary) at the final time.

Usage: gpu_validate_cmp.py CPU_DIR GPU_DIR [--max-rel TOL] [--fcs-integrals]

Checks the density (u0 * sqrt(4 pi)) of final.csv: max relative deviation
(relative to max |density| of the CPU run) and relative L2 deviation must be
<= TOL, and both runs must take the same number of steps. With
--fcs-integrals the lattice_fcs integrals in summary.json are compared too.
Exit status 0 = pass, 1 = fail. Used by tests/gpu_validate.sh.
"""
import argparse
import csv
import json
import math
import sys


def load_density(d):
    with open(d + "/final.csv") as f:
        r = csv.reader(f)
        h = next(r)
        i = h.index("u0")
        s = math.sqrt(4 * math.pi)
        return [float(row[i]) * s for row in r]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cpu")
    p.add_argument("gpu")
    p.add_argument("--max-rel", type=float, default=1e-8)
    p.add_argument("--fcs-integrals", action="store_true")
    a = p.parse_args()

    ok = True
    try:
        x, y = load_density(a.cpu), load_density(a.gpu)
        sa = json.load(open(a.cpu + "/summary.json"))
        sb = json.load(open(a.gpu + "/summary.json"))
    except (OSError, ValueError, KeyError) as e:
        print(f"  FAIL: cannot read results ({e})")
        return 1
    if len(x) != len(y) or not x:
        print(f"  FAIL: field sizes differ ({len(x)} vs {len(y)})")
        return 1

    m = max(abs(v) for v in x)
    d = [abs(u - v) for u, v in zip(x, y)]
    max_rel = max(d) / m
    l2_rel = math.sqrt(sum(v * v for v in d) / sum(v * v for v in x))
    t = lambda s: s.get("solver_seconds", s.get("total_seconds", float("nan")))
    ta, tb = t(sa), t(sb)
    print(f"  density max_rel {max_rel:.2e} l2_rel {l2_rel:.2e} (tol {a.max_rel:.0e}) | "
          f"cpu {ta:.1f} s gpu {tb:.1f} s ({ta / max(tb, 1e-9):.1f}x) | "
          f"steps {sa.get('steps')}/{sb.get('steps')}")
    if not (max_rel <= a.max_rel and l2_rel <= a.max_rel):
        print("  FAIL: density deviation above tolerance")
        ok = False
    if sa.get("steps") != sb.get("steps"):
        print("  FAIL: step counts differ")
        ok = False

    if a.fcs_integrals:
        fa, fb = sa.get("lattice_fcs"), sb.get("lattice_fcs")
        if not fa or not fb:
            print("  FAIL: lattice_fcs missing in summary.json")
            return 1
        rel = {k: abs(fa[k] - fb[k]) / max(abs(fa[k]), 1e-300)
               for k in fa if k.startswith(("total", "uncollided")) and isinstance(fa[k], (int, float))}
        worst = max(rel.values()) if rel else 0.0
        print(f"  lattice_fcs integrals max rel {worst:.2e} | table cpu {fa.get('table_seconds', float('nan')):.2f} s "
              f"gpu {fb.get('table_seconds', float('nan')):.2f} s")
        if worst > a.max_rel:
            print("  FAIL: lattice_fcs integral deviation above tolerance: " +
                  ", ".join(f"{k} {v:.1e}" for k, v in rel.items() if v > a.max_rel))
            ok = False

    print("  PASS" if ok else "  => FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
