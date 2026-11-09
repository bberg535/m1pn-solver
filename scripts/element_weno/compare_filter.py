#!/usr/bin/env python3
"""Compare element WENO runs with the mass-correction filter study.

Usage: compare_filter.py BINARY FILTER_RESULTS_DIR OUTPUT_DIR

The element WENO density is sampled on the Q1 nodes of the filter study and
measured with the same lumped-mass metrics against the same kinetic density.
"""
import csv, json, math, os, subprocess, sys
exe, filter_root, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="1")
W, BG, T = .07, .001, .32
def ref(x, y):
    p = lambda a, b: math.exp(-(a * a + b * b) / (2 * W * W))
    return BG + p(x - T - .3, y - .5) + p(x - .5, y - T - .3)
def weight(x, y, grid):
    h = 1 / grid
    wx = h / 2 if x < 1e-12 or x > 1 - 1e-12 else h
    wy = h / 2 if y < 1e-12 or y > 1 - 1e-12 else h
    return wx * wy
def metrics(path, grid):
    l1 = l1r = l2 = l2r = linf = 0.
    rmin = cmin = float("inf")
    for row in csv.DictReader(open(path)):
        x, y, rho = float(row["x"]), float(row["y"]), float(row["density"])
        j = math.sqrt(float(row["jx"])**2 + float(row["jy"])**2 + float(row["jz"])**2)
        r, m = ref(x, y), weight(x, y, grid)
        e = rho - r
        l1 += m * abs(e); l1r += m * abs(r); l2 += m * e * e; l2r += m * r * r
        linf = max(linf, abs(e)); rmin = min(rmin, rho); cmin = min(cmin, rho - j)
    return l1 / l1r, math.sqrt(l2 / l2r), linf, rmin, cmin
rows = []
for name in ("crossing_20261109", "extended_orders_20261109"):
    for r in csv.DictReader(open(f"{filter_root}/{name}/results.csv")):
        if r["coupling"] == "m1pn" and r["time_refined"] != "True":
            rows.append({"grid": int(r["grid"]), "order": int(r["order"]), "method": "B+" + r["filter"],
                         "L1": float(r["density_rel_L1"]), "L2": float(r["density_rel_L2"]), "Linf": float(r["density_Linf"]),
                         "rho_min": float(r["minimum_density"]), "cone": float(r["minimum_cone_margin"]),
                         "seconds": float(r["total_seconds"]), "steps": int(r["steps"]), "note": ""})
cases = sorted({(r["grid"], r["order"]) for r in rows})
print("cases", cases, flush=True)
def run(grid, order, degree, elements, label, flags, cfl=".5"):
    target = f"{out}/g{grid}_P{order}_p{degree}_{label}"
    r = subprocess.run([exe, "--problem", "crossing-study", "--dimension", "2", "--pn-order", str(order),
                        "--degree", str(degree), "--elements", str(elements), "--time", str(T), "--cfl", cfl,
                        "--sample-grid", str(grid), "--output", target] + flags, env=env, capture_output=True, text=True)
    if r.returncode:
        print("FAILED", grid, order, degree, label, r.stderr.strip()[:200], flush=True)
        return None
    s = json.loads(r.stdout.strip().splitlines()[-1])
    l1, l2, linf, rmin, cmin = metrics(target + "/final.csv", grid)
    os.remove(target + "/final.csv")
    row = {"grid": grid, "order": order, "method": f"EW p={degree} {label}", "L1": l1, "L2": l2, "Linf": linf,
           "rho_min": rmin, "cone": cmin, "seconds": s["wall_seconds"], "steps": s["steps"],
           "note": f"intervals={elements*degree} coef_min={s['min_coefficient_density']:.3e} coef_cone={s['min_coefficient_margin']:.3e} limited={s['limited_elements']}"}
    print(row, flush=True)
    return row
for grid, order in cases:
    for degree in (1, 2, 3, 4):
        elements = round(grid / degree)
        for label, w in (("w1e-3", "1e-3"), ("w0.2", "0.2")):
            result = run(grid, order, degree, elements, label, ["--weno-linear", w])
            if result: rows.append(result)
check = run(32, 3, 2, 16, "w0.2_cfl.25", ["--weno-linear", "0.2"], cfl=".25")
if check: rows.append(check)
with open(out + "/comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
