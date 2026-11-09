#!/usr/bin/env python3
"""Run element WENO and filtered mass correction on finer crossing grids.

Usage: large_comparison.py EW_BINARY FILTER_BINARY RUN_DIR RESULT_DIR [WORKERS]

Filter runs reuse the arguments of nn_filter/scripts/study.py, restricted to the
coupled path and without its 900 s limit. Each finished run is stored as JSON,
so a restarted campaign skips it. All densities use the Q1 nodes and the
lumped-mass metrics of the filter study.
"""
import concurrent.futures as cf, csv, json, math, os, shutil, subprocess, sys, time
ew, flt, runs, out = sys.argv[1:5]
workers = int(sys.argv[5]) if len(sys.argv) > 5 else 6
os.makedirs(runs, exist_ok=True); os.makedirs(out + "/runs", exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
W, BG, T = .07, .001, .32
CONST = {3: 31.75, 5: 27.27, 7: 16.52, 9: 13.62}
CASES = [(64, 3), (64, 5), (64, 7), (64, 9), (128, 3), (128, 7)]
def ref(x, y):
    p = lambda a, b: math.exp(-(a * a + b * b) / (2 * W * W))
    return BG + p(x - T - .3, y - .5) + p(x - .5, y - T - .3)
def q1(x, y, g):
    h = 1 / g
    return (h / 2 if x < 1e-12 or x > 1 - 1e-12 else h) * (h / 2 if y < 1e-12 or y > 1 - 1e-12 else h)
def score(samples, g):
    a = b = c = d = linf = 0.; rmin = cmin = float("inf"); peak = 0.
    for x, y, rho, j, m in samples:
        r = ref(x, y); m = q1(x, y, g) if m is None else m; e = rho - r
        a += m * abs(e); b += m * abs(r); c += m * e * e; d += m * r * r
        linf = max(linf, abs(e)); rmin = min(rmin, rho); cmin = min(cmin, rho - j); peak = max(peak, rho)
    return dict(L1=a / b, L2=math.sqrt(c / d), Linf=linf, rho_min=rmin, cone=cmin, rho_max=peak)
def run_ew(g, order, degree, label, weight):
    n = round(g / degree); name = f"ew_g{g}_P{order}_p{degree}_{label}"; target = f"{runs}/{name}"
    t = time.time()
    r = subprocess.run([ew, "--problem", "crossing-study", "--dimension", "2", "--pn-order", str(order), "--degree", str(degree),
                        "--elements", str(n), "--time", str(T), "--cfl", ".5", "--weno-linear", weight, "--sample-grid", str(g),
                        "--output", target], env=env, capture_output=True, text=True, timeout=10800)
    wall = time.time() - t
    if r.returncode: return {"name": name, "error": r.stderr.strip()[:300]}
    s = json.loads(r.stdout.strip().splitlines()[-1])
    rows = []
    for row in csv.DictReader(open(target + "/final.csv")):
        j = math.sqrt(float(row["jx"])**2 + float(row["jy"])**2 + float(row["jz"])**2)
        rows.append((float(row["x"]), float(row["y"]), float(row["density"]), j, None))
    return dict(score(rows, g), name=name, grid=g, order=order, method=f"EW p={degree} {label}", degree=degree, wall=wall, steps=s["steps"],
                intervals=n * degree, min_coef=s["min_coefficient_density"], limited=s["limited_elements"], cells=n * n, nudged=s["nudged_states"])
def run_b(g, order, mode, dt):
    name = f"b_g{g}_P{order}_{mode}"; target = f"{runs}/{name}"; shutil.rmtree(target, ignore_errors=True)
    t = time.time()
    with open(f"{runs}/{name}.log", "w") as log:
        code = subprocess.run([flt, "--dimension", "2", "--elements", str(g), "--pn-order", str(order), "--case", "crossing-study",
                               "--inflow", "crossing-study-transport", "--stabilization", "m1pn_mass_correction", "--coupling", "m1pn",
                               "--filter", mode, "--filter-strength", str(CONST[order]), "--time", str(T), "--cfl", ".2",
                               "--threads", "1", "--cuda", "off", "--output", target, "--dt", format(dt, ".17g")],
                              stdout=log, stderr=subprocess.STDOUT, env=env, timeout=10800).returncode
    wall = time.time() - t
    if code: return {"name": name, "error": open(f"{runs}/{name}.log").read()[-300:]}
    masses = [float(r["mass"]) for r in csv.DictReader(open(target + "/nodal_mass.csv"))]
    rows = []
    with open(target + "/final.csv") as f:
        reader = csv.reader(f); head = next(reader); first = head.index("u0")
        for k, r in enumerate(reader):
            a0, a1, a2, a3 = (float(r[first + i]) for i in range(4))
            rho = math.sqrt(4 * math.pi) * a0; c = math.sqrt(4 * math.pi / 3)
            rows.append((float(r[head.index("x")]), float(r[head.index("y")]), rho, c * math.sqrt(a1 * a1 + a2 * a2 + a3 * a3), masses[k]))
    summary = json.load(open(target + "/summary.json"))
    return dict(score(rows, g), name=name, grid=g, order=order, method=f"B {mode}", degree=1, wall=wall, steps=summary["steps"], intervals=g, dt=dt)
def probe(g, order):
    name = f"probe_g{g}_P{order}"; target = f"{runs}/{name}"; shutil.rmtree(target, ignore_errors=True)
    with open(f"{runs}/{name}.log", "w") as log:
        subprocess.run([flt, "--dimension", "2", "--elements", str(g), "--pn-order", str(order), "--case", "crossing-study",
                        "--inflow", "crossing-study-transport", "--stabilization", "m1pn_mass_correction", "--coupling", "m1pn",
                        "--filter", "off", "--filter-strength", str(CONST[order]), "--time", "0", "--cfl", ".2",
                        "--threads", "1", "--cuda", "off", "--output", target], stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=3600)
    return json.load(open(target + "/config.json"))["dt_selected"]
def task_b(g, order, mode):
    done = f"{out}/runs/b_g{g}_P{order}_{mode}.json"
    if os.path.exists(done): return json.load(open(done))
    dt_file = f"{runs}/dt_g{g}_P{order}.json"
    if os.path.exists(dt_file): dt = json.load(open(dt_file))
    else:
        dt = probe(g, order); json.dump(dt, open(dt_file, "w"))
    result = run_b(g, order, mode, dt)
    if "error" not in result: json.dump(result, open(done, "w"), indent=1)
    return result
def task_ew(g, order, degree, label, weight):
    done = f"{out}/runs/ew_g{g}_P{order}_p{degree}_{label}.json"
    if os.path.exists(done): return json.load(open(done))
    result = run_ew(g, order, degree, label, weight)
    if "error" not in result: json.dump(result, open(done, "w"), indent=1)
    return result
tasks = []
for g, order in sorted(CASES, key=lambda c: (c[0], c[1])):
    tasks.append(("probe", g, order))
pool = cf.ThreadPoolExecutor(workers)
for g, order in CASES:
    if not os.path.exists(f"{runs}/dt_g{g}_P{order}.json"):
        json.dump(probe(g, order), open(f"{runs}/dt_g{g}_P{order}.json", "w"))
jobs = []
for g, order in CASES:
    for mode in ("off", "constant", "nn"):
        jobs.append(pool.submit(task_b, g, order, mode))
    for degree in (1, 2, 3, 4):
        for label, weight in (("w1e-3", "1e-3"), ("w0.2", "0.2")):
            jobs.append(pool.submit(task_ew, g, order, degree, label, weight))
print("jobs", len(jobs), flush=True)
results = []
for job in cf.as_completed(jobs):
    r = job.result(); results.append(r)
    print(r.get("name"), "ERROR " + r["error"] if "error" in r else f"L1={r['L1']:.4f} wall={r['wall']:.0f}s steps={r['steps']}", flush=True)
good = [r for r in results if "error" not in r]
with open(out + "/comparison.csv", "w", newline="") as f:
    keys = ["grid", "order", "method", "degree", "intervals", "L1", "L2", "Linf", "rho_min", "rho_max", "cone", "wall", "steps"]
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
    w.writerows(sorted(good, key=lambda r: (r["grid"], r["order"], r["L1"])))
print("COMPLETED", len(good), "of", len(results), flush=True)
