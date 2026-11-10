#!/usr/bin/env python3
"""CLI checks for m1pn_fast: defaults, element-weno, filter and biharmonic.

Usage: fast_cli.py M1PN_FAST [M1PN_ELEMENT_WENO]
"""
import csv, json, math, os, subprocess, sys, tempfile
fast = sys.argv[1]
standalone = sys.argv[2] if len(sys.argv) > 2 else None
env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
root = tempfile.mkdtemp(prefix="fast_cli_")
failures = 0
def check(ok, what):
    global failures
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    failures += not ok
def run(name, *args, expect_ok=True):
    out = f"{root}/{name}"
    r = subprocess.run([fast, *args, "--output", out], env=env, capture_output=True, text=True, timeout=600)
    check((r.returncode == 0) == expect_ok, f"{name}: exit {r.returncode}" + ("" if expect_ok else " (rejected as expected)"))
    if r.returncode and expect_ok:
        print(r.stderr[-400:])
    return out if r.returncode == 0 else None
def density(path):
    rows = list(csv.DictReader(open(path + "/final.csv")))
    return [math.sqrt(4 * math.pi) * float(r["u0"]) for r in rows], rows
crossing = ["--case", "crossing-study", "--inflow", "crossing-study-transport", "--dimension", "2",
            "--elements", "8", "--pn-order", "3", "--time", ".02"]
out = run("default", *crossing)
if out:
    c = json.load(open(out + "/config.json")); s = json.load(open(out + "/summary.json"))
    check(c["rk"] == 2 and c["cfl_safety"] == 1 and c["stabilization"] == "m1pn_mass_correction"
          and c["mass_mode"] == "lumped" and c["kv_gamma"] == 0, "default is mass correction, SSPRK2, CFL 1, lumped")
    check(s["status"] == "completed" and s["global_Q_balance"] < 1e-12, "default run completes with balance")
ew = run("element_weno", *crossing, "--stabilization", "element-weno", "--degree", "2", "--elements", "4", "--dt", ".005")
if ew:
    c = json.load(open(ew + "/config.json")); s = json.load(open(ew + "/summary.json"))
    check(c["rk"] == 3 and c["degree"] == 2 and c["sample_points"] == 81, "element-weno uses SSPRK3 and a 9x9 sample grid")
    check(s["minimum_cone_margin"] >= 0 and s["nudged_states"] == 0, "element-weno stays in the cone without nudging")
    if standalone:
        ref = f"{root}/standalone"
        subprocess.run([standalone, "--problem", "crossing-study", "--dimension", "2", "--pn-order", "3", "--degree", "2",
                        "--elements", "4", "--time", ".02", "--dt", ".005", "--sample-grid", "8", "--output", ref],
                       env=env, check=True, capture_output=True)
        a, _ = density(ew)
        b = [float(r["density"]) for r in csv.DictReader(open(ref + "/final.csv"))]
        check(len(a) == len(b) and max(abs(x - y) for x, y in zip(a, b)) < 1e-13,
              "element-weno in m1pn_fast matches the stand-alone driver")
ewf = run("element_weno_filter", *crossing, "--stabilization", "element-weno", "--elements", "4", "--dt", ".005",
          "--filter", "constant", "--filter-strength", "31.75")
if ewf and ew:
    a, _ = density(ew); b, _ = density(ewf)
    check(max(abs(x - y) for x, y in zip(a, b)) > 1e-8, "constant filter changes the element-weno solution")
bf = run("biharmonic_filter", *crossing, "--stabilization", "biharmonic-corrected", "--time-integrator", "ssprk3",
         "--filter", "constant", "--filter-strength", "31.75")
if bf:
    check(json.load(open(bf + "/summary.json"))["status"] == "completed", "biharmonic with constant filter completes")
run("element_weno_nn", *crossing, "--stabilization", "element-weno", "--filter", "nn", expect_ok=False)
run("element_weno_pn_only", *crossing, "--stabilization", "element-weno", "--coupling", "pn-only", expect_ok=False)
run("element_weno_rk2", *crossing, "--stabilization", "element-weno", "--time-integrator", "ssprk2", expect_ok=False)
print("FAILED" if failures else "PASSED")
sys.exit(1 if failures else 0)
