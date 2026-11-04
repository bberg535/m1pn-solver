#!/usr/bin/env python3
"""Registered sequential paired benchmark, all-checkpoint 1e-9 gate.

Every process uses one pinned CPU and one thread. Existing trials are immutable.
No external dependencies beyond the installed NumPy are installed by this script.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Same discretization/time step for every member of a matched group.
EXPLICIT = [
    ("smooth_1d", "smooth", 1, 32, 3, 1e-4, 40),
    ("smooth_2d", "smooth", 2, 16, 3, 1e-4, 40),
    ("isotropic_2d", "isotropic", 2, 12, 7, 1e-4, 40),
    ("crossing_2d", "crossing", 2, 12, 7, 1e-4, 40),
    ("smooth_3d", "smooth", 3, 6, 3, 1e-4, 40),
    ("crossing_3d", "crossing", 3, 6, 3, 1e-4, 40),
]
IMPLICIT = [
    ("imex_smooth_2d", "smooth", 2, 8, 3, 1e-4, 40),
    ("imex_crossing_2d", "crossing", 2, 8, 3, 1e-4, 40),
    ("imex_smooth_3d", "smooth", 3, 4, 3, 1e-4, 40),
    ("imex_crossing_3d", "crossing", 3, 4, 3, 1e-4, 40),
    ("imex_pulse_2d", "pulse", 2, 8, 3, 1e-4, 40),
]


def read_field(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    xyz = a[:, 1:4]
    u = a[:, 4:]
    assert np.isfinite(u).all() and np.isfinite(xyz).all()
    return xyz, u


def physical(u):
    return u[:, [0, 3, 1, 2]] * np.array([math.sqrt(4*math.pi)] + [math.sqrt(4*math.pi/3)]*3)


def density_error(folder, scenario, end_time):
    points = sorted(folder.glob("checkpoint_step_*.csv"))
    xyz, u = read_field(folder / "final.csv" if (folder / "final.csv").exists() else points[-1])
    rho = physical(u)[:, 0]
    weights = np.loadtxt(folder / "nodal_mass.csv", delimiter=",", skiprows=1, usecols=1, ndmin=1)
    if scenario == "smooth":
        # Independent exact isotropic kinetic transport on the periodic domain.
        exact = 1 + .2*np.sin(2*math.pi*xyz[:, 0])*np.sinc(2*end_time)
    elif scenario == "isotropic":
        exact = np.ones_like(rho)
    elif scenario == "crossing":
        z = 0. if np.max(xyz[:, 2]) == 0 else .5
        exact = .01 + np.exp(-50*np.sum((xyz-np.array([.3+end_time, .5, z]))**2, axis=1))
        exact += np.exp(-50*np.sum((xyz-np.array([.5, .3+end_time, z]))**2, axis=1))
    else:
        return {"density_reference": "same_discrete_direct_imex_only"}
    error = rho-exact
    return {"density_reference": "analytic_kinetic_nodal", "rho_L1": float(np.dot(weights, abs(error))),
            "rho_L2": float(np.sqrt(np.dot(weights, error**2))), "rho_Linf": float(np.max(abs(error))),
            "rho_relative_L1": float(np.dot(weights, abs(error))/np.dot(weights, abs(exact)))}


def compare(reference, candidate):
    ref = sorted(reference.glob("checkpoint_step_*.csv"))
    cand = sorted(candidate.glob("checkpoint_step_*.csv"))
    assert [p.name for p in ref] == [p.name for p in cand] and ref
    max_pn = max_physical = max_rho = 0.
    pairs = list(zip(ref, cand))
    if (reference/"final.csv").exists() and (candidate/"final.csv").exists():
        pairs.append((reference/"final.csv", candidate/"final.csv"))
    for a, b in pairs:
        xa, ua = read_field(a); xb, ub = read_field(b)
        assert np.array_equal(xa, xb)
        max_pn = max(max_pn, float(np.max(abs(ua-ub))))
        delta = physical(ua)-physical(ub)
        max_physical = max(max_physical, float(np.max(abs(delta))))
        max_rho = max(max_rho, float(np.max(abs(delta[:, 0]))))
    return {"max_PN_difference": max_pn, "max_physical_difference": max_physical,
            "max_rho_difference": max_rho, "checkpoints": len(pairs),
            "field_gate_pass": max(max_pn, max_physical) <= 1e-9}


def run_trial(build, out, case, variant, cpu):
    name, scenario, dim, elements, pn, dt, steps = case
    binary = build / ("efficiency_reference" if variant == "reference" else "efficiency_benchmark")
    cmd = ["taskset", "-c", str(cpu), str(binary), str(out), scenario, str(dim), str(elements),
           str(pn), str(dt), str(steps), "1", variant]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    start = time.perf_counter()
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.perf_counter()-start
    out.mkdir(exist_ok=True)
    (out / "stdout.log").write_text(proc.stdout)
    (out / "stderr.log").write_text(proc.stderr)
    record = {"command": cmd, "returncode": proc.returncode, "process_seconds": elapsed}
    (out / "process.json").write_text(json.dumps(record, indent=2)+"\n")
    if proc.returncode:
        return dict(record, status="failed", error=proc.stderr[-2000:])
    # Benchmark writes one row per measured repeat; warmup is excluded.
    with (out / "repeat_timings.csv").open() as f:
        rows = list(csv.DictReader(f))
    return dict(record, status="completed", solver_seconds=float(rows[0]["solver_seconds"]),
                **density_error(out, scenario, dt*steps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=ROOT / "tmp/build")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--suite", choices=["explicit", "implicit", "all"], default="all")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--case", action="append")
    ap.add_argument("--steps", type=int, help="Override steps, recorded before execution")
    ap.add_argument("--variants", nargs="+", help="Subset of matched variants, must include baseline")
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(exist_ok=False, parents=True)
    cases = ([] if args.suite == "implicit" else EXPLICIT) + ([] if args.suite == "explicit" else IMPLICIT)
    if args.case: cases = [c for c in cases if c[0] in args.case]
    if args.steps: cases = [tuple(list(c[:-1])+[args.steps]) for c in cases]
    assert cases and args.repeats >= 1
    cpu = min(os.sched_getaffinity(0))
    spec = {"cases": cases, "repeats": args.repeats, "cpu": cpu, "absolute_field_gate": 1e-9,
            "sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/"src").glob("*")) if p.is_file()},
            "hostname": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "benchmark_sha256": hashlib.sha256((ROOT/"tests/efficiency.cpp").read_bytes()).hexdigest(),
            "binaries": {n: hashlib.sha256((args.build/n).read_bytes()).hexdigest()
                         for n in ["efficiency_reference", "efficiency_benchmark"]}}
    (out / "plan.json").write_text(json.dumps(spec, indent=2)+"\n")
    results = []
    for case in cases:
        variants = ["imex-direct", "defect", "pcg"] if case[0].startswith("imex_") else ["reference", "eager", "lazy", "cached", "cached-eager"]
        baseline = variants[0]
        if args.variants:
            variants = [v for v in variants if v in args.variants]
            assert baseline in variants
        for repeat in range(args.repeats):
            order = variants if repeat % 2 == 0 else list(reversed(variants))
            group = {}
            for variant in order:
                target = out / (case[0]+"_"+variant+"_"+str(repeat))
                print(f"RUN {case[0]} {variant} repeat={repeat}", flush=True)
                row = dict(case=case[0], variant=variant, repeat=repeat, path=str(target),
                           **run_trial(args.build.resolve(), target, case, variant, cpu))
                results.append(row); group[variant] = row
                (out / "trials.json").write_text(json.dumps(results, indent=2)+"\n")
            for variant in variants:
                row = group[variant]
                if row["status"] == "completed" and group[baseline]["status"] == "completed":
                    row.update(compare(Path(group[baseline]["path"]), Path(row["path"])))
                else: row["field_gate_pass"] = False
            (out / "trials.json").write_text(json.dumps(results, indent=2)+"\n")
    summary = []
    for case in cases:
        names = list(dict.fromkeys(r["variant"] for r in results if r["case"] == case[0]))
        base_name = "imex-direct" if case[0].startswith("imex_") else "reference"
        base = [r["solver_seconds"] for r in results if r["case"] == case[0] and r["variant"] == base_name and r["status"] == "completed"]
        for name in names:
            rows = [r for r in results if r["case"] == case[0] and r["variant"] == name]
            passed = all(r["status"] == "completed" and r["field_gate_pass"] for r in rows)
            times = [r["solver_seconds"] for r in rows if r["status"] == "completed"]
            item = dict(case=case[0], variant=name, passed=passed, completed=len(times), repeats=len(rows))
            if times:
                item.update(median_seconds=statistics.median(times), min_seconds=min(times), max_seconds=max(times),
                            speedup=statistics.median(base)/statistics.median(times) if base else None,
                            max_PN_difference=max(r.get("max_PN_difference", math.inf) for r in rows),
                            max_physical_difference=max(r.get("max_physical_difference", math.inf) for r in rows),
                            max_rho_difference=max(r.get("max_rho_difference", math.inf) for r in rows),
                            **{k: rows[0][k] for k in ["rho_L1", "rho_L2", "rho_Linf", "rho_relative_L1", "density_reference"] if k in rows[0]})
            summary.append(item)
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    keys = list(dict.fromkeys(k for r in summary for k in r))
    with (out / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
