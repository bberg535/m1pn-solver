#!/usr/bin/env python3
"""ctest of m1pn_lattice_mc (docs/LATTICE_MC.md): exact balance per history, bitwise thread independence, and the
2D lattice against Schotthoefer/Hauck Tab. 1 / 2pi within 5 standard errors (the table agrees with the 4e10-history
reference to 0.035 %, far below the error bar of this short run)."""
import json
import os
import subprocess
import sys
import tempfile

LITERATURE = {"mass": 2.0802, "absorbed": 1.1195, "rate": 0.7316}


def run(binary, case, threads, out):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    subprocess.run([binary, "--case", case, "--histories", "2e5", "--batches", "100", "--output", out],
                   check=True, env=env, stdout=subprocess.DEVNULL)
    with open(out) as f:
        return json.load(f)


def main():
    binary = sys.argv[1]
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for case in ("lattice-paul", "lattice-3d"):
            a = run(binary, case, 1, os.path.join(tmp, case + "_1.json"))
            b = run(binary, case, 3, os.path.join(tmp, case + "_3.json"))
            for d in (a, b):
                d.pop("seconds"), d.pop("threads")
            if a != b:
                failures.append(f"{case}: result depends on the thread count")
            if a["max_history_imbalance"] > 1e-14:
                failures.append(f"{case}: history imbalance {a['max_history_imbalance']}")
            tot, src, bg = a["total"], a["source"], a["background"]
            for k in range(len(a["times"])):
                lhs = tot["mass"][k] + tot["absorbed"][k] + tot["outflow"][k]
                rhs = src["injected"][k] + bg["injected"][k]
                if abs(lhs - rhs) > 1e-12 * max(1., rhs):
                    failures.append(f"{case}: balance at t={a['times'][k]}: {lhs} vs {rhs}")
            if case == "lattice-paul":
                for q, ref in LITERATURE.items():
                    m, s = tot[q][-1], tot[q + "_stderr"][-1]
                    if abs(m - ref) > 5 * s:
                        failures.append(f"lattice-paul {q}: {m} +- {s} vs literature {ref}")
    for f in failures:
        print("FAIL", f)
    print("lattice_mc_check:", "ok" if not failures else f"{len(failures)} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
