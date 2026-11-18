import csv
import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from campaign_paths import BASE, CPUS

base = BASE
runner = base / "physics"
cases = []
def add(case, dim, count, n, steps, configs, mode="pn"):
    for policy, lower, upper in configs:
        cases.append((case, dim, count, n, policy, steps, lower, upper, mode))
fixed = ("fixed", .002, .004)
warm = ("warm", .002, .004)
default = ("warm", .055, .10)
cold = ("cold", .002, .004)
middle = ("warm", .015, .030)
for name, n in itertools.product(("broad", "beam", "counter"), (7, 11, 15, 21)):
    if name == "broad" and n not in (7, 15):
        continue
    add(name, 2, 24, n, 16, (fixed, warm, default, cold))
for n, steps in itertools.product((7, 11, 15), (40, 8)):
    add("beam", 2, 24, n, steps, (fixed, warm))
for name in ("beam", "counter"):
    add(name, 2, 24, 15, 16, (middle,))
    add(name, 2, 24, 15, 40, (fixed, warm), "m1")
    add(name, 2, 24, 15, 16, (fixed, warm), "m1")
    add(name, 2, 48, 15, 16, (fixed, warm))
for n in (7, 15):
    add("beam", 3, 8, n, 8, (fixed, warm))
assert len(cases) == len(set(cases))

affinity = Queue()
for core in CPUS[:3]:
    affinity.put(core)
def call(case):
    core = affinity.get()
    try:
        cmd = ["taskset", "-c", str(core), str(runner), *map(str, (*case, 32))]
        finished = subprocess.run(cmd, capture_output=True, text=True, timeout=100)
        if finished.returncode:
            raise RuntimeError(f"{case}: {finished.stderr.strip()}")
        return case, next(csv.DictReader(finished.stdout.splitlines()))
    finally:
        affinity.put(core)

with ThreadPoolExecutor(max_workers=min(3, len(CPUS))) as pool, (base / "physics.csv").open("w", newline="") as out:
    header = None
    for case, row in pool.map(call, cases):
        if header is None:
            header = csv.DictWriter(out, fieldnames=row.keys())
            header.writeheader()
        header.writerow(row)
        out.flush()
        print(case, row["rho_error"], row["P_error"], row["mean_dofs"], row["timed_seconds"], flush=True)
