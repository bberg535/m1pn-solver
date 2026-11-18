import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from campaign_paths import BASE, CPUS

base = BASE
runner = base / "physics"
cases = []
for case in ("beam", "counter"):
    for threshold in (.0002, .00002):
        cases.append((case, 2, 48, 15, "warm", 16, threshold, 2 * threshold, "pn", 32, .2))
    for n, policy in ((11, "fixed"), (21, "warm")):
        cases.append((case, 2, 48, n, policy, 16, .002, .004, "pn", 32, .2))
    for policy in ("fixed", "warm"):
        cases.append((case, 2, 24, 15, policy, 8, .002, .004, "m1", 32, .99 * .5096537321044 * 8 / 24))
for quadrature in (24, 40, 48):
    cases.append(("counter", 2, 24, 15, "warm", 8, .002, .004, "pn", quadrature, .2))

cores = Queue()
for i in CPUS[:3]:
    cores.put(i)
def call(params):
    core = cores.get()
    try:
        result = subprocess.run(["taskset", "-c", str(core), str(runner), *map(str, params)], capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError(f"{params}: {result.stderr.strip()}")
        return next(csv.DictReader(result.stdout.splitlines())) | {"quadrature": params[-2], "end": params[-1]}
    finally:
        cores.put(core)
with ThreadPoolExecutor(max_workers=min(3, len(CPUS))) as pool, (base / "additional.csv").open("w", newline="") as out:
    header = None
    for row in pool.map(call, cases):
        if header is None:
            header = csv.DictWriter(out, fieldnames=row.keys())
            header.writeheader()
        header.writerow(row)
        out.flush()
        print(*(row[field] for field in ("case","count","N","CFL","policy","lower","rho_error","P_error","mean_dofs","limited_element_stages")), flush=True)
