import csv
import subprocess

from campaign_paths import BASE, CPUS

base = BASE
runner = base / "stability"
jobs = [
    (2, 6, 7, "fixed"),
    (2, 6, 15, "fixed"),
    *[(2, 6, n, m) for n in (7, 11, 15, 21) for m in ("patch", "checker")],
    (2, 12, 15, "patch"),
    (2, 12, 15, "graded"),
    (1, 24, 31, "patch"),
    (3, 4, 15, "checker"),
]
with (base / "stability.csv").open("w", newline="") as out:
    header = None
    for rank, count, n, mask in jobs:
        scan = (.4, .6, .75, .85, .875, .9, .925, .95, 1., 1.1)
        if rank == 1:
            scan = (.6, .8, .9, 1., 1.1, 1.2, 1.3, 1.4)
        elif rank == 3:
            scan = (.4, .5, .6, .7, .75, .8, .85, .9)
        line = ["taskset", "-c", str(CPUS[0]), str(runner), str(rank), str(count), str(n), mask, "22", *map(str, scan)]
        text = subprocess.check_output(line, text=True, timeout=90)
        rows = list(csv.DictReader(text.splitlines()))
        if header is None:
            header = csv.DictWriter(out, fieldnames=rows[0].keys())
            header.writeheader()
        header.writerows(rows)
        out.flush()
        print(rank, count, n, mask, [(x["cfl"], x["one_step_sigma_lower"], x["20step_factor"]) for x in rows], flush=True)
