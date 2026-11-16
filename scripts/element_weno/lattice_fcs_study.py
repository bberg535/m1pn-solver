#!/usr/bin/env python3
"""Lattice with and without the first-collision source (docs/FCS_LATTICE.md), CPU on the NX nodes.

  lattice_fcs_study.py prepare --campaign DIR [--fast-bin BIN]   freeze binary, write tasks.json and array.sbatch
  lattice_fcs_study.py submit  --campaign DIR                    sbatch the array (one task = one socket, 32 cores)
  lattice_fcs_study.py run     --campaign DIR --index K          (inside the array) run task K
  lattice_fcs_study.py collect --campaign DIR                    results.csv, REPORT.md

Deviations are against Schotthoefer/Hauck Table 1 / (2 pi) (kinetic S22) as in nx_convergence.py; the FCS rows use
the totals (collided grid + uncollided ray tracing).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

GRIDS = [56, 112, 224, 448]
ORDERS = [3, 7, 15, 21]
EXTRA = [(112, 35), (224, 35)]
KINETIC = {"mass": 13.07 / (2 * math.pi), "absorbed": 7.034 / (2 * math.pi), "rate": 4.597 / (2 * math.pi)}


def tasks() -> list[dict]:
    out = []
    pairs = [(n, pn) for pn in ORDERS for n in GRIDS] + EXTRA
    for n, pn in pairs:
        for fcs in ("off", "analytic"):
            out.append(dict(label=f"lattice_P{pn}_E{n}_{'fcs' if fcs == 'analytic' else 'std'}", elements=n, pn=pn,
                            fcs=fcs))
    # longest first (moments x nodes x steps)
    out.sort(key=lambda t: -((t["pn"] + 1) * (t["pn"] + 2) / 2 * t["elements"] ** 3))
    return out


def command(campaign: Path, t: dict) -> list[str]:
    cmd = [str(campaign / "bin" / "m1pn_fast"), "--dimension", "2", "--mesh-type", "hex", "--elements",
           str(t["elements"]), "--degree", "1", "--pn-order", str(t["pn"]), "--case", "lattice-paul", "--inflow",
           "vacuum", "--model", "r2", "--stabilization", "element-weno", "--time-integrator", "ssprk43", "--cfl", "0.2",
           "--dt", "0", "--time", "3.2", "--threads", "32", "--output", str(campaign / "runs" / t["label"]),
           "--background", "1e-10", "--weno", "on", "--element-limit", "on", "--weno-linear", "0.001",
           "--history-stride", "50", "--angular-reduction", "auto", "--mirror-reduction", "off", "--sample-moments",
           "low", "--fcs", t["fcs"]]
    if t["pn"] >= 40:
        cmd += ["--memory-mib", "16384"]
    return cmd


def prepare(ns) -> int:
    c = ns.campaign.resolve()
    if c.exists():
        raise SystemExit(f"{c} exists")
    (c / "bin").mkdir(parents=True)
    (c / "runs").mkdir()
    (c / "logs").mkdir()
    shutil.copy2(ns.fast_bin, c / "bin" / "m1pn_fast")
    shutil.copy2(__file__, c / "study.py")
    ts = tasks()
    (c / "tasks.json").write_text(json.dumps(ts, indent=1))
    (c / "array.sbatch").write_text(f"""#!/usr/bin/env bash
#SBATCH --job-name=lat-fcs
#SBATCH --partition=med
#SBATCH --constraint=nx
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-{len(ts) - 1}%8
#SBATCH --output={c}/logs/task-%a-%A.log
export OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores OPENBLAS_NUM_THREADS=1
python3 {c}/study.py run --campaign {c} --index $SLURM_ARRAY_TASK_ID
""")
    print(f"{len(ts)} tasks, campaign {c}")
    return 0


def submit(ns) -> int:
    c = ns.campaign.resolve()
    out = subprocess.check_output(["sbatch", str(c / "array.sbatch")], text=True).strip()
    print(out)
    (c / "submitted.txt").write_text(out + "\n")
    return 0


def run(ns) -> int:
    c = ns.campaign.resolve()
    t = json.loads((c / "tasks.json").read_text())[ns.index]
    out = c / "runs" / t["label"]
    if (out / "summary.json").exists():
        return 0
    if out.exists():
        shutil.rmtree(out)
    with open(c / "logs" / (t["label"] + ".log"), "w") as log:
        rc = subprocess.call(command(c, t), stdout=log, stderr=subprocess.STDOUT)
    return rc


def collect(ns) -> int:
    c = ns.campaign.resolve()
    rows = []
    for t in json.loads((c / "tasks.json").read_text()):
        s = c / "runs" / t["label"] / "summary.json"
        if not s.exists():
            rows.append(dict(t, status="missing"))
            continue
        d = json.loads(s.read_text())
        lf = d.get("lattice_fcs")
        mass = lf["total_mass_final"] if lf else d["final_density_mass"]
        absorbed = lf["total_absorbed"] if lf else d["ledger"]["absorbed"]
        rate = lf["total_absorption_rate_final"] if lf else d["absorption_rate_final"]
        rows.append(dict(t, status="done", mass=mass, absorbed=absorbed, rate=rate,
                         dev_mass=mass / KINETIC["mass"] - 1, dev_absorbed=absorbed / KINETIC["absorbed"] - 1,
                         dev_rate=rate / KINETIC["rate"] - 1, balance=d["ledger"]["balance_relative"],
                         uncollided_mass=lf["uncollided_mass_final"] if lf else 0.,
                         table_seconds=lf["table_seconds"] if lf else 0., solver_seconds=d["solver_seconds"],
                         total_seconds=d["total_seconds"], steps=d["steps"]))
    keys = sorted({k for r in rows for k in r})
    with open(c / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    done = [r for r in rows if r["status"] == "done"]
    lines = ["# Lattice mit und ohne FCS (NX, CPU, Q1)", "",
             f"{len(done)} / {len(rows)} fertig. Abweichung gegen Schotthoefer/Hauck Tab. 1 / 2 pi.", "",
             "| N | Gitter | FCS | Masse | Absorption | Rate | ungestoßen (Masse) | Tabelle s | gesamt s |",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(done, key=lambda r: (r["pn"], r["elements"], r["fcs"])):
        lines.append(f"| {r['pn']} | {r['elements']}² | {'ja' if r['fcs'] == 'analytic' else 'nein'} | "
                     f"{100 * r['dev_mass']:+.2f} % | {100 * r['dev_absorbed']:+.2f} % | {100 * r['dev_rate']:+.2f} % | "
                     f"{r['uncollided_mass']:.4f} | {r['table_seconds']:.0f} | {r['total_seconds']:.0f} |")
    (c / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    for name, fn in [("prepare", prepare), ("submit", submit), ("run", run), ("collect", collect)]:
        q = sub.add_parser(name)
        q.add_argument("--campaign", type=Path, required=True)
        if name == "prepare":
            q.add_argument("--fast-bin", type=Path,
                           default=Path(__file__).resolve().parents[2] / "tmp" / "build_cpu" / "m1pn_fast")
        if name == "run":
            q.add_argument("--index", type=int, required=True)
        q.set_defaults(fn=fn)
    ns = p.parse_args()
    return ns.fn(ns)


if __name__ == "__main__":
    raise SystemExit(main())
