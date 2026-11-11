#!/usr/bin/env python3
"""Run many element-WENO tasks on one machine in pinned slots (docs/ELEMENT_WENO.md, "Hardware").

Usage: pack_node.py TASKFILE [--slot-cores N] [--dry-run]

TASKFILE: one task per line, "NAME<TAB>COMMAND ...". Blank lines and lines starting with # are ignored.
Each command gets OMP_NUM_THREADS, OMP_PLACES=cores and OMP_PROC_BIND=close of its slot and runs under
numactl --physcpubind=<slot cpus> --localalloc (taskset if numactl is missing); stdout/stderr go to NAME.log
next to the task file. A slot never spans two NUMA nodes.

Element WENO is limited by memory bandwidth. Measured on nx (2 x EPYC 9354): one run on 32 cores of one socket is
as fast as on 64 cores, and four pinned runs of 16 cores give 2.5 times the throughput of single runs. Default
slot size: 16 physical cores, at most one NUMA node (on the 6-core workstation: one slot of 6).
"""
import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def physical_cores():
    """NUMA node -> list of one logical CPU per physical core, within the process affinity."""
    allowed = sorted(os.sched_getaffinity(0))
    nodes, seen = {}, set()
    for cpu in allowed:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        node = next((int(p.name[4:]) for p in base.glob("node[0-9]*")), 0)
        core = (int((base / "topology/physical_package_id").read_text()), int((base / "topology/core_id").read_text()))
        if core in seen:
            continue
        seen.add(core)
        nodes.setdefault(node, []).append(cpu)
    return nodes


def slots(slot_cores):
    out = []
    for node, cpus in sorted(physical_cores().items()):
        size = min(slot_cores, len(cpus))
        for start in range(0, len(cpus) - size + 1, size):
            out.append((node, cpus[start:start + size]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("taskfile", type=Path)
    parser.add_argument("--slot-cores", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tasks = []
    for line in args.taskfile.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            name, command = line.split("\t", 1)
            tasks.append((name.strip(), command.strip()))
    plan = slots(args.slot_cores)
    if not plan:
        sys.exit("no CPUs available")
    pin = shutil.which("numactl")
    print(f"{len(tasks)} tasks, {len(plan)} slots: " +
          ", ".join(f"node {n}: {len(c)} cores ({c[0]}..{c[-1]})" for n, c in plan), flush=True)
    work = queue.Queue()
    for task in tasks:
        work.put(task)
    failures = []

    def worker(node, cpus):
        cpu_list = ",".join(map(str, cpus))
        prefix = ([pin, f"--physcpubind={cpu_list}", "--localalloc"] if pin else ["taskset", "-c", cpu_list])
        env = dict(os.environ, OMP_NUM_THREADS=str(len(cpus)), OMP_PLACES="cores", OMP_PROC_BIND="close")
        while True:
            try:
                name, command = work.get_nowait()
            except queue.Empty:
                return
            full = prefix + ["bash", "-c", command]
            if args.dry_run:
                print(f"[node {node}] {name}: {' '.join(full)}", flush=True)
                continue
            start = time.time()
            with open(args.taskfile.parent / f"{name}.log", "w") as log:
                code = subprocess.run(full, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            print(f"[node {node}, {len(cpus)} cores] {name}: exit {code}, {time.time() - start:.1f} s", flush=True)
            if code:
                failures.append(name)

    threads = [threading.Thread(target=worker, args=slot) for slot in plan]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        sys.exit(f"{len(failures)} failed: {' '.join(failures)}")


if __name__ == "__main__":
    main()
