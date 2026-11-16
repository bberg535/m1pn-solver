#!/usr/bin/env python3
"""Convergence campaign for element-WENO M1PN on tardis (NVIDIA A100), extending the NX campaign.

Reuses nx_convergence.py unchanged (freezing, run validation, collect, plots) and only changes
  * the task matrix (larger grids, higher P_N; FCS line source as its own campaign so that EOCs are not mixed),
  * the solver flags of GPU tasks: --gpu, --mirror-reduction off (not on the GPU), --angular-reduction auto (exact for
    the even 2D data of these benchmarks, about half the moments),
  * the submission: sequential tasks on the single A100 of tardis, packed into Slurm jobs by estimated run time.
The 1D periodic smooth family (standalone binary) and anything the GPU path rejects run on the tardis CPU.

  python3 scripts/element_weno/tardis_convergence.py plan --profile tardis-paper
  python3 scripts/element_weno/tardis_convergence.py prepare --profile tardis-paper --campaign "$PWD/nx/tmp/tardis_paper"
  python3 <campaign>/tardis.py submit --campaign <campaign>            # dry run
  python3 <campaign>/tardis.py submit --campaign <campaign> --execute
  python3 <campaign>/tardis.py collect --campaign <campaign>            # results.csv, REPORT.md, runtimes.md, figures

No job is submitted without --execute.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nx_convergence as nx  # noqa: E402  (frozen next to this file inside a campaign)

PROFILES = ("tardis-smoke", "tardis-paper", "tardis-fcs", "tardis-hetero", "tardis-lattice", "tardis-lattice-mc")
HETERO = ("tardis-hetero", "tardis-lattice", "tardis-lattice-mc")
nx.HETERO_PROFILES = tuple(nx.HETERO_PROFILES) + HETERO
# Calibration (A100, SSPRK(4,3), Q1 512^2 P7, 37 moments, 722 steps: 7.2 s): seconds per node x moment x step, plus a
# fixed device/host overhead per stage. Estimates only (factor ~2).
GPU_COST = 1.03e-9
STAGE_OVERHEAD = 1.5e-3


def moments(t: dict) -> int:
    n = t["pn_order"]
    if t["dim"] < 3:
        return (n + 1) * (n + 2) // 2 + (1 if n >= 1 else 0)  # even moments plus J_z (angular reduction)
    return (n + 1) ** 2


def gpu_estimate(t: dict) -> float:
    nodes = (t["elements"] * t["degree"] + 1) ** t["dim"]
    h_node = 1. / (t["elements"] * t["degree"])
    if t["case"] == "lattice-paul":
        h_node *= 7  # domain [0, 7]^2
    elif t["case"] == "lattice-3d":
        h_node *= 7
    rk43 = t["rk"] == "ssprk43"
    factor = (.25 if rk43 else .125) * (1. if t["dim"] < 3 else .67)
    dt = t.get("fixed_dt") or factor * t["cfl"] * h_node
    steps = math.ceil(t["time"] / dt)
    stages = steps * (4 if rk43 else 3)
    return GPU_COST * nodes * moments(t) * steps + STAGE_OVERHEAD * stages


def gpu_task(t: dict) -> dict:
    """Mark a task for the GPU (fast binary, element-WENO, no periodic mesh)."""
    if t["binary"] == "fast" and t.get("stabilization", "element-weno") == "element-weno" and not t.get("periodic"):
        t["gpu"] = True
        if t.get("expect_mirror") is not None:
            t["expect_mirror"] = [False, False, False]  # mirror reduction is off on the GPU
    t["gpu_estimate_seconds"] = gpu_estimate(t) if t.get("gpu") else None
    return t


def make_tardis_tasks(profile: str) -> list[dict]:
    task = nx.task
    smoke = profile == "tardis-smoke"
    out: list[dict] = []
    if profile in ("tardis-smoke", "tardis-paper"):
        # 1D periodic smooth exact-PN study (standalone, CPU): finer grids than on NX.
        for p in ([1, 2] if smoke else [1, 2, 3, 4]):
            grids = [8, 16] if smoke else [8, 16, 32, 64, 128, 256]
            for n in grids:
                out.append(task("smooth", "standalone", "smooth", "vacuum", 1, n, p, 3, .1, kind="smooth_exact",
                                periodic=True, tag="spatial", cfl=.15))
                if n == grids[-1]:
                    out.append(task("smoothdt2", "standalone", "smooth", "vacuum", 1, n, p, 3, .1,
                                    kind="smooth_exact", periodic=True, tag="temporal_check", cfl=.075))
        # Open-boundary smooth transport (Q2), analytic kinetic inflow.
        for pn in ([3] if smoke else [3, 7, 11, 15]):
            for n in ([8, 16] if smoke else [8, 16, 32, 64, 128, 256, 512]):
                out.append(task("boundary", "fast", "smooth", "smooth-transport", 2, n, 2, pn, .15,
                                kind="boundary_exact"))
        # Crossing: Q1 up to 1024^2 (2048^2 for P <= 7), Q2 up to 512^2, Q3 up to 256^2.
        for pn in ([3] if smoke else [3, 7, 11, 15]):
            for p, grids in ([(1, [8, 16])] if smoke else
                             [(1, [16, 32, 64, 128, 256, 512, 1024] + ([2048] if pn <= 7 else [])),
                              (2, [8, 16, 32, 64, 128, 256, 512]),
                              (3, [8, 16, 32, 64, 128, 256])]):
                for n in grids:
                    out.append(task("crossing", "fast", "crossing-study", "crossing-study-transport", 2, n, p, pn,
                                    .32))
        # GHS line source without FCS (FCS: profile tardis-fcs).
        for pn in ([3] if smoke else [3, 7, 11, 15]):
            for p, grids in ([(1, [16])] if smoke else [(1, [32, 64, 128, 256, 512, 1024]),
                                                        (2, [16, 32, 64, 128, 256, 512])]):
                for n in grids:
                    out.append(task("line", "fast", "line-source", "vacuum", 2, n, p, pn, 1 / 3, sigma_s=3.,
                                    kind="line_exact"))
        # Time refinement (fixed grid, finest dt is the reference), as on NX but on finer grids.
        for name, case, inflow, dim, n, p, pn, end, scattering, absorption, filt in [
                ("time_smooth", "smooth", "vacuum", 1, 16 if smoke else 96, 2, 3, .12, 2., .5, 0.),
                ("time_crossing", "crossing-study", "crossing-study-transport", 2, 12 if smoke else 128, 2, 7, .08,
                 2., .5, .15)]:
            levels = [0, 1, 2] if smoke else [0, 1, 2, 3, 4]
            for level in levels:
                out.append(task(name, "fast", case, inflow, dim, n, p, pn, end, kind="temporal",
                                periodic=(dim == 1), sigma_s=scattering, sigma_a=absorption,
                                filter_strength=filt, dt_level=level, reference=(level == max(levels)), cfl=.3))
        # Ablation on a 4x finer grid than on NX.
        if not smoke:
            for key, weno, limit, rk in [("full", True, True, "ssprk43"), ("no_weno", False, True, "ssprk43"),
                                         ("no_limit", True, False, "ssprk43"), ("ssprk3", True, True, "ssprk3")]:
                out.append(task("ablation", "fast", "crossing-study", "crossing-study-transport", 2, 256, 2, 7,
                                .32, kind="ablation", tag=key, weno=weno, limiter=limit, rk=rk))
            out.append(task("ablation", "fast", "crossing-study", "crossing-study-transport", 2, 256, 2, 7, .32,
                            kind="ablation", tag="weno_weight_02", linear=.2))
    if profile == "tardis-fcs":
        # GHS line source with the first-collision source (separate campaign: same family name, own EOCs).
        for pn in [3, 7, 11, 15, 21]:
            for p, grids in [(1, [32, 64, 128, 256, 512, 1024]), (2, [16, 32, 64, 128, 256, 512])]:
                for n in grids:
                    t = task("line", "fast", "line-source", "vacuum", 2, n, p, pn, 1 / 3, sigma_s=3.,
                             kind="line_exact", tag="fcs")
                    t["fcs"] = True
                    out.append(t)
    if profile == "tardis-hetero":
        lat = dict(background=1e-10, expect_mirror=[True, False, False])
        for pn in [3, 7, 11, 15]:
            for p, grids in [(1, [112, 224, 448]), (2, [56, 112, 224]), (3, [35, 70, 140]), (4, [28, 56, 112])]:
                for n in grids:
                    out.append(task("lattice", "fast", "lattice-paul", "vacuum", 2, n, p, pn, 3.2, kind="lattice",
                                    tag="std", **lat))
        for pn in [7]:
            out.append(task("lattice", "fast", "lattice-paul", "vacuum", 2, 896, 1, pn, 3.2, kind="lattice",
                            tag="std", **lat))
        for pn in [3, 7]:
            for p, grids in [(1, [28, 56, 112]), (2, [14, 28, 56])]:
                for n in grids:
                    out.append(task("lattice3d", "fast", "lattice-3d", "vacuum", 3, n, p, pn, 3.2, kind="lattice3d",
                                    background=1e-10, skip_fields=True, expect_mirror=[False, False, False]))
    if profile == "tardis-lattice":
        # Lattice 2D, Q1 (most accurate per unknown in tardis-hetero), larger grids and higher N. Grids limited by the
        # A100 memory (~100 B per node x moment, kept below ~60 GB) and ~2 h per run.
        lat = dict(background=1e-10, expect_mirror=[True, False, False])
        for pn, grids in [(7, [224, 448, 896, 1792]), (15, [224, 448, 896, 1792]), (21, [224, 448, 896, 1344]),
                          (35, [224, 448, 896]), (45, [224, 448, 672])]:
            for n in grids:
                out.append(task("lattice", "fast", "lattice-paul", "vacuum", 2, n, 1, pn, 3.2, kind="lattice",
                                tag="std", **lat))
    if profile == "tardis-lattice-mc":
        # Lattice 2D/3D against the Monte-Carlo reference (docs/LATTICE_MC.md), standard CFL 1, Q1, T = 3.2.
        # A100 memory (80 GB): solver ~100 B per node x moment; the FCS time table holds points x (T/0.01 + 3)
        # doubles (3D 112^3: 29 GB, 168^3: ~100 GB). Hence 3D FCS up to 112^3 (P <= 7), 2D up to 1344^2.
        # K1, 3D: sigma_a h = 1.25 .. 0.31 (asymptotic regime?), FCS on/off, P_N part.
        l3 = dict(background=1e-10, skip_fields=True, expect_mirror=[False, False, False], cfl=1.)
        for n, std, fcs in [(56, [3, 7, 15], [3, 7, 15]), (112, [3, 7, 15], [3, 7]), (168, [3, 7], []), (224, [3], [])]:
            for with_fcs, pns in ((False, std), (True, fcs)):
                for pn in pns:
                    t = task("lattice3d", "fast", "lattice-3d", "vacuum", 3, n, 1, pn, 3.2, kind="lattice3d",
                             tag="fcs" if with_fcs else "std", **l3)
                    t["fcs"] = with_fcs
                    out.append(t)
        # K2, 2D: high resolution in grid and N, with FCS (P21 on 896^2 also without).
        lat = dict(background=1e-10, expect_mirror=[True, False, False], cfl=1.)
        for n, pn, with_fcs in [(896, 15, True), (896, 21, True), (896, 21, False), (896, 35, True),
                                (1344, 15, True), (1344, 21, True)]:
            t = task("lattice", "fast", "lattice-paul", "vacuum", 2, n, 1, pn, 3.2, kind="lattice",
                     tag="fcs" if with_fcs else "std", **lat)
            t["fcs"] = with_fcs
            out.append(t)
    seen = set()
    for number, t in enumerate(out):
        if t["label"] in seen:
            raise ValueError("duplicate task label: " + t["label"])
        seen.add(t["label"])
        t["id"] = number
        if t.get("kind") in ("lattice", "shadow", "lattice3d"):
            t.update(nx.cost_model(t))
        gpu_task(t)
    return out


_original_make_tasks = nx.make_tasks


def make_tasks(profile: str) -> list[dict]:
    return make_tardis_tasks(profile) if profile in PROFILES else _original_make_tasks(profile)


_original_flags = nx.flags


def flags(t: dict, output: Path, threads: int, **kwargs):
    args = list(_original_flags(t, output, threads, **kwargs))
    if t["binary"] == "fast":
        def set_value(key, value):
            if key in args:
                args[args.index(key) + 1] = value
            else:
                args.extend([key, value])
        if t.get("gpu"):
            set_value("--mirror-reduction", "off")
            args.append("--gpu")
        set_value("--angular-reduction", "auto")
        if t.get("stabilization", "element-weno") == "element-weno":
            set_value("--sample-moments", "low")  # the analysis uses u0..u3 only; full PN columns cost minutes
        if t.get("fcs"):
            args += ["--fcs", "analytic"]
        if t["pn_order"] >= 40:
            set_value("--memory-mib", "16384")  # angular setup (32 q^2 doubles) exceeds the 1 GiB default
    return args


nx.make_tasks = make_tasks
nx.flags = flags


def plan(ns) -> int:
    tasks = make_tardis_tasks(ns.profile)
    gpu = [t for t in tasks if t.get("gpu")]
    total = sum(t["gpu_estimate_seconds"] for t in gpu)
    print(f"{ns.profile}: {len(tasks)} tasks ({len(gpu)} GPU, {len(tasks) - len(gpu)} CPU); "
          f"estimated GPU time {total / 3600:.1f} h (factor ~2)")
    print(f"{'label':44s} {'nodes':>10s} {'mom':>4s} {'est. min':>9s}")
    for t in gpu:
        nodes = (t["elements"] * t["degree"] + 1) ** t["dim"]
        print(f"{t['label']:44s} {nodes:10d} {moments(t):4d} {t['gpu_estimate_seconds'] / 60:9.1f}")
    return 0


def prepare(ns) -> int:
    ns.profile = ns.profile
    rc = nx.prepare(ns)
    if rc or ns.dry_run:
        return rc
    campaign = ns.campaign.expanduser().resolve()
    shutil.copy2(HERE / "tardis_convergence.py", campaign / "tardis.py")
    shutil.copy2(HERE / "nx_convergence.py", campaign / "nx_convergence.py")
    meta = nx.load(campaign / "manifest.json")
    for relative in ("tardis.py", "nx_convergence.py"):
        meta["checksums"][relative] = nx.digest(campaign / relative)
    meta["machine"] = "tardis (NVIDIA A100 80GB PCIe, EPYC 7343); GPU tasks: m1pn_fast --gpu"
    meta["solver_flags"] = "GPU tasks: --gpu --mirror-reduction off; all fast tasks: --angular-reduction auto"
    nx.atomic(campaign / "manifest.json", meta)
    print("entry point:", campaign / "tardis.py")
    return 0


def submit(ns) -> int:
    """Pack the tasks into sequential Slurm jobs on tardis (one GPU), by estimated run time."""
    campaign = ns.campaign.resolve(strict=True)
    tasks = nx.check_manifest(campaign)
    pending = [t for t in tasks if not (campaign / "status" / (t["label"] + ".json")).exists()]
    budget = ns.job_hours * 3600
    jobs, current, used = [], [], 0.
    for t in sorted(pending, key=lambda t: t.get("gpu_estimate_seconds") or 30.):
        cost = 3 * (t.get("gpu_estimate_seconds") or 30.) + 30
        if current and used + cost > budget:
            jobs.append(current)
            current, used = [], 0.
        current.append(t)
        used += cost
    if current:
        jobs.append(current)
    (campaign / "packs").mkdir(exist_ok=True)
    scripts = []
    for k, job in enumerate(jobs):
        hours = sum(3 * (t.get("gpu_estimate_seconds") or 30.) + 30 for t in job) / 3600
        partition = "short" if hours <= 1.8 else ("med" if hours <= 7.5 else "long")
        limit = {"short": "02:00:00", "med": "08:00:00", "long": "2-00:00:00"}[partition]
        lines = ["#!/usr/bin/env bash", f"#SBATCH --job-name=tardis-{k:03d}", f"#SBATCH --partition={partition}",
                 "#SBATCH --constraint=ampere", "#SBATCH --gres=gpu:1", "#SBATCH --cpus-per-task=32",
                 "#SBATCH --mem=240G", f"#SBATCH --time={limit}",
                 f"#SBATCH --output={campaign}/packs/job-{k:03d}-%j.log",
                 "export OMP_NUM_THREADS=32 OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1",
                 "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"]
        for t in job:
            timeout = int(min(40 * 3600, max(1800, 6 * (t.get("gpu_estimate_seconds") or 300.))))
            lines.append(f"[ -e {campaign}/status/{t['label']}.json ] || python3 {campaign}/tardis.py run "
                         f"--campaign {campaign} --task {t['label']} --threads 32 --timeout {timeout}")
        path = campaign / "packs" / f"job-{k:03d}.sbatch"
        path.write_text("\n".join(lines) + "\n")
        scripts.append((path, partition, len(job), hours))
    for path, partition, count, hours in scripts:
        print(f"{path.name}: {partition}, {count} tasks, budget {hours:.2f} h")
    if not ns.execute:
        print("dry run; add --execute to submit", len(scripts), "jobs")
        return 0
    record = []
    for path, *_ in scripts:
        out = subprocess.check_output(["sbatch", str(path)], text=True).strip()
        print(out)
        record.append(dict(script=path.name, sbatch=out))
    nx.atomic(campaign / "packs" / "submitted.json", record)
    return 0


def collect(ns) -> int:
    rc = nx.collect(ns)
    # Run-time table of all tasks (GPU and CPU).
    campaign = ns.campaign.resolve(strict=True)
    tasks = nx.load(campaign / "tasks.json")
    rows = ["| task | device | nodes | moments | steps | wall s | solver s | estimate s |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for t in tasks:
        status = campaign / "status" / (t["label"] + ".json")
        if not status.is_file():
            continue
        s = nx.load(status)
        meta = s.get("metadata", {})
        nodes = (t["elements"] * t["degree"] + 1) ** t["dim"]
        rows.append(f"| {t['label']} | {'GPU' if t.get('gpu') else 'CPU'} | {nodes} | "
                    f"{meta.get('active_moments', '')} | {meta.get('steps', '')} | {s.get('elapsed_seconds', 0):.1f} | "
                    f"{(meta.get('solver_seconds') or 0):.1f} | {(t.get('gpu_estimate_seconds') or 0):.0f} |")
    (campaign / "runtimes.md").write_text("# Run times (tardis)\n\n" + "\n".join(rows) + "\n")
    print("run-time table:", campaign / "runtimes.md")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="operation", required=True)
    root = nx.ROOT if (nx.ROOT / "scripts/element_weno").is_dir() else Path(os.environ.get("M1PN_ROOT", "."))
    p = sub.add_parser("plan")
    p.add_argument("--profile", choices=PROFILES, default="tardis-paper")
    p.set_defaults(action=plan)
    p = sub.add_parser("prepare")
    p.add_argument("--profile", choices=PROFILES, default="tardis-paper")
    p.add_argument("--repo", type=Path, default=root)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--fast-bin", type=Path, default=root / "tmp/build_fast/m1pn_fast")
    p.add_argument("--standalone-bin", type=Path, default=root / "tmp/build_fast/m1pn_element_weno")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="+")
    p.add_argument("--slot-cores", type=int, choices=[16, 32])
    p.set_defaults(action=prepare)
    p = sub.add_parser("run")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--timeout", type=int, default=25200)
    p.set_defaults(action=nx.run)
    p = sub.add_parser("collect")
    p.add_argument("--campaign", type=Path, required=True)
    p.set_defaults(action=collect)
    p = sub.add_parser("submit")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--job-hours", type=float, default=6.)
    p.add_argument("--execute", action="store_true")
    p.set_defaults(action=submit)
    ns = parser.parse_args()
    return ns.action(ns)


if __name__ == "__main__":
    raise SystemExit(main())
