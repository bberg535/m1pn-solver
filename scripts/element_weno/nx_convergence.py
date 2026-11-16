#!/usr/bin/env python3
"""Reproducible NX convergence campaign for the element-WENO M1PN scheme.

This deliberately reuses:
  nx/scripts/analysis.py (moment conversion, analytical crossing, grid checks)
  scripts/line_source/reference.py (Ganapol/Bennett-McClarren reference)
  nx/batches/element_weno_pack.sbatch and scripts/element_weno/pack_node.py.

No jobs are submitted without the explicit 'submit --execute' flag.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = (HERE.parents[1] if HERE.name == "element_weno" else Path(os.environ.get("M1PN_ROOT", HERE))).resolve()
FROZEN = "nx_convergence_v1"
# v2: heterogeneous/3D profiles (hetero-smoke, hetero-paper); v1 campaigns stay readable and unchanged.
FROZEN_V2 = "nx_convergence_v2"
FORMATS = (FROZEN, FROZEN_V2)
HETERO_PROFILES = ("hetero-smoke", "hetero-paper")
# Schotthoefer/Hauck (arXiv:2505.17284) Table 1, converted to rho units by 1/(2 pi) (hemisphere concentration;
# mass + absorption = 2 pi * 3.2 confirms the factor): mass at T, total absorption, absorption rate at T.
LATTICE_KINETIC = {"mass": 13.07 / (2 * math.pi), "absorbed": 7.034 / (2 * math.pi),
                   "absorption_rate": 4.597 / (2 * math.pi)}
# Historical path-B lattice time step of the 15 NX runs (older binary).
LATTICE_B_DT = 0.0005209181181832981


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load(path: Path):
    return json.loads(path.read_text())


def checked_within(path: Path, parent: Path) -> Path:
    path = path.resolve()
    if path != parent and parent not in path.parents:
        raise ValueError(f"path escapes campaign: {path}")
    return path


def commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def task(family: str, binary: str, case: str, inflow: str, dim: int,
         n: int, p: int, pn: int, end: float, *, kind="spatial",
         sigma_s=0., sigma_a=0., filter_strength=0., weno=True,
         limiter=True, rk="ssprk43", periodic=False, dt_level=None,
         reference=False, linear=1e-3, tag="", cfl=.2,
         stabilization="element-weno", background=None, shadow_sigma=None,
         skip_fields=False, expect_mirror=None, fixed_dt=None):
    label = "_".join(str(x) for x in
                     (family, f"P{pn}", f"p{p}", f"E{n}",
                      ("dt" + str(dt_level)) if dt_level is not None else
                      (tag or "std")))
    return dict(label=label, family=family, kind=kind, binary=binary,
                case=case, inflow=inflow, dim=dim, elements=n, degree=p,
                pn_order=pn, time=end, sigma_s=sigma_s, sigma_a=sigma_a,
                filter_strength=filter_strength, weno=weno, limiter=limiter,
                rk=rk, periodic=periodic, dt_level=dt_level, reference=reference,
                weno_linear=linear, cfl=cfl, stabilization=stabilization,
                background=background, shadow_sigma=shadow_sigma, skip_fields=skip_fields,
                expect_mirror=expect_mirror, fixed_dt=fixed_dt)


def make_hetero_tasks(profile: str) -> list[dict]:
    """Heterogeneous/3D families (docs/ELEMENT_HETEROGENEOUS.md). Q1/112, Q2/56, Q3/35, Q4/28 have 113 (Q3: 106)
    control points per axis; mirror expectations are part of the validation."""
    smoke = profile == "hetero-smoke"
    out = []
    lat = dict(background=1e-10, expect_mirror=[True, False, False])
    # 2D lattice, Paul (44)-(47), T = 3.2.
    for pn in ([3] if smoke else [3, 7, 11]):
        for p, n in ([(1, 28), (2, 14)] if smoke else [(1, 112), (2, 56), (3, 35), (4, 28)]):
            t = task("lattice", "fast", "lattice-paul", "vacuum", 2, n, p, pn, 3.2, kind="lattice", **lat)
            if n % 2:
                t["expect_mirror"] = [False, False, False]
            out.append(t)
    if not smoke:
        for pn in (7, 11):  # spatial refinement at fixed P_N
            for p, n in ((1, 224), (2, 112)):
                out.append(task("lattice", "fast", "lattice-paul", "vacuum", 2, n, p, pn, 3.2,
                                kind="lattice", tag="fine", **lat))
            # same dt as the historical path-B runs (equal time step comparison)
            out.append(task("lattice", "fast", "lattice-paul", "vacuum", 2, 112, 1, pn, 3.2, kind="lattice",
                            tag="dtB", fixed_dt=LATTICE_B_DT, **lat))
    # 3D shadow, stage A (moderate, Beer-Lambert reference) and stage B (sigma = 1e7, Weih et al.).
    # Archived on 12.11.2026 (docs/ELEMENT_HETEROGENEOUS.md, section 11b): the collimated delta beam is outside what
    # P_N with N <= 11 can represent, so Beer-Lambert measures the angular model, not the scheme. Only the smoke
    # profile keeps one shadow task as a functional test of the 3D/mirror/material machinery.
    sh = dict(background=1e-6, skip_fields=True, expect_mirror=[False, True, True])
    if smoke:
        out.append(task("shadowA", "fast", "shadow-weih", "shadow-beam", 3, 4, 2, 3, 1.6,
                        kind="shadow", tag="s10", shadow_sigma=10., **sh))
        out.append(task("shadowB", "fast", "shadow-weih", "shadow-beam", 3, 4, 2, 3, 1.6,
                        kind="shadow", tag="s1e7", shadow_sigma=1e7, **sh))
    # Own 3D lattice extension (asymmetric: no mirror).
    for pn in ([3] if smoke else [3, 7]):
        for p, n in ([(1, 14)] if smoke else [(1, 28), (2, 14), (2, 28)]):
            out.append(task("lattice3d", "fast", "lattice-3d", "vacuum", 3, n, p, pn, 1. if smoke else 3.2,
                            kind="lattice3d", background=1e-10, skip_fields=True,
                            expect_mirror=[False, False, False]))
    return out


def make_tasks(profile: str) -> list[dict]:
    if profile in HETERO_PROFILES:
        out = make_hetero_tasks(profile)
        seen = set()
        for number, t in enumerate(out):
            if t["label"] in seen:
                raise ValueError("duplicate task label: " + t["label"])
            seen.add(t["label"])
            t["id"] = number
            t.update(cost_model(t))
        return out
    smoke = profile == "smoke"
    if profile not in ("smoke", "paper"):
        raise ValueError("profile must be smoke, paper, hetero-smoke or hetero-paper")
    out = []
    # True P_N smooth exact solution is built into the standalone executable.
    for p in ([1, 2] if smoke else [1, 2, 3, 4]):
        for n in ([8, 16] if smoke else [8, 16, 32, 64]):
            out.append(task("smooth", "standalone", "smooth", "vacuum", 1, n, p,
                            3, .1, kind="smooth_exact", periodic=True, tag="spatial",
                            cfl=.15))
            if n == (16 if smoke else 64):
                out.append(task("smoothdt2", "standalone", "smooth", "vacuum", 1,
                                n, p, 3, .1, kind="smooth_exact", periodic=True,
                                tag="temporal_check", cfl=.075))
    # Open-boundary compatible kinetic inflow, exact kinetic density available,
    # but finite-PN modeling error persists and must not be called FE error.
    for pn in ([3] if smoke else [3, 7]):
        for n in ([8, 16] if smoke else [8, 16, 32, 64]):
            out.append(task("boundary", "fast", "smooth", "smooth-transport",
                            2, n, 2, pn, .15, kind="boundary_exact"))
    # Crossing: same P_N compares against an analytical kinetic reference
    # AND against a finer spatial run; these are distinct error definitions.
    for pn in ([3] if smoke else [3, 7]):
        for p in ([1] if smoke else [1, 2, 3]):
            for n in ([8, 16] if smoke else [8, 16, 32, 64]):
                out.append(task("crossing", "fast", "crossing-study",
                                "crossing-study-transport", 2, n, p, pn, .32))
    # GHS physical line source rescaled to the element-WENO unit square.
    for pn in ([3] if smoke else [3, 7]):
        for p in ([1] if smoke else [1, 2]):
            for n in ([16] if smoke else [32, 64, 128]):
                out.append(task("line", "fast", "line-source", "vacuum", 2,
                                n, p, pn, 1 / 3, sigma_s=3., kind="line_exact"))
    # Time refinement: fixed spatial mesh and P_N, same semidiscrete system,
    # finest dt is self-reference, not a kinetic/analytic exact solution.
    for name, case, inflow, dim, n, p, pn, end, scattering, absorption, filt in [
        ("time_smooth", "smooth", "vacuum", 1, 16 if smoke else 48,
         2, 3, .12, 2., .5, 0.),
        ("time_crossing", "crossing-study", "crossing-study-transport",
         2, 12 if smoke else 32, 2, 7, .08, 2., .5, .15)]:
        levels = [0, 1, 2] if smoke else [0, 1, 2, 3, 4]
        for level in levels:
            out.append(task(name, "fast", case, inflow, dim, n, p, pn, end,
                            kind="temporal", periodic=(dim == 1),
                            sigma_s=scattering, sigma_a=absorption,
                            filter_strength=filt, dt_level=level,
                            reference=(level == max(levels)), cfl=.3))
    # Ablation and baseline: no-limit is diagnostic and may fail legitimately.
    for key, weno, limit, rk in [
        ("full", True, True, "ssprk43"),
        ("no_weno", False, True, "ssprk43"),
        ("no_limit", True, False, "ssprk43"),
        ("ssprk3", True, True, "ssprk3")]:
        out.append(task("ablation", "fast", "crossing-study",
                        "crossing-study-transport", 2, 16 if smoke else 64,
                        2, 7, .32, kind="ablation", tag=key, weno=weno,
                        limiter=limit, rk=rk))
    if not smoke:
        out.append(task("ablation", "fast", "crossing-study",
                        "crossing-study-transport", 2, 64, 2, 7, .32,
                        kind="ablation", tag="weno_weight_02", linear=.2))
    seen = set()
    for number, t in enumerate(out):
        if t["label"] in seen:
            raise ValueError("duplicate task label: " + t["label"])
        seen.add(t["label"])
        t["id"] = number
    return out


def active_moments(t: dict) -> int:
    n = t["pn_order"]
    return (n + 1) * (n + 2) // 2 + 1 if t["dim"] == 2 else (n + 1) ** 2


def cost_model(t: dict) -> dict:
    """Resource estimate (docs/ELEMENT_HETEROGENEOUS.md, Sec. 10), calibrated on the i5-12500 workstation
    (6 threads, 11.11.2026): core-seconds = (a + b * moments) * nodes * stages with a = 1.65e-6 (2D) / 5.0e-6 (3D),
    b = 4.5e-8; dt / h = c_p from the local runs (2D: 0.25, 0.15, 0.08, 0.05 for p = 1..4; 3D: x 0.67).
    Uncertainty about a factor 2; NX cores are not calibrated separately. Memory ~ 30 fields of nodes x moments."""
    p, dim = t["degree"], t["dim"]
    lengths = {"lattice-paul": 7., "lattice-3d": 7., "shadow-weih": .5}.get(t["case"], 1.)
    h = lengths / t["elements"]
    counts = [t["elements"]] * dim
    if t["case"] == "shadow-weih":
        counts[0] *= 3
    mirror = t.get("expect_mirror") or [False] * 3
    nodes = 1
    for d in range(dim):
        n = counts[d] * p + 1
        nodes *= (n + 1) // 2 if mirror[d] else n
    moments = active_moments(t) if t.get("stabilization", "element-weno") == "element-weno" else (t["pn_order"] + 1) ** 2
    c = {1: .25, 2: .15, 3: .08, 4: .05}.get(p, .05) * (.67 if dim == 3 else 1)
    # dt / h above was measured at cfl = 1; tasks use t["cfl"] (0.2 in this study).
    dt = t.get("fixed_dt") or c * h * t.get("cfl", 1.)
    stages = 4 * math.ceil(t["time"] / dt)
    a = 5.0e-6 if dim == 3 else 1.65e-6
    # Recalibrated on ew_hetero_paper (NX, 16 cores): 3D runs are 1.8 times slower than the workstation model.
    core_seconds = (a + 4.5e-8 * moments) * nodes * stages * (1.8 if dim == 3 else 1.)
    memory_gb = 30 * nodes * moments * 8 / 1e9 + .2
    slot = 32 if memory_gb > 48 or core_seconds > 16 * 6 * 3600 else 16
    return dict(estimate_nodes=nodes, estimate_moments=moments, estimate_steps=stages // 4,
                estimate_core_hours=core_seconds / 3600, estimate_memory_gb=memory_gb,
                estimate_wall_hours=core_seconds / 3600 / slot, slot_cores=slot)


def hetero_flags(t: dict, output: Path, threads: int, end, dt, cfl):
    args = ["--dimension", str(t["dim"]), "--mesh-type", "hex",
            "--elements", str(t["elements"]), "--degree", str(t["degree"]),
            "--pn-order", str(t["pn_order"]), "--case", t["case"], "--inflow", t["inflow"],
            "--model", "r2", "--stabilization", t["stabilization"], "--time-integrator", t["rk"],
            "--cfl", str(cfl), "--dt", f"{(t.get('fixed_dt') or 0.) if dt is None else dt:.17g}",
            "--time", f"{end:.17g}", "--threads", str(threads), "--output", str(output),
            "--background", f"{t['background']:.17g}"]
    if t["stabilization"] == "element-weno":
        args += ["--weno", "on" if t["weno"] else "off", "--element-limit", "on" if t["limiter"] else "off",
                 "--weno-linear", str(t["weno_linear"]), "--history-stride", "50",
                 "--angular-reduction", "auto", "--mirror-reduction", "auto"]
        if t.get("shadow_sigma") is not None:
            args += ["--shadow-sigma", f"{t['shadow_sigma']:.17g}"]
        if t.get("skip_fields"):
            args.append("--skip-fields")
    else:
        raise ValueError("only element-weno tasks (mass correction removed 2026-11-16)")
    return args


def flags(t: dict, output: Path, threads: int, *, end=None, dt=None, cfl=None):
    if end is None:
        end = t["time"]
    if cfl is None:
        cfl = t["cfl"]
    if t.get("kind") in ("lattice", "shadow", "lattice3d"):
        return hetero_flags(t, output, threads, end, dt, cfl)
    if t["binary"] == "standalone":
        args = ["--problem", "smooth", "--dimension", "1",
                "--pn-order", str(t["pn_order"]), "--degree", str(t["degree"]),
                "--elements", str(t["elements"]), "--time", str(end),
                "--cfl", str(cfl), "--output", str(output), "--ssprk43"]
        if dt is not None:
            args += ["--dt", f"{dt:.17g}"]
        return args
    args = ["--dimension", str(t["dim"]), "--mesh-type", "hex",
            "--elements", str(t["elements"]), "--degree", str(t["degree"]),
            "--pn-order", str(t["pn_order"]), "--case", t["case"],
            "--inflow", t["inflow"], "--model", "r2",
            "--stabilization", t.get("stabilization", "element-weno"),
            "--time-integrator", t["rk"], "--cfl", str(cfl),
            "--dt", f"{0. if dt is None else dt:.17g}",
            "--time", f"{end:.17g}", "--threads", str(threads),
            "--output", str(output), "--sigma-s", str(t["sigma_s"]),
            "--sigma-a", str(t["sigma_a"]), "--filter",
            "constant" if t["filter_strength"] else "off",
            "--filter-strength", str(t["filter_strength"]),
            "--angular-reduction", "off", "--mirror-reduction", "off"]
    if t["periodic"]:
        args.append("--periodic")
    if t["case"] == "crossing-study":
        args += ["--pulse-width", ".07", "--background", ".001"]
    if t["case"] == "line-source":
        args += ["--pulse-width", ".01", "--background", ".0009"]
    # Every task carries "stabilization" explicitly (smoke 2026-11-11: a missing key silently dropped
    # --weno/--element-limit/--weno-linear and passed mass-correction options to element-WENO runs).
    if t["stabilization"] == "element-weno":
        args += ["--weno", "on" if t["weno"] else "off",
                 "--element-limit", "on" if t["limiter"] else "off",
                 "--weno-linear", str(t["weno_linear"])]
    else:
        raise ValueError("only element-weno tasks (mass correction removed 2026-11-16)")
    return args


def prepare(ns) -> int:
    tasks = make_tasks(ns.profile)
    if getattr(ns, "only", None):
        # Supplementary campaign (e.g. recovery of failed tasks): a subset of the profile, same labels and settings.
        unknown = set(ns.only) - {t["label"] for t in tasks}
        if unknown:
            raise ValueError("unknown task labels: " + ", ".join(sorted(unknown)))
        tasks = [t for t in tasks if t["label"] in set(ns.only)]
    if getattr(ns, "slot_cores", None):
        for t in tasks:
            t["slot_cores"] = ns.slot_cores
            t["estimate_wall_hours"] = t.get("estimate_core_hours", 0) / ns.slot_cores
    print(f"{ns.profile}: {len(tasks)} tasks; groups: "
          + ", ".join(sorted(set(t["family"] for t in tasks))))
    if ns.dry_run:
        for t in tasks:
            print(t["label"], t["kind"], t["case"], t["rk"])
        return 0
    campaign = ns.campaign.expanduser().resolve()
    if campaign.exists():
        raise FileExistsError("campaign already exists: " + str(campaign))
    repo = ns.repo.expanduser().resolve()
    for path in (repo / "nx/scripts/analysis.py",
                 repo / "scripts/line_source/reference.py",
                 repo / "scripts/element_weno/pack_node.py",
                 repo / "nx/batches/element_weno_pack.sbatch"):
        if not path.is_file():
            raise FileNotFoundError(path)
    binaries = {"fast": ns.fast_bin.expanduser().resolve(strict=True),
                "standalone": ns.standalone_bin.expanduser().resolve(strict=True)}
    for path in binaries.values():
        if not os.access(path, os.X_OK):
            raise ValueError("not executable: " + str(path))
    campaign.mkdir(parents=True)
    for folder in ("bin", "runs", "logs", "status", "packs", "references",
                   "scripts/element_weno", "nx/batches"):
        (campaign / folder).mkdir(parents=True, exist_ok=True)
    for key, path in binaries.items():
        shutil.copy2(path, campaign / "bin" / key)
    sources = {
        HERE / "nx_convergence.py": campaign / "study.py",
        HERE / "nx_plot_convergence.py": campaign / "plot.py",
        repo / "nx/scripts/analysis.py": campaign / "analysis.py",
        repo / "scripts/line_source/reference.py": campaign / "reference.py",
        repo / "scripts/element_weno/pack_node.py": campaign / "scripts/element_weno/pack_node.py",
        repo / "nx/batches/element_weno_pack.sbatch": campaign / "nx/batches/element_weno_pack.sbatch"}
    if not (HERE / "nx_convergence.py").is_file():
        raise FileNotFoundError("study source must be launched from repository scripts/element_weno")
    for source, target in sources.items():
        shutil.copy2(source, target)
    atomic(campaign / "tasks.json", tasks)
    hashes = {}
    for relative in ["tasks.json", "study.py", "plot.py", "analysis.py", "reference.py",
                     "bin/fast", "bin/standalone",
                     "scripts/element_weno/pack_node.py",
                     "nx/batches/element_weno_pack.sbatch"]:
        hashes[relative] = digest(campaign / relative)
    atomic(campaign / "manifest.json",
           {"format": FROZEN_V2 if ns.profile in HETERO_PROFILES else FROZEN, "profile": ns.profile,
            "mfem_version": mfem_version(repo), "compiler_from": "config.json of every run",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repository": "bberg535/m1pn-solver", "git_head": commit(),
            "root": str(repo), "task_count": len(tasks),
            "checksums": hashes, "reactions": "Strang exact half steps",
            "interpretation": "kinetic reference includes PN truncation; temporal self reference only"})
    print("frozen campaign:", campaign)
    return 0


def mfem_version(repo: Path) -> str:
    """No MFEM since 2026-11-16 (element WENO only); kept for the manifest format."""
    return "none"

def runtime_environment() -> dict:
    """Threads, CPU affinity, NUMA nodes of the allowed CPUs and OpenMP variables of this task."""
    cpus = sorted(os.sched_getaffinity(0))
    nodes = set()
    for cpu in cpus:
        for entry in Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"):
            nodes.add(int(entry.name[4:]))
    return {"host": os.uname().nodename, "affinity_cpus": len(cpus),
            "affinity": f"{cpus[0]}-{cpus[-1]}" if cpus else "", "numa_nodes": sorted(nodes),
            "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
            "omp": {k: v for k, v in os.environ.items() if k.startswith(("OMP_", "OPENBLAS_", "MKL_"))}}


def check_manifest(campaign: Path) -> list[dict]:
    meta = load(campaign / "manifest.json")
    if meta.get("format") not in FORMATS:
        raise ValueError("invalid campaign format")
    for relative, expected in meta["checksums"].items():
        file = checked_within(campaign / relative, campaign)
        if not file.is_file() or digest(file) != expected:
            raise ValueError("frozen input hash mismatch: " + relative)
    tasks = load(campaign / "tasks.json")
    if len(tasks) != meta["task_count"]:
        raise ValueError("task count mismatch")
    return tasks


def run(ns) -> int:
    campaign = ns.campaign.resolve(strict=True)
    tasks = check_manifest(campaign)
    selected = [t for t in tasks if t["label"] == ns.task]
    if len(selected) != 1:
        raise ValueError("unknown task " + ns.task)
    t = selected[0]
    output = checked_within(campaign / "runs" / t["label"], campaign)
    status = campaign / "status" / (t["label"] + ".json")
    if output.exists() or status.exists():
        raise FileExistsError("run already exists; no implicit overwrite: " + t["label"])
    binary = campaign / "bin" / t["binary"]
    command = [str(binary)]
    if t["binary"] == "standalone":
        # dt scaling: temporal truncation should shrink faster than h^(p+1).
        h = 1 / t["elements"]
        exponent = (t["degree"] + 1) / 3 + .15
        target_dt = .25 * h ** exponent
        if t["family"] == "smoothdt2":
            target_dt /= 2
        options = flags(t, output, ns.threads, dt=target_dt)
    elif t["kind"] == "temporal":
        probe = campaign / "runs" / (t["label"] + "_probe")
        probe_args = flags(t, probe, ns.threads, end=0., dt=None)
        log = campaign / "logs" / (t["label"] + "_probe.log")
        with log.open("w") as out:
            proc = subprocess.run([str(binary), *probe_args], stdout=out,
                                  stderr=subprocess.STDOUT, check=False)
        if proc.returncode:
            atomic(status, {"status": "probe_failed", "rc": proc.returncode})
            return proc.returncode
        config = load(probe / "config.json")
        base = float(config["dt_selected"])
        if not math.isfinite(base) or base <= 0:
            raise ValueError("invalid pilot dt_selected")
        # Divide exact final time into an integer number of steps.
        tentative = base / (2 ** t["dt_level"])
        steps = max(1, math.ceil(t["time"] / tentative))
        dt = t["time"] / steps
        options = flags(t, output, ns.threads, dt=dt)
    else:
        options = flags(t, output, ns.threads)
    command += options
    log = campaign / "logs" / (t["label"] + ".log")
    started = time.monotonic()
    with log.open("w") as out:
        try:
            proc = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT,
                                  timeout=ns.timeout, check=False)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    elapsed = time.monotonic() - started
    outcome = dict(status="failed", label=t["label"], return_code=rc,
                   elapsed_seconds=elapsed, command=command, log=str(log))
    try:
        if rc != 0:
            raise ValueError("solver returned " + str(rc))
        summary = load(output / "summary.json")
        if t["binary"] == "standalone":
            for name in ("l2_density_error", "final_mass", "min_coefficient_margin"):
                if not math.isfinite(float(summary[name])):
                    raise ValueError("invalid standalone summary " + name)
            if not (math.isfinite(float(summary["dt"])) and float(summary["dt"]) > 0):
                raise ValueError("standalone dt invalid")
            if any(int(summary[k]) != t[v] for k, v in
                   (("pn_order", "pn_order"), ("degree", "degree"),
                    ("elements", "elements"), ("dimension", "dim"))):
                raise ValueError("standalone solver metadata mismatch")
        elif t.get("kind") in ("lattice", "shadow", "lattice3d"):
            config = load(output / "config.json")
            validate_hetero(t, config, summary)
        else:
            config = load(output / "config.json")
            if summary.get("status") != "completed" or config.get("case") != t["case"]:
                raise ValueError("solver did not complete requested case")
            if config.get("time_integrator") != t["rk"]:
                raise ValueError("time-integrator mismatch")
            if int(config.get("degree", -1)) != t["degree"]:
                raise ValueError("Bernstein degree mismatch")
            if abs(float(summary["time"]) - t["time"]) > 1e-10:
                raise ValueError("end time mismatch")
            if int(config.get("pn_order", -1)) != t["pn_order"] or \
               int(config.get("elements_per_axis", -1)) != t["elements"] or \
               int(config.get("dimension", -1)) != t["dim"]:
                raise ValueError("grid or PN order mismatch")
            if t.get("stabilization", "element-weno") == "element-weno":
                if config.get("method") != "element-weno" or \
                   config.get("weno") is not t["weno"] or \
                   config.get("element_limit") is not t["limiter"]:
                    raise ValueError("element-WENO configuration mismatch")
                if abs(float(config.get("weno_linear", -1)) - float(t["weno_linear"])) > 1e-15:
                    raise ValueError("WENO linear weight mismatch")
                if abs(float(config.get("sigma_s", -1)) - t["sigma_s"]) > 1e-12 or \
                   abs(float(config.get("sigma_a", -1)) - t["sigma_a"]) > 1e-12:
                    raise ValueError("reaction coefficients mismatch")
                expected_filter = "constant" if t["filter_strength"] else "off"
                if config.get("filter_mode") != expected_filter:
                    raise ValueError("filter mode mismatch")
            if t["kind"] == "temporal" and abs(float(config["dt_selected"]) - dt) > 1e-12 * max(1., dt):
                raise ValueError("requested temporal dt changed")
        outcome["summary_sha256"] = digest(output / "summary.json")
        if (output / "final.csv").is_file():
            outcome["final_sha256"] = digest(output / "final.csv")
        outcome["file_sha256"] = {
            name: digest(output / name)
            for name in ("summary.json", "final.csv", "config.json",
                         "initial.csv", "nodal_mass.csv", "history.csv", "slices.csv")
            if (output / name).is_file()
        }
        required = required_files(t)
        outcome["environment"] = runtime_environment()
        if t["binary"] == "fast":
            cfg = load(output / "config.json")
            outcome["metadata"] = {k: cfg.get(k) for k in (
                "compiler", "threads", "thread_policy", "pn_order", "Q", "active_moments", "degree",
                "coefficients", "spatial_dofs_full_domain", "dt_selected", "time_integrator",
                "mirror_directions", "angular_reduction")}
            outcome["metadata"].update({k: summary.get(k) for k in (
                "steps", "solver_seconds", "total_seconds", "peak_rss_kib_rank0", "phase_seconds", "ledger",
                "minimum_cone_margin", "nudged_states", "limited_element_stages")})
        if not all(name in outcome["file_sha256"] for name in required):
            raise ValueError("required saved-field files are missing")
        outcome["status"] = "completed"
    except Exception as error:
        outcome["status"] = "failed"
        outcome["error"] = str(error)
    atomic(status, outcome)
    print(t["label"], outcome["status"], "in", round(elapsed, 2), "s")
    return 0 if outcome["status"] == "completed" else 2


def required_files(t: dict) -> tuple:
    if t["binary"] == "standalone":
        return ("summary.json", "final.csv")
    if t.get("kind") in ("lattice", "shadow", "lattice3d") and t.get("stabilization") == "element-weno":
        files = ["summary.json", "config.json", "history.csv"]
        files += ["slices.csv"] if t["dim"] == 3 else []
        files += [] if t.get("skip_fields") else ["final.csv", "initial.csv", "nodal_mass.csv"]
        return tuple(files)
    return ("summary.json", "final.csv", "config.json", "initial.csv", "nodal_mass.csv")


def validate_hetero(t: dict, config: dict, summary: dict) -> None:
    """Heterogeneous/3D runs: case, configuration, material integrals, ledger balance, cone margin, mirrors."""
    if summary.get("status") != "completed" or config.get("case") != t["case"]:
        raise ValueError("solver did not complete requested case")
    if abs(float(summary["time"]) - t["time"]) > 1e-10:
        raise ValueError("end time mismatch")
    if int(config.get("pn_order", -1)) != t["pn_order"] or int(config.get("dimension", -1)) != t["dim"]:
        raise ValueError("PN order or dimension mismatch")
    if t.get("fixed_dt") and abs(float(config["dt_selected"]) - t["fixed_dt"]) > 1e-12 * t["fixed_dt"]:
        raise ValueError("fixed dt not used")
    if t["stabilization"] != "element-weno":
        return  # path B: its own checks (material integrals are part of its config)
    if config.get("method") != "element-weno" or config.get("time_integrator") != t["rk"] or \
       int(config.get("degree", -1)) != t["degree"] or int(config.get("elements_per_axis", -1)) != t["elements"]:
        raise ValueError("element-WENO configuration mismatch")
    if config.get("weno") is not t["weno"] or config.get("element_limit") is not t["limiter"] or \
       abs(float(config.get("weno_linear", -1)) - float(t["weno_linear"])) > 1e-15:
        raise ValueError("WENO/limiter configuration mismatch")
    expected, got = config.get("material_expected") or {}, config["material_integrals"]
    tolerance = {"lattice": 1e-10, "lattice3d": 1e-10, "shadow": 5e-3}[t["kind"]]
    for key in ("volume", "source", "sigma_a", "sigma_s"):
        if key in expected and abs(got[key] - expected[key]) > tolerance * max(1., abs(expected[key])):
            raise ValueError(f"material integral {key}: {got[key]} vs {expected[key]}")
    if t.get("expect_mirror") is not None and [bool(v) for v in config["mirror_directions"]] != list(t["expect_mirror"]):
        raise ValueError(f"mirror directions {config['mirror_directions']} != expected {t['expect_mirror']}")
    ledger = summary["ledger"]
    if not abs(float(ledger["balance_relative"])) <= 1e-9:
        raise ValueError("particle balance does not close: " + str(ledger["balance_relative"]))
    # Numerical cone check of the stored coefficients (rounding tolerance only; not a proof).
    if not float(summary["minimum_cone_margin"]) >= -1e-14 * max(1., float(summary["final_density_mass"])):
        raise ValueError("cone margin negative")


def read_field(path: Path):
    import numpy as np
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        head = next(reader)
        rows = [[float(v) for v in row] for row in reader if row]
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("invalid/nonfinite field " + str(path))
    return head, values


def rel_norm(error, exact, mass):
    import numpy as np
    den1 = np.sum(mass * np.abs(exact))
    den2 = np.sum(mass * exact ** 2)
    return dict(L1=float(np.sum(mass * np.abs(error)) / den1) if den1 > 0 else None,
                L2=float(np.sqrt(np.sum(mass * error ** 2) / den2)) if den2 > 0 else None,
                Linf=float(np.max(np.abs(error))))


def shadow_metrics(run: Path, sigma: float) -> dict:
    """Stage-A diagnostics from slices.csv (plane z = 0): L1 error of rho against the uncollided Beer-Lambert
    density of the collimated beam (radial profile of cases::shadow_beam_profile) behind the sphere front, x in [0.2, 0.7] (shadow region and its neighbourhood),
    centre-line density at x = 0.5 and transverse shadow width. The kinetic reference is exact for the collimated
    beam; the PN/M1 angular model and the background 1e-6 are part of the deviation."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    rows = [r for r in csv.DictReader(open(run / "slices.csv")) if r["plane"] == "2"]
    x = np.array([float(r["x"]) for r in rows]); y = np.array([float(r["y"]) for r in rows])
    rho = np.array([float(r["rho"]) for r in rows])
    R = .1
    def chord(px, py):
        r2 = py * py
        half = np.sqrt(np.maximum(R * R - r2, 0))
        a = np.maximum(-half, -.75); b = np.minimum(half, px)
        return np.where(r2 < R * R, np.maximum(0, b - a), 0.)
    r = np.abs(y)  # plane z = 0
    prof = np.where(r <= .15, 1., np.where(r >= .2, 0., np.cos(.5 * np.pi * (r - .15) / .05) ** 2))
    ref = prof * np.exp(-sigma * chord(x, y))
    mask = (x >= .2) & (x <= .7)
    l1 = float(np.sum(np.abs(rho[mask] - ref[mask])) / np.sum(ref[mask]))
    centre = rho[(np.abs(y) < 1e-12)]
    cx = x[(np.abs(y) < 1e-12)]
    at = float(np.interp(.5, cx, centre)) if len(cx) else float("nan")
    line = mask & (np.abs(x - .5) == np.min(np.abs(x[mask] - .5)))
    # width of the shadow: extent in y (|y| < 0.15, inside the beam core) where rho < 0.5 on the line x ~ 0.5
    core = line & (np.abs(y) < .15)
    width = float(np.sum(rho[core] < .5) * (np.ptp(y[line]) / max(1, line.sum() - 1)))
    return dict(shadow_L1_plane=l1, shadow_centre_x05=at, shadow_centre_ref=float(np.exp(-sigma * 2 * R)),
                shadow_width_half_density=width, shadow_width_ref=2 * R)


def collect_hetero(campaign: Path, tasks: list[dict]) -> int:
    rows, failures = [], []
    for t in tasks:
        label, output = t["label"], campaign / "runs" / t["label"]
        status_file = campaign / "status" / (label + ".json")
        if not status_file.is_file():
            failures.append((label, "missing status")); continue
        status = load(status_file)
        if status.get("status") != "completed":
            failures.append((label, status.get("error", status.get("status")))); continue
        try:
            for name, expected in status["file_sha256"].items():
                if digest(output / name) != expected:
                    raise ValueError("saved file changed after run: " + name)
            summary, config = load(output / "summary.json"), load(output / "config.json")
            row = dict(label=label, family=t["family"], kind=t["kind"], pn=t["pn_order"], p=t["degree"],
                       elements=t["elements"], stabilization=t["stabilization"], steps=summary.get("steps"),
                       dt=config.get("dt_selected"), solver_seconds=summary.get("solver_seconds"),
                       total_seconds=summary.get("total_seconds"), rss_mb=summary.get("peak_rss_kib_rank0", 0) / 1024,
                       threads=status.get("environment", {}).get("affinity_cpus"),
                       numa=status.get("environment", {}).get("numa_nodes"),
                       moments=config.get("active_moments", (t["pn_order"] + 1) ** 2),
                       dofs=config.get("spatial_dofs_full_domain", config.get("coefficients")),
                       mass=summary.get("final_density_mass"), rho_min=summary.get("minimum_coefficient_density"),
                       cone_margin=summary.get("minimum_cone_margin"), estimate_core_hours=t.get("estimate_core_hours"))
            if "ledger" in summary:
                row.update({"ledger_" + k: v for k, v in summary["ledger"].items()})
                row["absorption_rate"] = summary.get("absorption_rate_final")
                for k, v in (summary.get("phase_seconds") or {}).items():
                    row["phase_" + k] = v
            if "lattice_fcs" in summary:  # FCS: grid = collided part; compare the totals (docs/FCS_LATTICE.md)
                lf = summary["lattice_fcs"]
                row["fcs"] = True
                row["collided_mass"], row["uncollided_mass"] = row["mass"], lf["uncollided_mass_final"]
                row["mass"] = lf["total_mass_final"]
                row["ledger_absorbed"] = lf["total_absorbed"]
                row["absorption_rate"] = lf["total_absorption_rate_final"]
                row["fcs_table_seconds"] = lf["table_seconds"]
            if t["kind"] == "lattice" and t["stabilization"] == "element-weno":
                row["dev_mass_kinetic"] = row["mass"] / LATTICE_KINETIC["mass"] - 1
                row["dev_absorbed_kinetic"] = row["ledger_absorbed"] / LATTICE_KINETIC["absorbed"] - 1
                row["dev_absorption_rate_kinetic"] = row["absorption_rate"] / LATTICE_KINETIC["absorption_rate"] - 1
            if t["kind"] == "shadow":
                row["shadow_sigma"] = t["shadow_sigma"]
                row.update(shadow_metrics(output, t["shadow_sigma"]))
            rows.append(row)
        except Exception as error:  # report, never fabricate
            failures.append((label, str(error)))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("label", "family", "kind"), k))
    with open(campaign / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    atomic(campaign / "results.json", rows)
    lines = ["# Heterogeneous / 3D element-WENO campaign", "",
             f"Completed {len(rows)} / {len(tasks)} tasks; failures: {len(failures)}.", "",
             "- Lattice deviations are against Schotthoefer/Hauck Table 1 / (2 pi) (kinetic, S22, different initial",
             "  density 0 and inflow model); historical path-B NX runs used an older binary (not timed here).",
             "- Shadow stage A compares with the uncollided Beer-Lambert density of the collimated beam (radius 0.15,",
             "  cos^2 taper to 0.2; own choice, Weih et al. do not specify it); the PN/M1 angular model and the",
             "  background 1e-6 are part of the deviation.",
             "- lattice-3d is an own test case; no literature reference exists.", ""]
    if failures:
        lines += ["## Failed or missing", ""] + [f"- {a}: {b}" for a, b in failures]
    (campaign / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("completed", len(rows), "of", len(tasks), "; failures", len(failures))
    return 0 if not failures else 2


def collect(ns) -> int:
    import numpy as np
    campaign = ns.campaign.resolve(strict=True)
    tasks = check_manifest(campaign)
    if load(campaign / "manifest.json").get("profile") in HETERO_PROFILES:
        return collect_hetero(campaign, tasks)
    sys.path.insert(0, str(campaign))
    import analysis as ana  # frozen nx/scripts/analysis.py
    data, rows, failures = {}, [], []
    for t in tasks:
        label = t["label"]
        output = campaign / "runs" / label
        status_file = campaign / "status" / (label + ".json")
        if not status_file.is_file():
            failures.append((label, "missing status"))
            continue
        status = load(status_file)
        if status.get("status") != "completed":
            failures.append((label, status.get("error", status.get("status"))))
            continue
        try:
            required = {"summary.json", "final.csv"} if t["binary"] == "standalone" else {
                "summary.json", "final.csv", "config.json", "initial.csv", "nodal_mass.csv"}
            if not required.issubset(status.get("file_sha256", {})):
                raise ValueError("missing saved-field digests")
            for name, expected in status["file_sha256"].items():
                if digest(output / name) != expected:
                    raise ValueError("saved field changed after run: " + name)
            if digest(output / "summary.json") != status["summary_sha256"] or \
               digest(output / "final.csv") != status["final_sha256"]:
                raise ValueError("saved field changed after run")
            summary = load(output / "summary.json")
            config = load(output / "config.json") if t["binary"] == "fast" else {}
            head, final = read_field(output / "final.csv")
            row = dict(label=label, family=t["family"], kind=t["kind"], pn=t["pn_order"],
                       p=t["degree"], elements=t["elements"], h=1. / t["elements"],
                       rk=t["rk"], level=t["dt_level"],
                       elapsed_seconds=status["elapsed_seconds"],
                       steps=int(summary["steps"]))
            if t["binary"] == "standalone":
                if head != ["x", "y", "z", "density", "jx", "jy", "jz"]:
                    raise ValueError("unexpected standalone field header")
                physical = final[:, 3:7]
                coords = final[:, :3]
                row["dt"] = float(summary["dt"])
                row["error_exact_PN_L2"] = float(summary["l2_density_error"])
                row["minimum_cone_margin"] = float(summary["min_coefficient_margin"])
                # The builtin smooth exact comparison uses finite-element
                # quadrature; no claim of higher order without time checks.
                mass = None
            else:
                if head[:4] != ["node", "x", "y", "z"]:
                    raise ValueError("unexpected fast field header")
                coords = final[:, 1:4]
                physical = ana.physical_low(final[:, 4:])
                mh, mv = read_field(output / "nodal_mass.csv")
                if mh != ["node", "mass"] or len(mv) != len(final) or \
                   not np.array_equal(mv[:, 0], final[:, 0]):
                    raise ValueError("mass / final grid mismatch")
                mass = mv[:, 1]
                if np.any(mass <= 0) or abs(np.sum(mass) - 1.) > 1e-10:
                    raise ValueError("sample weights do not sum to unit domain volume")
                row["dt"] = float(config["dt_selected"])
                row["minimum_cone_margin"] = float(summary["minimum_cone_margin"])
                row["nudged_states"] = int(summary.get("nudged_states", 0))
                row["max_limited_zero_sum"] = float(summary.get("max_limited_zero_sum", 0))
                if (t.get("stabilization", "element-weno") == "element-weno"
                        and t["limiter"] and row["minimum_cone_margin"] < -1e-10):
                    raise ValueError("negative Bernstein coefficient cone margin")
                row["min_element_factor"] = float(summary.get("min_element_factor", 1.))
                row["min_weno_gamma"] = float(summary.get("min_weno_gamma", 1.))
                row["limited_element_stages"] = int(summary.get("limited_element_stages", 0))
                row["rss_kib"] = int(summary.get("peak_rss_kib_rank0", 0))
                if not np.isfinite(physical).all():
                    raise ValueError("nonfinite moments")
                # Bound is conditional on the limiter being enabled.
                if t["limiter"] and t.get("coupling", "m1pn") == "m1pn" and \
                   (float(physical[:, 0].min()) < -1e-10 or
                    float((physical[:, 0] - np.linalg.norm(physical[:, 1:4], axis=1)).min()) < -1e-9):
                    raise ValueError("M1 cone violation in stored field")
                if t["case"] == "crossing-study" and not t["sigma_s"] and not t["sigma_a"] and not t["filter_strength"]:
                    if "beam_geometry" in config:
                        reference, _ = ana.beam_exact(coords, t["time"], config["beam_geometry"])
                        row["error_kinetic_L2"] = rel_norm(physical[:, 0] - reference, reference, mass)["L2"]
                        row["reference"] = "analytic kinetic density (includes PN truncation)"
                if t["family"] == "line":
                    ih, initial = read_field(output / "initial.csv")
                    if ih[:4] != head[:4] or not np.array_equal(
                            initial[:, :4], final[:, :4]):
                        raise ValueError("line-source initial/final coordinate mismatch")
                    initial_rho = ana.physical_low(initial[:, 4:])[:, 0]
                    # Integral of the Gaussian + background on [0,1]^2,
                    # exponentially close to 1 + .0009 for width .01.
                    row["initial_mass_ratio"] = float(
                        np.sum(mass * initial_rho) / (1. + .0009))
                    row["initial_mass_relative_error"] = row["initial_mass_ratio"] - 1.
                if t["family"] == "boundary":
                    z = 2 * math.pi * t["time"]
                    reference = 1. + .2 * np.sin(2 * math.pi * coords[:, 0]) * np.sinc(z / math.pi)
                    row["error_kinetic_L2"] = rel_norm(physical[:, 0] - reference, reference, mass)["L2"]
                    row["reference"] = "analytic kinetic inflow; PN truncation included"
            data[label] = dict(t=t, coords=coords, physical=physical, mass=mass, row=row)
            rows.append(row)
        except Exception as error:
            failures.append((label, str(error)))
    # Rescaled exact line-source reference from existing independent reference generator.
    lines = [x for x in data.values() if x["t"]["family"] == "line"]
    if lines:
        target = campaign / "references" / "ganapol_GH.csv"
        if not target.is_file():
            cmd = [sys.executable, str(campaign / "reference.py"), "--medium", "scattering",
                   "--time", "1", "--sigma", ".03", "--floor", "1e-4",
                   "--rmax", "2.2", "--out", str(target)]
            subprocess.run(cmd, check=True)
        curve = np.loadtxt(target, delimiter=",", skiprows=1)
        for item in lines:
            xy, rho, mass = item["coords"], item["physical"][:, 0], item["mass"]
            r = 3 * np.linalg.norm(xy[:, :2] - .5, axis=1)
            expected = 9. * np.interp(r, curve[:, 0], curve[:, 1])
            item["row"]["error_kinetic_L2"] = rel_norm(rho - expected, expected, mass)["L2"]
            item["row"]["reference"] = "Ganapol smoothed GH; physical-to-unit-square scaling"
    # Temporal studies: compare each dt against finest *same-grid* solution.
    # Fast runs use identical nodal sample grids; check coordinates explicitly.
    for family in ("time_smooth", "time_crossing"):
        seq = sorted((item for item in data.values() if item["t"]["family"] == family),
                     key=lambda item: item["t"]["dt_level"])
        refs = [item for item in seq if item["t"]["reference"]]
        if not seq:
            continue
        if len(refs) != 1:
            failures.append((family, "missing unique temporal reference"))
            continue
        ref = refs[0]
        for item in seq:
            if item is ref:
                continue
            perm = ana.coordinate_permutation(item["coords"], ref["coords"])
            err = item["physical"][:, 0] - ref["physical"][perm, 0]
            item["row"]["error_temporal_self_L2"] = rel_norm(
                err, ref["physical"][perm, 0], item["mass"])["L2"]
            item["row"]["reference"] = ref["t"]["label"] + " (same grid, smaller dt)"
    # Fixed PN and degree: self differences avoid the kinetic truncation floor.
    for family in ("crossing", "boundary"):
        groups = {}
        for item in data.values():
            t = item["t"]
            if t["family"] == family:
                groups.setdefault((t["pn_order"], t["degree"]), []).append(item)
        for _, seq in groups.items():
            seq.sort(key=lambda item: item["t"]["elements"])
            for coarse, fine in zip(seq, seq[1:]):
                try:
                    interpolated = ana.interpolate_structured(
                        fine["coords"], fine["physical"][:, :1],
                        coarse["coords"], coarse["t"]["dim"])
                    err = coarse["physical"][:, 0] - interpolated[:, 0]
                    coarse["row"]["error_spatial_self_L2_to_next"] = rel_norm(
                        err, interpolated[:, 0], coarse["mass"])["L2"]
                except Exception as error:
                    failures.append((coarse["t"]["label"], "spatial interpolation: " + str(error)))
    # Paired dt at finest smooth grid: detect temporal contamination rather
    # than interpreting an SSPRK3 time plateau as spatial Qp convergence.
    for candidate in data.values():
        if candidate["t"]["family"] != "smoothdt2":
            continue
        t = candidate["t"]
        original = next((x for x in data.values()
                         if x["t"]["family"] == "smooth" and
                         x["t"]["pn_order"] == t["pn_order"] and
                         x["t"]["degree"] == t["degree"] and
                         x["t"]["elements"] == t["elements"]), None)
        if original is None:
            failures.append((t["label"], "missing matching full-dt smooth run"))
            continue
        if not np.allclose(candidate["coords"], original["coords"], atol=1e-13):
            failures.append((t["label"], "smooth time-check grids differ"))
            continue
        delta = original["physical"][:, 0] - candidate["physical"][:, 0]
        time_err = float(np.sqrt(np.mean(delta * delta)))
        original["row"]["smooth_dt_half_rms_difference"] = time_err
        original["row"]["smooth_time_warning"] = bool(
            time_err > .1 * original["row"]["error_exact_PN_L2"])
    # Experimental order of convergence only for comparisons with the SAME reference definition.
    for field, families in [
        ("error_exact_PN_L2", ["smooth"]),
        ("error_temporal_self_L2", ["time_smooth", "time_crossing"]),
        ("error_spatial_self_L2_to_next", ["crossing", "boundary"]),
        ("error_kinetic_L2", ["boundary", "crossing", "line"])]:
        for family in families:
            groups = {}
            for row in rows:
                if row["family"] != family or row.get(field) is None:
                    continue
                group = (row["pn"], row["p"], row["rk"])
                groups.setdefault(group, []).append(row)
            for seq in groups.values():
                seq.sort(key=(lambda r: -r["dt"]) if field == "error_temporal_self_L2" else
                         (lambda r: -r["h"]))
                for prev, current in zip(seq, seq[1:]):
                    x_prev = prev["dt"] if field == "error_temporal_self_L2" else prev["h"]
                    x_next = current["dt"] if field == "error_temporal_self_L2" else current["h"]
                    ep, en = prev[field], current[field]
                    if ep and en and ep > 0 and en > 0 and x_prev > x_next:
                        current["eoc_" + field] = math.log(ep / en) / math.log(x_prev / x_next)
    # Write a machine-readable publication table and a cautious report.
    columns = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with (campaign / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    atomic(campaign / "results.json", dict(rows=rows, failures=failures,
            completed=len(rows), expected=len(tasks)))
    headings = ["# NX Element-WENO convergence study", "",
                f"Completed {len(rows)} / {len(tasks)} tasks; failures: {len(failures)}.",
                "", "Source: frozen manifest " + (campaign / "manifest.json").as_posix(),
                "", "## Interpretation",
                "- Exact-PN smooth error uses the standalone L2-projected periodic test.",
                "- Kinetic Crossing / boundary / line-source errors include PN truncation.",
                "- Temporal rates use the finest fixed-grid time step as numerical reference.",
                "- Spatial self differences compare each mesh to the next finer mesh, not the exact solution.",
                "- SSPRK(4,3) transport is third order; Strang with noncommuting reactions is generally second order.",
                "- The estimated high-order CFL is empirical, not a general stability theorem.",
                "- Failed no-limiter ablations are diagnostic, not silently omitted.",
                "", "## Files", "- results.csv; results.json; failure details in status/*.json",
                ""]
    if failures:
        headings += ["## Failed or missing jobs", ""]
        headings += [f"- {a}: {b}" for a, b in failures]
    (campaign / "REPORT.md").write_text("\n".join(headings) + "\n")
    subprocess.run([sys.executable, str(campaign / "plot.py"), "--campaign", str(campaign)], check=True)
    print("completed", len(rows), "of", len(tasks), "; failures", len(failures))
    return 0 if not failures else 2


def submit(ns) -> int:
    campaign = ns.campaign.resolve(strict=True)
    tasks = check_manifest(campaign)
    lock = campaign / "submission.json"
    if lock.exists():
        raise FileExistsError("campaign already submitted/recorded; inspect " + str(lock))
    # Slot classes (v2): tasks with slot_cores 32 go to packs of two 32-core slots (one per socket), all others
    # to packs of four 16-core slots. v1 campaigns have no slot_cores and keep the 16-core packs.
    groups = {}
    for t in tasks:
        groups.setdefault(int(t.get("slot_cores", 16)), []).append(t)
    files = []
    for slot, members in sorted(groups.items()):
        batch_size = max(1, ns.tasks_per_job if slot == 16 else max(1, ns.tasks_per_job // 2))
        for batch in range(0, len(members), batch_size):
            selected = members[batch: batch + batch_size]
            path = campaign / "packs" / f"tasks_s{slot}_{batch//batch_size:03}.tsv"
            lines = []
            for t in selected:
                command = [sys.executable, str(campaign / "study.py"), "run",
                           "--campaign", str(campaign), "--task", t["label"],
                           "--threads", str(slot), "--timeout", str(ns.task_timeout)]
                lines.append(t["label"] + "\t" + shlex.join(command))
            if ns.execute:
                path.write_text("\n".join(lines) + "\n")
            files.append((path, lines, slot))
    sbatch = shutil.which("sbatch") or "sbatch"
    batch_script = campaign / "nx/batches/element_weno_pack.sbatch"
    commands = []
    jobids = []
    if ns.execute and not shutil.which("sbatch"):
        raise RuntimeError("sbatch missing; run submit --execute on the NX login node")
    if ns.execute:
        atomic(lock, {"status": "submitting", "jobids": [],
                      "commands": [], "warning": "partial submissions may be live"})
    for index, (path, _, slot) in enumerate(files):
        # Conservative dependency chain: no more than max_concurrent_packs
        # whole-node pack jobs may be eligible simultaneously.
        dependencies = []
        if index >= ns.max_concurrent_packs:
            predecessor = (jobids[index - ns.max_concurrent_packs]
                           if ns.execute else "PACK_JOB_" + str(index - ns.max_concurrent_packs))
            dependencies = ["--dependency=afterany:" + predecessor]
        overrides = ([f"--partition={ns.partition}"] if ns.partition else []) + ([f"--time={ns.time}"] if ns.time else [])
        cmd = [sbatch, "--parsable", *dependencies, *overrides, "--chdir=" + str(campaign),
               "--export=ALL,TASKFILE=" + str(path) + ",REPO=" + str(campaign) + ",SLOT_CORES=" + str(slot)
               + ",OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1",
               "--output=" + str(campaign / "packs" / (path.stem + "-%j.log")),
               str(batch_script)]
        commands.append(cmd)
        if ns.execute:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            jobids.append(proc.stdout.strip().split(";")[0])
            atomic(lock, {"status": "submitting", "jobids": jobids,
                          "commands": commands})
    if jobids:
        wrap = shlex.join([sys.executable, str(campaign / "study.py"),
                           "collect", "--campaign", str(campaign)])
        cmd = [sbatch, "--parsable", "--partition=med", "--constraint=nx",
               "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=32G",
               "--time=02:00:00", "--dependency=afterany:" + ":".join(jobids),
               "--output=" + str(campaign / "packs/collect-%j.log"),
               "--chdir=" + str(campaign), "--wrap=" + wrap]
        commands.append(cmd)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        jobids.append(proc.stdout.strip().split(";")[0])
        atomic(lock, {"status": "submitted", "jobids": jobids, "commands": commands,
                      "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    for cmd in commands:
        print(shlex.join(cmd))
    if not ns.execute:
        print(f"DRY RUN: {len(files)} pack jobs; no files written or jobs submitted")
    return 0


def plan(ns) -> int:
    """Resource plan of a profile (no files written, no submission)."""
    tasks = make_tasks(ns.profile)
    if ns.profile not in HETERO_PROFILES:
        for t in tasks:
            t.update(cost_model(t)) if t["binary"] == "fast" else None
    total = sum(t.get("estimate_core_hours", 0) for t in tasks)
    print(f"{ns.profile}: {len(tasks)} tasks, estimated {total:.1f} core-hours (calibration factor ~2 uncertain)")
    print(f"{'label':42s} {'nodes':>10s} {'mom':>4s} {'steps':>6s} {'mem GB':>7s} {'core-h':>7s} {'wall-h':>7s} {'slot':>4s}")
    warn = 0
    for t in tasks:
        if "estimate_core_hours" not in t:
            continue
        flag = ""
        if t["estimate_wall_hours"] > 8:
            flag, warn = " > med 8 h: use long", warn + 1
        print(f"{t['label']:42s} {t['estimate_nodes']:10d} {t['estimate_moments']:4d} {t['estimate_steps']:6d} "
              f"{t['estimate_memory_gb']:7.2f} {t['estimate_core_hours']:7.2f} {t['estimate_wall_hours']:7.2f} "
              f"{t['slot_cores']:4d}{flag}")
    packs16 = math.ceil(sum(t.get("slot_cores", 16) == 16 for t in tasks) / 4)
    packs32 = math.ceil(sum(t.get("slot_cores", 16) == 32 for t in tasks) / 2)
    print(f"packs: {packs16} x (4 x 16 cores) + {packs32} x (2 x 32 cores); tasks over 8 h wall: {warn}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--profile", choices=["smoke", "paper", *HETERO_PROFILES], default="paper")
    p.add_argument("--repo", type=Path, default=ROOT)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--fast-bin", type=Path, default=ROOT / "tmp/build/m1pn_fast")
    p.add_argument("--standalone-bin", type=Path, default=ROOT / "tmp/build/m1pn_element_weno")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="+", help="freeze only these task labels (supplementary/recovery campaign)")
    p.add_argument("--slot-cores", type=int, choices=[16, 32], help="override the slot size of all frozen tasks")
    p.set_defaults(action=prepare)
    p = sub.add_parser("plan", help="resource plan of a profile (dry, no files)")
    p.add_argument("--profile", choices=["smoke", "paper", *HETERO_PROFILES], default="hetero-paper")
    p.set_defaults(action=plan)
    p = sub.add_parser("run")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--timeout", type=int, default=25200)
    p.set_defaults(action=run)
    p = sub.add_parser("collect")
    p.add_argument("--campaign", type=Path, required=True)
    p.set_defaults(action=collect)
    p = sub.add_parser("submit")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--tasks-per-job", type=int, default=4)
    p.add_argument("--max-concurrent-packs", type=int, default=1)
    p.add_argument("--task-timeout", type=int, default=25200)
    p.add_argument("--partition", help="override the pack partition (e.g. short for smoke campaigns)")
    p.add_argument("--time", help="override the pack time limit (e.g. 00:30:00)")
    p.add_argument("--execute", action="store_true", help="Actually call sbatch")
    p.set_defaults(action=submit)
    ns = parser.parse_args()
    if hasattr(ns, "threads") and ns.threads < 1:
        parser.error("--threads must be positive")
    if hasattr(ns, "tasks_per_job") and ns.tasks_per_job < 1:
        parser.error("--tasks-per-job must be positive")
    if hasattr(ns, "max_concurrent_packs") and ns.max_concurrent_packs < 1:
        parser.error("--max-concurrent-packs must be positive")
    return ns.action(ns)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(1)
