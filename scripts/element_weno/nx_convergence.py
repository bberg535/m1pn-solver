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
         reference=False, linear=1e-3, tag="", cfl=.2):
    label = "_".join(str(x) for x in
                     (family, f"P{pn}", f"p{p}", f"E{n}",
                      ("dt" + str(dt_level)) if dt_level is not None else
                      (tag or "std")))
    return dict(label=label, family=family, kind=kind, binary=binary,
                case=case, inflow=inflow, dim=dim, elements=n, degree=p,
                pn_order=pn, time=end, sigma_s=sigma_s, sigma_a=sigma_a,
                filter_strength=filter_strength, weno=weno, limiter=limiter,
                rk=rk, periodic=periodic, dt_level=dt_level, reference=reference,
                weno_linear=linear, cfl=cfl)


def make_tasks(profile: str) -> list[dict]:
    smoke = profile == "smoke"
    if profile not in ("smoke", "paper"):
        raise ValueError("profile must be smoke or paper")
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
        # Same 129x129 sample-grid size as Q2/E64 element WENO, not same FE degree.
        for coupling in ("m1pn", "pn-only"):
            row = task("baseline", "fast", "crossing-study",
                       "crossing-study-transport", 2, 128, 1, 7, .32,
                       kind="baseline", tag=coupling)
            row["stabilization"] = "m1pn_mass_correction"
            row["coupling"] = coupling
            row["rk"] = "ssprk3"
            out.append(row)
    seen = set()
    for number, t in enumerate(out):
        if t["label"] in seen:
            raise ValueError("duplicate task label: " + t["label"])
        seen.add(t["label"])
        t["id"] = number
    return out


def flags(t: dict, output: Path, threads: int, *, end=None, dt=None, cfl=None):
    if end is None:
        end = t["time"]
    if cfl is None:
        cfl = t["cfl"]
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
    if t.get("stabilization") == "element-weno":
        args += ["--weno", "on" if t["weno"] else "off",
                 "--element-limit", "on" if t["limiter"] else "off",
                 "--weno-linear", str(t["weno_linear"])]
    else:
        args += ["--coupling", t.get("coupling", "m1pn"),
                 "--mass-mode", "lumped", "--pn-viscosity-scale", "1",
                 "--kv-gamma", "0", "--biharmonic-gamma", "0"]
    return args


def prepare(ns) -> int:
    tasks = make_tasks(ns.profile)
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
           {"format": FROZEN, "profile": ns.profile,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repository": "bberg535/m1pn-solver", "git_head": commit(),
            "root": str(repo), "task_count": len(tasks),
            "checksums": hashes, "reactions": "Strang exact half steps",
            "interpretation": "kinetic reference includes PN truncation; temporal self reference only"})
    print("frozen campaign:", campaign)
    return 0


def check_manifest(campaign: Path) -> list[dict]:
    meta = load(campaign / "manifest.json")
    if meta.get("format") != FROZEN:
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
        else:
            config = load(output / "config.json")
            if summary.get("status") != "completed" or config.get("case") != t["case"]:
                raise ValueError("solver did not complete requested case")
            if config.get("time_integrator") != t["rk"] and t.get("stabilization") != "m1pn_mass_correction":
                raise ValueError("time-integrator mismatch")
            if int(config.get("degree", -1)) != t["degree"] and t.get("stabilization") != "m1pn_mass_correction":
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
                if abs(float(config.get("sigma_s", -1)) - t["sigma_s"]) > 1e-12 or \
                   abs(float(config.get("sigma_a", -1)) - t["sigma_a"]) > 1e-12:
                    raise ValueError("reaction coefficients mismatch")
                expected_filter = "constant" if t["filter_strength"] else "off"
                if config.get("filter_mode") != expected_filter:
                    raise ValueError("filter mode mismatch")
            if t["kind"] == "temporal" and abs(float(config["dt_selected"]) - dt) > 1e-12 * max(1., dt):
                raise ValueError("requested temporal dt changed")
        outcome["summary_sha256"] = digest(output / "summary.json")
        outcome["final_sha256"] = digest(output / "final.csv")
        outcome["file_sha256"] = {
            name: digest(output / name)
            for name in ("summary.json", "final.csv", "config.json",
                         "initial.csv", "nodal_mass.csv")
            if (output / name).is_file()
        }
        required = ("summary.json", "final.csv") if t["binary"] == "standalone" else (
            "summary.json", "final.csv", "config.json", "initial.csv", "nodal_mass.csv")
        if not all(name in outcome["file_sha256"] for name in required):
            raise ValueError("required saved-field files are missing")
        outcome["status"] = "completed"
    except Exception as error:
        outcome["status"] = "failed"
        outcome["error"] = str(error)
    atomic(status, outcome)
    print(t["label"], outcome["status"], "in", round(elapsed, 2), "s")
    return 0 if outcome["status"] == "completed" else 2


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


def collect(ns) -> int:
    import numpy as np
    campaign = ns.campaign.resolve(strict=True)
    tasks = check_manifest(campaign)
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
    slot = 16
    batch_size = max(1, ns.tasks_per_job)
    files = []
    for batch in range(0, len(tasks), batch_size):
        selected = tasks[batch: batch + batch_size]
        path = campaign / "packs" / f"tasks_{batch//batch_size:03}.tsv"
        lines = []
        for t in selected:
            command = [sys.executable, str(campaign / "study.py"), "run",
                       "--campaign", str(campaign), "--task", t["label"],
                       "--threads", str(slot), "--timeout", str(ns.task_timeout)]
            lines.append(t["label"] + "\t" + shlex.join(command))
        if ns.execute:
            path.write_text("\n".join(lines) + "\n")
        files.append((path, lines))
    sbatch = shutil.which("sbatch") or "sbatch"
    batch_script = campaign / "nx/batches/element_weno_pack.sbatch"
    commands = []
    jobids = []
    if ns.execute and not shutil.which("sbatch"):
        raise RuntimeError("sbatch missing; run submit --execute on the NX login node")
    if ns.execute:
        atomic(lock, {"status": "submitting", "jobids": [],
                      "commands": [], "warning": "partial submissions may be live"})
    for path, _ in files:
        cmd = [sbatch, "--parsable", "--chdir=" + str(campaign),
               "--export=ALL,TASKFILE=" + str(path) + ",REPO=" + str(campaign) + ",OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--profile", choices=["smoke", "paper"], default="paper")
    p.add_argument("--repo", type=Path, default=ROOT)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--fast-bin", type=Path, default=ROOT / "tmp/build/m1pn_fast")
    p.add_argument("--standalone-bin", type=Path, default=ROOT / "tmp/build/m1pn_element_weno")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(action=prepare)
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
    p.add_argument("--task-timeout", type=int, default=25200)
    p.add_argument("--execute", action="store_true", help="Actually call sbatch")
    p.set_defaults(action=submit)
    ns = parser.parse_args()
    if hasattr(ns, "threads") and ns.threads < 1:
        parser.error("--threads must be positive")
    if hasattr(ns, "tasks_per_job") and ns.tasks_per_job < 1:
        parser.error("--tasks-per-job must be positive")
    return ns.action(ns)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(1)
