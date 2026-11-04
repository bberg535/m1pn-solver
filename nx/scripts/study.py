#!/usr/bin/env python3
"""Reproducible preparation and execution for the NX PN mass-correction study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
NX = HERE.parent
ROOT = NX.parent
PLAN_PATH = NX / "study_plan.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_plan() -> dict:
    return json.loads(PLAN_PATH.read_text())


def validate_plan(plan: dict) -> None:
    method, execution = plan["method"], plan["execution"]
    for key, required in (("model", "r2"), ("mass_mode", "lumped"),
                          ("integrator", "ssprk3"),
                          ("stabilization", "m1pn_mass_correction")):
        if method[key] != required:
            raise ValueError(f"This PN mass-correction study requires method.{key}={required}")
    if type(method["mass_corrections"]) is not int or method["mass_corrections"] != 0:
        raise ValueError("method.mass_corrections must be zero for lumped PN mass correction")
    if float(method.get("pn_viscosity_scale", -1)) != 1.0:
        raise ValueError("method.pn_viscosity_scale must be 1 for the standard PN mass correction")
    if float(method.get("biharmonic_gamma", -1)) != 0.0 or float(method.get("kv_gamma", -1)) != 0.0:
        raise ValueError("PN mass-correction study requires zero biharmonic and KV coefficients")
    for key in ("kv_gamma", "biharmonic_gamma"):
        value = float(method.get(key, .01))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"method.{key} must be finite and nonnegative")
    if not 0 < float(execution["cfl_safety"]) <= 1:
        raise ValueError("execution.cfl_safety must lie in (0,1]")
    for key in ("memory_mib", "max_steps", "timeout_seconds", "max_concurrent_tasks"):
        if type(execution[key]) is not int or execution[key] < 1:
            raise ValueError(f"execution.{key} must be a positive integer")
    if not 0 < float(execution["temporal_error_fraction"]) <= 1:
        raise ValueError("execution.temporal_error_fraction must lie in (0,1]")


def family_specs(plan: dict, profile: str, selected: list[str] | None) -> list[tuple[str, dict]]:
    specs = plan["profiles"][profile]
    names = selected or list(specs)
    unknown = sorted(set(names) - set(specs))
    if unknown:
        raise ValueError(f"Unknown case family: {', '.join(unknown)}")
    return [(name, specs[name]) for name in specs if name in names]


def freeze_common_dts(families: list[tuple[str, dict]], probes: list[dict], safety: float) -> dict[str, float]:
    result = {}
    for family, spec in families:
        family_probes = [p for p in probes if p["family"] == family]
        if not family_probes:
            raise ValueError(f"No stability probes for family {family}")
        limit = min(min(p["low_cfl_limit"], p["pn_step_limit"]) for p in family_probes)
        steps = math.ceil(spec["time"] / (safety * limit))
        result[family] = spec["time"] / max(1, steps)
    return result


def build_args(spec: dict, order: int, grid: int, output: str | Path, threads: int,
               max_steps: int, method: dict, *, t: float, dt: float | None = None,
               memory_mib: int = 49152) -> list[str]:
    args = ["--case", spec["case"], "--inflow", spec["inflow"],
            "--dimension", str(spec["dimension"]), "--mesh-type", "hex",
            "--elements", str(grid), "--pn-order", str(order), "--time", f"{t:.17g}",
            "--cfl", "1", "--model", method["model"], "--mass-mode", method["mass_mode"],
            "--stabilization", method["stabilization"],
            "--kv-gamma", str(method["kv_gamma"]), "--biharmonic-gamma", str(method["biharmonic_gamma"]),
            "--pn-viscosity-scale", str(method["pn_viscosity_scale"]), "--time-integrator", method["integrator"],
            "--max-steps", str(max_steps), "--memory-mib", str(memory_mib),
            "--pulse-width", f"{spec['width']:.17g}", "--background", f"{spec['background']:.17g}",
            "--threads", str(threads), "--output", str(output)]
    # The core parser requires [1,32] for the AFC iteration option. In lumped
    # mode the field is unused and config metadata correctly records zero.
    if method["mass_corrections"] > 0:
        args += ["--mass-corrections", str(method["mass_corrections"])]
    if dt is not None:
        args += ["--dt", f"{dt:.17g}"]
    return args


def run_logged(command: list[str], log: Path, timeout: int | None) -> tuple[int, float]:
    start = time.monotonic()
    with log.open("w") as f:
        try:
            p = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout, check=False)
            return p.returncode, time.monotonic() - start
        except subprocess.TimeoutExpired:
            f.write("\nORCHESTRATOR_TIMEOUT\n")
            return 124, time.monotonic() - start


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def source_manifest(exe: Path) -> dict:
    candidates = list((NX / "src").glob("*"))
    candidates += [p for p in (NX / "scripts").iterdir()
                  if p.is_file() and p.suffix in (".py", ".sh", ".sbatch")]
    candidates += [p for p in (NX / "tests").iterdir()
                  if p.is_file() and p.suffix in (".py", ".sh", ".sbatch")]
    candidates += list((NX / "batches").glob("*.sbatch"))
    candidates += list((ROOT / "src").glob("*"))
    candidates += [PLAN_PATH, ROOT / "CMakeLists.txt", NX / "CMakeLists.txt",
                  ROOT / "scripts" / "verify_run.py"]
    files = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(candidates))
             if p.is_file()}
    build_info = {}
    build_dir = exe.parent
    build_files = [build_dir / "CMakeCache.txt"]
    build_files += list((build_dir / "CMakeFiles").glob("*/link.txt"))
    build_files += list((build_dir / "CMakeFiles").glob("*/flags.make"))
    for p in build_files:
        if p.is_file():
            try:
                name = str(p.relative_to(ROOT))
            except ValueError:
                name = str(p)
            build_info[name] = sha256(p)
    return {"git_head": git_head(), "source_sha256": files,
            "build_configuration_sha256": build_info}


def prepare(ns: argparse.Namespace) -> int:
    plan = load_plan()
    validate_plan(plan)
    exe = ns.exe.resolve(strict=True)
    if not exe.is_file() or not os.access(exe, os.X_OK):
        raise ValueError(f"Executable is not executable: {exe}")
    campaign = ns.output.resolve()
    if campaign.exists():
        raise FileExistsError(f"Output already exists: {campaign}")
    if ROOT not in campaign.parents:
        raise ValueError(f"Campaign output must be inside {ROOT}")
    campaign.mkdir(parents=True)
    (campaign / "logs").mkdir()
    binary = campaign / "m1pn_mc_study"
    shutil.copy2(exe, binary)
    verifier_source = ROOT / "scripts" / "verify_run.py"
    if not verifier_source.is_file():
        raise FileNotFoundError(f"Run verifier is missing: {verifier_source}")
    verifier = campaign / "verify_run.py"
    shutil.copy2(verifier_source, verifier)
    families = family_specs(plan, ns.profile, ns.cases)
    run_root = campaign / "probes"
    records = []
    probe_failures = []
    all_specs = []
    for family, spec in families:
        for n in spec["grids"]:
            for order in spec["orders"]:
                all_specs.append((family, spec, n, order))
    for index, (family, spec, n, order) in enumerate(all_specs):
        label = f"{family}_N{n}_P{order}"
        out = run_root / label
        log = campaign / "logs" / f"probe_{label}.log"
        command = [str(binary), *build_args(spec, order, n, out, ns.threads,
                                             plan["execution"]["max_steps"], plan["method"], t=0,
                                             memory_mib=plan["execution"]["memory_mib"])]
        rc, wall = run_logged(command, log, plan["execution"]["timeout_seconds"])
        record = {"id": index, "family": family, "grid": n, "order": order,
                  "command": command, "output": str(out.relative_to(campaign)),
                  "log": str(log.relative_to(campaign)), "exit_code": rc,
                  "wall_seconds": wall, "status": "passed" if rc == 0 else "failed"}
        try:
            config = json.loads((out / "config.json").read_text())
            record["config"] = config
            method = plan["method"]
            if (config.get("mass_mode") != method["mass_mode"] or
                    config.get("mass_corrections") != method["mass_corrections"] or
                    config.get("stabilization") != method["stabilization"] or
                    config.get("pn_mass_correction_applications") != 1 or
                    config.get("uses_pn_low_order_predictor") is not True or
                    config.get("mass_factorized") is not False or
                    abs(float(config.get("gamma", -1)) - method["biharmonic_gamma"]) > 1e-14 or
                    abs(float(config.get("kv_gamma", -1)) - method["kv_gamma"]) > 1e-14 or
                    abs(float(config.get("pn_viscosity_scale", -1)) - method["pn_viscosity_scale"]) > 1e-14):
                raise ValueError("probe metadata does not match requested PN mass-correction method")
            if "ghost_edges" in config and config["ghost_edges"] != 0:
                raise ValueError("PN mass-correction probe unexpectedly uses ghost edges")
            record["low_cfl_limit"] = float(config["low_cfl_limit"])
            record["pn_step_limit"] = float(config["pn_step_limit"])
            if rc != 0:
                record["status"] = "failed"
            elif not all(math.isfinite(record[k]) and record[k] > 0
                         for k in ("low_cfl_limit", "pn_step_limit")):
                raise ValueError("probe step limits must be finite and positive")
        except Exception as e:
            record["probe_error"] = str(e)
            record["status"] = "failed"
        records.append(record)
        if record["status"] != "passed":
            probe_failures.append(record)
            atomic_json(campaign / "probe_failure.json", {"failed_probe": record,
                                                            "completed_probes": records})
            break
    if probe_failures:
        atomic_json(campaign / "probes.json", {"status": "failed", "probes": records})
        print(f"Probe failed; logs preserved in {campaign / 'logs'}", file=sys.stderr)
        return 2

    dt_by_family = freeze_common_dts(families, records, plan["execution"]["cfl_safety"])

    tasks = []
    for family, spec, n, order in all_specs:
        steps = round(spec["time"] / dt_by_family[family])
        tasks.append({"id": len(tasks), "label": f"{family}_N{n}_P{order}",
                      "family": family, "grid": n, "order": order,
                      "dimension": spec["dimension"],
                      "case_label": family, "grid_n": n, "angular_order": order,
                      "time": spec["time"], "dt": dt_by_family[family],
                      "steps": steps, "temporal_ref": False,
                      "output_dir": f"runs/{family}_N{n}_P{order}",
                      "inflow": spec["inflow"], "case": spec["case"],
                      "width": spec["width"], "background": spec["background"],
                      "kind": "spatial"})
    for family, spec in families:
        n, order = max(spec["grids"]), max(spec["orders"])
        steps = round(spec["time"] / (dt_by_family[family] / 2))
        tasks.append({"id": len(tasks), "label": f"{family}_N{n}_P{order}_dt2",
                      "family": family, "grid": n, "order": order,
                      "dimension": spec["dimension"],
                      "case_label": family, "grid_n": n, "angular_order": order,
                      "time": spec["time"], "dt": dt_by_family[family] / 2,
                      "steps": steps, "temporal_ref": True,
                      "output_dir": f"runs/{family}_N{n}_P{order}_dt2",
                      "inflow": spec["inflow"], "case": spec["case"],
                      "width": spec["width"], "background": spec["background"],
                      "kind": "temporal"})
    atomic_json(campaign / "probes.json", {"status": "passed", "probes": records})
    for task in tasks:
        task["method"] = dict(plan["method"])
        task["memory_mib"] = plan["execution"]["memory_mib"]
        task["solver_args"] = build_args(task, task["order"], task["grid"],
                                         task["output_dir"], ns.threads,
                                         plan["execution"]["max_steps"], plan["method"],
                                         t=task["time"], dt=task["dt"], memory_mib=task["memory_mib"])
    excessive = [t for t in tasks if t["steps"] > plan["execution"]["max_steps"]]
    if excessive:
        raise ValueError("estimated steps exceed max_steps for: " +
                         ", ".join(t["label"] for t in excessive))
    atomic_json(campaign / "tasks.json", tasks)
    tasks_hash = sha256(campaign / "tasks.json")
    atomic_json(campaign / "manifest.json", {
        "version": 1, "profile": ns.profile, "selected_families": [f for f, _ in families],
        "plan": plan, "plan_sha256": sha256(PLAN_PATH), "binary": binary.name,
        "binary_sha256": sha256(binary), "verifier": verifier.name,
        "verifier_sha256": sha256(verifier), "threads": ns.threads,
        "probes": "probes.json", "tasks": "tasks.json", "tasks_sha256": tasks_hash,
        "dt_by_family": dt_by_family,
        "provenance": source_manifest(exe), "warnings": [
            "Step counts are estimated as ceil(final_time / dt); nonlinear step rejection may increase them.",
            "A passing stored-field verifier is not a proof of the full PN cone or entropy inequality."],
        "estimated_total_steps": sum(t["steps"] for t in tasks)})
    if any(t["steps"] > 100000 for t in tasks):
        print(f"Warning: campaign includes a task above 100,000 estimated steps ({campaign})",
              file=sys.stderr)
    print(f"Prepared {len(tasks)} tasks in {campaign}")
    return 0


def read_task(campaign: Path, index: int) -> dict:
    tasks = json.loads((campaign / "tasks.json").read_text())
    if index < 0 or index >= len(tasks) or tasks[index].get("id") != index:
        raise ValueError(f"Task index must be between 0 and {len(tasks)-1}")
    return tasks[index]


def run_task(ns: argparse.Namespace) -> int:
    campaign = ns.campaign.resolve(strict=True)
    if ROOT not in campaign.parents:
        raise ValueError(f"Campaign must be inside {ROOT}")
    manifest = json.loads((campaign / "manifest.json").read_text())
    binary = campaign / manifest["binary"]
    if sha256(binary) != manifest["binary_sha256"]:
        raise ValueError("Frozen executable SHA256 mismatch")
    verifier = campaign / manifest["verifier"]
    if sha256(verifier) != manifest["verifier_sha256"]:
        raise ValueError("Frozen verifier SHA256 mismatch")
    if sha256(campaign / manifest["tasks"]) != manifest["tasks_sha256"]:
        raise ValueError("Frozen task manifest SHA256 mismatch")
    task = read_task(campaign, ns.task)
    runs = (campaign / task["output_dir"]).resolve()
    if campaign not in runs.parents:
        raise ValueError("Task output path escapes campaign directory")
    if runs.exists():
        raise FileExistsError(f"Run output already exists: {runs}")
    runs.parent.mkdir(parents=True, exist_ok=True)
    log_dir = campaign / "logs"
    log_dir.mkdir(exist_ok=True)
    log = log_dir / f"task_{task['id']}_{task['label']}.log"
    run_args = list(task["solver_args"])
    out_idx = run_args.index("--output") + 1
    run_args[out_idx] = str(runs)
    command = [str(binary), *run_args]
    if ns.launcher == "srun":
        command = ["srun", "--mpi=none", "--ntasks=1", f"--cpus-per-task={manifest['threads']}",
                   "--cpu-bind=cores", "--distribution=block:block", *command]
    status_path = campaign / "status" / f"{task['id']}.json"
    common = {"id": task["id"], "label": task["label"], "task": task,
              "output_dir": task["output_dir"], "command": command, "log": str(log.relative_to(campaign)),
              "binary_sha256": manifest["binary_sha256"],
              "allocation": {k: os.environ[k] for k in (
                  "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                  "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_JOB_PARTITION",
                  "SLURM_JOB_CONSTRAINTS") if k in os.environ}}
    atomic_json(status_path, {**common, "status": "running", "started_at": time.time()})
    timeout = ns.timeout or manifest["plan"]["execution"]["timeout_seconds"]
    rc, wall = run_logged(command, log, timeout)
    diagnostics = {}
    state = "failed"
    try:
        cfg = json.loads((runs / "config.json").read_text())
        summary = json.loads((runs / "summary.json").read_text())
        diagnostics = {"mass_mode": cfg.get("mass_mode"), "mass_corrections": cfg.get("mass_corrections"),
                       "mass_factorized": cfg.get("mass_factorized"),
                       "pn_mass_correction_applications": cfg.get("pn_mass_correction_applications"),
                       "integrator": cfg.get("integrator"), "low_cfl_limit": cfg.get("low_cfl_limit"),
                       "summary": summary}
        if rc == 0:
            tol = max(1e-13, 1e-12 * task["time"])
            method = manifest["plan"]["method"]
            if (cfg.get("mass_mode") != method["mass_mode"] or cfg.get("mass_corrections") != method["mass_corrections"] or
                    cfg.get("stabilization") != method["stabilization"] or
                    abs(float(cfg.get("kv_gamma", -1)) - method["kv_gamma"]) > 1e-14 or
                    abs(float(cfg.get("gamma", -1)) - method["biharmonic_gamma"]) > 1e-14 or
                    abs(float(cfg.get("pn_viscosity_scale", -1)) - method["pn_viscosity_scale"]) > 1e-14 or
                    cfg.get("pn_mass_correction_applications") != 1 or
                    cfg.get("uses_pn_low_order_predictor") is not True or
                    cfg.get("memory_budget_bytes") != manifest["plan"]["execution"]["memory_mib"] * 1024 * 1024 or
                    cfg.get("mass_factorized") is not False or
                    summary.get("mass_solve_calls") != 0 or
                    abs(float(summary.get("biharmonic_seconds", -1))) > 1e-15 or
                    not math.isfinite(float(summary.get("pn_predictor_seconds", -1))) or
                    float(summary.get("pn_predictor_seconds", -1)) < 0 or
                    not math.isfinite(float(summary.get("pn_mass_correction_seconds", -1))) or
                    float(summary.get("pn_mass_correction_seconds", -1)) < 0 or
                    ("ghost_edges" in cfg and cfg.get("ghost_edges") != 0) or
                    summary.get("status") != "completed" or
                    abs(float(summary.get("time", -1)) - task["time"]) > tol or
                    cfg.get("rk") != 3 or cfg.get("pn_order") != task["order"] or
                    cfg.get("elements_per_axis") != task["grid"] or
                    cfg.get("dimension") != task["dimension"] or
                    cfg.get("case") != task["case"] or cfg.get("inflow") != task["inflow"] or
                    abs(float(cfg.get("dt_requested", -1)) - task["dt"]) > max(1e-15, task["dt"]*1e-12) or
                    abs(float(cfg.get("dt_selected", -1)) - task["dt"]) > max(1e-15, task["dt"]*1e-12) or
                    abs(float(cfg.get("final_time", -1)) - task["time"]) > tol or
                    summary.get("steps") != task["steps"] or summary.get("rejected_attempts", 0) != 0 or
                    ("background" in cfg and abs(float(cfg["background"]) - task["background"]) > 1e-14) or
                    (task["case"] != "lattice-paul" and
                     abs(float(cfg.get("pulse_width", -1)) - task["width"]) > 1e-14) or
                    not cfg.get("low_cfl_limit")):
                raise ValueError("run metadata violates requested controls or final-time target")
            vr = subprocess.run([sys.executable, str(verifier), str(runs)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                timeout=min(timeout, 600), check=False)
            diagnostics["verification_output"] = vr.stdout
            if vr.returncode:
                state = "validation_failed"
            else:
                state = "completed"
        else:
            diagnostics["reason"] = f"solver exited with code {rc}"
    except Exception as e:
        diagnostics["validation_error"] = str(e)
        state = "validation_failed" if rc == 0 else "failed"
    atomic_json(status_path, {**common, "status": state, "exit_code": rc,
                              "wall_seconds": wall, "finished_at": time.time(),
            "validated": state == "completed", "error": diagnostics.get("reason") or diagnostics.get("validation_error"),
            "diagnostics": diagnostics})
    print(f"Task {task['id']} {state} ({wall:.1f}s)")
    return 0 if state == "completed" else 1


def collect(ns: argparse.Namespace) -> int:
    campaign = ns.campaign.resolve(strict=True)
    script = HERE / "analysis.py"
    if not script.is_file():
        raise FileNotFoundError("Collector is not installed yet: nx/scripts/analysis.py")
    return subprocess.run([sys.executable, str(script), "--campaign", str(campaign)],
                          check=False).returncode


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--exe", type=Path, required=True)
    prep.add_argument("--profile", choices=("production", "smoke"), required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--threads", type=int, required=True)
    prep.add_argument("--cases", nargs="+", choices=("crossing", "lattice", "three_d"))
    prep.set_defaults(func=prepare)
    run = sub.add_parser("run")
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--task", type=int, required=True)
    run.add_argument("--launcher", choices=("srun", "direct"), default="direct")
    run.add_argument("--timeout", type=int)
    run.set_defaults(func=run_task)
    col = sub.add_parser("collect")
    col.add_argument("--campaign", type=Path, required=True)
    col.set_defaults(func=collect)
    return p


def main(argv: list[str] | None = None) -> int:
    ns = parser().parse_args(argv)
    try:
        if hasattr(ns, "threads") and ns.threads < 1:
            raise ValueError("threads must be positive")
        if hasattr(ns, "timeout") and ns.timeout is not None and ns.timeout <= 0:
            raise ValueError("timeout must be positive")
        return ns.func(ns)
    except Exception as e:
        print(f"study.py: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
