#!/usr/bin/env python3
"""Move four pending timeout-risk tasks to a dependent long array without duplicate runs."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from zoneinfo import ZoneInfo

from study import atomic_json, git_head, sha256

TASK_IDS = (17, 45, 46, 47)
TIMEOUT_SECONDS = 23 * 3600


def frozen_inputs(campaign: Path, require_unstarted: bool = True) -> dict:
    manifest = json.loads((campaign / "manifest.json").read_text())
    files = {manifest["binary"]: manifest["binary_sha256"],
             manifest["verifier"]: manifest["verifier_sha256"],
             manifest["tasks"]: manifest["tasks_sha256"]}
    for name, expected in files.items():
        path = (campaign / name).resolve()
        path.relative_to(campaign)
        if sha256(path) != expected:
            raise ValueError("Frozen input hash mismatch: " + name)
    tasks = json.loads((campaign / manifest["tasks"]).read_text())
    if (manifest["threads"] != 16 or len(tasks) != 48 or
            manifest["plan"]["method"]["stabilization"] != "m1pn_mass_correction"):
        raise ValueError("Expected the existing 48-task, 16-thread campaign.")
    for index in TASK_IDS:
        task = tasks[index]
        if task["id"] != index:
            raise ValueError("Task ID mismatch.")
        output = (campaign / task["output_dir"]).resolve()
        output.relative_to(campaign)
        if require_unstarted and (output.exists() or (campaign / "status" / f"{index}.json").exists()):
            raise ValueError(f"Task {index} has already started; preserve its outputs and review it separately.")
    return {"manifest_sha256": sha256(campaign / "manifest.json"), "frozen_sha256": files,
            "tasks": [{"id": i, "label": tasks[i]["label"]} for i in TASK_IDS]}


def commands(nx: Path, campaign: Path, array: str, collector: str, new: str, binaries: dict) -> dict:
    return {
        "submit": [binaries["sbatch"], "--parsable", "--hold", "--chdir=" + str(nx),
                   "--export=ALL,PN_AFC_ROOT=" + str(nx.parent),
                   "--output=" + str(nx / "tmp/slurm_logs/pn-mc-long-%A_%a.log"),
                   "--array=17,45-47%2", "--dependency=afterany:" + array,
                   "--kill-on-invalid-dep=yes", str(nx / "batches/run_long.sbatch"),
                   str(campaign), str(TIMEOUT_SECONDS)],
        "collector_dependency": [binaries["scontrol"], "update", "JobId=" + collector,
                                 "Dependency=afterany:" + array + ":" + new],
        "cancel_original_pending": [binaries["scancel"], *[array + "_" + str(i) for i in TASK_IDS]],
        "release": [binaries["scontrol"], "release", new],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    ns = parser.parse_args()
    nx = Path(__file__).resolve().parents[1]
    launch_path = Path(json.loads((nx / "results/current_launch.json").read_text())["launch_record"]).resolve()
    launch_path.relative_to(nx / "results")
    launch = json.loads(launch_path.read_text())
    if launch.get("long_migration"):
        raise RuntimeError("Long tasks already migrated; inspect " + launch["long_migration"])
    campaign = Path(launch["campaign"]).resolve(strict=True)
    campaign.relative_to(nx / "tmp")
    array, collector = str(launch["run_job_id"]), str(launch["collect_job_id"])
    if not re.fullmatch(r"\d+", array) or not re.fullmatch(r"\d+", collector):
        raise ValueError("Invalid recorded job IDs.")
    inputs = frozen_inputs(campaign)
    binaries = {name: shutil.which(name) or name for name in ("sbatch", "scontrol", "scancel", "squeue")}
    if ns.dry_run:
        for command in commands(nx, campaign, array, collector, "LONG_ARRAY_JOB_ID", binaries).values():
            print(shlex.join(command))
        return 0
    if any(shutil.which(name) is None for name in binaries):
        raise RuntimeError("Slurm tools unavailable.")
    now = datetime.datetime.now(ZoneInfo("Europe/Berlin"))
    record = nx / "results" / f"long_migration_{now:%Y%m%d_%H%M%S_%f}.json"
    state = {"created_berlin": now.isoformat(), "campaign": str(campaign), "git_head": git_head(),
             "original_array_job_id": array, "collector_job_id": collector,
             "task_ids": list(TASK_IDS), "partition": "long", "slurm_limit_hours": 24,
             "effective_timeout_seconds": TIMEOUT_SECONDS, "max_concurrent_tasks": 2,
             "threads_per_task": 16, "inputs_before": inputs,
             "rationale": {"17": "128^2 P15 full step: estimated about 8h",
                           "45": "128^2 P15 half step: estimated about 16h",
                           "46": "finest lattice half step: estimate near 7h with uncertain reaction cost",
                           "47": "finest 3D half step: estimate near 7h without measured 3D production timing"},
             "status": "checking", "operations": []}
    # Journal the attempt before the first mutation to prevent duplicate submissions on retry.
    atomic_json(record, state)
    launch["long_migration"] = str(record)
    atomic_json(launch_path, launch)

    def execute(command: list[str], stage: str) -> str:
        state["status"] = stage
        atomic_json(record, state)
        result = subprocess.run(command, capture_output=True, text=True)
        state["operations"].append({"stage": stage, "command": command,
                                    "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        atomic_json(record, state)
        if result.returncode:
            raise RuntimeError(stage + ": " + result.stderr.strip())
        return result.stdout

    def require_pending(job: str) -> None:
        detail = execute([binaries["scontrol"], "show", "job", job, "-o"], "check_pending_" + job)
        if "JobState=PENDING" not in detail.split():
            raise RuntimeError("Job is no longer pending: " + job)

    try:
        partition = execute([binaries["scontrol"], "show", "partition", "long", "-o"], "check_partition")
        if "State=UP" not in partition.split():
            raise RuntimeError("Partition long is not UP.")
        for index in TASK_IDS:
            require_pending(array + "_" + str(index))
        require_pending(collector)
        command = commands(nx, campaign, array, collector, "LONG_ARRAY_JOB_ID", binaries)["submit"]
        reply = execute(command, "submit_held_long_array")
        new = reply.strip().split(";")[0]
        if not re.fullmatch(r"\d+", new):
            raise RuntimeError("Ambiguous sbatch response; inspect record before retrying.")
        state["long_array_job_id"] = new
        atomic_json(record, state)
        planned = commands(nx, campaign, array, collector, new, binaries)
        execute(planned["collector_dependency"], "extend_collector_dependency")
        # Recheck immediately before removing only the still-pending originals.
        frozen_inputs(campaign)
        for index in TASK_IDS:
            require_pending(array + "_" + str(index))
        execute(planned["cancel_original_pending"], "cancel_replaced_pending_tasks")
        execute(planned["release"], "release_long_array")
        state["queue_snapshot"] = execute([binaries["squeue"], "-j", array + "," + new + "," + collector,
                                            "-o", "%.24i %.24j %.10T %.10M %.12P %R"], "verify_queue")
        state["inputs_after"] = frozen_inputs(campaign, require_unstarted=False)
        if state["inputs_after"] != inputs:
            raise RuntimeError("Frozen campaign inputs changed during migration.")
        state["status"] = "migrated"
        atomic_json(record, state)
        launch["long_migration"] = str(record)
        launch["long_array_job_id"] = new
        atomic_json(launch_path, launch)
        print(f"Long array: {new}; tasks 17,45-47; 24h Slurm / 23h process timeout.")
        print(f"Collector {collector} waits for arrays {array} and {new}.")
        print("Record: " + str(record))
        print(state["queue_snapshot"])
    except Exception as error:
        state["status"] = "migration_error"
        state["error"] = str(error)
        atomic_json(record, state)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
