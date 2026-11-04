#!/usr/bin/env python3
"""Start the requested NX preparation, dependent 48-task array and collector."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from zoneinfo import ZoneInfo

from study import atomic_json, git_head, sha256, validate_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print all commands without submitting or writing a launch record.")
    ns = parser.parse_args()
    nx = Path(__file__).resolve().parents[1]
    plan_path = nx / "study_plan.json"
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    families = plan["profiles"]["production"]
    count = sum(len(s["orders"]) * len(s["grids"]) for s in families.values()) + len(families)
    if count != 48 or plan["execution"]["max_concurrent_tasks"] != 2:
        raise RuntimeError("Expected the requested 48-task study with concurrency two.")
    now = datetime.datetime.now(ZoneInfo("Europe/Berlin"))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    campaign = nx / "tmp" / ("pn_mc_production_" + stamp)
    record = nx / "results" / ("launch_" + stamp + ".json")
    pointer = nx / "results" / "current_launch.json"
    if pointer.exists() and not ns.dry_run:
        previous_record = Path(json.loads(pointer.read_text())["launch_record"])
        previous_record.resolve().relative_to(nx / "results")
        previous = json.loads(previous_record.read_text())
        raise RuntimeError("A launch is already recorded; inspect its jobs before starting another: " + str(previous_record) + " (" + previous["status"] + ")")
    sbatch = shutil.which("sbatch")
    if not ns.dry_run and sbatch is None:
        raise RuntimeError("sbatch is unavailable; execute this launcher on the cluster login host.")
    sbatch = sbatch or "sbatch"
    logs = nx / "tmp" / "slurm_logs"
    state = {"status": "ready_to_submit", "created_berlin": now.isoformat(),
             "project": str(nx.parent), "campaign": str(campaign),
             "task_count": count, "threads_per_task": 16, "max_concurrent_tasks": 2,
             "method": plan["method"], "plan_sha256": sha256(plan_path),
             "git_head": git_head(),
             "launch_record": str(record)}

    def submit(stage: str, arguments: list[str], extra: list[str]) -> str:
        command = [sbatch, "--parsable", "--chdir=" + str(nx),
                   "--export=ALL,PN_AFC_ROOT=" + str(nx.parent),
                   "--output=" + str(logs / ("pn-mc-" + stage + "-%A_%a.log")),
                   *extra, str(nx / "batches" / (stage + ".sbatch")), *arguments]
        if ns.dry_run:
            print(shlex.join(command))
            return {"prepare": "PREPARE_JOB_ID", "run": "ARRAY_JOB_ID", "collect": "COLLECT_JOB_ID"}[stage]
        state["status"] = "submitting_" + stage
        state[stage + "_command"] = command
        atomic_json(record, state)
        result = subprocess.run(command, capture_output=True, text=True)
        state[stage + "_submission_stderr"] = result.stderr
        state[stage + "_submission_stdout"] = result.stdout
        state[stage + "_submission_returncode"] = result.returncode
        atomic_json(record, state)
        if result.returncode:
            raise RuntimeError(stage + " submission failed: " + result.stderr.strip())
        job = result.stdout.strip().split(";")[0]
        if not re.fullmatch(r"\d+", job):
            raise RuntimeError("Ambiguous sbatch response; inspect recorded commands before retrying.")
        state[stage + "_job_id"] = job
        atomic_json(record, state)
        print(stage + " job: " + job, flush=True)
        return job

    lock = nx / "results" / ".launch.lock"
    if not ns.dry_run:
        record.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        logs.mkdir(parents=True, exist_ok=True)
        atomic_json(record, state)
        atomic_json(pointer, {"launch_record": str(record)})
    try:
        prepare = submit("prepare", ["production", str(campaign)], [])
        array = submit("run", [str(campaign)], ["--dependency=afterok:" + prepare,
                       "--kill-on-invalid-dep=yes", "--cpus-per-task=16", "--array=0-47%2"])
        submit("collect", [str(campaign)], ["--dependency=afterok:" + prepare + ",afterany:" + array,
               "--kill-on-invalid-dep=yes"])
        if not ns.dry_run:
            state["status"] = "submitted"
            atomic_json(record, state)
            print("Launch record: " + str(record), flush=True)
    except Exception as error:
        if not ns.dry_run:
            state["status"] = "submission_error"
            state["error"] = str(error)
            atomic_json(record, state)
        raise
    finally:
        if not ns.dry_run:
            lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
