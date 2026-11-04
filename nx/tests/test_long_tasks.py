"""Exercise long-task migration with fake Slurm commands; never submit real jobs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

NX = Path(__file__).resolve().parents[1]


class LongMigrationTests(unittest.TestCase):
    def fixture(self, base):
        nx = base / "nx"
        (nx / "scripts").mkdir(parents=True)
        (nx / "results").mkdir()
        (nx / "batches").mkdir()
        for name in ("submit_long_tasks.py", "study.py"):
            shutil.copy2(NX / "scripts" / name, nx / "scripts" / name)
        campaign = nx / "tmp/campaign"
        campaign.mkdir(parents=True)
        tasks = [{"id": i, "label": f"task_{i}", "output_dir": f"runs/task_{i}"} for i in range(48)]
        for name, text in (("solver", "frozen binary"), ("verifier.py", "frozen verifier"),
                           ("tasks.json", json.dumps(tasks))):
            (campaign / name).write_text(text)
        digest = lambda name: hashlib.sha256((campaign / name).read_bytes()).hexdigest()
        manifest = {"threads": 16, "plan": {"method": {"stabilization": "m1pn_mass_correction"}},
                    "binary": "solver", "binary_sha256": digest("solver"),
                    "verifier": "verifier.py", "verifier_sha256": digest("verifier.py"),
                    "tasks": "tasks.json", "tasks_sha256": digest("tasks.json")}
        (campaign / "manifest.json").write_text(json.dumps(manifest))
        launch = nx / "results/launch.json"
        launch.write_text(json.dumps({"campaign": str(campaign), "run_job_id": "100", "collect_job_id": "101"}))
        (nx / "results/current_launch.json").write_text(json.dumps({"launch_record": str(launch)}))
        bin_dir = base / "bin"
        bin_dir.mkdir()
        fake = '''#!/usr/bin/env python3
import json, os, pathlib, sys
name=pathlib.Path(sys.argv[0]).name
with open(os.environ['FAKE_SLURM_JOURNAL'],'a') as f:f.write(json.dumps([name,*sys.argv[1:]])+'\\n')
if name=='sbatch':print('200')
elif name=='scontrol' and sys.argv[1:3]==['show','partition']:print('PartitionName=long State=UP MaxTime=2-00:00:00')
elif name=='scontrol' and sys.argv[1:3]==['show','job']:print('JobState=PENDING')
elif name=='squeue':print('100 med RUNNING\\n200 long PENDING Dependency\\n101 med PENDING Dependency')
'''
        for name in ("sbatch", "scontrol", "scancel", "squeue"):
            p = bin_dir / name
            p.write_text(fake)
            p.chmod(0o755)
        journal = base / "commands.jsonl"
        env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
                   FAKE_SLURM_JOURNAL=str(journal))
        return nx, campaign, launch, journal, env

    def test_migration_and_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            nx, campaign, launch, journal, env = self.fixture(Path(tmp))
            before = {name: (campaign / name).read_bytes() for name in ("manifest.json", "tasks.json", "solver", "verifier.py")}
            command = [sys.executable, str(nx / "scripts/submit_long_tasks.py")]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in journal.read_text().splitlines()]
            submit = next(c for c in calls if c[0] == "sbatch")
            self.assertIn("--hold", submit)
            self.assertIn("--array=17,45-47%2", submit)
            self.assertIn("--dependency=afterany:100", submit)
            self.assertEqual(submit[-1], "82800")
            update = ["scontrol", "update", "JobId=101", "Dependency=afterany:100:200"]
            cancel = ["scancel", "100_17", "100_45", "100_46", "100_47"]
            release = ["scontrol", "release", "200"]
            self.assertLess(calls.index(update), calls.index(cancel))
            self.assertLess(calls.index(cancel), calls.index(release))
            self.assertEqual(before, {name: (campaign / name).read_bytes() for name in before})
            migration = json.loads(Path(json.loads(launch.read_text())["long_migration"]).read_text())
            self.assertEqual(migration["status"], "migrated")
            self.assertEqual(migration["inputs_before"], migration["inputs_after"])
            repeated = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(len(calls), len(journal.read_text().splitlines()))

    def test_dry_run_and_started_task_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            nx, campaign, launch, journal, env = self.fixture(Path(tmp))
            command = [sys.executable, str(nx / "scripts/submit_long_tasks.py"), "--dry-run"]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(result.stdout.strip().splitlines()), 4)
            self.assertFalse(journal.exists())
            self.assertNotIn("long_migration", json.loads(launch.read_text()))
            (campaign / "runs/task_17").mkdir(parents=True)
            blocked = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("already started", blocked.stderr)
            self.assertFalse(journal.exists())


if __name__ == "__main__":
    unittest.main()
