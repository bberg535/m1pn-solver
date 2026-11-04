import importlib.util
import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "study.py"
spec = importlib.util.spec_from_file_location("nx_study", SCRIPT)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
ANALYSIS_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis.py"
analysis_spec = importlib.util.spec_from_file_location("nx_analysis_for_timeout_test", ANALYSIS_SCRIPT)
analysis = importlib.util.module_from_spec(analysis_spec)
analysis_spec.loader.exec_module(analysis)


class StudyPlanTests(unittest.TestCase):
    def test_configuration_reaches_solver_arguments(self):
        plan=study.load_plan()
        method=dict(plan['method'])
        configured=dict(plan,method=method)
        study.validate_plan(configured)
        argv=study.build_args(plan['profiles']['smoke']['crossing'],3,8,'output',4,
                              1234,method,t=.01,dt=.0001,memory_mib=3072)
        flags=dict(zip(argv[::2],argv[1::2]))
        self.assertEqual(flags['--mass-mode'],'lumped')
        self.assertNotIn('--mass-corrections',flags)
        self.assertEqual(flags['--stabilization'],'m1pn_mass_correction')
        self.assertEqual(flags['--kv-gamma'],'0.0')
        self.assertEqual(flags['--biharmonic-gamma'],'0.0')
        self.assertEqual(flags['--pn-viscosity-scale'],'1.0')
        self.assertEqual(flags['--memory-mib'],'3072')
        self.assertEqual(flags['--max-steps'],'1234')

    def test_unsupported_method_rejected_explicitly(self):
        plan=study.load_plan()
        for key,value in [('model','legacy'),('mass_mode','lumped_afc'),('integrator','ssprk2'),
                          ('stabilization','biharmonic-corrected'),('mass_corrections',1),
                          ('pn_viscosity_scale',2.),('biharmonic_gamma',.004),
                          ('kv_gamma',float('nan'))]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                study.validate_plan(dict(plan,method=dict(plan['method'],**{key:value})))

    def test_plan_cartesian_counts(self):
        plan = study.load_plan()
        expected = {"crossing": 18, "lattice": 15, "three_d": 12}
        for family, count in expected.items():
            case = plan["profiles"]["production"][family]
            self.assertEqual(len(case["orders"]) * len(case["grids"]), count)
            smoke = plan["profiles"]["smoke"][family]
            self.assertEqual(len(smoke["orders"]) * len(smoke["grids"]), 4)
        self.assertEqual(sum(expected.values()), 45)
        self.assertEqual(sum(expected.values()) + len(expected), 48)

    def test_probe_step_limits_freeze_common_case_dt(self):
        # Each probe contributes two independently calculated stability limits.
        probes = [{"family": "crossing", "low_cfl_limit": 0.08, "pn_step_limit": 0.05},
                  {"family": "crossing", "low_cfl_limit": 0.12, "pn_step_limit": 0.04}]
        families = [("crossing", {"time": 0.32})]
        common_dt = study.freeze_common_dts(families, probes, 0.2)["crossing"]
        self.assertAlmostEqual(common_dt, 0.008)
        assigned = [common_dt] * 6
        self.assertEqual(len(set(assigned)), 1)

    def test_probe_failure_exit_code_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / "fail.py"
            exe.write_text("#!/usr/bin/env python3\nraise SystemExit(7)\n")
            exe.chmod(0o755)
            log = Path(td) / "probe.log"
            rc, _ = study.run_logged([str(exe)], log, 5)
            self.assertEqual(rc, 7)
            self.assertTrue(log.is_file())

    def test_existing_campaign_path_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=study.ROOT) as td:
            campaign = Path(td) / "occupied"
            campaign.mkdir()
            marker = campaign / "keep"
            marker.write_text("do not overwrite")
            class Args:
                pass
            args = Args()
            args.exe = Path("/bin/true")
            args.profile = "smoke"
            args.output = campaign
            args.threads = 1
            args.cases = ["crossing"]
            with self.assertRaises(FileExistsError):
                study.prepare(args)
            self.assertEqual(marker.read_text(), "do not overwrite")

    def test_fake_t0_prepare_freezes_tasks_and_creates_logs(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            exe = td / "fake_solver.py"
            exe.write_text("""#!/usr/bin/env python3
import json, pathlib, sys
a = sys.argv[1:]
out = pathlib.Path(a[a.index('--output') + 1])
out.mkdir(parents=True, exist_ok=True)
args = dict(zip(a[::2], a[1::2]))
(out / 'config.json').write_text(json.dumps({
 'low_cfl_limit': .05, 'pn_step_limit': .04, 'mass_mode': 'lumped',
 'mass_corrections': 0, 'stabilization': 'm1pn_mass_correction',
 'pn_mass_correction_applications': 1, 'uses_pn_low_order_predictor': True,
 'mass_factorized': False, 'gamma': 0., 'kv_gamma': 0.,
 'pn_viscosity_scale': 1.}))
""")
            exe.chmod(0o755)
            campaign = study.ROOT / "tmp" / ("test_study_" + str(os.getpid()))
            if campaign.exists():
                self.fail(f"temporary campaign unexpectedly exists: {campaign}")
            rc = study.main(["prepare", "--exe", str(exe), "--profile", "smoke",
                             "--output", str(campaign), "--threads", "1", "--cases", "crossing"])
            try:
                self.assertEqual(rc, 0)
                self.assertTrue((campaign / "logs").is_dir())
                tasks = json.loads((campaign / "tasks.json").read_text())
                spatial = [t for t in tasks if not t["temporal_ref"]]
                self.assertEqual(len(spatial), 4)
                self.assertEqual(len({t["dt"] for t in spatial}), 1)
                self.assertTrue(all("solver_args" in t for t in tasks))
                self.assertEqual([t["steps"] for t in tasks if not t["temporal_ref"]], [1] * 4)
                self.assertEqual([t["steps"] for t in tasks if t["temporal_ref"]], [2])
                manifest = json.loads((campaign / "manifest.json").read_text())
                self.assertEqual(manifest["tasks_sha256"], hashlib.sha256(
                    (campaign / "tasks.json").read_bytes()).hexdigest())
            finally:
                import shutil
                shutil.rmtree(campaign)

    def test_run_timeout_override_preserves_frozen_solver_command_and_collector_accepts(self):
        for override, expected, source in ((82800, 82800, "cli_override"),
                                           (None, 25200, "manifest")):
            with self.subTest(timeout=override), tempfile.TemporaryDirectory(dir=study.ROOT) as td:
                campaign = Path(td)
                binary = campaign / "m1pn_mc_study"
                binary.write_bytes(b"frozen executable")
                verifier = campaign / "verify_run.py"
                verifier.write_text("# frozen verifier fixture\n")
                task = {
                    "id": 0, "label": "timeout_test", "output_dir": "runs/timeout_test",
                    "solver_args": ["--case", "crossing-study", "--inflow", "crossing-study-transport",
                                    "--dimension", "2", "--elements", "4", "--pn-order", "3",
                                    "--time", "0.1", "--dt", "0.1", "--output", "runs/timeout_test"],
                    "family": "lattice", "kind": "spatial", "case": "crossing-study",
                    "inflow": "crossing-study-transport", "dimension": 2, "grid": 4,
                    "order": 3, "time": .1, "dt": .1, "steps": 1, "width": .07,
                    "background": .001, "method": {"mass_mode":"lumped", "mass_corrections":0,
                        "stabilization":"m1pn_mass_correction", "kv_gamma":0.,
                        "biharmonic_gamma":0., "pn_viscosity_scale":1.}, "memory_mib": 12,
                }
                tasks_path = campaign / "tasks.json"
                tasks_path.write_text(json.dumps([task]))
                digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
                manifest = {
                    "binary": binary.name, "binary_sha256": digest(binary),
                    "verifier": verifier.name, "verifier_sha256": digest(verifier),
                    "tasks": "tasks.json", "tasks_sha256": digest(tasks_path), "threads": 1,
                    "plan": {"execution": {"timeout_seconds": 25200, "memory_mib": 12},
                             "method": task["method"]},
                }
                (campaign / "manifest.json").write_text(json.dumps(manifest))
                expected_command = [str(binary), *task["solver_args"]]
                output_arg = expected_command.index("--output") + 1
                expected_command[output_arg] = str(campaign / task["output_dir"])
                observed = {}

                def fake_run_logged(command, log, timeout):
                    observed["command"] = command
                    observed["timeout"] = timeout
                    running = json.loads((campaign / "status" / "0.json").read_text())
                    self.assertEqual(running["status"], "running")
                    self.assertEqual(running["effective_timeout_seconds"], expected)
                    self.assertEqual(running["timeout_source"], source)
                    out = Path(command[command.index("--output") + 1])
                    out.mkdir(parents=True)
                    cfg = {"mass_mode":"lumped", "mass_corrections":0,
                           "stabilization":"m1pn_mass_correction", "kv_gamma":0.,
                           "gamma":0., "pn_viscosity_scale":1.,
                           "pn_mass_correction_applications":1,
                           "uses_pn_low_order_predictor":True, "memory_budget_bytes":12*1024*1024,
                           "mass_factorized":False, "rk":3, "pn_order":3,
                           "elements_per_axis":4, "dimension":2, "case":"crossing-study",
                           "inflow":"crossing-study-transport", "dt_requested":.1,
                           "dt_selected":.1, "final_time":.1, "low_cfl_limit":.2,
                           "background":.001, "pulse_width":.07}
                    summary = {"mass_solve_calls":0, "biharmonic_seconds":0.,
                               "pn_predictor_seconds":.02, "pn_mass_correction_seconds":.001,
                               "status":"completed", "time":.1, "steps":1,
                               "rejected_attempts":0}
                    (out / "config.json").write_text(json.dumps(cfg))
                    (out / "summary.json").write_text(json.dumps(summary))
                    return 0, .5

                ns = type("Args", (), {"campaign":campaign, "task":0, "launcher":"direct",
                                         "timeout":override})()
                with mock.patch.object(study, "run_logged", side_effect=fake_run_logged), \
                     mock.patch.object(study.subprocess, "run",
                                       return_value=subprocess.CompletedProcess([], 0, stdout="verified")):
                    self.assertEqual(study.run_task(ns), 0)
                self.assertEqual(observed["timeout"], expected)
                self.assertEqual(observed["command"], expected_command)
                status = json.loads((campaign / "status" / "0.json").read_text())
                self.assertEqual(status["status"], "completed")
                self.assertEqual(status["effective_timeout_seconds"], expected)
                self.assertEqual(status["timeout_source"], source)
                self.assertEqual(status["command"], expected_command)

                # The collector consumes the completed status without consulting
                # the per-attempt timeout or changing the frozen task manifest.
                with mock.patch.object(analysis, "task_result", return_value={
                        "id":0, "label":"timeout_test", "family":"lattice", "grid":4,
                        "order":3, "kind":"spatial", "time":.1, "dt":.1,
                        "coordinates":analysis.np.array([[0.,0.,0.]]),
                        "physical":analysis.np.zeros((1,4)), "mass":analysis.np.ones(1),
                        "domain_volume":1.}), mock.patch.object(analysis, "make_plots"):
                    self.assertEqual(analysis.analyze(campaign), 0)
                collected = json.loads((campaign / "analysis_summary.json").read_text())
                self.assertEqual(collected["expected_tasks"], 1)
                self.assertEqual(collected["analyzed_tasks"], 1)
                self.assertTrue(collected["execution_complete"])
                self.assertTrue(collected["quality_accepted"])

    def test_run_wrapper_rejects_invalid_timeout_before_nx_setup(self):
        wrapper = SCRIPT.parent / "run_task_nx.sh"
        for invalid in ("0", "abc", "12.5"):
            with self.subTest(timeout=invalid):
                result = subprocess.run(["bash", str(wrapper), "campaign", invalid],
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("TIMEOUT_SECONDS must be a positive integer", result.stdout)
                self.assertNotIn("prepared sbatch wrappers", result.stdout)

    def test_run_wrapper_forwards_optional_timeout_and_keeps_default_invocation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scripts = root / "scripts"
            scripts.mkdir()
            import shutil
            shutil.copy2(SCRIPT.parent / "run_task_nx.sh", scripts / "run_task_nx.sh")
            (scripts / "nx_common.sh").write_text(
                f"TASK_NX={str(root)!r}\n"
                "nx_fail() { echo \"$*\" >&2; exit 2; }\n"
                "nx_inside() { return 0; }\n"
                "nx_setup() { TASK_PYTHON=python3; OMP_NUM_THREADS=1; export OMP_NUM_THREADS; }\n")
            (root / "scripts" / "study.py").write_text(
                "import json,sys\nprint(json.dumps(sys.argv[1:]))\n")
            campaign = root / "campaign"
            campaign.mkdir()
            (campaign / "manifest.json").write_text(json.dumps({"threads":1}))
            env = dict(os.environ, SLURM_ARRAY_TASK_ID="7")
            for timeout in (None, "82800"):
                command = ["bash", str(scripts / "run_task_nx.sh"), str(campaign)]
                if timeout is not None:
                    command.append(timeout)
                result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stdout)
                forwarded = json.loads(result.stdout)
                self.assertEqual(forwarded[:4], ["run", "--campaign", str(campaign), "--task"])
                self.assertEqual(forwarded[4], "7")
                if timeout is None:
                    self.assertNotIn("--timeout", forwarded)
                else:
                    self.assertEqual(forwarded[-2:], ["--timeout", "82800"])


if __name__ == "__main__":
    unittest.main()
