import importlib.util
import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "study.py"
spec = importlib.util.spec_from_file_location("nx_study", SCRIPT)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


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


if __name__ == "__main__":
    unittest.main()
