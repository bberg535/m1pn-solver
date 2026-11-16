#!/usr/bin/env python3
"""Offline structural tests; no MFEM build or NX allocation needed."""
from __future__ import annotations
import importlib.util
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts/element_weno/nx_convergence.py"
spec = importlib.util.spec_from_file_location("nx_convergence_study", SRC)
study = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(study)


class CampaignPlan(unittest.TestCase):
    def test_unique_and_stable_tasks(self):
        for profile in ("smoke", "paper"):
            tasks = study.make_tasks(profile)
            self.assertTrue(tasks)
            self.assertEqual([t["id"] for t in tasks], list(range(len(tasks))))
            self.assertEqual(len({t["label"] for t in tasks}), len(tasks))
            self.assertEqual(tasks, study.make_tasks(profile))
            self.assertEqual(len(tasks), 21 if profile == "smoke" else 79)  # 81 minus the two mass-correction baselines

    def test_scientific_families(self):
        tasks = study.make_tasks("paper")
        families = {t["family"] for t in tasks}
        self.assertTrue({"smooth", "smoothdt2", "boundary", "crossing",
                         "line", "time_smooth", "time_crossing",
                         "ablation"} <= families)
        self.assertNotIn("baseline", families)  # mass correction removed 2026-11-16
        for name in ("time_smooth", "time_crossing"):
            family = [t for t in tasks if t["family"] == name]
            self.assertEqual(len(family), 5)
            self.assertEqual(sum(t["reference"] for t in family), 1)
            self.assertEqual([t["dt_level"] for t in family], list(range(5)))

    def test_ssprk43_explicit_and_baselines_labeled(self):
        for t in study.make_tasks("paper"):
            if t["binary"] == "standalone":
                opts = study.flags(t, Path("/tmp/result"), 16, dt=.001)
                self.assertIn("--ssprk43", opts)
                self.assertEqual(t["kind"], "smooth_exact")
            else:
                opts = study.flags(t, Path("/tmp/result"), 16)
                self.assertEqual(opts[opts.index("--time-integrator") + 1], t["rk"])
                self.assertEqual(opts[opts.index("--angular-reduction") + 1], "off")
                self.assertEqual(opts[opts.index("--mirror-reduction") + 1], "off")
                self.assertEqual(t["stabilization"], "element-weno")
                if t["family"] != "ablation":
                    self.assertEqual(t["rk"], "ssprk43")

    def test_no_silent_override_and_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            f = root / "data.txt"
            f.write_text("test")
            h = study.digest(f)
            study.atomic(root / "manifest.json", {"format": study.FROZEN,
                          "task_count": 0, "checksums": {"data.txt": h}})
            study.atomic(root / "tasks.json", [])
            self.assertEqual(study.check_manifest(root), [])
            f.write_text("changed")
            with self.assertRaises(ValueError):
                study.check_manifest(root)

    def test_nonlinear_weno_weight_is_represented(self):
        tasks = study.make_tasks("paper")
        weno = [t for t in tasks if t["family"] == "ablation"
                and t["weno_linear"] == .2]
        self.assertEqual(len(weno), 1)
        opts = study.flags(weno[0], Path("/tmp/results"), 16)
        self.assertEqual(opts[opts.index("--weno-linear") + 1], "0.2")

    def test_every_element_weno_task_passes_its_weno_flags(self):
        # Regression of the 2026-11-11 smoke failure: element-WENO options must reach the solver for all tasks.
        for profile in ("smoke", "paper"):
            for t in study.make_tasks(profile):
                if t["binary"] != "fast":
                    continue
                opts = study.flags(t, Path("/tmp/out"), 16)
                if t["stabilization"] == "element-weno":
                    self.assertEqual(opts[opts.index("--weno") + 1], "on" if t["weno"] else "off", t["label"])
                    self.assertEqual(opts[opts.index("--element-limit") + 1],
                                     "on" if t["limiter"] else "off", t["label"])
                    self.assertEqual(float(opts[opts.index("--weno-linear") + 1]), t["weno_linear"], t["label"])
                    self.assertNotIn("--mass-mode", opts, t["label"])
                else:
                    self.assertNotIn("--weno", opts, t["label"])
                    self.assertIn("--mass-mode", opts, t["label"])

    def test_solver_flags_split_reaction(self):
        t = next(t for t in study.make_tasks("paper") if t["family"] == "time_crossing")
        opts = study.flags(t, Path("/tmp/output"), 16, end=0.,
                           dt=None)
        self.assertEqual(opts[opts.index("--sigma-s") + 1], "2.0")
        self.assertEqual(opts[opts.index("--sigma-a") + 1], "0.5")
        self.assertEqual(opts[opts.index("--filter") + 1], "constant")
        self.assertEqual(opts[opts.index("--dt") + 1], "0")
        self.assertEqual(opts[opts.index("--time") + 1], "0")


class HeterogeneousProfiles(unittest.TestCase):
    def test_profiles_are_separate_and_unique(self):
        for profile in study.HETERO_PROFILES:
            tasks = study.make_tasks(profile)
            labels = [t["label"] for t in tasks]
            self.assertEqual(len(labels), len(set(labels)))
            kinds = {t["kind"] for t in tasks}
            self.assertTrue({"lattice", "lattice3d"} <= kinds)
            # shadow archived (12.11.2026): only the smoke profile keeps it as a functional test
            self.assertEqual("shadow" in kinds, profile == "hetero-smoke")
        # the frozen v1 matrices are unchanged by the new profiles
        self.assertEqual(len(study.make_tasks("paper")), 79)
        self.assertEqual(len(study.make_tasks("smoke")), 21)

    def test_flags_and_expectations(self):
        for t in study.make_tasks("hetero-paper"):
            opts = study.flags(t, Path("/tmp/out"), 16)
            self.assertEqual(opts[opts.index("--case") + 1], t["case"], t["label"])
            self.assertEqual(float(opts[opts.index("--background") + 1]), t["background"])
            if t["stabilization"] == "element-weno":
                self.assertEqual(opts[opts.index("--weno-linear") + 1], str(t["weno_linear"]))
                self.assertEqual("--skip-fields" in opts, bool(t["skip_fields"]))
                if t["kind"] == "shadow":
                    self.assertEqual(float(opts[opts.index("--shadow-sigma") + 1]), t["shadow_sigma"])
                    self.assertEqual(t["expect_mirror"], [False, True, True])
                if t["kind"] == "lattice3d":
                    self.assertEqual(t["expect_mirror"], [False, False, False])
            else:
                self.assertNotIn("--weno", opts)
            if t.get("fixed_dt"):
                self.assertEqual(float(opts[opts.index("--dt") + 1]), study.LATTICE_B_DT)
            self.assertGreater(t["estimate_core_hours"], 0)
            self.assertIn(t["slot_cores"], (16, 32))

    def test_only_element_weno_tasks(self):
        for profile in ("paper", "hetero-paper", "hetero-smoke"):
            for t in study.make_tasks(profile):
                self.assertEqual(t.get("stabilization", "element-weno"), "element-weno", t["label"])

    def test_required_files(self):
        shadow = next(t for t in study.make_tasks("hetero-smoke") if t["kind"] == "shadow")
        self.assertIn("slices.csv", study.required_files(shadow))
        self.assertNotIn("final.csv", study.required_files(shadow))
        lattice = next(t for t in study.make_tasks("hetero-smoke") if t["kind"] == "lattice")
        self.assertIn("final.csv", study.required_files(lattice))

    def test_validation_accepts_and_rejects(self):
        t = next(t for t in study.make_tasks("hetero-smoke") if t["kind"] == "lattice" and t["elements"] == 28)
        config = {"case": "lattice-paul", "pn_order": t["pn_order"], "dimension": 2, "method": "element-weno",
                  "time_integrator": t["rk"], "degree": t["degree"], "elements_per_axis": t["elements"],
                  "weno": True, "element_limit": True, "weno_linear": t["weno_linear"], "dt_selected": .01,
                  "material_expected": {"volume": 49, "source": 1, "sigma_a": 110, "sigma_s": 38},
                  "material_integrals": {"volume": 49., "source": 1., "sigma_a": 110., "sigma_s": 38.},
                  "mirror_directions": [1, 0, 0]}
        summary = {"status": "completed", "time": 3.2, "final_density_mass": 2.,
                   "ledger": {"balance_relative": 1e-13}, "minimum_cone_margin": 1e-12}
        study.validate_hetero(t, config, summary)
        for key, bad in (("material_integrals", {"volume": 49., "source": 1., "sigma_a": 109., "sigma_s": 38.}),
                         ("mirror_directions", [1, 1, 0])):
            broken = dict(config, **{key: bad})
            with self.assertRaises(ValueError):
                study.validate_hetero(t, broken, summary)
        with self.assertRaises(ValueError):
            study.validate_hetero(t, config, dict(summary, ledger={"balance_relative": 1e-6}))
        with self.assertRaises(ValueError):
            study.validate_hetero(t, config, dict(summary, minimum_cone_margin=-1e-9))

    def test_plan_runs(self):
        import contextlib, io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(study.plan(type("NS", (), {"profile": "hetero-paper"})()), 0)
        self.assertIn("core-hours", buffer.getvalue())

if __name__ == "__main__":
    unittest.main()
