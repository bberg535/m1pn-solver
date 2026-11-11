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

    def test_scientific_families(self):
        tasks = study.make_tasks("paper")
        families = {t["family"] for t in tasks}
        self.assertTrue({"smooth", "smoothdt2", "boundary", "crossing",
                         "line", "time_smooth", "time_crossing",
                         "ablation", "baseline"} <= families)
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
                if t["family"] == "baseline":
                    self.assertEqual(t["stabilization"], "m1pn_mass_correction")
                    self.assertEqual(t["rk"], "ssprk3")
                elif t["family"] != "ablation":
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

    def test_solver_flags_split_reaction(self):
        t = next(t for t in study.make_tasks("paper") if t["family"] == "time_crossing")
        opts = study.flags(t, Path("/tmp/output"), 16, end=0.,
                           dt=None)
        self.assertEqual(opts[opts.index("--sigma-s") + 1], "2.0")
        self.assertEqual(opts[opts.index("--sigma-a") + 1], "0.5")
        self.assertEqual(opts[opts.index("--filter") + 1], "constant")
        self.assertEqual(opts[opts.index("--dt") + 1], "0")
        self.assertEqual(opts[opts.index("--time") + 1], "0")


if __name__ == "__main__":
    unittest.main()
