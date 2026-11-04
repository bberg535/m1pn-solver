import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis.py"
spec = importlib.util.spec_from_file_location("nx_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AnalysisMathTests(unittest.TestCase):
    def test_physical_harmonics_reordering_and_scaling(self):
        # Stored [u0,u1,u2,u3], with physical rho=2 and J=(.3,.4,0).
        physical = np.array([[2., .3, .4, 0.]])
        stored = physical[:, [0, 2, 3, 1]] / np.array(
            [np.sqrt(4*np.pi), np.sqrt(4*np.pi/3), np.sqrt(4*np.pi/3), np.sqrt(4*np.pi/3)])
        np.testing.assert_allclose(analysis.physical_low(stored), physical)

    def test_beam_solution_density_and_flux(self):
        geom = {"background": .1, "width_exponent": -2., "beams": [
            {"center": [0., 0., 0.], "direction": [1., 0., 0.]},
            {"center": [1., 0., 0.], "direction": [0., 1., 0.]}]}
        rho, flux = analysis.beam_exact(np.array([[0., 0., 0.]]), 0., geom)
        self.assertAlmostEqual(rho[0], 1.1 + np.exp(-2.))
        self.assertAlmostEqual(flux[0, 0], 1.)
        self.assertAlmostEqual(flux[0, 1], np.exp(-2.))

    def test_trilinear_interpolation_on_2d_grid(self):
        axes = np.linspace(0., 1., 3)
        coords = np.array([[x, y, 0.] for x in axes for y in axes])
        values = (coords[:, 0]**2 + 2*coords[:, 1])[:, None]
        target = np.array([[.25, .25, 0.], [.75, .5, 0.]])
        got = analysis.interpolate_structured(coords, values, target, 2)
        # The stated interpolation is piecewise linear, so these are midpoint averages.
        np.testing.assert_allclose(got[:, 0], [.625, 1.625])

    def test_coordinate_alignment_reorders_and_rejects_mismatch(self):
        reference = np.array([[0., 0., 0.], [1., 0., 0.]])
        other = reference[::-1].copy()
        np.testing.assert_array_equal(analysis.coordinate_permutation(reference, other), [1, 0])
        with self.assertRaises(ValueError):
            analysis.coordinate_permutation(reference, np.array([[0., 0., 0.], [.5, 0., 0.]]))

    def test_task_result_accepts_standard_b_and_rejects_biharmonic_work(self):
        with tempfile.TemporaryDirectory() as td:
            campaign = Path(td)
            out = campaign / "runs" / "case"
            out.mkdir(parents=True)
            (out / "initial.csv").write_text("node,x,y,z,u0,u1,u2,u3\n0,0,0,0,1,0,0,0\n")
            (out / "final.csv").write_text("node,x,y,z,u0,u1,u2,u3\n0,0,0,0,1,0,0,0\n")
            (out / "nodal_mass.csv").write_text("node,mass\n0,1\n")
            (out / "boundary_integral.csv").write_text("time,value\n0,0\n")
            (out / "reaction_integral.csv").write_text("time,value\n0,0\n")
            cfg = {"mass_mode":"lumped", "mass_corrections":0,
                   "stabilization":"m1pn_mass_correction", "kv_gamma":0.,
                   "gamma":0., "pn_viscosity_scale":1.,
                   "pn_mass_correction_applications":1,
                   "uses_pn_low_order_predictor":True, "mass_factorized":False,
                   "rk":3, "case":"crossing-study", "dimension":2,
                   "elements_per_axis":1, "pn_order":1, "final_time":.1,
                   "dt_selected":.1}
            summary = {"mass_solve_calls":0, "status":"completed", "time":.1,
                       "biharmonic_seconds":0., "pn_predictor_seconds":.2,
                       "pn_mass_correction_seconds":.01}
            (out / "config.json").write_text(json.dumps(cfg))
            (out / "summary.json").write_text(json.dumps(summary))
            names = ["initial.csv", "final.csv", "nodal_mass.csv", "config.json",
                     "summary.json", "boundary_integral.csv", "reaction_integral.csv"]
            digests = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                       for name in names}
            (out / "verification.json").write_text(json.dumps({"status":"passed", "sha256":digests}))
            task = {"id":0, "label":"case", "output_dir":"runs/case", "time":.1,
                    "dt":.1, "grid":1, "order":1, "dimension":2,
                    "case":"crossing-study", "inflow":"vacuum", "kind":"spatial",
                    "method":{"mass_mode":"lumped", "mass_corrections":0,
                              "stabilization":"m1pn_mass_correction", "kv_gamma":0.,
                              "biharmonic_gamma":0., "pn_viscosity_scale":1.}}
            status = {"validated":True, "output_dir":"runs/case"}
            analysis.task_result(campaign, task, status)
            summary["biharmonic_seconds"] = .001
            (out / "summary.json").write_text(json.dumps(summary))
            digests["summary.json"] = hashlib.sha256((out / "summary.json").read_bytes()).hexdigest()
            (out / "verification.json").write_text(json.dumps({"status":"passed", "sha256":digests}))
            with self.assertRaisesRegex(ValueError, "no_biharmonic_work"):
                analysis.task_result(campaign, task, status)


if __name__ == "__main__":
    unittest.main()
