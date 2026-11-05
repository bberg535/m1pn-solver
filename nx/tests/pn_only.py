#!/usr/bin/env python3
"""Focused CLI and ledger checks for the additive standalone-PN path."""
import json
import pathlib
import subprocess
import sys
import tempfile

NX = pathlib.Path(__file__).resolve().parents[1]
binary = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(tempfile.mkdtemp(prefix="pn_only_checks_", dir=NX / "tmp"))

base = [str(binary), "--case", "crossing-study", "--dimension", "2", "--elements", "4",
        "--pn-order", "3", "--model", "r2", "--time-integrator", "ssprk3",
        "--stabilization", "m1pn_mass_correction", "--pn-viscosity-scale", "1",
        "--mass-mode", "lumped", "--inflow", "crossing-study-transport",
        "--background", ".001", "--pulse-width", ".1", "--threads", "1"]

def run(label, extra):
    folder = out / label
    command = [*base, *extra, "--output", str(folder)]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (out / f"{label}.log").write_text(json.dumps(command) + "\n" + proc.stdout + proc.stderr)
    assert proc.returncode == 0, (label, proc.stderr)
    return folder, proc.stdout

# This deliberately probes a finite state outside the M1 cone. It must be
# evaluated by the PN target without triggering M1 admissibility rejection.
_, stdout = run("oracle", ["--coupling", "pn-only", "--pn-only-self-test", "--time", "0"])
assert '"status":"passed"' in stdout and '"outside_cone_nodes":25' in stdout, stdout

folder, _ = run("short_run", ["--coupling", "pn-only", "--time", ".00002", "--dt", ".00001"])
cfg = json.loads((folder / "config.json").read_text())
summary = json.loads((folder / "summary.json").read_text())
assert cfg["model"] == "r2" and cfg["method"] == "m1pn_mass_correction"
assert cfg["coupling"] == cfg["coupling_mode"] == "pn-only"
assert cfg["pn_target"] == "standard_B_mass_corrected"
for key in ("m1_coupling_enabled", "m1_closure_enabled", "pair_limiter_enabled",
            "cone_rejection_enabled"):
    assert cfg[key] is False, (key, cfg[key])
assert cfg["mass_mode"] == "lumped" and cfg["mass_corrections"] == 0
assert cfg["mass_factorized"] is False and cfg["pn_viscosity_scale"] == 1
assert cfg["uses_pn_low_order_predictor"] is True
assert cfg["pn_mass_correction_applications"] == 1
assert summary["status"] == "completed" and summary["steps"] == 2
for key in ("mass_solve_calls", "m1_closure_evaluations", "pair_limiter_evaluations",
            "cone_rejections", "mass_iterations"):
    assert summary[key] == 0, (key, summary[key])
assert summary["biharmonic_seconds"] == 0
assert summary["pn_only_target_recovery_max"] < 2e-10
assert summary["pn_only_rate_residual_max"] < 2e-12
assert summary["maximum_step_balance"] < 2e-10

# Omitting --coupling must retain the established coupled default.
default_folder, _ = run("default_mode", ["--time", "0"])
default_cfg = json.loads((default_folder / "config.json").read_text())
assert default_cfg["coupling"] == default_cfg["coupling_mode"] == "m1pn"
assert default_cfg["m1_coupling_enabled"] is True
print(json.dumps({"status": "passed", "raw_outputs": str(out),
                  "pn_only_steps": summary["steps"],
                  "minimum_density_observed": summary["minimum_density_observed"],
                  "minimum_cone_margin_observed": summary["minimum_cone_margin_observed"]}))
