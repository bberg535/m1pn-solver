#!/usr/bin/env python3
"""Native rejection and rollback tests using only the Python standard library."""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def read(path):
    with path.open() as file:
        return list(csv.reader(file))


binary = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="m1pn_cli_") as tmp:
    root = Path(tmp)
    cases = [
        ("invalid_pair_limiter", ["--pair-limiter", "vector"], "Invalid pair limiter"),
        ("memory", ["--pn-order", "15", "--memory-mib", "1"], "Memory budget"),
        ("overflow", ["--elements", "1000000000"], "Mesh index overflow"),
        ("pn_step", ["--periodic", "--elements", "4", "--pn-order", "7", "--biharmonic-gamma", ".1", "--dt", ".03125", "--time", ".03125"], "independent PN time"),
        ("rk2_centered", ["--periodic", "--elements", "4", "--biharmonic-gamma", "0", "--time-integrator", "ssprk2"], "no positive energy-stable"),
        ("c0_data", ["--boundary", "characteristic-c0", "--inflow", "isotropic"], "C0 h=0 requires vacuum"),
        ("vacuum_beam", ["--dimension", "2", "--pn-order", "7", "--case", "vacuum", "--inflow", "beam"], "13 conservative retries"),
        ("pure_beam", ["--dimension", "2", "--pn-order", "7", "--case", "pure-beam"], "13 conservative retries"),
    ]
    for name, options, message in cases:
        out = root / name
        result = subprocess.run([binary, "--model", "legacy", "--time", ".0005", "--output", str(out), *options], text=True, capture_output=True, timeout=30)
        assert result.returncode == 2 and message in result.stderr, (name, result.returncode, result.stderr)
        if "retries" in message:
            assert read(out / "initial.csv") == read(out / "rollback_state.csv")
            for filename in ["rollback_boundary.csv", "rollback_reaction.csv"]:
                assert all(float(value) == 0 for row in read(out / filename)[1:] for value in row[4:])
            report = json.loads((out / "rejection.json").read_text())
            assert report["accepted_steps"] == 0 and report["rejected_attempts"] == 13
        print(json.dumps({"check": name, "status": "passed"}))
