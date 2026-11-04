#!/usr/bin/env python3
"""The 2D crossing profiles must have full amplitude in their own plane."""
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

binary = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="m1pn_crossing_") as tmp:
    for dimension in [2, 3]:
        out = Path(tmp)/str(dimension)
        result = subprocess.run([binary, "--model", "legacy", "--dimension", str(dimension), "--pn-order", "3",
            "--elements", "10", "--case", "crossing", "--inflow", "crossing-transport",
            "--time", "0", "--output", str(out)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        maximum = 0
        with (out/"initial.csv").open() as file:
            for row in csv.DictReader(file):
                x, y, z = (float(row[k]) for k in ["x", "y", "z"])
                center = 0 if dimension == 2 else .5
                a = math.exp(-50*((x-.3)**2+(y-.5)**2+(z-center)**2))
                b = math.exp(-50*((x-.5)**2+(y-.3)**2+(z-center)**2))
                rho = math.sqrt(4*math.pi)*float(row["u0"])
                assert math.isclose(rho, .01+a+b, rel_tol=2e-14, abs_tol=2e-14)
                assert math.isclose(math.sqrt(4*math.pi/3)*float(row["u3"]), a, rel_tol=2e-14, abs_tol=2e-14)
                assert math.isclose(math.sqrt(4*math.pi/3)*float(row["u1"]), b, rel_tol=2e-14, abs_tol=2e-14)
                maximum = max(maximum, rho)
        assert maximum > 1
        print(json.dumps({"check": f"crossing_profile_{dimension}d", "passed": True, "maximum_density": maximum}))
