#!/usr/bin/env python3
"""Independent file readback: stored physical cone and all-Q global ledger."""
import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import numpy as np


def field(path):
    return np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))


def verify(path):
    path = Path(path)
    initial = field(path / "initial.csv")[:, 4:]
    final = field(path / "final.csv")[:, 4:]
    boundary = field(path / "boundary_integral.csv")[:, 4:]
    reaction = field(path / "reaction_integral.csv")[:, 4:]
    mass = field(path / "nodal_mass.csv")[:, 1]
    assert initial.shape == final.shape == boundary.shape == reaction.shape
    assert np.isfinite(final).all()
    scale = [math.sqrt(4 * math.pi)] + [math.sqrt(4 * math.pi / 3)] * 3
    low = final[:, [0, 3, 1, 2]] * scale
    # This tests the exact dyadic values of rounded physical projections.
    for row in low:
        r, x, y, z = map(Fraction.from_float, map(float, row))
        assert r >= 0 and r * r >= x * x + y * y + z * z, row
    balance = np.array([math.fsum(float(mass[i]) * (float(final[i, k]) - float(initial[i, k]))
                        + float(boundary[i, k]) - float(reaction[i, k])
                        for i in range(len(mass))) for k in range(final.shape[1])])
    error = float(np.max(np.abs(balance)))
    assert error < 2e-10, error
    result = {"status": "passed", "nodes": len(mass), "Q": final.shape[1],
              "stored_projection_exact_dyadic_cone": True,
              "maximum_global_Q_balance": error,
              "balance_by_component": balance.tolist(),
              "minimum_margin": float(np.min(low[:, 0] - np.linalg.norm(low[:, 1:], axis=1))),
              "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in path.iterdir() if p.is_file() and p.name != "verification.json"},
              "scope": "Saved initial/final fields and global boundary/reaction ledger; not a full PN cone or entropy certificate."}
    (path / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run), indent=2))
