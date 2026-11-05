#!/usr/bin/env python3
"""Read back a PN-only run and verify finite fields plus the all-Q global ledger.

This verifier deliberately reports low-moment density/cone diagnostics without
requiring positivity: PN-only evolution has no M1 cone guarantee.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NX = ROOT / "nx"
sys.path.insert(0, str(NX / "scripts"))
import analysis as study_analysis

REQUIRED = ("initial.csv", "final.csv", "nodal_mass.csv", "config.json",
            "summary.json", "boundary_integral.csv", "reaction_integral.csv")
TOLERANCE = 2e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    try:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [[float(x) for x in row] for row in reader if row]
    except (OSError, StopIteration, ValueError) as exc:
        raise ValueError(f"cannot read finite numeric CSV {path.name}: {exc}") from exc
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(header) or not np.isfinite(data).all():
        raise ValueError(f"invalid/non-finite CSV data: {path.name}")
    return header, data


def compute(run: Path) -> tuple[dict, dict]:
    run = run.resolve(strict=True)
    raw = {}
    headers = {}
    for name in REQUIRED:
        path = run / name
        if not path.is_file():
            raise ValueError(f"required PN run file is missing: {name}")
        if path.suffix == ".csv":
            headers[name], raw[name] = read_csv(path)
    initial = raw["initial.csv"]
    final = raw["final.csv"]
    mass_data = raw["nodal_mass.csv"]
    boundary = raw["boundary_integral.csv"]
    reaction = raw["reaction_integral.csv"]
    if headers["initial.csv"][:4] != ["node", "x", "y", "z"] or \
            headers["final.csv"] != headers["initial.csv"]:
        raise ValueError("initial/final field headers do not match expected node-coordinate layout")
    if headers["boundary_integral.csv"] != headers["initial.csv"] or \
            headers["reaction_integral.csv"] != headers["initial.csv"]:
        raise ValueError("boundary/reaction ledger headers do not match all-Q field layout")
    if headers["nodal_mass.csv"][:2] != ["node", "mass"]:
        raise ValueError("nodal_mass.csv must begin with node,mass")
    if initial.shape != final.shape or boundary.shape != final.shape or reaction.shape != final.shape:
        raise ValueError("initial/final/boundary/reaction arrays have inconsistent shapes")
    if len(mass_data) != len(final) or mass_data.shape[1] < 2:
        raise ValueError("nodal mass and field row counts differ")
    if not np.array_equal(initial[:, :4], final[:, :4]) or \
            not np.array_equal(initial[:, :4], boundary[:, :4]) or \
            not np.array_equal(initial[:, :4], reaction[:, :4]) or \
            not np.array_equal(initial[:, 0], mass_data[:, 0]):
        raise ValueError("node IDs/coordinates differ between fields and ledgers")
    if len(np.unique(final[:, 0])) != len(final):
        raise ValueError("duplicate node IDs in PN field")
    mass = mass_data[:, 1]
    if np.any(mass <= 0):
        raise ValueError("nodal masses must be positive")
    initial_q, final_q = initial[:, 4:], final[:, 4:]
    boundary_q, reaction_q = boundary[:, 4:], reaction[:, 4:]
    if final_q.shape[1] < 4:
        raise ValueError("PN field must contain at least the four low-order harmonics")
    balance = np.asarray([
        math.fsum(float(mass[i]) * (float(final_q[i, k]) - float(initial_q[i, k]))
                  + float(boundary_q[i, k]) - float(reaction_q[i, k])
                  for i in range(len(mass)))
        for k in range(final_q.shape[1])], dtype=float)
    max_balance = float(np.max(np.abs(balance)))
    if not math.isfinite(max_balance) or max_balance > TOLERANCE:
        raise ValueError(f"all-Q global balance {max_balance:.17g} exceeds {TOLERANCE:g}")
    physical = study_analysis.physical_low(final_q)
    rho, current = physical[:, 0], physical[:, 1:]
    margin = rho - np.linalg.norm(current, axis=1)
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    expected_config = {
        "coupling_mode": "pn-only", "coupling": "pn-only",
        "m1_coupling_enabled": False, "m1_closure_enabled": False,
        "pair_limiter_enabled": False, "cone_rejection_enabled": False,
        "model": "r2", "stabilization": "m1pn_mass_correction",
        "pn_target": "standard_B_mass_corrected",
        "mass_mode": "lumped", "mass_corrections": 0,
        "pn_mass_correction_applications": 1,
        "uses_pn_low_order_predictor": True,
        "pn_viscosity_scale": 1.0,
        "kv_gamma": 0.0, "gamma": 0.0,
        "mass_factorized": False,
    }
    mismatches = [key for key, value in expected_config.items() if config.get(key) != value]
    if mismatches:
        raise ValueError("run is not the requested PN-only standard-B configuration: " +
                         ", ".join(mismatches))
    if summary.get("status") != "completed":
        raise ValueError("PN-only summary does not report completed")
    if summary.get("mass_solve_calls") != 0:
        raise ValueError("PN-only summary must report mass_solve_calls=0")
    final_time = float(config.get("final_time", math.nan))
    summary_time = float(summary.get("time", math.nan))
    selected_dt = float(config.get("dt_selected", math.nan))
    steps = summary.get("steps")
    if not all(math.isfinite(v) and v > 0 for v in (final_time, summary_time, selected_dt)):
        raise ValueError("PN-only config/summary has non-positive or non-finite time controls")
    if abs(final_time - summary_time) > max(1e-13, 1e-12 * abs(final_time)):
        raise ValueError("PN-only summary time differs from configured final_time")
    if type(steps) is not int or steps < 1:
        raise ValueError("PN-only summary must contain a positive integer step count")
    if not np.isfinite(rho).all() or not np.isfinite(margin).all():
        raise ValueError("non-finite derived physical low moments")
    record = {
        "status": "passed", "nodes": len(mass), "Q": final_q.shape[1],
        "balance_tolerance": TOLERANCE,
        "maximum_global_Q_balance": max_balance,
        "balance_by_component": balance.tolist(),
        "minimum_density": float(np.min(rho)),
        "rho_min": float(np.min(rho)),
        "negative_density_nodes": int(np.count_nonzero(rho < 0.0)),
        "minimum_M1_low_moment_cone_margin": float(np.min(margin)),
        "minimum_cone_margin": float(np.min(margin)),
        "outside_M1_low_moment_cone_nodes": int(np.count_nonzero(margin < 0.0)),
        "outside_cone_nodes": int(np.count_nonzero(margin < 0.0)),
        "scope": "PN-only finite fields and all-Q conservation only; negative density and M1 low-moment cone violations are reported, not rejected; no positivity, M1 cone, or higher-moment realizability guarantee.",
        "sha256": {name: sha256(run / name) for name in REQUIRED},
        "configuration": {key: config.get(key) for key in (
            "model", "stabilization", "pn_target", "coupling", "coupling_mode", "mass_mode",
            "mass_corrections", "dimension", "elements_per_axis", "pn_order",
            "mesh_type", "final_time", "dt_selected", "beam_geometry") if key in config},
        "summary": {key: summary.get(key) for key in (
            "status", "time", "steps", "global_Q_balance", "solver_seconds",
            "mass_solve_calls", "m1_closure_evaluations",
            "pair_limiter_evaluations", "cone_rejections") if key in summary},
    }
    arrays = {"initial": initial, "final": final, "mass": mass,
              "coordinates": final[:, 1:4], "physical": physical}
    return record, arrays


def check(run: Path) -> tuple[dict, dict]:
    """Recheck a previously saved verification record without rewriting it."""
    run = run.resolve(strict=True)
    path = run / "verification.json"
    if not path.is_file():
        raise ValueError("PN verification.json is missing")
    saved = json.loads(path.read_text())
    if saved.get("status") != "passed" or "PN-only" not in saved.get("scope", ""):
        raise ValueError("verification record is not a passed PN-only check")
    for name in REQUIRED:
        if saved.get("sha256", {}).get(name) != sha256(run / name):
            raise ValueError(f"PN verification hash missing or mismatched for {name}")
    calculated, arrays = compute(run)
    if calculated["maximum_global_Q_balance"] > TOLERANCE:
        raise ValueError("PN all-Q balance failed on readback")
    # Compare diagnostics too, so stale/edited verification records cannot mask
    # a changed summary even when the raw hashes still match.
    for key in ("nodes", "Q", "minimum_density", "rho_min", "negative_density_nodes",
                "minimum_M1_low_moment_cone_margin", "minimum_cone_margin",
                "outside_M1_low_moment_cone_nodes", "outside_cone_nodes"):
        if saved.get(key) != calculated.get(key):
            raise ValueError(f"PN verification diagnostic mismatch: {key}")
    return saved, arrays


def verify(run: Path) -> dict:
    run = run.resolve(strict=True)
    record, _ = compute(run)
    record["created_utc"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat()
    (run / "verification.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"verify_pn_run.py: {exc}", file=sys.stderr)
        raise SystemExit(2)
