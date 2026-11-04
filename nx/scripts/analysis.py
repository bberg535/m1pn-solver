#!/usr/bin/env python3
"""Analyze frozen NX AFC campaigns without modifying solver outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(x) for x in row] for row in reader if row]
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(header) or not np.isfinite(data).all():
        raise ValueError(f"invalid/non-finite CSV data: {path}")
    return header, data


def physical_low(values: np.ndarray) -> np.ndarray:
    """Convert stored [u0,u1,u2,u3] harmonics to [rho,Jx,Jy,Jz]."""
    if values.shape[1] < 4:
        raise ValueError("field must contain at least four harmonics")
    scale = np.array([math.sqrt(4 * math.pi), math.sqrt(4 * math.pi / 3),
                      math.sqrt(4 * math.pi / 3), math.sqrt(4 * math.pi / 3)])
    return values[:, [0, 3, 1, 2]] * scale


def beam_exact(xyz: np.ndarray, time: float, geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    bg = float(geometry["background"])
    width_exp = float(geometry["width_exponent"])
    rho = np.full(len(xyz), bg)
    flux = np.zeros((len(xyz), 3))
    for beam in geometry["beams"]:
        direction = np.asarray(beam["direction"], dtype=float)
        center = np.asarray(beam["center"], dtype=float)
        amplitude = float(beam.get("amplitude", 1.0))
        value = amplitude * np.exp(width_exp * np.sum((xyz - center - time * direction) ** 2, axis=1))
        rho += value
        flux += value[:, None] * direction[None, :]
    return rho, flux


def metrics(error: np.ndarray, mass: np.ndarray, volume: float) -> dict:
    w = mass / volume
    return {"L1": float(np.sum(w[:, None] * np.abs(error)) / error.shape[1]),
            "L2": float(np.sqrt(np.sum(w[:, None] * error * error) / error.shape[1])),
            "Linf": float(np.max(np.abs(error)))}


def structured_axes(coords: np.ndarray, dimension: int) -> list[np.ndarray]:
    axes = [np.unique(coords[:, d]) for d in range(dimension)]
    expected = math.prod(len(a) for a in axes)
    if expected != len(coords):
        raise ValueError("mesh is not a complete structured tensor grid")
    return axes


def grid_tensor(coords: np.ndarray, values: np.ndarray, dimension: int):
    axes = structured_axes(coords, dimension)
    ids = [{float(v): i for i, v in enumerate(a)} for a in axes]
    tensor = np.empty(tuple(map(len, axes)) + (values.shape[1],), dtype=float)
    for c, v in zip(coords, values):
        tensor[tuple(ids[d][float(c[d])] for d in range(dimension))] = v
    return axes, tensor


def interpolate_structured(source_xyz: np.ndarray, source_values: np.ndarray,
                           target_xyz: np.ndarray, dimension: int) -> np.ndarray:
    """Multilinear interpolation on a Cartesian structured grid (no SciPy)."""
    axes, data = grid_tensor(source_xyz, source_values, dimension)
    out = np.zeros((len(target_xyz), source_values.shape[1]))
    for row, point in enumerate(target_xyz[:, :dimension]):
        choices = []
        for d, axis in enumerate(axes):
            if point[d] < axis[0] - 1e-12 or point[d] > axis[-1] + 1e-12:
                raise ValueError("comparison point outside interpolation grid")
            hi = int(np.searchsorted(axis, point[d], side="right"))
            if hi == 0:
                choices.append([(0, 1.)])
            elif hi >= len(axis):
                choices.append([(len(axis)-1, 1.)])
            else:
                lo = hi - 1
                alpha = (point[d] - axis[lo]) / (axis[hi] - axis[lo])
                if abs(alpha) < 1e-13:
                    choices.append([(lo, 1.)])
                elif abs(alpha - 1) < 1e-13:
                    choices.append([(hi, 1.)])
                else:
                    choices.append([(lo, 1-alpha), (hi, alpha)])
        for combo in np.ndindex(*(len(c) for c in choices)):
            index, weight = [], 1.
            for d, j in enumerate(combo):
                idx, w = choices[d][j]
                index.append(idx)
                weight *= w
            out[row] += weight * data[tuple(index)]
    return out


def volume_of(coords: np.ndarray, dimension: int) -> float:
    return float(np.prod([np.ptp(np.unique(coords[:, d])) for d in range(dimension)]))


def coordinate_permutation(reference_xyz: np.ndarray, other_xyz: np.ndarray) -> np.ndarray:
    """Return rows of other_xyz matching reference_xyz, rejecting mismatched grids."""
    if reference_xyz.shape != other_xyz.shape:
        raise ValueError("comparison meshes have different coordinate shapes")
    # Solver meshes are generated from the same dyadic coordinates; rounding only
    # absorbs harmless CSV serialization differences while checking collisions.
    def key(row):
        return tuple(round(float(v), 13) for v in row)
    lookup = {key(x): i for i, x in enumerate(other_xyz)}
    if len(lookup) != len(other_xyz):
        raise ValueError("duplicate coordinates in comparison mesh")
    try:
        permutation = np.asarray([lookup[key(x)] for x in reference_xyz], dtype=int)
    except KeyError as e:
        raise ValueError("comparison coordinate sets differ") from e
    if len(set(permutation.tolist())) != len(permutation):
        raise ValueError("comparison coordinate sets differ")
    return permutation


def verify_saved_inputs(out: Path) -> dict:
    """Validate verifier's recorded digests against current raw run inputs."""
    verification_path = out / "verification.json"
    verification = json.loads(verification_path.read_text())
    if verification.get("status") != "passed":
        raise ValueError("verification.json does not report passed")
    recorded = verification.get("sha256", {})
    required = ["initial.csv", "final.csv", "nodal_mass.csv", "config.json", "summary.json",
                "boundary_integral.csv", "reaction_integral.csv"]
    for name in required:
        path = out / name
        if not path.is_file() or recorded.get(name) != sha256(path):
            raise ValueError(f"verification hash missing or mismatched for {name}")
    if (out / "material.csv").is_file() and recorded.get("material.csv") != sha256(out / "material.csv"):
        raise ValueError("verification hash missing or mismatched for material.csv")
    return verification


def task_result(campaign: Path, task: dict, status: dict) -> dict:
    out = (campaign / status.get("output_dir", task.get("output_dir", ""))).resolve()
    if campaign not in out.parents:
        raise ValueError("task output path escapes campaign")
    if not status.get("validated", False):
        raise ValueError("saved field verifier did not pass")
    verification = verify_saved_inputs(out)
    cfg = json.loads((out / "config.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    expected_grid = task.get("grid_n", task.get("grid"))
    expected_order = task.get("angular_order", task.get("order"))
    expected_case = task.get("case")
    method = task.get("method", {"mass_mode":"lumped", "mass_corrections":0,
                                 "stabilization":"m1pn_mass_correction", "kv_gamma":0.,
                                 "biharmonic_gamma":0., "pn_viscosity_scale":1.})
    checks = {
        "mass_mode": cfg.get("mass_mode") == method["mass_mode"],
        "mass_corrections": cfg.get("mass_corrections") == method["mass_corrections"],
        "stabilization": cfg.get("stabilization") == method["stabilization"],
        "kv_gamma": abs(float(cfg.get("kv_gamma", -1)) - method["kv_gamma"]) <= 1e-14,
        "mass_factorized": cfg.get("mass_factorized") is False,
        "mass_solve_calls": summary.get("mass_solve_calls") == 0,
        "pn_low_order_predictor": cfg.get("uses_pn_low_order_predictor") is True,
        "pn_mass_correction_applications": cfg.get("pn_mass_correction_applications") == 1,
        "pn_viscosity_scale": abs(float(cfg.get("pn_viscosity_scale", -1)) -
                                   method["pn_viscosity_scale"]) <= 1e-14,
        "biharmonic_gamma": abs(float(cfg.get("gamma", -1)) -
                                method["biharmonic_gamma"]) <= 1e-14,
        "no_biharmonic_work": abs(float(summary.get("biharmonic_seconds", -1))) <= 1e-15,
        "predictor_timing": math.isfinite(float(summary.get("pn_predictor_seconds", -1))) and
                            float(summary.get("pn_predictor_seconds", -1)) >= 0,
        "mass_correction_timing": math.isfinite(float(summary.get("pn_mass_correction_seconds", -1))) and
                                  float(summary.get("pn_mass_correction_seconds", -1)) >= 0,
        "summary_status": summary.get("status") == "completed",
        "rk": cfg.get("rk") == 3,
        "case": expected_case is None or cfg.get("case") == expected_case,
        "dimension": int(cfg.get("dimension", -1)) == int(task.get("dimension", -2)),
        "grid": int(cfg.get("elements_per_axis", -1)) == int(expected_grid),
        "angular_order": int(cfg.get("pn_order", -1)) == int(expected_order),
        "final_time": abs(float(cfg.get("final_time", -1)) - float(task["time"])) <=
                      max(1e-13, 1e-12 * float(task["time"])),
        "selected_dt": abs(float(cfg.get("dt_selected", -1)) - float(task["dt"])) <=
                       max(1e-14, 1e-12 * float(task["dt"])),
        "summary_time": abs(float(summary.get("time", -1)) - float(task["time"])) <=
                        max(1e-13, 1e-12 * float(task["time"])),
    }
    if "ghost_edges" in cfg:
        checks["no_ghost_edges"] = cfg["ghost_edges"] == 0
    if not all(checks.values()):
        bad = ", ".join(k for k, ok in checks.items() if not ok)
        raise ValueError(f"run metadata violates requested controls: {bad}")
    header, final = read_csv(out / "final.csv")
    _, masses = read_csv(out / "nodal_mass.csv")
    if header[:4] != ["node", "x", "y", "z"]:
        raise ValueError("unexpected final.csv columns")
    if len(masses) != len(final) or not np.array_equal(masses[:, 0], final[:, 0]):
        raise ValueError("nodal mass and final field node IDs differ")
    xyz, mass = final[:, 1:4], masses[:, 1]
    dim = int(task.get("dimension", cfg.get("dimension", 2)))
    physical = physical_low(final[:, 4:])
    volume = volume_of(xyz, dim)
    result = {"id": task["id"], "label": task.get("label", str(task["id"])),
              "family": task.get("family"), "grid": task.get("grid_n", task.get("grid")),
              "order": task.get("angular_order", task.get("order")),
              "kind": task.get("kind", "spatial"), "time": float(task["time"]),
              "dt": float(task["dt"]), "nodes": len(final), "domain_volume": volume,
              "run_output": str(out.relative_to(campaign)), "field_sha256": sha256(out / "final.csv"),
              "config_sha256": sha256(out / "config.json"),
              "summary_sha256": sha256(out / "summary.json"),
              "verification_sha256": sha256(out / "verification.json"),
              "verified_input_hashes": verification["sha256"],
              "coordinates": xyz, "mass": mass, "physical": physical,
              "rho_min": float(physical[:, 0].min()), "rho_max": float(physical[:, 0].max())}
    geom = cfg.get("beam_geometry")
    if geom:
        exact_rho, exact_flux = beam_exact(xyz, float(task["time"]), geom)
        exact = np.column_stack((exact_rho, exact_flux))
        result["reference_kind"] = "analytic_kinetic_transport_moments"
        result["exact_error"] = metrics(physical - exact, mass, volume)
        result["exact_error"]["density_L1"] = float(np.sum(mass * np.abs(physical[:, 0] - exact[:, 0])) / volume)
        result["exact_error"]["density_L2"] = float(np.sqrt(np.sum(mass * (physical[:, 0] - exact[:, 0])**2) / volume))
        result["exact_error"]["density_Linf"] = float(np.max(np.abs(physical[:, 0] - exact[:, 0])))
        result["exact"] = exact
    if task.get("family") == "lattice":
        mh, mat = read_csv(out / "material.csv")
        if mh != ["node", "sigma_a", "sigma_s", "source_density"] or len(mat) != len(mass):
            raise ValueError("unexpected lattice material.csv layout")
        indicators = {"source_integral": float(np.sum(mass * mat[:, 3])),
                      "absorption_integral": float(np.sum(mass * mat[:, 1])),
                      "scattering_integral": float(np.sum(mass * mat[:, 2])),
                      "volume": float(np.sum(mass))}
        expected = {"source_integral": 1., "absorption_integral": 110.,
                    "scattering_integral": 38., "volume": 49.}
        indicators["valid"] = all(abs(indicators[k] - v) <= 1e-8 * max(1., abs(v))
                                   for k, v in expected.items())
        if not indicators["valid"]:
            raise ValueError(f"lattice material integral checks failed: {indicators}")
        result["lattice_indicators"] = indicators
    return result


def jsonable_result(r: dict) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in r.items() if k not in ("coordinates", "mass", "physical", "exact")}


def analyze(campaign: Path, allow_incomplete: bool = False) -> int:
    campaign = campaign.resolve(strict=True)
    manifest = json.loads((campaign / "manifest.json").read_text())
    tasks_path = campaign / manifest.get("tasks", "tasks.json")
    if not tasks_path.is_file() or manifest.get("tasks_sha256") != sha256(tasks_path):
        raise ValueError("frozen task manifest SHA256 missing or mismatched")
    tasks = json.loads(tasks_path.read_text())
    rows, data, failures, missing = [], {}, [], []
    execution_failed = False
    for task in tasks:
        sid = str(task["id"])
        status_path = campaign / "status" / f"{sid}.json"
        if not status_path.is_file():
            missing.append({"id": task["id"], "label": task.get("label"), "reason": "missing status"})
            execution_failed = True
            continue
        status = json.loads(status_path.read_text())
        if status.get("status") != "completed":
            execution_failed = True
            failures.append({"id": task["id"], "label": task.get("label"),
                             "status": status.get("status"), "error": status.get("error") or
                             status.get("diagnostics", {}).get("validation_error")})
            continue
        try:
            result = task_result(campaign, task, status)
            rows.append(jsonable_result(result))
            data[int(task["id"])] = result
        except Exception as e:
            failures.append({"id": task["id"], "label": task.get("label"),
                             "status": "analysis_failed", "error": str(e)})
    # Grid and angular comparisons use only same-case, same-time spatial tasks.
    comparisons = []
    spatial = [r for r in data.values() if r["kind"] == "spatial"]
    groups: dict[tuple, list[dict]] = {}
    for r in spatial:
        groups.setdefault((r["family"], r["time"]), []).append(r)
    for (family, time), group in groups.items():
        for coarse in group:
            for fine in group:
                if coarse["order"] != fine["order"] or coarse["grid"] >= fine["grid"]:
                    continue
                try:
                    fine_at_coarse = interpolate_structured(fine["coordinates"], fine["physical"],
                                                              coarse["coordinates"],
                                                              3 if family == "three_d" else 2)
                    err = metrics(coarse["physical"] - fine_at_coarse,
                                  coarse["mass"], coarse["domain_volume"])
                    comparisons.append({"family": family, "time": time, "type": "grid_difference",
                                        "coarse_grid": coarse["grid"], "fine_grid": fine["grid"],
                                        "order": coarse["order"], "difference": err,
                                        "interpretation": "self difference; lattice is not an exact solution"})
                except Exception as e:
                    failures.append({"status": "comparison_failed", "family": family,
                                     "type": "grid_difference", "error": str(e)})
        grids = sorted({r["grid"] for r in group})
        for grid in grids:
            same = [r for r in group if r["grid"] == grid]
            if not same:
                continue
            reference = max(same, key=lambda r: r["order"])
            for r in same:
                if r["order"] == reference["order"]:
                    continue
                perm = coordinate_permutation(r["coordinates"], reference["coordinates"])
                comparisons.append({"family": family, "time": time, "type": "angular_difference",
                                    "grid": grid, "order": r["order"],
                                    "reference_order": reference["order"],
                                    "difference": metrics(r["physical"] - reference["physical"][perm],
                                                          r["mass"], r["domain_volume"]),
                                    "interpretation": "same-grid PN self difference"})
        if family != "lattice":
            for order in sorted({r["order"] for r in group}):
                seq = sorted([r for r in group if r["order"] == order and "exact_error" in r],
                             key=lambda r: r["grid"])
                for coarse, fine in zip(seq, seq[1:]):
                    if fine["grid"] != 2 * coarse["grid"]:
                        continue
                    ratio = coarse["exact_error"]["L2"] / fine["exact_error"]["L2"] \
                        if fine["exact_error"]["L2"] > 0 else None
                    p = (math.log(ratio) / math.log(fine["grid"] / coarse["grid"])
                         if ratio is not None and ratio > 0 else None)
                    comparisons.append({"family": family, "time": time,
                                        "type": "transport_reference_error_order",
                                        "order": order, "coarse_grid": coarse["grid"],
                                        "fine_grid": fine["grid"], "coarse_L2": coarse["exact_error"]["L2"],
                                        "fine_L2": fine["exact_error"]["L2"], "observed_order": p})
    # Temporal error compared at the same finest-grid/highest-order coordinates.
    for r in data.values():
        if r["kind"] != "temporal":
            continue
        peers = [x for x in data.values() if x["family"] == r["family"] and
                 x["kind"] == "spatial" and x["grid"] == r["grid"] and x["order"] == r["order"]]
        if not peers:
            continue
        base = peers[0]
        perm = coordinate_permutation(base["coordinates"], r["coordinates"])
        dt_err = metrics(r["physical"][perm] - base["physical"], base["mass"], base["domain_volume"])
        lower_grids = [c["coarse_grid"] for c in comparisons if c["type"] == "grid_difference" and
                       c["family"] == r["family"] and c.get("order") == r["order"] and
                       c.get("fine_grid") == r["grid"]]
        nearest_grid = max(lower_grids) if lower_grids else None
        spatial_rows = [c for c in comparisons if c["type"] == "grid_difference" and
                        c["family"] == r["family"] and c.get("order") == r["order"] and
                        c.get("fine_grid") == r["grid"] and c.get("coarse_grid") == nearest_grid]
        spatial_reference = spatial_rows[0]["difference"]["L2"] if spatial_rows else 0.
        tolerance_fraction = float(manifest.get("plan", {}).get("execution", {}).get(
            "temporal_error_fraction", .1))
        threshold = max(1e-10, tolerance_fraction * spatial_reference)
        contaminated = dt_err["L2"] > threshold
        comparisons.append({"family": r["family"], "type": "temporal_difference",
                            "grid": r["grid"], "order": r["order"], "dt": r["dt"],
                            "dt_full": base["dt"], "dt_half": r["dt"], "difference": dt_err,
                            "spatial_reference_L2": spatial_reference,
                            "spatial_reference_coarse_grid": nearest_grid,
                            "temporal_error_fraction": tolerance_fraction,
                            "threshold_L2": threshold, "temporal_contamination": contaminated})
    # Estimate observed order from consecutive grid self-difference ratios.
    for family, time in groups:
        for order in sorted({r["order"] for r in groups[(family, time)]}):
            seq = sorted([c for c in comparisons if c["type"] == "grid_difference" and
                          c["family"] == family and c["time"] == time and c["order"] == order],
                         key=lambda c: c["coarse_grid"])
            seq = [c for c in seq if c["fine_grid"] == 2 * c["coarse_grid"]]
            for a, b in zip(seq, seq[1:]):
                ratio = a["difference"]["L2"] / b["difference"]["L2"] if b["difference"]["L2"] > 0 else None
                a["observed_order"] = (math.log(ratio) / math.log(b["coarse_grid"] / a["coarse_grid"])
                                        if ratio and ratio > 0 else None)
    execution_complete = not execution_failed
    analysis_complete = len(rows) == len(tasks)
    temporal_contaminated = any(c.get("temporal_contamination", False) for c in comparisons
                                if c["type"] == "temporal_difference")
    quality_accepted = execution_complete and analysis_complete and not temporal_contaminated
    incomplete = not analysis_complete
    output = {"schema_version": 1, "campaign": campaign.name,
              "manifest_sha256": sha256(campaign / "manifest.json"),
              "tasks_sha256": sha256(campaign / "tasks.json"),
              "plan_sha256": manifest.get("plan_sha256"),
              "analysis_script_sha256": sha256(Path(__file__).resolve()),
              "analysis_tests_sha256": sha256(Path(__file__).resolve().parents[1] / "tests" / "test_analysis.py"),
              "source_provenance": manifest.get("provenance"),
              "expected_tasks": len(tasks), "analyzed_tasks": len(rows),
              "execution_complete": execution_complete, "analysis_complete": analysis_complete,
              "quality_accepted": quality_accepted,
              "temporal_contamination": temporal_contaminated,
              "incomplete": incomplete, "missing": missing, "failures": failures,
              "tasks": rows, "comparisons": comparisons,
              "caveats": ["Beam reference is analytic kinetic free-transport moments, not the finite-N PN/R2 solution.",
                          "Beam reference errors include angular, model, boundary, and limiter errors; observed h-rates do not isolate spatial discretization error.",
                          "Plateauing or non-monotone errors do not support a generic convergence claim.",
                          "Lattice has no exact solution; differences are self differences.",
                          "Short smoke-profile runs are functional smoke checks, not performance or accuracy evidence.",
                          "Temporal contamination rejects quality acceptance; it does not mean solver execution failed.",
                          "Saved-field digests are rechecked against verification.json before analysis."]}
    (campaign / "analysis.csv").write_text(csv_text(rows))
    (campaign / "comparisons.csv").write_text(comparisons_csv(comparisons))
    (campaign / "analysis_summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    make_plots(campaign, data, comparisons)
    print(f"Analyzed {len(rows)}/{len(tasks)} tasks; execution_complete={execution_complete}; "
          f"quality_accepted={quality_accepted}")
    if incomplete and not allow_incomplete:
        return 1
    if temporal_contaminated:
        return 3
    return 0


def csv_text(rows: list[dict]) -> str:
    if not rows:
        return "id,label,family,grid,order,kind,time,dt,exact_error\n"
    keys = ["id", "label", "family", "grid", "order", "kind", "reference_kind", "time", "dt", "nodes",
            "domain_volume", "rho_min", "rho_max", "exact_error", "lattice_indicators", "run_output",
            "field_sha256"]
    from io import StringIO
    s = StringIO()
    w = csv.DictWriter(s, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow({**row, "exact_error": json.dumps(row.get("exact_error")),
                    "lattice_indicators": json.dumps(row.get("lattice_indicators"))})
    return s.getvalue()


def comparisons_csv(comparisons: list[dict]) -> str:
    from io import StringIO
    keys = ["family", "time", "type", "grid", "coarse_grid", "fine_grid", "order",
            "reference_order", "observed_order", "coarse_L2", "fine_L2", "L1", "L2", "Linf",
            "spatial_reference_L2", "spatial_reference_coarse_grid", "temporal_error_fraction",
            "threshold_L2", "temporal_contamination", "interpretation"]
    s = StringIO()
    writer = csv.DictWriter(s, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for item in comparisons:
        row = dict(item)
        difference = row.pop("difference", {})
        row.update({name: difference.get(name) for name in ("L1", "L2", "Linf")})
        writer.writerow(row)
    return s.getvalue()


def make_plots(campaign: Path, data: dict, campaign_comparisons: list[dict]) -> None:
    try:
        mplconfig = campaign / "tmp" / "mplconfig"
        mplconfig.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mplconfig)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_dir = campaign / "plots"
    plot_dir.mkdir(exist_ok=True)
    families = sorted({r["family"] for r in data.values() if r["kind"] == "spatial"})
    for family in families:
        fig, ax = plt.subplots()
        orders = sorted({r["order"] for r in data.values() if r["family"] == family and r["kind"] == "spatial"})
        for order in orders:
            if family != "lattice":
                seq = sorted([r for r in data.values() if r["family"] == family and r["order"] == order and
                              r["kind"] == "spatial" and "exact_error" in r], key=lambda r: r["grid"])
                xx = [r["grid"] for r in seq]
                yy = [r["exact_error"]["L2"] for r in seq]
                ylabel, title = ("Physical moment error (RMS)",
                                 f"{family}: transport-reference error by grid")
            else:
                differences = [c for c in campaign_comparisons if c["family"] == family and
                               c["type"] == "grid_difference" and c["order"] == order]
                if differences:
                    finest = max(c["fine_grid"] for c in differences)
                    differences = sorted([c for c in differences if c["fine_grid"] == finest],
                                         key=lambda c: c["coarse_grid"])
                xx = [c["coarse_grid"] for c in differences]
                yy = [c["difference"]["L2"] for c in differences]
                ylabel, title = "Physical moment self difference (RMS)", f"{family}: self difference by grid"
            positive = [(x, y) for x, y in zip(xx, yy) if x > 0 and y > 0]
            if positive:
                ax.loglog([x for x, _ in positive], [y for _, y in positive], "o-", label=f"P{order}")
        ax.set(xlabel="grid elements per axis", ylabel=ylabel, title=title)
        ax.grid(True, which="both", alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{family}_convergence.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots()
        grids = sorted({r["grid"] for r in data.values() if r["family"] == family and r["kind"] == "spatial"})
        for grid in grids:
            if family == "lattice":
                seq = sorted([c for c in campaign_comparisons if c["family"] == family and
                              c["type"] == "angular_difference" and c["grid"] == grid],
                             key=lambda c: c["order"])
                points = [(c["order"], c["difference"]["L2"]) for c in seq]
            else:
                seq = sorted([r for r in data.values() if r["family"] == family and
                              r["kind"] == "spatial" and r["grid"] == grid and "exact_error" in r],
                             key=lambda r: r["order"])
                points = [(r["order"], r["exact_error"]["L2"]) for r in seq]
            points = [(n, e) for n, e in points if e > 0]
            if points:
                ax.semilogy([n for n, _ in points], [e for _, e in points], 'o-', label=f'n={grid}')
        ax.set(xlabel='PN order N', ylabel='Physical moment self difference (RMS)' if family=='lattice'
               else 'Physical moment error (RMS)', title=f'{family}: '+
               ('difference to highest N on same grid' if family=='lattice' else 'kinetic-reference error by PN order'))
        ax.grid(True, which='both', alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f'{family}_angular_convergence.png', dpi=160, bbox_inches='tight')
        plt.close(fig)
        field_runs = [r for r in data.values() if r["family"] == family and r["kind"] == "spatial"]
        if field_runs:
            finest = max(field_runs, key=lambda r: (r["grid"], r["order"]))
            dim = 3 if family == "three_d" else 2
            axes, tensor = grid_tensor(finest["coordinates"], finest["physical"][:, :1], dim)
            fig, ax = plt.subplots()
            if dim == 2:
                image = ax.pcolormesh(axes[0], axes[1], tensor[:, :, 0].T, shading="auto")
                ax.set(xlabel="x", ylabel="y", title=f"{family}: final density")
            else:
                iz = len(axes[2]) // 2
                image = ax.pcolormesh(axes[0], axes[1], tensor[:, :, iz, 0].T, shading="auto")
                ax.set(xlabel="x", ylabel="y", title=f"{family}: final density, middle z slice")
            fig.colorbar(image, ax=ax, label="density")
            ax.set_xlim(axes[0][0], axes[0][-1])
            ax.set_ylim(axes[1][0], axes[1][-1])
            fig.tight_layout()
            fig.savefig(plot_dir / f"{family}_density.png", dpi=160, bbox_inches="tight")
            plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    ns = p.parse_args(argv)
    try:
        return analyze(ns.campaign, ns.allow_incomplete)
    except Exception as e:
        print(f"analysis.py: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
