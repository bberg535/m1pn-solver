#!/usr/bin/env python3
"""Compare a verified PN-only run with its matched coupled M1PN-B run."""
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
import verify_pn_run


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def find_coupled_task(run: Path) -> tuple[Path, dict, dict]:
    run = run.resolve(strict=True)
    for parent in run.parents:
        manifest_path = parent / "manifest.json"
        tasks_path = parent / "tasks.json"
        if not manifest_path.is_file() or not tasks_path.is_file():
            continue
        manifest = read_json(manifest_path)
        if study_analysis.sha256(tasks_path) != manifest.get("tasks_sha256"):
            raise ValueError("Coupled campaign frozen task manifest SHA256 mismatch")
        for task in json.loads(tasks_path.read_text()):
            output = (parent / task["output_dir"]).resolve()
            if output == run:
                status_path = parent / "status" / f"{task['id']}.json"
                if not status_path.is_file():
                    raise ValueError("Coupled run has no task status record")
                return parent, task, read_json(status_path)
    raise ValueError("Coupled run is not inside a frozen NX campaign task directory")


def require_equal(label: str, left, right) -> None:
    if left != right:
        raise ValueError(f"PN-only/coupled comparison mismatch for {label}: {left!r} != {right!r}")


def metric_row(method: str, target: str, error: np.ndarray,
               mass: np.ndarray, volume: float, denominator: float,
               meta: dict) -> dict:
    l1 = float(np.sum(mass * np.abs(error)) / volume)
    return {**meta, "method": method, "target": target,
            "density_L1": l1,
            "density_L2": float(np.sqrt(np.sum(mass * error**2) / volume)),
            "density_Linf": float(np.max(np.abs(error))),
            "relative_density_L1": l1 / denominator if denominator > 0 else math.nan,
            "relative_density_L1_denominator": denominator}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pn-run", required=True, type=Path)
    parser.add_argument("--coupled-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    pn_run = args.pn_run.resolve(strict=True)
    coupled_run = args.coupled_run.resolve(strict=True)
    output = args.output.resolve()
    if ROOT not in pn_run.parents:
        raise ValueError("PN-only run must be inside the experiment root")
    if NX not in coupled_run.parents:
        raise ValueError("Coupled run must be inside the experiment NX tree")
    if ROOT not in output.parents:
        raise ValueError(f"Output must be inside experiment root {ROOT}")
    if pn_run in output.parents or coupled_run in output.parents or output in pn_run.parents or output in coupled_run.parents:
        raise ValueError("Output directory must be separate from both run folders")

    pn_verification, pn_arrays = verify_pn_run.check(pn_run)
    pn_config = read_json(pn_run / "config.json")
    pn_summary = read_json(pn_run / "summary.json")
    campaign, task, status = find_coupled_task(coupled_run)
    coupled = study_analysis.task_result(campaign, task, status)
    coupled_config = read_json(coupled_run / "config.json")
    coupled_summary = read_json(coupled_run / "summary.json")

    # Explicitly bind the requested run pair: both start from the same saved
    # nodal state and lumped FE masses, on the same mesh and time schedule.
    same_files = {}
    for name in ("initial.csv", "nodal_mass.csv"):
        pn_hash = study_analysis.sha256(pn_run / name)
        coupled_hash = study_analysis.sha256(coupled_run / name)
        require_equal(f"{name} SHA256", pn_hash, coupled_hash)
        same_files[name] = pn_hash
    require_equal("dimension", pn_config.get("dimension"), coupled_config.get("dimension"))
    require_equal("mesh_type", pn_config.get("mesh_type"), coupled_config.get("mesh_type"))
    require_equal("grid", pn_config.get("elements_per_axis"), coupled_config.get("elements_per_axis"))
    require_equal("PN order", pn_config.get("pn_order"), coupled_config.get("pn_order"))
    require_equal("final time", pn_config.get("final_time"), coupled_config.get("final_time"))
    require_equal("selected dt", pn_config.get("dt_selected"), coupled_config.get("dt_selected"))
    require_equal("completed steps", pn_summary.get("steps"), coupled_summary.get("steps"))
    require_equal("beam geometry", pn_config.get("beam_geometry"), coupled_config.get("beam_geometry"))
    require_equal("case", pn_config.get("case"), coupled_config.get("case"))
    require_equal("inflow", pn_config.get("inflow"), coupled_config.get("inflow"))
    if pn_config.get("coupling_mode") != "pn-only" or pn_config.get("coupling") != "pn-only":
        raise ValueError("PN run config does not identify coupling_mode/coupling as pn-only")
    if pn_config.get("m1_coupling_enabled") is not False or \
            pn_config.get("m1_closure_enabled") is not False or \
            pn_config.get("pair_limiter_enabled") is not False or \
            pn_config.get("cone_rejection_enabled") is not False:
        raise ValueError("PN-only run metadata reports an enabled M1 coupling, limiter, or cone rejection")
    if pn_config.get("model") != "r2" or \
            pn_config.get("stabilization") != "m1pn_mass_correction":
        raise ValueError("PN-only run is not standard B under R2")
    require_equal("PN order P3", pn_config.get("pn_order"), 3)
    require_equal("128x128 grid", pn_config.get("elements_per_axis"), 128)
    if task.get("family") != "crossing" or task.get("kind") != "spatial":
        raise ValueError("Coupled comparator must be a spatial Crossing run")
    require_equal("coupled order P3", task.get("order", task.get("angular_order")), 3)
    require_equal("coupled 128x128 grid", task.get("grid", task.get("grid_n")), 128)
    if not np.array_equal(pn_arrays["coordinates"], coupled["coordinates"]):
        raise ValueError("PN-only and coupled final coordinates differ")
    if not np.array_equal(pn_arrays["mass"], coupled["mass"]):
        raise ValueError("PN-only and coupled nodal mass arrays differ despite matching hashes")

    geometry = pn_config.get("beam_geometry")
    if not geometry:
        raise ValueError("PN-only config lacks analytic kinetic beam geometry")
    time = float(pn_config["final_time"])
    exact_rho, _ = study_analysis.beam_exact(pn_arrays["coordinates"], time, geometry)
    rho_pn = pn_arrays["physical"][:, 0]
    rho_coupled = coupled["physical"][:, 0]
    mass = pn_arrays["mass"]
    volume = study_analysis.volume_of(pn_arrays["coordinates"], 2)
    reference_denominator = float(np.sum(mass * np.abs(exact_rho)) / volume)
    coupled_denominator = float(np.sum(mass * np.abs(rho_coupled)) / volume)
    meta = {"grid": 128, "pn_order": 3, "time": time,
            "dt": float(pn_config["dt_selected"]), "steps": int(pn_summary["steps"]),
            "domain_volume": volume}
    metric_rows = [
        metric_row("PN-only B", "kinetic_reference", rho_pn - exact_rho,
                   mass, volume, reference_denominator, meta),
        metric_row("coupled M1PN B", "kinetic_reference", rho_coupled - exact_rho,
                   mass, volume, reference_denominator, meta),
        metric_row("PN-only B minus coupled M1PN B", "method_difference",
                   rho_pn - rho_coupled, mass, volume, coupled_denominator, meta),
    ]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "pn_comparison_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140,
                         "savefig.dpi": 200, "pdf.fonttype": 42})
    coords = pn_arrays["coordinates"]
    density_fields = [exact_rho, rho_pn, rho_coupled]
    rho_min = min(float(np.min(v)) for v in density_fields)
    rho_max = max(float(np.max(v)) for v in density_fields)
    density_norm = Normalize(vmin=min(0., rho_min), vmax=rho_max)
    density_cmap = "RdBu_r" if rho_min < 0 else "viridis"
    density_titles = ["Kinetische Referenz", "PN-only B", "gekoppeltes M1PN B"]
    density_minima = [float(np.min(v)) for v in density_fields]
    density_label = ("ρ (gemeinsame Skala; negative Werte enthalten)"
                     if rho_min < 0 else "ρ (gemeinsame Skala)")

    error_fields = [rho_pn - exact_rho, rho_coupled - exact_rho, rho_pn - rho_coupled]
    error_max = max(float(np.max(np.abs(v))) for v in error_fields)
    error_max = max(error_max, np.finfo(float).tiny)
    error_norm = TwoSlopeNorm(vmin=-error_max, vcenter=0., vmax=error_max)
    error_titles = ["PN-only B − kinetische Referenz",
                    "gekoppeltes M1PN B − kinetische Referenz",
                    "PN-only B − gekoppeltes M1PN B"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
    for col, (values, title, minimum) in enumerate(zip(density_fields, density_titles, density_minima)):
        _, tensor = study_analysis.grid_tensor(coords, values[:, None], 2)
        mesh = axes[0, col].pcolormesh(np.unique(coords[:, 0]), np.unique(coords[:, 1]),
                                       tensor[:, :, 0].T, shading="auto", cmap=density_cmap,
                                       norm=density_norm)
        axes[0, col].set(title=f"{title}\nmin ρ={minimum:.3e}", xlabel="x", ylabel="y",
                         xlim=(0, 1), ylim=(0, 1), aspect="equal")
        fig.colorbar(mesh, ax=axes[0, col], shrink=.84, label=density_label)
    for col, (values, title) in enumerate(zip(error_fields, error_titles)):
        _, tensor = study_analysis.grid_tensor(coords, values[:, None], 2)
        mesh = axes[1, col].pcolormesh(np.unique(coords[:, 0]), np.unique(coords[:, 1]),
                                       tensor[:, :, 0].T, shading="auto", cmap="RdBu_r",
                                       norm=error_norm)
        axes[1, col].set(title=title, xlabel="x", ylabel="y", xlim=(0, 1), ylim=(0, 1),
                         aspect="equal")
        fig.colorbar(mesh, ax=axes[1, col], shrink=.84, label="signierter Dichtefehler")
    fig.suptitle("PN-only B und gekoppeltes M1PN B · Crossing P3 auf 128²\n"
                 "Kinetische Referenzfehler enthalten endliche-PN-Modellfehler")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"pn_density_and_errors.{ext}", bbox_inches="tight")
    plt.close(fig)

    # Midline uses the same tensorized y=0.5 row for all fields and references.
    axes_grid, rho_tensor = study_analysis.grid_tensor(coords,
        np.column_stack((exact_rho, rho_pn, rho_coupled)), 2)
    yidx = int(np.argmin(np.abs(axes_grid[1] - .5)))
    if abs(float(axes_grid[1][yidx]) - .5) > 1e-12:
        raise ValueError("128² comparison grid has no y=0.5 node")
    line_rows = []
    for j, x in enumerate(axes_grid[0]):
        line_rows.append({"x": float(x), "y": float(axes_grid[1][yidx]),
                          "rho_kinetic_reference": float(rho_tensor[j, yidx, 0]),
                          "rho_pn_only": float(rho_tensor[j, yidx, 1]),
                          "rho_coupled_m1pn_b": float(rho_tensor[j, yidx, 2]),
                          "pn_only_minus_reference": float(rho_tensor[j, yidx, 1] - rho_tensor[j, yidx, 0]),
                          "coupled_minus_reference": float(rho_tensor[j, yidx, 2] - rho_tensor[j, yidx, 0]),
                          "pn_only_minus_coupled": float(rho_tensor[j, yidx, 1] - rho_tensor[j, yidx, 2])})
    with (output / "pn_midline_y_0_5.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(line_rows[0]))
        writer.writeheader()
        writer.writerows(line_rows)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True, layout="constrained")
    xline = axes_grid[0]
    axes[0].plot(xline, rho_tensor[:, yidx, 0], "k--", label="Kinetische Referenz")
    axes[0].plot(xline, rho_tensor[:, yidx, 1], label="PN-only B")
    axes[0].plot(xline, rho_tensor[:, yidx, 2], label="gekoppeltes M1PN B")
    axes[1].plot(xline, rho_tensor[:, yidx, 1] - rho_tensor[:, yidx, 0],
                 label="PN-only B − Referenz")
    axes[1].plot(xline, rho_tensor[:, yidx, 2] - rho_tensor[:, yidx, 0],
                 label="gekoppeltes M1PN B − Referenz")
    axes[1].plot(xline, rho_tensor[:, yidx, 1] - rho_tensor[:, yidx, 2],
                 label="PN-only B − M1PN B")
    axes[0].set(ylabel="ρ(x, 0.5)", title="Dichte am Mittelschnitt")
    axes[1].set(xlabel="x", ylabel="signierter Fehler", title="Signierte Differenzen")
    axes[0].legend(fontsize=8, ncols=2)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle("PN-only / gekoppeltes M1PN B · y=0.5")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"pn_midline_y_0_5.{ext}", bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "comparison": "PN-only standard B versus coupled M1PN standard B",
        "pn_run": str(pn_run), "coupled_run": str(coupled_run),
        "coupled_campaign": str(campaign), "output": str(output),
        "metadata_match": {"same_initial_csv_sha256": same_files["initial.csv"],
                           "same_nodal_mass_csv_sha256": same_files["nodal_mass.csv"],
                           "same_dimension": pn_config["dimension"],
                           "same_mesh_type": pn_config["mesh_type"],
                           "same_grid": pn_config["elements_per_axis"],
                           "same_pn_order": pn_config["pn_order"],
                           "same_final_time": pn_config["final_time"],
                           "same_dt": pn_config["dt_selected"],
                           "same_steps": pn_summary["steps"],
                           "same_beam_geometry": True},
        "pn_config": pn_verification["configuration"],
        "pn_verification": {"sha256": pn_verification["sha256"],
                             "verification_json_sha256": study_analysis.sha256(pn_run / "verification.json"),
                             "maximum_global_Q_balance": pn_verification["maximum_global_Q_balance"],
                             "scope": pn_verification["scope"]},
        "coupled_provenance": {"campaign_manifest_sha256": study_analysis.sha256(campaign / "manifest.json"),
                               "tasks_sha256": study_analysis.sha256(campaign / "tasks.json"),
                               "binary_sha256": read_json(campaign / "manifest.json").get("binary_sha256"),
                               "verified_input_hashes": coupled["verified_input_hashes"],
                               "field_sha256": coupled["field_sha256"]},
        "scripts": {str(Path(__file__).resolve()): study_analysis.sha256(Path(__file__)),
                    str(Path(verify_pn_run.__file__).resolve()): study_analysis.sha256(Path(verify_pn_run.__file__)),
                    str(Path(study_analysis.__file__).resolve()): study_analysis.sha256(Path(study_analysis.__file__))},
        "metrics": metric_rows,
        "norms": {"density_L1": "sum_i m_i |e_i| / domain_volume",
                  "density_L2": "sqrt(sum_i m_i e_i^2 / domain_volume)",
                  "relative_density_L1_reference_denominator": reference_denominator,
                  "method_difference_relative_L1_denominator": coupled_denominator},
        "reference_caveat": "Analytic kinetic free-transport moments include finite-PN model error and are not an exact solution of the discrete finite-PN system.",
        "positivity_caveat": "PN-only verifier checks finite fields and all-Q conservation; it reports negative density/cone diagnostics but does not assert positivity or M1 realizability.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "metrics": metric_rows,
                      "pn_rho_min": pn_verification["rho_min"],
                      "pn_negative_density_nodes": pn_verification["negative_density_nodes"],
                      "pn_outside_M1_low_moment_cone_nodes": pn_verification["outside_M1_low_moment_cone_nodes"],
                      "pn_all_Q_balance": pn_verification["maximum_global_Q_balance"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"plot_pn_comparison.py: {exc}", file=sys.stderr)
        raise SystemExit(2)
