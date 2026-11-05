#!/usr/bin/env python3
"""Create preliminary, hash-validated Crossing comparisons from completed NX runs."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
NX = ROOT / "nx"
sys.path.insert(0, str(NX / "scripts"))
import analysis as study_analysis

DEFAULT_CAMPAIGN = NX / "tmp" / "pn_mc_production_20261104_161438_760701"
DEFAULT_OUTPUT = ROOT / "results" / "nx_comparisons" / "20261105_114629" / "crossing"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    if NX not in campaign.parents:
        raise ValueError(f"Campaign must be inside {NX}")
    output = args.output.resolve()
    if ROOT not in output.parents:
        raise ValueError(f"Output must be inside {ROOT}")
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140,
                         "savefig.dpi": 200, "pdf.fonttype": 42})

    manifest = read_json(campaign / "manifest.json")
    tasks_path = campaign / manifest.get("tasks", "tasks.json")
    if study_analysis.sha256(tasks_path) != manifest.get("tasks_sha256"):
        raise ValueError("Frozen task manifest SHA256 mismatch")
    tasks = json.loads(tasks_path.read_text())
    status_counts: dict[str, int] = {}
    data: list[dict] = []
    rows: list[dict] = []
    statuses: dict[int, dict] = {}
    for task in tasks:
        status_path = campaign / "status" / f"{task['id']}.json"
        status = read_json(status_path) if status_path.is_file() else {"status": "not_started"}
        statuses[int(task["id"])] = status
        status_counts[status["status"]] = status_counts.get(status["status"], 0) + 1
        if task.get("family") != "crossing" or task.get("kind") != "spatial":
            continue
        if status.get("status") != "completed":
            continue
        # This rechecks the status validation, solver settings, and every saved
        # field/config/summary/integral digest before any values enter a plot.
        result = study_analysis.task_result(campaign, task, status)
        if "exact" not in result or "exact_error" not in result:
            raise ValueError(f"Crossing task {task['id']} has no kinetic reference")
        cfg = read_json(campaign / task["output_dir"] / "config.json")
        summary = read_json(campaign / task["output_dir"] / "summary.json")
        exact_rho = result["exact"][:, 0]
        rho = result["physical"][:, 0]
        mass = result["mass"]
        volume = result["domain_volume"]
        relative_denominator = float(np.sum(mass * np.abs(exact_rho)) / volume)
        metrics = {
            "density_L1": float(np.sum(mass * np.abs(rho - exact_rho)) / volume),
            "density_L2": float(np.sqrt(np.sum(mass * (rho - exact_rho) ** 2) / volume)),
            "density_Linf": float(np.max(np.abs(rho - exact_rho))),
            "relative_density_L1": (float(np.sum(mass * np.abs(rho - exact_rho)) / volume /
                                               relative_denominator)
                                     if relative_denominator > 0 else math.nan),
            "relative_density_L1_denominator": relative_denominator,
        }
        record = {"id": task["id"], "label": task.get("label"),
                  "grid": int(result["grid"]), "pn_order": int(result["order"]),
                  "dimension": int(task["dimension"]), "time": float(task["time"]),
                  "dt": float(task["dt"]), "steps": int(summary["steps"]),
                  "solver_seconds": float(summary["solver_seconds"]),
                  "wall_seconds": float(status["wall_seconds"]),
                  "peak_rss_kib_rank0": int(summary["peak_rss_kib_rank0"]),
                  "mass_mode": cfg["mass_mode"], "stabilization": cfg["stabilization"],
                  **metrics, "run_output": task["output_dir"],
                  "field_sha256": result["field_sha256"],
                  "config_sha256": result["config_sha256"],
                  "summary_sha256": result["summary_sha256"],
                  "verification_sha256": result["verification_sha256"],
                  "verified_input_hashes": result["verified_input_hashes"]}
        result.update(record)
        result["exact_rho"] = exact_rho
        data.append(result)
        rows.append(record)

    if not rows:
        raise ValueError("No hash-validated completed Crossing spatial runs are available")
    rows.sort(key=lambda r: (r["grid"], r["pn_order"]))
    with (output / "crossing_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            csvrow = dict(row)
            csvrow["verified_input_hashes"] = json.dumps(csvrow["verified_input_hashes"], sort_keys=True)
            writer.writerow(csvrow)

    timestamp = dt.datetime.now().astimezone()
    timestamp_text = timestamp.strftime("%d.%m.%Y %H:%M %Z")
    complete_by_grid = {n: sorted([r["pn_order"] for r in rows if r["grid"] == n])
                        for n in sorted({r["grid"] for r in rows})}
    incomplete_labels = [task["label"] for task in tasks
                         if task.get("family") == "crossing" and task.get("kind") == "spatial"
                         and statuses[int(task["id"])].get("status") != "completed"]
    subtitle = (f"PN-Massenkorrektur B · t={rows[0]['time']:g} · "
                f"{len(rows)}/18 räumliche Läufe validiert · {timestamp_text}")

    def save(fig, stem: str) -> None:
        fig.savefig(output / f"{stem}.png", bbox_inches="tight")
        fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Nodal lumped-mass L2 density error against the analytic kinetic moments.
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.8), layout="constrained")
    for grid in sorted({r["grid"] for r in rows}):
        subset = sorted((r for r in rows if r["grid"] == grid), key=lambda r: r["pn_order"])
        axes[0].semilogy([r["pn_order"] for r in subset], [r["density_L2"] for r in subset],
                         "o-", linewidth=1.8, label=f"{grid}×{grid}")
    for order in sorted({r["pn_order"] for r in rows}):
        subset = sorted((r for r in rows if r["pn_order"] == order), key=lambda r: r["grid"])
        axes[1].semilogy([r["grid"] for r in subset], [r["density_L2"] for r in subset],
                         "o-", linewidth=1.8, label=f"P{order}")
    axes[0].set(xlabel="PN-Ordnung", ylabel="Dichtefehler L₂, ML-gewichtet",
                title="PN-Ordnungsvariation")
    axes[0].set_xticks(sorted({r["pn_order"] for r in rows}))
    axes[1].set(xlabel="Zellen je Achse", title="Gittervariation bei fester PN-Ordnung")
    axes[1].set_xticks([32, 64, 128], labels=["32", "64", "128"])
    axes[1].minorticks_off()
    for ax in axes:
        ax.grid(True, which="both", alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Crossing beams: nodaler Dichtefehler gegen kinetische Referenz\n" + subtitle)
    save(fig, "crossing_error_convergence")

    # Same-grid 128^2 maps: intentionally use only orders completed at that grid.
    # In particular, there is no P15 field here while its run is pending.
    fine = {r["pn_order"]: r for r in data if r["grid"] == 128}
    required_orders = (3, 7, 11)
    missing = [n for n in required_orders if n not in fine]
    if missing:
        raise ValueError(f"Missing completed 128^2 comparison orders: {missing}")
    reference = fine[3]["exact_rho"]
    density_values = [reference] + [fine[n]["physical"][:, 0] for n in required_orders]
    error_values = [fine[n]["physical"][:, 0] - fine[n]["exact_rho"] for n in required_orders]
    density_max = max(float(np.max(a)) for a in density_values)
    error_max = max(float(np.max(np.abs(a))) for a in error_values)
    density_norm = Normalize(vmin=0., vmax=density_max)
    error_norm = TwoSlopeNorm(vmin=-error_max, vcenter=0., vmax=error_max)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.1), layout="constrained")
    map_fields = [(None, reference, "Kinetische Referenz", False)]
    map_fields += [(fine[n], fine[n]["physical"][:, 0], f"PN-MC B, 128² P{n}", False)
                   for n in required_orders]
    map_fields += [(fine[n], fine[n]["physical"][:, 0] - fine[n]["exact_rho"],
                    f"Dichtefehler P{n}", True) for n in required_orders]
    axes[1, 0].axis("off")
    for ax, (run, values, title, is_error) in zip(
            [axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3],
             axes[1, 1], axes[1, 2], axes[1, 3]], map_fields):
        field_run = run if run is not None else fine[3]
        grid_axes, tensor = study_analysis.grid_tensor(
            field_run["coordinates"], np.asarray(values)[:, None], 2)
        mesh = ax.pcolormesh(grid_axes[0], grid_axes[1], tensor[:, :, 0].T,
                             shading="auto", cmap="RdBu_r" if is_error else "viridis",
                             norm=error_norm if is_error else density_norm)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y",
               title=title, aspect="equal")
        fig.colorbar(mesh, ax=ax, shrink=.83,
                     label="ρ − ρₖᵢₙ" if is_error else "ρ")
    fig.suptitle("Crossing beams: gleiche Auflösung 128², gemeinsame Farbskalen\n" + subtitle)
    save(fig, "crossing_128_density_and_error")

    # Accuracy/runtime frontier for every completed Crossing spatial run.
    grids = sorted({r["grid"] for r in rows})
    orders = sorted({r["pn_order"] for r in rows})
    colors = plt.get_cmap("viridis")(np.linspace(.15, .85, len(grids)))
    color_for = dict(zip(grids, colors))
    marker_cycle = ["o", "s", "^", "D", "v", "P"]
    marker_for = {order: marker_cycle[i % len(marker_cycle)] for i, order in enumerate(orders)}
    fig, ax = plt.subplots(figsize=(9.5, 6), layout="constrained")
    for row in rows:
        ax.scatter(row["solver_seconds"], row["density_L2"], color=color_for[row["grid"]],
                   marker=marker_for[row["pn_order"]], s=65, edgecolor="black", linewidth=.4)
        ax.annotate(f"{row['grid']}/P{row['pn_order']}",
                    (row["solver_seconds"], row["density_L2"]), xytext=(5, 4),
                    textcoords="offset points", fontsize=7)
    from matplotlib.lines import Line2D
    grid_handles = [Line2D([], [], marker="o", linestyle="", color=color_for[g], label=f"{g}²")
                    for g in grids]
    order_handles = [Line2D([], [], marker=marker_for[n], linestyle="", color="0.25", label=f"P{n}")
                     for n in orders]
    legend1 = ax.legend(handles=grid_handles, title="Gitter", loc="upper right", fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=order_handles, title="PN-Ordnung", loc="lower left", fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel="Solverzeit [s]",
           ylabel="Dichtefehler L₂, ML-gewichtet",
           title="Praktische Genauigkeit und Solverzeit")
    ax.grid(True, which="both", alpha=.25)
    fig.suptitle("Crossing beams: alle abgeschlossenen Läufe\n" + subtitle)
    save(fig, "crossing_accuracy_vs_solver_seconds")

    # A common y=0.5 density lineout from the 128^2 P3/P7/P11 fields.
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True, layout="constrained")
    reference_axes, reference_tensor = study_analysis.grid_tensor(
        fine[3]["coordinates"], fine[3]["exact_rho"][:, None], 2)
    reference_yidx = int(np.argmin(np.abs(reference_axes[1] - .5)))
    if abs(float(reference_axes[1][reference_yidx]) - .5) > 1e-12:
        raise ValueError("128² reference grid has no y=0.5 node")
    axes[0].plot(reference_axes[0], reference_tensor[:, reference_yidx, 0],
                 "k--", linewidth=2, label="Kinetische Referenz")
    for order in required_orders:
        run = fine[order]
        axes2, tensor = study_analysis.grid_tensor(run["coordinates"],
                                                    run["physical"][:, [0]], 2)
        yidx = int(np.argmin(np.abs(axes2[1] - .5)))
        if abs(float(axes2[1][yidx]) - .5) > 1e-12:
            raise ValueError("128² lineout grid has no y=0.5 node")
        if not (np.array_equal(reference_axes[0], axes2[0]) and
                np.array_equal(reference_axes[1], axes2[1]) and yidx == reference_yidx):
            raise ValueError("128² run grids do not align for the y=0.5 lineout")
        axes[0].plot(axes2[0], tensor[:, yidx, 0], label=f"PN-MC B P{order}")
        exact_axes, exact_tensor = study_analysis.grid_tensor(
            run["coordinates"], run["exact_rho"][:, None], 2)
        axes[1].plot(exact_axes[0], tensor[:, yidx, 0] - exact_tensor[:, yidx, 0],
                     label=f"P{order}")
    axes[0].set(ylabel="ρ(x, 0.5)", title="128²-Dichte entlang y=0.5")
    axes[1].set(xlabel="x", ylabel="ρ − ρₖᵢₙ", title="Signierter Dichtefehler")
    axes[0].legend(fontsize=8, ncols=2)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle("Crossing beams: gleiche Auflösung, P3/P7/P11\n" + subtitle)
    save(fig, "crossing_128_midline")

    record = {
        "created_local": timestamp.isoformat(), "campaign": str(campaign),
        "output": str(output), "campaign_manifest_sha256": study_analysis.sha256(campaign / "manifest.json"),
        "frozen_binary_sha256": manifest.get("binary_sha256"),
        "source_provenance": manifest.get("provenance"),
        "tasks_sha256": study_analysis.sha256(tasks_path),
        "analysis_script_sha256": study_analysis.sha256(Path(study_analysis.__file__)),
        "plot_script_sha256": study_analysis.sha256(Path(__file__)),
        "counts": status_counts, "completed_crossing_spatial": len(rows),
        "completed_orders_by_grid": complete_by_grid,
        "incomplete_crossing_spatial": incomplete_labels,
        "runs": rows,
        "reference": "Analytic kinetic free-transport moments; includes finite-PN model error and is not an exact finite-PN/R2 solution.",
        "error_norm": "Nodal lumped-mass weighted density L2, sqrt(sum_i m_i (rho_i-rho_ref_i)^2 / domain_volume); L1 normalized analogously.",
        "relative_density_L1": "density_L1 divided by nodal lumped-mass weighted L1 of the kinetic reference density.",
        "comparison_scope": "Preliminary spatial Crossing runs only; temporal-reference tasks are pending and temporal contamination has not yet been assessed.",
        "same_grid_maps": "128x128 P3, P7 and P11 plus kinetic reference, shared density and symmetric error scales; P15 omitted because the 128x128 P15 run is not completed.",
    }
    (output / "snapshot.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "completed_crossing_spatial": len(rows),
                      "orders_by_grid": complete_by_grid,
                      "pending_crossing_spatial": incomplete_labels,
                      "artifacts": sorted(p.name for p in output.iterdir() if p.suffix in (".png", ".pdf", ".csv", ".json"))},
                     indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"plot_crossing_snapshot.py: {exc}", file=sys.stderr)
        raise SystemExit(2)
