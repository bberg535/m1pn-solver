#!/usr/bin/env python3
"""Verified first-look density comparisons for the completed NX lattice runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NX = ROOT / "nx"
CAMPAIGN_DEFAULT = NX / "tmp" / "pn_mc_production_20261104_161438_760701"
OUTPUT_DEFAULT = ROOT / "results" / "nx_comparisons" / "20261105_114629" / "lattice"
spec = importlib.util.spec_from_file_location("nx_analysis", NX / "scripts" / "analysis.py")
analysis = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = analysis
spec.loader.exec_module(analysis)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def weighted_density_metrics(error: np.ndarray, mass: np.ndarray, volume: float) -> dict[str, float]:
    """Nodal-mass weighted density-only norms, normalized by physical volume."""
    return {
        "L1": float(np.sum(mass * np.abs(error)) / volume),
        "L2": float(np.sqrt(np.sum(mass * error * error) / volume)),
        "Linf": float(np.max(np.abs(error))),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    tasks = json.loads((campaign / "tasks.json").read_text())
    manifest = json.loads((campaign / "manifest.json").read_text())
    if digest(campaign / "tasks.json") != manifest.get("tasks_sha256"):
        raise ValueError("Frozen campaign task list hash does not match manifest")
    completed = {}
    run_records = []
    for task in tasks:
        if task.get("family") != "lattice" or task.get("kind", "spatial") != "spatial":
            continue
        status_path = campaign / "status" / f"{task['id']}.json"
        status = json.loads(status_path.read_text())
        if status.get("status") != "completed" or status.get("validated") is not True:
            raise RuntimeError(f"lattice spatial task {task['id']} is not completed and validated")
        result = analysis.task_result(campaign, task, status)
        out = campaign / status["output_dir"]
        axes, tensor = analysis.grid_tensor(result["coordinates"], result["physical"][:, :1], 2)
        expected_axis = np.linspace(0., 7., int(task["grid"]) + 1)
        if any(not np.allclose(a, expected_axis, rtol=0, atol=2e-12) for a in axes):
            raise ValueError(f"unexpected lattice coordinate axes for task {task['id']}")
        if not result.get("lattice_indicators", {}).get("valid"):
            raise ValueError(f"lattice material indicators did not validate for task {task['id']}")
        entry = {"task": task, "result": result, "output": out,
                 "axes": axes, "rho_tensor": tensor[..., 0]}
        completed[(int(task["grid"]), int(task["angular_order"]))] = entry
        run_records.append({
            "id": task["id"], "label": task["label"], "grid": task["grid"],
            "order": task["angular_order"], "steps": status["diagnostics"]["summary"]["steps"],
            "time": status["diagnostics"]["summary"]["time"], "dt": task["dt"],
            "rho_min": result["rho_min"], "rho_max": result["rho_max"],
            "field_sha256": result["field_sha256"], "config_sha256": result["config_sha256"],
            "summary_sha256": result["summary_sha256"],
            "verification_sha256": result["verification_sha256"],
            "lattice_volume": result["lattice_indicators"]["volume"],
            "source_integral": result["lattice_indicators"]["source_integral"],
            "absorption_integral": result["lattice_indicators"]["absorption_integral"],
            "scattering_integral": result["lattice_indicators"]["scattering_integral"],
        })
    expected = {(n, p) for n in (28, 56, 112) for p in (3, 5, 7, 9, 11)}
    if set(completed) != expected:
        raise ValueError(f"expected 15 spatial lattice results; found {sorted(completed)}")
    common_times = {float(entry["task"]["time"]) for entry in completed.values()}
    common_dts = {float(entry["task"]["dt"]) for entry in completed.values()}
    if len(common_times) != 1 or len(common_dts) != 1:
        raise ValueError("spatial lattice runs do not share a common final time and dt")
    common_time, common_dt = next(iter(common_times)), next(iter(common_dts))
    if abs(common_time - 3.2) > 1e-13:
        raise ValueError(f"unexpected common lattice final time: {common_time}")
    write_csv(output / "verified_runs.csv", run_records)

    # Same-grid angular snapshots. The same nodes and FE masses are used for all orders.
    fine_entries = {p: completed[(112, p)] for p in (3, 5, 7, 9, 11)}
    ref = fine_entries[11]
    ref_order_rows = []
    for order, entry in fine_entries.items():
        perm = analysis.coordinate_permutation(ref["result"]["coordinates"],
                                               entry["result"]["coordinates"])
        rho = entry["result"]["physical"][perm, 0]
        ref_rho = ref["result"]["physical"][:, 0]
        mass = ref["result"]["mass"]
        norms = weighted_density_metrics(rho - ref_rho, mass, ref["result"]["domain_volume"])
        ref_order_rows.append({"comparison": "same_grid_angular_difference",
                               "grid": 112, "order": order, "reference_order": 11,
                               "reference_status": "PN11 finite-order reference; not exact solution",
                               **norms})

    # Spatial self differences: interpolate the coarser structured FE nodal field
    # to the finer grid, then apply the finer grid's nodal FE masses.
    grid_rows = []
    for order in (3, 5, 7, 9, 11):
        for coarse, fine in ((28, 56), (56, 112)):
            left, right = completed[(coarse, order)], completed[(fine, order)]
            coarse_rho = left["result"]["physical"][:, 0]
            fine_xyz = right["result"]["coordinates"]
            mapped = analysis.interpolate_structured(left["result"]["coordinates"],
                                                     coarse_rho[:, None], fine_xyz, 2)[:, 0]
            fine_rho = right["result"]["physical"][:, 0]
            norms = weighted_density_metrics(fine_rho - mapped, right["result"]["mass"],
                                             right["result"]["domain_volume"])
            grid_rows.append({"comparison": "same_order_grid_difference",
                              "coarse_grid": coarse, "fine_grid": fine, "order": order,
                              "norm_evaluation": "finer-grid nodal FE masses; coarse field multilinearly interpolated",
                              **norms})
    metrics_rows = grid_rows + ref_order_rows
    write_csv(output / "density_difference_metrics.csv", metrics_rows)

    # Density maps on the identical 112 x 112 cell mesh with one shared linear scale.
    map_orders = (3, 7, 11)
    vmin = min(float(fine_entries[p]["rho_tensor"].min()) for p in map_orders)
    vmax = max(float(fine_entries[p]["rho_tensor"].max()) for p in map_orders)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.25), sharex=True, sharey=True,
                             constrained_layout=True)
    image = None
    for ax, order in zip(axes, map_orders):
        entry = fine_entries[order]
        image = ax.pcolormesh(entry["axes"][0], entry["axes"][1], entry["rho_tensor"].T,
                              shading="nearest", cmap="viridis", vmin=vmin, vmax=vmax,
                              rasterized=True)
        ax.set_title(f"PN{order}, 112² cells")
        ax.set_xlabel("x")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y")
    cbar = fig.colorbar(image, ax=axes, shrink=.88)
    cbar.set_label("density ρ (shared linear scale)")
    fig.suptitle("Lattice-PN: Dichte bei t = 3,2")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"density_maps_P3_P7_P11_N112.{ext}", dpi=220)
    plt.close(fig)

    # Centerline y=3.5 is a grid line for N=112; retain the nodal data in CSV.
    line_y = 3.5
    line_rows = []
    xaxis = ref["axes"][0]
    for order, entry in fine_entries.items():
        iy = int(np.argmin(np.abs(entry["axes"][1] - line_y)))
        if abs(float(entry["axes"][1][iy]) - line_y) > 1e-12:
            raise ValueError("centerline y=3.5 is absent from N=112 lattice grid")
        for x, rho in zip(xaxis, entry["rho_tensor"][:, iy]):
            line_rows.append({"x": float(x), "y": line_y, "order": order, "density": float(rho)})
    write_csv(output / "density_lineout_y3_5_N112.csv", line_rows)
    fig, ax = plt.subplots(figsize=(8.8, 4.7), constrained_layout=True)
    for order, entry in fine_entries.items():
        iy = int(np.argmin(np.abs(entry["axes"][1] - line_y)))
        ax.plot(xaxis, entry["rho_tensor"][:, iy], label=f"PN{order}", lw=1.5)
    ax.set(xlabel="x bei y = 3,5", ylabel="Dichte ρ",
           title="Lattice-Dichteschnitt bei t = 3,2, 112² Zellen")
    ax.grid(True, alpha=.25)
    ax.legend(ncol=5, frameon=False)
    fig.savefig(output / "density_lineout_y3_5_N112.png", dpi=220)
    fig.savefig(output / "density_lineout_y3_5_N112.pdf")
    plt.close(fig)

    # Compact metric plot. Left: fixed-order grid differences. Right: same-grid
    # angular differences against the finite PN11 result, never called exact.
    fig, (ax_grid, ax_order) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                           constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for i, order in enumerate((3, 5, 7, 9, 11)):
        rows = [r for r in grid_rows if r["order"] == order]
        ax_grid.plot([r["fine_grid"] for r in rows], [r["L2"] for r in rows],
                     marker="o", label=f"PN{order}", color=colors(i))
    ax_grid.set(xscale="log", yscale="log", xticks=[56, 112],
                xticklabels=["56", "112"], xlabel="feineres Gitter (Zellen je Achse)",
                ylabel="massengewichtete Dichte-L₂-Differenz",
                title="Gittervergleich bei gleicher PN-Ordnung")
    ax_grid.tick_params(axis="x", which="minor", labelbottom=False)
    ax_grid.grid(True, which="both", alpha=.25)
    ax_grid.legend(title="PN-Ordnung", frameon=False, ncol=2)
    order_rows = sorted(ref_order_rows, key=lambda r: r["order"])
    nonself_order_rows = [r for r in order_rows if r["order"] != 11]
    ax_order.plot([r["order"] for r in nonself_order_rows], [r["L2"] for r in nonself_order_rows],
                  marker="o", color="tab:purple")
    ax_order.set(yscale="log", xticks=[3, 5, 7, 9, 11], xlabel="PN-Ordnung",
                 ylabel="massengewichtete Dichte-L₂-Differenz",
                 title="N=112 gegen PN11")
    ax_order.grid(True, which="both", alpha=.25)
    ax_order.text(.02, .02, "PN11 ist eine endliche Referenz, keine exakte Lösung.",
                  transform=ax_order.transAxes, fontsize=8, va="bottom")
    ax_order.text(.98, .98, "PN11-Selbstdifferenz = 0 (definitionsgemäß)",
                  transform=ax_order.transAxes, fontsize=8, ha="right", va="top")
    fig.suptitle(f"Dichtevergleiche bei gemeinsamem t = {common_time:g} und Δt = {common_dt:.9g}")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"density_difference_metrics.{ext}", dpi=220)
    plt.close(fig)

    lattice_112 = completed[(112, 11)]
    temporal_task = next(t for t in tasks if t.get("label") == "lattice_N112_P11_dt2")
    temporal_status_path = campaign / "status" / f"{temporal_task['id']}.json"
    temporal_status = json.loads(temporal_status_path.read_text()) if temporal_status_path.exists() else None
    temporal_state = ("completed_validated" if temporal_status and temporal_status.get("status") == "completed"
                      and temporal_status.get("validated") else
                      (temporal_status.get("status", "pending") if temporal_status else "pending"))

    # Recompute numerical highlights from the saved task outputs.
    top_grid = [r for r in grid_rows if r["fine_grid"] == 112]
    top_grid_min = min(top_grid, key=lambda r: r["L2"])
    top_grid_max = max(top_grid, key=lambda r: r["L2"])
    order_by_error = sorted(ref_order_rows, key=lambda r: r["L2"])
    nonself_order_by_error = sorted((r for r in ref_order_rows if r["order"] != 11),
                                    key=lambda r: r["L2"])
    if any(abs(completed[(112, p)]["result"]["domain_volume"] - 49.) > 1e-8 for p in (3,5,7,9,11)):
        raise ValueError("lattice domain volume differs from expected 49")
    summary_lines = [
        "# Erste NX-Lattice-Vergleiche",
        "",
        "Alle 15 räumlichen Läufe (N=28, 56, 112; PN3, 5, 7, 9, 11) wurden über `analysis.task_result` geprüft. Die eingefrorene Taskliste stimmt mit ihrem Manifest-Hash überein; die gespeicherten Eingabe-Hashes, Solver-Metadaten und Lattice-Materialintegrale stimmen ebenfalls. Gemeinsame Endzeit ist t=3.2, gemeinsames Δt=0.0005209181181832981.",
        "",
        f"Bei N=112 sinkt der Dichteunterschied zur PN11-Referenz über PN3→PN9 von L2={nonself_order_by_error[-1]['L2']:.6g} auf {nonself_order_by_error[0]['L2']:.6g}; PN11 selbst hat definitionsgemäß null Differenz. PN11 ist eine endliche Winkelordnung und keine exakte Lösung.",
        f"Für die Verfeinerung 56→112 ist die kleinste massengewichtete L2-Differenz PN{top_grid_min['order']} ({top_grid_min['L2']:.6g}), die größte PN{top_grid_max['order']} ({top_grid_max['L2']:.6g}). Das sind Same-order-Felddifferenzen nach multilinearer Interpolation auf die feineren Knoten, keine Fehler gegen eine exakte Lösung.",
        "",
        f"Der zusätzliche Lauf N=112, PN11 mit halbiertem Δt ist derzeit **{temporal_state}**. Daher enthalten diese ersten Vergleiche keine zeitliche Fehlerabschätzung und begründen keine formale Konvergenzordnung.",
        "",
        "Dichtekarten verwenden eine gemeinsame lineare Farbskala. Alle L1/L2-Werte sind Dichte-only, durch das Domänenvolumen normiert und mit den feineren Läufen gespeicherten lumped nodal masses gewichtet. Der PN11-Vergleich ist als endliche Referenz gekennzeichnet.",
        "",
        "Dateien: `density_maps_P3_P7_P11_N112.png/.pdf`, `density_lineout_y3_5_N112.png/.pdf`, `density_difference_metrics.png/.pdf`, `density_difference_metrics.csv`, `density_lineout_y3_5_N112.csv`, `verified_runs.csv`, `provenance.json`.",
        "",
    ]
    (output / "findings.md").write_text("\n".join(summary_lines))

    input_hashes = {}
    for entry in completed.values():
        task, run = entry["task"], entry["output"]
        input_hashes[task["label"]] = {
            name: digest(run / name) for name in
            ("initial.csv", "final.csv", "nodal_mass.csv", "config.json", "summary.json",
             "verification.json", "material.csv")
        }
    provenance = {
        "campaign": str(campaign),
        "campaign_manifest_sha256": digest(campaign / "manifest.json"),
        "tasks_sha256": digest(campaign / "tasks.json"),
        "frozen_tasks_hash_validated": True,
        "common_spatial_final_time": common_time,
        "common_spatial_dt": common_dt,
        "campaign_git_head": manifest.get("provenance", {}).get("git_head"),
        "campaign_binary_sha256": manifest.get("binary_sha256"),
        "campaign_source_sha256": manifest.get("provenance", {}).get("source_sha256"),
        "analysis_script_sha256": digest(NX / "scripts" / "analysis.py"),
        "plot_script_sha256": digest(Path(__file__).resolve()),
        "verified_run_input_sha256": input_hashes,
        "included_tasks": [completed[k]["task"]["id"] for k in sorted(completed)],
        "excluded_temporal_reference": {
            "task_id": temporal_task["id"], "label": temporal_task["label"],
            "status": temporal_state, "included_in_metrics": False,
        },
        "metric_definition": "density-only; L1=sum(m_i*abs(diff_i))/49; L2=sqrt(sum(m_i*diff_i^2)/49); Linf=max(abs(diff_i)); grid self differences interpolate coarse structured nodal density to finer coordinates and use finer-run nodal masses",
        "same_grid_order_reference": "PN11 at N=112 is a finite-order reference, not an exact solution",
        "display_scale": {"density_maps": "shared linear vmin/vmax across PN3, PN7, PN11 at N=112",
                          "vmin": vmin, "vmax": vmax},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", "output": str(output),
                      "spatial_runs_verified": len(completed),
                      "temporal_reference_status": temporal_state,
                      "finest_grid_difference_L2_min": top_grid_min,
                      "finest_grid_difference_L2_max": top_grid_max,
                      "same_grid_angular_L2_by_order": {r["order"]: r["L2"] for r in order_by_error}}, indent=2))


if __name__ == "__main__":
    main()
