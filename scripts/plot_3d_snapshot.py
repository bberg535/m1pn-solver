#!/usr/bin/env python3
"""Read-only 3D comparison plots for completed tasks in a partial campaign."""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx/scripts"))
import analysis as study_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.png")):
        raise FileExistsError("Choose a new plot directory.")
    manifest = json.loads((campaign / "manifest.json").read_text())
    if study_analysis.sha256(campaign / "tasks.json") != manifest["tasks_sha256"]:
        raise ValueError("Frozen tasks have changed.")
    tasks = json.loads((campaign / "tasks.json").read_text())
    data, rows, unfinished = [], [], []
    for task in tasks:
        if task["family"] != "three_d":
            continue
        status_file = campaign / "status" / f"{task['id']}.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        if status.get("status") != "completed":
            unfinished.append({"id": task["id"], "label": task["label"], "status": status.get("status", "not_started")})
            continue
        result = study_analysis.task_result(campaign, task, status)
        cfg = json.loads((campaign / task["output_dir"] / "config.json").read_text())
        summary = json.loads((campaign / task["output_dir"] / "summary.json").read_text())
        if summary["steps"] != task["steps"] or summary.get("rejected_attempts", 0) != 0:
            raise ValueError("Unexpected steps/rejections in " + task["label"])
        if cfg["threads"] != manifest["threads"]:
            raise ValueError("Thread count mismatch.")
        rho = result["physical"][:, 0]
        reference = result["exact"][:, 0]
        relative = float(np.sum(result["mass"] * np.abs(rho - reference)) /
                         np.sum(result["mass"] * np.abs(reference)))
        row = {"id": task["id"], "label": task["label"], "grid": task["grid"], "pn_order": task["order"],
               "kind": task["kind"], "time": result["time"], "dt": result["dt"], "steps": summary["steps"],
               "density_L1": result["exact_error"]["density_L1"], "density_L2": result["exact_error"]["density_L2"],
               "density_Linf": result["exact_error"]["density_Linf"], "relative_density_L1": relative,
               "solver_seconds": summary["solver_seconds"], "wall_seconds": status["wall_seconds"],
               "peak_rss_MiB": summary["peak_rss_kib_rank0"] / 1024,
               "threads": cfg["threads"], "global_Q_balance": summary["global_Q_balance"],
               "field_sha256": result["field_sha256"]}
        rows.append(row)
        if task["kind"] == "spatial":
            result["row"] = row
            data.append(result)
    if not data:
        raise ValueError("No completed spatial 3D tasks.")
    if len({(r["time"], r["dt"]) for r in data}) != 1:
        raise ValueError("Spatial comparisons require common final time and dt.")
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/mpl_3d_snapshot"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120, "savefig.dpi": 180, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(out / (name + "." + suffix), bbox_inches="tight")
        plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    for grid in sorted({r["grid"] for r in data}):
        subset = sorted((r for r in data if r["grid"] == grid), key=lambda r: r["order"])
        axs[0].semilogy([r["order"] for r in subset], [r["row"]["density_L2"] for r in subset],
                        "o-", label=f"{grid}³ Zellen")
    for order in sorted({r["order"] for r in data}):
        subset = sorted((r for r in data if r["order"] == order), key=lambda r: r["grid"])
        axs[1].loglog([r["grid"] for r in subset], [r["row"]["density_L2"] for r in subset],
                      "o-", label=f"P{order}")
    for ax in axs:
        ax.set_ylabel("Dichte-L₂ zur kinetischen Referenz")
        ax.grid(True, which="both", alpha=.25)
        ax.legend()
    axs[0].set(xlabel="PN-Ordnung N", xticks=sorted({r["order"] for r in data}), title="PN-Ordnung bei gleichem Gitter")
    axs[1].set(xlabel="Zellen je Achse", xticks=[8, 16, 32], xticklabels=["8", "16", "32"], title="Gitter bei gleicher PN-Ordnung")
    axs[1].minorticks_off()
    fig.suptitle(f"3D-Crossing · Standard B · t={data[0]['time']:g} · {len(data)}/12 reguläre Läufe\n"
                 "Vorläufig: kinetische Referenz enthält PN-Modellfehler; Δt/2-Prüfung steht aus", fontsize=11)
    save(fig, "convergence")

    finest = max(r["grid"] for r in data)
    selected = sorted((r for r in data if r["grid"] == finest), key=lambda r: r["order"])
    slices = []
    for result in selected:
        xyz = result["coordinates"]
        z_values = np.unique(xyz[:, 2])
        z = float(z_values[np.argmin(np.abs(z_values - .5))])
        mask = np.abs(xyz[:, 2] - z) < 1e-13
        values = np.column_stack((result["physical"][mask, 0], result["exact"][mask, 0],
                                  result["physical"][mask, 0] - result["exact"][mask, 0]))
        axes, tensor = study_analysis.grid_tensor(xyz[mask], values, 2)
        slices.append((result, z, axes, tensor))
    density_max = max(float(np.max(s[3][:, :, :2])) for s in slices)
    error_max = max(float(np.max(np.abs(s[3][:, :, 2]))) for s in slices)
    density_norm = Normalize(0, density_max)
    error_norm = TwoSlopeNorm(vcenter=0, vmin=-max(error_max, 1e-15), vmax=max(error_max, 1e-15))
    fig, axs = plt.subplots(len(slices), 2, figsize=(9.5, 3.8 * len(slices)), layout="constrained", squeeze=False)
    for row_index, (result, z, axes, tensor) in enumerate(slices):
        for column, index in enumerate((0, 2)):
            ax = axs[row_index, column]
            mesh = ax.pcolormesh(axes[0], axes[1], tensor[:, :, index].T, shading="auto",
                                 cmap="viridis" if column == 0 else "RdBu_r",
                                 norm=density_norm if column == 0 else error_norm)
            ax.set(xlabel="x", ylabel="y", aspect="equal", xlim=(0, 1), ylim=(0, 1),
                   title=f"P{result['order']} · " + ("Dichte" if column == 0 else "ρ − ρ_ref"))
            fig.colorbar(mesh, ax=ax, shrink=.85)
    fig.suptitle(f"3D-Crossing · {finest}³ Zellen · t={selected[0]['time']:g} · Schnitt z={slices[0][1]:g}\n"
                 "Gleiche Skalen innerhalb der Dichte- und Fehlerpanels", fontsize=12)
    save(fig, "density_slices")

    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    line_rows = []
    for index, result in enumerate(selected):
        xyz = result["coordinates"]
        mask = (np.abs(xyz[:, 1] - .5) < 1e-13) & (np.abs(xyz[:, 2] - .5) < 1e-13)
        order = np.argsort(xyz[mask, 0])
        x = xyz[mask, 0][order]
        reference = result["exact"][mask, 0][order]
        numerical = result["physical"][mask, 0][order]
        if index == 0:
            ax.plot(x, reference, "k--", lw=2, label="Kinetische Referenz")
        ax.plot(x, numerical, "o-", ms=3, label=f"P{result['order']}")
        line_rows.extend({"pn_order": result["order"], "x": a, "numerical_density": b, "kinetic_density": c}
                         for a, b, c in zip(x, numerical, reference))
    ax.set(xlabel="x", ylabel="Dichte ρ", title=f"3D-Crossing · {finest}³ · y=z=0,5 · t={selected[0]['time']:g}")
    ax.grid(alpha=.25)
    ax.legend()
    save(fig, "lineouts")
    with (out / "lineouts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(line_rows[0]))
        writer.writeheader()
        writer.writerows(line_rows)
    provenance = {"created_berlin": datetime.datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
                  "campaign": str(campaign), "manifest_sha256": study_analysis.sha256(campaign / "manifest.json"),
                  "tasks_sha256": manifest["tasks_sha256"], "script_sha256": study_analysis.sha256(Path(__file__)),
                  "completed_tasks": [study_analysis.jsonable_result(r) for r in data], "unfinished_tasks": unfinished,
                  "norm": "Density errors weighted by lumped nodal mass and normalized by domain volume; Linf over nodes.",
                  "reference": "Analytic kinetic free-transport density; includes PN model/boundary/limiter and spatial error.",
                  "temporal_reference_available": any(r["kind"] == "temporal" for r in rows),
                  "preliminary": True, "slice_z": slices[0][1]}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"completed_regular": len(data), "output": str(out), "metrics": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
