#!/usr/bin/env python3
"""Plot the latest completed local Crossing run, preserving all raw inputs."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx/scripts"))
import analysis as study_analysis
import numpy as np


def latest_run() -> Path:
    candidates = []
    for folder in (ROOT / "tmp").glob("local_B_*"):
        if not folder.is_dir() or not (folder / "final.csv").is_file():
            continue
        try:
            summary = json.loads((folder / "summary.json").read_text())
            cfg = json.loads((folder / "config.json").read_text())
            if summary["status"] == "completed" and abs(summary["time"] - cfg["final_time"]) < 1e-12:
                candidates.append(folder)
        except (OSError, ValueError, KeyError):
            continue
    if not candidates:
        raise ValueError("No completed local run with final.csv found.")
    return max(candidates, key=lambda p: (p / "final.csv").stat().st_mtime_ns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Default: latest completed tmp/local_B_*.")
    parser.add_argument("--output", type=Path, help="New plot directory.")
    args = parser.parse_args()
    run = args.run.resolve(strict=True) if args.run else latest_run()
    cfg = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    if cfg["dimension"] != 2 or not cfg.get("beam_geometry"):
        raise ValueError("This plot requires a 2D Crossing run with beam metadata.")
    if summary["status"] != "completed" or abs(summary["time"] - cfg["final_time"]) > 1e-12:
        raise ValueError("Run has not completed at its requested final time.")
    verification = study_analysis.verify_saved_inputs(run)
    _, initial = study_analysis.read_csv(run / "initial.csv")
    _, final = study_analysis.read_csv(run / "final.csv")
    _, mass_data = study_analysis.read_csv(run / "nodal_mass.csv")
    if not np.array_equal(initial[:, :4], final[:, :4]) or not np.array_equal(final[:, 0], mass_data[:, 0]):
        raise ValueError("Field coordinates or mass node IDs differ.")
    xyz, mass = final[:, 1:4], mass_data[:, 1]
    initial_rho = study_analysis.physical_low(initial[:, 4:])[:, 0]
    rho = study_analysis.physical_low(final[:, 4:])[:, 0]
    reference, _ = study_analysis.beam_exact(xyz, summary["time"], cfg["beam_geometry"])
    error = rho - reference
    volume = study_analysis.volume_of(xyz, 2)
    metrics = {"density_L1": float(np.sum(mass * np.abs(error)) / volume),
               "density_L2": float(np.sqrt(np.sum(mass * error**2) / volume)),
               "density_Linf": float(np.max(np.abs(error))),
               "relative_density_L1": float(np.sum(mass * np.abs(error)) / np.sum(mass * np.abs(reference)))}
    stamp = datetime.datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%d_%H%M%S_%f")
    out = (args.output or ROOT / "results/local_plots" / (run.name + "_" + stamp)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib_local"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120, "savefig.dpi": 180,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    values = np.column_stack((initial_rho, reference, rho, error))
    axes, tensor = study_analysis.grid_tensor(xyz, values, 2)
    density_norm = Normalize(0, float(np.max(values[:, :3])))
    error_max = max(float(np.max(np.abs(error))), 1e-15)
    error_norm = TwoSlopeNorm(vcenter=0, vmin=-error_max, vmax=error_max)
    title = (f"PN-Massenkorrektur B · {cfg['elements_per_axis']}×{cfg['elements_per_axis']} Zellen"
             f" · P{cfg['pn_order']} · t={summary['time']:g} · Δt={cfg['dt_selected']:g}")
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 9), layout="constrained")
    titles = ["Anfangsdichte", "Kinetische Referenz am Endzeitpunkt",
              "Berechnete Enddichte", "Signierter Dichtefehler ρ − ρ_ref"]
    for index, (ax, panel_title) in enumerate(zip(axs.flat, titles)):
        is_error = index == 3
        mesh = ax.pcolormesh(axes[0], axes[1], tensor[:, :, index].T, shading="auto",
                             cmap="RdBu_r" if is_error else "viridis",
                             norm=error_norm if is_error else density_norm)
        ax.set(xlabel="x", ylabel="y", title=panel_title, aspect="equal",
               xlim=(axes[0][0], axes[0][-1]), ylim=(axes[1][0], axes[1][-1]))
        fig.colorbar(mesh, ax=ax, shrink=.86, label="ρ − ρ_ref" if is_error else "ρ")
    fig.suptitle(title + f"\nDichte-L₂ = {metrics['density_L2']:.4g} · relativer Dichte-L₁ = {metrics['relative_density_L1']:.2%}", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("density_comparison." + suffix), bbox_inches="tight")
    plt.close(fig)
    mid = int(np.argmin(np.abs(axes[1] - .5)))
    line = np.column_stack((axes[0], tensor[:, mid, 1], tensor[:, mid, 2], tensor[:, mid, 3]))
    np.savetxt(out / "lineout.csv", line, delimiter=",", comments="", header="x,reference_density,numerical_density,signed_error")
    fig, ax = plt.subplots(figsize=(8.3, 4.2), layout="constrained")
    ax.plot(line[:, 0], line[:, 1], color="black", lw=2, label="Kinetische Referenz")
    ax.plot(line[:, 0], line[:, 2], "o-", color="tab:blue", ms=3, lw=1.4, label="PN-Massenkorrektur B")
    ax.set(xlabel="x", ylabel="Dichte ρ", title=title + f"\nSchnitt bei y={axes[1][mid]:g}")
    ax.grid(alpha=.25)
    ax.legend()
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("lineout." + suffix), bbox_inches="tight")
    plt.close(fig)
    record = {"run": str(run), "config": cfg, "summary": summary, "error_metrics": metrics,
              "reference": "Analytic kinetic transport density; differences include finite-PN model error.",
              "norm": "Lumped-node-mass weighted errors divided by domain volume; Linf over nodes.",
              "verified_input_sha256": verification["sha256"],
              "plot_script_sha256": study_analysis.sha256(Path(__file__)), "lineout_y": float(axes[1][mid])}
    (out / "plot_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"run": str(run), "plots": str(out), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
