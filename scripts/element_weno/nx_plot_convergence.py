#!/usr/bin/env python3
"""Produce publication-style convergence plots from a *verified* frozen campaign."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    campaign = args.campaign.resolve(strict=True)
    source = campaign / "results.json"
    data = json.loads(source.read_text())
    rows = data["rows"]
    output = campaign / "figures"
    output.mkdir(exist_ok=True)
    plots = [
        ("smooth", "error_exact_PN_L2", "h", "Spatial error vs exact P_N", "Exact P_N density (CG-WENO)"),
        ("boundary", "error_kinetic_L2", "h", "Open-boundary kinetic error", "Includes P_N truncation"),
        ("crossing", "error_kinetic_L2", "h", "Crossing-beams kinetic error", "Includes P_N truncation"),
        ("crossing", "error_spatial_self_L2_to_next", "h", "Crossing fixed-P_N self difference", "Next finer grid"),
        ("line", "error_kinetic_L2", "h", "Smoothed line-source kinetic error", "Ganapol reference, rescaled"),
        ("time_smooth", "error_temporal_self_L2", "dt", "Temporal self convergence, smooth + scattering", "Strang + SSPRK(4,3)"),
        ("time_crossing", "error_temporal_self_L2", "dt", "Temporal self convergence, crossing + reactions", "Strang + SSPRK(4,3)") ]
    created = []
    for family, metric, axis, title, subtitle in plots:
        group = [r for r in rows if r["family"] == family and r.get(metric) is not None
                 and r[axis] is not None and r[metric] > 0]
        if not group:
            continue
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        identifiers = sorted(set((r["pn"], r["p"], r["rk"]) for r in group))
        for pn, p, rk in identifiers:
            seq = sorted((r for r in group if (r["pn"], r["p"], r["rk"]) == (pn, p, rk)),
                         key=lambda r: r[axis], reverse=True)
            xs = [r[axis] for r in seq]
            ys = [r[metric] for r in seq]
            ax.loglog(xs, ys, "o-", linewidth=1.4,
                      label=f"P{pn} Q{p} {rk}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time step Δt" if axis == "dt" else "Element diameter h")
        ax.set_ylabel("Relative L² error" if "exact_PN" not in metric else "Exact P_N density L² error")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=.25)
        ax.legend(fontsize=8)
        fig.text(.13, .01, subtitle, fontsize=8, alpha=.75)
        fig.tight_layout(rect=(0, .04, 1, 1))
        name = f"{family}_{metric}"
        fig.savefig(output / (name + ".pdf"))
        fig.savefig(output / (name + ".png"), dpi=180)
        plt.close(fig)
        created.append(name)
    benchmarks = [r for r in rows if r["family"] in ("ablation", "baseline")]
    if benchmarks:
        labels = [r["label"].replace("ablation_", "").replace("baseline_", "")
                  for r in benchmarks]
        seconds = [r["elapsed_seconds"] for r in benchmarks]
        fig, ax = plt.subplots(figsize=(8., max(3.8, .44 * len(labels) + 1.)))
        ax.barh(labels, seconds)
        ax.invert_yaxis()
        ax.set_xlabel("Wall-clock seconds")
        ax.set_title("Crossing ablation: measured run times")
        ax.grid(True, axis="x", alpha=.25)
        fig.tight_layout()
        fig.savefig(output / "ablation_wall_seconds.pdf")
        fig.savefig(output / "ablation_wall_seconds.png", dpi=180)
        plt.close(fig)
        created.append("ablation_wall_seconds")
    # Reusable compact LaTeX table: values are reported as they are;
    # time and space self-reference errors are not relabelled as exact.
    columns = ["family", "pn", "p", "elements", "dt", "error_exact_PN_L2",
               "error_kinetic_L2", "error_temporal_self_L2",
               "error_spatial_self_L2_to_next", "elapsed_seconds"]
    def cell(v):
        if v is None:
            return "--"
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v).replace("_", r"\_")
    lines = [r"\begin{tabular}{lrrrrrrrrr}",
             r"\hline",
             " & ".join(c.replace("_", r"\_") for c in columns) + r" \\",
             r"\hline"]
    for row in rows:
        lines.append(" & ".join(cell(row.get(k)) for k in columns) + r" \\")
    lines += [r"\hline", r"\end{tabular}", ""]
    (campaign / "tables.tex").write_text("\n".join(lines))
    (campaign / "figures.json").write_text(json.dumps({"figures": created}, indent=2) + "\n")
    print(f"{len(created)} plot families, tables.tex")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
