#!/usr/bin/env python3
"""Tables and figures of the positivity-projection NX campaign (scripts/positivity/campaign.py).

Usage: analyze.py --campaign DIR --output results/positivity/<name>
Writes summary.json, runtime.md, accuracy.md, scaling.md and fig_*.png/pdf. Missing runs are listed, never filled in.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import load  # noqa: E402

COLOR = {"ref": "#8a8a85", "c6": "#2a78d6", "proj": "#eb6834", "projN1": "#1baf7a"}
LABEL = {"ref": "Referenz bdd5d3f (C6)", "c6": "optimiert, C6", "proj": "Projektion, Regel 2N+1",
         "projN1": "Projektion, Regel N+1"}
MARK = {"ref": "s", "c6": "o", "proj": "^", "projN1": "D"}


def style(ax):
    ax.grid(True, color="#e4e3df", linewidth=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#8a8a85")
    ax.spines["bottom"].set_color("#8a8a85")
    ax.tick_params(colors="#55554f", labelsize=8)


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=160)
    plt.close(fig)


def field_diff(a, b):
    ra, rb = load(a), load(b)
    scale = max(max(abs(v) for v in ra["density"]), 1e-300)
    return max(max(abs(x - y) for x, y in zip(ra[k], rb[k])) for k in ("density", "jx", "jy", "jz")) / scale


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    ns = p.parse_args()
    camp, out = ns.campaign.resolve(), ns.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((camp / "manifest.json").read_text())
    runs, missing = {}, []
    for pack in manifest["packs"].values():
        for name in pack["commands"]:
            if name == "bench":
                continue
            f = camp / "runs" / name / "summary.json"
            if f.exists():
                runs[name] = json.loads(f.read_text())
            else:
                missing.append(name)
    bench = []
    if (camp / "runs" / "bench.jsonl").exists():
        bench = [json.loads(l) for l in (camp / "runs" / "bench.jsonl").read_text().splitlines() if l.strip()]
    summary = {"campaign": str(camp), "commit": manifest["commit"], "missing": missing, "runtime": [], "accuracy": [],
               "order_scaling": [], "grid_scaling": [], "bench": bench}

    # ---- Runtime and the <= 1e-6 check (ref vs optimized C6), projection overhead.
    lines = ["| p | E | N | Wand ref [s] | Wand C6 [s] | Gesamt-Speedup | Rand ref [s] | Rand C6 [s] | Rand-Speedup "
             "| max. rel. Feldabw. ref/C6 | Wand Proj [s] | Proj/C6 | Ghost-Anteil Proj |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p_ in (1, 2):
        for E in (32, 64, 128, 256):
            for N in (3, 7, 15):
                key = f"p{p_}_E{E}_N{N}"
                r, c, j = runs.get(f"rt_ref_{key}"), runs.get(f"rt_c6_{key}"), runs.get(f"rt_proj_{key}")
                if not (r and c):
                    continue
                diff = field_diff(camp / "runs" / f"rt_ref_{key}", camp / "runs" / f"rt_c6_{key}")
                row = {"p": p_, "E": E, "N": N, "wall_ref": r["wall_seconds"], "wall_c6": c["wall_seconds"],
                       "boundary_ref": r["seconds"]["boundary"], "boundary_c6": c["seconds"]["boundary"],
                       "max_field_rel_diff_ref_c6": diff, "steps": c["steps"]}
                if j:
                    row.update(wall_proj=j["wall_seconds"], ghost_proj=j["seconds"]["ghost"],
                               ghost_c6=c["seconds"]["ghost"])
                summary["runtime"].append(row)
                lines.append(
                    f"| {p_} | {E} | {N} | {r['wall_seconds']:.2f} | {c['wall_seconds']:.2f} | "
                    f"{r['wall_seconds'] / c['wall_seconds']:.2f} | {r['seconds']['boundary']:.2f} | "
                    f"{c['seconds']['boundary']:.2f} | {r['seconds']['boundary'] / c['seconds']['boundary']:.2f} | "
                    f"{diff:.1e} | " + (f"{j['wall_seconds']:.2f} | {j['wall_seconds'] / c['wall_seconds']:.3f} | "
                                        f"{j['seconds']['ghost'] / j['wall_seconds']:.1%} |" if j else "– | – | – |"))
    (out / "runtime.md").write_text("\n".join(lines) + "\n")

    # ---- Accuracy against the kinetic solution (contains the PN model error).
    lines = ["| p | E | N | Variante | kin. L2-Fehler | L2 Ausflussstreifen | rel. Massenfehler | min. Dichte (Koeff.) "
             "| ergänzt | Newton/Aufruf | max. Momentenrest |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p_ in (1, 2):
        for E in (32, 64, 128, 256):
            for N in (3, 7, 15):
                for v in ("c6", "proj", "projN1"):
                    d = runs.get(f"rt_{v}_p{p_}_E{E}_N{N}")
                    if not d:
                        continue
                    pr = d.get("projection", {})
                    row = {"p": p_, "E": E, "N": N, "variant": v, "kinetic": d["kinetic_l2_error"],
                           "strip": d["kinetic_l2_error_outflow_strip"],
                           "mass_error": abs(d["final_mass"] - d["exact_mass"]) / d["exact_mass"],
                           "min_density": d["min_coefficient_density"], **{f"proj_{k}": v2 for k, v2 in pr.items()}}
                    summary["accuracy"].append(row)
                    it = pr.get("iterations", 0) / max(pr.get("calls", 1), 1)
                    lines.append(f"| {p_} | {E} | {N} | {v} | {row['kinetic']:.4e} | {row['strip']:.4e} | "
                                 f"{row['mass_error']:.3e} | {row['min_density']:.2e} | "
                                 + (f"{pr['augmented'] / pr['calls']:.2%} | {it:.2f} | {pr['max_residual']:.1e} |"
                                    if pr else "– | – | – |"))
    (out / "accuracy.md").write_text("\n".join(lines) + "\n")

    # ---- Scaling in N (E=64) and in E (N=7).
    def scaling_row(c, j, **key):
        pr = j["projection"]
        calls = max(pr["calls"], 1)
        return {**key, "wall_c6": c["wall_seconds"], "wall_proj": j["wall_seconds"], "ghost_c6": c["seconds"]["ghost"],
                "ghost_proj": j["seconds"]["ghost"], "calls": pr["calls"],
                "us_ghost_c6_per_node": 1e6 * c["seconds"]["ghost"] / calls,
                "us_ghost_proj_per_node": 1e6 * j["seconds"]["ghost"] / calls,
                "iterations_per_call": pr["iterations"] / calls, "max_iterations": pr["max_iterations"],
                "augmented_share": pr["augmented"] / calls, "identity_share": pr["identities"] / calls,
                "failures": pr["failures"], "max_residual": pr["max_residual"],
                "overhead": j["wall_seconds"] / c["wall_seconds"] - 1,
                "kinetic_c6": c["kinetic_l2_error"], "kinetic_proj": j["kinetic_l2_error"]}

    for N in (3, 5, 7, 11, 15, 21, 31):
        c, j = runs.get(f"order_c6_p1_E64_N{N}"), runs.get(f"order_proj_p1_E64_N{N}")
        if c and j:
            summary["order_scaling"].append(scaling_row(c, j, N=N, E=64))
    for E in (32, 64, 128, 256, 512, 1024):
        if E <= 256:
            c, j = runs.get(f"rt_c6_p1_E{E}_N7"), runs.get(f"rt_proj_p1_E{E}_N7")
        else:
            c, j = runs.get(f"grid_c6_p1_E{E}_N7"), runs.get(f"grid_proj_p1_E{E}_N7")
        if c and j:
            summary["grid_scaling"].append(scaling_row(c, j, N=7, E=E))
    lines = ["### Ordnung N (E=64, p=1)", "",
             "| N | Wand C6 [s] | Wand Proj [s] | Mehraufwand | Ghost/Knoten C6 [µs] | Ghost/Knoten Proj [µs] "
             "| Newton/Aufruf | max. Newton | Identität | ergänzt | Fehlschläge |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["order_scaling"]:
        lines.append(f"| {r['N']} | {r['wall_c6']:.1f} | {r['wall_proj']:.1f} | {r['overhead']:+.1%} | "
                     f"{r['us_ghost_c6_per_node']:.2f} | {r['us_ghost_proj_per_node']:.2f} | "
                     f"{r['iterations_per_call']:.2f} | {r['max_iterations']} | {r['identity_share']:.1%} | "
                     f"{r['augmented_share']:.2%} | {r['failures']} |")
    lines += ["", "### Gitter E (N=7, p=1)", "",
              "| E | Wand C6 [s] | Wand Proj [s] | Mehraufwand | Ghost/Knoten C6 [µs] | Ghost/Knoten Proj [µs] "
              "| Newton/Aufruf | max. Newton | Identität | ergänzt | Fehlschläge |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["grid_scaling"]:
        lines.append(f"| {r['E']} | {r['wall_c6']:.1f} | {r['wall_proj']:.1f} | {r['overhead']:+.1%} | "
                     f"{r['us_ghost_c6_per_node']:.2f} | {r['us_ghost_proj_per_node']:.2f} | "
                     f"{r['iterations_per_call']:.2f} | {r['max_iterations']} | {r['identity_share']:.1%} | "
                     f"{r['augmented_share']:.2%} | {r['failures']} |")
    if bench:
        lines += ["", "### Kernel-Benchmark (ein Thread, 2000 Zustände je Klasse)", "",
                  "| N | Regel | Punkte | Klasse | Trace dicht [µs] | Trace separabel [µs] | C6 [µs] | Proj kalt [µs] "
                  "| Proj warm [µs] | Newton kalt | Newton warm | ergänzt | Quadraturfehler Proj | Quadraturfehler C6 "
                  "| max. C6−Proj |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for b in bench:
            lines.append(f"| {b['N']} | {b['rule_label']} | {b['points']} | {b['class']} | {b['us_trace_dense']:.2f} | "
                         f"{b['us_trace_separable']:.2f} | {b['us_c6']:.2f} | {b['us_projection_cold']:.2f} | "
                         f"{b['us_projection_warm']:.2f} | {b['iterations_cold_mean']:.2f} | "
                         f"{b['iterations_warm_mean']:.2f} | {b['augmented_share']:.1%} | "
                         f"{b['max_outgoing_error_vs_fine_rule']:.1e} | {b['max_c6_error_vs_fine_rule']:.1e} | "
                         f"{b['max_c6_minus_projection']:.2f} |")
    (out / "scaling.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    # ---- Figures.
    # 1. Runtime speedup of the optimized boundary path (ref / C6), per p, over E for each N.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, p_ in zip(axes, (1, 2)):
        for N, ls in ((3, ":"), (7, "--"), (15, "-")):
            rows = [r for r in summary["runtime"] if r["p"] == p_ and r["N"] == N]
            if rows:
                ax.plot([r["E"] for r in rows], [r["boundary_ref"] / r["boundary_c6"] for r in rows], ls,
                        color=COLOR["c6"], marker="o", ms=5, lw=2, label=f"Rand, N={N}")
                ax.plot([r["E"] for r in rows], [r["wall_ref"] / r["wall_c6"] for r in rows], ls,
                        color=COLOR["ref"], marker="s", ms=5, lw=2, label=f"gesamt, N={N}")
        ax.set_xscale("log", base=2)
        ax.set_title(f"Q{p_}: Speedup gegenüber bdd5d3f", fontsize=9)
        ax.set_xlabel("Elemente je Achse E", fontsize=8)
        style(ax)
    axes[0].set_ylabel("Laufzeit ref / optimiert", fontsize=8)
    axes[1].legend(fontsize=7, frameon=False, ncol=2)
    save(fig, out, "fig_runtime_speedup")

    # 2. Accuracy: kinetic L2 error and outflow-strip error vs E (N=7 and 15, p=2).
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True)
    for col, N in enumerate((3, 7, 15)):
        for row, (key, title) in enumerate((("kinetic", "kin. L2-Fehler"), ("strip", "L2-Fehler Ausflussstreifen"))):
            ax = axes[row, col]
            for v in ("c6", "proj", "projN1"):
                rows = [r for r in summary["accuracy"] if r["p"] == 2 and r["N"] == N and r["variant"] == v]
                if rows:
                    ax.loglog([r["E"] for r in rows], [r[key] for r in rows], color=COLOR[v], marker=MARK[v],
                              ms=5, lw=2, label=LABEL[v])
            ax.set_xscale("log", base=2)
            ax.set_title(f"{title}, N={N}, Q2", fontsize=9)
            style(ax)
            if row == 1:
                ax.set_xlabel("E", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    save(fig, out, "fig_accuracy")

    # 3. Scaling in N: ghost cost per boundary node and Newton iterations.
    if summary["order_scaling"]:
        rs = summary["order_scaling"]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        Ns = [r["N"] for r in rs]
        axes[0].loglog(Ns, [r["us_ghost_c6_per_node"] for r in rs], color=COLOR["c6"], marker="o", lw=2, label="C6")
        axes[0].loglog(Ns, [r["us_ghost_proj_per_node"] for r in rs], color=COLOR["proj"], marker="^", lw=2,
                       label="Projektion")
        axes[0].set_title("Ghost-Stufe je Randknoten [µs]", fontsize=9)
        axes[0].legend(fontsize=7, frameon=False)
        axes[1].plot(Ns, [r["iterations_per_call"] for r in rs], color=COLOR["proj"], marker="^", lw=2)
        axes[1].set_title("Newton-Schritte je Aufruf (Mittel)", fontsize=9)
        axes[2].plot(Ns, [100 * r["overhead"] for r in rs], color=COLOR["proj"], marker="^", lw=2)
        axes[2].set_title("Mehraufwand Gesamtlaufzeit [%]", fontsize=9)
        for ax in axes:
            ax.set_xlabel("PN-Ordnung N", fontsize=8)
            style(ax)
        save(fig, out, "fig_scaling_order")
    if summary["grid_scaling"]:
        rs = summary["grid_scaling"]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        Es = [r["E"] for r in rs]
        axes[0].semilogx(Es, [r["us_ghost_c6_per_node"] for r in rs], color=COLOR["c6"], marker="o", lw=2, label="C6",
                         base=2)
        axes[0].semilogx(Es, [r["us_ghost_proj_per_node"] for r in rs], color=COLOR["proj"], marker="^", lw=2,
                         label="Projektion", base=2)
        axes[0].set_title("Ghost-Stufe je Randknoten [µs]", fontsize=9)
        axes[0].legend(fontsize=7, frameon=False)
        axes[1].semilogx(Es, [r["iterations_per_call"] for r in rs], color=COLOR["proj"], marker="^", lw=2, base=2)
        axes[1].set_title("Newton-Schritte je Aufruf (Mittel)", fontsize=9)
        axes[2].semilogx(Es, [100 * r["overhead"] for r in rs], color=COLOR["proj"], marker="^", lw=2, base=2)
        axes[2].set_title("Mehraufwand Gesamtlaufzeit [%]", fontsize=9)
        for ax in axes:
            ax.set_xlabel("Elemente je Achse E (N=7, Q1)", fontsize=8)
            style(ax)
        save(fig, out, "fig_scaling_grid")
    if bench:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
        for ax, cls in zip(axes, ("smooth", "peaked", "beam")):
            rows = [b for b in bench if b["class"] == cls and b["rule_label"] == "2N+1"]
            Ns = [b["N"] for b in rows]
            ax.loglog(Ns, [b["us_trace_dense"] for b in rows], color=COLOR["ref"], marker="s", lw=2, label="Trace dicht")
            ax.loglog(Ns, [b["us_trace_separable"] for b in rows], color=COLOR["c6"], marker="o", lw=2,
                      label="Trace separabel")
            ax.loglog(Ns, [b["us_projection_warm"] for b in rows], color=COLOR["proj"], marker="^", lw=2,
                      label="Projektion (warm)")
            ax.loglog(Ns, [b["us_c6"] for b in rows], color=COLOR["projN1"], marker="D", lw=2, label="C6 Positivteil")
            ax.set_title({"smooth": "glatt", "peaked": "gepeakt", "beam": "Strahl + Hintergrund"}[cls]
                         + ", Regel 2N+1", fontsize=9)
            ax.set_xlabel("PN-Ordnung N", fontsize=8)
            style(ax)
        axes[0].set_ylabel("µs je Randknoten (ein Thread)", fontsize=8)
        axes[0].legend(fontsize=7, frameon=False)
        save(fig, out, "fig_kernel")
    print(json.dumps({"runs": len(runs), "missing": len(missing), "bench_rows": len(bench)}))


if __name__ == "__main__":
    main()
