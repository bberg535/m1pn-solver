#!/usr/bin/env python3
"""Lattice (Paul (44)-(47)) analysis for element-WENO runs (docs/ELEMENT_HETEROGENEOUS.md, Sec. 5).

Usage: lattice_analysis.py OUTDIR RUN_DIR... [--b-lineout CSV] [--b-runs CSV]

Per run: steps, solver time, material integrals, global ledger and balance residual, cone margin, rho_max, the
Schotthoefer/Hauck quantities (mass at T, total absorption, absorption rate at T) and their relative deviation
from the kinetic reference (arXiv:2505.17284, Table 1, converted to our rho units by the factor 2*pi: their
concentration integrates over one hemisphere; mass + absorption = 2*pi*3.2 confirms it). Optionally the
historical path-B line-out y = 3.5 (N = 112, older binary) and rho_max of the 15 historical B runs.
Writes lattice_table.csv/.md, lattice_lineout.png/.pdf and lattice_maps.png/.pdf.
"""
import argparse
import csv
import json
import math
from pathlib import Path

SH = {"mass": 13.07, "absorbed_total": 7.034, "absorption_rate": 4.597}  # Table 1, QoI-5, QoI-4, QoI-3
SH_FACTOR = 2 * math.pi


def load_field(run):
    rows = list(csv.reader(open(run / "final.csv")))
    head = rows[0]
    i0 = head.index("u0")
    pts = [(float(r[1]), float(r[2]), math.sqrt(4 * math.pi) * float(r[i0])) for r in rows[1:]]
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--b-lineout", type=Path)
    ap.add_argument("--b-runs", type=Path)
    ns = ap.parse_args()
    ns.outdir.mkdir(parents=True, exist_ok=True)
    table = []
    fields = {}
    for run in ns.runs:
        c = json.load(open(run / "config.json"))
        s = json.load(open(run / "summary.json"))
        led = s["ledger"]
        pts = load_field(run)
        fields[run.name] = (c, pts)
        row = dict(run=run.name, P=c["pn_order"], Q=c["degree"], E=c["elements_per_axis"],
                   control_points=c["elements_per_axis"] * c["degree"] + 1, moments=c["active_moments"],
                   mirror="".join("xyz"[d] for d in range(3) if c["mirror_directions"][d]),
                   steps=s["steps"], dt=c["dt_selected"], solver_s=s["solver_seconds"], wall_s=s["total_seconds"],
                   rss_mb=s["peak_rss_kib_rank0"] / 1024,
                   int_sigma_a=c["material_integrals"]["sigma_a"], int_sigma_s=c["material_integrals"]["sigma_s"],
                   int_source=c["material_integrals"]["source"], volume=c["material_integrals"]["volume"],
                   mass=s["final_density_mass"], absorbed=led["absorbed"], absorption_rate=s["absorption_rate_final"],
                   outflow=led["boundary_outflow"], injected=led["injected"], nudge=led["nudge_mass"],
                   balance_rel=led["balance_relative"], min_margin=s["minimum_cone_margin"],
                   rho_max=max(p[2] for p in pts), limited_stages=s["limited_element_stages"])
        for key, ref in (("mass", "mass"), ("absorbed", "absorbed_total"), ("absorption_rate", "absorption_rate")):
            row["dev_" + key] = row[key] / (SH[ref] / SH_FACTOR) - 1
        table.append(row)
    keys = list(table[0].keys())
    with open(ns.outdir / "lattice_table.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    with open(ns.outdir / "lattice_table.md", "w") as f:
        f.write("| P | Q/E | Punkte | Momente | Spiegel | Schritte | Löser (s) | Masse (Abw.) | Absorption gesamt (Abw.) | "
                "Absorptionsrate T (Abw.) | Ausfluss | Bilanz rel. | min. Marge | ρ_max |\n|---|---|---:|---:|---|---:|---:|---|---|---|---:|---:|---:|---:|\n")
        for r in table:
            f.write(f"| P{r['P']} | Q{r['Q']}/{r['E']} | {r['control_points']} | {r['moments']} | {r['mirror'] or '-'} | {r['steps']} | "
                    f"{r['solver_s']:.2f} | {r['mass']:.4f} ({100 * r['dev_mass']:+.1f} %) | {r['absorbed']:.4f} "
                    f"({100 * r['dev_absorbed']:+.1f} %) | {r['absorption_rate']:.4f} ({100 * r['dev_absorption_rate']:+.1f} %) | "
                    f"{r['outflow']:.2e} | {r['balance_rel']:.1e} | {r['min_margin']:.1e} | {r['rho_max']:.4f} |\n")
        f.write(f"\nKinetische Referenz (Schotthöfer/Hauck Tab. 1, /2π): Masse {SH['mass'] / SH_FACTOR:.4f}, "
                f"Absorption gesamt {SH['absorbed_total'] / SH_FACTOR:.4f}, Absorptionsrate {SH['absorption_rate'] / SH_FACTOR:.4f}.\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    # Line-out y = 3.5
    fig, ax = plt.subplots(figsize=(7, 4))
    if ns.b_lineout and ns.b_lineout.exists():
        b = list(csv.DictReader(open(ns.b_lineout)))
        for order in sorted({int(r["order"]) for r in b}):
            if order in (3, 7, 11):
                xs = [float(r["x"]) for r in b if int(r["order"]) == order]
                ys = [float(r["density"]) for r in b if int(r["order"]) == order]
                ax.semilogy(xs, ys, "--", lw=1, label=f"B Q1/112 P{order} (historisch)")
    for name, (c, pts) in fields.items():
        if c["degree"] != 2:
            continue
        line = sorted((x, rho) for x, y, rho in pts if abs(y - 3.5) < 1e-9)
        ax.semilogy([p[0] for p in line], [max(p[1], 1e-12) for p in line], lw=1.3, label=f"EW Q2/{c['elements_per_axis']} P{c['pn_order']}")
    ax.set_xlabel("x"); ax.set_ylabel("Dichte ρ (y = 3.5, t = 3.2)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ns.outdir / f"lattice_lineout.{ext}", dpi=150)
    plt.close(fig)
    # Density maps (log) for Q2 runs
    sel = [(n, cp) for n, cp in fields.items() if cp[0]["degree"] == 2]
    fig, axes = plt.subplots(1, len(sel), figsize=(4 * len(sel), 3.6), squeeze=False)
    for ax, (name, (c, pts)) in zip(axes[0], sorted(sel, key=lambda t: t[1][0]["pn_order"])):
        xs = sorted({p[0] for p in pts}); ys = sorted({p[1] for p in pts})
        grid = {(p[0], p[1]): p[2] for p in pts}
        import numpy as np
        Z = np.array([[max(grid[(x, y)], 1e-10) for x in xs] for y in ys])
        im = ax.pcolormesh(xs, ys, Z, norm=LogNorm(vmin=1e-7, vmax=Z.max()), shading="auto", cmap="magma")
        ax.set_aspect("equal"); ax.set_title(f"EW Q2/{c['elements_per_axis']} P{c['pn_order']}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ns.outdir / f"lattice_maps.{ext}", dpi=150)
    plt.close(fig)
    print(open(ns.outdir / "lattice_table.md").read())


if __name__ == "__main__":
    main()
