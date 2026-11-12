#!/usr/bin/env python3
"""Tables and accuracy-vs-time plots of a DLRA campaign (scripts/dlra/campaign.py).

Usage: report.py CAMPAIGN_DIR REFERENCE_GH.csv

Per run: family (LS line source with FCS, LAT lattice, CR crossing), P_N, variant, solver seconds, peak memory, rank,
error against the reference (LS: Ganapol, relative L1 of rho; CR: exact kinetic density, relative L1; LAT: deviation
of mass, total absorption and absorption rate at T from Schotthoefer/Hauck) and the relative L1 distance of the
density to the full-rank run with the same P_N (same sample grid). Writes report.csv, report.md, accuracy_time.png/pdf.
"""
import csv, json, math, sys
from pathlib import Path
import numpy as np

SH = {"mass": 13.07, "absorbed": 7.034, "absorption_rate": 4.597}
SH_FACTOR = 2 * math.pi


def field(run):
    with open(run / "final.csv") as fh:
        head = fh.readline().strip().split(",")
    ix, iy, iu = head.index("x"), head.index("y"), head.index("u0")
    d = np.loadtxt(run / "final.csv", delimiter=",", skiprows=1, usecols=(ix, iy, iu))
    m = np.loadtxt(run / "nodal_mass.csv", delimiter=",", skiprows=1, usecols=1)
    return d[:, 0], d[:, 1], math.sqrt(4 * math.pi) * d[:, 2], m


def main():
    camp, ref_path = Path(sys.argv[1]), sys.argv[2]
    ref = np.loadtxt(ref_path, delimiter=",", skiprows=1)
    rows = []
    fields = {}
    for line in open(camp / "tasks.tsv"):
        name = line.split("\t")[0]
        run = camp / "runs" / name
        if not (run / "summary.json").exists():
            continue
        s = json.load(open(run / "summary.json"))
        c = json.load(open(run / "config.json"))
        fam, kind = name.split("_")[0], name.split("_")[1]
        variant = "_".join(name.split("_")[1:-1])
        x, y, rho, m = field(run)
        fields[name] = (rho, m)
        row = dict(name=name, family=fam, P=c["pn_order"], variant=variant, moments=c["active_moments"],
                   nodes=c["coefficients"], steps=s["steps"], solver_s=s["solver_seconds"],
                   rss_mib=s["peak_rss_kib_rank0"] / 1024, balance_rel=s["ledger"]["balance_relative"],
                   rank_mean="", rank_max="")
        if s.get("dlra", {}).get("enabled"):
            row.update(rank_mean=round(s["dlra"]["mean_rank"], 1), rank_max=s["dlra"]["max_rank"])
        if fam == "LS":
            X, Y, R_rho, M = .5 + 3 * (x - .5), .5 + 3 * (y - .5), rho / 9, 9 * m
            R = np.interp(np.hypot(X - .5, Y - .5), ref[:, 0], ref[:, 1])
            row["error"] = float(np.sum(M * abs(R_rho - R)) / np.sum(M * abs(R)))
        elif fam == "CR":
            T, w, bg = s["time"], .07, c["background"]
            p = lambda cx, cy: np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w * w))
            R = bg + p(.3 + T, .5) + p(.5, .3 + T)
            row["error"] = float(np.sum(m * abs(rho - R)) / np.sum(m * abs(R)))
        else:
            dev = dict(mass=s["final_density_mass"], absorbed=s["ledger"]["absorbed"],
                       absorption_rate=s["absorption_rate_final"])
            for k, v in dev.items():
                row["dev_" + k] = v / (SH[k] / SH_FACTOR) - 1
            row["error"] = max(abs(row["dev_" + k]) for k in dev)
        rows.append(row)
    # Distance to the full-rank run (mirror auto) with the same family and P_N.
    for row in rows:
        full = f"{row['family']}_full_mauto_P{row['P']}" if row["family"] != "CR" else f"CR_full_P{row['P']}"
        if full in fields and full != row["name"]:
            a, m = fields[full]
            b, _ = fields[row["name"]]
            row["dist_full"] = float(np.sum(m * abs(a - b)) / np.sum(m * abs(a))) if len(a) == len(b) else ""
        else:
            row["dist_full"] = ""
    keys = ["name", "family", "P", "variant", "moments", "nodes", "steps", "solver_s", "rss_mib", "rank_mean",
            "rank_max", "error", "dist_full", "dev_mass", "dev_absorbed", "dev_absorption_rate", "balance_rel"]
    with open(camp / "report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    titles = {"LS": "Line Source mit FCS (GHS, Q1/128, T = 1/3): relativer L1-Fehler gegen Ganapol",
              "LAT": "Lattice (Q1/112, T = 3.2): max. |Abweichung| von Masse, Absorption, Rate gegen Schotthöfer/Hauck",
              "CR": "Crossing (Q2/64, T = 0.32): relativer L1-Fehler gegen die kinetische Dichte"}
    fmt = lambda v, f: (f.format(v) if isinstance(v, (int, float)) else "–")
    with open(camp / "report.md", "w") as f:
        for fam in ("LS", "LAT", "CR"):
            sel = sorted([r for r in rows if r["family"] == fam], key=lambda r: (r["P"], r["variant"]))
            if not sel:
                continue
            f.write(f"## {titles[fam]}\n\n| Lauf | P | q aktiv | Knoten | Rang (mittel/max) | Fehler | Abstand zu Vollrang | "
                    f"Löser (s) | RSS (MiB) |\n|---|---:|---:|---:|---|---:|---:|---:|---:|\n")
            for r in sel:
                rank = f"{r['rank_mean']}/{r['rank_max']}" if r["rank_max"] != "" else "–"
                f.write(f"| {r['variant']} | {r['P']} | {r['moments']} | {r['nodes']} | {rank} | {r['error']:.4f} | "
                        f"{fmt(r['dist_full'], '{:.4f}')} | {r['solver_s']:.1f} | {r['rss_mib']:.0f} |\n")
            f.write("\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = [fam for fam in ("LS", "LAT", "CR") if any(r["family"] == fam for r in rows)]
    fig, axes = plt.subplots(1, len(fams), figsize=(6 * len(fams), 4.6), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        groups = {}
        for r in rows:
            if r["family"] != fam:
                continue
            key = r["variant"] if not r["variant"].startswith("dlra_r") else r["variant"]
            groups.setdefault(key, []).append(r)
        for key, g in sorted(groups.items()):
            g = sorted(g, key=lambda r: r["P"])
            style = "-o" if key.startswith("full") else "--s"
            ax.loglog([r["solver_s"] for r in g], [r["error"] for r in g], style, ms=4, lw=1.2, label=key)
            for r in g:
                ax.annotate(f"P{r['P']}", (r["solver_s"], r["error"]), fontsize=6, xytext=(2, 2),
                            textcoords="offset points")
        ax.set_xlabel("Lösezeit (s, 6 Threads)")
        ax.set_ylabel("Fehler")
        ax.set_title(fam, fontsize=10)
        ax.grid(alpha=.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(camp / f"accuracy_time.{ext}", dpi=140)
    print(open(camp / "report.md").read())


if __name__ == "__main__":
    main()
