#!/usr/bin/env python3
"""Evaluate the line-source runs against the exact solutions.

Usage: analyze.py LINE_SOURCE_DIR RESULT_DIR

GH runs:   Ganapol line source (sigma_s=1, T=1, width .03, floor 1e-4) in
           physical coordinates, source at (0.5, 0.5).
GHS runs:  element-weno on the rescaled GH problem (unit square, sigma_s=3,
           T=1/3); mapped back with r = 3 r', rho = rho'/9 and reported as GH.
VOID runs: uncollided line source on the unit square (T=1/3, width .01,
           floor 9e-4).  Radii are reported in physical units (x3).
Errors use the lumped nodal masses: relative L1/L2 of rho = sqrt(4 pi) u0
against the exact rho at the node radius.  Profiles: ray y=0.5 (x>0.5) and
the diagonal.
"""
import csv, json, math, os, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

camp, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
REF = {"GH": dict(args=["--medium", "scattering", "--time", "1", "--sigma", ".03", "--floor", "1e-4",
                        "--rmax", "2.2"], scale=1., width=.03, floor=1e-4),
       "VOID": dict(args=["--medium", "void", "--time", str(1 / 3), "--sigma", ".01", "--floor", "9e-4",
                          "--rmax", ".72"], scale=3., width=.01, floor=9e-4)}
ref = {}
for k, v in REF.items():
    f = f"{out}/reference_{k}.csv"
    if not os.path.exists(f):
        subprocess.run([sys.executable, f"{here}/reference.py", *v["args"], "--out", f], check=True)
    d = np.loadtxt(f, delimiter=",", skiprows=1)
    ref[k] = d[:, 0], d[:, 1]


def load(path):
    with open(path) as fh:
        head = fh.readline().strip().split(",")
    ix, iy, iu = head.index("x"), head.index("y"), head.index("u0")
    d = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(ix, iy, iu, iu + 1, iu + 2, iu + 3))
    return d[:, 0], d[:, 1], math.sqrt(4 * math.pi) * d[:, 2], \
        math.sqrt(4 * math.pi / 3) * np.sqrt(d[:, 3] ** 2 + d[:, 4] ** 2 + d[:, 5] ** 2)


rows, missing, prof = [], [], {}
for line in open(f"{camp}/tasks.tsv"):
    name = line.split("\t")[0]
    run = f"{camp}/runs/{name}"
    if not os.path.exists(run + "/summary.json"):
        missing.append(name)
        continue
    kind = name.split("_")[0]
    s = json.load(open(run + "/summary.json"))
    m = np.loadtxt(run + "/nodal_mass.csv", delimiter=",", skiprows=1, usecols=1)
    x, y, rho, j = load(run + "/final.csv")
    x0, y0, rho0, _ = load(run + "/initial.csv")
    if kind == "GHS":  # exact rescaling to physical GH coordinates
        x, y, x0, y0 = (.5 + 3 * (v - .5) for v in (x, y, x0, y0))
        rho, j, rho0, m = rho / 9, j / 9, rho0 / 9, 9 * m
    case = "GH" if kind == "GHS" else kind
    r = np.hypot(x - .5, y - .5)
    R = np.interp(r, *ref[case])
    e = rho - R
    # Lumped initial mass against the exact 1 + floor * area (sampling of the narrow Gaussian).
    area = (x0.max() - x0.min()) * (y0.max() - y0.min())
    mass0_exact = 1 + REF[case]["floor"] * area
    parts = name.split("_")
    method = "B " + parts[2] if parts[1] == "B" else f"EW {parts[2]} {parts[3]}"
    rows.append(dict(name=name, case=case, grid=int(parts[-2][1:]), order=int(parts[-1][1:]), method=method,
                     L1=float(np.sum(m * abs(e)) / np.sum(m * abs(R))),
                     L2=float(math.sqrt(np.sum(m * e * e) / np.sum(m * R * R))),
                     mass0_error=float(np.sum(m * rho0) / mass0_exact - 1),
                     mass_ratio=float(np.sum(m * rho) / np.sum(m * rho0)),
                     rho_min=float(rho.min()), cone=float((rho - j).min()), steps=s["steps"],
                     total_seconds=s["total_seconds"]))
    sc = REF[case]["scale"] if kind != "GHS" else 1.
    ray = (abs(y - .5) < 1e-9) & (x >= .5)
    dia = (abs(x - y) < 1e-9) & (x >= .5)
    prof[name] = dict(ray=(sc * (x[ray] - .5), rho[ray]), dia=(sc * r[dia], rho[dia]))

with open(out + "/comparison.csv", "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(rows[0])); wtr.writeheader()
    wtr.writerows(sorted(rows, key=lambda q: (q["case"], q["grid"], q["order"], q["L1"])))
json.dump({"missing": missing, "completed": len(rows)}, open(out + "/status.json", "w"), indent=1)
get = {(q["case"], q["grid"], q["order"], q["method"]): q for q in rows}


def cell(*k, key="L1", fmt="{:.4f}"):
    q = get.get(k); return fmt.format(q[key]) if q else "–"


# Profiles: one figure per case and grid, one panel per PN order.
GHM = ["B off", "B constant", "B nn", "EW p1 off", "EW p2 off", "EW p3 off", "EW p4 off", "EW p3 constant"]
for kind, grid, methods in (("GH", 256, GHM), ("GH", 512, GHM),
                            ("VOID", 256, ["B off", "B constant", "EW p1 off", "EW p2 off", "EW p3 off", "EW p4 off",
                                           "EW p3 constant"])):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for col, order in enumerate((3, 7, 11)):
        for rowi, cut in enumerate(("ray", "dia")):
            ax = axes[rowi, col]
            rr, ph = ref[kind]
            ax.plot(REF[kind]["scale"] * rr, ph, "k-", lw=2, label="exakt")
            for meth in methods:
                q = get.get((kind, grid, order, meth))
                if not q: continue
                px, py = prof[q["name"]][cut]; o = np.argsort(px)
                ax.plot(px[o], py[o], lw=1, label=meth)
            ax.set_xlim(0, 1.3)
            ax.set_ylim(-.05 if kind == "GH" else -.1, 1.2 if kind == "GH" else 3.)
            ax.set_title(f"P{order}, {'y=0' if cut == 'ray' else 'Diagonale'}")
            ax.set_xlabel("Abstand vom Ursprung (physikalisch)"); ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=8)
    title = "Line Source mit Streuung (Ganapol), T=1" if kind == "GH" else "Line Source im Vakuum, T=1 (physikalisch)"
    fig.suptitle(f"{title}, {grid}² Zellen (B) bzw. gleiche Knotenzahl (EW)")
    fig.tight_layout(); fig.savefig(f"{out}/profiles_{kind}_g{grid}.png", dpi=110); plt.close(fig)

L = ["# Line Source gegen exakte Lösung", "",
     f"{len(rows)} von {len(rows) + len(missing)} Läufen abgeschlossen. Standardprogramm `m1pn_fast`, 16 Threads.", "",
     "**GH**: Benchmark nach Garrett & Hauck (2013): Gebiet [−1.5,1.5]², σ_s=1, σ_a=0, T=1, Gauß ς=0.03, "
     "Boden 1e-4, Vakuumrand. Referenz: Ganapol (AZURV1) Linienquelle, mit dem Gauß gefaltet "
     "(Formeln nach Bennett & McClarren, arXiv:2205.15783, Gl. 38–41; Massenkontrolle 1e-5).", "",
     "**VOID**: dasselbe ohne Material, auf das Einheitsquadrat skaliert (T=1/3, ς=0.01), damit Element-WENO "
     "mitlaufen kann. Referenz: analytische ungestreute Linienquelle, gefaltet.", "",
     "Relativer L1-Fehler der Dichte (enthält den PN-Abbruchfehler).", "",
     "**GHS**: Element-WENO rechnet auf [0,1]²; GH wird dafür exakt skaliert (σ_s=3, T=1/3, ς=0.01, "
     "Boden 9e-4, Filterstärke ×3) und auf physikalische Koordinaten zurückgerechnet (r=3r', ρ=ρ'/9).", "",
     "## GH (mit Streuung)", "",
     "| Gitter | PN | B ohne | B konst. | B NN | EW p1 | EW p2 | EW p3 | EW p4 | EW p1 +F | EW p2 +F | EW p3 +F | EW p4 +F | Fehler Anfangsmasse B |",
     "|---|---|" + "---:|" * 12]
for g in (256, 512):
    for o in (3, 7, 11):
        cols = [cell("GH", g, o, mm) for mm in ("B off", "B constant", "B nn")]
        cols += [cell("GH", g, o, f"EW p{p} {md}") for md in ("off", "constant") for p in (1, 2, 3, 4)]
        L.append(f"| {g}² | P{o} | " + " | ".join(cols)
                 + f" | {cell('GH', g, o, 'B off', key='mass0_error', fmt='{:+.1e}')} |")
L += ["", "## VOID (ohne Material, 256² bzw. gleiche Knotenzahl)", "",
      "| PN | B ohne | B konst. | B NN | EW p1 | EW p2 | EW p3 | EW p4 | EW p1 +F | EW p2 +F | EW p3 +F | EW p4 +F |",
      "|---|" + "---:|" * 11]
for o in (3, 7, 11):
    cols = [cell("VOID", 256, o, mm) for mm in ("B off", "B constant", "B nn")]
    cols += [cell("VOID", 256, o, f"EW p{p} {md}") for md in ("off", "constant") for p in (1, 2, 3, 4)]
    L.append(f"| P{o} | " + " | ".join(cols) + " |")
L += ["", "## Positivität und Laufzeit", "",
      "| Lauf | ρ_min | min(ρ−|J|) | Masse T/0 | Schritte | Zeit s |", "|---|---:|---:|---:|---:|---:|"]
for q in sorted(rows, key=lambda q: q["name"]):
    L.append(f"| {q['name']} | {q['rho_min']:.2e} | {q['cone']:.2e} | {q['mass_ratio']:.4f} | {q['steps']} | {q['total_seconds']:.0f} |")
L += ["", "Profile: `profiles_GH_g256.png`, `profiles_GH_g512.png`, `profiles_VOID_g256.png`.", "",
      "Fehlende Läufe: " + (", ".join(missing) if missing else "keine"), ""]
open(out + "/REPORT.md", "w").write("\n".join(L))
print("\n".join(L[:40]))
