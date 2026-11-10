#!/usr/bin/env python3
"""Evaluate the 256/512 crossing campaign.

Usage: analyze.py CAMPAIGN_DIR RESULT_DIR

Every run is measured from its own final.csv and nodal_mass.csv with the
lumped-mass metrics of nn_filter/scripts/study.py against the analytic
kinetic crossing density at T=0.32.
"""
import csv, json, math, os, sys
camp, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
W, BG, T = .07, .001, .32
def ref(x, y):
    p = lambda a, b: math.exp(-(a * a + b * b) / (2 * W * W))
    return BG + p(x - T - .3, y - .5) + p(x - .5, y - T - .3)
rows, missing = [], []
for line in open(f"{camp}/tasks.tsv"):
    name = line.split("\t")[0]
    run = f"{camp}/runs/{name}"
    if not os.path.exists(run + "/summary.json"):
        missing.append(name)
        continue
    s = json.load(open(run + "/summary.json"))
    masses = [float(r["mass"]) for r in csv.DictReader(open(run + "/nodal_mass.csv"))]
    a = b = c = d = linf = 0.
    rmin = margin = float("inf"); rmax = 0.
    with open(run + "/final.csv") as f:
        reader = csv.reader(f); head = next(reader)
        ix, iy, iu = head.index("x"), head.index("y"), head.index("u0")
        for k, r in enumerate(reader):
            u0, u1, u2, u3 = (float(r[iu + i]) for i in range(4))
            rho = math.sqrt(4 * math.pi) * u0
            j = math.sqrt(4 * math.pi / 3) * math.sqrt(u1 * u1 + u2 * u2 + u3 * u3)
            x, y = float(r[ix]), float(r[iy]); R = ref(x, y); e = rho - R; m = masses[k]
            a += m * abs(e); b += m * abs(R); c += m * e * e; d += m * R * R
            linf = max(linf, abs(e)); rmin = min(rmin, rho); rmax = max(rmax, rho); margin = min(margin, rho - j)
    parts = name.split("_")
    method = "B " + parts[1] if parts[0] == "B" else f"EW {parts[1]} {parts[2]}"
    grid = int(parts[-2][1:]); order = int(parts[-1][1:])
    rows.append(dict(name=name, grid=grid, order=order, method=method, L1=a / b, L2=math.sqrt(c / d), Linf=linf,
                     rho_min=rmin, rho_max=rmax, cone=margin, steps=s["steps"],
                     total_seconds=s["total_seconds"], solver_seconds=s.get("solver_seconds", float("nan"))))
with open(out + "/comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r["grid"], r["order"], r["L1"])))
json.dump({"missing": missing, "completed": len(rows)}, open(out + "/status.json", "w"), indent=1)
get = {(r["grid"], r["order"], r["method"]): r for r in rows}
def cell(g, o, m, key="L1", fmt="{:.4f}"):
    r = get.get((g, o, m)); return fmt.format(r[key]) if r else "–"
L = ["# Element-WENO und Massenkorrektur mit Filter: 256² und 512²", "",
     f"Crossing-Studie, `T=0.32`, Standardprogramm `m1pn_fast` auf nx-Knoten (16 Threads je Lauf). "
     f"{len(rows)} von {len(rows) + len(missing)} Läufen abgeschlossen.", "",
     "B: Massenkorrektur mit Produktionseinstellungen (SSPRK2, CFL 1). EW: Element-WENO, SSPRK3, "
     "Schritt `min(Realisierbarkeit, Hochordnungs-Stabilität)`. Filterstärke konstant: P3 31.75, P7 16.52 (Paper), P11 6 (wie nx511).", "",
     "Relativer L1-Dichtefehler gegen die analytische kinetische Dichte (enthält den PN-Modellfehler).", "",
     "| Gitter | PN | B ohne | B konstant | B NN | EW p=1 | EW p=2 | EW p=3 | EW p=4 | EW p=1 +F | EW p=2 +F | EW p=3 +F | EW p=4 +F |",
     "|---|---|" + "---:|" * 11]
for g in (256, 512):
    for o in (3, 7, 11):
        cols = [cell(g, o, m) for m in ("B off", "B constant", "B nn")]
        cols += [cell(g, o, f"EW p{p} off") for p in (1, 2, 3, 4)]
        cols += [cell(g, o, f"EW p{p} constant") for p in (1, 2, 3, 4)]
        L.append(f"| {g}² | P{o} | " + " | ".join(cols) + " |")
L += ["", "`+F`: mit konstantem Filter. p=3 hat 255 bzw. 513 Intervalle pro Richtung.", "",
      "## Laufzeit (Gesamtzeit in s, 16 Threads)", "",
      "| Gitter | PN | B ohne | B konstant | EW p=1 | EW p=2 | EW p=3 | EW p=4 |", "|---|---|" + "---:|" * 6]
for g in (256, 512):
    for o in (3, 7, 11):
        cols = [cell(g, o, m, "total_seconds", "{:.0f}") for m in ("B off", "B constant")]
        cols += [cell(g, o, f"EW p{p} off", "total_seconds", "{:.0f}") for p in (1, 2, 3, 4)]
        L.append(f"| {g}² | P{o} | " + " | ".join(cols) + " |")
L += ["", "## Positivität", "",
      f"Kleinste Dichte über alle Läufe: {min(r['rho_min'] for r in rows):.3e}; kleinster Kegelabstand: {min(r['cone'] for r in rows):.3e}.",
      "", "Fehlende Läufe: " + (", ".join(missing) if missing else "keine"), ""]
open(out + "/REPORT.md", "w").write("\n".join(L))
print("\n".join(L))
