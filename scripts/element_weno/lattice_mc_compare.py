#!/usr/bin/env python3
"""Lattice results against the own Monte-Carlo reference (docs/LATTICE_MC.md).

Reads the MC JSON files of m1pn_lattice_mc (quantities at the final time, block "total") and the versioned solver
tables, and writes REPORT.md and comparison.csv with relative deviations from MC (and, for lattice-paul, from
Schotthoefer/Hauck Tab. 1 / 2pi).

    python3 -I scripts/element_weno/lattice_mc_compare.py --mc2d DIR/lattice-paul.json --mc3d DIR/lattice-3d.json \
        --out results/heterogeneous/lattice_mc_20261116
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LITERATURE = {"mass": 2.0802, "absorbed": 1.1195, "rate": 0.7316}  # Schotthoefer/Hauck Tab. 1 / (2 pi), S22
QOIS = ("mass", "absorbed", "rate")


def mc_final(path: Path) -> dict:
    d = json.loads(path.read_text())
    k = len(d["times"]) - 1
    tot = d["total"]
    out = {"histories": d["histories"], "seconds": d["seconds"], "seed": d["seed"], "time": d["times"][k]}
    for q in QOIS + ("outflow", "uncollided_mass", "uncollided_absorbed", "uncollided_rate"):
        out[q] = tot[q][k]
        out[q + "_stderr"] = tot[q + "_stderr"][k]
    return out


def solver_rows() -> list[dict]:
    rows: list[dict] = []
    fcs = ROOT / "results/heterogeneous/lattice_fcs_20261116/results.csv"
    for r in csv.DictReader(fcs.open()):
        if r["status"] != "done":
            continue
        rows.append({"case": "lattice-paul", "source": "lattice_fcs_20261116 (NX)", "label": r["label"],
                     "pn": int(r["pn"]), "p": 1, "elements": int(r["elements"]), "fcs": r["fcs"],
                     "mass": float(r["mass"]), "absorbed": float(r["absorbed"]), "rate": float(r["rate"])})
    for sub in ("lattice", "hetero"):
        f = ROOT / f"results/convergence/tardis_202611/{sub}/results.csv"
        for r in csv.DictReader(f.open()):
            if not r.get("mass"):
                continue
            case = "lattice-3d" if r["family"] == "lattice3d" else "lattice-paul"
            rows.append({"case": case, "source": f"tardis_202611/{sub} (A100)", "label": r["label"],
                         "pn": int(r["pn"]), "p": int(r["p"]), "elements": int(r["elements"]), "fcs": "off",
                         "mass": float(r["mass"]), "absorbed": float(r["ledger_absorbed"]),
                         "rate": float(r["absorption_rate"])})
    # Duplicates (same configuration in two campaigns) are kept: they document reproducibility across machines.
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc2d", type=Path, required=True)
    ap.add_argument("--mc3d", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    mc = {"lattice-paul": mc_final(a.mc2d), "lattice-3d": mc_final(a.mc3d)}
    rows = solver_rows()
    for r in rows:
        ref = mc[r["case"]]
        for q in QOIS:
            r[f"dev_mc_{q}"] = r[q] / ref[q] - 1
            if r["case"] == "lattice-paul":
                r[f"dev_lit_{q}"] = r[q] / LITERATURE[q] - 1
    a.out.mkdir(parents=True, exist_ok=True)
    keys = ["case", "source", "label", "pn", "p", "elements", "fcs", *QOIS,
            *[f"dev_mc_{q}" for q in QOIS], *[f"dev_lit_{q}" for q in QOIS]]
    with (a.out / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def pct(x: float | None) -> str:
        return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.3f} %"

    lines = ["# Lattice gegen die eigene Monte-Carlo-Referenz", "",
             "Erzeugt von `scripts/element_weno/lattice_mc_compare.py`; Verfahren in docs/LATTICE_MC.md.", "",
             "## Referenzwerte (Endzeit, Einheiten wie `summary.json`)", "",
             "| Fall | T | Historien | Masse | Absorption gesamt | Rate bei T | Ausfluss | ungestoßene Masse |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case, m in mc.items():
        lines.append(f"| {case} | {m['time']:g} | {m['histories']:.2g} | {m['mass']:.6f} ± {m['mass_stderr']:.1e} | "
                     f"{m['absorbed']:.6f} ± {m['absorbed_stderr']:.1e} | {m['rate']:.6f} ± {m['rate_stderr']:.1e} | "
                     f"{m['outflow']:.2e} | {m['uncollided_mass']:.6f} ± {m['uncollided_mass_stderr']:.1e} |")
    m = mc["lattice-paul"]
    lines += ["", "Literatur (Schotthöfer/Hauck Tab. 1 / 2π, S22, ψ0 = 0) gegen MC: " +
              ", ".join(f"{q} {LITERATURE[q]} ({pct(LITERATURE[q] / m[q] - 1)})" for q in QOIS) + ".", ""]
    for case in ("lattice-paul", "lattice-3d"):
        sel = sorted((r for r in rows if r["case"] == case),
                     key=lambda r: (r["fcs"], r["p"], r["pn"], r["elements"], r["source"]))
        lines += [f"## {case}: Abweichung gegen MC", "",
                  "| Quelle | P | Q | Gitter | FCS | Masse | Absorption | Rate |" +
                  (" Masse (Lit.) | Absorption (Lit.) | Rate (Lit.) |" if case == "lattice-paul" else ""),
                  "|---|---:|---:|---:|---|---:|---:|---:|" + ("---:|---:|---:|" if case == "lattice-paul" else "")]
        for r in sel:
            s = (f"| {r['source']} | {r['pn']} | {r['p']} | {r['elements']} | {r['fcs']} | "
                 f"{pct(r['dev_mc_mass'])} | {pct(r['dev_mc_absorbed'])} | {pct(r['dev_mc_rate'])} |")
            if case == "lattice-paul":
                s += f" {pct(r['dev_lit_mass'])} | {pct(r['dev_lit_absorbed'])} | {pct(r['dev_lit_rate'])} |"
            lines.append(s)
        lines.append("")
    (a.out / "REPORT.md").write_text("\n".join(lines))
    print(f"wrote {a.out / 'REPORT.md'} ({len(rows)} solver runs)")


if __name__ == "__main__":
    main()
