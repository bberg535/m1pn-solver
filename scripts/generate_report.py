#!/usr/bin/env python3
"""Create a report from completed, immutable efficiency trial directories."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess

import numpy as np

from run_efficiency import ROOT, physical, read_field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suites", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, default=ROOT / "docs/RESULTS.md")
    args = ap.parse_args()
    lines = ["# Gemessener Vergleich — 4. November 2026", "",
             "Native MFEM-Portierung auf einer isolierten Kopie des bisherigen AFC-Codes. "
             "Die Python/Numba-Quellarchive der PDF lagen nicht vor. Die folgenden Werte "
             "stammen ausschließlich aus tatsächlich ausgeführten lokalen Läufen.", "",
             "Ein gebundener CPU-Kern, ein Thread, fünf sequenzielle Wiederholungen in "
             "wechselnder Reihenfolge, je eine ungemessene Aufwärmrunde. Solverzeit umfasst "
             "alle `advance`-Berechnungen und Prüfungen, ohne Aufbau und Datei-I/O. "
             "Die zusätzlich gespeicherte Prozesszeit enthält Aufwärmrunde und I/O.", "",
             "Alle Varianten einer Gruppe verwenden identische Gitter, PN-Ordnungen, "
             "Zeitschritte und Parameter: R2, korrigierte Biharmonik mit gamma=0.004, "
             "alte Element-Paarzerlegung und gelumpte PN-Flächenlast. "
             "Die konsistente Flächenlast und faktorisierten BH-Paare sind gesonderte Methodenvarianten.", ""]
    all_rows = []
    for suite in args.suites:
        data = json.loads((suite / "summary.json").read_text())
        plan = json.loads((suite / "plan.json").read_text())
        lines += ["## " + suite.name, "",
                  "Endzeiten: " + ", ".join(f"{c[0]}: {c[-1]*c[-2]:.6g}" for c in plan["cases"]) + ".", "",
                  "| Fall | Variante | Solver [s], Median | Min–Max [s] | Faktor zur Referenz | max. phys. Differenz | Abnahme |",
                  "|---|---|---:|---:|---:|---:|---|"]
        for r in data:
            all_rows.append(dict(suite=suite.name, **r))
            if r.get("completed", 0):
                lines.append(f"| {r['case']} | {r['variant']} | {r['median_seconds']:.6f} | "
                             f"{r['min_seconds']:.6f}–{r['max_seconds']:.6f} | {r['speedup']:.3f} | "
                             f"{r['max_physical_difference']:.3e} | {'bestanden' if r['passed'] else 'nicht bestanden'} |")
            else:
                lines.append(f"| {r['case']} | {r['variant']} | — | — | — | — | fehlgeschlagen ({r['completed']}/{r['repeats']}) |")
        lines += ["", "Dichtefehler sind massengewichtete nodale Fehler gegen die exakte kinetische "
                  "Transportlösung, sofern unten angegeben. Das ist keine Fehlerzerlegung in Raum, "
                  "Zeit, Winkel und Limiter.", "",
                  "| Fall | Variante | Dichte L1 | Dichte L2 | Dichte Linf | relatives L1 |",
                  "|---|---|---:|---:|---:|---:|"]
        for r in data:
            if "rho_L1" in r:
                lines.append(f"| {r['case']} | {r['variant']} | {r['rho_L1']:.6e} | {r['rho_L2']:.6e} | "
                             f"{r['rho_Linf']:.6e} | {r['rho_relative_L1']:.6e} |")
        trials = json.loads((suite / "trials.json").read_text())
        failed = [r for r in trials if r["status"] != "completed"]
        for r in failed:
            lines += ["", f"Fehlversuch {r['case']}/{r['variant']}/{r['repeat']}: " + r.get("error", "unbekannt").replace("\n", " ")]
        lines += ["", "Prozesszeit enthält Initialisierung, Aufwärmrunde und Datei-I/O; "
                  "sie ist separat von der reinen Solverzeit zu beurteilen.", "",
                  "| Fall | Variante | Prozesszeit Median [s] | Prozessfaktor |",
                  "|---|---|---:|---:|"]
        for r in data:
            selected = [t for t in trials if t["case"] == r["case"] and t["variant"] == r["variant"] and t["status"] == "completed"]
            base_name = "imex-direct" if r["case"].startswith("imex_") else "reference"
            base = [t["process_seconds"] for t in trials if t["case"] == r["case"] and t["variant"] == base_name and t["status"] == "completed"]
            if selected and base:
                elapsed = statistics.median(t["process_seconds"] for t in selected)
                lines.append(f"| {r['case']} | {r['variant']} | {elapsed:.6f} | {statistics.median(base)/elapsed:.3f} |")
        lines += ["", "Rohdaten und Einzelbefehle: `" + str(suite.relative_to(ROOT)) + "`. "
                  "`plan.json` bindet die ausführbaren Dateien an ihre SHA-256-Werte; "
                  "`trials.json` enthält alle Prozesse einschließlich Fehlern.", ""]
    explicit = [r for r in all_rows if not r["case"].startswith("imex_") and r["variant"] not in ("reference", "eager")]
    if explicit:
        max_pn = max(r.get("max_PN_difference", 0) for r in explicit)
        max_phys = max(r.get("max_physical_difference", 0) for r in explicit)
        lines += ["## Abnahme und Grenzen", "",
                  f"Maximale beobachtete Abweichung aller geprüften expliziten Optimierungen: "
                  f"PN-Koeffizienten {max_pn:.6e}, physikalische Momente {max_phys:.6e}. "
                  "Die Abnahmegrenze ist 1e-9 absolut an jedem registrierten Checkpoint.", "",
                  "IMEX-Euler und SSPRK3 sind verschiedene Zeitverfahren. Ihre Laufzeiten werden "
                  "nicht als operatoridentischer Speedup gegeneinander verrechnet. PCG und Defektiteration "
                  "werden gegen den direkten konsistenten Solve desselben IMEX-Schritts verglichen. "
                  "Für den 2D-Einstrompuls ist hier nur die Differenz zur diskreten direkten IMEX-Referenz "
                  "bestimmt; eine unabhängige kinetische 2D-Pulsreferenz wird nicht behauptet.", "",
                  "Die Messungen sind Prototypmessungen, kein Konvergenz- oder Stabilitätsbeweis. "
                  "Die künstliche Biharmonik und die nichtlineare Begrenzung beeinflussen den absoluten "
                  "Dichtefehler. Kein FCS-/Low-Rank-Speedup wurde mit diesen Zahlen multipliziert.", ""]
    cli_path = ROOT/"results/cli_v3/summary.json"
    if cli_path.exists():
        lines += ["## Tatsächliche CLI-Läufe mit Aufbau und Ausgabe", "",
                  "Je fünf Läufe bis T=0.04. Gesamtzeit wird im Programm ab vor der Initialisierung "
                  "gemessen; Prozesszeit umfasst zusätzlich Prozess-/MPI-Start und Programmende. "
                  "Beide Varianten schreiben dieselben Ausgaben und Checkpoints. Kein Aufwärmlauf.", "",
                  "| Fall | Variante | Aufbau [s] | Programm gesamt [s] | Prozess [s] | Prozessfaktor | Dichte L2 |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for r in json.loads(cli_path.read_text()):
            if "process_seconds" in r:
                lines.append(f"| {r['case']} | {r['variant']} | {r['initialization_seconds']:.6f} | {r['total_seconds']:.6f} | {r['process_seconds']:.6f} | {r['process_speedup']:.3f} | {r['rho_L2']:.6e} |")
            else: lines.append(f"| {r['case']} | {r['variant']} | — | — | — | — | fehlgeschlagen |")
        lines += ["", "Die zusätzliche Solveriteration R3 baut unbenutzte BH-Matrizen und "
                  "Routentabellen erst bei Bedarf auf. Der erste kurze R1-Aufbau war dadurch "
                  "teilweise langsamer; seine negativen Prozessmessungen bleiben vollständig erhalten.", ""]
    pulse_path = ROOT/"results/pulse_v3/results.json"
    if pulse_path.exists():
        lines += ["## Unabhängige 1D-Einstrompuls-Dichte", "",
                  "Q1 mit 32 Zellen, P3, IMEX-Euler, dt=1e-4 bis T=0.16, Pulseingang auf "
                  "der linken Seite für 0<=t<0.06, übriger Rand Vakuum. Referenz ist die "
                  "exakte kinetische Dichte nach Abschaltung: "
                  "rho(t,x)=max(0,min(1,x/(t-0.06))-x/t)/2. "
                  "Die Fehler enthalten Raum-, Zeit-, Winkel- und Limitereffekte.", "",
                  "| Solver | Dichte L1 | Dichte L2 | Dichte Linf | relatives L1 | max. Dichtedifferenz zur direkten IMEX-Referenz |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in json.loads(pulse_path.read_text()):
            if r["status"] == "completed":
                delta = f"{r['max_rho_difference']:.6e}" if 'max_rho_difference' in r else "Zeitgitter verschieden"
                lines.append(f"| {r['mode']} | {r['rho_L1']:.6e} | {r['rho_L2']:.6e} | {r['rho_Linf']:.6e} | {r['rho_relative_L1']:.6e} | {delta} |")
                if not r.get("field_gate_pass", False):
                    lines += ["", "Checkpoint-Gate nicht bestanden: " + r.get("gate_error", "Zustandsdifferenz überschreitet 1e-9"), ""]
            else: lines += [f"| {r['mode']} | — | — | — | — | fehlgeschlagen |", "", r.get("error", "")]
        lines += ["", "Einzellaufzeiten dieses Genauigkeitstests sind keine wiederholte Performanceabnahme. "
                  "Alle 1600 beziehungsweise durch Endzeit-Rundung zusätzlich entstehenden Schritte "
                  "werden als Checkpoints gegen die direkte Lösung geprüft.", ""]
    lines += ["## Tests und unabhängige Prüfung", "",
              "Die Abnahmeprotokolle stehen unter `results/ctest_v5.log`; frühere Fehlversuche "
              "bleiben erhalten. Der unabhängige numerische Quellaudit korrigierte während der "
              "Entwicklung eine falsche Drucktensorrichtung, die Reihenfolge der IMEX-Randlast, "
              "den endlichen Massenprädiktor und Nullspaltenbehandlung. Die zugehörigen "
              "Gegenbeispiele und Prüfungen sind Bestandteil der neuen Tests. "
              "`minimum_bar` wird bei Lazy-Zertifikaten nicht als vollständiges gemessenes "
              "Barminimum ausgewertet; Abnahme stützt sich auf Zertifikate und gespeicherte "
              "Zustände. Teilzeiten der Komponenten sind unvollständig; Laufzeitaussagen benutzen "
              "die einschließende Solver-/Programm-/Prozesszeit.", ""]
    args.output.write_text("\n".join(lines)+"\n")
    # Source and frozen-reference integrity, separate from run binary identities.
    manifest = json.loads((ROOT / "reference_manifest.json").read_text())
    checks = {name: hashlib.sha256((Path(manifest["source"])/name).read_bytes()).hexdigest() == digest
              and hashlib.sha256((ROOT/"reference"/name).read_bytes()).hexdigest() == digest
              for name, digest in manifest["files"].items()}
    require = all(checks.values())
    assert require, "Original or frozen reference changed"
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for base in ["src", "tests", "scripts"] for p in sorted((ROOT/base).rglob("*"))
               if p.is_file() and "__pycache__" not in str(p)}
    provenance = {"hostname": platform.node(), "platform": platform.platform(),
                  "original_and_frozen_reference_unchanged": checks, "sources": sources,
                  "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
                  "suites": [str(p.relative_to(ROOT)) for p in args.suites]}
    (ROOT / "results/provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        data = json.loads((args.suites[-1]/"summary.json").read_text())
        rows = [r for r in data if r["variant"] in ["cached", "cached-eager"] and r["passed"]]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh([r["case"]+" / "+r["variant"] for r in rows], [r["speedup"] for r in rows])
        ax.axvline(1, color="black", linewidth=.8); ax.set_xlabel("Solver speedup versus frozen native reference")
        fig.tight_layout(); fig.savefig(ROOT/"docs/speedup.pdf"); fig.savefig(ROOT/"docs/speedup.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    print(args.output)


if __name__ == "__main__": main()
