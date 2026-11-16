# PN-Massenkorrektur im nativen M1PN-MCL-Solver

> **Aktueller Stand und Standard (16.11.2026):** Branch `feature/system-202611`, Tag `standard-20261116`. Er enthält
> den bisherigen Standard `standard-20261114` (Element-WENO mit Positivitätsprojektion am Rand, optimierter Randpfad,
> DLRA-Speedups ohne DLRA, Details unten) und zusätzlich:
> - **SSPRK(4,3)** als Zeitintegrator (`--ssprk3` für das frühere SSPRK3; gleiche Genauigkeit, 1,36–1,5× schneller);
> - **MPI** für Element-WENO (Scheiben entlang der letzten Achse, bitgleich zu seriell;
>   [docs/ELEMENT_WENO_MPI.md](docs/ELEMENT_WENO_MPI.md), aus `feature/element-weno-mpi`);
> - den **GPU-Pfad** (`-DM1PN_ELEMENT_GPU=ON`, `m1pn_fast ... --gpu --mirror-reduction off`): Zeitschleife,
>   Ghost-Projektion, Reaktion und FCS auf der GPU; Abweichung zur CPU auf Rundungsniveau
>   ([docs/ELEMENT_WENO_GPU.md](docs/ELEMENT_WENO_GPU.md), [docs/TARDIS_CAMPAIGN.md](docs/TARDIS_CAMPAIGN.md));
> - **First-Collision-Source** für Line/Plane Source ([docs/FCS.md](docs/FCS.md)) und den **Lattice** mit exakter
>   Strahlverfolgung des ungestoßenen Anteils, CPU und GPU (`--case lattice-paul --fcs analytic`,
>   [docs/FCS_LATTICE.md](docs/FCS_LATTICE.md));
> - die Kampagnenskripte für tardis und NX (`scripts/element_weno/tardis_convergence.py`, `lattice_fcs_study.py`;
>   Ergebnisse [docs/TARDIS_RESULTS.md](docs/TARDIS_RESULTS.md)).
>
> Nicht im Standard: AMR (`experimental/amr`, bei gleicher Genauigkeit langsamer), genäherte Masseninverse
> (`experimental/approx-mass-inverse`, ändert das Verfahren), Schur-Löser für MPI (opt-in), Lazy-Limiter (gemessen:
> höchstens 7–18 % bei glatten Fällen, nichts bei Lattice/Crossing; [docs/SPEEDUP_OPTIONS.md](docs/SPEEDUP_OPTIONS.md)).
>
> **Vorheriger Standard (14.11.2026):** Branch `feature/boundary-positivity-projection`, Tag `standard-20261114`:
> - der **momententreuen Positivitätsprojektion am Rand** als Standard-Ghost (C6 nur noch mit `--c6-ghost`),
>   Projektionsregel N+1, Warmstart mit Kaltstart-Retry ([docs/POSITIVITY_PROJECTION.md](docs/POSITIVITY_PROJECTION.md));
> - dem **optimierten Randpfad und Zeitschritt** (separable Trace-Auswertung, Face-Zeilen, Thread-Skalierung, blockweises
>   SpMM; 1.6–3.0× schneller als `bdd5d3f`, [results/positivity](results/positivity/README.md));
> - den **Optimierungen aus dem DLRA-Zweig** (`846e6c9` schnelleres Setup/Initialisierung, `bdd5d3f` gebündelte
>   Randprodukte), **ohne DLRA** (DLRA selbst ist seit 16.11.2026 entfernt, [docs/DLRA.md](docs/DLRA.md)).
>
> Verworfen: adaptive PN-Ordnung (Branch gelöscht; auf NX bei 54–63 % aktiven Momenten 1.2–1.4× langsamer als feste
> Ordnung), Projektion im Hochordnungs-Randfluss (`--projection-high-order`, schlechter).

Isoliertes Experiment im eigenen Git-Repository, Branch
`experiment/m1pn-pn-mass-correction`. Der eingefrorene Ausgangsstand steht
unverändert in `baseline/src` (Manifest und Baseline-Commit `ce1e785`).

Der neue Standardpfad `m1pn_mass_correction` berechnet einen vollständigen
PN-Low-Order-Prädiktor und genau eine lokale Massenkorrektur. Der getrennte
Pfad `m1pn_mass_correction_boundary_consistent` setzt auf offenen uniformen
kartesischen P1/Q1-Gittern die Randprädiktoren aus dem nächstliegenden ganz
inneren Knoten fort. Beide verwenden den bestehenden M1-Low-Order-Operator,
Kegellimiter, physikalischen Randfluss und SSPRK3.

Der [Ergebnisbericht](docs/REPORT.md), die [Herleitung](docs/MATHEMATICS.md),
[Quellenprüfung](docs/LITERATURE.md), [Versuchsmatrix](docs/EXPERIMENT_PLAN.json)
und [Vergleichsdaten](results/comparison/analysis.json) enthalten Evidenz und Grenzen.

Die ausführliche interne Gegenprüfung steht in [AUDIT.md](docs/AUDIT.md). Die Ergebnisempfehlung lautet: bestehende Biharmonik für das Paper beibehalten. Die vier Vakuum-Einström-Stressläufe brechen ab; dies wird im Bericht gesondert ausgewiesen.

Die neue [NX-Clusterstudie](nx/README.md) überträgt den 48-Aufgaben-Plan der
Biharmonik-Studie auf Standardvariante B. Sie berechnet ihre Schrittweiten neu
und startet mit `python3 nx/scripts/start_mass_correction.py`.

Die [ersten NX-Vergleichsplots](results/nx_comparisons/README.md) umfassen
44 abgeschlossene Aufgaben. Der [lokale Vergleich von reinem P3 und
gekoppeltem M1PN auf 128²](results/pn_comparisons/20261105_123328/README.md)
zeigt die Wirkung der abgeschalteten M1-Kopplung bei gleichen Anfangsdaten
und gleicher Schrittweite.

Die [momententreue Positivitätsprojektion am Rand](docs/POSITIVITY_PROJECTION.md)
(Standard seit 14.11.2026, C6 mit `m1pn_element_weno --c6-ghost`) ersetzt den C6-Ghost durch die Projektion des PN-Traces auf
nichtnegative, momententreue Winkelmaße; der Randpfad wurde vorab optimiert (Ergebnisse unter
[results/positivity](results/positivity/README.md)).

## Standardprogramm `m1pn_fast`

`m1pn_fast` ist das Standardprogramm (Quelle `nn_filter/src/main.cpp`, gebaut mit
`M1PN_FAST_DEFAULTS`). **Seit 16.11.2026 (`feature/combine-202611`)** rechnet es ohne weitere Optionen mit
Element-WENO, SSPRK(4,3), `--cfl 1 --dt 0` (CFL-Scan: [docs/ELEMENT_WENO.md](docs/ELEMENT_WENO.md)) und den
physischen Kernen eines NUMA-Knotens. Die Massenkorrektur mit den Einstellungen der schnellen 512er-Läufe
(gelumpte Masse, SSPRK2, `--cfl 1`, `--kv-gamma 0`) gibt es mit `--stabilization m1pn_mass_correction`; dieser
Q1-Pfad ist bitgleich zur bisherigen `m1pn_nn_filter` mit diesen expliziten Flags. Ältere Skripte, die sich auf den
früheren Default verlassen, müssen `--stabilization m1pn_mass_correction` ergänzen.

Wählbar über Optionen:

| Verfahren | Option |
|---|---|
| Massenkorrektur | `--stabilization m1pn_mass_correction` |
| Biharmonik | `--stabilization biharmonic-corrected` bzw. `biharmonic-original`, `--biharmonic-gamma G` |
| Element-WENO (Standard; Bernstein `Q_p`, elementbasiertes Limiting) | `--stabilization element-weno --degree p` |
| Winkelfilter | `--filter constant --filter-strength S` oder `--filter nn` (nicht mit Element-WENO) |
| Line Source (Garrett–Hauck) | `--case line-source --domain-length 3 --pulse-width .03 --background 1e-4 --sigma-s 1 --time 1` (nur B; Referenz `scripts/line_source/reference.py`) |

Element-WENO läuft auf generierten kartesischen Gittern (konstante isotrope Streuung und Absorption, heterogene
Fälle `lattice-paul`, `lattice-3d`, `shadow-weih` mit Material und Quelle) und nutzt SSPRK(4,3) (Standard seit 15.11.2026, gleiche Genauigkeit,
1,36-1,5x schneller; `--ssprk3` für das frühere SSPRK3). Die Felddateien enthalten dieselben Spalten wie der Q1-Pfad, ausgewertet
auf einem gleichmäßigen Gitter mit `elements*degree` Zellen pro Achse. Details in
[docs/ELEMENT_WENO.md](docs/ELEMENT_WENO.md). Die bisherige `m1pn_nn_filter`
behält ihre alten Defaults, damit bestehende Studien reproduzierbar bleiben.

## Bauen und testen

Vorhandene MFEM/MPI/Eigen-Installation verwenden; es wird nichts installiert:

```bash
bash scripts/build.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash scripts/test.sh
```

Ein lokaler Crossing-Test mit Fortschritt im Terminal verwendet die bereits
gebaute Studienbinary aus `nx/tmp/build_local`:

```bash
# Kurzer Test: 16×16, P3, T=0.01, ein Thread.
python3 scripts/run_local_progress.py --time .01

# Vollständige Crossing-Zeit T=0.32, gleiche lokale Größe.
python3 scripts/run_local_progress.py

# Höhere PN-Ordnung auf demselben Gitter.
python3 scripts/run_local_progress.py --pn-order 7

# Größeres Gitter: Budget erhöhen und Schrittweite automatisch bestimmen.
python3 scripts/run_local_progress.py --elements 256 --pn-order 11 \
  --memory-mib 16384 --dt 0 --cfl .2
```

Die Anzeige enthält Prozent, Schrittzahl, Simulationszeit, Laufzeit und eine
geschätzte Restzeit. Neue Ergebnisse liegen unter `tmp/local_B_<Zeitstempel>`,
das Solverlog daneben. Nach dem Lauf werden die gespeicherten Felder auf den
M1-Kegel und die globale Bilanz geprüft. `Strg+C` beendet den lokalen Lauf und
behält die bisherigen Daten. Die Clusterjobs werden davon nicht beeinflusst.
`--memory-mib` setzt das interne Solverbudget (Standard 4096 MiB).
Mit `--dt 0` wird die Schrittweite als `cfl * min(M1-Grenze, PN-Grenze)`
berechnet; ein positiver `--dt`-Wert gibt sie fest vor und wird gegen beide
Grenzen geprüft. Der Standardwert bleibt `--dt .0001`.

Ein reiner PN-Lauf mit demselben Standard-B-Operator lässt sich über
`--coupling pn-only` starten. Er verwendet die vollständigen PN-Randflüsse
einschließlich vorgeschriebener Einstrahlung, aber keine M1-Abschlussrelation,
keinen Kegellimiter und keine Synchronisierung der niedrigen Momente. Negative
Dichten und Verletzungen des M1-Kegels werden erfasst und nicht begrenzt.
Dieser zusätzliche Studienpfad unterstützt derzeit das materialfreie
`crossing-study`-Problem mit Standard B und SSPRK3.

```bash
# P3 auf 128×128 bis T=0.32, gleiche Schrittweite wie der NX-Vergleichslauf.
python3 scripts/run_local_progress.py --coupling pn-only \
  --elements 128 --pn-order 3 --time .32 \
  --dt 6.459426725878078e-05 --threads 8
```

Die Ergebnisse liegen unter `tmp/local_PN_B_<Zeitstempel>`. Die automatische
Nachprüfung kontrolliert endliche Felder und die globale Bilanz aller
PN-Komponenten; sie protokolliert zusätzlich negative Dichten und
M1-Kegelverletzungen.

Der direkte Vergleich mit dem fertigen gekoppelten P3/128²-Lauf verlangt
identische gespeicherte Anfangsdaten, Knotenmassen und Zeitschritte:

```bash
python3 scripts/plot_pn_comparison.py \
  --pn-run tmp/local_PN_B_<Zeitstempel> \
  --coupled-run nx/tmp/pn_mc_production_20261104_161438_760701/runs/crossing_N128_P3 \
  --output results/pn_comparisons/<neuer_Ordner>
```

Den zuletzt abgeschlossenen lokalen Crossing-Lauf mit kinetischer Referenz,
signiertem Dichtefehler und Mittelschnitt plotten:

```bash
python3 scripts/plot_local_run.py
# Oder einen bestimmten Lauf auswählen:
python3 scripts/plot_local_run.py --run tmp/local_B_<Zeitstempel>
```

PNG, PDF, Schnittdaten und Herkunftsnachweis landen in einem neuen Ordner unter
`results/local_plots/`. Laufende oder abgebrochene Läufe werden übersprungen.

Ein bereits vorhandener anderer kompatibler Stack kann über `MFEM_CONFIG`
ausgewählt werden. Builds und große Rohfelder bleiben unter `tmp/`.

```bash
# Neue Methode; --mass-mode lumped wird automatisch gesetzt.
tmp/build/m1pn --dimension 2 --elements 16 --pn-order 7 \
  --case crossing --inflow crossing-transport \
  --stabilization m1pn_mass_correction \
  --time .02 --dt .0001 --threads 1 --output tmp/my_mc_run

# Getrennte Randvariante (mindestens drei Zellen je offener Achse).
tmp/build/m1pn --dimension 1 --elements 32 --pn-order 3 \
  --case affine --inflow affine --pn-boundary-quadrature consistent \
  --stabilization m1pn_mass_correction_boundary_consistent \
  --time .05 --dt .0001 --threads 1 --output tmp/my_mc_boundary_run
```

`--ledger-stride S` (Standard 64) steuert die Prüfung der Standardvariante B: Jeder
`S`-te Schritt, beginnend mit dem ersten, läuft über den vollständigen Pfad mit allen
Q-Momenten. Dazwischen wertet `src/fast_mc.cpp` Paarflüsse nur für die vier limitierten
Momente aus. In diesen Schritten decken `ledger_residual` und `target_recovery` nur diese
vier Momente ab, und `target_energy` in `stages.csv` ist `nan`. `--ledger-stride 1`
erzwingt auf jedem Schritt den vollständigen Pfad.

Ausgabeziele müssen neu sein. Ein ausdrücklich inkompatibler Massenmodus,
SSPRK2/IMEX oder `--model legacy` wird für die neuen Methoden abgewiesen.
`--pn-viscosity-scale` hat Standardwert 1 und muss mindestens 1 sein. Die
M1-Graphviskosität wird damit nicht verändert. Bisherige Standards bleiben erhalten.
Die neue Methode baut weder einen Biharmonik-/Ghost-Operator noch eine
konsistente Massenfaktorisierung auf. `mass_solve_calls` bleibt null.

## Messungen reproduzieren

```bash
python3 scripts/analyze_scalar.py --output results/my_scalar
for variant in A PN B C Aex; do
  for suite in patches convergence spectrum; do
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 tmp/build/pn_mc_study \
      --suite "$suite" --variant "$variant" --output "tmp/my_study"
  done
done
python3 scripts/run_mc_campaign.py --suite screen --variants A Aex B C --output tmp/my_screen
python3 scripts/run_mc_campaign.py --suite main --variants A Aex B C --output tmp/my_main
python3 scripts/run_mc_campaign.py --suite performance --variants A Aex B C --output tmp/my_performance
python3 scripts/run_mc_campaign.py --suite boundary --variants A Aex B C --output tmp/my_boundary
python3 scripts/run_mc_campaign.py --suite cfl --variants A Aex B C --output tmp/my_cfl
python3 scripts/run_mc_campaign.py --suite regression --output tmp/my_regression
python3 scripts/analyze_mc_results.py --help
```

A bezeichnet die erhaltene randkorrigierte Steifigkeitsbiharmonik mit
`gamma=.004`. Aex ist die vorhandene quartische Ghost-Gram-Biharmonik mit dem
bereits dokumentierten 2D-Koeffizienten `.00032911848351755427`; sie ist nur auf
offenen kartesischen Gittern definiert. PN ist der unstabilisierte konsistente
Galerkinpfad. B und C sind die beiden neuen Methoden. Diese Koeffizienten sind
keine Behauptung gleicher Dissipation. Insbesondere ist `gamma_eff=1/12`
eine spezifische asymptotische 1D-Innenraumaussage.

Die Fourier- und Energierechnungen im unabhängigen Skalarskript ersetzen keine
PN- oder MCL-Tests. Konservativität impliziert keine quadratische Dissipativität;
der Bericht enthält explizite Gegenbeispiele. Crossing-Fehler zur analytischen
kinetischen Dichte enthalten PN-Modellfehler. Abweichungen zwischen Methoden
werden separat mit ausgewiesenem Nenner berichtet.
