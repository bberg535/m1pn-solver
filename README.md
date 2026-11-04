# PN-Massenkorrektur im nativen M1PN-MCL-Solver

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

## Bauen und testen

Vorhandene MFEM/MPI/Eigen-Installation verwenden; es wird nichts installiert:

```bash
bash scripts/build.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash scripts/test.sh
```

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
