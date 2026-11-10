# NN-Filter aus Plumridge–Hauck–Schotthöfer im nativen M1PN-Solver

Isoliertes Experiment im neuen Unterordner `nn_filter/`. Der vorhandene
Standard-B-Massenkorrektur-Operator wird unverändert eingebunden. Änderungen
liegen ausschließlich in diesem Ordner. Die bereits vorhandenen Änderungen
im übergeordneten Repository wurden übernommen, nicht zurückgesetzt.
`parent_manifest.json` dokumentiert Commit und SHA256 des tatsächlich
verwendeten, nicht sauberen Elternstands.

Quelle: `02_literature/NN_Filter_PN.pdf`, *Data-Driven Filtering of the
Spherical Harmonics Method*, arXiv:2510.05452v1, Gleichungen (16)–(18),
(48)–(51), Tabelle 1 und Abschnitt 4.2.2. Trainierte 2D-Modelle aus
[benplumridge/NN_FPN](https://github.com/benplumridge/NN_FPN), eingefrorener
Commit `dba16410a4bbb0749897b12b04119f4ca061f647`.

## Umfang

* Originalgewichte für P3, P5, P7 und P9: vier tanh-Schichten, residuale
  Verbindungen, LayerNorm, abschließendes ReLU. Kein neues Training.
* C++/Eigen-Inferenz ohne PyTorch im Solver. Originalcheckpoints,
  exportierte Textgewichte, Prüfsummen und PyTorch-Testwerte in `models/`.
* `--filter off|constant|nn`, kombiniert mit dem vorhandenen gekoppelten
  M1PN-B-Pfad oder dem vorhandenen `--coupling pn-only`-Crossing-Pfad.
* Gefilterte Läufe sind auf 2D, CPU, R2 und Standard B beschränkt.
  Der NN-Modellgrad muss exakt zum PN-Grad passen. Der vorhandene
  PN-only-Pfad bleibt auf materialfreies `crossing-study` beschränkt.
* Getrennte Ausgabe des Filters in `filter_integral.csv` und
  `filter_strength.csv`; Bilanzkontrolle berücksichtigt alle PN-Momente.

Der Filter ergänzt die räumliche Massenkorrektur. Er ersetzt sie nicht.
Die Modelle sind für die FV-Diskretisierung der Autoren trainiert; ihre
Übertragung auf FE/M1PN ist ein Experiment, keine Reproduktion der
Paperfehler oder Garantie einer Verbesserung.

## Filter und Zeitschritt

Für jeden harmonischen Grad `l` gilt

```
F_l = (l/(N+1))^4
du_lm/dt = -sigma_f(u) * F_l * u_lm.
```

Alle `2*l+1` Komponenten eines Grades erhalten denselben Faktor. Insbesondere
ist `F_0=0`: das Dichtemoment wird im Filter bitgenau kopiert. Die drei
Flussmomente erhalten denselben Faktor aus `[0,1]`. Der reine Filterschritt
erhält daher den M1-Kegel und verringert die euklidische Norm der Momente.
Dies ist keine Positivitätsgarantie für den vollständigen ungekuppelten
PN-Transport.

Konstante Stärke wird exakt exponentiell integriert. Bei NN-Stärke wird
ein exponentieller Mittelpunktsschritt verwendet: Stärke am Eingang
auswerten, mit halbem Intervall einen Prädiktor bilden, dort erneut
auswerten, schließlich den Eingang über das volle Intervall dämpfen.
Je ein halber Filterschritt liegt vor und nach dem bestehenden
Material/SSPRK3-Transport-Schritt. Das ist bei glatter Zustandsabhängigkeit
eine zeitlich zweite Ordnung; eine allgemeine Konvergenzaussage über
ReLU- und Limiter-Schaltflächen wird damit nicht behauptet.

Der Filter integriert die Reaktion ohne zusätzliche explizite
Filter-CFL-Beschränkung. Die Schrittweitenkontrolle des Transportoperators
bleibt bestehen. Ein separater Test mit halbierter Schrittweite prüft
die Empfindlichkeit der vollständigen Lösung. Die Momentenbilanz lautet

```
sum_i [m_i*(u_final-u_initial) + boundary - reaction - filter]_i = 0.
```

In `filter_integral.csv` stehen bereits massengewichtete Änderungen;
die Dichtespalte ist exakt null. Rückgewiesene Versuche werden nicht
in die endgültige Filterbilanz übernommen. Filterlaufzeit und Zahl der
Inferenzauswertungen in `summary.json` schließen Wiederholungsversuche ein.

## Tatsächlicher Merkmalsvertrag der Originalgewichte

Der veröffentlichte 2D-Code weicht von der abstrakten PDF-Formel ab. Um die
Originalgewichte sinnvoll zu benutzen, wird sein tatsächlicher Vertrag aus
`2D/src/funcs_common.py` nachgebildet:

1. Gradnormen von `(sigma_a+sigma_s)*u`.
2. Gradnormen der komponentenweisen Wurzeln aus
   `(A_x*d_x u)^2 + (A_y*d_y u)^2`, mit Originalschwelle `1e-12`.
3. `sigma_s*u_0` und isotroper Quellkoeffizient `s_0`.
4. Jede Spalte wird über die räumlichen Knoten mit Mittelwert und
   Stichprobenstandardabweichung (`correction=1`, Nennerzusatz `1e-10`)
   normalisiert.

Im FE-Solver werden `d_x` und `d_y` durch `D^-1 C_x` bzw. `D^-1 C_y`
ersetzt. Diese Auswertung ist Teil des Diskretisierungstransfers.
Die volle harmonische Basis enthält die reduzierte 2D-Basis der Autoren;
deren `l+1` spiegelungssymmetrische Komponenten entsprechen hier den
Indizes mit geradem `l+m`. Die Schwelle wird auf diese Komponenten
angewandt, alle übrigen Komponenten tragen ohne künstliche Schwelle bei.
Die tatsächlich getesteten Crossing-Daten sind spiegelsymmetrisch in z.

Die räumliche Normalisierung macht die Stärke von der gesamten aktuellen
Momentaufnahme abhängig. Zudem ist die getrennte Norm der x/y-Beiträge
nicht dieselbe wie `P(A·grad u)` aus dem PDF. Für die übernommene
Merkmalsaufbereitung wird deshalb keine allgemeine räumliche
Rotationsinvarianz behauptet. Die gradweise Dämpfung bei gegebener
skalaren Stärke kommutiert mit harmonischen Rotationen.

In der Originaldatei ist die im PDF erwähnte Eingangs-BatchNorm auskommentiert.
Außerdem enthält der dortige Trainingszweig `tt_flag==0` die auffällige
Formel `sigs ** psi_0`; der Testzweig verwendet Multiplikation. Dieser
Trainingscode wird hier nicht ausgeführt oder nachträglich verändert.
Die archivierten Originaldateien unter `upstream/` dienen der Quellenprüfung.

## Bauen und testen

Vorhandenen Stack verwenden; keine Installation, keine Slurm-Submission:

```bash
cd 03_projects/M1PN_MassCorrection_20261104/nn_filter
bash scripts/build.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir tmp/build -R '^nn_filter_' --output-on-failure
```

Das Buildskript liest standardmäßig `MFEM_CONFIG` aus dem bestehenden
`../tmp/opt_20261109/build_opt/CMakeCache.txt`. Alternativ darf ausdrücklich
`MFEM_CONFIG=/pfad/zum/bereits/installierten/config.mk` gesetzt werden.
Der bestehende MFEM/MPI-Compiler wird aus dieser Konfiguration ausgelesen.

PyTorch wird nur zum erneuten Export und für den optionalen direkten
Merkmalsvergleich benötigt. Zum ursprünglichen Export wurde das vorhandene
PyTorch 2.8.0 verwendet. Die Vergleichstests mit C++ verwenden float64 und
identische, aus den float32-Checkpoints übernommene Gewichte.
`torch.load(weights_only=True)` erlaubt ausschließlich
die benötigten Tensor-/Modulklassen; upstream Initialisierungs- oder
Trainingscode wird vom Export nicht ausgeführt.

```bash
OMP_NUM_THREADS=1 python3 scripts/export_models.py
```

`src/main.cpp` ist eine isolierte Kopie des vorhandenen Studienprogramms mit
Filterargumenten, Filterschritten und Bilanzfeldern. Sie ist bereits vorhanden
und muss zum Bauen nicht erzeugt werden. `scripts/prepare_driver.py` erzeugt
sie bei einem bewusst gewünschten Abgleich mit dem Elternstand erneut;
dabei wird auch das Elternmanifest aktualisiert.

## Einzelner Lauf und Vergleichsstudie

```bash
tmp/build/m1pn_nn_filter --dimension 2 --elements 32 --pn-order 3 \
  --case crossing-study --inflow crossing-study-transport \
  --stabilization m1pn_mass_correction --filter nn \
  --time .32 --dt 0 --cfl .2 --threads 1 --cuda off --output tmp/my_nn_run

# Konstante 2D-P3-Stärke aus Tabelle 1:
tmp/build/m1pn_nn_filter --dimension 2 --elements 32 --pn-order 3 \
  --case crossing-study --inflow crossing-study-transport \
  --stabilization m1pn_mass_correction --filter constant --filter-strength 31.75 \
  --time .32 --dt 0 --cfl .2 --threads 1 --cuda off --output tmp/my_constant_run

python3 scripts/study.py --output results/my_new_crossing
python3 scripts/analyze.py --study results/my_new_crossing
```

Alle Ausgabeordner müssen neu sein. Ohne `--filter` ist der Filter aus.
Bei `--filter constant` muss eine positive Stärke ausdrücklich angegeben
werden, sonst ist die Standardstärke null. `--filter-model FILE` erlaubt
ein anderes exportiertes Modell mit passendem PN-Grad.

Die vorab festgelegte Studie steht in `protocol.json`. Sie vergleicht
P3/P7 auf 16²/32², beide Kopplungen, drei Filtervarianten und drei zusätzliche
halbierte Zeitschritte bei gekoppeltem P3/32²: insgesamt 27 Läufe bis T=0.32.
Die Eingangsfelder, Knotenmassen und akzeptierten Zeitfolgen müssen innerhalb
jedes Vergleichs identisch sein. Die Referenz ist die analytische kinetische
Crossing-Dichte; deren Fehler enthält auch den PN-Modellfehler.

Ergebnisse, Plots und Bewertung stehen im
[Versuchsbericht](results/crossing_20261109/REPORT.md).
