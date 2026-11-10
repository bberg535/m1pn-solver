# Ergänzende NN-Filtertests für P5 und P9

Lokale Validierung vom 09.11.2026 mit den unveränderten Originalgewichten. Diese Ergänzung schließt die End-to-End-Lücke für Modellgrade P5 und P9; sie ist kein Konvergenz- oder Produktionsbenchmark.

## Abdeckung

- Regressionstests: 3/3 CTest-Tests bestanden (Algebra, CLI/Transport, Vergleich der Merkmalsaufbereitung mit dem Autoren-Code).
- End-to-End: je P5/P9 auf 16² mit M1PN- und PN-only-Kopplung sowie ohne Filter, konstanter Paperstärke und NN; 12 Läufe bis T=0.32.
- Innerhalb jeder Vergleichsgruppe stimmen Anfangsfeld, Knotenmassen und akzeptierte Zeitschrittfolge überein. Sämtliche Läufe wurden aus den gespeicherten Feldern unabhängig auf Bilanz, Filter-Dichteledger und Kegeldiagnostik geprüft.
- Zeitschrittprüfung: gekoppelte NN-Läufe für P5/P9 zusätzlich mit dt/2.
- Die bereits vorhandene P3/P7-Kampagne umfasst weitere 27 Läufe. Ihre Ergebnisdateien, alle 270 dort referenzierten Feld-/Konfigurationsdateien, vier Modellgrade und 31 gehashte Elternquellen wurden erneut verifiziert.

## Dichtefehler und Stabilität

Relativer massengewichteter L1-Dichtefehler zur analytischen kinetischen Crossing-Referenz bei T=0.32:

| PN | Kopplung | Ohne Filter | Konstant | NN | NN: min. Dichte | NN: min. Kegelabstand | NN: Bilanzrest |
|---|---|---:|---:|---:|---:|---:|---:|
| P5 | m1pn | 0.458522 | 0.467367 | 0.466381 | 0.000416228 | 6.38317e-05 | 4.34e-16 |
| P5 | pn-only | 0.557653 | 0.463289 | 0.462835 | -0.0690349 | -0.083939 | 3.92e-16 |
| P9 | m1pn | 0.431270 | 0.432182 | 0.433386 | 0.000536543 | 0.000208949 | 8.71e-16 |
| P9 | pn-only | 0.378170 | 0.382427 | 0.386521 | -0.0400158 | -0.0665305 | 7.57e-16 |

Im gekoppelten Lauf bleibt die Dichte für beide Grade positiv und im M1-Kegel. Im PN-only-Lauf bleiben Dichte und Kegel auch mit Filter verletzt; der Filter ist kein Ersatz für den M1-Limiter. Die NN-Stärke blieb in allen Läufen endlich und nichtnegativ, und der Dichtebeitrag des Filters war exakt null.

Das NN verbessert den Dichtefehler nicht durchgehend: bei gekoppeltem P5 ist es knapp besser als die konstante Stärke, aber schlechter als ungefiltert; bei gekoppeltem P9 ist es schlechter als beide Kontrollen. PN-only profitiert in diesen zwei Fällen ebenfalls nicht konsistent.

## Zeitschritt halbiert

| PN | Schritte dt | Schritte dt/2 | Dichtefeldänderung / Norm bei dt/2 | Bilanzrest | Min. Dichte | Min. Kegelabstand | Rückweisungen |
|---|---:|---:|---:|---:|---:|---:|---:|
| P5 | 584 | 1168 | 2.548e-06 | 8.99e-16 | 0.000416234 | 6.38336e-05 | 0 |
| P9 | 610 | 1220 | 1.449e-06 | 1.75e-15 | 0.000536548 | 0.000208944 | 0 |

Die Feldänderung ist die massengewichtete L1-Differenz zwischen dt und dt/2, geteilt durch die Dichtenorm der dt/2-Lösung. Sie ist eine Empfindlichkeitsmessung und keine Ordnungsschätzung. Beide Läufe beendeten T=0.32 mit verdoppelter Schrittzahl, ohne Rückweisungen, mit positiver Dichte und Kegeltreue.

## Reproduktion

Die 12 Hauptläufe wurden mit folgendem Befehl erzeugt:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/study.py \
  --output results/extended_orders_20261109 --grids 16 --orders 5 9
```

Die Rohfelder stehen unter `../../tmp/extended_orders_20261109/`; die Zeitschrittläufe liegen in `../../tmp/extended_orders_20261109/half_dt_retry/`. `results.json`, `results.csv`, `manifest.json` und `temporal_refinement.json` enthalten aggregierte Werte und Prüfsummen.
