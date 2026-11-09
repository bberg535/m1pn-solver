# Element-WENO gegen Massenkorrektur mit Filter: 64² und 128²

Lokaler Vergleich vom 9.11.2026, `crossing-study`, `T=0.32`, gekoppelt (M1PN), 66 von 66 Läufen abgeschlossen.
Variante B nutzt dieselbe Binary und dieselben Argumente wie `nn_filter/scripts/study.py`, mit `dt` aus der Null-Zeit-Probe und `--cfl .2`.
Element-WENO nutzt `--cfl .5` der Schranke (19). Beide Dichten werden auf den Q1-Knoten mit der gelumpten Masse gegen die analytische kinetische Dichte gemessen.
Der Fehler enthält den endlichen PN-Modellfehler. Die portierte Metrik reproduziert auf dem gespeicherten Feld 32²/P3/off den Wert der Filterstudie.

## Relativer L1-Fehler

| Gitter | PN | B ohne | B konstant | B NN | EW p=1 | EW p=2 | EW p=3 | EW p=4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 64² | P3 | 0.4786 | 0.3973 | 0.4320 | 0.4945 | 0.4740 | 0.4718 | 0.4841 |
| 64² | P5 | 0.3243 | 0.3191 | 0.3162 | 0.3330 | 0.3261 | 0.3447 | 0.3840 |
| 64² | P7 | 0.1054 | 0.1730 | 0.1975 | 0.1024 | 0.1086 | 0.1503 | 0.2652 |
| 64² | P9 | 0.0633 | 0.0893 | 0.1176 | 0.0651 | 0.0719 | 0.0921 | 0.2351 |
| 128² | P3 | 0.5242 | 0.4134 | 0.4649 | 0.5328 | 0.5078 | 0.4819 | 0.4621 |
| 128² | P7 | 0.1244 | 0.1848 | 0.2098 | 0.1233 | 0.1117 | 0.1195 | 0.1380 |

EW-Spalten mit linearem WENO-Gewicht 1e-3. p=3 hat 63 bzw. 129 Intervalle pro Richtung.

## Gewicht 0.2

| Gitter | PN | EW p=1 | EW p=2 | EW p=3 | EW p=4 |
|---|---|---:|---:|---:|---:|
| 64² | P3 | 0.4791 | 0.4651 | 0.4574 | 0.4829 |
| 64² | P5 | 0.3308 | 0.3259 | 0.3702 | 0.3871 |
| 64² | P7 | 0.1199 | 0.1345 | 0.2741 | 0.2864 |
| 64² | P9 | 0.0830 | 0.0937 | 0.2626 | 0.2712 |
| 128² | P3 | 0.5283 | 0.5012 | 0.4648 | 0.4493 |
| 128² | P7 | 0.1244 | 0.1162 | 0.1528 | 0.1427 |

## Laufzeit und Schritte (128², ein Thread, sechs Prozesse parallel)

| PN | Methode | Schritte | Gesamt (s) | s/Schritt |
|---|---|---:|---:|---:|
| P3 | B off | 4313 | 6117 | 1.42 |
| P3 | B constant | 4313 | 5350 | 1.24 |
| P3 | B nn | 4313 | 6872 | 1.59 |
| P3 | EW p=1 w1e-3 | 650 | 865 | 1.33 |
| P3 | EW p=2 w1e-3 | 494 | 441 | 0.89 |
| P3 | EW p=3 w1e-3 | 445 | 350 | 0.79 |
| P3 | EW p=4 w1e-3 | 416 | 312 | 0.75 |
| P7 | B off | 4808 | 6508 | 1.35 |
| P7 | B constant | 4808 | 6249 | 1.30 |
| P7 | B nn | 4808 | 7758 | 1.61 |
| P7 | EW p=1 w1e-3 | 650 | 1994 | 3.07 |
| P7 | EW p=2 w1e-3 | 494 | 1046 | 2.12 |
| P7 | EW p=3 w1e-3 | 445 | 853 | 1.92 |
| P7 | EW p=4 w1e-3 | 416 | 773 | 1.86 |

## Einordnung

- Filter senkt den Fehler von B nur bei P3 deutlich. Bei P7 und P9 ist B ohne Filter klar besser als mit Filter.
- Bei P3 ist B mit konstanter Stärke am besten (EW 9–15 % darüber). Bei P5 und P9 liegen B ohne Filter und EW p=1–2 innerhalb von 3 %, bei P7 ist EW um 3 bzw. 10 % besser.
- P3 ändert sich mit dem Gitter kaum: der PN-Modellfehler dominiert. Der Filter wirkt auf den Winkel, EW auf den Raum; beide adressieren nicht denselben Fehler.
- Höherer Grad hilft bei feinerem Gitter: 128²/P3 sinkt L1 mit p von 0.533 auf 0.462; 128²/P7 ist p=2 am besten, p=4 aber schlechter als p=1. Auf 16²/32² war p>1 durchweg schlechter.
- Beim Gewicht gibt es keine einheitlich bessere Wahl: 0.2 ist bei P3 meist besser, 1e-3 bei P7 und P9.
- Alle Läufe bleiben positiv und im Kegel auf den Auswertepunkten.

## Grenzen

- Laufzeiten sind nicht fair vergleichbar: B rechnet bei `--cfl .2`, EW bei der eigenen Schranke. Pro Schritt ist EW bei 128² teurer (s. Tabelle). Mit vergleichbarem Stabilitätsausnutzen könnten sich beide angleichen; das ist nicht gemessen.
- Nur materialfreies Crossing, keine feineren Gitter als 128², keine Zeitschrittstudie auf den feinen Gittern.
- Keine Kombination von Element-WENO mit Filter getestet.
