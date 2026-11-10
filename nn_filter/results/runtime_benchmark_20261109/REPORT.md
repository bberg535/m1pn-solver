# Laufzeitvergleich des NN-Filters

Lokaler Screening-Benchmark vom 09.11.2026: M1PN mit `m1pn_mass_correction`, Gitter 16², T=0.32, ein Thread, identische feste Schrittweite je PN-Grad. Drei Wiederholungen je P3/P5/P7/P9 und Filtermodus; Modus- und Gradreihenfolge wurden rotiert. Alle 36 Läufe schlossen ab, hatten identische akzeptierte Schritte innerhalb jeder Vergleichsgruppe, null Rückweisungen, globalen Bilanzrest <1e-9 und Kegeltreue.

Die Tabellenwerte sind Median aus drei Läufen. Gesamtzeit umfasst Setup, Solver und Ausgabe; Filterzeit ist die vom Solver separat gemessene Filterarbeit. Die Spannweite pro Messgröße ist in `runtime.json` und `runtime.csv` enthalten.

| PN | Gesamt off (s) | konstant (s) | NN (s) | NN vs off | NN vs konstant | Filter konstant (s) | Filter NN (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| P3 | 3.616 | 2.840 | 3.793 | +4.9% | +33.6% | 0.024 | 0.513 |
| P5 | 3.722 | 3.367 | 4.197 | +12.8% | +24.7% | 0.055 | 0.896 |
| P7 | 3.981 | 3.709 | 4.914 | +23.4% | +32.5% | 0.104 | 1.418 |
| P9 | 3.927 | 4.160 | 6.045 | +53.9% | +45.3% | 0.155 | 2.002 |

Das NN verursacht in diesem Setup zusätzliche Laufzeit gegenüber off und konstant; der Zuwachs wächst mit dem PN-Grad. Das ist ein lokaler Einzelmaschinenvergleich, kein allgemeiner Hardwarebenchmark. Drei Wiederholungen reichen für eine grobe belastbare Einordnung, aber nicht für enge Leistungsintervalle.

Rohmessungen: [runtime.csv](runtime.csv), vollständige Mediane/Spannweiten und Validierungsdaten: [runtime.json](runtime.json). Ausgabelogs und Felder liegen unter `../../tmp/runtime_benchmark_20261109/`.
