# Element-WENO gegen Massenkorrektur mit Filter

Lokaler Vergleich vom 9.11.2026, `crossing-study`, `T=0.32`, gekoppelt (M1PN).
Vergleichsdaten: `nn_filter/results/{crossing,extended_orders}_20261109`
(Standard B, ohne Filter, konstante Paperstärke, NN). Rohdaten: `comparison.csv`.

Die Element-WENO-Dichte wird auf den Q1-Knoten der Filterstudie ausgewertet und
mit derselben Metrik gemessen: L1 mit gelumpter Masse, relativ zur analytischen
kinetischen Dichte. Der Fehler enthält den endlichen PN-Modellfehler.

## Relativer L1-Fehler, p=1 (identische Knotenzahl und Auswertepunkte)

| Gitter | PN | B ohne | B konstant | B NN | EW p=1, 1e-3 | EW p=1, 0.2 | EW 1e-3 / bestes B |
|---|---|---:|---:|---:|---:|---:|---:|
| 16² | P3 | 0.5654 | 0.4957 | 0.5442 | 0.5893 | 0.6650 | 1.19 |
| 32² | P3 | 0.4621 | 0.4037 | 0.4299 | 0.4886 | 0.4702 | 1.21 |
| 16² | P5 | 0.4585 | 0.4674 | 0.4664 | 0.4623 | 0.6086 | 1.01 |
| 16² | P7 | 0.4301 | 0.4347 | 0.4385 | 0.3925 | 0.5889 | 0.91 |
| 32² | P7 | 0.1562 | 0.1933 | 0.2159 | 0.1427 | 0.2335 | 0.91 |
| 16² | P9 | 0.4313 | 0.4322 | 0.4334 | 0.3844 | 0.5843 | 0.89 |

"Bestes B" ist der kleinste Wert der drei B-Spalten.

## Höherer Polynomgrad, gleiche Knotenzahl (EW, Gewicht 1e-3)

| Gitter | PN | p=1 | p=2 | p=3 | p=4 |
|---|---|---:|---:|---:|---:|
| 16² | P3 | 0.589 | 0.763 | 0.912 | 1.010 |
| 32² | P3 | 0.489 | 0.534 | 0.590 | 0.666 |
| 32² | P7 | 0.143 | 0.286 | 0.380 | 0.526 |

p=3 hat 15 bzw. 33 statt 16 bzw. 32 Intervalle pro Richtung.

Relativer L1-Fehler der Anfangsdaten (`t≈0`, Koeffizienten = Punktwerte):

| Gitter | p=1 | p=2 | p=3 | p=4 |
|---|---:|---:|---:|---:|
| 16² | 0.000 | 0.135 | 0.316 | 0.280 |
| 32² | 0.000 | 0.039 | 0.066 | 0.086 |
| 96² | 0.000 | 0.004 | 0.008 | 0.010 |

## Einordnung

- EW p=1 mit Gewicht 1e-3 ist bei P7 und P9 um rund 9–11 % genauer als die
  beste B-Variante, bei P3 rund 20 % schlechter, bei P5 gleichauf.
- Das Gewicht 1e-3 ist auf diesen Gittern meist besser als 0.2. Das kehrt die
  Tendenz der 96²-Läufe um.
- p>1 ist auf 16² und 32² nicht konkurrenzfähig. Die Glättung der Anfangsdaten
  erklärt nur einen Teil davon. Auf den 96²-Läufen war die Reihenfolge umgekehrt
  (siehe `docs/ELEMENT_WENO.md`), mit einer abweichenden Metrik auf 97 Punkten.
- Alle Läufe beider Verfahren bleiben positiv und im M1-Kegel; EW ohne
  rundungsbedingte Korrekturen.
- Halber Zeitschritt (P3, 32², p=2): L1 0.5411 gegen 0.5410.

## Grenzen

- Nur 16² und 32², nur materialfreies Crossing. Die Studie ist nicht auf
  Konvergenz angelegt.
- Laufzeiten sind nicht direkt vergleichbar: B rechnet mit rund 6–7-mal mehr
  Schritten (CFL-Politik `0.2`), und EW misst nur die Zeitschleife. Pro Schritt
  liegen beide bei P3/32² gleichauf (etwa 0.02 s).
- Die Randbehandlung und die Initialisierung für p>1 unterscheiden sich von B.
- `p=1` nutzt nicht das Argument für höhere Grade, das die Variante motiviert.

## Reproduktion

    python3 -I scripts/element_weno/compare_filter.py \
      tmp/build/m1pn_element_weno <solver>/nn_filter/results <ausgabe>
