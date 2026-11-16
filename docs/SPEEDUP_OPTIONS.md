# Weitere Beschleunigungsoptionen: Messungen (15.11.2026)

Crossing-Studie, P7, T = 0.32. CPU-Läufe auf 3 NX-Servern (6 Ränge x 32 Threads, MPI-Branch; die NX-Knoten sind
EPYC 9354, Zen4), GPU-Läufe auf der A100 (tardis, `experimental/gpu-a100`). Genauigkeit: relativer L²-Fehler der
Dichte an den Elementecken gegen die PN-Lösung Q1 1024² (Rundungs-Unsicherheit dieser Referenz: 1,7e-4). Der
kinetische Fehler eignet sich nicht, er wird vom P7-Modellfehler dominiert (gröbere Gitter haben dort sogar weniger).
Rohdaten: `nx/tmp/rest_20261115/`.

| Variante | Fehler | Schleife (3 NX) | gegen Q1 512² SSPRK3 |
|---|---|---|---|
| Q1 512², SSPRK3 (Standard) | 1,93e-3 | 46,0 s | 1 |
| Q1 512², SSPRK(4,3) | 1,93e-3 | 33,9 s | 1,36x, gleiche Genauigkeit |
| Q1 512², gelumpte Masse (`--mass-iterations 0`, Approx-Branch) | 1,97e-3 | 37,0 s | 1,24x |
| Q1 512², gelumpt + SSPRK(4,3) | 1,97e-3 | 24,6 s | 1,87x |
| Q1 362² | 2,31e-3 | 28,2 s | |
| Q1 256² | 2,85e-3 | 17,3 s | |
| Q2 256² | 2,49e-3 | 52,9 s | schlechter als Q1 bei gleicher Zeit |
| Q2 128² | 3,57e-3 | 24,4 s | |
| Q3 170² | 2,99e-3 | 64,6 s | |

GPU: SSPRK(4,3) 10,2 s gesamt (Schleife 7,2 s) statt 13,8 s (10,9 s), Fehler 1,93e-3 wie SSPRK3. Q2 256² auf der
GPU 23,6 s (generische Kernel).

- Zeitintegrator: SSPRK(4,3) halbiert die Stufen pro Zeit nicht ganz (doppelter Schritt, 4 statt 3 Stufen) und lässt
  die Genauigkeit unverändert: 1,36x (CPU) bzw. 1,5x (GPU). Die Grenze ist weiterhin die Low-Order-CFL.
- Höherer Polynomgrad: lohnt sich nicht. Bei gleicher Genauigkeit ist Q1 schneller (Q2 256²: 2,49e-3 in 52,9 s; Q1
  erreicht das zwischen 362² und 512² in 28-46 s). Die Lösung ist nicht glatt genug (Limiter, Strahlkanten); Q1
  konvergiert hier nur mit Ordnung ~0,6.
- Elementlokale (gelumpte) Masse: ändert das Verfahren, die Genauigkeit hier aber kaum (+2 %). 1,24x allein, 1,87x mit
  SSPRK(4,3); skaliert besser mit Rängen (6 -> 8 Ränge: 37,0 -> 32,1 s).
- Mehr Ränke (Slabs): 8 statt 6 Ränge bringen bei 512² nichts (47,7 s statt 46,0 s), bei 1024² 1,15x (187,6 statt
  216,2 s). Die y-Kette begrenzt. Pencils wurden nicht gebaut (großer Umbau, Schätzung weiter ~10-30 % nur bei
  vielen Rängen).
- Roofline (Schätzung, die likwid-Uncore-Zähler liefern auf den Zen4-Knoten 0): STREAM-Triad eines NX-Sockels 279 GB/s.
  Mindestverkehr pro Stufe etwa 28 Feld-Durchgänge x 78 MB = 2,2 GB; ein Sockel braucht 20,5 ms pro Stufe, also
  ~107 GB/s = 38 % der Bandbreite (Obergrenze ~2,6x pro Sockel); auf 3 Servern nur ~12 % pro Sockel (Kette). Die A100
  erreicht ~0,9 TB/s (~55 % des Erreichbaren).
- Single Precision: nicht portiert (Kegelprädikat, Limiter und Rundungs-Nudges bauen auf binary64). Micro-Benchmark
  des dominanten Kernels (Linienlösung 513² x 37 bzw. 1025² x 37 auf der A100): float 1,63x bzw. 1,83x schneller
  als double. Bei der Rundungsempfindlichkeit des Problems (siehe ELEMENT_WENO_GPU.md) wären die Abweichungen bei
  T = 0.32 groß.
- Adaptives Gitter (nur Potenzial, aus den Zuständen 512² bei T = 0.02/0.08/0.32): Anteil der Elemente mit
  Dichtevariation > 1 % der Maximaldichte 10-14 %, > 0,1 %: 35-42 %. Mit 3 gröberen Stufen und globalem dt wären das
  7-9x bzw. 2,3-2,8x (Obergrenze ohne AMR-Overhead). Die PN-Wellen (Ray-Effekte) füllen große Teile des Gebiets.
- Adaptive Winkelordnung (Potenzial): Ordnung pro Knoten mit Abbruchschranke 1e-2 bzw. 1e-3 der Knotennorm: bei
  T = 0.32 brauchen 98-99 % der Knoten P7, Gewinn 1,0x (früh 1,1-1,3x). Lohnt sich nicht.

## Lazy-Limiter: Messung der Limiter-Aktivität (16.11.2026)

Frage: Lohnt es sich, Limiter und Schließung in Stufen ohne Eingriff zu überspringen? `summary.json` zählt dazu
`rate_calls` und `unlimited_rate_calls` (Stufen, in denen kein Element und kein Randknoten begrenzt wurde; reine
Diagnose). NX, CPU, 32 Kerne:

| Fall | Stufen ohne Eingriff | begrenzte Element-Stufen | Limiter + Schließung | Obergrenze Ersparnis |
|---|---:|---:|---:|---:|
| Glatter Transport am Rand, Q2 128², P7 | 100 % | 0 % | 18 % | ≤ 18 % |
| Crossing Q2 128², P15 | 80 % | 1,9 % | 9,5 % | ≤ 7,5 % |
| Crossing Q1 256², P7 | 27 % | 30 % | 38 % | ≤ 10 % |
| Line Source mit FCS, Q1 256², P7 | 24 % | 0,7 % | 13 % | ≤ 3 % |
| Line Source, Q1 256², P7 | 11 % | 8 % | 22 % | ≤ 2 % |
| Lattice Q1 224², P7 | 0 % | 9,6 % | 28 % | 0 % |

Die Obergrenze setzt voraus, dass Stufen ohne Eingriff vorher kostenlos erkannt werden. A posteriori (Stufe ohne
Limiter rechnen, Kegel prüfen, bei Verletzung wiederholen) kostet jede Fehlvorhersage eine ganze Rate und ändert das
Verfahren (der Limiter greift an den Bars ein, nicht erst am Ergebnis); elementweise bräuchte es einen Sensor, der
billiger ist als die Limiterprüfung selbst. Nicht umgesetzt.
