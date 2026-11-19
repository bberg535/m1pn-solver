# Große Umbauten (Branch `experiment/big-rebuilds`, ab 19.11.2026)

Aufbauend auf `experiment/break-dt` (geteilter Schritt an Trennebenen, docs/BREAK_DT.md). Auftrag des Nutzers: die
großen Punkte der Optimierungsliste testen. Bezug: lattice-paul, GB = `--cells-per-unit 32,64,128,128,128,64,32`,
Trennebenen mit geteiltem Schritt, Spiegel in x, A100, 16 Kerne; Standard GB P35 126 s, P21 64 s (Job 162569).

## 1. SSPRK(n²,3) statt SSPRK(4,3) (`--ssp-n n`)

Ketchesons SSPRK(n²,3): n² Euler-Stufen der Länge Δt/(n² − n), SSP-Koeffizient n² − n, gespeicherter Zustand nach
(n−1)(n−2)/2 Stufen, Kombination n/(2n−1)·y_m + (n−1)/(2n−1)·(x + τL(x)) nach n(n+1)/2 − 1 Stufen. n = 2 ist exakt
SSPRK(4,3) (Gewichte 1/6, 1/6, 1/6, 1/2, Zeiten 0, ½, 1, ½; nachgeprüft). Stabilitätsfunktion dritter Ordnung für
n = 2..4 nachgeprüft (Vorsicht: vertauschte Kombinationsgewichte ergeben ein Verfahren erster Ordnung).

Stabilitätsintervalle (Bisektion auf |R(z)| = 1):

| Verfahren | Stufen | reelle Achse | je Stufe | imaginäre Achse je Stufe | C/s |
|---|---|---|---|---|---|
| SSPRK(4,3) | 4 | 5,149 | 1,287 | 0,539 | 0,5 |
| SSPRK(9,3) | 9 | 13,290 | 1,477 | 0,458 | 0,667 |
| SSPRK(16,3) | 16 | 25,419 | 1,589 | 0,374 | 0,75 |

Die Hochordnungsschranke nutzt die reelle Grenze je Verfahren; die Realisierbarkeitsschranke das C. Umsetzung:
`ElementWeno::ssp_plan` (Stufenplan mit Zeiten und Bilanzgewichten), `advance_ssp_n3` (CPU), `ElementWenoGpu::step`
(drei Puffer, y_m in einem davon). Ohne Option bitgleich.

Ergebnisse (Job 162661, GB, Trennebenen geteilt, Spiegel, gegen MC):

| Verfahren | Schritte | P35 Zeit | P35 Masse/Absorption/Rate % | P35 Feld L1 | P21 Zeit | P21 Rate % |
|---|---|---|---|---|---|---|
| SSPRK(4,3) | 1059 | 126 s | −0,026 / +0,042 / +0,049 | 7,7e-4 | 64 s | +0,132 |
| SSPRK(9,3) | 411 | 107 s | −0,025 / +0,040 / +0,031 | 7,8e-4 | 55 s | +0,114 |
| SSPRK(16,3) | 215 | 99 s | −0,024 / +0,038 / +0,013 | 7,7e-4 | 51 s | +0,096 |

Stabil bei CFL 1 mit der jeweiligen Grenze; 15 % bzw. 21 % schneller. Masse, Absorption und Feld fast unverändert,
die Rate am Endzeitpunkt verschiebt sich aber um 0,018 bzw. 0,036 Prozentpunkte: Zeitfehler (größere Schritte,
Strang-Kopplung zweiter Ordnung), der zufällig zur MC-Referenz hin zeigt. Für SSPRK(4,3) liegt der Zeitfehler unter
0,01 Prozentpunkten (Paper). Kleiner Testfall (CPU, P7, gegen Δt = 5e-4): Rate 3,9e-4 / 1,5e-3 / 9,4e-3, Feld
9e-4 / 5,5e-3 / 3,1e-2. CPU gegen GPU bei gleichem Δt (SSPRK(9,3), ohne Trennebenen): Integrale 4e-9, Dichte 2,3e-7
max (30 von 540 000 Limiter-Entscheidungen verschieden; größeres Δt bringt den Limiter näher an die Kegelgrenze).
**Bewertung:** brauchbar als Option für schnelle Läufe; kein Standard, weil der Zeitfehler die Rate sichtbar ändert.

## 2. Ein CUDA-Stream je GPU-Kontext, Blöcke gleichzeitig

Alle 68 Kernelstarts, die asynchronen Kopien/Memsets, Events und das cuBLAS-Handle laufen auf `ctx->stream`
(blockierender Stream: verbleibende Operationen auf dem Legacy-Stream, z. B. `upload_async` und synchrone Kopien,
bleiben korrekt geordnet). `m1pn_blocks` bedient die Blöcke einer Stufenphase gleichzeitig (OpenMP, verschachtelt:
`min(B, threads)` Blöcke zugleich mit je `threads / min(B, threads)` Threads für ihre Host-Phasen, größte zuerst;
`M1PN_BLOCKS_SERIAL=1`: nacheinander). Zwischen den Phasen bleibt die Barriere.

Prüfung (Job 162663): ein Kontext bitgleich zum Binary vor dem Umbau (GB P35 geteilt, t = 0,4); GPU-Validierung
29/29; Blöcke seriell gegen gleichzeitig bitgleich. Zeiten 364², P21-Karten: 14×14 (71 Blöcke) 74,6 → 25,7 s,
7×7 (21 Blöcke) 39,2 → 38,0 s. Erste Fassung ohne verschachtelte Threads: ein einzelner Block lief mit einem
Host-Thread (130 s statt ~40 s); behoben (B = 1 ohne äußere Schleife, sonst Threads aufgeteilt).

## 3. Ordnungskarten auf dem abgestuften Gitter (`m1pn_blocks --cells-per-unit`)

Globale Ebenen wie `m1pn_fast --cells-per-unit`, jeder Block erhält die Ebenen seines Bereichs (konforme Blockflächen,
knotenweise Kopplung); Trennebenen im Blockinneren aus den Ebenen; `--split-breaks on` für Blöcke mit inneren
Trennebenen. Ein Block mit GB-Ebenen ist in den Integralen bitgleich zu `m1pn_fast` (CPU, P7, t = 0,5).
Einschränkung: Blockgrenzen sind einseitige Knoten wie Trennebenen, der geteilte Schritt deckt sie nicht ab (Partner
in einem anderen Block); der Blocktreiber hat keine Spiegelreduktion.

**Fehler gefunden und behoben (077789d):** Die Klassentabellen des Streifenkerns lagen im `__constant__`-Speicher, den
alle GPU-Kontexte teilen. Mit mehreren abgestuften Blöcken überschrieb der zuletzt aufgebaute Block die Tabellen der
anderen: falsche Ergebnisse (Masse 0,659 statt 0,680 ohne Trennebenen) und Abbrüche „limiter endpoint rounding check
failed“ mit Trennebenen (auch seriell, Job 162676). Jetzt je Kontext im globalen Speicher (`__ldg`); CPU gegen GPU
1,9e-11 (ohne) / 7,8e-14 (mit Trennebenen), Job 162681. Ein Kontext bleibt bitgleich.

**Geteilter Schritt an Blockgrenzen (18623bd):** Blockgrenzflächen zählen wie Trennebenen (`split_face`, `split_row`;
eigene Schranke, getrennte Summen). Die Stufe zerfällt in zwei Hälften mit Barriere: `split_compute_y` (Zwischenzustand
y, für die Partner abgelegt), dann `split_apply` mit dem y des Partnerblocks (`interface_split_y` →
`split_y_state`). CPU: `ssprk43_stage_a/b`; GPU: `stage_end(defer)` + `stage_end_finish`, `lockstep_end_a/b`.
Ohne Blockgrenzen bitgleich zum bisherigen geteilten Schritt. 5 abgestufte Blöcke P21/P7: 203 → 134 Schritte, Integrale
wie beim Einblock-Fall (Rate 2e-4 gegen ungeteilt, Bilanz 4e-8); CPU gegen GPU 2e-10 (c21_7) bzw. 2,3e-7 (c7_3, mehr
Begrenzung). Job 162683.

**Spiegel im Blocktreiber (9f8a949, `--mirror-x on`):** Rechengebiet x ≥ 3,5, Blöcke links entfallen, über 3,5
abgeschnitten (Spiegelrand), Karte muss symmetrisch sein, Integrale × 2, Ausgabe gespiegelt. Ein Block bitgleich zu
`m1pn_fast` mit Spiegel; Karte mit/ohne Spiegel 1,3e-5 (Δt-Schätzung auf dem Halbgebiet).

**Karten gegen einheitliches P35 (Jobs 162687, 162692; GB, Trennebenen geteilt, Spiegel, gegen MC).** Pilotkosten
gehören zur Rechenzeit: `m1pn_lattice_mc` bis l = 35 braucht auf NX (64 Kerne) 2 s (1e7), 12 s (1e8), 1e9 220 s auf
32 Kernen (≈ 110 s auf 64). Auf den 16 Kernen eines tardis-Jobs etwa viermal so lange.

| Karte (Pilot, Pixel, tol) | Koeff. | Lauf (wall) | Masse/Absorption/Rate % | Feld L1 |
|---|---|---|---|---|
| einheitlich P35 (`m1pn_fast`) | 114 M | 121 s | −0,026 / +0,042 / +0,049 | 7,7e-4 |
| `all35` (ein Block) | 114 M | 120 s | −0,026 / +0,042 / +0,049 | 7,6e-4 |
| Ringe 35/27/21 (7 Blöcke) | 83 M | 138 s | −0,027 / +0,045 / +0,057 | 7,8e-4 |
| 1e9, 28, 3e-4 | 44 M | 100 s | −0,008 / +0,008 / −0,022 | 8,1e-4 |
| 1e9, 28, 1e-4 | 73 M | 166 s | −0,025 / +0,040 / +0,048 | 7,8e-4 |
| 1e9, 28, 1e-3 | 14 M | 40 s | −0,069 / +0,121 / +0,191 | 1,6e-3 |
| 1e8, 7, 3e-4 | 19 M | 71 s | −0,051 / +0,087 / +0,057 | 2,4e-3 |
| 1e8, 14, 3e-4 | 36 M | 103 s | −0,029 / +0,047 / +0,065 | 1,2e-3 |
| 1e8, 28, 3e-4 | 66 M | 179 s | −0,023 / +0,036 / +0,035 | 8,5e-4 |
| 1e8, 14, 1e-4 | 71 M | 167 s | −0,028 / +0,046 / +0,061 | 8,3e-4 |
| 1e7, 7, 3e-4 | 31 M | 93 s | +0,041 / −0,083 / −0,139 | 1,9e-3 |
| 1e7, 28, 1e-4 | 94 M | 203 s | −0,026 / +0,043 / +0,053 | 7,7e-4 |

Befund: (i) Die Blockkarte kostet selbst: Ringe 35/27/21 mit 73 % der Koeffizienten sind langsamer als einheitlich
(Grenzflächen-Teilschritt, Barrieren, kleinere Kernel). (ii) Die 1e9-Karte spart ~20 s im Lauf, der Pilot kostet ein
Mehrfaches. (iii) Billige Piloten sind verrauscht (Tail-Rauschen / Schranke 6–200): die Bias-Korrektur E − Var hilft
im Mittel, aber Schwankungen setzen viele Blöcke unnötig auf L (1e7/1e8 bei 28 Pixeln: 66–94 M Koeffizienten, langsamer
als einheitlich) oder grobe Pixel mitteln die Winkelstruktur weg (7 Pixel: ungenau). **Stand: keine Karte schlägt
einheitliches P35 bei gleicher Genauigkeit, wenn der Pilot mitgezählt wird.**

Rauschfeste Karte (`order_map_from_mc.py --significance 3 --mirror-x`): nur Koeffizienten mit |u_lm| > 3·stderr zählen.
1e8 bei 28 Pixeln weicht dann im Mittel 1,2 Ordnungen von der 1e9-Karte ab (vorher 3,1), Zustandsanteil 0,24 statt
0,34. Ergebnis (Job 162698, Binary 602d65f):

| Karte (3σ) | Koeff. | Lauf | + Pilot (64 NX-Kerne) | Masse/Absorption/Rate % | Feld L1 |
|---|---|---|---|---|---|
| 1e8, 28, 3e-4 | 50 M | 117 s | 129 s | −0,027 / +0,044 / +0,052 | 9,0e-4 |
| 1e8, 28, 2e-4 | 60 M | 130 s | 142 s | −0,022 / +0,035 / +0,037 | 8,1e-4 |
| 1e8, 14, 3e-4 | 33 M | 95 s | 107 s | +0,023 / −0,049 / −0,067 | 1,2e-3 |
| 1e9, 28, 3e-4 | 38 M | 93 s | ≈ 200 s | −0,013 / +0,017 / +0,015 | 7,9e-4 |
| 1e9, 28, 2e-4 | 48 M | 119 s | ≈ 230 s | −0,013 / +0,018 / +0,006 | 7,7e-4 |

Die Schwelle macht die 1e8-Karten deutlich billiger (117 statt 179 s bei gleicher Genauigkeit der Integrale), mit
Pilot aber nicht schneller als einheitliches P35 (121 s). Höhere einheitliche Ordnung (Job 162689): P45 198 s,
−0,021 / +0,033 / +0,032 %, L1 7,0e-4; P63 489 s, −0,018 / +0,028 / +0,022 %, L1 6,8e-4: ab P35 dominiert im Feld der
Ortsfehler. **Fazit Karten:** auf lattice-paul kein Nettogewinn durch MC-Pilotkarten; einheitliches GB P35 bleibt der
Standard. Die 1e9-Karten zeigen nur, dass die Integrale mit ~40 % der Koeffizienten genauer werden können als mit P45.

## 5. CUDA-Kernel

nsys (Job 162662, GB P35 geteilt, Spiegel, t = 0,8; Kernel 28,1 s ≈ Rechenzeit 25,7 s, also GPU-gebunden): Zeilenlöser
40 % (3,2 ms, ~1,1 TB/s von ~1,9), Streifenkern 20 % (4,8 ms), Kombination 14 %, Reaktion 13 % (6,8 ms bei ~1,2 ms
Bandbreitenminimum).
- **Reaktion (eb68ff2):** ein Thread je (Knoten, Moment) statt je Knoten (vorher kein Coalescing, 666 `exp` seriell).
  Bitgleich; GB P35 t = 0,8: 25,4 → 23,4 s (−8 %), Job 162681.
- **Fusion PN-Kombination in die letzte Massenlösung (690ec1b):** bitgleich, aber langsamer (29,5 statt 23,0 s;
  verstreute Lesezugriffe auf Flussmatrix und Gradienten); Standard aus, `M1PN_GPU_FUSE=1`.
- **Konstantenspeicher → global** (siehe 3): Zeit unverändert innerhalb der Messgenauigkeit.
- Offen: Vorauslade-Tiefe/Blockgröße des Zeilenlösers (`M1PN_GPU_LINE_CHUNK/BLOCK`, Job 162672), Nsight Compute
  (Job 162674).

## 4. Lokale Zeitschritte: Kostenmodell statt Umsetzung

Ideale Ersparnis, wenn jedes Element mit dem zu seiner kleinsten Weite passenden Schritt läuft (Zweierpotenzen):

| Gitter | Beschleunigung |
|---|---|
| GB (Tensor) 2D / 3D | 1,06× / 1,02× |
| Tensor 16,32,64,128,64,32,16 2D / 3D | 1,29× / 1,15× |
| isotrope Ringblöcke 128,128,64,32 je Ring, 2D / 3D | 1,27× / 1,26× |
| isotrope Ringblöcke 128,64,32,16, 2D / 3D | 1,90× / 2,03× |

Auf Tensorgittern nutzlos: die gestreckten Übergangszellen haben in einer Richtung die feine Weite. Ein Gewinn von
1,3–2× setzt isotrop abgestufte Blöcke voraus, deren Grenzen nicht konform sind (Punktkopplung, auf der GPU heute
langsam). Reihenfolge, falls gewollt: schnelle nichtkonforme GPU-Kopplung → isotrope Blöcke → lokale Zeitschritte
mit Flussregistern (Mathematikänderung). Nicht begonnen.
