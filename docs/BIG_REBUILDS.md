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

KARTEN_FOLGEN

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
