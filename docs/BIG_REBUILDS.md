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

ERGEBNISSE_FOLGEN
