# Experiment: Zeitschritt mit Trennebenen (Branch `experiment/break-dt`, 18.11.2026)

Ausgang (docs/STATIC_HP_BLOCKS.md, „Zeitschrittschranke“): Mit Trennebenen bestimmt die Realisierbarkeitsschranke
Δt ≤ m_i/λ_i den Schritt (GB: 0,00197 statt der Hochordnungsschranke 0,00302, 1623 statt 1052 Schritte). Ein Knoten
auf einer Trennebene hat nur die einseitige gelumpte Masse; die Schranke ist für das Verfahren scharf.

Der Nutzer hat am 18.11. abends ein Experiment auf eigenem Branch freigegeben. Der Standard bleibt unverändert;
alle Optionen hier sind Studienoptionen von `m1pn_fast`.

## b) Δt nach der Hochordnungsschranke (`--dt-bound high-order`)

Wählt Δt = CFL · high_order_step und lässt die Prüfung gegen die Realisierbarkeitsschranke weg.

Ergebnis (CPU, lattice-paul FCS, 224 Zellen abgestuft 16,32,32,32,32,32,16, P7, Trennebenen, Spiegel): Δt 1,52-mal
die Realisierbarkeitsschranke; **im ersten Schritt** verlässt ein Zustand den M1-Kegel (Knoten im Hintergrund,
ρ ≈ 5e-11, ρ − |J| = −1e-11), Abbruch. Die Schranke wird also sofort gebraucht; ohne Schrittweitenrücknahme ist b)
nicht brauchbar, mit Rücknahme gingen die gewonnenen Schritte wieder verloren. Nicht weiter verfolgt.

## c) Trennebenen nur auf ausgewählten Linien (`--break-at 1,6`)

Mit `--break-units on` und `--break-at L1,L2,...` werden nur die Einheitslinien x = L bzw. y = L gebrochen (beide
Achsen, abgestufte Gitter). Keine Änderung der Mathematik, nur die Wahl der Linien.

Idee: Die Strafe entsteht nur, wenn die Knoten einer Trennebene auf der feinen Seite liegen. Bei GB
(32,64,128,128,128,64,32) grenzen an x = 1 und x = 6 Zellen der Weite 1/32 und 1/64; dort liegt die
Realisierbarkeitsschranke über der Hochordnungsschranke. Prüfung (CPU, 8,16,32,32,32,16,8, P3):

| Trennebenen | Realisierbarkeit | Hochordnung | Δt |
|---|---|---|---|
| keine | 0,0159 | 0,0122 | 0,0122 |
| alle | 0,0079 | 0,0121 | 0,0079 |
| 1, 6 | 0,0154 | 0,0122 | 0,0122 |
| 1, 2, 5, 6 | 0,0079 | 0,0121 | 0,0079 |

Genauigkeit gegen MC (GB P35/P21, A100, 16 Kerne, Spiegel in x, Job 162424; „keine“/„alle“ aus Job 162329):

| Trennebenen | Schritte | P35 Zeit | P35 Masse/Absorption/Rate % | P35 Feld L1 | P21 Zeit | P21 Absorption % |
|---|---|---|---|---|---|---|
| keine | 1052 | 122 s | −0,046 / +0,079 / +0,076 | 8,6e-4 | 61 s | +0,127 |
| 1, 6 | 1052 | 125 s | −0,046 / +0,079 / +0,076 | 8,6e-4 | 62 s | +0,127 |
| 1, 2, 5, 6 | 1623 | 183 s | −0,042 / +0,073 / +0,071 | 8,4e-4 | 89 s | +0,120 |
| 3, 4 | 1623 | 179 s | −0,029 / +0,049 / +0,057 | 7,9e-4 | 87 s | +0,102 |
| alle | 1623 | 185 s | −0,026 / +0,042 / +0,050 | 7,7e-4 | 89 s | +0,096 |

Ergebnis: Der Gewinn der Trennebenen kommt fast ganz aus dem feinen Zentrum (Linien 3, 4); die schrittfreien
Linien 1, 6 bringen nichts (bitgleich zu „keine“ in den Kennzahlen). c) löst das Problem nicht: Die Genauigkeit
braucht die Trennebenen genau dort, wo die feine Seite die Schranke senkt.

## a) Lokale Teilschritte (nicht begonnen)

Nur die Knoten der Trennebenen (bzw. die Elemente daneben) mit 2 Teilschritten, Kopplung über Flussregister,
Theorem 3.1 je Teilschritt. Großer Umbau (SSPRK(4,3)-Stufen, GPU-Pipeline), Ausgang offen. Nach c) der einzige
Weg, die Strafe (Faktor 1,5 bei P35: 185 s statt ~125 s) zu vermeiden. Skizze einer einfacheren Variante, die vor
einem Umbau zu prüfen wäre: nur der Tiefordnungsschritt der Trennebenenknoten mit zwei Teilschritten bei
eingefrorenen Nachbarzuständen (jeder Teilschritt konvex, also realisierbar); die Masse bliebe aber nur mit einem
Flussregister (Ausgleich auf der Partnerseite) erhalten, und die Kopplung an den Limiter (Hochordnungsanteil gegen
Tiefordnungszustand) ist zu klären. Entscheidung des Nutzers.
