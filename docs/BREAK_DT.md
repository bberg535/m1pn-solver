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

Genauigkeit gegen MC (GB P35/P21, A100, Job 162424): ERGEBNISSE_FOLGEN

## a) Lokale Teilschritte (nicht begonnen)

Nur die Knoten der Trennebenen (bzw. die Elemente daneben) mit 2 Teilschritten, Kopplung über Flussregister,
Theorem 3.1 je Teilschritt. Großer Umbau (SSPRK(4,3)-Stufen, GPU-Pipeline), Ausgang offen; erst angehen, wenn c)
nicht genügt.
