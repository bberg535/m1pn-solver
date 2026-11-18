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

## d) Verbreiterte Zellen an den Trennebenen (`--break-widen w`)

Gitterwahl statt Mathematik: Auf abgestuften Gittern werden die w Zellen auf jeder Seite einer Trennebene zu einer
Zelle verschmolzen (Breite w·h normal zur Linie, tangential unverändert). Die einseitige Knotenmasse wächst, die
Kopplung nur wenig.

Schranken (CPU, 8,16,32,32,32,16,8, P3): w = 2: Realisierbarkeit 0,0154, Hochordnung 0,0123, Δt 0,0123 (ohne
Trennebenen 0,0122) — die Strafe ist vollständig weg.

Genauigkeit gegen MC (GB, A100, Spiegel, Job 162477):

| Gitter | Schritte | P35 Zeit | P35 Masse/Absorption/Rate % | P35 Feld L1 | P21 Zeit | P21 Absorption % | P21 Feld L1 |
|---|---|---|---|---|---|---|---|
| ohne Trennebenen | 1052 | 122 s | −0,046 / +0,079 / +0,076 | 8,6e-4 | 61 s | +0,127 | 1,2e-3 |
| Trennebenen, w = 2 | 1051 | 125 s | −0,049 / +0,068 / +0,067 | 1,2e-3 | 61 s | +0,114 | 1,6e-3 |
| Trennebenen, w = 3 | 1051 | 121 s | −0,080 / +0,097 / +0,087 | 2,0e-3 | 59 s | +0,137 | 2,2e-3 |
| Trennebenen, w = 1 | 1623 | 185 s | −0,026 / +0,042 / +0,050 | 7,7e-4 | 89 s | +0,096 | 1,1e-3 |

Ergebnis: Die gröberen Zellen liegen genau dort, wo die Lösung am steilsten ist; w = 2 behält nur einen kleinen Teil
des Absorptionsgewinns und verschlechtert das Feld (L1 schlechter als ganz ohne Trennebenen). Kein Gewinn.
Nebenbefund: CPU gegen GPU auf diesem Gitter (w = 2, P7, gleiches Δt) 1,0e-7 max / 1,8e-8 L2, über der Schwelle
1e-8; nicht untersucht (einzelne Limiter-Entscheidung oder GPU-Pfad mit verschmolzenen Zellen neben Trennebenen).

## a) Geteilter Schritt an den Trennebenen (`--split-breaks on`, CPU-Prototyp)

Statt echter lokaler Teilschritte (Zwischenzustände der Nachbarn, Flussregister, SSPRK-Stufen; mehrere Tage) eine
Umformung jedes Vorwärts-Euler-Teilschritts τ = Δt/2 der SSPRK(4,3)-Stufen an den Trennebenenknoten:

1. **Element- und Randschritt:** y = u + τ/m (E + B_Rand). E = Σ_e [2 tv (ū^e − u) + θ_e f^e] hat schon die konvexe
   Form (ū^e im Kegel, der Randfluss steckt in ū^e), also realisierbar für τ (Σ 2 tv + A_Rand) ≤ m.
2. **Trennebenenschritt:** z = y + τA/m (B̄(y) + θ g − y) mit B̄(y) dem knotenweisen LLF-Barzustand aus y und dem
   y des Partners (`broken_face_low`), g = c/A, c = T − A (B̄(y) − y), T = P b^H − Σ_d (∫φ n_d) G_d der unbegrenzte
   Flächenterm des ungeteilten Schritts. θ = 1 ergibt **genau** u + τ (E + T)/m, also den ungeteilten Schritt;
   θ < 1 nur dort, wo z sonst den Kegel verließe (wie der Randlimiter). Realisierbar für τ A ≤ m.

Schranke: Δt ≤ 2 min_i min(m_i / (Σ 2 tv_i + A_Rand,i), m_i / A_TE,i) (`node_dt_bound`); die Rate der
Trennebenenknoten wird (z − u)/τ, `combine` und die SSP-Gewichte bleiben. Masse bleibt erhalten bis auf den
Begrenzungsanteil τ(θ − 1)c₀ (Statistik `split_mass`). Code: `ElementWenoOptions::split_breaks`,
`apply_split_breaks`, getrennte Summen der Trennebenenzeilen in `add_boundary` (`BreakAggregates`).

Prüfung (CPU, lattice-paul FCS, 8,16,32,32,32,16,8, P7, Trennebenen, Spiegel, t = 1,6):

| Lauf | Δt | Schritte | gegen ungeteilt: Integrale | Feld | Massenbilanz |
|---|---|---|---|---|---|
| ohne Option | 0,00789 | 203 | bitgleich zum Stand vor dem Umbau | | 4e-14 |
| geteilt, altes Δt | 0,00789 | 203 | 4e-9 / 4e-8 / 6e-10 | 3e-8 | 3e-8 |
| geteilt, neues Δt | 0,01203 | 133 | 7e-6 / 5e-5 / 2e-4 | 5e-4 | 5e-8 |

θ < 1 tritt auch beim alten Δt oft auf (65 554 Knotenstufen, min θ 0,0025), aber nur in fast leeren Bereichen
(Wirkung 1e-8). Beim neuen Δt Abweichung 2e-4 in der Rate — Zeitfehler oder Teilungsfehler; entscheidet der
Vergleich gegen MC.

**GPU:** Der Kernel `k_boundary_rows_add` liefert zusätzlich die l ≤ 1-Summen nur der Trennebenenzeilen (Zeilenflag
`row_break`); der Host summiert in Phase B die Trennebenenzeilen getrennt, rechnet in `boundary_correction` nur
den Randanteil und danach den zweiten Schritt auf `load_b_` mit derselben Funktion wie die CPU
(`ElementWeno::split_second_step`). (Erster Versuch falsch: die l ≤ 1-Koeffizientensummen nicht nach (ρ, J)
projiziert; −0,5 % Absorption. Behoben.)

Ergebnisse gegen MC (lattice-paul, Spiegel in x, t = 3,2):

| Lauf | Gerät | Zeit | Schritte | Masse/Absorption/Rate % | Feld L1 | Median |
|---|---|---|---|---|---|---|
| GB P21, Trennebenen, ungeteilt | CPU NX 32 Thr. | 674 s | 1623 | −0,055 / +0,096 / +0,133 | 1,1e-3 | 2,6e-3 |
| GB P21, Trennebenen, geteilt | CPU NX 32 Thr. | 582 s | 1059 | −0,055 / +0,096 / +0,132 | 1,1e-3 | 2,6e-3 |
| GB P35, ohne Trennebenen | A100 | 122 s | 1052 | −0,046 / +0,079 / +0,076 | 8,6e-4 | 2,8e-3 |
| GB P35, Trennebenen, ungeteilt | A100 | 185 s | 1623 | −0,026 / +0,042 / +0,050 | 7,7e-4 | 2,2e-3 |
| **GB P35, Trennebenen, geteilt** | A100 | **131 s** | 1059 | **−0,026 / +0,042 / +0,049** | **7,7e-4** | 2,2e-3 |
| GB P21, Trennebenen, ungeteilt | A100 | 89 s | 1623 | −0,055 / +0,096 / +0,133 | 1,1e-3 | 2,6e-3 |
| **GB P21, Trennebenen, geteilt** | A100 | **64 s** | 1059 | −0,055 / +0,096 / +0,132 | 1,1e-3 | 2,6e-3 |

(Jobs 162520, 162541.) Die Trennebenen behalten ihren vollen Genauigkeitsgewinn, die Schrittzahl ist die des
Gitters ohne Trennebenen; Massenbilanz 7e-8 (`split_mass` 1,4e-7). Host-Zusatzzeit des zweiten Schritts ~10 s
(`host_correction`), überlappt nicht mit der GPU.

Offen/Einschränkungen:
- Der zweite Schritt begrenzt oft (θ < 1 in ~6 % der Knotenstufen, min θ ~1e-3): sein Ziel y + T/A verlässt an
  steilen Stellen den Kegel. Die Wirkung auf Integrale, Bilanz und MC-Fehler ist vernachlässigbar, aber die diskreten
  θ-Entscheidungen machen den Lauf empfindlich gegen Rundung: CPU gegen GPU bei gleichem Δt 2,7e-7 max / 2,9e-8 L2
  in der Dichte (Integrale 7e-9), ungeteilt 1e-12. Eine glattere Variante (nur den Hochordnungsanteil begrenzen)
  verliert ohne Zusatzmaßnahme die Erhaltung.
- Zeitgenauigkeit an den Trennebenen: formal ist jede SSP-Stufe exakt der ungeteilte Euler-Schritt, solange θ = 1;
  bei θ < 1 ist der Fehler lokal erster Ordnung. Nicht gesondert gemessen (Abweichung beim größeren Δt von gleicher
  Größe wie der reine Zeitfehler ohne Trennebenen).
- 3D: CPU-Code dimensionsunabhängig; abgestufte Gitter und Trennebenen auf der GPU sind derzeit nur 2D.
- CPU-Prototyp ~45 % teurer je Schritt (volle Felder je Stufe, serielle Summen) — für die GPU unerheblich.

## Echte lokale Teilschritte (nicht begonnen)

Nur die Knoten der Trennebenen (bzw. die Elemente daneben) mit 2 Teilschritten, Kopplung über Flussregister,
Theorem 3.1 je Teilschritt. Großer Umbau (SSPRK(4,3)-Stufen, GPU-Pipeline), Ausgang offen. Nach c) der einzige
Weg, die Strafe (Faktor 1,5 bei P35: 185 s statt ~125 s) zu vermeiden. Skizze einer einfacheren Variante, die vor
einem Umbau zu prüfen wäre: nur der Tiefordnungsschritt der Trennebenenknoten mit zwei Teilschritten bei
eingefrorenen Nachbarzuständen (jeder Teilschritt konvex, also realisierbar); die Masse bliebe aber nur mit einem
Flussregister (Ausgleich auf der Partnerseite) erhalten, und die Kopplung an den Limiter (Hochordnungsanteil gegen
Tiefordnungszustand) ist zu klären. Entscheidung des Nutzers.
