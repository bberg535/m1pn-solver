# First-Collision-Source für den Lattice (Stand 16.11.2026)

Branch `experimental/lattice-fcs` (von `experimental/gpu-campaign`). Erweitert [FCS.md](FCS.md) vom homogenen
Line-Source-Puls auf den heterogenen Lattice (Moujaes/Kuzmin/Baeumer (44)-(45), Schotthöfer/Hauck (12)):
σ_a = 10 auf den elf Absorberzellen, σ_s = 1 sonst, Quelle q0 = 1 auf [3,4]² ab t = 0, Vakuumrand.

## Formulierung

Wie in FCS.md: ψ = ψ_u + ψ_c. Die gesamte Quelle geht in ψ_u; das Gitter (Element-WENO M1PN) trägt ψ_c mit der
isotropen Quelle σ_s ρ_u, Anfangswert der Boden 1e-10. Ausgewertet werden die Summen (Masse, Absorptionsrate,
absorbierte Gesamtmenge).

**ψ_u exakt per Strahlverfolgung** (`src/uncollided_lattice.hpp`). Felder hängen von (x, y) ab, Richtungen sind 3D,
μ = sin θ. Mit dem ebenen Abstand l = sμ (sin θ dθ ds = dθ dl):

    ρ_u(x, t) = q0/(2π) ∫dφ ∫_0^{π/2} dθ ∫_0^{tμ} χ_S(l) e^{−T(l)/μ} dl,

T(l) die ebene optische Tiefe entlang x − l(cos φ, sin φ). S ist konvex, ein Strahl trifft S in [l_in, l_out]
mit σ_S = 1 dort, T_in = T(l_in) aus dem Zelldurchlauf (≤ 14 Zellen). Das l-Integral ist geschlossen:
e^{−T_in/μ} μ/σ_S (1 − e^{−σ_S L/μ}), L = min(l_out, tμ) − l_in.

**Quadratur.** φ über das Fenster, in dem S von x aus sichtbar ist, geteilt an den Winkeln aller Gitterecken
(konstante Zellfolge) und an den Schnittpunkten des Kreises vom Radius t mit dem Rand von S (l_in = t bzw. l_out = t);
auf jedem Stück Gauß-Legendre nach der Substitution φ = a + (c − a)(3s² − 2s³), die das (t − l_in)^{3/2}-Verhalten
an den Frontpunkten glättet. θ geteilt bei μ = l_in/t und l_out/t, dazu geometrische Panels lo·2^k (Randschicht von
e^{−l/μ}), je 10 Gauß-Punkte.

**Pruning (16.11.):** Nur Gitterecken vor dem Eintritt in S sind Bruchstellen (die übrigen ändern die Zellen auf
[0, l_in] nicht); dafür 24 statt 8 Punkte je φ-Stück. Über 2000 zufällige (x, t) gegen die Referenz ohne Pruning
(16/20 Punkte): alle Ecken mit 8 Punkten 1.0e-4 relativ in 0.083 s, Pruning mit 24 Punkten 1.7e-6 in 0.040 s
(`--prune-study`; die frühere Angabe < 1e-9 galt nur für die festen Testpunkte). Tabelle 56²: 15.5 s → 5.5 s,
Summen ändern sich um ≤ 1e-7 relativ.

**Zeittabelle.** ρ_u an den FCS-Quadraturpunkten (Gauß p+1 je Richtung und Element, exakt für ∫φ_i) für
t_m = 0.01 m, kubisch interpoliert (Fehler ≤ 5e-6 absolut); Zeitintegral für die ungestoßene Absorption mit
Simpson auf der Tabelle. Kosten ≈ 120 µs pro Punkt und Tabellenzeit auf einem Kern
(vor dem Pruning; 56²: 16 s auf 32 Kernen, 448² ≈ 16 min; danach etwa 2.8x schneller).

## Tests (`tests/uncollided_lattice.cpp`, `--full` für 5 und 6)

1. Quellmitte, t ≤ 0.5: ρ_u = 1 − e^{−t}, Fehler ≤ 1e-15.
2. Quadraturkonvergenz (Standard gegen 16/20 Punkte): < 1e-9 relativ.
3. Null vor Ankunft der Front.
4. Zeittabelle gegen direkte Auswertung: 4.9e-6 absolut; Zeitintegral in der Mitte 8e-9.
5. Unabhängige Formulierung als Flächenintegral über S mit dem Kern der unendlichen Linienquelle
   K = 1/(2πr) ∫_0^{acos(r/t)} e^{−τ/cos a} da, τ durch feines Abschreiten: ohne Absorber auf dem Weg 1e-11 bis 6e-7;
   mit Absorbern konvergiert die Referenz mit der Schrittzahl (2.4e-4 → 1.1e-4, 2.8e-4 → 3.2e-5 bei 4e3 → 4e4 Schritten).
6. Bilanz ohne Ausstrom (t < 3): dM_u/dt = 1 − ∫σ_t ρ_u. Residuum 3.5e-4 → 8.7e-5 → 2.2e-5 (t = 1) und
   6.0e-4 → 1.5e-4 → 3.8e-5 (t = 2.5) bei Halbierung der Raumquadratur: zweite Ordnung gegen null.

## Treiber

`m1pn_fast --case lattice-paul --stabilization element-weno --fcs analytic` (nur CPU; der GPU-Pfad kennt nur den
Line-Source-Puls). `summary.json` → `lattice_fcs`: ungestoßene Masse/Rate/Absorption, Summen `total_*`,
Tabellenzeit. `final.csv`: `u0` = Gesamtdichte, `rho_uncollided` direkt ausgewertet. `history.csv` und `ledger`
beziehen sich auf den gestoßenen Anteil (Bilanz dort ≤ 1e-12). `nx_convergence.py` vergleicht bei FCS die Summen.

Erster Lauf Q1 56², P7, T = 3.2 (Abweichung gegen Schotthöfer/Hauck Tab. 1 / 2π):

| | Masse | Absorption gesamt | Rate |
|---|---:|---:|---:|
| ohne FCS | −5.27 % | +9.81 % | +5.19 % |
| mit FCS | −3.59 % | +5.27 % | +3.48 % |

## Studie

`scripts/element_weno/lattice_fcs_study.py` (NX, CPU, eine Aufgabe = ein Sockel): Q1 56²-448², N = 3, 7, 15, 21
(35 auf 112², 224²), jeweils mit und ohne FCS; Kampagne `nx/tmp/lattice_fcs` (`REPORT.md`, `results.csv`; Tabelle noch
ohne Pruning). Abweichung gegen Schotthöfer/Hauck Tab. 1 / 2π (Masse / Absorption gesamt / Rate):

| N | 448² ohne FCS | 448² mit FCS | Laufzeit ohne / mit (s) |
|---:|---|---|---|
| 3 | −2.24 / +4.20 / +6.36 % | −1.23 / +2.30 / +3.60 % | 252 / 1142 |
| 7 | −0.61 / +1.16 / +1.05 % | −0.36 / +0.67 / +0.81 % | 305 / 1218 |
| 15 | −0.22 / +0.43 / +0.40 % | −0.13 / +0.26 / +0.29 % | 754 / 1563 |
| 21 | −0.17 / +0.34 / +0.28 % | −0.10 / +0.20 / +0.20 % | 1148 / 2006 |

- FCS verkleinert Masse- und Absorptionsabweichung auf allen Gittern und für alle N um etwa den Faktor 1.7
  (Rate 1.2-1.8), nicht um eine Größenordnung wie beim Line Source.
- Ab N = 15 ändert sich bei festem Gitter kaum noch etwas (224²: N = 15/21/35 gleich bis 0.02 %): der Rest ist
  Raumfehler, er fällt pro Gitterverdopplung etwa um den Faktor 3. FCS wirkt deshalb vor allem auf den Raumfehler
  (das steile ungestoßene Feld wird nicht auf dem Gitter transportiert).
- Kosten: die Tabelle dominierte (448²: 896 s); mit Pruning ≈ 320 s. Damit kostet FCS bei 448² etwa so viel wie der
  Schritt N = 15 → 21 und bringt mehr (P15 mit FCS −0.13/+0.26 % gegen P21 ohne −0.17/+0.34 %); gegenüber einer
  Gitterverfeinerung (Faktor 3, ~8x Kosten) ist es der billigere Hebel.
- Die Abweichungen nähern sich der Auflösung der Referenz (4 Stellen, S22, andere Anfangsdichte/Einstrom).
