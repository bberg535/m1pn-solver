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

## 3D: `lattice-3d` (16.11.2026, Branch `feature/combine-202611`)

`m1pn_fast --case lattice-3d --fcs analytic` (CPU und `--gpu`). Quelle q0 = 1 auf der Zelle S = [3,4]³, σ_t
zellweise (Absorber 10, Schicht k = 5 Streuung 5, sonst 1; `src/benchmark_cases.hpp`), Vakuum außerhalb von [0,7]³.

**Formulierung** (`src/uncollided_lattice3d_core.hpp`, Host und Device):

    ρ_u(x, t) = q0/(4π) ∫_{S²} e^{−τ_in(Ω)} (1 − e^{−σ_S L}) / σ_S dΩ,   L = min(s_out, t) − s_in > 0.

Für x außerhalb von S wird über die von x aus sichtbaren Flächen F von S integriert (höchstens drei); der Strahl zum
Flächenpunkt y tritt dort ein (s_in = |y − x|), dΩ = |n·(y − x)| / |y − x|³ dA. Liegt x in S, sind die sechs Flächen
Austrittsflächen (s_in = τ_in = 0). τ_in durch Voxel-Traversierung (Amanatides–Woo), s_out durch den Slab-Test.

**Quadratur.** Je Fläche iteriert Gauß (6 Punkte je Stück), außen u, innen v:
- innen geteilt, wo der Strahl x → y(u, v) eine Gitterkantenlinie (achsenparallele Linie durch ganzzahlige
  Koordinaten) vor seinem Austritt aus S trifft: dort ändert sich die Zellfolge vor S oder die Austrittsfläche;
- außen geteilt an den Projektionen der Gitterecken und dort, wo projizierte Kantenlinien die Flächenränder v = v0, v1
  kreuzen (sonst Sprung der zweiten Ableitung des inneren Integrals in u; ohne diese Teilung nur algebraische
  Konvergenz, 1e-5);
- jedes Stück zusätzlich in 8 × 8 gleiche Teile (Front s_in = t bzw. s_out = t: Knick, nicht ausgerichtet).

Ohne Front in S ist die Regel auf 1e-10 bis 1e-15 genau (exponentielle Konvergenz in der Punktzahl); mit Front ≈ 1e-5
relativ, direkt nach Ankunft der Front (ρ_u winzig) bis 1e-4. Etwa 5 ms je Punkt für eine ganze Zeittabelle.

**Zeittabelle.** Je Strahl ist die Zeitabhängigkeit exakt (0 vor s_in, A(1 − e^{−σ_S(t − s_in)}) bis s_out, danach
konstant); sie wird über drei Differenzfelder auf dem Gitter t_m = 0.01 m aufsummiert (`lattice3d_row`), also eine
Auswertung je Punkt für alle Zeiten. Punkte, die die Front bis T nicht erreicht, bleiben null. Interpolation und
Zeitintegral wie in 2D (`Lattice3dUncollidedTable`, `src/uncollided_lattice3d.hpp`).

**Tests** (`tests/uncollided_lattice3d.cpp`, ctest; `--full` für 6 und 7):

| Test | Ergebnis |
|---|---|
| 1. Quellmitte, t ≤ 0.5: ρ_u = 1 − e^{−t} | 9.5e-15 |
| 2. homogen σ = 1, außerhalb S, gegen Volumenintegral über S (glatt) | 8.9e-15 |
| 3. null vor Ankunft der Front | ok |
| 4. Tabelle gegen direkte Auswertung an t_m / zwischen t_m (kubisch) | 1.5e-14 / 1.2e-5 |
| 5. Standardregel gegen doppelte Teilung und 8 Punkte, 200 zufällige (x, t) | 2.6e-8 (relativ zum Maximum) |
| 6. mit Absorbern gegen Volumenintegral, τ exakt über Ebenenschnitte (unabhängig von der Traversierung) | 5e-3 → 1e-3 → 1.9e-4 → 3.5e-5 bei sub 8 → 64 (zweite Ordnung der Referenz); ohne Absorber auf dem Weg 1e-15 |

**GPU.** `--gpu` baut die Tabelle auf dem Gerät (`k_lattice3d_table`, ein Thread je Punkt, `lattice3d_row`); die
Reaktion nutzt denselben Interpolationskern wie in 2D. CPU gegen GPU (`tests/gpu_validate.sh`, 28³, T = 0.5): Dichte
1.1e-16, FCS-Integrale 1.1e-15.

**Läufe** Q1, P3, T = 3.2, CFL 1 (NX-Sockel bzw. tardis), Summen gestoßen + ungestoßen:

| Gitter | ohne FCS: Masse / Absorption / Rate | mit FCS | Tabelle CPU (NX, 32 Kerne) / GPU (A100) | Gesamt GPU |
|---:|---|---|---:|---:|
| 14³ | 0.2375 / 2.9620 / 0.9986 | 0.5150 / 2.1410 / 0.8125 | 2.4 s / – | – |
| 28³ | 0.4057 / 2.7943 / 0.9981 | 0.5902 / 2.4354 / 0.9391 | 21.7 s / 7.4 s | 9.7 s |
| 56³ | 0.5524 / 2.6476 / 0.9974 | 0.6489 / 2.5043 / 0.9812 | 169 s / 49 s | 55 s |

Ungestoßene Masse bei T: 0.4230 / 0.4299 / 0.4325. Bilanz des gestoßenen Anteils ≤ 5e-12. Beide Varianten sind auf
diesen Gittern noch nicht konvergiert (eine Quellzelle hat bei 14³ nur 2³ Elemente); die FCS-Summen ändern sich je
Verdopplung um etwa die Hälfte weniger als ohne FCS. `lattice-3d` ist ein eigener Fall ohne Literaturreferenz.
Seit 16.11.2026 gibt es eine eigene Monte-Carlo-Referenz für beide Lattice-Fälle; Abweichungen dagegen in
[LATTICE_MC.md](LATTICE_MC.md) (3D mit FCS, P3: 28³ −15.9 %, 56³ −7.6 % in der Masse).

## 3D: Polar-Regel der Zeittabelle (20.11.2026, Branch `feature/fcs3d-table`, Option)

**Befund zur kartesischen Regel.** Test 5 (Standard gegen doppelte Teilung, 2.6e-8) zieht Zufallspunkte, die kaum nahe
an S liegen. Für Punkte im Abstand h vor einer Fläche von S ist der Kern h/r³ fast singulär: Fehler (max über t, gegen
eine sinh-transformierte Referenz 10×64) 6e-5 bei h = 0.3, 1.1e-3 bei h = 0.026 (Gauß-Punkte von 56³), 3.2e-3 bei
0.013 (112³), 9.5e-3 bei 0.0066 (G3B), 4e-2 bei 0.003. Auf die Summen wirkt das kaum (56³ P3: Masse, Absorption und
Rate ändern sich gegen genauere Regeln um ≤ 0.03 %), weil nur wenige Punkte betroffen sind. Kosten: im Mittel 50 000
Strahlen je Punkt (8×8-Unterteilung je Stück wegen der Frontknicke, 6×6 Gauß).

**Polar-Regel** (`lattice3d_row_polar`, `src/uncollided_lattice3d_core.hpp`, Host und Device). Polarkoordinaten (ρ, φ)
um den Fußpunkt von x auf der Fläche, s = √(h² + ρ²): dΩ = (h/s²) ds dφ, beschränkt (Hayami, PART). Außerhalb von S ist
s = s_in, die Front s_in = t liegt bei festem s für alle t; mit Chebyshev-Stammfunktionen je radialem Panel (8 Knoten)
ist ρ_u(t_m) = ∫dφ [G₀(t_m) − e^{−σ_S t_m} G₁(t_m)] exakt in s an jeder Tabellenzeit (Sättigung nach s_out knotenweise wie
bisher). In S: s = s_out, analog. Bruchstellen radial: Gitterkantenkreuzungen längs der radialen Linie, geometrisch
(ρ = h·2^k) am Fußpunkt; in φ: Flächenecken, projizierte Gitterecken, Schnitte projizierter Kanten mit dem Flächenrand.
Adaptiv: Panels und φ-Stücke werden halbiert, bis die letzten zwei Chebyshev-Koeffizienten (von A in s, vom radialen
Integral Q in φ) unter `polar_tol`·max + `polar_floor` liegen (1e-4, 1e-9); ein φ-Stück wird in einer temporären Zeile
akkumuliert und nur bei Annahme übernommen (fester Stack, Tiefensuche). Arbeitsspeicher 10 (M + 2) je Punkt.

Schalter (Experiment): `M1PN_FCS3D_RULE=polar`, `M1PN_FCS3D_TOL`. Standard bleibt kartesisch.

| 300 Zufallspunkte außerhalb S, Referenz sinh 10×64 | Strahlen/Punkt | ms/Punkt CPU | max Fehler / max | Zeitintegral global | je Punkt Median / p90 |
|---|---:|---:|---:|---:|---|
| kartesisch 6×8 | 49 700 | 8.5 | 1.3e-2 | 2.5e-3 | 1.9e-7 / 3.6e-6 |
| polar adaptiv (1e-4, 1e-9) | 8 100 | 3.0 (Prototyp; Kern 3.7× schneller als kartesisch) | 8.9e-5 | 1.8e-7 | 1.0e-6 / 5.2e-5 |

Lauf 56³ P3 FCS T = 3.2 (A2000): Tabelle 616 → 351 s (1.75×), Masse/Absorption/Rate gegen MC −7.501/+0.234/−1.234 % →
−7.497/+0.241/−1.226 %. CPU gegen GPU (28³) 2e-11, FCS-Integrale 4e-11. Test 8 in `tests/uncollided_lattice3d.cpp`.

**Offen.** (1) Auf der GPU nur 1.75× bei 6× weniger Strahlen: die adaptive Tiefe divergiert zwischen benachbarten
Threads (ein Thread je Punkt); Abhilfe: Arbeit je φ-Stück verteilen. (2) Läuft der Frontkreis über den Flächenrand, hat
das radiale Teilintegral in φ einen Knick bei einem zeitabhängigen Winkel, den die Adaptivität nicht teilt: bis 8e-4 des
Zeilenmaximums kurz nach Ankunft (Punkt (3.4, 3.6, 0.8)), Zeitintegral 2e-6; 2D teilt dort je Zeit. (3) Speicher
unverändert (U168 mit FCS und P15 auf 112³ passen nicht auf die A100).

**Verworfen** (56³ P3, gegen 8 Gauß-Punkte je Element): Zeilen nur an den Knoten und Q1-Interpolation auf die
Gauß-Punkte (Tabelle 8× billiger) ergibt Absorption +41 % statt +0.24 %; ein Gauß-Punkt je Element −11.5 %. In den
Absorbern ist ein Element 1.25 freie Weglängen dick (σ = 10, h = 1/8), ρ_u ändert sich im Element um den Faktor drei;
die 2×2×2-Regel ist dort nötig.
