# Elementbasiertes Limiting mit CG-WENO-Ziel

Eigener Pfad, unabhängig von `Transport`: `src/element_weno.{hpp,cpp}`, Binary
`m1pn_element_weno`, Test `tests/element_weno.cpp`. Uniforme kartesische Gitter
auf `[0,1]^d`, Tensor-Bernstein `Q_p`.

## Verfahren

Hochordnungsziel für alle PN-Koeffizienten, mit konsistenter Masse:

    M ȧ^H = -C·A a + b^H - s_h(a)

`s_h` nach Kuzmin, Hajduk, Vedral (arXiv:2509.04673v2), Gl. (6)/(7):

    s_h^e(v,u) = ν_e ∫_e ∇v·(∇u - γ_e g_h) dx,  ν_e = h/(2p)
    γ_e = 1 - min(1, ‖u - u*‖_e / ‖u‖_e)^q

`g_h` ist die konsistente L²-Projektion von `∇u_h`, `u*` die Hermite-WENO-
Rekonstruktion aus den von-Neumann-Nachbarn (Kuzmin, Vedral, arXiv:2212.14224).

Antidiffusive Elementbeiträge:

    f_ij^e = m_ij^e (q̇_i - q̇_j) + d_ij^e (u_i - u_j) + c_ji^e·G_i - c_ij^e·G_j
    f_i^e  = Σ_j f_ij^e - P s_h^e(φ_i, a)

mit `q̇ = P ȧ^H`, `G = F^H - F^L` und dem Limiter aus Bericht §6.
Die Ränder ergänzen `h_i = P b_i^H - b_i^L - (∮φ_i n)·G_i`.

## Abweichungen von der ursprünglichen Formel

- Abschlussterm `c_ji·G_i - c_ij·G_j` statt `c_ij·(G_i - G_j)`. Beide ergeben
  die gleiche Knotensumme, aber nur die erste summiert auf jedem Element zu null.
- `-s_h^e`, weil `s_h` im Paper auf der linken Seite steht.

## Getroffene Festlegungen

| Punkt | Wahl | Quelle |
|---|---|---|
| `γ_e` | ein Wert pro Element aus `ψ^(0)`, für alle PN-Komponenten | eigene Wahl, Literatur legt es nicht fest |
| `q`, `r`, `ε` | 1, 2, 1e-6 | Paper; `--weno-q` |
| lineare Gewichte | Nachbarn 1e-3 | Kuzmin, Vedral §7; `--weno-linear` |
| Positivitätsanteil `ε`, Faktor `θ` | 1e-3, 0.999 | im Bericht offen (`[CHECK]`) |
| Anfangsdaten Crossing | Koeffizienten = Punktwerte | Paper, nur 2. Ordnung |
| Randkorrektur | knotenweise, eigener Faktor | eigene Wahl |

## Ergebnisse (P3, 9.11.2026, ein Kern)

Periodische Sinuswelle, 1D, ohne aktiven Limiter (EOC der letzten Verfeinerung):

| p | WENO, q=1 | keine Stabilisierung |
|---|---|---|
| 1 | 2.05 | 2.00 |
| 2 | 3.03 | 2.00 |
| 3 | 3.23 (q=3 oder γ=1: 3.79, bei n=128) | 4.00 |
| 4 | 4.99 | 4.01 |

Bei `q=1` skaliert `1-γ` wie `h²`; das kostet bei p=3 Ordnung. Mit `q=3` ist das
Ergebnis identisch zu `γ=1`.

Crossing-Studie, gleiche Knotenzahl, `T=0.32`, CFL 0.5. Rohdaten in
`results/element_weno/crossing_20261109.txt`.

| Grad, Gitter | Gewicht | ρ_max | L² gegen Kinetik | Laufzeit |
|---|---|---|---|---|
| p=1, 96² | 1e-3 | 1.511 | 8.65e-2 | 77 s |
| p=1, 96² | 0.2 | 1.376 | 8.41e-2 | 77 s |
| p=2, 48² | 1e-3 | 1.320 | 7.98e-2 | 39 s |
| p=2, 48² | 0.2 | 1.269 | 7.72e-2 | 40 s |
| p=3, 32² | 1e-3 | 1.248 | 7.47e-2 | 29 s |
| p=3, 32² | 0.2 | 1.115 | 6.93e-2 | 29 s |

Der L²-Wert enthält den P3-Abbruchfehler gegen die kinetische Lösung und ist
deshalb nur als Vergleich innerhalb der Tabelle aussagekräftig.

## Grenzen

- Nur uniforme kartesische Gitter auf `[0,1]^d`. Konstante isotrope Streuung
  (`--sigma-s`) und Absorption (`--sigma-a`) werden seit 10.11.2026 exakt
  integriert: `a_lm *= exp(-τ(σ_a + σ_s[l≥1]))`, je ein halber Schritt vor und
  nach SSPRK3 zusammen mit dem Filter (Strang). Erhält Dichte (Streuung) und
  M1-Kegel. Quellen, Materialsprünge und `lattice` fehlen weiterhin.
- Line Source (Garrett–Hauck) läuft für Element-WENO exakt skaliert auf dem
  Einheitsquadrat (`σ_s=3`, `T=1/3`, Filter ×3); Auswertung in
  `scripts/line_source/analyze.py`.
- Vergleich mit Variante B und Filter siehe
  `results/element_weno/filter_comparison_20261109/REPORT.md`. Dort ist p>1 auf
  16² und 32² schlechter als p=1; die Tabelle oben gilt nur für 96 Intervalle.
  Feinere Gitter: `results/element_weno/large_comparison_20261109/REPORT.md`.
- Mit Gewicht 1e-3 ist der Sensor im Crossing fast wirkungslos; `ρ_max` und L²
  liegen praktisch bei `γ=1`.
- Der Limiter ist in 41–43 % der Element-Stufen aktiv, mit Faktor bis 0.
- Puffer pro Element brauchen `Elemente × Knoten × q` Doubles pro Stufe. Für
  hohe PN-Ordnung auf feinen Gittern ist das zu teuer.

## Laufzeit

Optimierungen vom 10.11.2026: paralleles Sammeln über eine Knoten-Inzidenzliste,
zeilenweise LDLᵀ-Linienlösungen für alle PN-Komponenten, parallele Randfaces mit
einmal ausgewerteter konstanter Einstrahldichte, vorab summierte Viskositäten.
128²-Knoten, P7, p=2: 0.411 → 0.227 s/Schritt (1 Thread), 0.213 → 0.095 s/Schritt
(8 Threads); Ergebnis gleich bis auf 1e-13. Zum Vergleich braucht die
Massenkorrektur mit Produktionseinstellungen auf 128²/P7 etwa 0.025 s/Schritt (8 Threads).

## Bauen und ausführen

    cmake --build tmp/build --target m1pn_element_weno m1pn_element_weno_tests
    tmp/build/m1pn_element_weno_tests
    tmp/build/m1pn_element_weno --problem crossing-study --dimension 2 \
      --pn-order 3 --degree 3 --elements 32 --time .32 --cfl .5 --weno-linear .2
