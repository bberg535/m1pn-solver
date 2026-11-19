# Kobayashi-Benchmarks (3D, Hohlraum und Kanal)

Externe 3D-Referenz für Element-WENO mit und ohne First-Collision-Source (Stand 17.11.2026, Branch `feature/paper-tests`).

## Definition

Kobayashi, Sugimura, Nagaya, *3D radiation transport benchmark problems and results for simple geometries with void
region*, Prog. Nucl. Energy 39 (2001) 119–144; OECD/NEA NSC/DOC(2000)4. Stationär, Einheiten cm, Quelle
q0 = 1 cm⁻³ s⁻¹ isotrop im Würfel [0,10]³, σ_t = 0.1 cm⁻¹ (Quelle, Absorber) bzw. 10⁻⁴ cm⁻¹ (Hohlraum). Fall i:
reiner Absorber; Fall ii: Streuverhältnis 1/2 (σ_s = σ_t/2 überall). Reflektierend auf x = 0, y = 0, z = 0, Vakuum
sonst.

| Problem | Gebiet (Oktant) | Hohlraum |
|---|---|---|
| 1 | [0,100]³ | [0,50]³ ohne Quelle |
| 2 | [0,60]×[0,100]×[0,60] | gerader Kanal x, z < 10, 10 < y < 100 |
| 3 (noch nicht umgesetzt) | wie 2 | Kanal mit Knick |

Gerechnet wird das volle gespiegelte Gebiet; die Spiegelreduktion (`--mirror-reduction auto`, Standard) rechnet den
Oktanten mit den reflektierenden Flächen (prüft die Symmetrie selbst, `config.json → mirror_directions`). Nicht mit
MPI kombinierbar (dann volles Gebiet).

```bash
m1pn_fast --case kobayashi-1ii --inflow vacuum --dimension 3 --degree 1 --elements 100 --pn-order 15 \
  --time 400 --background 1e-10 --fcs analytic --skip-fields ...   # --elements n: Elemente je 200 cm, n % 20 == 0
```

## First-Collision-Source (stationär)

`src/uncollided_box3d.hpp`: dieselbe Strahlintegral-Quadratur wie der Lattice-Kern (`uncollided_lattice3d_core.hpp`),
aber auf einem Rechteckgitter mit beliebigen Ebenen je Achse (Problem 1: ±10, ±50, ±100; Problem 2: ±10, ±60 bzw.
±100) und für t → ∞. Der stationäre Zustand hängt nicht davon ab, ab wann die Quelle σ_s ρ_u wirkt; der gestoßene
Anteil bekommt daher die stationäre ungestoßene Dichte ab t = 0 (keine Zeittabelle). CPU only.

Validierung (`kobayashi_uncollided_tests`, ctest): In Fall i ist ρ_u die ganze Lösung. Alle 46 Detektorwerte von
1i und 2i stimmen mit der analytischen Kyoto-Referenz innerhalb der Rundung ihrer drei Stellen (≤ 0.50 Einheiten der
letzten Stelle); Quadratur gegen 10 Punkte / doppelte Panels ≤ 4e-8; 0.3 ms je Punkt.

## Ausgabe und Auswertung

`detectors.csv`: Detektorlinien des Benchmarks (A: y-Richtung bei x = z = 5; Problem 1: B Diagonale, C x-Richtung
bei y = 55, z = 5; Problem 2: B x-Richtung bei y = 95, z = 5), `rho_grid` (Bernstein-Auswertung des Endzustands),
`rho_uncollided` (exakt am Punkt, mit FCS) und `rho` (Summe).

`scripts/kobayashi/compare.py CASE RUN...`: Abweichung gegen die Referenz und gegen Ardra P21
(Brown/Chang/Hanebutte, UCRL-JC-135163, Tab. III–VI; dort steht für Linie C von Problem 1 „y = 95“, die Werte gehören
aber zu y = 55). `--summary out.csv RUN...` für Kampagnen.

## Literatur (P_N ohne FCS)

Ardra P21 (LLNL), 50³ Zellen (Problem 1) bzw. 56×96×56 (Problem 2), Abweichung gegen die Referenz: 1ii meist 2–16 %,
bis −32 % am Rand; 2ii am Kanalende −17 bis −33 %; 2i einzelne Punkte −99 % / +376 %. 6 000–21 000 s auf 200–512
Prozessoren (ASCI Blue-Pacific, 1999).

## Ergebnisse

Folgen (Kampagne `nx/tmp/kobayashi/`).
