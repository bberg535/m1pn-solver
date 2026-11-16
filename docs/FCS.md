# First-Collision-Source (FCS) für Element-WENO

Stand 12.11.2026, Branch `feature/element-weno-fcs`.

## 1. Literatur

| Quelle | Inhalt | verwendet für |
|---|---|---|
| Hauck & McClarren, *A collision-based hybrid method for time-dependent, linear, kinetic transport equations*, Multiscale Model. Simul. 11(4), 1197–1227 (2013) | Aufspaltung ψ = ψ_u + ψ_c, ψ_u hochaufgelöst (S_N), ψ_c niedrig aufgelöst; nach jedem Schritt Umlabeln (ψ_c ← ψ_c + ψ_u, ψ_u ← 0) | Grundformulierung; Stufe 2 (Umlabeln) |
| Kusch & Stammer, *A robust collision source method for rank adaptive dynamical low-rank approximation in radiation therapy*, arXiv:2111.07160 (ESAIM M2AN 2023) | Gl. (7a/b) Split; (24) unkollidiert S_N mit gerichteter Quadratur, kollidiert als Momente; Line Source | Gleichungen, Momente für ψ_c |
| Crockatt, Christlieb, Garrett, Hauck, JCP 2017 (IDC) und JCP 2019 (DIRK, Raum-Zeit-DG) | Hybridverfahren beliebiger Zeitordnung | Zeitordnung bei Umlabeln (Stufe 2) |
| Crockatt, Christlieb, Hauck, JCP 422 (2020) | Nyström-Rekonstruktion, Defektkorrektur des Splittingfehlers | Stufe 2 |
| Galindo-Olarte, DeCaria, Hauck, J. Comput. Theor. Transp. (2025) | Fehleranalyse des Hybridverfahrens | Einordnung |
| Whewell et al., Nucl. Sci. Eng. | Multigruppen-Erweiterung | – |
| Krotz, Hauck, McClarren, arXiv:2312.04217 | Hybrid Monte Carlo / DG mit Umlabeln | Alternative für ψ_u |
| Bennett & McClarren, arXiv:2205.15783; arXiv:2206.13445 | Analytische unkollidierte Lösungen (Plane, Line, Gauß), exakte unkollidierte Quelle in einem Zeitschrittverfahren | Formeln in `src/uncollided.hpp` |
| Garrett & Hauck, TTSP 42 (2013); Ganapol, AZURV1 (OECD/NEA 2008) | Line-Source-Benchmark und exakte Lösung | Referenz (`scripts/line_source/reference.py`) |
| Alcouffe, O'Dell, Brinkley | Klassische First-Collision-Source mit diskreter S_N-Bilanz | Hintergrund |
| Masterarbeit, `03_projects/Masterarbeit/FCS_TestSuite` | M1PN-Plane-Source mit analytischem ballistischem Anteil, Quelle zur Schrittmitte | Vorlage für die Kopplung |

## 2. Formulierung

ψ = ψ_u + ψ_c mit (Kusch & Stammer (7)):

```
∂_t ψ_u + Ω·∇ψ_u + σ_t ψ_u = 0,                                   ψ_u(0) = Puls,
∂_t ψ_c + Ω·∇ψ_c + σ_t ψ_c = σ_s/(4π) (ρ_u + ρ_c) + q/(4π),       ψ_c(0) = Boden.
```

Der kollidierte Anteil wird mit Element-WENO (M1PN) gerechnet, die einzige Änderung ist die zusätzliche isotrope
Quelle σ_s ρ_u. Sie ist nichtnegativ, der M1-Kegel bleibt daher erhalten. Ausgegeben wird ρ = ρ_c + ρ_u.

**Unkollidierter Anteil (Stufe 1, analytisch):** homogenes, unendliches Medium, isotroper Gauß-Puls
ρ_0 = m N(x; c, ς²):

- Plane (1D): ρ_u(x,t) = e^{−σ_t t} m/(4t) [erf((y+t)/(√2ς)) − erf((y−t)/(√2ς))], y = x − c.
- Line (2D): ρ_u(r,t) = e^{−σ_t t} m ∫_0^{π/2} sin a · ring(r, t sin a) da, ring = Mittel des normierten
  2D-Gauß über den Kreis vom Radius s im Abstand r: e^{−(r−s)²/(2ς²)} I0e(rs/ς²)/(2πς²).
  Ausgewertet wird nur das Fenster |t sin a − r| ≤ 10ς (6 Panels × 8 Gauß-Punkte), tabelliert in r mit ς/8 und
  kubischer Interpolation (Catmull-Rom, bei null abgeschnitten).

## 3. Umsetzung

- `ElementWenoOptions::uncollided_density(t, points, out)`: ρ_u an einem Satz physikalischer Punkte.
- Quadratur (`collided_source`): Tensor-Gauß mit p + 3 Punkten je Richtung und Element; Knotenquelle
  q_i = ∫σ_s ρ_u φ_i / ∫φ_i ≥ 0. Die Regel integriert die Bernstein-Funktionen exakt (∫φ_i = gelumpte Masse),
  die Masse der Quelle ist daher exakt σ_s ∫ρ_u (dieselbe Regel).
- Reaktion: q_i geht in die vorhandene exakte Reaktionslösung ein (ρ += q τ φ_1(−σ_a τ)); ausgewertet zur Mitte
  jedes Strang-Halbschritts (t + dt/4 und t + 3dt/4), also zweite Ordnung in der Zeit.
- Bilanz: `injected` enthält die Stoßquelle; die Bilanz des kollidierten Anteils bleibt bis auf Rundung erfüllt.
- Treiber: `m1pn_fast --stabilization element-weno --fcs analytic` für `line-source` (2D) und `plane-source` (1D),
  homogenes Medium (`--sigma-s`, `--sigma-a`), Vakuum-Einstrom. Der Träger t + 6ς muss bis T im Gebiet bleiben
  (sonst Abbruch). `final.csv`: `u0` ist die Gesamtdichte, `rho_uncollided` steht zusätzlich in der letzten Spalte;
  u1–u3 sind die kollidierten Momente. `summary.json`: `fcs`, `uncollided_mass_final`, `uncollided_mass_exact`.
- Tests: `tests/element_weno_fcs.cpp` (i0e, Masse und Faltung der Pulse, Tabellierung, Quadratur, Reaktion, Bilanz).

## 4. Ergebnisse (lokal, ein Kern, Q1, `--degree 1`)

**Plane Source 1D** (σ_s = 1, T = 0.2, 400 Elemente, ohne Boden; Referenz P39 ohne FCS auf demselben Gitter):
relativer L1-Abstand P3 6.4 % → 0.17 % mit FCS, P7 0.05 % → 0.02 %.

**Line Source GHS** (Garrett–Hauck, auf [0,1]² skaliert: σ_s = 3, T = 1/3, ς = 0.01, Boden 9e-4; relativer
L1-Fehler gegen die gefaltete exakte Lösung, Auswertung wie `scripts/line_source/analyze.py`):

| | P1 | P3 | P7 | P11 |
|---|---:|---:|---:|---:|
| FCS, 128² | 0.444 | 0.226 | 0.099 | 0.043 |
| FCS, 256² | 0.445 | 0.240 | 0.116 | 0.064 |
| ohne FCS, 128² | – | – | 0.819 | 0.572 |
| ohne FCS, 256² (Kampagne 10.11.) | – | 1.400 | 1.045 | 0.844 |
| ohne FCS mit Winkelfilter, 256² (Kampagne 10.11.) | – | 0.530 | 0.130 | 0.421 |

- Der unkollidierte Anteil ist exakt: Sein Fehleranteil gegen `phi_uncollided` der Referenz liegt unter 1e-4,
  die unkollidierte Masse bei 6e-8 relativ. Der gesamte Restfehler ist der P_N-Fehler des kollidierten Anteils.
- Der Fehler hängt stark von P ab und kaum vom Gitter (256² leicht schlechter als 128²: das gröbere Gitter
  verschmiert die verbleibenden Winkelstrukturen der einmal gestoßenen Teilchen). Ohne FCS zeigt sich dasselbe
  (P3: 1.40 auf 256², 1.44 auf 512²).
- FCS ohne Filter ist bei P7 und P11 genauer als die gefilterten Läufe ohne FCS; der Filter wird nicht gebraucht.
- VOID (σ_s = 0): L1 3e-4 (nur der abfließende Boden), also exakt.
- Bilanz des kollidierten Anteils ≤ 4e-14; eingespeiste Stoßmasse 0.632121 gegen 1 − e⁻¹ = 0.632121.

## 4a. Effizienzschleife

GHS, 128², P3, ein Kern, Löserzeit (`phase_seconds`; `fcs_density` = ρ_u, `fcs_quadrature` = Knotenquelle):

| Stand | Löser | FCS-Reaktion | davon ρ_u | davon Quadratur |
|---|---:|---:|---:|---:|
| ohne FCS | 5.15 s | – | – | – |
| FCS, erste Fassung (16 × 16 θ-Punkte, Tabelle ς/32 linear) | 10.69 s | 6.96 s | 6.32 s | 0.61 s |
| θ: 8 × 8 Punkte, Tabelle ς/16 kubisch, Punktschleife parallel | 5.27 s | 1.58 s | 0.99 s | 0.57 s |
| Elemente außerhalb des Trägers von ρ_u überspringen | 4.69 s | 1.08 s | 0.95 s | 0.11 s |
| θ: 6 × 8 Punkte, Tabelle ς/8 kubisch | **4.20 s** | **0.58 s** | 0.45 s | 0.11 s |

- Genauigkeit je Schritt durch Tests abgesichert: Fensterintegral gegen volles Integral 1e-7, Tabelle 2e-5 relativ
  zum Peak; L1 der Line Source unverändert (0.2262 in allen Fassungen).
- **Mit FCS ist der Lauf schneller als ohne:** Ohne den δ-Puls im Gitter greift der Limiter in 60 250 statt
  437 372 Element-Stufen (Limiter 0.96 s statt 2.48 s). Das gleicht die FCS-Kosten mehr als aus.
- Bei höherem P (128²): P7 6.6 s mit FCS gegen 6.5 s ohne (L1 0.099 gegen 0.819), P11 11.2 s gegen 11.1 s (L1 0.043
  gegen 0.572). FCS kostet hier netto nichts und senkt den Fehler um den Faktor 8 bzw. 13.

## 4b. Nachmessung 15.11.2026 (NX, ein Sockel, 32 Threads, SSPRK(4,3))

GHS Line Source, Q1, `m1pn_fast --stabilization element-weno --time-integrator ssprk43`, Gesamtzeit des Prozesses,
relativer L1-Fehler gegen die gefaltete exakte Lösung (`scripts/line_source/analyze.py`). Rohdaten
`nx/tmp/fcs_speed_20261115/`.

| Gitter | PN | ohne FCS: L1 / Zeit | mit FCS: L1 / Zeit |
|---|---|---|---|
| 256² | P7 | 1.045 / 8.4 s | 0.116 / 8.7 s |
| 256² | P11 | 0.844 / 13.6 s | 0.064 / 14.0 s |
| 256² | P15 | 0.636 / 20.4 s | 0.035 / 20.3 s |
| 512² | P7 | 1.110 / 41.4 s | 0.120 / 43.7 s |
| 512² | P11 | 0.944 / 60.3 s | 0.069 / 62.6 s |
| 512² | P15 | 0.736 / 85.4 s | 0.041 / 88.2 s |

FCS kostet 0-5 % Laufzeit (Reaktion mit Stoßquelle, ρ_u 3.7 s bei 512²) und senkt den Fehler um den Faktor 9-18.
FCS mit P7 (0.116 in 8.7 s) ist genauer als ohne FCS mit P15 (0.636 in 20.4 s). Der Fehler wird vom P_N-Fehler des
kollidierten Anteils bestimmt und fällt mit P, nicht mit dem Gitter (512² leicht schlechter als 256²).

## 5. Grenzen und Stufe 2

- Nur homogene Medien mit analytischem ψ_u. Für den Lattice (Quelle im Gebiet, Absorber) braucht ψ_u einen eigenen
  Löser: S_N mit wenigen gerichteten Richtungen bzw. Ray Tracing und Umlabeln nach Hauck & McClarren (2013).
  Bei einer isotropen ausgedehnten Quelle wandern die Ray Effects dann in ψ_u; dort ist der Gewinn kleiner.
- Die unkollidierte Stromdichte wird nicht ausgegeben (u1–u3 sind kollidiert).
- Rand: ψ_u ist die Lösung im unendlichen Medium; der Boden des kollidierten Anteils fließt wie ohne FCS über den
  Vakuumrand ab.
