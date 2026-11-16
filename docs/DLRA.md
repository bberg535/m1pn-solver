> **Entfernt am 16.11.2026** (Branch `feature/combine-202611`): DLRA wird nicht mehr benötigt; Code, Tests und Skripte
> sind aus dem Solver genommen. Letzter Stand mit DLRA: Tag `standard-20261116`. Dieses Dokument bleibt als Bericht.

# Dynamische Niedrigrang-Approximation (DLRA) für Element-WENO

Stand 12.11.2026, Branch `feature/element-weno-dlra` (Basis: `feature/element-weno-fcs` @ d43c0ad).
Code: `src/element_weno_dlra.{hpp,cpp}`, Treiber `m1pn_fast --dlra on`, Tests `tests/element_weno_dlra.cpp`,
Skripte `scripts/dlra/`.

## 1. Literatur

| Quelle | Inhalt | verwendet für |
|---|---|---|
| Kusch & Stammer, *A robust collision source method for rank adaptive dynamical low-rank approximation in radiation therapy*, ESAIM: M2AN 57 (2023), arXiv:2111.07160 | P_N-Momente u ≈ X S Wᵀ (Raum × Rang × Winkel), rangadaptiver BUG-Integrator, Kollidiert/Unkollidiert-Split (unkollidiert gerichtete Quadratur, kollidiert niedrigrangig), L²-Stabilität unabhängig von steifer Streuung, Toleranz ϑ = ϑ̄‖S‖_F; Line Source: PN 5408 s gegen DLRA 1009–1278 s, „hohe Ränge nötig“ | Grundidee, Split, Toleranzkonvention |
| Stammer, Burlacu, Wahl, Lathouwers, Kusch, *A Deterministic Dynamical Low-rank Approach for Charged Particle Transport*, M&C 2025, arXiv:2412.09484 | unkollidiert per Ray Tracing, kollidiert P_N + DLRA, rangadaptiv | Bestätigung: Strahl außerhalb des Niedrigrangteils |
| Stammer, *A high-order deterministic dynamical low-rank method for proton transport in heterogeneous media*, J. Comput. Phys. 558 (2026) | dieselbe Zerlegung, heterogene Medien | Einordnung |
| Ceruti & Lubich, *An unconventional robust integrator for dynamical low-rank approximation*, BIT 62 (2022) | BUG-Integrator: K-, L-Schritt parallel, Galerkin-S-Schritt | Integrator |
| Ceruti, Kusch, Lubich, *A rank-adaptive robust integrator for dynamical low-rank approximation*, BIT 62 (2022), arXiv:2104.05247 | augmentierte Basen [X, K₁], [W, L₁] (Rang ≤ 2r), Trunkierung per SVD, ϑ ∝ h | Augmentierung, Trunkierung, Toleranz pro Zeiteinheit |
| Einkemmer, Kusch, Schotthöfer, *Conservation properties of the augmented basis update & Galerkin integrator for kinetic problems* (v3: *Construction of high-order conservative BUG integrators*), arXiv:2311.06399 | erhaltende Trunkierung: Erhaltungs-Basisfunktionen bleiben fest in der Basis; Augmentierung für RK-S-Schritte | Feste l ≤ 1-Momente außerhalb des Niedrigrangfaktors |
| Kusch, Einkemmer, Ceruti, *On the stability of robust dynamical low-rank approximations for hyperbolic problems*, SISC 45 (2023) | Stabilität/CFL des BUG für P_N mit Upwind | Zeitschritt bleibt der des Vollrangverfahrens |
| Ceruti, Kusch, Lubich, *A parallel rank-adaptive integrator for dynamical low-rank approximation*, arXiv:2304.05660; Kusch, *Second-order robust parallel integrators*, arXiv:2403.02834 | parallele bzw. zweite Ordnung | Ausblick (unser BUG ist erster Ordnung) |
| Peng, McClarren, Frank, *A low-rank method for two-dimensional time-dependent radiation transport calculations*, J. Comput. Phys. 421 (2020), arXiv:1912.07522 | P_N + DLRA (Projektor-Splitting), Line Source; „höhere Genauigkeit durch mehr Basisfunktionen bei festem Rang“ | Studie mit festem Rang und steigendem N |

## 2. Formulierung für Element-WENO (M1PN)

Zustand pro Knoten: aktive P_N-Koeffizienten a ∈ ℝ^{n × q}. Zerlegung

    a = [ a_L | X S Wᵀ ],   a_L ∈ ℝ^{n×4} (l ≤ 1),   X ∈ ℝ^{n×r},  XᵀMX = I,   W ∈ ℝ^{q_H×r},  WᵀW = I.

- **Die M1-Momente bleiben vollrangig.** Sie durchlaufen unverändert den limitierten Niedrigordnungsschritt
  (`ElementWeno::limited_low_load`, aus `rate()` herausgezogen, bitgleich): Kegel, Massenbilanz, Randkorrekturen
  C1–C6 gelten exakt wie im Vollrangverfahren. Das ist die erhaltende Trunkierung von Einkemmer/Kusch/Schotthöfer
  in der Form „Erhaltungsgrößen außerhalb des Faktors“; die Massenbilanz schließt auf Rundung (Tests: 1e-16).
- **Nur l ≥ 2 ist niedrigrangig.** Der Hochordnungs-P_N-Operator ist bei festem WENO-γ linear und von der Form
  Σ_k T_k(a) B_k mit räumlichen Operatoren T_k, die komponentenweise wirken (M⁻¹C_d, s_h, M⁻¹, Randflächenmassen),
  und Winkelmatrizen B_k (A_d, (A_n − llf)ᵀ, llf_datum). Für a = Z Qᵀ mit Z = [a_L | Y] wirken alle T_k auf die
  4 + r Spalten von Z statt auf q Spalten. `ElementWenoDlra::factored_rate` reproduziert `rate()` auf 1e-15
  (offenes Gitter mit Einstrom und C6-Ghost, periodisch, 1D/2D, mit/ohne Paritätsreduktion).
- **Skalarprodukt:** X ist M-orthonormal (konsistente Bernstein-Masse). Dann ist der S-Schritt ein Galerkin-Schritt
  Ṡ = X̂ᵀ R(a) Ŵ des Residuums R = M F *ohne* Massenlösung, und ‖a_H‖_M = ‖S‖_F.

### Zeitschritt

1. Reaktion Δt/2 (Filter, Streuung, Absorption, Quellen, FCS): a_L wie bisher; l ≥ 2 ist separabel
   (e^{−τ(σ_a+σ_s)(x)} · e^{−τ f (l/(N+1))⁴}) und wirkt auf X bzw. W mit Re-Orthonormalisierung.
2. K- und L-Schritt (explizites Euler, a_L eingefroren), **eine gemeinsame Operatorauswertung** bei aⁿ:
   K₁ = XS + Δt F(aⁿ)W, L₁ = WSᵀ + Δt (XᵀM F(aⁿ))ᵀ.
3. Augmentierung X̂ = orth_M[X, K₁] (CGS2 + Eigenzerlegung der Gram-Matrix + Cholesky-QR), Ŵ = orth[W, L₁],
   Ŝ₀ = X̂ᵀMX S WᵀŴ.
4. S-Schritt gekoppelt mit dem limitierten M1-Teil in den SSP-Stufen des Verfahrens (SSPRK3 bzw. SSPRK(4,3)).
   **Schnelle Variante (Standard):** Alle Operatoren auf X̂ (Gradienten, s_h, Randlasten, Projektionen X̂ᵀ·,
   (MX̂)ᵀ·) werden einmal pro Schritt vorberechnet; pro Stufe bleiben Operatoren auf den 4 M1-Spalten und r×r-Produkte.
   Einzige Abweichung: γ in s_h des l ≥ 2-Galerkin-Terms ist das von aⁿ (der M1-Teil nutzt das aktuelle γ).
   Mit γ ≡ 1 identisch zur stufenweisen Auswertung (Test: 2e-14), mit WENO 6e-4 nach 10 Schritten.
5. Trunkierung: SVD von Ŝ, kleinster Rang mit (Σ_{j>r₁} σ_j²)^{1/2} ≤ ϑ, ϑ = ϑ̄ ‖a‖_M (Modus `step`, Kusch/Stammer)
   bzw. ϑ = ϑ̄ ‖a‖_M Δt (Modus `rate`, Ceruti/Kusch/Lubich); alternativ fester Rang.
6. Reaktion Δt/2.

Der Zeitschritt ist derselbe wie im Vollrangverfahren (Realisierbarkeitsschranke des M1-Teils, Hochordnungsschranke).
