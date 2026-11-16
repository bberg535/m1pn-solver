# Randkorrekturen C1, C2, C5, C6 und Stabilisator-Regression L3

Branch `fix/element-weno-boundary-c1-c6` (Basis: `experiment/element-weno` @ `8efe517` plus Audit-Harness `90e6739`).
Hinweis (16.11.2026): Der Ordner `audit/` (Prüfprogramme, Ergebnisse) ist aus dem Arbeitsstand entfernt und liegt in der
Git-Historie (bis Tag `standard-20261116`); die Tests `tests/element_weno_fixes.cpp` bleiben.

Auslöser: Gegenbeispiele C1, C2, C5, C6 in `audit/REPORT.md`. Die WENO-/Element-MCL-Architektur ist unverändert;
alle Änderungen sind Schalter in `ElementWenoOptions` (`src/element_weno.hpp`). **Alle Schalter `false` reproduzieren 8efe517 bitgleich**
(`--legacy-boundary` in `m1pn_element_weno`, `AUDIT_FIX=legacy` im Audit): 29 Records der kleinen Tests und 4 Crossing-Records (64², P7) stimmen in
allen Gleitkommafeldern exakt überein (`results/fix/comparison.json`, `legacy_bitwise_identical_to_8efe517_*`). Die alten Resultate unter
`audit/results/*.jsonl` und die Referenzversion (Commit 8efe517) bleiben unverändert erhalten.

Neue Standardwerte: `vacuum_sign_limit`, `outward_bar_rounding`, `fullsphere_ghost`, `scale_invariant_weno` = **true**; `symmetric_residual` = **false**.
Achtung: `m1pn_fast` (`src/fast_main.cpp`) übernimmt diese Standardwerte ohne eigenen Legacy-Schalter.

Reproduktion: `bash audit/run_fix.sh` (ca. 25 min, ein Kern). Tests: `tests/element_weno_fixes.cpp` (43 Checks, CMake-Ziel `m1pn_element_weno_fix_tests`),
unverändert `tests/element_weno.cpp` (75 Checks, laufen auf dem Produktionscode mit den neuen Standardwerten: PASSED).

---

## C2 – Vorzeichenbegrenzung der akzeptierten Randresiduen (`vacuum_sign_limit`)

**Befund vorher.** Vakuum-Einstrom, Eckknoten ρ=1, J=0.75(1,1,0)/√2 (P3, p=1, 8×8): globale akzeptierte Massenrate max **+7.15e‑3**, PN-Target −6.33e‑3. Knotenweise
akzeptierte Randrate max +7.15e‑3 (Ecke), +4.5e‑3 (Kantenknoten), +1.9e‑7 (Ecke, auslaufend). Ursache: der Randlimiter beschränkt nur Kegelverletzungen; positive Dichtekorrekturen θ·h_i[0] > 0 sind
immer kegelzulässig und blieben unbegrenzt.

**Bilanz am Randknoten i** (alle Flächen f ∋ i, Ecken eingeschlossen). Mit `A_f = ∮_f φ_i`, `fa = Σ_f A_f n_f = ∮φ_i n`, `bar_i = Σ_f A_f U^bar_f / Σ A_f`:
Die Elementbeiträge summieren pro Element auf null (A.4 des Berichts; Messung 1.5e‑15), die geometrischen Beiträge der Elementsummen verbleiben als `−fa·F^L(u_i)`. Damit ist die
tatsächliche Randbilanz der Masse

    B_i = L_i + θ_i h_i[0],      L_i = A_i (bar_i − u_i)[0] − fa·F^L_i[0]     (Code: `lo_rate`, element_weno.cpp:868-871)
    d/dt ∫ρ = Σ_i B_i   (exakt bis auf Rundung: 1.2e‑15 absolut, `mass_balance_resid_abs_max`).

Mit LLF-Randzustand `U^bar_f = ½(u+F_n(u)) + ½(ext − F_n(ext))` ist `F_n(u)[0] = n·J` und somit pro Fläche

    L_i = −½ Σ_f A_f [ (ρ + n_f·J) − (ρ_ext,f − n_f·J_ext,f) ].

**Neue Bedingung.** Mit der kinetischen Einstromrate `I_i = Σ_f ∮_f φ_i (−kinetic[0]) ≥ 0` wird
`θ_i := min(θ_i^Kegel, θ_i^Vorzeichen)`, `θ_i^Vorzeichen = min(1, a_i/h_i[0])` für `h_i[0] > a_i`, `a_i = max(0, I_i − L_i)` (nach unten gerundet, `nextafter`).
Folge: `B_i ≤ max(L_i, I_i)`. Im Vakuum (`I_i = 0`, `L_i ≤ 0`, siehe C6/λ) gilt `a_i = −L_i = q^L_i ≥ 0`, also `B_i ≤ 0`: das ist das aus `BOUNDARY_AUDIT.md` (B13) bekannte
α^sign = min(1, q^L/δ), übertragen auf die elementbasierten Knotenresiduen; die Eckgeometrie steckt in `fa` und der Summe über f.
Bei vorgeschriebenem Einstrom bleibt die Aussage die schwächere `B_i ≤ max(L_i, I_i)` (kein Anspruch auf Monotonie, siehe Grenzen).

**Beweis der Kegelzulässigkeit.** Der knotenweise Randzustand ist `ū_i* = bar_i + θ_i g_i`, `g_i = h_i/A_i`. Nach Konstruktion ist `bar_i` im Kegel (LLF-Mittel zweier Kegelzustände, `ext` im Kegel)
und `bar_i + θ_i^Kegel g_i` im Kegel. Für `0 ≤ θ_i ≤ θ_i^Kegel` ist
`bar_i + θ_i g_i = (1 − θ_i/θ^K) bar_i + (θ_i/θ^K)(bar_i + θ^K g_i)` (θ^K > 0; für θ^K = 0 ist θ_i = 0 und nichts zu zeigen), eine Konvexkombination zweier Kegelelemente, also im Kegel (Konvexität des Lorentz-Kegels).
Die Forward-Euler-Aktualisierung ist unter der unveränderten CFL (`dt_bound`, Randflächenmassen eingeschlossen) die Konvexkombination
`(1 − dt(2Σd + A_i)/m_i) u_i + (dt/m_i) Σ_e 2d_i^e(ū_i^e + g_i^e) + (dt/m_i) A_i ū_i*`; die SSPRK3-Stufen sind Konvexkombinationen solcher Schritte. Der Kegel bleibt erhalten; die Elementsummen (Erhaltung) sind unberührt, weil nur der Randfaktor θ_i sinkt.
Eine Verkleinerung von θ_i ändert weder `dt_bound` noch die Elementlimiter.

**Test vor/nach** (`audit/results/fix/comparison.json`, `tests/element_weno_fixes.cpp`):

| Fall (T2, vacuum, P3 p=1 8×8, 7 Schritte, dt≈0.9·step_bound) | max. globale Massenrate vorher | nachher | max. nodale Randrate vorher | nachher |
|---|---|---|---|---|
| Ecke, J einwärts | **+7.15e‑3** | 3.5e‑18 | +7.15e‑3 | 6.9e‑18 |
| Ecke, J auswärts | −1.53e‑2 | −1.39e‑2 | +1.9e‑7 | 1.5e‑20 |
| Kantenknoten | −2.66e‑2 | −2.69e‑2 | +4.5e‑3 | 8.7e‑19 |
| Innenknoten (erreicht Rand nach ≈4 Elementen) | 2.3e‑17 | 1.2e‑16 | +1.5e‑5 | 6.8e‑21 |

T1 (isotrop, passende Einströmdaten) bleibt stationär (1.8e‑14, `sign_limited_nodes = 0`); Einstrom in ein leeres Gebiet füllt es weiterhin (Test „kinetic inflow into an empty domain still fills it“).
Crossing 64²/P7 (Einstrom ≠ 0): akzeptierte Massenrate − kinetische Einstromrate max −6.7e‑4 (nachher) bzw. −8.2e‑4 (vorher), also nie über dem Einstrom.

## C1 – Bar-State am Kegelrand (`outward_bar_rounding`)

**Reproduktion** (`audit/audit_c1_probe.cpp`, `audit/c1_highprecision.py`, 100 Stellen, exakte binary64-Eingaben): T2 `corner_cone_boundary_J`, Schritt 0, Stufe 2, Paar (Element 0, i=0, j=1):
`u_i = (0x1.2baa127db801fp-1, −0x1.2baa127db801fp-1, 0, 0)` (exakt auf dem Kegelrand, Lücke 0), `u_j = (0x1.26af4dc5756dbp-8, −0x1.26af4dc5756dap-8, 0, 0)` (relative Lücke 3.9e‑16),
`d = 0x1.7d9f4cf754635p-6`. Ergebnis in 100-stelliger Arithmetik: `ρ̄ = 0.03515410544183704201015707`, `|J̄| = 0.03515410544183704118858024`,
`ρ̄² − |J̄|² = +5.78e‑20 ≥ 0` – **der exakte Bar-State liegt im Kegel**. Auf nächste binary64 gerundet ist `ρ̄² − |J̄|² = −3.76e‑38 < 0`; **1 ulp in ρ** (relativ 1.97e‑16) genügt.
Klassifikation: reines Rundungsproblem, keine mathematische Verletzung (`classification: rounding_only`).

**Änderung.** `stable_base_bar_outward` (`src/base_bar.hpp`, `stable_base_bar` bleibt unverändert und wird von den anderen Pfaden weiter benutzt): Auswertung wie bisher in `long double`; Dichte gerichtet nach oben gerundet
(bei `double(ρ) < ρ_exakt` ein ulp), danach höchstens 8 weitere ulp **nur der Dichte**, bis `exact::cone` gilt; sonst `StepRejected` (echte Verletzung). Kein Floor, kein Clipping der Impulse; jede Anhebung wird gezählt
(`ElementWenoStats::outward_bars`). Dieselbe Konvention (nach außen gerundete Dichte, gezählt) verwenden `viscosity_ij_` und `Boundary::data` schon. Die Kegelzulässigkeit des Elementbarzustands bleibt gesichert, weil
`ū + ε e_ρ` für ε ≥ 0 im Kegel bleibt; die Massenerhaltungs-Summen ändern sich um ≤ 8 ulp(ρ).

**Test vor/nach.** Vorher: `StepRejected: Stable M1 base bar outside stored binary64 cone` im ersten Schritt (Ecke ρ=1, J=(−1,0,0), Vakuum sonst; auch Low-Order-Kontrolle). Nachher: 7 Schritte, `outward_bars = 48`,
0 Kegelverletzungen im Endzustand; Unit-Test mit den exakten Hexfloat-Eingaben (`legacy stable_base_bar rejects`, `outward bar: 1 ulp`, Dichte stimmt mit dem 100-Stellen-Wert auf 2e‑16 relativ).
**Offen/ehrlich:** In diesem Lauf hat eine Euler-Teilaktualisierung (Audit-Treiber, `adm`) den Kegel um 1 ulp verfehlt; `advance_ssprk3` repariert das bereits seit 8efe517 mit dem dokumentierten, gezählten `nudge_into_cone`
(`silent_nudges_ulp_repairs = 3`, `max_ulps_per_nudge = 1`). Ein Nachweis „mathematisch im Kegel“ für diese Teilaktualisierung (Rundungsfehler des Update-Ausdrucks) wurde nicht separat in hoher Genauigkeit geführt.

## C6 – Positive Vollkugel-Fortsetzung als M1-Ghost (`fullsphere_ghost`)

**Befund vorher.** Ghost = Halbraummomente der einlaufenden Daten (ρ_ext = ρ₀/2 im isotropen Fall). Low-Order-Kontrolle des stationären kompatiblen Zustands: max. Rate **0.846** (T1) bzw. 10.6–25.8 (C6-Reihe).

**Konstruktion.** Für jeden Randknoten i und jede Fläche f: `ext = m_in(g) + λ_if · m_out(ψ_i⁺)`, wobei
`m_in(g)` die bisherigen Halbraummomente der einlaufenden kinetischen Daten sind und `m_out(ψ_i⁺)` die Momente der **positiven Teilfunktion** `ψ_i⁺ = max(ψ_N,i, 0)` der PN-Rekonstruktion des Knotens auf der **auslaufenden** Halbkugel
(Gauss-Halbkugelregel, Grad 2N+1, exakt für polynomiale ψ). Die Vollkugel-Fortsetzung `ĝ = g (einlaufend) ∪ ψ_i⁺ (auslaufend)` ist nichtnegativ, daher liegt `ext = ∫ĝ(1,ω)dω` im Kegel (Dreiecksungleichung mit |ω|=1; bei Rundung
nach außen gerundete Dichte wie bei `data.moments`). Für nichtnegative, darstellbare ψ mit `g = ψ|_einlaufend` ist `ext = u` exakt, damit `U^bar = u`, `L_i = 0` und das Low-Order-Verfahren stationär.
Skalierung λ: Für positive ψ gilt `a_ext = ρ_o − n·J_o = ∫_out(1−μ)ψ ≤ ∫_out(1+μ)ψ ≤ b = ρ + n·J` (μ>0), also λ=1. Für PN-Rekonstruktionen mit negativen Lappen kann `a_ext > b` auftreten und das Low-Order-Verfahren würde
im Vakuum Masse erzeugen (nachgewiesen im ersten Versuch, Innenknoten-Fall: +1.3e‑5); dann wird der auslaufende Anteil mit `λ = b/a_ext ∈ [0,1)` skaliert (Kegelskalierung), und es gilt `L_i ≤ 0` im Vakuum.
Positive und nichtpositive Rekonstruktionen sind damit getrennt behandelt: Konsistenz nur im positiven Fall, im nichtpositiven Fall bleibt das Low-Order-Residuum ≠ 0 (aber kleiner als zuvor).
CFL: die Randrate `A_i` und damit `dt_bound` ändern sich nicht, da `U^bar` ein LLF-Mittel (Konvexkombination mit Gewicht ½) bleibt. Nicht verwendet wird die Ghost-Konstruktion, wenn `exterior_m1`/`exterior_state` vorgegeben ist.

**Test vor/nach** (`audit/audit_c6.cpp`, `results/fix/c6.jsonl`; Low-Order-Kontrolle = alle antidiffusiven Korrekturen aus, gleichförmiger Zustand, Einstrom g = ψ; max. Rate der Momentenzeilen):

| ψ (P3) | Ghost alt | Ghost neu | | ψ (P5/P7) | Ghost alt | Ghost neu |
|---|---|---|---|---|---|---|
| isotrop | 10.6 | 2.4e‑14 | | isotrop (P7) | 10.6 | 2.9e‑14 |
| 1+0.5 ω_x | 11.5 | 4.8e‑14 | | (1−ω_x)³ (P5/P7) | 25.8 | 2.8e‑13 / 1.9e‑13 |
| (1−ω_x)³ | 25.8 | 8.4e‑14 | | 1+0.9 ω_y (P7) | 12.2 | 1.1e‑13 |
| 1+0.9 ω_y | 12.2 | 5.5e‑14 | | exp(3ω_x) (P5, min ψ_N<0) | 42.1 | 1.6e‑2 |
| exp(3ω_x), P3: min ψ_N=−0.81 | 42.1 | 3.0 | | exp(8ω_x) (P7, min ψ_N=−28) | 2791 | 132 |

T1: Low-Order-Rate 0.846 → 2.3e‑15; Vollverfahren unverändert 1.8e‑14. **Nicht behoben:** die Randinkompatibilität des Onsager-Exterior bei P3 (C7; T3 bleibt mit Rate 1.54 nicht stationär, Ursache im PN-Rand, nicht im Ghost).

## C5 – Skaleninvarianter WENO-Sensor (`scale_invariant_weno`)

**Befund vorher.** `ω_l = lin_l/(‖c_l‖^r + ε)` mit absolutem `ε = 1e‑6`: `γ_min = 0.854` bei ρ~1, `0.993` ab ρ ≤ 1e‑4 (Sensor aus); 3 Schritte später 2.7e‑3 relative Abweichung zwischen s=1 und s ≤ 1e‑3.

**Konstruktion.** `ε_e = ε · S_e^r`, `S_e = max_l ‖c_l‖` über den Kandidaten des Elements (eigenes Polynom und Nachbarextensionen, alle linear in u). Für `u → s u` (s>0): `‖c_l‖^r → s^r ‖c_l‖^r`, `ε_e → s^r ε_e`, die normierten Gewichte `ω_l/Σω` und damit `u*` (∝ s) und `γ = 1 − min(1, ‖u−u*‖/‖u‖)^q` sind exakt invariant (bis Rundung).
**Exaktes Vakuum:** ist `‖own‖ = 0` (Element exakt konstant/leer), gilt wie bisher `γ = 1` per Definition; leeres Element neben nichtleerem: der Sensor des nichtleeren Elements sieht das Vakuum als glatten Kandidaten, ohne Floor, ohne Division durch 0 (`S_e > 0`, weil `‖own‖ > 0`).
Das ε-Verhältnis `1e‑6` bleibt der Parameter `weno_epsilon`; bei ρ~1 stimmt das neue Verhalten mit dem alten bis auf den Faktor `S^r ~ O(1)` überein (T8 s=1: `γ_min` 0.854 → 0.854 im 1D-Test, 0.626 → 0.602 im 2D-Test).

**Test vor/nach.** γ-Abweichung über s ∈ [1e‑100, 1e100]: **0.144 → 2.9e‑14**. T8 (2D, 3 Schritte): relative Abweichung gegen s=1 bei s=1e‑3: 2.7e‑3 → **3.2e‑14**; s=1e‑30: 2.7e‑3 → 9.9e‑14; s=1e‑100: 2.7e‑3 → 1.4e‑13; `rate(su)/s − rate(u)` ≤ 4.8e‑15 relativ (Test). Halb-Vakuum-Fälle: `γ_min`, Limiter-Anteil (52.4 %) und Kegel unabhängig von s. Exaktes Vakuum: Rate 0, 0 Kegelverletzungen, 0 Reparaturen.

## L3 – Analytischer Regressionstest und Alternativen

**Test (`tests/element_weno_fixes.cpp`, `l3_analytic`, unabhängige Dichtmatrix-Referenz).** 1D P1, `h₁=1`, `h₂=1/2`, Gradienten (1, 3/2), γ=1, `ν_e = h_e/2`, ungewichtete L²-Projektion auf stetiges P1.
Rechnung: `M g = b` mit `g = (11/12, 7/6, 5/3)`; Element 1: `ν₁∫u'(u'−g) = ½·(−1/24) = −1/48`, Element 2: `¼·(1/16) = 1/64`; Summe `−1/48 + 1/64 = −1/192`.
Testwert `s_h(u,u) = −0.00520833… = −1/192` auf 1e‑15 – **bestanden**, ebenso `λ_min(S) < 0` (−3.4e‑3).
Bedeutung: Die Positivität `uᵀS(1)u = ν(‖∇u‖² − ‖g_h‖²) ≥ 0` (Pythagoras) gilt nur für **konstantes** ν. Der Produktionscode (gleichmäßiges Gitter) ist davon nicht betroffen; für Gitter mit variablem `ν_e` oder variablem γ gilt sie nicht.

**Alternativen** (im Test geprüft):
1. *Gewichtete Projektion* (Gewichte `ν_e` im Projektions-Innenprodukt): `s_h(u,u) = ‖∇u‖²_ν − ‖g‖²_ν ≥ 0` (Pythagoras im ν-gewichteten Innenprodukt) – im Beispiel **+0.00625 = 1/160**, symmetrisch (4e‑17) und PSD für γ=1, aber **nicht** für variables γ (Asymmetrie bleibt).
2. *Symmetrische Residualstabilisierung* `s_sym(v,u) = Σ_e ν_e[(1−γ_e)(∇v,∇u)_e + γ_e((I−P)∇v,(I−P)∇u)_e]` (Option `symmetric_residual`): Summe positiv-semidefiniter, symmetrischer Terme, daher **symmetrisch und PSD für jedes γ_e ∈ [0,1] und jedes ν_e > 0**, `S·1 = 0` und Reproduktion von Q_p (γ=1) wie zuvor; bei γ ≡ 1 und konstantem ν mit dem Original identisch (P ist orthogonale Projektion, globale Orthogonalität), sonst nicht.
   Gemessen: unregelmäßiges Gitter 6 Elemente, zufälliges γ: λ_min ≥ −2.7e‑16, Symmetriedefekt < 1e‑14 (Original: Defekt 0.38); Produktion p=1,2,3 (uniform, zufälliges γ): Symmetrie 1e‑16, PSD, Original-Defekt 0.13–0.29, Gleichheit bei γ=1 auf 4e‑16.
   Kosten: zwei zusätzliche globale Massenlösungen je Richtung; Crossing 64²/P7: Laufzeit 51.8 s statt 40.7 s (+27 %), Dichte unterscheidet sich von der Originalform um 1.0e‑3 relativ nach dem Vollverfahren (identische Kegel-/Massen-/Energiebudgets, `energy_integral_total` −2.552e‑5 vs −2.559e‑5).
**Empfehlung.** Original beibehalten, solange das Gitter uniform ist (PSD für alle γ ∈ [0,1] wurde durch Eckenaufzählung auf 7 Netzen bestätigt, Beweis offen, siehe Bericht E/L3); `symmetric_residual` für variable `ν_e`/Teilgitter bereithalten (nicht Standard).

---

## Regression: vollständige Budgets (Crossing 64², P7, p=1, T=0.32, CFL 0.5·step_bound, 325 Schritte)

Alle Werte aus `results/fix/comparison.json` (legacy = 8efe517; new = Standardwerte; sym = zusätzlich `symmetric_residual`).

| Größe (Variante *full*) | legacy | new | sym |
|---|---|---|---|
| Kegelverletzungen (Euler-Teilschritte / SSP-Stufen) | 0 / 0 | 0 / 0 | 0 / 0 |
| min. relative Kegelmarge | 2.993e‑4 | 2.991e‑4 | 2.992e‑4 |
| stille Reparaturen / Outward-Bars | 0 / – | 0 / 0 | 0 / 0 |
| max. Elementsummenfehler (roh / limitiert, relativ) | 1.5e‑15 / 1.3e‑15 | 1.5e‑15 / 1.3e‑15 | 1.6e‑15 / 1.3e‑15 |
| Massenbilanz-Residuum absolut (Rundung) | 1.2e‑15 | 1.2e‑15 | (relativ 1.1e‑10, Skala klein) |
| Energiebudget-Schlussfehler (relativ) | 5.5e‑15 | 4.6e‑15 | 3.6e‑15 |
| max. Synchronisierungsrate (relativ zu ‖a‖²_M) | 9.5e‑16 | 9.6e‑16 | 9.5e‑16 |
| ∫ Energierate gesamt | −2.501e‑5 | −2.559e‑5 | −2.552e‑5 |
| Synchronisierungsdefekt (M-Norm, relativ) | 8.98 % | 8.94 % | 8.94 % |
| akzeptierte Massenrate − kinet. Einstrom (max) | −8.2e‑4 | −6.7e‑4 | – |
| min. Rand-θ / Anteil Randknoten mit θ<1 | 0.013 / 8.9 % | 0 / 11.0 % (θ=0 an Knoten ohne Spielraum) | 0 / 11.0 % |
| Elementlimiter aktiv innen / Rand | 31.96 % / 13.8 % | 31.97 % / 13.9 % | 31.97 % / 13.9 % |
| Endmasse | 0.06255 | 0.06257 | 0.06257 |
| relative L²-Dichtedifferenz zu legacy | – | 1.33e‑3 | 1.31e‑3 |

Zuordnung der Differenz new − legacy (nur *full*, einzelne Schalter): Ghost allein 5.0e‑3, Vorzeichenlimiter allein 1.3e‑3, WENO-Skala allein 3.6e‑4, Ghost+Vorzeichen+C1 1.28e‑3 (`attribution_density_diff_to_legacy.json`).
Es gibt für diesen Fall keine Referenzlösung; die Differenzen sind Effektgrößen, **keine** Genauigkeitsaussage. Low-Order-Kontrolle: unverändert (Differenz zu legacy 1.3e‑3 relativ, `sync_defect` 22.1 %); *no‑limit*-Diagnose bricht weiterhin nach 113 Schritten durch Kegelverletzung ab (−3.1 %).

Operatorebene unverändert (`results/fix/ops_new.jsonl`, `energy_random_new.jsonl`): `λ_max(Q_impl) = 2.4e‑16`; diskrete Randenergie ≤ −0.24 ∮|u|²; PSD aller γ-Ecken ≥ −5.2e‑16; Reproduktion 4.2e‑16; Hill-Climbing-Minimum 1.6e‑3 > 0.

## Grenzen / nicht behoben

* **C3 (positive Synchronisierungsenergie)** unverändert (T2: max. +18; Low-Order-Kontrolle +0.082); **C7** (Onsager-Rand P3, nicht-axialsymmetrische Normale) unverändert.
* C2 beweist Monotonie nur im Vakuum (`I_i = 0`); mit Einstrom gilt `B_i ≤ max(L_i, I_i)`. Der Beweis setzt die CFL und die exakte Elementsummen-Null voraus; sie ist in der Praxis bis auf Rundung (≤ 1.2e‑15 absolut) erfüllt, die Vorzeichenaussage daher ebenfalls nur bis auf Rundung (gemessene Überschreitungen ≤ 7e‑18 in T2).
* C6: Konsistenz nur für positive PN-Rekonstruktionen; bei negativen Lappen nur Verbesserung, kein Verschwinden des Residuums. Hemisphärenregel: Grad 2N+1, Kosten ∝ (Randknoten × Quadraturpunkte × q) pro Stufe (Crossing-Laufzeit +10 %).
* C1: Rundungsklassifikation für diesen einen Fall in 100 Stellen, nicht für beliebige Paare; Grenze 8 ulp ist eine Wahl, keine Herleitung.
* Keine Genauigkeitsverbesserung behauptet. T6 (glatt, kompatibler Einstrom, PN3, p=2, E=4/8/16): **vorher = nachher** in allen gedruckten Stellen (RMS-Dichtefehler Rand 1.56e‑3 / 1.66e‑4 / 3.72e‑5, innen 1.77e‑3 / 1.77e‑4 / 2.02e‑5; Ordnungen 3.24 / 2.16 am Rand, 3.32 / 3.13 innen; 0 Kegelverletzungen, 0 Reparaturen; `results/fix/T6_{legacy,new}.jsonl`) – die Änderungen berühren glatte Lösungen nicht.
  T7 (steiler Gradient, unterauflösend): Fehler gegen E=48 0.5028→0.5037 (Rand), 0.5564→0.5565 (innen); Rand-θ_min Rand 0.0253→0 (Vorzeichen), innen 1→1; `γ_min` 0→0.18 (Rand und innen, skaleninvarianter Sensor bei Hintergrund ρ=1e‑3), Limiteranteil 18.8 %/20.3 %→18.8 %/21.1 % (Rand-/Innenelemente, Randfall) bzw. 4.5 %/38.6 %→4.5 %/38.5 % (Innenfall).
