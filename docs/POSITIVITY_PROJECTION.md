# Momententreue Positivitätsprojektion am Rand (Element-WENO)

Branch `feature/boundary-positivity-projection` (Basis `feature/element-weno-dlra` @ `bdd5d3f`).
Grundlage: Notiz *Momententreue Positivitätsprojektion – Nichtnegative Winkelrekonstruktion für den M1–PN-Randzustand*
(13.11.2026). Ergebnisse der NX-Kampagne: [results/positivity/](../results/positivity/README.md).

## 1. Was ersetzt wird

Der C6-Ghost (`fullsphere_ghost`, [BOUNDARY_FIXES.md](BOUNDARY_FIXES.md)) setzt am Randknoten i

    ext = m_in(g) + λ · m_out(max(ψ_N,i, 0)),     λ = min(1, b/a_ext)

mit den auslaufenden Halbraummomenten des **punktweise abgeschnittenen** PN-Traces. Das Abschneiden ändert die
Momente, und λ < 1 skaliert sie nachträglich. Neu (Option `positivity_projection`, CLI `--positivity-projection`):
Der Trace wird auf der **ganzen Kugel** auf die nichtnegativen, momententreuen Winkelmaße projiziert (Notiz, Gl. 4)

    min_z ½ Σ_k w_k (z_k − y_k)²   u. d. N.   z_k ≥ 0,   Σ_k w_k m_k z_k = U_i = (ρ_i, J_i),

y_k = ψ_N,i(Ω_k), und der Ghost verwendet m_out(z*) = Σ_{Ω_k·n>0} w_k m_k z_k*. U_i ist genau `low.row(i)`, also
`project(a_i)`. Da z* ≥ 0 die vollen Momente U_i hat, gilt a_ext ≤ b automatisch; λ = 1 bis auf Rundung
(gezählt als `scaled`, in allen Läufen 0).

## 2. Löser

KKT (Notiz, Gl. 5–7): z_k = max(0, y_k − ξ·m_k), ξ ∈ ℝ⁴ löst F(ξ) = Σ w m z − U = 0. F ist der Gradient der konkaven
dualen Funktion φ(ξ) = −½ Σ w z(ξ)² − ξ·U. Verfahren (`src/positivity_projection.cpp`):

* **Identität zuerst:** y > 0 und |Σ w m y − U| ≤ tol ⇒ z = y exakt (bitgleich). Die Regel (2N+1 je Halbkugel)
  integriert m·ψ_N exakt; nichtnegative PN-Traces bleiben unverändert.
* **Semiglattes Newton** d = (H + μI)⁻¹F, H = Σ_{aktiv} w m mᵀ (4×4, LDLT; μ = 1e−14·tr H, bei Singularität erhöht),
  **Armijo** auf φ. Ein Durchlauf pro Auswertung berechnet F, H, φ zusammen (SIMD, O(Q)).
* **Abbruch:** ‖F‖∞ ≤ 1e−12 · (Σ w|y| + ρ). z ≥ 0 und Komplementarität gelten per Konstruktion, das Momentenresiduum
  ist damit das einzige zu prüfende KKT-Residuum (Notiz, „Numerische Sicherheit“). Gemessen: ≤ 1e−12.
* **Warmstart:** ξ des letzten Aufrufs desselben Randknotens (getrennt für einfache und ergänzte Regel). Nur
  Startwert; die Lösung ist eindeutig (gemessen: Abweichung zum Kaltstart 1.8e−12). `--projection-cold-start`
  schaltet ihn ab.

### Lösbarkeit (Notiz, Abschnitt 5)

U ∈ R1 garantiert nicht U ∈ cone{m_k}. Für ρ > 0 gilt U = ρ(1, u) ∈ int cone{m_k} ⇔ u ∈ int conv{Ω_k}. Die Regel
ist ein Breiten-Längen-Gitter (Ringe z_i, gemeinsames Azimutgitter); die konvexe Hülle besteht **exakt** aus den
gleichschenkligen Trapezen zwischen benachbarten Ringen und den zwei Polkappen (`lat_long_facets`, beim Aufbau
gegen alle Punkte verifiziert, Fehler < 1e−13). Der Test u ∈ int conv ist damit exakt und O(Q) (übersprungen für
|u| < Inkugelradius). Unit-Test: 800/800 Übereinstimmungen mit dem tatsächlichen Newton-Verhalten.

Unlösbare Zustände (stark gerichtet, |J|/ρ nahe 1, vor allem Strahlen am Rand) werden **nicht** geclippt. Die Regel
wird um **einen Punkt Ω* = J/|J|** (mittleres Gewicht 4π/Q, Vorwert y* = 0) ergänzt („angepasste Winkelquadratur“
der Notiz): U = |J|(1, Ĵ) + (ρ−|J|)(1, 0) liegt dann für jedes ρ > |J| im Inneren des ergänzten Kegels; für
ρ = |J| ist das Atom selbst die einzige nichtnegative Darstellung. Das Ergebnis bleibt ein nichtnegatives,
momententreues diskretes Winkelmaß; das Atom zählt zur auslaufenden Hälfte, wenn Ω*·n > 0. Jeder Fall wird
gezählt (`augmented`, `diracs`). Scheitert auch das (nicht beobachtet), wird mit `--projection-strict` abgebrochen,
sonst für diesen Knoten der C6-Ghost verwendet und als `failures` gezählt.

### Grenzen (unverändert aus der Notiz)

Positivität nur an den Quadraturpunkten, nicht für Interpolationen dazwischen; höhere PN-Momente werden nicht
erhalten; keine globale Stabilitätsaussage für den FEM/MCL-Löser. Die Projektion ändert nur den M1-Ghost (Bar-States
des Low-Order-Randflusses); der PN-Randfluss (LLF mit Onsager-Datum) ist unverändert.

## 3. Optimierung des Randpfads (vorab, Methode unverändert)

Ziel: wesentlich schneller bei relativer Feldabweichung ≤ 1e−6 gegenüber `bdd5d3f`. Profil vorher (callgrind,
`add_boundary`, P15): 52 % dichte Trace-GEMMs, 29 % Einstromdaten (davon 13 % `harmonics()`), Rest Allokationen.

| Änderung | Datei |
|---|---|
| **Separable Trace-Auswertung**: Y_l(FΩ) = D_l Y_l(Ω) (Wigner-Block je l) und Y_lm = Θ_lm(z_i) T_m(φ_j) im lokalen Tensorgitter: Q(2N+1) statt Q·q je Knoten; D, Θ per Quadratur (konventionsunabhängig), beim Aufbau gegen die dichte Auswertung geprüft (< 1e−12) | `src/separable_trace.{hpp,cpp}` |
| Ghost-Stufe in Blöcken von ≤ 32 Knoten (Speicher O(Q·32) statt O(Q·Randknoten)), Blockgröße an die Threadzahl angepasst | `add_boundary` |
| Einstromdaten ohne Allokation (`face_data_into`), Atom-Harmonische je (Rand, Richtung) gecacht, exaktes Kegelprädikat einmal statt zweimal ausgewertet, `require(bool, const char*)` ohne `std::string` | `boundary.cpp`, `element_weno.cpp`, `common.hpp` |

Gleiche Operationen in gleicher Reihenfolge außer der Trace-Summation; gemessene maximale relative Feldabweichung
1e−12 … 3e−11 (alle Felder, normiert auf max ρ), unabhängig von der Threadzahl. Bestehende Tests (element_weno,
material, fixes, FCS, DLRA) bestehen unverändert.

### Zweite Runde: ganzer Zeitschritt und Thread-Skalierung (System unverändert)

Profil eines ganzen Schritts (callgrind mit Zeileninformation) und Thread-Skalierungsproben auf nx (E=256, Q1, N=7):

| Änderung | Wirkung (nx, 31 Threads, E=256, Q1, N=7, T=0.05) |
|---|---|
| `require/cone_require(bool, const char*)`: kein `std::string` je Aufruf (Limiter, Levermore) | Limiter 2.6× schneller (malloc-Konkurrenz bei 32 Threads) |
| PN-Advektion −Σ g_d A_d als Block-SpMM (8 Knoten, transponierter Puffer) statt skalarer CSR-Schleife | Massenphase −24 % bei P15 (ein Thread) |
| Knotengroße Puffer je `rate()` parallel genullt (First Touch) statt seriell | Closure 1.39 s → 0.17 s (seriell wuchs mit der Threadzahl) |
| Keine Kopie des Zustands am Schrittende (`cond ? react(..) : last` kopierte) | −0.75 s unerfasste serielle Zeit |
| Randstufe: Flächenzeilen statt tausender Kleinmatrizen, parallele Knotenreduktion (gleiche Summationsreihenfolge), Seitenprodukte in einem Parallelbereich | Rand 0.66 s → 0.25 s |
| Gesamt | 5.60 s (`bdd5d3f`) → 2.22 s, Skalierung 1→31 Threads 6× → 13.8× |

Neue Zeitmessungen in `summary.json`: `closure_setup`, `finite`, `ssp_stage`, `rate`, `advance`, `boundary_faces`,
`boundary_products`, `boundary_reduce`.

### Grenze des 1e−6-Kriteriums

Für einzelne Operatorauswertungen und kurze Läufe (T = 0.1) liegt jede Änderung bei ≤ 3e−11. Über tausende Schritte
verstärkt das nichtlineare Verfahren (Limiter-, Kegel- und WENO-Schalter) Rundungsunterschiede jedoch. Die
**unveränderte Referenz** `bdd5d3f` ist deshalb nicht einmal mit sich selbst auf 1e−6 reproduzierbar, wenn nur die
Threadzahl wechselt (32 vs. 16, T = 0.8, max. relative Feldabweichung):

| Fall | Referenz 32 vs. 16 Threads | Referenz vs. optimiert |
|---|---|---|
| E=256, Q2, N=7 | 5.7e−6 | 1.5e−6 |
| E=128, Q1, N=7 | 2.1e−10 | 1.2e−10 |
| E=128, Q1, N=15 | 6.5e−5 | 3.5e−4 |

Die Abweichungen der Optimierung liegen damit in derselben Größenordnung wie die Eigenvariabilität der Referenz;
bei N=15 sitzt das Maximum am Ausflussrand (RMS 2e−5).

## 4. Benutzung

```bash
cmake --build tmp/build --target m1pn_element_weno m1pn_positivity_projection_tests m1pn_positivity_projection_bench
OMP_NUM_THREADS=1 tmp/build/m1pn_positivity_projection_tests            # 150 Checks, Referenz: Dykstra
tmp/build/m1pn_element_weno --problem crossing-study --dimension 2 --elements 64 --degree 2 \
    --pn-order 7 --time .8 --positivity-projection [--projection-rule R] [--projection-strict] \
    [--projection-tolerance 1e-12] [--projection-cold-start] --output tmp/my_run
tmp/build/m1pn_positivity_projection_bench --orders 3,7,15 --states 2000   # Kernkosten je Randknoten
```

`summary.json` enthält zusätzlich `kinetic_l2_error` (gegen die exakte kinetische Dichte, enthält den
PN-Modellfehler), `kinetic_l2_error_outflow_strip` (nur max(x,y) > 1 − 4·Breite), `exact_mass`, `seconds.ghost`
und bei Projektion `projection{calls, identities, augmented, certificates, diracs, farkas, failures, iterations,
backtracks, max_iterations, max_residual, min_active_fraction}`. `--projection-rule R` setzt die Hemisphärenregel
(Standard 2N+1, Minimum N+1 für Momentenkonsistenz).

NX-Kampagne: `scripts/positivity/campaign.py prepare|submit|collect`, Auswertung `scripts/positivity/analyze.py`.
