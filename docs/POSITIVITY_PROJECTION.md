# Momententreue Positivitätsprojektion am Rand (Element-WENO)

Branch `feature/boundary-positivity-projection` (Basis `feature/element-weno-dlra` @ `bdd5d3f`).
Grundlage: Notiz *Momententreue Positivitätsprojektion – Nichtnegative Winkelrekonstruktion für den M1–PN-Randzustand*
(13.11.2026). Ergebnisse der NX-Kampagne: [results/positivity/](../results/positivity/README.md).

## 1. Was ersetzt wird

Der C6-Ghost (`fullsphere_ghost`, [BOUNDARY_FIXES.md](BOUNDARY_FIXES.md)) setzt am Randknoten i

    ext = m_in(g) + λ · m_out(max(ψ_N,i, 0)),     λ = min(1, b/a_ext)

mit den auslaufenden Halbraummomenten des **punktweise abgeschnittenen** PN-Traces. Das Abschneiden ändert die
Momente, und λ < 1 skaliert sie nachträglich. Neu (Option `positivity_projection`, **seit 14.11.2026 Standard**;
C6 mit `--c6-ghost` bzw. `positivity_projection = false`, `--legacy-boundary` schaltet sie mit ab):
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

## 2b. Varianten für U_aus (untersucht 13.11.2026)

Verglichen an den **exakten kinetischen Halbraummomenten** der Randknoten (Option `--ghost-study K`, Funktion
`ElementWeno::ghost_estimates`; [Tabelle](../results/positivity/ghost_study_20261113/README.md)):

| Variante (Option) | Idee | Ergebnis |
|---|---|---|
| C6 (`fullsphere_ghost`) | max(0, ψ_N) auslaufend, λ-Skalierung | Referenz |
| **volle Projektion** (Notiz, Standard) | W-Projektion auf der ganzen Kugel, volle Momente U | **4–12× genauer als C6** in a = ρ − n·J (glatt N=7: 2.2 % vs 8.4 %; Crossing N=7: 0.035 vs 0.44) |
| gewichtet (`--projection-incoming-weight c`) | Gewicht c auf der einlaufenden Hälfte; c → 0 lässt sie die Korrektur aufnehmen | praktisch wie volle Projektion: der Rest U − U_aus ist auf der einlaufenden Hälfte meist nicht nichtnegativ darstellbar |
| Halbraum (`--projection-mode half`) | nur auslaufend, erhält die Halbraummomente des PN-Traces | glatt wie volle Projektion, Crossing kaum besser als C6 (Halbraummomente der Strahlen oft nicht darstellbar) |
| Hybrid (`--projection-mode hybrid`) | C6, Projektion nur statt der λ-Skalierung | nahe C6 |
| Filter (`--ghost-filter α`) | exp(−α(l/(N+1))⁴) auf l ≥ 2 vor dem Trace | mit C6: Crossing besser, glatt schlechter; mit Projektion wirkungslos |

Der globale Lösungsfehler reagiert kaum auf die Güte des Ghosts (er wird vom PN-Fehler im Inneren dominiert); die
leichte Verschlechterung im glatten Ausflusstest (+0.5…1.2 %) geht daher nicht auf einen schlechteren Randzustand
zurück.

**Warum der globale Fehler trotzdem leicht steigt (Befund 14.11.2026).** Die Randlast ist
`A(bar − u) + θ h` mit `h = P b^H − b^L`; der Ghost geht nur in die Tiefordnungsteile (bar, b^L) ein, und für θ = 1
heben sie sich heraus: die Randlast ist dann die PN-Hochordnungslast P b^H **aus dem unprojizierten Trace**, unabhängig
vom Ghost. Der Ghost wirkt also nur über den Limiter (Kegelschranke, C2-Vorzeichenlimiter). Für einen auslaufend
positiven Trace ist C6 gleich den Halbraummomenten des Traces, also konsistent mit b^H (h klein). Die Projektion
verschiebt über ξ·m auch die auslaufenden Werte, sobald die einlaufende Hälfte negative Keulen hat; dann ist h größer
und es wird öfter limitiert (glatter Ausfluss, E=32, N=7, Q1: 711 statt 452 limitierte Randknoten), was am
Ausflussrand diffundiert. Beim Crossing wird ohnehin limitiert, dort hilft der bessere Tiefordnungszustand; bei N=15
ist der Trace positiv (Identität, kein Unterschied). Konsequenz: den Gewinn hebt erst ein Hochordnungsrandfluss, der
für die auslaufende Hälfte ebenfalls z* statt ψ_N verwendet.

**Getestet (14.11.2026, Option `projection_high_order`, CLI `--projection-high-order`, Standard aus).** Der
PN-Randfluss ist kein Halbraumintegral, sondern der charakteristische LLF/C1-Fluss ½(A_n − I)(u − ext(u, d)) mit
ext = exterior_map·u + Datenanteil. Konsistente Übertragung: im Außenzustand ũ = u + δ statt u, mit
δ = Σ_k w_k Y(Ω_k)(z*_k − y_k) (+ Atom), d. h. die L2-Projektion der Korrektur (δ = 0 bei Identität, M1-Momente von δ
null; Σ w Y Yᵀ = I auf 1e−15 geprüft). Zusatzterm −(llf_homogeneous − ½(A_n + I)) δ, nur an projizierten Knoten.
Ergebnis (E=32, Q1, kin. L2-Fehler; C6 / Projektion / Projektion+HO):

| Problem | N=3 | N=7 |
|---|---|---|
| glatter Ausfluss T=0.6 | 1.5065e−2 / 1.4997e−2 / 1.5338e−2 | 2.2593e−3 / 2.2720e−3 / 2.3944e−3 |
| Crossing T=0.8 | 6.5006e−2 / 6.4986e−2 / 6.5769e−2 | 2.3610e−2 / 2.2360e−2 / 2.5779e−2 |

Die Variante ist durchweg schlechter, obwohl sie beim Crossing die Limitierung senkt (40529 statt 46033 limitierte
Randknoten). Deutung: der charakteristische Fluss soll die auslaufenden Charakteristiken des **PN-Zustands** selbst
transportieren; ersetzt man sie durch die der positiven Rekonstruktion, ist der PN-Ausfluss nicht mehr konsistent
mit der PN-Lösung im Inneren (O(δ)-Störung). Verworfen; der Ghost bleibt ein reiner Tiefordnungsbaustein.

**Regel:** N+1 je Halbkugel liefert dieselben Ghost-Momente und Lösungen wie 2N+1 und 4N+3 (3–4 Stellen) bei
2.5–3× geringeren Kosten und ist seitdem Standard (`--projection-rule` überschreibt); Aufpreis im Gesamtsolver damit
+0.5…+5.5 % (nx, E=64–256).

### Robustheit (14.11.2026, beim Umschalten des Standards)

- Warmstart: Stockt Newton vom Warmstart aus (Liniensuche, max_iterations), wird kalt neu gestartet, bevor die Regel
  ergänzt wird. Vorher entschied der Verlauf eines Knotens in seltenen Fällen über die Ergänzung (DLRA-Test: 8e−6
  zwischen zwei Pfaden mit gemeinsamem Warmstart-Speicher, jetzt 2e−14). Der Warmstart beeinflusst damit nur noch
  die Iterationszahl.
- Auslaufende Masse ≤ ρ: die exakte Projektion erfüllt sie (z ≥ 0, Gesamtmasse ρ); die Newton-Toleranz ist relativ
  zu Σw|y| + ρ, nahe am Vakuum kann der Trace ρ um Größenordnungen übersteigen. Überschreitungen werden gekappt und
  als `capped` gezählt (glatter Ausfluss E=32: 154 von 67320, Relativgröße ~1e−12).
- Rand-Bar-Zustand ½(u + F_n(u)) + ½(ext − F_n(ext)): für einen einlaufenden Strahl u = ρ(1, −n) ist der erste Teil
  exakt 0 und entsteht durch Auslöschung (O(eps ρ_u)), weit mehr als ein ulp des winzigen Ergebnisses. Der unechte
  auslaufende C6-Ghost hat das verdeckt; mit dem exakten Null-Ghost der Projektion schlug der C1-Test fehl. Jetzt
  Rundung nach außen relativ zur Größe der Summanden (≤ 64 eps, Option `outward_bar_rounding`).

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
    --pn-order 7 --time .8 [--c6-ghost] [--projection-rule R] [--projection-strict] \
    [--projection-tolerance 1e-12] [--projection-cold-start] --output tmp/my_run
tmp/build/m1pn_positivity_projection_bench --orders 3,7,15 --states 2000   # Kernkosten je Randknoten
```

`summary.json` enthält zusätzlich `kinetic_l2_error` (gegen die exakte kinetische Dichte, enthält den
PN-Modellfehler), `kinetic_l2_error_outflow_strip` (nur max(x,y) > 1 − 4·Breite), `exact_mass`, `seconds.ghost`
und bei Projektion `projection{calls, identities, augmented, certificates, diracs, farkas, failures, iterations,
backtracks, max_iterations, max_residual, min_active_fraction, capped}`. `--projection-rule R` setzt die Hemisphärenregel
(Standard N+1 für die Projektion, C6 behält 2N+1; Minimum N+1 für Momentenkonsistenz).

NX-Kampagne: `scripts/positivity/campaign.py prepare|submit|collect`, Auswertung `scripts/positivity/analyze.py`.
