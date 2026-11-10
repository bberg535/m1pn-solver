# Falsifikationsbericht: PN-Randbehandlung × WENO × Elementlimiting × PN–M1-Synchronisierung

Stand 10.11.2026, Code `experiment/element-weno` @ `8efe517` (Arbeitsbaum ohne Änderungen an getrackten Dateien;
nur untracked Logs/Ergebnisordner). Alle Zahlen stammen aus `audit/results/*.jsonl` und `audit/results/summary.json`;
Reproduktion: `bash audit/run_all.sh` (eine Kern, deterministisch, ca. 10–15 min). Keine Pushes, kein Merge.

## 0. Umfang, Methode, Vorbehalte

* **Neue Methode ist vorhanden.** `src/element_weno.{hpp,cpp}` (Binary `m1pn_element_weno`, in `m1pn_fast --stabilization element-weno`):
  nichtlineares CG-WENO (`γ_e` pro Element) + elementbasiertes Convex Limiting + knotenweises Randlimiting + SSPRK3.
  Das ist **nicht** der alte lineare Kuzmin–Vedral-Operator (das ist der Fall `γ_e ≡ 1`, `--no-weno`) und nicht die Local-KV-/Biharmonik-Pfade
  (`src/stabilization.cpp`, `transport.cpp`; hier nicht verwendet).
* **Instrumentierung isoliert.** `audit/make_audit_source.py` patcht beim Bauen eine **Kopie** von `element_weno.cpp`
  (Hooks: Low-Order-Kontrolle, Limiter getrennt abschaltbar, Elementfaktoren/θ/Skalen exportieren; Standardverhalten unverändert).
  `src/`, Referenzdaten und Benchmarks sind unberührt. Private Member werden im Testcode per `#define private public` gelesen.
  Die 75 vorhandenen Checks von `m1pn_element_weno_tests` laufen unverändert durch (`existing_element_weno_tests.txt`).
* **Build:** g++ 13.2.0 über OpenMPI-5.0.10-`mpic++`, `-O3 -DNDEBUG -std=gnu++17 -march=x86-64-v3 -ffp-contract=fast -fopenmp`, 1 Thread.
* **Kegel-Prüfung exakt** (`exact::cone` auf den gespeicherten Doubles, kein Epsilon). Es wird nichts geclippt; Reparaturen (`nudge`, ≤256 ulp auf ρ)
  werden **gezählt** – in allen berichteten Läufen: **0**.
* **Nicht vorhanden im Solver:** Streuung, Absorption, FCS-Quellen. Die Beiträge „Streuung“ und „FCS“ des Energiebudgets sind strukturell 0;
  FCS ist damit trivial „aus“. Kein Teil von `03_projects/M1PN_FCS_Performance` oder `research/` wurde berührt.
* **Zeitbudget:** ca. 35 min Rechen-/Schreibzeit. Zeitschritte stets `dt ≤ step_bound() = min(dt_bound, high_order_step)` (Crossing: 0.4993 × step_bound; T2: 0.906 ×).
  `high_order_step()` ist eine 60-Schritt-Potenziteration (keine rigorose Schranke).
* Literaturangaben unten stammen aus `docs/LITERATURE.md` bzw. dem Gedächtnis und wurden **in diesem Lauf nicht neu geprüft**.

### Zuordnung Formel → Code (`src/element_weno.cpp`, `src/boundary.cpp`)

| Formel | Code |
|---|---|
| Hochordnungsziel `M ȧ = −C·A a − s_h(a) + b^H` | `rate()` :552 ff., Residuum :601–626, `solve_mass` |
| `s_h^e(v,u)=ν_e∫∇v·(∇u−γ_e g_h)`, `g_h` = globale L²-Projektion | `stabilization_load` :423, `ν = h/(2p)` :79 |
| `γ_e = 1 − min(1,‖u−u*‖/‖u‖)^q`, WENO-Mischung (nur ψ⁽⁰⁾, ein γ für alle PN-Komponenten) | `weno_gamma` :384; Rand: fehlende Nachbarn werden ausgelassen (einseitig), Fallback `‖u‖=0 ⇒ γ=1` |
| Graphenviskosität `d_ij`, Barzustände `ū_i` (konvexe Zerlegung) | :173–188, :660–676 |
| Antidiffusive Beiträge `f_i^e` | :667–672 |
| Limiter Schritt 1 (ρ-Clip + Skalierung), Schritt 2 (Elementfaktor gegen Kegel, θ=0.999) | :685–712 |
| Randkorrektur `h_i = P b^H − b^L − (∮φ_i n)·G_i`, knotenweises θ_i | `add_boundary` :467, :737–764 |
| PN→M1-Synchronisierung: Zeilen l=0,1 der PN-Rate := limitierte M1-Rate / `lumped` | :765–770 (`set_low`) |
| CFL | `dt_bound` :270–282 (inkl. Randflächenmassen), `high_order_step` :819 |
| Stufen / Reparatur | `advance_ssprk3` :853, `nudge_into_cone` :47 |
| PN-Rand: Onsager-C1-Exterior `ũ = R₋ incoming + R₊ outgoing + Nullanteil`, Fluss `½(A_n−I)(u−ũ)` (LLF mit Einheitsdissipation) | `Boundary::exterior` boundary.cpp:174, Integrand element_weno.cpp:517 |

## A. Was im aktuellen Code wirklich erfüllt ist (gemessen)

1. **Spektrum von `A_n`** (n = e_x, e_y, (1,1,0)/√2, (1,1,1)/√3, rotiert): P3 (6 positiv, 6 negativ, 4 null), P5 (15,15,6), P7 (28,28,8) = (r, r, N+1) für alle Normalen.
   `A_n = H⁺ + H⁻` (kinetische Halbraumintegrale) bis 2e‑15.
2. **Randenergie, implementierter Fluss (LLF + Onsager-Exterior), d = 0:** `λ_max(sym Q) ≤ 2.4e‑16` für P3/5/7 und alle Normalen
   (`Q = ½(A−I)(I−E) − ½A`, `uᵀQu` = Energierate pro Flächenpunkt, Volumenanteil `−½uᵀAu` eingerechnet).
   Nullmoden tragen exakt nichts bei (`|ZᵀQZ| ≤ 5.8e‑17`, `Eᶻ = z` bis 3.6e‑15).
   **Diskret mit den tatsächlichen Flächenmassen** (konsistente Gauss-Flächenregeln, `audit_energy_random`, 200 Zufallsfelder je Fall, P3/5/7 × p=1,2,3):
   `(aᵀR_vol + aᵀB^H)/∮|u_h|² ∈ [−0.39, −0.24]` – strikt dissipativ.
3. **Stabilisierung:** `S(γ)·1 = 0` (8e‑17); Reproduktion von `x^a y^b` (a,b ≤ p) auf offenen Netzen: 4e‑16, auch mit dem tatsächlichen WENO-γ (`|γ−1| ≤ 8e‑9`);
   für `γ ≡ 1` ist `S` symmetrisch (1.2e‑16) und PSD (`uᵀSu = ν(‖∇u‖²−‖g_h‖²) ≥ 0`, elementar aus der L²-Projektion).
   **Der Symmetrieanteil von `S(γ)` ist für alle γ ∈ [0,1]^cells PSD** auf den getesteten Netzen (1D p=1,2,3 periodisch/offen, 2D p=2 3×3 periodisch/offen):
   `uᵀS(γ)u` ist affin in γ, also genügt die Prüfung aller 2^cells Ecken: min. Eigenwert ≥ −5.2e‑16 relativ. Die Hill-Climbing-Suche über echte γ(u) (15 000 Auswertungen je Fall)
   fand kein `uᵀS(γ(u))u < 0` (min. Rayleigh-Quotient 1.6e‑3 > 0).
   Im Crossing ist der Stabilisierungsterm in **jeder** gemessenen Stufe dissipativ (max. Rate −1.0e‑2·‖a‖²_M).
4. **Elementbilanzen:** Summe der Rohkorrekturen und der akzeptierten Korrekturen pro Element = 0 bis 1.5e‑15 (relativ zur größten Korrektur).
5. **Kegel:** Crossing 64² (p=1, P7, T=0.32, CFL 0.5, 325 Schritte), Varianten *full*, *no‑WENO*, *low‑order*: **0 Verletzungen** nach jeder Euler-Teilaktualisierung und jeder SSP-Stufe, 0 Reparaturen;
   kleinste Marge absolut 1.58e‑6, relativ 3.0e‑4. Ebenso T1, T3, T6, T7, T8 (alle ρ-Skalen 1…1e‑100, exaktes Vakuum, Halb-Vakuum).
6. **Globale M1-Massenbilanz:** akzeptierte Rate = unabhängig aus Randgrößen rekonstruierte Rate (`Σ_i[bl_i + θ_i h_i − (∮φ_i n)·F^L_i]₀`) bis 8.9e‑13 relativ (Rundung); T3: 1.6e‑17.
7. **Energiebudget schließt** algebraisch (`vol+stab+bnd+sync = aᵀM ȧ_tatsächlich`): Fehler ≤ 5.5e‑15 relativ.
8. **T1:** isotroper Zustand mit passenden Einströmdaten ist für das Gesamtverfahren stationär (1.8e‑14), `ũ = u` (9e‑17), θ ≡ 1.
9. **T6 (glatt, kompatible zeitabhängige Einströmdaten, PN3, p=2, offen, T=0.25, CFL 0.5, `h` und `dt` gemeinsam):**
   Dichte-RMS-Fehler Randschicht/Innen: E=4: 1.56e‑3/1.77e‑3; E=8: 1.66e‑4/1.77e‑4 (Ordnung 3.24/3.32); E=16: 3.72e‑5/2.02e‑5 (Ordnung **2.16 am Rand**, 3.13 innen).
   Der Limiter ist hier inaktiv (Varianten full = no‑WENO = no‑limit bis 1e‑3 relativ). PN5 nur E=4,8 (Ordnung 3.2/3.3; Lauf danach durch Timeout abgebrochen).
10. **WENO im Crossing:** Mischung ändert die Dichte nur um 7.9e‑4 (relative L²-Differenz full vs. no‑WENO); γ<0.99 in 0.19 % (Rand) / 0.10 % (innen) der Elementauswertungen.

## B. Übertragbare Literaturbeweise (unter expliziten Voraussetzungen)

| Eigenschaft | Quelle (nicht neu geprüft) | Voraussetzungen, die hier **gemessen** wurden |
|---|---|---|
| Kegelinvarianz einer Convex-Limiting-Stufe, lokale Erhaltung | Kuzmin–Hajduk 2023 (§3.2.7), Moujaes–Kuzmin–Bäumer arXiv:2509.07689 (M1), Guermond–Popov (Invariant Domain) | konvexe Zerlegung in `ū_i^e` und Randbarzustand, `dt ≤ dt_bound` inkl. Randflächenmassen, Elementsummen = 0 (A.4), exakte Kegelarithmetik. Beobachtet: 0 Verletzungen. Der Beweis überträgt sich **nur für die M1-Momentenzeilen**; PN-Zeilen l ≥ 2 sind nicht Teil der Aussage. |
| `L²`-Stabilität von PN mit Marshak/Onsager-Rand | klassisch (Marshak-PN-Galerkin) | Hier andere Realisierung (LLF + Onsager-Exterior). Face-Ebene numerisch bestätigt (A.2), **Beweis für diesen Fluss nicht aus der Literatur übernommen**. |
| CG-WENO `γ_e∈[0,1]`, Konsistenz, `S1=0` | Kuzmin–Hajduk–Vedral arXiv:2509.04673 | `S(1)` PSD ist elementar; für variables γ lag mir kein Beweis vor (siehe E, L3). |

## C. Gescheiterte Eigenschaften – minimale Gegenbeispiele

Aufruf: `audit/build/audit_runs small` (dort Fälle T1–T3, T8); Fall‑IDs = `label` in `results/small.jsonl`.

**C1. Schrittabbruch an Kegelrand mit Vakuumnachbarn (T2, `corner_cone_boundary_J`).** P3, p=1, 8×8, offen, kein Einstrom; Knoten 0 (Ecke) ρ=1, J=(−1,0,0) (nur ρ um ulps angehoben, damit der Anfangszustand zulässig ist), alle anderen Knoten exaktes Vakuum.
Ergebnis: `StepRejected: Stable M1 base bar outside stored binary64 cone` im ersten Schritt, sowohl *full* als auch Low-Order-Kontrolle. Kein Clipping, aber auch keine Fortsetzung: der Solver deckt diesen Randfall des Kegels nicht ab (ähnlich der README-Notiz zu abbrechenden Vakuum-Stressläufen im Massenkorrektur-Pfad; anderer Code, hier nicht verglichen).

**C2. Massenzuwachs aus Vakuum-Einstrom (T2, `corner_inflow_J_full`).** P3, p=1, 8×8, kein Einstrom, ρ=1 an der Ecke, J = 0.75·(1,1,0)/√2.
Akzeptierte globale Massenrate max **+7.15e‑3** (Anfangsmasse 3.9e‑3), PN‑Target max −6.33e‑3, Low-Order max −5.9e‑3.
Knotenweise akzeptierte Dichte‑Randrate max +7.15e‑3 (`corner_inflow`), +4.5e‑3 (`edge_node`), +1.9e‑7 (`corner_outflow`); die Low-Order-Kontrolle ist überall ≤ 0.
Mechanismus aus dem Code: das Randlimiting (:749–763) und der Elementlimiter beschränken nur Kegelverletzungen; **positive** ρ-Korrekturen `θ_i h_i[0] > 0` sind immer kegelzulässig und werden nie gedeckelt. Das PN‑Target selbst verliert Masse, die limitierte Summe gewinnt welche (Differenz der jeweiligen Maxima akzeptiert − Target ≈ +1.3e‑2).
Die Bilanz mit den *akzeptierten* Flüssen schließt dennoch exakt (A.6): Erhaltung ja, Vorzeichen-/Monotonie-Eigenschaft nein.

**C3. Synchronisierung erzeugt positive Energie (T2, `corner_outflow_J_full`).** Gleicher Aufbau, J = 1.5·(−1,−1,0)/√2/2.
`⟨a, M_C E(L−Ph)⟩/‖a‖²_M` erreicht **+18.0** (max. Rate), kumulativ `+1.27e‑4` bei Gesamtintegral `−2.36e‑4`. Quelle: Elementlimiter (+17.8) statt Randlimiter (+0.24). `corner_inflow_J_full`: +3.19 (Element +1.6, Rand +1.6), kumulativ +1.3e‑5.
Die Gesamtenergie sinkt in diesen Läufen trotzdem (Stabilisierung/Rand überkompensieren) – das ist **kein** Beweis.
Low-Order-Kontrolle auf Crossing 64²: Sync max **+0.082**, Gesamtrate max **+0.060 > 0** (PN‑Energie wächst momentan). Im vollständigen Crossing‑Lauf bleibt der Sync-Term dagegen ≤ 1e‑15 (kein Zuwachs jenseits Rundung), Integral −7.7e‑6.

**C4. Variables γ macht den effektiven Operator unsymmetrisch.** `‖S−Sᵀ‖/‖S₀‖` = 6–13 % (zufälliges γ), bis 23 % (Eckenmuster). Nur der Symmetrieanteil ist PSD (A.3); eine Argumentation über einen symmetrischen, positiven `s_h` greift nicht.

**C5. WENO-Sensor nicht skaleninvariant (T8).** `ε = 1e‑6` ist absolut: `γ_min = 0.854` bei ρ~1, `0.957` bei 1e‑2, `0.993` ab ≤1e‑4 (Sensor praktisch aus). Nach 3 Schritten weicht `S(sρ)/s` für s ≤ 1e‑3 um 2.7e‑3 relativ von s=1 ab. Keine Kegelverletzung, keine Reparatur; aber Einheitenabhängigkeit.

**C6. Low-Order-Randfluss ist nicht konsistent (T1).** Low-Order-Kontrolle des isotropen Gleichgewichts: max. Rate **0.846** bei ρ=1 (Ghost = Halbraummomente ρ/2). Konsistenz stammt allein von der antidiffusiven Randkorrektur; wo θ<1 (Crossing min θ=0.013; T7 0.025; T2 0), weicht die akzeptierte Rate vom PN‑Target ab:
Synchronisierungsdefekt (M‑Norm, relativ) **9.0 %** (Crossing), 22 % (Low-Order), 49–54 % (T7), 32–65 % (T2); Massenratenabweichung Target↔akzeptiert max 5.1e‑4 bei max. |akzeptierter Rate| 1.75e‑4.

**C7. Onsager-Rand reproduziert nicht jede PN-darstellbare stationäre Kinetik (T3/Probe).** P3, ψ=(1−Ω_x)³ (exakt in P3, ρ=25.1, |J|/ρ=0.6), Einstrom g=ψ auf der einlaufenden Halbkugel:
für n=±e_x `ũ=u` (1e‑14), für n=±e_y und (1,1,0)/√2 `‖ũ−u‖/‖u‖ = 7.7e‑3`. P5/P7: exakt. Folge: Gesamtverfahren nicht stationär (max. Rate 1.54; PN‑Target 2.16), obwohl θ ≡ 1 – der Fehler liegt im PN‑Rand, nicht im Limiter. Möglicher Zusammenhang mit der Randordnung 2.16 in T6 (E=16) – **nicht isoliert**.

**C8. Charakteristischer C1-Fluss `Boundary::flux` ist im Euklid‑Energiemaß nicht dissipativ.** `λ_max(−sym H) = +1.7e‑3 (P3), 3.7e‑3 (P5), 5.3e‑3 (P7)`. Er wird vom Element‑WENO‑Pfad **nicht** benutzt (dort LLF, A.2) und weicht von diesem um 43–47 % (Frobenius) ab. Ein Onsager-gewichtetes Maß wurde nicht getestet.

**C9. Kinetisch vs. spektral (T4/T5).** `‖H⁺−A⁺‖/‖A‖ = 16.5 % (P3), 11.8 % (P5), 9.6 % (P7)`. Für Nullmoden `z` (`A_n z = 0`) ist `H⁺z ≠ 0`: `zᵀH⁺z ∈ [0.124, 0.137]` (P3), `[0.094, 0.113]` (P5), `[0.077, 0.098]` (P7); die Massenzeile von `H⁺z` ist 0, höhere Momente nicht. Der implementierte Rand reicht Nullmoden unverändert durch (Energie 0), die kinetische Halbraumbilanz sähe positive einseitige Energie. Nullmoden sind winkelnegativ (min ψ_z = −0.67 … −1.19), also keine zulässigen Kinetikdichten. Streifwinkel (μ=0, ±0.01): 56–63 % der Strahlenergie liegen in Nullmoden; Randenergierate des implementierten Flusses −0.11…−0.13·‖u‖².

**C10. Diagnosevariante ohne Limiter.** Crossing 64² *no_limit*: nach 113 Schritten Kegelverletzung (Marge −3.2 %), Abbruch >256 ulp. Der Limiter ist notwendig, nicht dekorativ (Element‑Faktor min 1.1e‑4, in 32 % der inneren, 14 % der Randelemente aktiv; Rand‑θ min 0.013 in 8.9 % der Randknoten).

## D. Nur ungetestet / unbewiesen

* Allgemeiner Beweis, dass `S(γ)+S(γ)ᵀ ⪰ 0` für alle Netze/Grade (nur Eckenaufzählung auf 7 kleinen Netzen).
* Energiebeschränkung der Synchronisierung; Entropie: PN‑Quadratenergie und M1‑Entropie werden getrennt behandelt, **keine** gemeinsame Entropieabschätzung behauptet.
* 3D, p ≥ 3 im Crossing, Crossing mit p=2 bei 64² (zu teuer für das Budget), gedrehte Flächen innerhalb von `ElementWeno` (nur achsparallel), Onsager‑gewichtete Energie, Stoßterme.
* T4 als Zeitintegration (nur Flächenebene), T5 nur als Flächenoperator, P5‑Konvergenz E=16 (Timeout), `dt_bound` bei CFL 1 separat vom Hochordnungsschritt.
* Energiezerlegung wird nur an den ersten Stufen von ca. 8 (Crossing) bzw. 20 (T2) Schritten gemessen, nicht an jeder Stufe.
* Ursache der Randordnung 2.16 (T6) nicht isoliert; T7 (Gaußbreite 0.03 < h) ist unterauflösend – der Fehler gegen E=48 (≈0.5) belegt nur Unterauflösung; Vergleich: Randfall zeigt Randlimiter θ_min=0.025 gegenüber 1 im inneren Fall, γ_min=0 (Innenelemente) in beiden Fällen.

## E. Fehlende Lemmata für einen allgemeinen M1PN-Randstabilitätsbeweis

* **L1 (Vorzeichen-/Vakuumlemma des akzeptierten Randflusses).** Für Randlimiter mit Kegelschranke allein gilt keine Massenmonotonie bei Vakuum‑Einstrom (C2). Gesucht: eine zusätzliche, Konvexität erhaltende Schranke an `θ_i h_i[0]` (z. B. `B_i ≤ max(0, kinetischer Einstrom)`) samt Beweis, dass `{Kegel} ∩ {Massenmonotonie}` unter Forward Euler/SSPRK3 invariant bleibt.
* **L2 (Energielemma der Synchronisierung).** Abschätzung oder Konstruktion von `Σ_c a_cᵀ M (L−Pḣ)_c ≤ 0` (oder ≤ Anteil der Stabilisierungs‑/Flächendissipation); Gegenbeispiele C3 zeigen, dass der aktuelle Ersatz der Momentenzeilen das nicht garantiert.
* **L3 (Stabilisator-Lemma).** Für die L²-Projektion `g_h` auf stetige `Q_p`: `Σ_e [(∇u, g_h)_e]₊ ≤ ‖∇u‖²` (äquivalent: `S(γ)+Sᵀ ⪰ 0 ∀ γ∈[0,1]^cells`). Numerisch an allen Ecken bestätigt, Beweis offen.
  (Optional L4: Charakterisierung der Polynomklasse, für die der Onsager‑Exterior exakt ist; C7.)

## F. Empfehlung für Claude Fable 5.2

**L3 zuerst** (abgeschlossenes, präzises, mit `audit_ops` (Eckenaufzählung) und `dense_S` sofort falsifizierbar, ein Beweis oder ein Gegenbeispiel in bis zu ~1 Sitzung),
danach **L2**: die eigentliche Forschungslücke, weil sie das einzige gemessene Energie-Defizit betrifft (C3). Eine Modifikation der Synchronisierung (energie-bewusste Projektion der Momentenzeilen) wäre zu entwerfen und mit der vorhandenen Messroutine `measure()` (`audit/audit_runs.cpp`) gegenzuprüfen.

## Dateien

`audit/make_audit_source.py` (Patch der Kopie), `audit/audit_common.hpp`, `audit/audit_ops.cpp` (A, B, T4, T5, T8‑Sensor), `audit/audit_runs.cpp` (T1–T3, T6–T8, Crossing, Energie/Massen/Kegel),
`audit/audit_energy_random.cpp`, `audit/audit_probe*.cpp` (Rand‑Kompatibilität), `audit/build.sh`, `audit/run_all.sh`, `audit/make_summary.py`, `audit/show.py`,
`audit/results/*.jsonl`, `audit/results/summary.json` (maschinenlesbar, mit Commit/Compiler).
