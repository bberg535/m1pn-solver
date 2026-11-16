# Mathematischer Konsistenz-Audit des aktuellen Solvers (16.11.2026)

Gegenstand: `M1PN_Solver`, Tag `standard-20261116-ewonly` plus Branch `feature/block3-numerics` (Default bitgleich
zum Tag, geprüft). Element-WENO auf Tensor-Bernstein-Elementen, elementbasiertes Limiting der M1-Momente,
SSPRK(4,3), Strang-Reaktion, First-Collision-Source (FCS), momententreue Positivitätsprojektion am Rand.

Grundlage: der ausführbare Code (Datei:Funktion angegeben), die Projektdokumente in `docs/` und die heutigen Messungen.
Frühere Chats waren nicht zugänglich. Gleichungsnummern der Literatur stehen nur dort, wo sie in den Projektdokumenten
bereits geprüft vermerkt sind (v. a. docs/ELEMENT_HETEROGENEOUS.md §1, docs/ELEMENT_WENO.md, docs/FCS.md); sonst steht
„Nummer nicht geprüft“.

Nicht Gegenstand, weil im aktuellen Code nicht vorhanden: M2, Biharmonik, lokales KV, kantenbasiertes MCL,
Ray-Trace-/Source-adaptive-FCS-Backends, Energieabhängigkeit, stationärer Löser, reflektierende Ränder, anisotrope
Streuung. Sie erscheinen nur als „nicht implementiert“.

---

## A. Übersicht (nach Relevanz geordnet)

| # | Thema | Mathematische Bedeutung | Hergeleitet? | Im Code | Relevanz | Handlungsbedarf |
|---|---|---|---|---|---|---|
| 1 | Sensor-Stabilisierungs-Rückkopplung (q = 1) | Komposition zweier konsistenter Bausteine wird auf feinen Gittern instabil | heute diagnostiziert (docs/ELEMENT_WENO.md) | Standard, bewusst beibehalten | hoch (Paper: Aussage zur Ordnung) | im Paper erwähnen; Linearisierung (B.1) aufschreiben |
| 2 | Keine Energie-/L²-Stabilität der PN-Zeilen mit elementweisem γ_e | s_h ist nur für konstantes γ positiv semidefinit | nein | — | hoch (Beweislücke) | B.1 als offene Frage formulieren |
| 3 | Invariante Menge = M1-Kegel von (ρ, J), nicht PN-Positivität | IDP-Aussage gilt nur für die limitierten Momente | teilweise (Paper Moujaes) | ja | hoch (Begriffsklarheit) | B.4 |
| 4 | FCS-Zerlegung: Gültigkeitsannahmen | konvexes Gebiet, Vakuum, Quellen in D̄; B + C erfüllt die volle Aufgabe | teilweise (docs/FCS.md) | ja, per `require` erzwungen | hoch | B.2 |
| 5 | FCS-Bilanz nur bis auf Quadraturfehler | Verlust von B analytisch, Gewinn von C per Gauß-Quadratur | gemessen (2. Ordnung) | ja | mittel | B.3 |
| 6 | Gesamtordnung in der Zeit = 2 | Strang mit nichtkommutierender Reaktion/Quelle | heute gemessen | ja | mittel | B.5 |
| 7 | Kegelerhalt der exakten Reaktion | Dämpfung l ≥ 1 mindestens so stark wie ρ | im Code-Kommentar | ja | mittel | B.6 sauber aufschreiben |
| 8 | Optische Dicke pro Element (3D-Lattice) | σh ≈ 0.6 → präasymptotisch, Schema nicht AP | nein | — | mittel (Deutung der 3D-Konvergenz) | Hypothese prüfen (D.2) |
| 9 | Dirac-Einstrom (Atome) | Datum über ungerade Halbraumprojektion, M1-Außenzustand auf dem Kegelrand | nein (nur Code) | ja | mittel | B.7 |
| 10 | Gleitkomma und invariante Menge | IDP-Beweis gilt exakt; Code braucht exakte Prädikate, gerichtete Rundung, gezählte Nudges | dokumentiert (Code) | ja | mittel | Erklärung ergänzen |
| 11 | FCS on/off diskretisieren dieselbe Streuung verschieden | Punktquadratur von σ_sρ_B gegen nodale Mittelung σ_s,iρ_C,i | nein | ja | niedrig–mittel | B.3 Bemerkung |
| 12 | Randoperator: charakteristische Aufteilung, Kernmoden | Kern von A(n) wird durchgereicht; C1-Kontraktion < 1 | im Code (Konstruktor-Prüfungen) | ja | mittel | kurze Herleitung |
| 13 | Symmetrie-/Winkelreduktion | nur gültig bei geraden Daten; Laufzeitprüfung | dokumentiert | ja | niedrig | — |
| 14 | MPI/Threads | Felder bitgleich, dt rangunabhängig | geprüft | ja | niedrig | — |
| 15 | Zeitschrittschranke der Hochordnung | Potenziteration ohne Randterme; Heuristik | dokumentiert | ja | niedrig–mittel | als Heuristik benennen |
| 16 | Literaturreferenz Lattice | Tabelle nur 4 Stellen; eigene MC-Referenz 1e-5 | heute | ja (`m1pn_lattice_mc`) | erledigt | — |
| — | anisotrope Streuung, gerichtete Volumenquellen, reflektierende Ränder, Energie, M2 | — | — | nicht implementiert | Zukunft | nur bei Bedarf |

---

## B. Herleitungen der wichtigen Zusammenhänge

Notation: ψ(x, Ω, t), ρ = ∫_{S²} ψ dΩ, J = ∫ Ω ψ dΩ; PN-Koeffizienten a = (a_lm); Bernstein-Basis φ_i ≥ 0,
∑_i φ_i = 1; gelumpte Masse m_i = ∫ φ_i dx; σ_t = σ_a + σ_s.

### B.1 Warum WENO-Sensor und Stabilisierung zusammen instabil werden können (Befund 1, Lücke 2)

Stabilisierung nach Kuzmin/Hajduk/Vedral (arXiv:2509.04673v2, Gl. (6)/(7); docs/ELEMENT_WENO.md):

    s_h(v, u) = ∑_e ν_e ∫_e ∇v · (∇u − γ_e g_h) dx,   ν_e = h/(2p),   g_h = L²-Projektion von ∇u_h (global),
    γ_e = 1 − r_e^q,   r_e = min(1, ‖u − u*‖_e / ‖u‖_e).

(a) **Konstantes γ:** Da g_h die globale L²-Projektion ist, gilt (∇u, g_h) = ‖g_h‖². Daraus folgt

    s_h(u, u) = ν (‖∇u‖² − γ ‖g_h‖²) = ν [(1−γ)‖∇u‖² + γ ‖∇u − g_h‖²] ≥ 0.

Die Stabilisierung ist dissipativ.

(b) **Elementweise γ_e, ν_e:** Die Orthogonalität (∇u − g_h, g_h) = 0 gilt nur global, nicht je Element. Der Term
∑_e ν_e γ_e (∇u − g_h, g_h)_e hat kein festes Vorzeichen; s_h ist nicht mehr positiv semidefinit. Ein Energiebeweis
für die PN-Zeilen fehlt daher. Die Option `symmetric_residual` (src/element_weno.hpp) ist genau die symmetrische
Alternative; sie ist nicht Standard.

(c) **Nichtglattheit des Sensors:** r_e ist eine Norm geteilt durch eine Norm. Für q = 1 ist γ in r_e = 0 nur
Lipschitz, nicht differenzierbar. Für eine Störung δ gilt |δγ_e| ≈ ‖δ − δ*‖_e / ‖ū‖_e. Die Stabilisierung koppelt das
mit dem Faktor ν_e ∫∇v·g_h(ū) zurück:

    |δ(rate)| ≲ ν_e |∇ū| · ‖δ‖ / ‖ū‖_e.

Am glatten Extremum ist die skalierte Halbnorm ‖ū‖_e ~ h²|ū''|, also wächst die Verstärkung wie ν_e |∇ū| / h²|ū''|.
Bei festem Problem steigt sie mit feinerem Gitter. Ab einer Auflösung übersteigt sie die Dämpfung, und
Rundungsfehler wachsen exponentiell.

Für q ≥ 2 ist ∂γ/∂r = 0 in r = 0, die lineare Rückkopplung verschwindet.

*Belegt durch Messung* (`tests/weno_roundoff_study.cpp`, docs/ELEMENT_WENO.md):
- Mit γ ≡ 1 gibt es kein Wachstum, ohne Limiter unverändertes Wachstum. Der Effekt hängt nicht von dt ab, ist also
  semidiskret.
- q = 3 ist in 20 Fällen identisch zu γ ≡ 1.
- Q4/256 4.7e-8, Q4/512 2.3e-7 statt 1e-13; Q1 EOC 1.83, Q3 3.33 statt 2.00 bzw. 3.91.

*Nicht bewiesen:* das genaue Spektrum der linearisierten Rückkopplung. Die Skalierungsaussage ist eine
Größenordnungsabschätzung.

### B.2 FCS-Zerlegung und ihre Annahmen (Thema 4)

Kontinuierlich, mit Primärquelle Q, Anfangswert ψ⁰ und Einstrom ψ_in auf Γ₋:

    B:  ∂_t B + Ω·∇B + σ_t B = Q_B,                 B|_{Γ₋} = ψ_in^B,  B(0) = ψ⁰_B
    C:  ∂_t C + Ω·∇C + σ_t C = σ_s/(4π) (ρ_B + ρ_C) + Q_C,   C|_{Γ₋} = ψ_in^C,  C(0) = ψ⁰_C

(Kusch & Stammer, arXiv:2111.07160, Gl. (7); Hauck & McClarren, MMS 11 (2013); docs/FCS.md.)

Addieren ergibt die volle Gleichung, sofern die Aufteilungen sich zu den Daten summieren:
Q_B + Q_C = Q, ψ_in^B + ψ_in^C = ψ_in, ψ⁰_B + ψ⁰_C = ψ⁰. Der Standardfall „B|Γ₋ = ψ_in, C|Γ₋ = 0“ ist also eine
Wahl, keine Notwendigkeit. Er ist nur dann sinnvoll, wenn B den Einstrom analytisch enthält. Sonst muss der Einstrom
in C, und es darf ihn nur einer der beiden Teile tragen; sonst wird er doppelt gezählt.

**Im Code** (src/fast_main.cpp, FCS-Setup):

| Fall | B trägt | C trägt | Annahme | Prüfung |
|---|---|---|---|---|
| Line/Plane Source | Puls ψ⁰ im homogenen unendlichen Medium | Boden ψ⁰ (isotrop), Vakuum-Einstrom | homogen, Vakuum | `require(!material && inflow == vacuum)`; Front + 6 Breiten ≤ halbe Box |
| Lattice 2D/3D | ganze Volumenquelle q0 auf der Quellzelle | Boden 1e-10, Vakuum | Zellgeometrie, „nichts tritt wieder ein“ | Fall erzwingt Vakuum; Quelle aus `material->source` entfernt |

Warum die Einschränkung auf das Gebiet exakt ist: D ist konvex (Quader). Ein Strahl, der D verlässt, kehrt nie
zurück. Liegen alle Quellen und Anfangsdaten von B in D̄, dann stimmt das unendlich ausgedehnte B, eingeschränkt auf
D, mit dem B des Gebiets mit Vakuum-Einstrom überein. Die Zeitbedingung der Line Source wird für die Zerlegung
selbst nicht gebraucht. Sie sichert zweierlei:
- Die analytische Buchführung der ungestoßenen Masse (`pulse.mass · e^{−σ_t t}` in summary.json) setzt voraus, dass
  nichts von B das Gebiet verlassen hat.
- Die Vergleichbarkeit mit der Referenz im unendlichen Medium bleibt erhalten.

*Konsequenz:* FCS mit nichtvakuum Einstrom (Crossing, Strahlen) ist bewusst nicht unterstützt. Eine Erweiterung
bräuchte B|Γ₋ = ψ_in als Ray-Trace mit Beer–Lambert-Dämpfung (wie der Shadow-Test Stufe A) und C|Γ₋ = 0.

### B.3 Diskrete FCS-Bilanz (Themen 5, 11)

Implementierung (src/element_weno.cpp, `collided_source`; Quadratur im Konstruktor):

    q_i = ( ∑_e ∑_k w_k φ_i(x_k) σ_s(x_k) ρ_B(x_k, t) ) / μ_i,     μ_i = ∑_e ∑_k w_k φ_i(x_k).

Die Gauß-Regel hat p+1 Punkte je Richtung (`uncollided_points = degree + 1`). Daraus folgt:
- μ_i = m_i exakt, bis auf Rundung.
- Positivität: w_k > 0, φ_i ≥ 0 (Bernstein), σ_s ≥ 0, ρ_B ≥ 0 (geprüft, sonst Abbruch) ⇒ q_i ≥ 0. Die Quelle
  erhält den M1-Kegel (B.6).
- Gesamttransfer: ∑_i m_i q_i = Q_h[σ_s ρ_B], also die Quadratur des exakten Integrals (Zerlegung der Eins).

Bilanz:
- **C:** Das Ledger bucht m_i q_i τ als eingespeist, die Absorption analytisch. Bilanz bis auf Rundung (≤ 5e-12
  gemessen).
- **B:** Verliert analytisch ∫σ_s ρ_B dx dt.
- **Gesamt (B + C):** Fehler = ∫ (Q_h − ∫)[σ_s ρ_B] dt. Das ist ein reiner Quadraturfehler, gemessen zweiter Ordnung
  (docs/FCS_LATTICE.md). Er ist nicht null, weil ρ_B an der Front und an Materialgrenzen nicht glatt ist. Die
  Materialgrenzen liegen auf Elementgrenzen (n durch 7 teilbar), die Front nicht.

Zur Frage q_i ≠ σ_{s,i} ρ_{B,i}: Der Code verwendet keine nodale Größe ρ_{B,i}, sondern die Punktquadratur des
Produkts. Für C dagegen nutzt die Reaktion nodal gemittelte Koeffizienten σ_{s,i} = ∫σ_s φ_i / ∫φ_i angewandt auf
a_i (src/element_weno.cpp, `react`). Die gleiche physikalische Streuung wird also für B und C verschieden
diskretisiert. Beide sind konsistent; bei endlichem h sind FCS-on und FCS-off aber nicht dieselbe Diskretisierung
desselben Problems. Unterschiede zwischen beiden sind daher kein reiner „FCS-Gewinn“.

### B.4 Was genau realisierbar bleibt (Thema 3)

- Invariante Menge: Für jeden Knoten liegt (ρ_i, J_i) im M1-Kegel |J| ≤ ρ. Das ist die Aussage des Limitings nach
  Moujaes et al. (arXiv:2509.07689v1; Kegelerhalt der Reaktion Lemma 2, Gl. (28); Quelle (26)). Im Code ersetzen die
  limitierten Momente die PN-Zeilen 0/1 (`rate`), unter dt ≤ `realizability_step()`.
- PN-Zeilen l ≥ 2 werden nicht limitiert. Die PN-Rekonstruktion ψ_N(Ω) = ∑ a_lm Y_lm darf negativ sein, auch wenn
  (ρ, J) realisierbar ist. Gegenbeispiel: isotroper Zustand plus großes a_20 hat ρ > 0, J = 0 im Kegel, aber ψ_N < 0
  bei Ω ∥ z.
- Wo das zählt: nur am Rand, weil der Ghost-Zustand aus der Spur gebildet wird. Dort projiziert der Code die Spur auf
  ein nichtnegatives, momententreues Maß (docs/POSITIVITY_PROJECTION.md, Notiz Gl. 4). Im Inneren hat ψ_N < 0 keine
  Folge für die M1-Aussage.
- FCS: B ≥ 0 ist exakt, C liegt (über die M1-Momente) im Kegel. Der Kegel ist konvex, also ist auch B + C
  realisierbar. Das braucht kein eigenes Argument.
- Low-Order-Fluss: Er nutzt die M1-Schließung (Levermore), der High-Order-Fluss den PN-Fluss. Ohne aktiven Limiter
  folgen die Zeilen 0/1 dem PN-Modell. Das Verfahren ist PN-konsistent mit M1-Realisierbarkeit als Nebenbedingung,
  kein M1-Modell.

### B.5 Zeitordnung des gekoppelten Verfahrens (Thema 6)

Schritt: R(τ/2) ∘ T_SSPRK(4,3)(τ) ∘ R(τ/2). R ist die exakte Reaktion mit eingefrorener Quelle, T der Transport
dritter Ordnung. Strang ist zweiter Ordnung, wenn beide Teilflüsse mindestens zweiter Ordnung sind. Der
Kommutator [T, R] ≠ 0 bei ortsabhängigem σ, Quelle oder Filter.

FCS-Quelle: q(t) wird je Halbschritt in der Mitte ausgewertet (t + τ/4, t + 3τ/4; docs/FCS.md) und im Halbschritt
konstant gehalten. Das ist die Mittelpunktregel, Fehler O(τ³) je Schritt; die Ordnung 2 bleibt.

Die Ordnung 3 des isolierten Transports überträgt sich daher nicht. Gemessen (docs/ELEMENT_WENO.md, Abschnitt
„Zeitordnung mit Reaktion“): Ordnung 2.0–2.2. Bei CFL 1 liegt der Zeitfehler 2–3 Größenordnungen unter dem
Ortsfehler.

Weitere Ordnungsbegrenzer derselben Art, alle räumlich:
- Limiter an glatten Extrema
- nodale Materialmittelung an Sprüngen
- FCS-Quadratur über die Front

### B.6 Kegelerhalt der Reaktion, und was anisotrope Streuung ändern würde (Thema 7)

Exakte lokale Lösung je Knoten (src/element_weno.cpp, `react`):

    ρ(τ) = e^{−σ_a τ} ρ + q0 τ φ1(−σ_a τ),   a_lm(τ) = e^{−(σ_a + σ_s + f_l) τ} a_lm  (l ≥ 1).

Daraus |J(τ)| = e^{−(σ_a+σ_s+f_1)τ}|J| ≤ e^{−σ_a τ}|J| ≤ e^{−σ_a τ} ρ ≤ ρ(τ), weil q0 ≥ 0 und φ1 > 0.

**Anisotrope Streuung** (nicht implementiert): Bei rotationsinvariantem Kern k(Ω·Ω′) diagonalisiert Funk–Hecke den
Operator in Y_lm mit Eigenwerten σ_s k_l, k_0 = 1, |k_l| ≤ 1 (Nummer nicht geprüft). Dann zerfällt a_lm mit
σ_a + σ_s(1 − k_l). Der Kegelerhalt braucht σ_s(1 − k_1) ≥ 0, also k_1 ≤ 1. Das gilt für jeden physikalischen Kern.
Die FCS-Quelle würde zu q_lm = σ_s k_l u_lm^B; ihr M1-Teil (q_0, q_1) mit |q_1| ≤ q_0 bleibt zulässig, weil q die
Momente eines nichtnegativen Maßes sind. Für allgemeine, nicht rotationsinvariante Kerne ist der Operator nicht
diagonal, und das Argument entfällt.

### B.7 Dirac-Einstrom (Thema 9)

Atom ψ_in = m δ_{Ω0} auf S² (∫_{S²} δ_{Ω0} dΩ = 1, also ρ-Beitrag m). Code (src/boundary.cpp, `add_atoms`):
- Onsager-/C1-Datum d_j = 2 m Y_j^odd(Ω0) über die ungeraden Harmonischen im Randrahmen. Das ist dieselbe
  Normierung wie die Quadraturversion `2 w g Y_odd` in `Boundary::data`; der Faktor 2 gehört zum
  Halbraum-Ansatz.
- M1-Außenzustand (m, m Ω0) liegt genau auf dem Kegelrand. Rundung kann ihn um ein ulp hinausschieben. Der Code rundet
  die Dichte gerichtet nach außen (`finish_data`, höchstens 64 ulp, exaktes Prädikat), ohne Toleranz.
- Die PN-Projektion eines Dirac ist stark vorzeichenwechselnd (Gibbs auf S²). Das betrifft nur das PN-Datum; der
  M1-Außenzustand ist exakt das kinetische Halbraummoment.

2D-Slab: Ein Atom ist ein S²-Dirac mit Richtung in der Ebene. Die Winkelreduktion setzt gerade Daten in z voraus und
wird zur Laufzeit geprüft (`dropped_inflow` ≤ 1e-12).

*Erklärungslücke:* Die Herleitung des Faktors 2 und der C1-Datumsabbildung `llf_datum` steht nur im Code
(src/boundary.cpp, Konstruktor), nicht in docs.

### B.8 Optische Dicke pro Element (Thema 8, Hypothese)

LLF-/Graph-Viskosität ~ h. Bei σh ≳ 1 ist ein Element optisch dick:
- Das Profil e^{−σx} im Absorber ist nicht aufgelöst.
- Das Verfahren ist nicht asymptotisch erhaltend (keine Diffusionsgrenze).

| Lauf | σ_a h |
|---|---|
| 2D-Lattice 448² | 0.16 |
| 3D-Lattice 112³ | 0.63 |
| 3D 56³ | 1.25 |
| 3D 28³ | 2.5 |

Die beobachtete langsame Konvergenz der 3D-Masse (−42 → −20 → −8.9 % gegen MC) passt zu einem präasymptotischen
Regime. Die Quellzelle grenzt direkt an Absorber (Zellen mit gerader Indexsumme).

*Nicht geprüft:* dass die Konvergenz bei σh ≪ 1 in die erwartete Ordnung übergeht. Testvorschrift in D.2.

---

## C. Theorie-Code-Zuordnung

| Eigenschaft | Formel | Ort | Status |
|---|---|---|---|
| Hochordnungsziel mit konsistenter Masse | M ȧ^H = −C·A a + b^H − s_h(a) | `ElementWeno::rate` | korrekt umgesetzt |
| WENO-Stabilisierung | B.1 | `weno_gamma`, `stabilization_load` | umgesetzt; Stabilität mit variablem γ_e mathematisch nicht abgesichert |
| Elementweise Erhaltung des Limiters | ∑_j f_ij^e = 0 je Element (Abschlussterm c_ji G_i − c_ij G_j) | `rate`, docs/ELEMENT_WENO.md | korrekt; Erhaltung bis auf gezählte Nudges |
| M1-Realisierbarkeit | dt ≤ SSP-Koeff. · dt_bound | `realizability_step` | korrekt (exakte Arithmetik); Gleitkomma über exakte Prädikate (src/cone.hpp) |
| Hochordnungs-Zeitschritt | 0.9 c/ρ(L) per Potenziteration | `high_order_step` | Heuristik (Operator ohne Randterme, nicht dissipativ; docs/ELEMENT_WENO.md) |
| Exakte Reaktion, Kegelerhalt | B.6 | `react` | korrekt für isotrope Streuung, q0 ≥ 0 |
| Strang | R(τ/2) T(τ) R(τ/2) | Treiber, `advance` | korrekt, Ordnung 2 (gemessen) |
| FCS-Zerlegung | B.2 | src/fast_main.cpp (FCS-Setup), src/uncollided*.hpp | korrekt unter den erzwungenen Annahmen |
| FCS-Deposition | B.3 | `collided_source` | korrekt; positiv; global nur bis auf Quadratur |
| Ungestoßene Analytik Lattice | Strahlintegrale mit Gauß | src/uncollided_lattice*.hpp | gegen MC validiert (2D 3e-5, 3D konvergent) |
| Randfluss PN | ½(A_n − I)(u − ext(u, d)), C1-Reflexion | `Boundary`, `add_boundary` | umgesetzt; Kontraktion < 1 beim Aufbau geprüft |
| Positivitätsprojektion Ghost | Notiz Gl. 4 | `project_positive_moments` | umgesetzt, Statistiken in summary |
| Dirac-Einstrom | B.7 | `Boundary::add_atoms` | korrekt; Herleitung undokumentiert |
| Winkel-/Spiegelreduktion | gerade Daten, symmetrisches Material | Treiber (Laufzeitprüfungen) | korrekt, geprüft |
| MPI | gleiche Felder für 1/2/4 Ränge | `ElementWenoMpi`, `layered_norm` | geprüft (bitgleich) |
| anisotrope Streuung, gerichtete Volumenquelle, Reflexion, Energie, M2 | — | — | nicht implementiert |

---

## D. Kritische Befunde (getrennt)

### D.1 Nachgewiesene Fehler

Keine Programmfehler gefunden. Ein nachgewiesener **Verfahrensdefekt**: die Rückkopplung des q = 1-Sensors (B.1).
Er ist reproduzierbar, die Ursache ist isoliert, die Wirkung gemessen. Entscheidung vom 16.11.: Standard bleibt;
im Paper erwähnen.

### D.2 Konkrete, unbestätigte Hypothesen

1. **3D-Lattice präasymptotisch** (B.8). Test: Ein Würfel mit einem Absorber neben der Quelle, gleiche Daten,
   Gitter mit σh = 2 … 0.1, gegen `m1pn_lattice_mc`. Erwartung: Ordnung steigt bei σh < 0.3.
2. **FCS-on/off-Unterschied enthält einen Diskretisierungsanteil** (B.3). Test: homogenes Medium ohne Absorber,
   reine Streuung, beide Varianten auf feinem Gitter gegen MC. Der Unterschied sollte wie h² fallen, unabhängig vom
   FCS-Gewinn bei grober Auflösung.

### D.3 Mathematische Beweislücken

1. L²-/Energiestabilität des Hochordnungsziels mit elementweisem γ_e (B.1 b). Bekannt ist nur der Fall
   konstantes γ.
2. Stabilität der Gesamtkomposition aus Hochordnungsziel, Limiter, Randprojektion und Strang. Bewiesen ist nur die
   Invarianz der M1-Momente, keine Stabilität der PN-Zeilen l ≥ 2.
3. Der Hochordnungs-Zeitschritt ist eine Spektralschätzung ohne Randterme, kein Stabilitätsbeweis.

### D.4 Bewusste Modellapproximationen

- PN-Abbruch (dominiert Line Source und Crossing)
- Winkelfilter, wo verwendet (verändert nur l ≥ 2)
- nodale Materialmittelung
- FCS-Quadratur
- Vakuum statt unendlichen Mediums beim Vergleich mit Referenzen im unendlichen Medium
- Anfangsboden 1e-10 (gegen ψ⁰ = 0 der Literatur; Effekt < 5e-9)

### D.5 Fehlende Erklärungen für korrekt Implementiertes

- Faktor 2 und C1-Datumsabbildung am Rand (B.7)
- gerichtete Rundung und Nudges als Gleitkomma-Ersatz für den exakten IDP-Beweis; das Ledger zählt sie
- Durchreichen der Kernmoden von A(n) im Randzustand (`Boundary::exterior`)
- warum B + C ohne eigenes Argument realisierbar ist (Konvexität, B.4)

---

## E. Heute ausgeführte Gegenproben

| Aussage | Referenz | Parameter | Ergebnis | Was folgt / was nicht |
|---|---|---|---|---|
| Sensor auf glatten Daten in Ordnung, Defekt dynamisch | exakte PN-Lösung, γ ≡ 1 | 1D periodisch, P3, Q1–Q4, n = 32–512 | γ = 1 auf Daten; Wachstum nur mit WENO, q = 3 stabil | Ursache isoliert; kein Spektralbeweis |
| Default unverändert | Tag-Binary | Crossing, Lattice, glatt Q4, Line + FCS | bitgleich | Studienoptionen neutral |
| Strang-Ordnung 2, Zeitfehler klein | feinstes dt | Lattice 112²/224², Line + FCS 256², CFL 1 … 1/8 | Ordnung 2.0–2.2; Zeitfehler 1e-4 gegen 1–2 % Ort | keine höhere Splitting-Ordnung nötig |
| Ungestoßene Analytik korrekt | unabhängiges MC | 2D 4e9, 3D 1e8 Historien | 2D \|z\| ≤ 1.6; 3D Quadratur konvergiert gegen MC | validiert bis 3e-5 bzw. Quadraturgrenze |
| Literatur-Lattice zuverlässig | MC 4e10 | T = 3.2 | Abweichung ≤ 0.036 % | eigene Referenz jetzt 1e-5 |
| MPI-Invarianz | 1 Rang | 9 Fälle, 2/4 Ränge | Felder bitgleich | — |

Sinnvolle weitere Proben, nur beschrieben:
- die beiden Tests aus D.2
- reiner Absorptionsstrahl mit Beer–Lambert-Referenz (existiert als Shadow Stufe A)
- Rotationstest: Lattice um 90° gedreht, Integrale bitgleich bzw. bis auf Rundung (prüft Randaufbau und Quadratur auf
  Richtungsfehler)

---

## F. Herleitungs-Checkliste (priorisiert)

| Priorität | Thema | Zweck |
|---|---|---|
| 1 | Positivität von s_h bei konstantem γ und ihr Verlust bei γ_e (B.1 a/b) | Masterarbeit, Methodenbeschreibung |
| 2 | Invarianz der M1-Momente vs. Vorzeichen der PN-Rekonstruktion, Konvexität für B + C (B.4) | Masterarbeit |
| 3 | FCS-Zerlegung mit Rand- und Anfangsdaten, Konvexitätsargument (B.2) | Masterarbeit |
| 4 | Diskrete FCS-Bilanz und Positivität der Deposition (B.3) | Methodenbeschreibung |
| 5 | Strang mit Mittelpunktquelle, Gesamtordnung 2 (B.5) | Masterarbeit |
| 6 | Kegelerhalt der exakten Reaktion (B.6) | Masterarbeit (kurz) |
| 7 | Charakteristischer C1-Rand, Kernmoden, Faktor 2 für Atome (B.7) | Methodenbeschreibung |
| 8 | Sensor-Rückkopplung als Linearisierung (B.1 c) | Paper |
| 9 | Optische Dicke, fehlende AP-Eigenschaft (B.8) | Paper, falls 3D-Lattice im Paper |
| 10 | Funk–Hecke und anisotrope Streuung (B.6) | Zukunft |

Vorschlag für Unterabschnitte im Kapitel „Verfahren“ (docs/tex):
1. „Invariante Menge und PN-Rekonstruktion“ (B.4)
2. „Zerlegung in ungestoßenen und gestoßenen Anteil“ (B.2 + B.3)
3. „Zeitintegration und Ordnung“ (B.5, mit der Messtabelle)
4. „Bemerkung zum WENO-Sensor“ (B.1)

Literatur, wie in den Projektdokumenten belegt:
- Kuzmin/Hajduk/Vedral arXiv:2509.04673v2, Gl. (6)/(7)
- Kuzmin/Vedral arXiv:2212.14224, §7
- Moujaes et al. arXiv:2509.07689v1, Gl. (26)–(28), (44)–(47)
- Schotthöfer/Hauck arXiv:2505.17284, (12a), Tab. 1
- Kusch/Stammer arXiv:2111.07160, Gl. (7)
- Schneider/Leibner, (4.12)
