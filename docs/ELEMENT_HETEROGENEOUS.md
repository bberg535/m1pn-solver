# Element-WENO mit heterogenen Reaktionen und 3D-Transportbenchmarks

Branch `feature/element-weno-heterogeneous-3d` (Basis `study/nx-element-weno-convergence-20261111` @ a62b016,
diese wiederum auf `experiment/element-weno` @ c49290b). Stand 11.11.2026.
Rohdaten und Tabellen: `results/element_heterogeneous_20261111/`.

## 0. Kurzfassung

- **Heterogene Reaktionen:** Element-WENO kennt jetzt ortsabhängige isotrope Absorption σ_a(x), Streuung σ_s(x) und
  Quelle q₀(x) ≥ 0 (`ElementWenoMaterial`). Die Knotenwerte sind positive Mittel `∫σφ_i/∫φ_i`. Der Reaktionsschritt
  wird exakt gelöst (`expm1`) und steht wie bisher als Strang-Halbschritt um den SSP-Transport. Ein Bilanzbuch
  schließt die globale Teilchenbilanz auf Rundungsniveau (≤ 1.5e-12 relativ in allen Läufen).
- **Mathematischer Hauptbefund:** Die nodale Mittelung ist für glatte variable Koeffizienten nur O(h²) (Q1, Q2) bzw.
  O(h) (Q3, Q4). Die konsistente FE-Reaktion erreicht O(h^{p+1}), erzeugt aber negative Bernstein-Koeffizienten und
  verletzt damit die Realisierbarkeit. Bei Materialsprüngen sind beide O(h).
- **Zeitordnung:** Transport allein ist dritter Ordnung (SSPRK3 und SSPRK(4,3)), mit heterogener Reaktion im
  Strang-Splitting zweiter Ordnung (gemessen 2.00).
- **2D-Lattice (Paul (44)–(47)):** Die Materialintegrale 49/1/110/38 sind exakt.
  - Gegen die kinetische Referenz von Schotthöfer/Hauck: P11, Q1/112 bei Masse −1.8 %, Gesamtabsorption +3.4 %,
    Absorptionsrate +2.0 %.
  - ρ_max 1.1280 gegenüber 1.1278 der historischen Pfad-B-Läufe P11/112.
  - Bei gleicher Kontrollpunktzahl ist Q1 am genauesten.
- **3D:** Elementzahlen pro Achse (würfelförmige Zellen) für das Weih-Gebiet 3:1:1. Implementiert sind ein Shadow-Test
  (Stufe A mit Beer-Lambert-Referenz, Stufe B mit σ = 10⁷) und ein eigener asymmetrischer 3D-Lattice. Beide sind
  stabil, kegelerhaltend und konservativ. Die lokale Auflösung (bis 48·16·16) reicht für quantitative
  Schattengenauigkeit nicht; das ist Aufgabe der vorbereiteten nx-Kampagne.
- **Effizienzschleife:** Shadow 3D 1.86-mal, 3D-Lattice 1.4-mal schneller, mit unveränderten Operatoren.
- **Kampagnenskript:** Die Smoke-Fehlschläge sind behoben (Skriptfehler, nicht Solver). Die neuen Profile
  `hetero-smoke`/`hetero-paper` haben Validierung, Metadaten, Ressourcenplan und Slotklassen. Der nx-Smoke-Test lief
  5/5 durch.

Die vollständige Abnahme steht in §12.

## 1. Literatur: geprüfte Aussagen und Gleichungsnummern

| Quelle | eingesehene Fassung | verwendet |
|---|---|---|
| Moujaes, Kuzmin, Bäumer („Paul“) | arXiv:2509.07689v1 (lokale Kopie); Journalfassung Math. Comput. Simul. 2026 **nicht** eingesehen | §5.4 (44)–(47) Lattice; §3.2 (26) Quelle; §3.3 (27), Lemma 2 (28) Reaktion |
| Schneider, Leibner | arXiv:1904.03098 (Teil II = JCP 2022, 111040) | §4 (4.1a/b) Strang; (4.3) Mittelpunktsregel; (4.7)–(4.12) exakte Reaktion |
| Schotthöfer, Hauck | arXiv:2505.17284 | §3.1 (7)–(14) Lattice; Tab. 1 kinetische Referenz-QoIs |
| Weih et al. | arXiv:2007.05718v2 (MNRAS 498, 3374) | §4.2.1 Shadow-Geometrie und Opazität |
| Kuzmin, Hajduk, Vedral; Kuzmin 2026; Kuzmin, Vedral | arXiv:2509.04673v2, 2602.02095, 2212.14224 | bestehendes Verfahren (`ELEMENT_WENO.md`); hier nichts Neues übernommen |
| Guermond, Popov, Tomas (2019) | nicht eingesehen | – |
| OECD/NEA Kobayashi (2000) | Server von hier nicht erreichbar | nur Sekundärangaben; kein Test daraus abgeleitet |

Aussagen im Einzelnen:

- **Paul (44):** `D_a` besteht aus 11 Einheitszellen, also |D_a| = 11, ∫σ_a = 110 und ∫σ_s = 49 − 11 = 38.
  Paul legt einen **do-nothing-Auslass** fest (Text vor (47)); (45) ist die isotrope Quelle auf [3,4]²; (47) die
  Anfangsdaten ψ⁽⁰⁾ = 10⁻¹⁰, ψ⁽¹⁾ = 0. Die Reaktion steht implizit in jeder SSP-Stufe mit gelumpter Reaktionsmasse
  `m_i^σ = ∫σφ_i` (27). Lemma 2 (28) zeigt den Kegelerhalt mit σ_t ≥ σ_a. Die Quelle `s_i = ∫φ_i q` (26) erhält den
  Kegel für q ∈ R₁.
- **Schotthöfer/Hauck (12a):** dieselben 11 Blöcke, auf [−3.5, 3.5]² verschoben (Block für Block geprüft).
  Abweichend: Anfangswert ψ₀ = 0 und Einstrom ψ⁻ = 0. Ihre Konzentration integriert über eine Halbkugel; die
  Bilanz bestätigt das (Masse 13.07 + Gesamtabsorption 7.034 + Ausfluss ≈ 0 = 2π·3.2). Umrechnung in unsere
  ρ-Einheiten: Faktor 1/(2π).
- **Schneider/Leibner (4.12):** Der Faktor `(1 − e^{−σ_a t})/σ_a` ist bei σ_a → 0 singulär geschrieben. Hier wird
  die Form `t·expm1(z)/z` verwendet. Sie beanspruchen insgesamt nur zweite Ordnung (Strang).
- **Weih et al. §4.2.1:** Gebiet (−0.75, 0.75)×(−0.25, 0.25)², 150×50×50 Zellen, κ_a = 10⁷ in der Kugel R = 0.1 um den
  Ursprung, Strahl vom linken Rand. **Das Strahlprofil wird nicht angegeben** (Verweis auf den Strahltest in §4.1.1).
  Das Ergebnis ist rein qualitativ (Abb. 7).

Keine der Stabilitäts- oder Konvergenzaussagen der Quellen wird auf das M1PN-Verfahren übertragen; alles hier
Behauptete ist gemessen oder in §4 bewiesen.

## 2. Bestehende Struktur (unverändert)

```
M_C ȧ^H = R_C^{PN}(a) − S_WENO(a)                       Hochordnungsziel, konsistente Masse (ELEMENT_WENO.md)
ρ̇, J̇ ← limitierte Elementbeiträge aus ȧ^H                M1-Projektion und Limiter (Bericht §6)
a^{n+1} = R_{Δt/2} ∘ Φ_{Δt}^{SSP} ∘ R_{Δt/2} a^n           Reaktion außerhalb der semidiskreten Rate
```

Die Reaktion steht nicht in `rate()`. Hochordnungsziel, Elementkorrekturen, Nullsummen und Randkorrekturen sind
daher von ihr nicht berührt; die neuen Materialdaten gehen **nur** in `react()` ein. Pfad B
(`R_C + (M_L − M_C) ȧ^L`, Massenkorrektur) ist ein eigener Solver. Er teilt mit Element-WENO nur die
Mengendefinition des Lattice (`src/benchmark_cases.hpp`); eine Massenkorrektur ist in Element-WENO nicht eingebaut.

## 3. Heterogene Koeffizienten: nodale Mittelung gegen konsistente Reaktion

`σ_i = ∫σφ_i / ∫φ_i` ist für Bernstein-Basen (φ_i ≥ 0) ein positives Mittel. Die Quadratur ist zusammengesetzter
Gauß mit p + 4 Punkten pro Richtung und `subcells` Unterzellen. Für elementausgerichtete Sprünge (Lattice) ist sie
exakt (Test: ausgerichtetes Schachbrett exakt; Lattice-Integrale auf 1e-13). Bei nicht ausgerichteten Sprüngen
konvergiert sie mit den Unterzellen (Kugel: 4 Unterzellen, relativer Integralfehler 5e-4).

**Studie** (`scripts/element_weno/reaction_model_study.py`, NumPy, unabhängig vom Solver; 1D periodisch,
σ = 1 + ½ sin 2πx, Fehler gegen die L²-Projektion der exakten Lösung):

| Q_p | nodal: Rate / Lösung t=1 | konsistent: Rate / Lösung | Sprung (L1, nicht ausgerichtet) | min. Koeffizient nodal / konsistent |
|---|---|---|---|---|
| Q1 | O(h²) / O(h²) | O(h²) / O(h²) | beide O(h) | 0 / −4e-4 |
| Q2 | O(h²) / O(h²) | O(h³) / O(h³) | beide O(h) | 0 / −2.5e-3 |
| Q3 | → O(h) / → O(h) | O(h⁴) / O(h⁴) | beide O(h) | 0 / −3.7e-3 |
| Q4 | → O(h) / → O(h) | O(h⁵) / O(h⁵) | beide O(h) | 0 / −9.1e-3 |

Folgerungen:

1. Die nodale Mittelung ist für konstante σ exakt. Für glatte variable σ ist sie **nicht hochordentlich
   konsistent**: höchstens zweiter Ordnung, für Q3/Q4 asymptotisch erster Ordnung. Die koeffizientenweise
   Multiplikation von Bernstein-Koeffizienten ist kein Produkt von Funktionen. Das passt zum Befund des
   Parallelzweigs, dass koeffizientenweise Operatoren für p ≥ 3 nur schwach konsistent sind.
2. Die konsistente Reaktion `M⁻¹M_σ` ist hochordentlich, ihre Exponentialfunktion erzeugt aber negative
   Koeffizienten aus nichtnegativen Daten. Sie ist mit der koeffizientenweisen Realisierbarkeit des Limiters
   nicht verträglich.
3. Bei Materialsprüngen ist der Unterschied klein (beide O(h)). Für Lattice und Shadow ist die nodale Mittelung
   also kein Ordnungsverlust. Sie verschmiert die Sprünge über die Elementbreite, nicht über den Knotenabstand.
   Bei gleicher Kontrollpunktzahl verlieren hohe Grade deshalb Genauigkeit (§5).
4. Die exakte Lösung des nodalen Modells ist **nicht** die exakte Lösung der konsistenten FE-Reaktion; beide
   unterscheiden sich um die Größen der Tabelle.

## 4. Reaktionsschritt: exakte Lösung, Realisierbarkeit, Bilanz, Zeitordnung

Pro Knoten und Momentgrad l (`ElementWeno::react`, f_l = Filterrate, nur für l ≥ 1 ungleich 0):

```
ρ(τ)    = e^{−σ_a τ} ρ + q₀ τ φ₁(−σ_a τ),     φ₁(z) = expm1(z)/z,  φ₁(0) = 1
a_lm(τ) = e^{−(σ_a + σ_s + f_l) τ} a_lm        (l ≥ 1)
```

Der Grenzfall σ_a → 0 ist stetig (Test: σ_a = 10⁻³⁰⁰ gegen 0 auf 1e-15). Für σ_a = 10⁷ ergibt sich ρ = q₀/σ_a im
Kegel. Gegenprobe: Die Reaktion stimmt auf 1e-16 mit der unabhängigen `reaction()` von Pfad B (`src/time.cpp`)
überein.

**Satz (Realisierbarkeit des Reaktionsschritts).** Ist ρ ≥ |J| und sind σ_a, σ_s, f₁ ≥ 0 und q₀ ≥ 0, dann gilt
ρ(τ) ≥ |J(τ)|.
*Beweis.* q₀ τ φ₁ ≥ 0, also ρ(τ) ≥ e^{−σ_a τ} ρ ≥ e^{−σ_a τ} |J| ≥ e^{−(σ_a+σ_s+f₁)τ} |J| = |J(τ)|. ∎
In binary64 kann das Runden einen Zustand auf dem Kegelrand um wenige ulp herausschieben. Dann wird der
gespeicherte Dichtekoeffizient ulp-weise erhöht, bis die gespeicherte Zeile selbst im Kegel liegt (koeffizientengenau,
keine `set_low`-Rundreise); diese Masse wird gezählt (`nudge_mass`). Das ist dieselbe Rundungskorrektur wie nach
den Transportstufen, kein Floor und kein physikalisches Clipping. Die hohen PN-Momente werden nur exakt gedämpft.

**Bilanzbuch** (ρ-Einheiten, `ElementWenoStats`): Randausfluss aus den Randraten jeder Stufe mit den effektiven
SSP-Gewichten (1/6, 1/6, 2/3 bzw. 1/6, 1/6, 1/6, 1/2). Er enthält den gelumpten Randfluss
`−(∮φ_i n)·F^L_i` der Low-Order-Elementterme, denn nur die antidiffusiven Beiträge summieren sich elementweise zu
null. Absorption analytisch `m_i[ρ(1 − e^{−σ_aτ}) + q₀τ(1 − φ₁)]`, nicht als Differenz. Damit
`M(T) − M(0) + B + A − S − N = 0` auf ≤ 1.5e-12 relativ in allen Läufen (Lattice, Shadow, 3D-Lattice, Crossing mit
Einstrom, gespiegelter Line Source).

**Zeitordnung** (`tests/element_weno_time_study.cpp`; periodisch 2D, P3, Q2, 8², glatte heterogene σ_a, σ_s, q₀;
Selbstreferenz mit dt/2⁹):

| | SSPRK3 | SSPRK(4,3) |
|---|---|---|
| nur Transport | 3.00 | 3.00 |
| mit Strang-Reaktion | 2.00 | 2.00 |

SSPRK(4,3) macht das Gesamtverfahren also nicht dritter Ordnung, sobald Reaktionen mit räumlich variablen
Koeffizienten vorhanden sind. Die Reaktionsoperatoren kommutieren nicht mit dem Transport. Bei konstanten
Koeffizienten kommutiert die Reaktion mit dem Transport, nicht aber zwingend mit Limiter und Randbehandlung; das
ist nicht separat gemessen.

## 5. 2D-Lattice (Paul (44)–(47))

`m1pn_fast --case lattice-paul --dimension 2 --inflow vacuum --background 1e-10 --stabilization element-weno …`,
Gebiet (0,7)², Elementzahl durch 7 teilbar (Sprünge auf Elementkanten). Material, Quelle und Integrale stammen aus
`src/benchmark_cases.hpp`; Pfad B verwendet dieselben Prädikate (unveränderter Ausdruck).

**Randbehandlung:** Element-WENO hat am offenen Rand den schwachen C1/LLF-Rand mit Vakuum-Einstrom, Paul den
do-nothing-Auslass. Beides ist **nicht** dasselbe. Bis T = 3.2 verlassen aber nur 5e-7 bis 8e-5 der injizierten
Masse das Gebiet (Tabelle unten, Spalte Ausfluss). Für die globalen Größen ist der Unterschied damit vernachlässigbar,
für die Dichte direkt am Rand nicht untersucht. Schotthöfer/Hauck verwenden Vakuum-Einstrom wie wir.

**Symmetrie:** Material und Quelle sind spiegelsymmetrisch in x um 3.5, in y nicht (Absorber [3,4]×[1,2]). Die
automatische Reduktion halbiert daher nur in x; bei ungerader Elementzahl (Q3/35) wird nicht gespiegelt.

Lokale Serie (i5-12500, 6 Threads, SSPRK(4,3), Standard-dt, `results/element_heterogeneous_20261111/lattice_local/`):

| P | Q/E | Punkte | Schritte | Löser (s) | Masse | Absorption gesamt | Absorptionsrate T | Ausfluss | Bilanz | ρ_max |
|---|---|---:|---:|---:|---|---|---|---:|---:|---:|
| P3 | Q1/112 | 113 | 203 | 1.9 | 2.015 (−3.2 %) | 1.185 (+5.9 %) | 0.783 (+7.1 %) | 5e-7 | 1e-13 | 1.0461 |
| P3 | Q2/56 | 113 | 165 | 1.7 | 1.992 (−4.2 %) | 1.208 (+7.9 %) | 0.791 (+8.1 %) | 1e-6 | 4e-13 | 1.0390 |
| P7 | Q1/112 | 113 | 203 | 2.3 | 2.040 (−1.9 %) | 1.160 (+3.6 %) | 0.747 (+2.1 %) | 1e-5 | 1e-13 | 1.1409 |
| P7 | Q2/56 | 113 | 166 | 2.0 | 2.020 (−2.9 %) | 1.180 (+5.4 %) | 0.757 (+3.4 %) | 2e-5 | 3e-13 | 1.1373 |
| P11 | Q1/112 | 113 | 203 | 4.5 | 2.043 (−1.8 %) | 1.157 (+3.4 %) | 0.746 (+2.0 %) | 2e-5 | 1e-13 | 1.1280 |
| P11 | Q2/56 | 113 | 166 | 3.1 | 2.024 (−2.7 %) | 1.176 (+5.1 %) | 0.755 (+3.2 %) | 2e-5 | 5e-13 | 1.1220 |
| P11 | Q3/35 | 106 | 197 | 6.7 | 1.998 (−3.9 %) | 1.202 (+7.3 %) | 0.759 (+3.7 %) | 4e-5 | 1e-13 | 1.1167 |
| P11 | Q4/28 | 113 | 265 | 5.6 | 1.987 (−4.5 %) | 1.213 (+8.4 %) | 0.761 (+4.0 %) | 8e-5 | 1e-12 | 1.1214 |

Klammern: Abweichung von Schotthöfer/Hauck Tab. 1 (kinetisch, S22, h = 10⁻³) / 2π: Masse 2.0802, Absorption 1.1195,
Rate 0.7316. Darin stecken PN-Modellfehler, Ortsfehler und die abweichenden Anfangs-/Einstromdaten.

- **N:** P3 → P7 halbiert die Abweichungen ungefähr; P7 → P11 bringt wenig. Der Rest ist räumlich bzw.
  modellbedingt; die Feingitterläufe (Q1/224, Q2/112) sind für die nx-Kampagne geplant.
- **Grad bei gleicher Punktzahl:** Q1 > Q2 > Q3 > Q4. Das ist mit §3.3 verträglich, aber nicht allein dadurch
  bewiesen.
- **Pfad B (historisch, 15 NX-Läufe, älteres Binary):** Die Felder sind gelöscht, erhalten sind ρ_max und der Schnitt
  y = 3.5 bei N = 112. ρ_max: P3 1.04607 (EW Q1/112: 1.0461), P11 1.12782 (EW: 1.1280). Abbildung
  `lattice_lineout.png`: die Schnitte von B und EW liegen in der Log-Skala übereinander. Laufzeiten von B sind nicht
  vergleichbar (älteres Binary, nicht gemessen); die Neuberechnung mit aktuellem Binary ist in `hetero-paper`
  enthalten (`lattice_B_*_B_new`, gleiches dt wie historisch).

## 6. Geometrie: Elementzahl pro Achse

`ElementWenoOptions::elements_per_axis` mit würfelförmigen Zellen h = length/elements. Die Box ist
origin + h·(n_x, n_y, n_z). Pro Richtung: Konnektivität, Nachbarn, Ursprünge, Knotenpositionen, 1D-Masse und
-Konvektion, LDLᵀ-Faktor, Linienlösungen, Auswertung. Gradienten, Massenmatrizen, Viskosität ν = h/(2p),
CFL- und Spektralschranke, Randflächen und Normalen bleiben unverändert korrekt, weil h in allen Richtungen
gleich ist.
**Nicht unterstützt:** nicht-würfelförmige Zellen. Für Lattice und Shadow reicht das; eine nichtisometrische Skalierung
wird nicht verwendet.
Test: x-abhängige Daten auf periodischen Boxen 12×4 und 6×2×3 ergeben knotenweise dasselbe wie der Würfel
(Rate ≤ 1.2e-14). Der Test fand während der Entwicklung einen Stride-Fehler in der Konnektivität.

## 7. 3D-Shadow

`--case shadow-weih --dimension 3 --inflow shadow-beam --shadow-sigma σ --background 1e-6`, Gebiet nach Weih
(n = Zellen pro 0.5, also 3n·n·n). **Strahl (eigene Festlegung):** kollimiert in +x, Dichte 1 für
r = |(y,z)| ≤ 0.15, cos²-Übergang bis 0.2, danach 0.

Ein erster Versuch mit einer Ebenenwelle über die ganze Eintrittsfläche zeigte: Ein Strahl, der tangential an den
Seitenflächen entlangläuft, wird dort durch den Identitätsanteil des LLF-Randflusses `½(A_n − I)(u − ext)`
abgesaugt, obwohl Ω·n = 0. Das ist eine Eigenschaft der vorhandenen schwachen Randbehandlung bei streifender
Einstrahlung. Hier wurde sie nicht verändert, sondern umgangen.

**Kugelabsorption:** nodale Mittelung mit 4 Unterzellen (∫σ_a relativ 5e-4 zum exakten 4/3·πR³σ).
**Spiegelung:** automatisch in y und z (Viertel); x wird wegen der Einstrahlung abgelehnt.
**Referenz Stufe A:** unkollidierte Beer-Lambert-Dichte ρ_b(y,z)·exp(−σ·Sehne), exakt für den kinetischen
δ-Strahl. Die Abweichung enthält den Winkelmodellfehler von PN/M1 bei einem δ-Strahl, die Ortsauflösung und den
Hintergrund 10⁻⁶.

Lokale Ergebnisse (Q2, T = 1.6, 6 Threads; vollständig in
`results/element_heterogeneous_20261111/shadow_local/shadow_table.md`, Profile in `shadow_profiles.png`):

| Kennzahl (Referenz) | P3, n=8 → 16 | P7, n=8 → 16 | P11, n=8 → 16 |
|---|---|---|---|
| ρ auf der Achse bei x = −0.25, freier Strahl (1), σ = 10 | 0.37 → 0.53 | 0.55 → 0.80 | 0.48 → 0.73 |
| ρ auf der Achse bei x = 0.5, σ = 10 (exp(−2) = 0.135) | 0.051 → 0.032 | 0.099 → 0.126 | 0.097 → 0.138 |
| ρ auf der Achse bei x = 0.5, σ = 100 (2e-9) | 0.036 → 0.014 | 0.017 → 0.077 | – |
| ρ auf der Achse bei x = 0.5, σ = 10⁷ (0) | 0.015 → 0.007 | 0.002 → 0.025 | 0.010 (n=8) |
| Löserzeit n = 16 | 50–55 s | 101 s | 221 s |

Die relative L1-Abweichung in der Ebene z = 0 (x ∈ [0.2, 0.7]) liegt bei 0.76–1.09. Sie wird von der seitlichen
Aufweitung des Strahls dominiert und ist auf diesen Gittern noch kein Konvergenzmaß.

- **Stabilität:** Bei allen σ einschließlich 10⁷ ist die Bilanz ≤ 8e-15 und die Kegelmarge ≥ 0. Der steife Fall
  braucht keine Sonderbehandlung, weil die Reaktion exakt integriert wird.
- **Genauigkeit:** Bei n = 8 (Strahlradius etwa 2.4 Elemente) fällt die Strahldichte schon im freien Raum stark ab.
  Mit n = 16 und P ≥ 7 erholt sie sich deutlich (0.80 statt 0.55). P3 bleibt bei etwa 0.53; dort begrenzt das
  Winkelmodell. Für σ = 10 trifft P11 auf n = 16 die Beer-Lambert-Schattentiefe (0.138 gegen 0.135). Bei σ ≥ 100
  füllt sich der Schatten seitlich auf (P7, n = 16: 0.025–0.077 statt ≈ 0); der Anteil wächst zunächst mit P, weil
  der Strahl bei höherem P weniger zerfließt und mehr Dichte am Kugelrand vorbeiträgt. Quantitative Konvergenz erst ab
  n ≥ 32 (nx).
- **Stufe B:** Weih et al. zeigen einen nahezu vollständigen Schatten bei 150×50×50 (Lattice-Boltzmann, erster
  Ordnung). Ein quantitativer Vergleich ist nicht möglich (anderes Winkelmodell, Strahlprofil unbekannt) und wird
  nicht behauptet.

## 8. Eigener 3D-Lattice (keine Literaturreferenz)

D = (0,7)³, Zellen C_ijk = [i,i+1]×[j,j+1]×[k,k+1]:

```
A = { (i,j,k) ∈ {1,…,5}³ : i+j+k gerade } \ {(4,2,2)}  ∪  {(3,3,1)}     (62 Zellen)
σ_a = 10 auf A, sonst 0;   σ_s = 0 auf A, 5 auf den nicht absorbierenden Zellen mit k = 5, sonst 1;   q₀ = 1 auf C_333
```

- **Unabhängige Zählung** (`cases::lattice3d_counts`): 62 absorbierende Zellen, 37 stark streuende, 244 streuende,
  also ∫σ_a = 620, ∫σ_s = 429, ∫q₀ = 1, |D| = 343. Die FE-Quadratur reproduziert das auf 2e-13 (Test, p = 1, 2).
- **Asymmetrie:** Das Entfernen von (4,2,2) bricht die Spiegelung in x und die Vertauschung x↔y; (3,3,1) und die
  Schicht k = 5 brechen z. Die Spiegelreduktion schaltet sich daher korrekt ab.
- **Abgedeckt:** Materialgrenzen in allen drei Richtungen, Transport entlang der Achsen und diagonal (zentrale Quelle),
  Quelle in streuendem Gebiet, Absorptionsschatten und Mehrfachstreuung.
- **Geprüft:** Bilanz (≤ 3e-15) und Kegel. Konvergenzaussagen gibt es mangels Referenz nur als Selbstvergleich
  (nx-Kampagne).
- **Extrudierter 2D-Lattice** als dimensionsbezogener Regressionstest: **nicht implementiert.**

## 9. Symmetriereduktionen

Die Spiegelreduktion wirkt jetzt pro Richtung um die Boxmitte und nur, wenn **alle** Bedingungen erfüllt sind:
- gerade Elementzahl und offenes Gitter;
- Anfangsdaten an allen Knoten spiegelsymmetrisch (mit gespiegelten PN-Momenten);
- σ_a, σ_s und q₀ an allen Punkten der Materialquadratur gleich (bis 2·10⁷ Punkte, darüber eine deterministische
  Stichprobe von 2·10⁶; vermerkt in `mirror_reasons`);
- Einstrom: Dichte und Atome auf allen Flächen zu 4 Zeiten spiegelsymmetrisch. Externe Zustandsfunktionen führen
  zur Ablehnung.

Ergebnisse:

| Fall | gespiegelt |
|---|---|
| Lattice | x |
| Shadow | y, z |
| 3D-Lattice | keine Richtung |
| Crossing | keine Richtung (Anfangsdaten, Einstrom) |
| Line Source | x, y (wie bisher, bitgleich) |

Die Winkelreduktion bleibt auf 1D/2D beschränkt; isotrope Reaktionen erhalten die z-Parität. In 3D werden alle
(N+1)² Momente gerechnet.

## 10. Effizienzschleife

Alle Änderungen sind operatorgleich. Die Felder sind bitgleich, bis auf die neue Summationsreihenfolge der
Materialmittelung (≤ 8e-16); diese ist dafür unabhängig von der Threadzahl.

| Maßnahme | Wirkung |
|---|---|
| `Inflow::steady`: Randdaten pro Flächenpunkt einmal | weniger kinetische Auswertungen |
| Memo pro Flächenknoten (Außenzustand, Levermore, Bar State bei gleicher Einstrahlung) | Rand Shadow 1.71 → 0.38 s |
| Randkorrekturen parallel über sortierte Randknoten | `finish` 0.66 → 0.14 s |
| Paartabelle im Limiter | Limiter −10…18 % |
| Materialmittelung parallel, `evaluate` mit 1D-Werten, paralleles Abtasten | Init 1.00 → 0.36 s |

Ohne Feldausgabe, 6 Threads: Shadow P3/Q2/n = 8 von 5.71 auf 3.07 s, 3D-Lattice P3/Q2/14 von 3.45 auf 2.39 s.
Der 2D-Lattice ist unverändert (0.6 s; kaum Rand).

## 11. NX-Kampagne

`scripts/element_weno/nx_convergence.py`:

- **Smoke-Fehler behoben:** Aufgaben ohne Schlüssel `stabilization` verloren `--weno`, `--element-limit` und
  `--weno-linear`. Die zwei Smoke-Ablationen liefen deshalb mit WENO und Limiter; `weno_weight_02` im Paper-Profil
  wäre unbemerkt mit 1e-3 gelaufen. Der vorhandene Offline-Test schlug auf a62b016 schon fehl.
- **Neue Profile** `hetero-smoke` (5 Aufgaben) und `hetero-paper` (63 Aufgaben); die v1-Profile mit 81/21 Aufgaben
  bleiben unverändert.
- **Ressourcenplan** `plan --profile hetero-paper`: 62.7 Kernstunden. Das teuerste ist P11 auf 150×50×50
  (etwa 16 Kernstunden, 27 GB); alles passt in `med` mit 16-Kern-Slots. Der Plan ist auf dem Arbeitsplatz
  kalibriert, Unsicherheit etwa Faktor 2.
- **Metadaten:** Git-SHA, MFEM-Version, Compiler, Threads, Affinität, NUMA-Knoten, OMP-Variablen, Momente,
  Freiheitsgrade, Schritte, dt, Wand- und Lösezeit, RSS, Phasenzeiten inkl. Reaktion, Bilanzbuch, Kegelmarge.
- **nx-Smoke** (`--partition short`, Jobs 159276–159278): 5/5 auf nx-06, NUMA-gepinnt (zwei Slots pro Sockel),
  `collect` ohne Fehler.

Produktionslauf (nicht gestartet):

```bash
python3 scripts/element_weno/nx_convergence.py plan --profile hetero-paper
python3 scripts/element_weno/nx_convergence.py prepare --profile hetero-paper --campaign "$PWD/nx/tmp/ew_hetero_paper"
python3 nx/tmp/ew_hetero_paper/study.py submit --campaign "$PWD/nx/tmp/ew_hetero_paper"            # Dry-Run
python3 nx/tmp/ew_hetero_paper/study.py submit --campaign "$PWD/nx/tmp/ew_hetero_paper" --execute
```

## 12. Abnahme

| Prüfpunkt | Ergebnis |
|---|---|
| Reaktion: reine Absorption, Streuung, Quelle, kombiniert, σ_a → 0, σ_a = 10⁷ | bestanden |
| Reaktion = unabhängige Pfad-B-Reaktion | bestanden (1e-16) |
| Realisierbarkeit Reaktionsschritt (Beweis + Zufallszustände inkl. Kegelrand) | bestanden |
| Kegel nach jedem Schritt (Lattice, Shadow inkl. 10⁷, 3D-Lattice) | bestanden (Rundungskorrektur gezählt) |
| Globale Bilanz M(T) − M(0) + B + A − S − N = 0 | bestanden (≤ 1.5e-12) |
| Materialintegrale Lattice 49/1/110/38, 3D-Lattice 343/1/620/429 | bestanden |
| Nodale Mittelung: konstante/lineare Koeffizienten exakt, ausgerichtete Sprünge exakt | bestanden |
| Nodale Mittelung hochordentlich für glatte variable Koeffizienten | **nicht bestanden** (O(h²), Q3/Q4 O(h)) |
| Konsistente FE-Reaktion positivitätserhaltend | **nicht bestanden** (negative Koeffizienten) |
| Zeitordnung Transport 3, mit Strang-Reaktion 2 | gemessen (3.00 / 2.00) |
| Element-Korrektursummen, Randkorrekturen, WENO-Beiträge unverändert | bestanden (bitgleiche Altfälle, Bestandstests 140 + 43) |
| Spiegelreduktion nur bei symmetrischem Material, Quelle, Einstrom | bestanden (Lattice x, Shadow yz, 3D-Lattice keine) |
| Geometrie mit Elementzahl pro Achse | bestanden (Box = Würfel knotenweise) |
| Lattice gegen kinetische Referenz | gemessen (P11: −1.8 / +3.4 / +2.0 %) |
| Lattice gegen historische B-Läufe | ρ_max und Schnitt übereinstimmend; Felder nicht mehr vorhanden |
| Shadow Stufe A gegen Beer-Lambert | lokal bis n = 16: σ = 10 Schattentiefe P11 0.138 vs 0.135; Konvergenz **nicht untersucht** (nx) |
| Shadow Stufe B (σ = 10⁷) stabil | bestanden; quantitative Reproduktion nicht behauptet |
| 3D-Lattice Konvergenz | **nicht untersucht** (nx, Selbstvergleich) |
| Manufactured Solution mit variablen Koeffizienten im vollen Solver | **nicht untersucht** (Studie §3 nur Reaktion, §4 Zeit) |
| Extrudierter 2D-Lattice in 3D | **nicht untersucht** |
| Kobayashi-Probleme (OECD/NEA) | **nicht untersucht** (Quelle nicht erreichbar) |
| Entropiestabilität, volle PN-Realisierbarkeit, hochordentliche Reaktionskonsistenz | nicht behauptet |

## 13. Offene Punkte

1. **Reaktionsdiskretisierung:** Ein hochordentliches *und* realisierbares Reaktionsmodell fehlt. Denkbar wäre
   konsistent für l ≥ 2 und nodal für l ≤ 1, mit Limiter-Kopplung; nicht untersucht.
2. **Streifende Einstrahlung am LLF-Rand:** Der Dissipationsanteil am Rand bei Ω·n = 0 ist für tangentiale Strahlen
   unphysikalisch. Für Hohlraum-artige Fälle sollte er geprüft werden.
3. **Zweiter Sockel bei Einzelläufen:** NUMA-gerechtes paralleles Anlegen der Felder fehlt weiter; Kampagnen
   umgehen es durch Packen.
4. **C3/C7:** Die offenen Punkte aus `BOUNDARY_FIXES.md`/Audit bleiben offen.
