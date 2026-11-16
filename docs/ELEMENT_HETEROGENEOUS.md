# Element-WENO mit heterogenen Reaktionen und 3D-Transportbenchmarks

Branch `feature/element-weno-heterogeneous-3d` (Basis `study/nx-element-weno-convergence-20261111` @ a62b016,
diese wiederum auf `experiment/element-weno` @ c49290b). Stand 11.11.2026.
Rohdaten und Tabellen: `results/heterogeneous/nx_hetero_20261111/`.

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

Lokale Serie (i5-12500, 6 Threads, SSPRK(4,3), Standard-dt, `results/heterogeneous/nx_hetero_20261111/lattice_local/`):

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

> **Archiviert am 12.11.2026** (Begründung in §11b). Der Fall bleibt im Code und im Profil `hetero-smoke`
> als Funktionstest; `hetero-paper` enthält ihn nicht mehr.

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
`results/heterogeneous/nx_hetero_20261111/shadow_local/shadow_table.md`, Profile in `shadow_profiles.png`):

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
- **Neue Profile** `hetero-smoke` (5 Aufgaben) und `hetero-paper` (63 Aufgaben; seit §11b ohne Shadow 27); die v1-Profile mit 81/21 Aufgaben
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

## 11a. Ergebnisse der nx-Kampagne `ew_hetero_paper` (12.11.2026)

Profil `hetero-paper`, Git 47b8752, Binary eingefroren, 16 Packs à 4 × 16 Kerne (NUMA-gepinnt), `--cfl 0.2`.
60 von 63 Aufgaben validiert. Bilder und Tabellen: `results/heterogeneous/nx_hetero_20261111/nx_hetero_paper/`
(`lattice_*`, `shadow_*`, `lattice3d_*`, PNG und PDF; Tabellen `lattice.md`, `shadow.md`, `lattice3d.md`).

**Lattice** (`lattice.md`, `lattice_convergence`, `lattice_density_maps`, `lattice_lineout`, `lattice_error_vs_time`):

| P | Q1/112 → Q1/224: Masse | Absorption gesamt | Rate bei T |
|---|---|---|---|
| P7 | −1.91 % → −0.87 % | +3.58 % → +1.65 % | +2.20 % → +1.28 % |
| P11 | −1.80 % → −0.63 % | +3.37 % → +1.20 % | +2.06 % → +0.96 % |

Abweichungen gegen Schotthöfer/Hauck Tab. 1 / 2π.

- **Konvergenz:** Die Abweichungen halbieren sich bis drittelns sich bei Verdopplung der Auflösung, der Fehler ist also
  räumlich dominiert. Mit dem historischen B-Zeitschritt (6.5-mal kleiner) ändern sich die Größen um < 1e-4; der
  Zeitfehler ist vernachlässigbar.
- **Pfad B mit aktuellem Binary,** gleiches Gitter Q1/112 und P, auf nx mit 16 Kernen:

  | | Element-WENO | Pfad B |
  |---|---|---|
  | Masse P3 | −3.14 % | −2.98 % |
  | Masse P7 | −1.91 % | −1.83 % |
  | Rate P7 | +2.20 % | +2.05 % |
  | Lösezeit P3 | 7.5 s | 88 s |
  | Lösezeit P7 | 9.5 s | 122 s |

  Gleiche Genauigkeit bei etwa 12–13-mal kürzerer Rechenzeit. Pfad B nutzt das kleinere historische dt; mit gleichem dt
  braucht Element-WENO 57 s (P7). Pfad B P11 ist am internen Speicherbudget gescheitert (Skriptfehler, behoben,
  Nachlauf).
- **Grad bei gleicher Punktzahl:** wie lokal Q1 > Q2 > Q3 > Q4.

**Shadow** (`shadow.md`, `shadow_convergence`, `shadow_density_maps`, `shadow_profiles`):

- **σ = 10:** Die Schattentiefe hinter der Kugel konvergiert gegen Beer-Lambert. P7: 0.100 → 0.126 → 0.131
  (n = 8, 16, 32; Referenz 0.135); P11: 0.098 → 0.141 → 0.148.
- **σ ≥ 100:** Der Schatten füllt sich mit wachsender Auflösung (P7/P11 bei n = 32: 0.08–0.10 statt ≈ 0). Die
  Dichtekarten zeigen brennpunktartige Strukturen auf der Achse hinter der Kugel. Das ist der Winkelmodellfehler
  eines endlichen PN: Die PN-Darstellung des δ-Strahls enthält breite Winkelanteile, die schräg hinter den Absorber
  laufen. Feineres Gitter dämpft sie weniger. Eine Konvergenz gegen den kinetischen Schatten ist mit P ≤ 11 nicht
  zu erwarten und wird nicht behauptet.
- **Überhöhung vor dem Absorber (neuer Befund):** Der freie Strahl bei x = −0.25 erreicht bei P7 auf n = 32 bis
  1.29, auf n = 50 bis 1.25 (Referenz 1). Der Effekt wächst mit σ; bei P11 bleibt der Wert ≤ 0.92, bei P3 ≤ 0.72.
  Kinetisch gibt es bei reiner Absorption keine Rückstreuung. Die Überhöhung ist ein Artefakt des PN/M1-Modells
  bzw. der Kopplung vor einem stark absorbierenden Hindernis. Ihre Ursache ist **nicht untersucht.**
- **Querausbreitung:** Der Strahl füllt quer fast die ganze Box. Die M1/PN-Strahlaufweitung in zwei Querrichtungen
  ist in 3D erheblich.
- **Robustheit:** Bilanz ≤ 6e-12 und Kegel in allen erfolgreichen Läufen.
  - σ = 10⁷, P3, n = 32 brach bei t = 0.90 ab: Ein Bar State mit subnormalen Werten (ρ ≈ 8.6e-316) lag 636 ulp
    außerhalb des Kegels, weil binary64 dort nur noch etwa 24 signifikante Bits hat. Behoben: Unterhalb von 1e-290
    wird |J| nach exakter Skalierung mit 2⁶⁰⁰ berechnet und ρ darauf aufgerundet. Das bleibt eine reine
    Rundungskorrektur, die Zusatzmasse liegt bei ~1e-317 und wird gezählt. Test `subnormal_rounding`.
  - P11, n = 50 (Weih-Auflösung) überschritt das Zeitlimit von 7 h bei t = 1.24 von 1.6.

**3D-Lattice** (`lattice3d.md`, `lattice3d_slices`, `lattice3d_convergence`):

- Bilanz ≤ 6e-12, Kegelmarge > 0.
- Die drei Mittelebenen zeigen die asymmetrische 3D-Struktur (z. B. den Absorber (3,3,1) unter der Quelle).
- Masse bei T über die Auflösungen (Q2/14, Q1/28, Q2/28): 0.33, 0.41, 0.49 (P3) bzw. 0.50 (P7). Das ist **nicht
  konvergiert**; für Konvergenzaussagen braucht es feinere Gitter (Zellbreite ≤ 1/8 der Materialwürfel).

**Kostenmodell:** Die Kampagne lief mit `--cfl 0.2`, das Modell war auf CFL 1 kalibriert und 3D lief auf nx 1.8-mal
langsamer. Das Modell ist korrigiert (CFL der Aufgabe, 3D-Faktor 1.8). `hetero-paper` entspricht damit etwa 550
Kernstunden. Nachlauf der drei Fehlschläge: Kampagne `ew_hetero_paper_recovery` (2 × 32 Kerne, Partition `long`).

## 11b. Shadow-Test archiviert (12.11.2026)

**Entscheidung:** Der Shadow-Test wird nicht weiter als Validierung verwendet. `hetero-paper` hat damit 27 statt 63
Aufgaben; `hetero-smoke` behält eine Shadow-Aufgabe je Stufe als Funktionstest (3D, Spiegel, Material, σ = 10⁷).
Die Ergebnisse aus §11a bleiben unverändert in `results/heterogeneous/nx_hetero_20261111/nx_hetero_paper/shadow*`.
Nachlauf: `shadowB_P3_p2_E32_s1e7` abgeschlossen; `shadowB_P11_p2_E50_s1e7` (Job 159521) abgebrochen.

**Begründung:**

- **Der freie Strahl ist schon falsch, nicht erst der Schatten.** Bei σ = 1 enthält das Gebiet 0.124 (P7) bzw.
  0.133 (P11) Masse; der Einstrom ist korrekt (Anfangsrate 0.095 = Strahlquerschnitt). ρ auf der Achse bei x = 0.5
  liegt aber bei 0.24 bzw. 0.48 statt 0.82: Der Strahl verteilt sich über den Boxquerschnitt (0.098 / 0.25 ≈ 0.39).
  Ursache ist die PN-Darstellung des δ-Strahls in der Richtung; die Box (Halbbreite 0.25, Strahlradius 0.2) lässt
  keinen Raum. Bei σ ≥ 100 füllt sich der Schatten seitlich auf.
- **Nicht stationär bei T = 1.6:** P11 gewinnt in den letzten 0.14 noch etwa 5 % Masse; Beer-Lambert ist stationär.
- **Literatur:** Weih et al. (arXiv:2007.05718) ist ein Lattice-Boltzmann-Paper. Der Stencil enthält eine diskrete
  Richtung genau entlang x („All other directions do not matter“); der Test ist dort trivial. Das Ergebnis ist ein
  einziges Farbbild, ohne Referenz und ohne Zahl. M1 oder PN wurden für den Shadow-Test nicht gerechnet.
- **FCS:** Mit First-Collision-Aufspaltung ist die Lösung rein unkollidiert (σ_s = 0) und gleich Beer-Lambert; der
  Test prüft dann nur den unkollidierten Löser.

**M1-Ziel (Diagnose, `--m1-target on`):** Option `ElementWenoOptions::m1_target`. Hochordnungsrate der vier
Momente ist die Gruppen-FE-Galerkin-Rate −Σ_d M⁻¹C_d F^L_d des nodalen Levermore-Flusses plus s_h, G = F^H − F^L = 0,
Rand nur mit dem M1-Fluss niedriger Ordnung (Korrekturfaktor 0). Nur P1, offenes Netz, ohne Spiegel. Bei n = 8,
σ = 10 (Q2, ein Kern):

| | Achse x = 0.5 (Ref. 0.135) | Strahlrand y = 0.175, x = 0.5 (Ref. 0.5) | freier Strahl x = −0.25 (Ref. 1) | Masse |
|---|---:|---:|---:|---:|
| P1, PN-Ziel | 0.094 | 0.067 | 0.40 | 0.066 |
| P7, PN-Ziel | 0.100 | 0.152 | 0.55 | 0.098 |
| M1-Ziel | 0.037 | 0.446 | 0.36 | 0.091 |

M1 hält den Strahl zusammen und wirft einen Schatten mit zwei Strahlmaxima bei y ≈ ±0.15. Hinter der Kugel ist der
Schatten zu tief und schließt sich ab x ≈ 0.55 durch Verschmelzen der Teilstrahlen (bekannte M1-Eigenschaft bei
sich kreuzenden Strahlen). Bilanz 4e-14. Die Option ist nur ein Diagnosewerkzeug und nicht Teil der Kampagnen.

**Was stattdessen nötig wäre:** ein Strahltest mit Streuung und kinetischer Referenz (S_N), eine breitere Box und
T ≈ 3, oder der Vergleich mit FCS.

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
| Lattice gegen kinetische Referenz | konvergierend (P11 Q1/224: −0.63 / +1.20 / +0.96 %) |
| Lattice gegen Pfad B (aktuelles Binary) | gleiche Genauigkeit, Element-WENO 12–13× schneller (P3, P7) |
| Lattice gegen historische B-Läufe | ρ_max und Schnitt übereinstimmend; Felder nicht mehr vorhanden |
| Shadow Stufe A gegen Beer-Lambert | σ = 10: konvergiert (P7 0.131, P11 0.148 vs 0.135 bei n = 32); σ ≥ 100: Schatten füllt sich (Winkelmodell) |
| Shadow: kein Aufstau vor dem Absorber | **nicht bestanden** (P7 bis 1.29 statt 1; Ursache offen) |
| Robustheit subnormaler Zustände (σ = 10⁷) | nach Korrektur bestanden (Test; Nachlauf n = 32) |
| Shadow Stufe B (σ = 10⁷) stabil | bestanden; quantitative Reproduktion nicht behauptet |
| 3D-Lattice Konvergenz | **nicht erreicht** auf Q2/28 (Masse 0.33 → 0.41 → 0.50); feinere Gitter nötig |
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
