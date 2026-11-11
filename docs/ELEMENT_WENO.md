# Elementbasiertes Limiting mit CG-WENO-Ziel

Eigener Pfad, unabhängig von `Transport`: `src/element_weno.{hpp,cpp}`, Binary
`m1pn_element_weno`, Test `tests/element_weno.cpp`. Uniforme kartesische Gitter
auf `[0,1]^d`, Tensor-Bernstein `Q_p`.

## Verfahren

Hochordnungsziel für alle PN-Koeffizienten, mit konsistenter Masse:

    M ȧ^H = -C·A a + b^H - s_h(a)

`s_h` nach Kuzmin, Hajduk, Vedral (arXiv:2509.04673v2), Gl. (6)/(7):

    s_h^e(v,u) = ν_e ∫_e ∇v·(∇u - γ_e g_h) dx,  ν_e = h/(2p)
    γ_e = 1 - min(1, ‖u - u*‖_e / ‖u‖_e)^q

`g_h` ist die konsistente L²-Projektion von `∇u_h`, `u*` die Hermite-WENO-
Rekonstruktion aus den von-Neumann-Nachbarn (Kuzmin, Vedral, arXiv:2212.14224).

Antidiffusive Elementbeiträge:

    f_ij^e = m_ij^e (q̇_i - q̇_j) + d_ij^e (u_i - u_j) + c_ji^e·G_i - c_ij^e·G_j
    f_i^e  = Σ_j f_ij^e - P s_h^e(φ_i, a)

mit `q̇ = P ȧ^H`, `G = F^H - F^L` und dem Limiter aus Bericht §6.
Die Ränder ergänzen `h_i = P b_i^H - b_i^L - (∮φ_i n)·G_i`.

## Abweichungen von der ursprünglichen Formel

- Abschlussterm `c_ji·G_i - c_ij·G_j` statt `c_ij·(G_i - G_j)`. Beide ergeben
  die gleiche Knotensumme, aber nur die erste summiert auf jedem Element zu null.
- `-s_h^e`, weil `s_h` im Paper auf der linken Seite steht.

## Getroffene Festlegungen

| Punkt | Wahl | Quelle |
|---|---|---|
| `γ_e` | ein Wert pro Element aus `ψ^(0)`, für alle PN-Komponenten | eigene Wahl, Literatur legt es nicht fest |
| `q`, `r`, `ε` | 1, 2, 1e-6 | Paper; `--weno-q` |
| lineare Gewichte | Nachbarn 1e-3 | Kuzmin, Vedral §7; `--weno-linear` |
| Positivitätsanteil `ε`, Faktor `θ` | 1e-3, 0.999 | im Bericht offen (`[CHECK]`) |
| Anfangsdaten Crossing | Koeffizienten = Punktwerte | Paper, nur 2. Ordnung |
| Randkorrektur | knotenweise, eigener Faktor | eigene Wahl |

## Ergebnisse (P3, 9.11.2026, ein Kern)

Periodische Sinuswelle, 1D, ohne aktiven Limiter (EOC der letzten Verfeinerung):

| p | WENO, q=1 | keine Stabilisierung |
|---|---|---|
| 1 | 2.05 | 2.00 |
| 2 | 3.03 | 2.00 |
| 3 | 3.23 (q=3 oder γ=1: 3.79, bei n=128) | 4.00 |
| 4 | 4.99 | 4.01 |

Bei `q=1` skaliert `1-γ` wie `h²`; das kostet bei p=3 Ordnung. Mit `q=3` ist das
Ergebnis identisch zu `γ=1`.

Crossing-Studie, gleiche Knotenzahl, `T=0.32`, CFL 0.5. Rohdaten in
`results/element_weno/crossing_20261109.txt`.

| Grad, Gitter | Gewicht | ρ_max | L² gegen Kinetik | Laufzeit |
|---|---|---|---|---|
| p=1, 96² | 1e-3 | 1.511 | 8.65e-2 | 77 s |
| p=1, 96² | 0.2 | 1.376 | 8.41e-2 | 77 s |
| p=2, 48² | 1e-3 | 1.320 | 7.98e-2 | 39 s |
| p=2, 48² | 0.2 | 1.269 | 7.72e-2 | 40 s |
| p=3, 32² | 1e-3 | 1.248 | 7.47e-2 | 29 s |
| p=3, 32² | 0.2 | 1.115 | 6.93e-2 | 29 s |

Der L²-Wert enthält den P3-Abbruchfehler gegen die kinetische Lösung und ist
deshalb nur als Vergleich innerhalb der Tabelle aussagekräftig.

## Grenzen

- Nur uniforme kartesische Gitter auf `[0,1]^d`. Konstante isotrope Streuung
  (`--sigma-s`) und Absorption (`--sigma-a`) werden seit 10.11.2026 exakt
  integriert: `a_lm *= exp(-τ(σ_a + σ_s[l≥1]))`, je ein halber Schritt vor und
  nach SSPRK3 zusammen mit dem Filter (Strang). Erhält Dichte (Streuung) und
  M1-Kegel. Quellen, Materialsprünge und `lattice` fehlen weiterhin.
- Line Source (Garrett–Hauck) läuft für Element-WENO exakt skaliert auf dem
  Einheitsquadrat (`σ_s=3`, `T=1/3`, Filter ×3); Auswertung in
  `scripts/line_source/analyze.py`.
- Vergleich mit Variante B und Filter siehe
  `results/element_weno/filter_comparison_20261109/REPORT.md`. Dort ist p>1 auf
  16² und 32² schlechter als p=1; die Tabelle oben gilt nur für 96 Intervalle.
  Feinere Gitter: `results/element_weno/large_comparison_20261109/REPORT.md`.
- Mit Gewicht 1e-3 ist der Sensor im Crossing fast wirkungslos; `ρ_max` und L²
  liegen praktisch bei `γ=1`.
- Der Limiter ist in 41–43 % der Element-Stufen aktiv, mit Faktor bis 0.
- Puffer pro Element brauchen `Elemente × Knoten × q` Doubles pro Stufe. Für
  hohe PN-Ordnung auf feinen Gittern ist das zu teuer.

## Laufzeit

Optimierungen vom 10.11.2026: paralleles Sammeln über eine Knoten-Inzidenzliste,
zeilenweise LDLᵀ-Linienlösungen für alle PN-Komponenten, parallele Randfaces mit
einmal ausgewerteter konstanter Einstrahldichte, vorab summierte Viskositäten.
128²-Knoten, P7, p=2: 0.411 → 0.227 s/Schritt (1 Thread), 0.213 → 0.095 s/Schritt
(8 Threads); Ergebnis gleich bis auf 1e-13. Zum Vergleich braucht die
Massenkorrektur mit Produktionseinstellungen auf 128²/P7 etwa 0.025 s/Schritt (8 Threads).
Eine Nachmessung am 11.11.2026 ergab dafür 0.049 s/Schritt (129² Knoten, P7, 8 Threads, Standardwerte von `m1pn_fast`).

Optimierungen vom 11.11.2026 (nach den Randkorrekturen, algebraisch identische Umformungen):

- `M⁻¹C_d = I ⊗ (M₁⁻¹C₁)` auf dem Tensorgitter: der projizierte Gradient `g_d` ist ein
  Linienoperator entlang `d` (ein bandiertes Produkt, eine Linienlösung) statt
  Elementschleife, Sammeln und `d` Linienlösungen.
- Advektionsanteil der Hochordnungsrate direkt als `-Σ_d g_d A_d`; die Felder `a A_d`
  und die Element-Residuen entfallen. Für `G = F^H - F^L` werden nur die vier
  niedrigen Momente von `a A_d` gebildet.
- `s_h` wird knotenweise ohne Elementpuffer gesammelt (Zeilenkerne ohne Kopien und
  Allokationen); pro Element bleiben nur die vier projizierten Momente für den Limiter.
- Rand: der PN-Flächenintegrand ist affin in (Spur, Datum),
  `(A_n - llf_homogeneous) u - llf_datum d`; pro Face eine Flächenmassenmatrix statt
  einer Onsager-Lösung pro Quadraturpunkt. C6-Ghost einmal pro (Seite, Randknoten).
- WENO-Sensor, Limiter-Schleife und SSPRK3-Stufen ohne Allokationen bzw. Kopien.

128²-Knoten, P7, p=2, Crossing, CPU-Zeit mit einem Thread: 0.348 → 0.149 s/Schritt
(gemessen auf einer ausgelasteten Maschine, daher CPU- statt Wandzeit). Mit 8 Threads
sinkt die Wandzeit eines 30-Schritte-Laufs einschließlich Aufbau von 13.3 auf 6.0 s.

Abweichung der Dichte gegenüber dem Stand davor in 12 Vergleichsläufen (p=1..4,
1D/2D/3D, `--legacy-boundary`, `--symmetric-residual`, Filter, ohne WENO/Stabilisierung):
1e-15 bis 5e-10. Ausnahme p=3, 16², `--weno-linear .2`, T=0.32: 9e-4. Dieser Fall
verstärkt Rundungsunterschiede; derselbe alte Code weicht ohne FMA-Kontraktion
(`-ffp-contract=off`) von sich selbst um 1.2e-3 ab.

## Zeitschritt und SSPRK(4,3) (11.11.2026)

**Hochordnungsschranke.** Bisher galt `high_order_step = 0.9·√3/ρ`, also die Grenze von SSPRK3 auf der
imaginären Achse. Das ist um den Faktor 1.6 zu vorsichtig. Potenziteration auf dem Verstärkungsoperator
`R(dt·L)` des periodischen Hochordnungsziels (p = 1..4, P3/P7, γ = 0 und 1) zeigt: Die betragsgrößten
Eigenwerte sind dissipativ und liegen nahe der negativen reellen Achse. SSPRK3 kippt genau bei `2.5127/ρ`,
der Grenze von `R₃` auf der reellen Achse. Neu gilt `high_order_step = 0.9·c/ρ` mit `c = 2.5127` (SSPRK3)
bzw. `c = 5.1495` (SSPRK(4,3)). Der Test `high_order_stability` in `tests/element_weno.cpp` prüft beides:
kein Wachstum bei `high_order_step`, Wachstum beim 1.25-Fachen der Grenze.

Hinweis: Die Potenziteration läuft auf dem offenen Gitter ohne Randterme. Dieser Operator ist selbst nicht
dissipativ (scheinbares Wachstum auch bei kleinem dt). Für ρ spielt das keine Rolle, ein Stabilitätstest muss
aber periodisch rechnen.

**SSPRK(4,3)** (Kraaijevanger; Spiteri und Ruuth), `--time-integrator ssprk43` (`m1pn_fast`) bzw. `--ssprk43`
(`m1pn_element_weno`): vier Euler-Stufen der Länge dt/2, SSP-Koeffizient 2, dritte Ordnung. Jede Stufe ist ein
limitierter Euler-Schritt wie bisher; die Realisierbarkeit gilt daher für `dt ≤ 2·dt_bound`
(`realizability_step()`). Filter, Streuung und Absorption bleiben halbe Schritte vor und nach dem Zeitschritt.

**Overhead.** Der serielle Test `a.allFinite()` zu Beginn von `rate()` und die seriellen Eigen-Linearkombinationen
der Stufen kosteten mit 8 Threads etwa 25 % der Laufzeit. Beide laufen jetzt zeilenparallel.

Crossing, `T = 0.32`, 129² Knoten (p = 2, 64²), Gesamtstand mit Randkorrekturen und den Optimierungen
vom 11.11. (Abschnitt oben), Standard-`dt`. Die Massenkorrektur (Q1, 128², SSPRK2, Standardwerte von `m1pn_fast`)
steht zum Vergleich darunter.

| Variante | PN | Threads | Schritte | dt | Rechenzeit | L1 gegen Kinetik | Abweichung |
|---|---|---:|---:|---:|---:|---:|---:|
| SSPRK3, alte Schranke, Code vor dieser Änderung | P7 | 8 | 394 | 8.14e-4 | 17.2 s | 0.11169 | – |
| SSPRK3, gleiches dt | P7 | 8 | 394 | 8.14e-4 | 13.2 s | 0.11169 | 1e-11 |
| SSPRK3, neue Schranke | P7 | 8 | 271 | 1.18e-3 | 9.2 s | 0.11169 | 5e-6 |
| SSPRK(4,3) | P7 | 8 | 133 | 2.42e-3 | 5.9 s | 0.11169 | 2e-5 |
| Massenkorrektur | P7 | 8 | 325 | 9.86e-4 | 16.0 s | 0.12439 | |
| SSPRK(4,3) | P7 | 1 | 133 | 2.42e-3 | 22.7 s | 0.11169 | |
| Massenkorrektur | P7 | 1 | 325 | 9.86e-4 | 59.5 s | 0.12439 | |
| SSPRK3, alte Schranke, Code vor dieser Änderung | P3 | 8 | 393 | 8.16e-4 | 10.8 s | 0.50633 | – |
| SSPRK3, neue Schranke | P3 | 8 | 271 | 1.18e-3 | 6.0 s | 0.50635 | 7e-5 |
| SSPRK(4,3) | P3 | 8 | 132 | 2.43e-3 | 3.9 s | 0.50634 | 1e-4 |
| Massenkorrektur | P3 | 8 | 325 | 9.86e-4 | 14.4 s | 0.52416 | |

Abweichung: relative L1-Differenz der Dichte zum alten Lauf. Sie ist 3 bis 4 Größenordnungen kleiner als der
Fehler gegen die kinetische Lösung; der Zeitfehler spielt keine Rolle. Mit Streuung und Absorption
(P3, 32², σ_s = 2, σ_a = 0.5) stimmt der Stand bei gleichem dt bis auf 4e-12 mit Merge 60a9343 überein.
Mit dem Standard-`dt` ändern sich alle bisherigen SSPRK3-Läufe (größerer Schritt); mit `--dt` lassen sie sich
reproduzieren.

## Winkelreduktion über Parität (11.11.2026)

In 1D und 2D hängt die Lösung nicht von z (2D) bzw. y, z (1D) ab. Die Transportmatrizen der aufgelösten
Richtungen (`A_x`, in 2D auch `A_y`) und die Randmatrizen für Normalen in dieser Ebene koppeln Momente
verschiedener Parität unter `z → −z` (1D zusätzlich `y → −y`) nicht. Sind Anfangs- und Einstromdaten gerade,
bleiben alle ungeraden Momente für alle Zeiten exakt null. Gemessen im Crossing ohne Reduktion: |a| < 1e-17.

`ElementWenoOptions::reduce_angular` rechnet daher nur mit den geraden Momenten. Die M1-Momente (l ≤ 1)
bleiben immer erhalten, also auch J_z bzw. J_y, die dann null sind. Closure, Limiter und Bar States
bleiben dadurch unverändert. Aktive Momente:

| | volle Basis | 2D: `(N+1)(N+2)/2 + 1` | 1D: `Σ_l (⌊l/2⌋+1) + 2` |
|---|---:|---:|---:|
| P7 | 64 | 37 | 22 |
| P11 | 144 | 79 | 44 |
| P15 | 256 | 137 | 74 |

- **Paritätsbestimmung:** aus den Werten der Kugelflächenfunktionen an gespiegelten Richtungen, also
  unabhängig von der Konvention der Basis.
- **Absicherung beim Aufbau:** `require` prüft, dass `A_d` und `A_n − llf_homogeneous` keine Einträge zwischen
  gerader und ungerader Klasse haben (Toleranz 1e-12).
- **Einstrahlung:** `dropped_inflow(t)` misst den ungeraden Anteil von `llf_datum·d` auf allen Randpunkten.
- **`m1pn_fast --angular-reduction auto|on|off`** (Standard `auto`): `auto` schaltet die Reduktion in 1D/2D
  ein, wenn die Anfangskoeffizienten aller Knoten und der Einstromterm zu neun Zeiten in [0, T] gerade sind,
  sonst bleibt die volle Basis (Hinweis auf stderr). `config.json` enthält `angular_reduction`,
  `active_moments` und die gemessenen ungeraden Anteile. Die Ausgabe (`final.csv`) hat immer die volle Basis.
- **`m1pn_element_weno`:** Reduktion standardmäßig an (beide Probleme sind gerade, die Einstrahlung wird
  geprüft), abschaltbar mit `--no-angular-reduction`.

Tests (`tests/element_weno.cpp`, `angular_reduction`): `rate()` und drei SSPRK-Schritte stimmen auf geraden
Daten mit dem vollen Schema überein (≤ 2.5e-14), in 1D periodisch, 2D periodisch, 2D offen mit Einstrahlung
und 2D mit Filter und Streuung. Ein um 0.4 rad aus der Ebene gekippter Strahl wird als ungerade erkannt.

Crossing, T = 0.32, p = 2 auf 64², SSPRK(4,3), 6 Threads:

| PN | aktive Momente | ohne Reduktion | mit Reduktion | Faktor | Speicher |
|---|---:|---:|---:|---:|---|
| P7 | 37 / 64 | 5.84 s | 4.74 s | 1.23 | 89 → 64 MB |
| P11 | 79 / 144 | 10.97 s | 5.92 s | 1.85 | 180 → 118 MB |
| P15 | 137 / 256 | 21.25 s | 10.03 s | 2.12 | 351 → 201 MB |

Die Dichte unterscheidet sich um 2e-6 bis 1e-5, weil die Potenziteration für `high_order_step` im
reduzierten Raum ein um bis zu 7e-4 anderes ρ schätzt. Mit gleichem `--dt` sind es 4e-11 (P11). Für gerade
Daten genügt die Stabilität im geraden Teilraum, das kleinere ρ ist also korrekt. Der glatte 1D-Fall
(P7, p = 3) liefert mit und ohne Reduktion denselben L2-Fehler auf 10 Stellen.

## Spiegelränder und Halb-/Viertelgebiet (11.11.2026)

`ElementWenoOptions::origin`, `length` erlauben das Gebiet `origin + [0, length]^d`. `mirror[2d + side]`
macht eine Seite zum Spiegel (spekulare Reflexion). Für spiegelsymmetrische Lösungen ist das Galerkin-Verfahren
auf dem halben Gebiet mit symmetrischen Testfunktionen äquivalent zum vollen Gebiet. Daraus folgt:

- **Keine Randterme an der Spiegelfläche.** Spiegelknoten haben die halbe Masse und den halben Beitrag.
- **Ungerade Momente auf der Spiegelebene sind null.** Für die Momente, die unter `x_d → −x_d` ungerade sind,
  wird die konsistente Masse entlang `d` mit Dirichlet-Zeile am Spiegelknoten gelöst (eigene LDLᵀ-Faktorisierung
  ohne diesen Knoten), für die geraden natürlich. Beim projizierten Gradienten `g_d` ist die Parität vertauscht.
  Tangentiale Gradienten brauchen nichts, weil die ungeraden Komponenten am Spiegel schon null sind.
- **Limiter und Low-Order** bleiben elementweise. An Spiegelknoten wird die Rate auf die geraden Momente
  projiziert. Das Weglassen von `J_d` verkleinert |J| und erhält den Kegel; die Realisierbarkeit gilt also
  weiter mit derselben Schranke `dt_bound`.
- **WENO-Sensor:** Der Nachbar hinter dem Spiegel ist das eigene, in `d` gespiegelte Element.

`m1pn_fast --mirror-reduction auto|off` (Standard `auto`): Ist das Netz offen, die Einstrahlung `vacuum`,
die Elementzahl gerade und sind die Anfangsdaten an allen Knoten des vollen Gitters in jeder Richtung um ½
spiegelsymmetrisch (`reflect_pn`, Toleranz 1e-14 relativ), rechnet `m1pn_fast` auf `[½, 1]^d` mit halber
Elementzahl. Das betrifft Line Source (Viertel) und Plane Source (Hälfte), nicht aber das Crossing (Einstrahlung).
`initial.csv`/`final.csv` werden auf das volle Gitter zurückgespiegelt, `final_density_mass` gilt für das volle
Gebiet. In `config.json` stehen `mirror_reduction`, `mirror_asymmetry` und `computed_elements_per_axis`.

Tests (`mirror_quarter`): Viertelgebiet gegen Vollgebiet [0,1]² mit symmetrischen, den Limiter aktivierenden Daten:
gleiche `dt_bound` (2e-16), Rate auf allen gemeinsamen Knoten bis 8e-15, vier SSPRK-Schritte bis 2e-16, Masse
genau ein Viertel. Auch mit Winkelreduktion, Streuung und SSPRK(4,3).

| Fall | voll | Spiegel | Faktor | Abweichung bei gleichem `--dt` |
|---|---:|---:|---:|---:|
| Line Source (GHS), P7, 64 Intervalle | 0.40 s | 0.15 s | 2.7 | 1e-12 |
| Line Source (GHS), P11, 128 Intervalle | 5.24 s | 1.34 s | 3.9 | – |
| Plane Source 1D, P7 | 0.056 s | 0.044 s | 1.3 | 5e-12 |

Ohne `--dt` schätzt die Potenziteration auf dem kleineren Gebiet ein etwas anderes ρ (Abweichung der Dichte
dann 1e-4).

## Hardware (11.11.2026)

Element-WENO ist durch die Speicherbandbreite begrenzt; ab N ≈ 11 dominieren die Phasen, die mit der Zahl der
Momente wachsen. Messungen:

**Arbeitsplatz** (i5-12500, 6 Kerne mit Hyperthreading), P11, 193² Knoten: 4 Threads 4.31 s, **6 Threads
3.45 s**, 8 Threads 4.34 s, 12 Threads 4.73 s.

**nx-01…06** (2 × AMD EPYC 9354, Zen 4, je Sockel ein NUMA-Knoten mit 32 Kernen, 512 MB L3, 1.5 TB RAM),
`OMP_PLACES=cores`:

| Threads (close) | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| P11, 257² Knoten, 28 Schritte | 17.9 s | 5.1 | 2.86 | 1.89 | **1.48** | 1.85 |
| P15, 513² Knoten, 11 Schritte | – | – | 6.89 | 3.96 | **2.74** | 2.77 |

Der zweite Sockel bringt einem einzelnen Lauf nichts, auch nicht bei Feldern weit über L3 (P15/513²: 288 MB pro
Feld), weil der Hauptthread die Felder zuerst berührt und der Speicher deshalb auf Sockel 0 liegt. `spread` über
beide Sockel ist langsamer (32 Threads: 3.26 statt 2.74 s). Mehrere gepinnte Läufe pro Knoten mit lokalem Speicher
(`numactl --physcpubind … --localalloc`) beeinflussen sich dagegen kaum:

| Packung auf einem nx-Knoten | P11/257²: Zeit je Lauf | Durchsatz | P15/513²: Zeit je Lauf | Durchsatz |
|---|---:|---:|---:|---:|
| 1 × 32 Threads | 1.48 s | 1.48 s/Lauf | 2.74 s | 2.74 s/Lauf |
| 2 × 32 (je Sockel) | 1.49 s | 0.75 s/Lauf | 2.75 s | 1.37 s/Lauf |
| 4 × 16 (je Viertel) | 1.95 s | **0.49 s/Lauf** | 4.3 s | **1.08 s/Lauf** |

Umgesetzt:

- **Threadzahl** (`src/hardware.hpp`): Ohne `--threads` und ohne `OMP_NUM_THREADS` nehmen `m1pn_fast` (nur
  Element-WENO) und `m1pn_element_weno` die physischen Kerne der erlaubten CPUs im NUMA-Knoten der ersten
  erlaubten CPU: 6 auf dem Arbeitsplatz, 32 auf einem ganzen nx-Knoten, 16 bei `--cpus-per-task=16`. Bei
  gesetztem `OMP_PLACES` bindet libgomp den Hauptthread schon vor `main` an einen Kern; die erlaubten CPUs kommen
  dann aus der OpenMP-Platzliste. `config.json` enthält `thread_policy`. Pfad B behält das OpenMP-Maximum.
- **Packung** (`scripts/element_weno/pack_node.py`): Aufgabenliste `NAME<TAB>KOMMANDO`, Slots aus physischen
  Kernen innerhalb eines NUMA-Knotens (Standard 16 Kerne), `numactl --physcpubind --localalloc`,
  `OMP_PLACES=cores`, `OMP_PROC_BIND=close`, gemeinsame Warteschlange.
  `nx/batches/element_weno_pack.sbatch` belegt einen nx-Knoten exklusiv mit vier Slots
  (`sbatch --export=ALL,TASKFILE=… nx/batches/element_weno_pack.sbatch` aus dem Repository-Wurzelverzeichnis).
- Die Binary (x86-64-v3, `-mtune=znver3`) läuft auf beiden Maschinen. Ein eigener Zen-4-Build (AVX-512) ist
  nicht gemacht; bei bandbreitenbegrenzten Phasen ist wenig davon zu erwarten.

Offen: NUMA-gerechtes paralleles Anlegen der Felder, damit ein Einzellauf beide Sockel nutzen kann. Für
Kampagnen ist die Packung einfacher und schon effizienter.

## Bauen und ausführen

    cmake --build tmp/build --target m1pn_element_weno m1pn_element_weno_tests
    tmp/build/m1pn_element_weno_tests
    tmp/build/m1pn_element_weno --problem crossing-study --dimension 2 \
      --pn-order 3 --degree 3 --elements 32 --time .32 --cfl .5 --weno-linear .2

## Randkorrekturen (Branch fix/element-weno-boundary-c1-c6)

Vorzeichenlimiter am Rand, nach außen gerundeter Bar-State, positive Vollkugel-Ghost-Zustände, skaleninvarianter WENO-Sensor und die symmetrische Residualform
sind in [BOUNDARY_FIXES.md](BOUNDARY_FIXES.md) mit Herleitung und Vorher/Nachher-Tests dokumentiert. Option `--legacy-boundary` reproduziert Commit 8efe517.
