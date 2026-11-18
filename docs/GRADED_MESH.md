# Abgestufte Tensorproduktgitter (Plan, 17.11.2026, Branch `feature/graded-mesh`)

Ziel: lokale Verfeinerung wie im abgestuften Dreiecksgitter von KiT-RT (fein an der Quelle, außen gröber), ohne die
Tensorstruktur aufzugeben. Rechteckgitter mit variabler Elementbreite je Achse (Ebenen x_0 < x_1 < … je Achse),
Elemente sind Quader, Q_p-Bernstein wie bisher.

## Was bleibt

- Globale Operatoren sind Tensorprodukte der 1D-Matrizen je Achse: M = M_x ⊗ M_y ⊗ M_z, C_d = … ⊗ C_d,1D ⊗ …,
  also M⁻¹ C_d = I ⊗ (M_d⁻¹ C_d,1D) ⊗ I. Gilt mit variablen Breiten weiter (die 1D-Massen werden mit der Breite
  jedes 1D-Elements assembliert, C_1D ist breitenunabhängig). Linienlöser, `natural_`/`dirichlet_`-Faktoren,
  MPI-Ketten bleiben strukturell gleich.
- Skalierte Sobolev-Halbnorm des WENO-Sensors: auf dem Referenzelement definiert, bleibt unverändert (mit
  achsenweiser Skalierung h_d^{α_d} ist sie für Quader exakt die Referenzhalbnorm).

## Was sich ändert (Bestandsaufnahme `src/element_weno.cpp`, Stand 4cf885d)

Heute gibt es **einen** Satz Elementdaten für alle Elemente (gleich große Würfel):

| Daten | Ort | neu |
|---|---|---|
| `mass_`, `convection_[d]`, `stiffness_`, `viscosity_ij_` | Konstruktor (~350–445) | je **Größenklasse** (Breitentripel); M = h_x h_y h_z M̂, C_d = (∏_{k≠d} h_k) Ĉ_d, K = Σ_d (∏_{k≠d} h_k / h_d) K̂_d; d_ij aus den skalierten c_ij (nicht skalar skalierbar bei Anisotropie) |
| `pair_table_`, `total_viscosity_`, `stiffness_rows_`, `convection_t_rows_` | Limiter, Stabilisierung, GPU-Setup | je Klasse, Element → Klassenindex |
| `viscosity` (= scale · h / 2p) | Stabilisierung | je Klasse (charakteristische Länge, für Würfel = h) |
| `extension_[d][side]` | WENO-Kandidaten vom Nachbarn | je Achse, Seite und Breitenverhältnis Nachbar/Element |
| `rules_[side]` (Randquadratur, `rule.mass`, physikalische Gewichte mit `area`) | Rand, Bar-States, GPU-Randpfad | Referenzgewichte; je Randfläche Faktor ∏ tangentiale Breiten |
| `lumped`, `rates`/`dt_bound`, `fcs_node_mass_` | Konstruktor | aus den Klassen-Zeilensummen |
| `element_origin`, `node_position`, `evaluate` (Punktsuche `(x − origin)/h`) | Geometrie | Ebenenlisten, Präfixsummen, Binärsuche |
| Material-, FCS- und Randquadratur (`x[d] += h * ref`, Gewicht `h^d`) | ~600, ~670, ~919, ~1748, ~2685, ~2774 | Breiten des Elements |
| `mass_1d_` (MPI: Band und Tripel mit `s_.h * m1`) | 1D-Assemblierung | Breite je 1D-Element |
| GPU (`element_weno_gpu.cpp/.cu`) | Paartabelle, `total_viscosity`, Steifigkeitszeilen, Randzeilen | Klassenindex je Element, Tabellen je Klasse |

Zeitschritt: global durch das kleinste Element (wie bei KiT-RT); der Gewinn kommt aus der Elementzahl.

## Gitterwahl

`--cells-per-unit n_0,n_1,…` je Achse (Elemente pro Einheitsintervall des Falls, z. B. Lattice
`32,64,128,256,128,64,32`): Material- und Quellgrenzen liegen immer auf Elementgrenzen, Spiegelsymmetrie bleibt bei
symmetrischen Listen erhalten. Größenklassen: wenige Breiten je Achse ⇒ wenige Klassen.

## Stufen

| Stufe | Inhalt | Abnahme |
|---|---|---|
| A | CPU (OpenMP), 1D–3D, ohne MPI/GPU: Klassen, 1D-Assemblierung, Quadraturen, Punktsuche, dt, Spiegel | gleichmäßiges Gitter bitgleich zu 4cf885d; glatter Fall konvergiert auf abgestuftem Gitter mit Ordnung wie gleichmäßig; Lattice abgestuft gegen MC |
| B | MPI | `tests/fast_mpi_equivalence.sh` + abgestufter Fall |
| C | GPU | `tests/gpu_validate.sh` + abgestufter Fall |

Entscheidungshilfe (nicht für den Paper-Vergleich): KiT-RT auf abgestuftem gegen gleichmäßiges Dreiecksgitter mit
gleicher Zellzahl, Felder gegen MC (`external/kitrt_runs/diag.sbatch`).

## Stufe A: Ergebnisse (18.11.2026, `nx/tmp/graded/v4`, CPU 64 Threads, lattice-paul mit FCS, t = 3,2)

Zwei Fehler auf dem Weg: (1) eine Viskosität je Klasse ~ w_max machte die Steifigkeit gestreckter Elemente um
w_max/w_min zu stark und kürzte den Hochordnungsschritt entsprechend (GA: 7977 statt ~1050 Schritte), jetzt ν_d ~ w_d
je Achse; (2) `fast_main --cells-per-unit` überschrieb `counts[0]` vor der y-Ausdehnung (y-Achse falsch lang, mit
konstanter Liste unsichtbar). Kontrolle in v4: Verfeinern nur der Außenzelle lässt die ungestreuten Integrale bitgleich.

| Gitter | Elemente | Zeit P11 / P21 | Absorption % P11 / P21 | Feld L1 P11 / P21 |
|---|---|---|---|---|
| U364 | 132k | 128 / 180 s | +0,455 / +0,317 | 3,8e-3 / 2,6e-3 |
| GA 21,32,64,128,64,32,21 | 131k | 200 / 283 s | +0,353 / +0,173 | 3,2e-3 / 1,7e-3 |
| U560 | 314k | 346 / 509 s | +0,356 / +0,181 | 3,0e-3 / 1,6e-3 |
| GB 32,64,128,128,128,64,32 | 332k | 431 / 601 s | +0,322 / +0,127 | 2,8e-3 / 1,2e-3 |
| U896 | 803k | 1043 / 1612 s | +0,319 / +0,124 | 2,8e-3 / 1,2e-3 |

GB erreicht U896 (alle QoIs und Feld) in 37 % der Zeit, GA erreicht U560 in 56 %. Kosten je Element und Schritt
abgestuft ~1,2 µs gegen 0,8 µs gleichmäßig (Klassenzugriff; Optimierungspotential).
