# Element WENO auf der GPU (experimentell, 15.11.2026)

Branch `experimental/gpu-a100` (von `feature/element-weno-mpi`, `afb1c55`), Option `--gpu` von
`m1pn_element_weno`. Build nur mit `-DM1PN_ELEMENT_ONLY=ON -DENABLE_MPI=OFF -DM1PN_ELEMENT_GPU=ON` und CUDA
(`/sfw/cuda/12.6`, Zielarchitektur sm_80). Ziel: A100 80 GB PCIe auf `tardis` (Slurm: `--constraint=ampere
--gres=gpu:1`, 16 Kerne EPYC 7343 mit SMT; `-w tardis` wird von Slurm abgelehnt, `--constraint=ampere` geht).

## Aufbau

Der Zustand bleibt während der ganzen Zeitschleife auf der GPU. Pro Stufe:

| Teil | Wo | Bemerkung |
|---|---|---|
| Closure (Levermore), Dichtespalte | GPU | fusioniert in den letzten Kernel der Vorstufe |
| WENO-Gewichte | GPU | ein Thread pro Element |
| Gradienten `M^-1 C_d a` | GPU | Konvektion im Vorwärtssweep, gebänderte LDL^T in natürlicher Ordnung |
| Stabilisierung (gesammelt) | GPU | 2D Q1: Streifen-Kernel, 3x3-Fenster in Registern |
| PN-Randlast (Teil A) | GPU (seit 16.11.2026, Block 4) | Spurkernel, cuBLAS-DGEMM je Seite, Knotensumme; Datum-Anteil `D B_datum` vom Host; `M1PN_GPU_BOUNDARY_HIGH=0`: Host |
| Massenlösung, Fluss der Momente | GPU | |
| Limiter | GPU | ein Thread pro (Element, Knoten), Elementsummen per Warp-Shuffle in Host-Reihenfolge |
| Ghost (Positivitätsprojektion), Face-Zustände (Teil B) | Host | überlappt mit der GPU-Arbeit |
| Randkorrekturen | Host | nur Randknoten |
| SSP-Kombination, Kegel-Nudge | GPU | PN-Zeilen vor den Randkorrekturen, M1-Zeilen danach |

Der Rand bleibt auf dem Host (Newton-Verfahren der Projektion, Inflow-Funktionen). Er braucht nur die Randzeilen
(512²: 2052 Knoten), die per Pinned-Memory und Events asynchron übertragen werden. Inflow-Daten hängen nur von der
Zeit ab und werden für die nächste Stufe vorbereitet, während der Host auf die GPU wartet.

Exaktheit: Das exakte M1-Kegelprädikat ist auf der GPU exakt (Filter wie auf dem Host, sonst fehlerfreie Produkte
und Shewchuk-Expansion nach exakter Zweierpotenz-Skalierung). Pfade, die auf dem Host `long double` brauchen
(Levermore nahe am Strahl, stabile Basis-Bar), werden markiert und auf dem Host mit dem Originalcode nachgerechnet
(in den Läufen hier nie aufgetreten). Gleiche Ergebnisse wie die CPU bis auf Rundung: andere Summationsreihenfolge,
FMA, die Linienfaktoren in natürlicher statt AMD-Ordnung. Ein GPU-Lauf ist deterministisch (drei Läufe bitgleich).

Nicht unterstützt (Abbruch mit Meldung): MPI, symmetrisches Residuum, M1-Target,
`projection_high_order/hybrid/diagnostics`, äußere Zustände (`exterior_state`). Spiegelseiten und periodische Gitter
seit 16.11.2026 (Abschnitt „Spiegelreduktion und periodische Gitter“).

## Laufzeit

Crossing-Studie (Breite 0,1, Hintergrund 0,01), Q1, P7, T = 0.32, automatisches dt (SSPRK3). Gesamt = Wandzeit
des Prozesses einschließlich Aufbau und Auswertung, Schleife = `wall_seconds`.

512² Elemente (1443 Schritte):

| Lauf | Gesamt | Schleife | GPU schneller (gesamt / Schleife) |
|---|---|---|---|
| NX, 3 Server (6 Ränge x 32 Threads, MPI-Branch) | 49,3 s | 46,0 s | 3,6x / 4,2x |
| NX, 1 Sockel (32 Threads) | – | 88,9 s | – / 8,2x |
| tardis nur CPU (16 Kerne, 32 Threads) | 157,5 s | 154,1 s | 11,5x / 14x |
| **tardis A100** | **13,7 s** | **10,85 s** | |

Drei GPU-Läufe: Schleife 10,84 / 10,85 / 10,87 s (bitgleich), Aufbau 1,95 s (davon Spektralradius auf der GPU
0,3 s), Auswertung 0,45 s.

1024² Elemente (2886 Schritte): NX 3 Server 223,5 s gesamt (Schleife 216,2 s), A100 87,5 s (82,0 s), also 2,6x.
Die 3 NX-Server skalieren mit der Größe besser (Kette und Rand fallen weniger ins Gewicht), die GPU braucht pro Stufe
etwa 4x so lang bei 4x so vielen Knoten.

Entwicklung der Schleifenzeit (512²): 22,6 s (erste Version) → 21,4 s (asynchrone Pipeline, Rand geteilt) →
15,8 s (Kernel) → 13,7 s (Inflow-Daten vorab) → 11,9 s (Limiter pro Knoten, Statistik per Warp-Reduktion) →
10,85 s (Fluss im ELL-Format, Stabilisierung mit vorskalierten Koeffizienten, Software-Pipelining der Linienlösungen).
Pro Stufe (2,5 ms) rechnet die GPU fast durchgehend: Linienlösungen 0,9 ms (latenzgebunden, nur 19k Threads pro
Richtung), Kombination 0,4 ms, Stabilisierung 0,32 ms, Limiter 0,27 ms, Rest 0,5 ms. Der Host-Rand (Projektion, Faces) braucht mit 32 Threads etwa
1 ms pro Stufe und liegt neben der GPU-Arbeit. Verworfen: Linienlösung im Shared Memory (8 Komponenten pro Block,
3x langsamer), Limiter mit Shared-Memory-Staging (langsamer), Chunk 32 statt 16 in den Linienlösungen, Kombination
der M1-Zeilen im Knoten-Kernel (unkoalesziert, langsamer). Streifenlänge der Stabilisierung: 16 (8/32/64 gleich oder
langsamer). Die Projektion auf die GPU zu bringen hilft nicht: der Host-Rand liegt vollständig neben der GPU-Arbeit.

## Abweichung

Dichte (sqrt(4 pi) a_0 an allen Knoten) gegen den CPU-Referenzlauf (deterministisch; NX-MPI, NX-Sockel und tardis-CPU
sind bitgleich), Maximum relativ zur maximalen Dichte und L² relativ:

| Fall | GPU gegen CPU | CPU ohne FMA gegen CPU |
|---|---|---|
| 64², T = 0.32 | 2,5e-12 / 5,9e-13 | |
| 512², T = 0.02 | 9,2e-12 / 6,0e-13 | 1,3e-11 / 1,6e-12 |
| 512², T = 0.08 | 1,9e-10 / 7,0e-12 | 1,8e-10 / 7,9e-12 |
| 512², T = 0.32 | 9,1e-5 / 7,3e-6 | 3,1e-5 / 4,6e-6 |
| 1024², T = 0.08 | 1,9e-9 / 5,7e-11 | |
| 1024², T = 0.16 | 1,5e-5 / 1,1e-6 | |
| 1024², T = 0.32 | 1,4e-2 / 8,0e-4 | 1,5e-2 / 7,9e-4 |

"CPU ohne FMA" ist derselbe CPU-Code mit `-mno-fma`, also dasselbe Verfahren mit anderer Rundung. Er weicht genauso
stark ab wie die GPU. Die Abweichung ist keine Eigenschaft des GPU-Codes, sondern die Rundungsempfindlichkeit des
Verfahrens bei diesem Problem: Sie wächst exponentiell mit der Zeit (Limiter- und Kegelentscheidungen kippen an
Schwellen), bei 1024² stärker als bei 512². Integrale Größen bleiben dabei stabil: kinetischer L²-Fehler 0,0116131 /
0,0116132 (512²) und 0,0123551 / 0,0123541 (1024²), Masse auf 8 Stellen gleich. Eine Abweichungsschranke von 1e-8 bei
T = 0.32 kann daher keine Änderung einhalten, die nicht bitgleich ist (auch kein anderer Compiler-Schalter).

Rohdaten und Skripte: `nx/tmp/gpu_20261115/` (nicht versioniert).

## Positivitätsprojektion auf der GPU (Branch `experimental/gpu-projection`, 15.11.2026)

Der Ghost (Spur an der Vollkugelregel, Identitätstest, semismooth Newton mit 4x4-LDL^T und Armijo, Warmstart pro Paar)
läuft auf der GPU, ein Warp pro Ghost-Paar, auf einem nicht blockierenden Stream niedriger Priorität parallel zu den
Volumenkerneln. Hüllzertifikat, augmentierte Regel und nicht konvergierte Fälle werden markiert und vom Host mit
`project_positive_moments` gelöst (in allen Läufen hier: 0 Fälle). Der Host behält nur die Face-Terme.
`M1PN_GPU_GHOST=0` schaltet zurück auf die Host-Projektion. Die Statistik stimmt mit dem Host überein (512²: 8883108
Aufrufe, 5309118 Identitäten, 3945191 statt 3945159 Newton-Schritte).

| Fall | Projektion auf dem Host | Projektion auf der GPU |
|---|---|---|
| 512², 32 Host-Threads | 13,7 s (Schleife 10,85 s) | 14,3 s (11,5 s) |
| 512², 4 Host-Threads | 18,6 s (15,0 s) | 15,3 s (11,5 s) |
| 1024², 32 Host-Threads | 87,5 s (82,0 s) | 90,3 s (84,7 s) |

Der Ghost-Kernel braucht etwa 0,16 ms pro Stufe GPU-Zeit. Mit 32 Host-Threads lag die Host-Projektion vollständig
neben der GPU-Arbeit, daher ist die GPU-Version dort 3-6 % langsamer. Sie lohnt sich, wenn der Host knapp ist (4
Threads: 21 % schneller) oder wenn die GPU später durch weitere Optimierung nicht mehr der Engpass ist. Abweichung
gegen die CPU wie zuvor auf dem Rundungsniveau des Problems (512², T = 0.32: 5,8e-5 / 4,9e-6; 1024²: 1,2e-2 / 8,6e-4).

## Validierung des Systemstands (16.11.2026, `feature/system-202611`)

Gleiches Binary, CPU gegen `--gpu` (tardis, A100), Dichte am Ende. Reproduzierbar mit
`sbatch tests/gpu_validate.sh [m1pn_fast] [Ausgabeordner]` (Standard `tmp/build_gpu/m1pn_fast`; Schwelle 1e-8 für
max- und L2-Abweichung, gleiche Schrittzahl, bei Lattice-FCS zusätzlich die Integrale; `GPU_VALIDATE_TIMING=1` ergänzt
den Zeitlauf 224², P15). Nachgerechnet 16.11.2026 (Job 160891, `93fa881`): alle Fälle bestanden, Werte wie unten.

| Fall | max rel | L2 rel |
|---|---:|---:|
| Crossing Q1 64², P7, T = 0.05 | 2.9e-12 | 1.0e-12 |
| Glatter Transport am Rand, Q2 32², P7 | 8.4e-16 | 2.2e-16 |
| Line Source mit FCS, Q1 128², P7, T = 0.02 | 1.0e-14 | 1.8e-14 |
| Lattice 56², P7, T = 0.2 | 1.4e-15 | 4.5e-16 |
| Lattice mit FCS 56², P7, T = 0.2 | 1.7e-10 | 6.1e-11 |
| Lattice 56², P35, T = 0.2 | 2.2e-15 | 6.4e-16 |
| Lattice 3D Q2 14³, P3, T = 0.1 | 3.0e-16 | 2.9e-16 |

Lattice-FCS: die Tabelle entsteht auf der GPU (Rundung von sin/exp auf dem Gerät, daher 1e-10). Lattice-FCS 224²,
P15, T = 3.2 auf der GPU: Tabelle 1.5 s (CPU ≈ 55 s), Löser 21.6 s, gesamt 27 s; Masse 2.072102 (−0.39 % gegen
Schotthöfer/Hauck, wie der CPU-Lauf auf NX).

## Spiegelreduktion und periodische Gitter (16.11.2026, Branch `feature/combine-202611`)

`m1pn_fast --gpu` rechnet jetzt auch mit `--mirror-reduction auto` (Halb-/Viertel-/Achtelgebiet) und auf periodischen
Gittern (`--periodic`). Gleiche Verfahren wie auf der CPU ([ELEMENT_WENO.md](ELEMENT_WENO.md), Spiegelränder):

- **Spiegel:** Zeilenlöser `k_line_solve` mit Dirichlet-Variante je Komponente (ungerade Momente; beim Gradienten
  die geraden) auf der Linie ohne Spiegelknoten, eigener bandförmiger LDLᵀ-Faktor (`dlower`), Wert am Spiegelknoten
  0; WENO-Nachbar hinter dem Spiegel ist das gespiegelte eigene Element (`reflected`); in den Stufenkombinationen
  (`k_combine`, `k_admissible`) ist die Rate der ungeraden Momente auf der Spiegelebene null (wie `rate()` auf der
  CPU). Ohne Spiegel laufen exakt dieselben Operationen wie vorher.
- **Periodisch:** LDLᵀ der zyklischen 1D-Masse in natürlicher Ordnung; Bandzeilen ohne Auffüllung, die letzten p
  Zeilen dicht („Pfeil“, `tail`). Eigener Kernel `k_line_solve_cyclic`, Konvektionsband mit zyklischen Spalten. Der
  2D-Q1-Streifenkernel der Stabilisierung setzt das offene Gitter voraus und wird periodisch durch den allgemeinen
  ersetzt. Bedingung: mindestens 2p + 2 Knoten je periodischer Richtung.

Validierung (`tests/gpu_validate.sh`, tardis, Job 160904, gleiches Binary; Schwelle 1e-8, alle 24 Fälle bestanden):

| Fall | GPU gegen CPU (max rel) | GPU Spiegel gegen GPU voll, gleiches dt |
|---|---:|---:|
| Line Source Q1 128², P7 (Viertel) | 3.8e-15 | 1.4e-12 |
| Line Source + FCS Q1 128², P7 | 8.0e-15 | 1.2e-14 |
| Line Source Q2 64², P7 | 3.2e-12 | 8.1e-12 |
| Lattice 56², P7 | 1.5e-15 | 1.1e-15 |
| Lattice + FCS 56², P7 | 1.4e-16 | 4.3e-10 (FCS-Tabelle auf der GPU) |
| Glatt 3D Q2 12³, P3, σ_s = 1 (Spiegel in y, z) | 1.3e-15 | 1.3e-15 |
| periodisch 1D Q3 32, P7 | 7.5e-16 | |
| periodisch 2D Q1 64², Q2 32², Q4 16², P7 | 7.5e-16 / 8.4e-16 / 1.9e-15 | |
| periodisch 3D Q2 8³, P3 | 8.4e-16 | |

Laufzeit auf der A100 (Job 160905, CFL 0.2, Löserzeit voll → Spiegel, Masse jeweils auf 8 Stellen gleich):

| Fall | voll | Spiegel (Viertel) | Faktor |
|---|---:|---:|---:|
| Lattice Q1 224², P15, T = 3.2 | 20.6 s | 11.7 s | 1.8 |
| Lattice + FCS Q1 224², P15 | 20.9 s | 11.9 s | 1.8 |
| Line Source Q1 512², P7, T = 0.3 | 33.7 s | 12.4 s | 2.7 |
| Line Source + FCS Q1 512², P7 | 33.1 s | 11.2 s | 3.0 |

Weniger als Faktor 4, weil das Viertelgebiet die GPU schlechter auslastet und der Host-Randpfad nicht mitschrumpft.

`lattice-3d` ist absichtlich asymmetrisch (Zelle (4,2,2) frei, (3,3,1) Absorber) und wird nicht gespiegelt.

## Host-Randpfad (Block 4, 16.11.2026, Branch `feature/block4-host-boundary`)

**Befund** (tardis, A100, `results/performance/host_boundary_20261116/`): In 3D und bei hoher PN-Ordnung lag fast die
ganze Löserzeit im Host-Randpfad, die GPU wartete. Lattice 3D Q1 56³, T = 1: P7 30.5 s Löserzeit, davon Phase B
(Face-Terme) 22.7 s und Phase A (PN-Randlast) 7.1 s; P15 163 s, davon 64 s bzw. 97 s. 2D-Lattice P35 224²: Phase A
24.4 von 25.6 s.

**Phase B (Face-Terme niedriger Ordnung), CPU und GPU-Host.** Die Face-Regel hat `p + N + 3` Gauß-Punkte je
Richtung (für die exakte Face-Masse des PN-Terms; Q1/P15 in 3D: 361 Punkte je Face). Die Face-Terme niedriger
Ordnung liefen über alle Punkte × Face-Knoten, mit je einem `Incoming`-Datensatz pro Punkt. Bei stationärem Einstrom
ohne Rückruf-Zustand, dessen Daten auf einem Face an allen Punkten bitgleich sind (Vakuum, isotrop), hängen alle
Summanden je Face-Knoten nicht vom Punkt ab; die Schleife reduziert sich auf die Face-Gewichte
W = Σ_Punkte w_p φ_n(x_p) (`face_uniform_`, `face_weight_` in `ElementWeno`, einmal beim Aufbau geprüft). Gilt für
`ElementWeno::add_boundary` (CPU, MPI) und `ElementWenoGpu::boundary_low`. Ortsabhängiger Einstrom (Strahlen,
Crossing) bleibt auf dem alten Pfad (bitgleich).

**Phase A (PN-Randlast) auf der GPU** (`gpu::boundary_high`): H_r = Σ_m M_side(n,m) x_node(f,m) · B_side − DB_r mit
einem Spurkernel, einem cuBLAS-DGEMM je Randseite (B_side = (A_n − llf_homogeneous)ᵀ, q × q) und der Summe der
Zeilen je Randknoten in fester Reihenfolge (wie auf dem Host); die ersten vier Spalten der Knotensummen (l ≤ 1) gehen
asynchron zurück für die Randkorrekturen. Der Datumsanteil wird bei zeitabhängigem Einstrom je Stufe hochgeladen.
cuBLAS ist Teil von CUDA 12.6 (keine Installation). Deterministisch (feste Konfiguration, ein Stream).

**Ergebnis** (Löserzeit, T = 1):

| Fall | vorher | nur Phase B | Phase A + B | Faktor |
|---|---:|---:|---:|---:|
| Lattice 3D 56³, P7 | 30.5 s | 2.0 s | 2.0 s | 15 |
| Lattice 3D 56³, P15 | 163.0 s | 13.2 s | 13.1 s | 12 |
| Lattice 2D 224², P35 | 25.6 s | 18.6 s | 18.6 s | 1.4 |

Phase B allein beschleunigt auch die Host-Phase A um etwa den Faktor 10 (vermutlich, weil die Punktdaten den Cache
und die Speicherbandbreite nicht mehr belegen); Phase A auf der GPU bringt danach in diesen Fällen wenig, entlastet
aber den Host. Auf der CPU (NX) halbiert Phase B die Randphase im 3D-Lattice (56³ P7 1.1 → 0.5 s).

**Abweichungen.** Nur Summationsreihenfolge: 3D-Lattice 3e-16, Lattice 2D 112² 5e-14, Line Source + FCS 2e-12 gegen
`standard-20261116-ewonly`; Crossing (ortsabhängiger Einstrom) bitgleich; `tests/gpu_validate.sh` alle Fälle PASS
(1e-16 … 4e-10). Ausnahme: 2D-Lattice ohne FCS bei hoher PN-Ordnung (P35 224², T = 1) verstärkt jede
Rundungsänderung auf 1e-4 (alter Stand gegen nur Phase B 2.4e-4, gegen Phase A + B 6.9e-5), wie schon CPU gegen GPU
(Abschnitt „Abweichung“). MPI 1/2/4 Ränge bleiben bitgleich (`tests/fast_mpi_equivalence.sh`).

**Nächster Engpass:** die Ghost-Projektion auf der GPU (`k_ghost`, P15 56³ ≈ 51 von ≈ 70 ms GPU-Zeit je Stufe,
`nsys_lattice3d_P15_E56.txt`); die Rechnung ist jetzt GPU-gebunden. Nebenbefund: Mit `--fcs analytic` in 3D wertet
die Ausgabe die exakte ungestoßene Dichte an allen Ausgabeknoten aus (≈ 5 ms je Punkt; 56³: 12 min nach 30 s Löser);
Kampagnen mit `skip_fields` sind davon nicht betroffen.

Die frühere Auffälligkeit der tardis-Lattice-Kampagne (P35 224² kaum schneller als 448²) tritt in Einzelläufen nicht
auf (T = 1: 25.6 s gegen 174.7 s); sie kam vermutlich von parallelen Aufgaben im damaligen Job.

## Adaptive PN: CPU und GPU gleich (20.11.2026, Branch `fix/gpu-adaptive-consistency`)

Die dynamische Tile-adaptive PN (`--adaptive-pn on`) wich auf der GPU von der CPU ab (Dichte max-rel bis 1.8e-6 bei
Graded Mesh + Broken Lines, 8e-8 im 2D-Lattice 56² P7 T = 0.2), nur bei gemischten Tile-Ordnungen. Uniform und bei
einheitlicher Ordnung (Toleranz 1 oder 1e-30) stimmten beide bis auf Rundung. Zwei GPU-Fehler:

1. **Stabilisierung an Ordnungsgrenzen.** Die Gradienten-Segmente reichen einen Knoten in das Tile niedrigerer Ordnung;
   dort sind Gradienten von Momenten, die an diesem Knoten inaktiv sind, ungleich null. Die CPU
   (`stabilization_residual`, `cm_of`) nimmt Beiträge eines Nachbarknotens nur für dort aktive Momente,
   `k_stabilization_adaptive` und `k_stabilization_strip_adaptive` nahmen alle. Jetzt maskiert (Tile-Ordnung des
   Nachbarknotens). Ebenso summieren `k_combine_adaptive` und `k_flux_rows` die Flusszeilen nur über Gradientenspalten
   unter `qpad` des Knotens, wie `ElementWeno::rate`.
2. **Closure nach dem Update.** Closure und Low-Flux der ersten Stufe eines Schritts entstehen schon in der letzten
   Stufe des vorigen (`closure_ready_`). Das Update nullt danach Momente geschrumpfter Tiles (l = 2 geht in den
   Low-Flux ein); `adapt_gpu` setzt jetzt `closure_ready_ = false`, wenn sich eine Tile-Ordnung ändert.

Gefunden mit `m1pn_adaptive_cpu_gpu_debug` (`tests/adaptive_cpu_gpu_debug.cpp`, nur GPU-Build): vergleicht Init-Update,
Gradienten, Massensolve, eine Stufe, PN-Momentenrate, `stab_low`, Low-Order-Last und mehrere Schritte operatorweise
(`ElementWenoGpu::debug_*`). Validierung: `tests/gpu_validate_adaptive.sh` (2D-Lattice Feature für Feature, Line Source
mit FCS + Mirror, Crossing Q1/Q2, 3D-Lattice inkl. FCS und Broken Lines + Split-Breaks; jeweils uniform und adaptiv),
alle 31 Fälle PASS (1e-17 … 1e-11, FCS-Fälle 4.9e-10 wie uniform); vorher 15 FAIL. `tests/gpu_validate.sh` PASS.
`adaptive_pn_dynamic` prüft CPU gegen GPU jetzt relativ zur uniformen Abweichung derselben Konfiguration (vorher 5e-4).

Offen (CPU und GPU gleich): Der Massensolve läuft je 32er-Moment-Chunk über Segmente, die auch Knoten enthalten, an
denen einzelne Momente des Chunks inaktiv sind; Einträge ungleich null an diesen Stellen der rechten Seite wirken auf
die aktiven Ergebnisse. Bei exakten Nullen harmlos, aber kein exakter Submatrix-Solve je Moment.

## Ghost-Spuren als DGEMM (20.11.2026, Branch `perf/solver-202611`)

**Befund.** Bei uniformem hohem N war nicht der Volumenteil, sondern die Ghost-Projektion am Vakuumrand der Engpass:
GBH P35 auf der A100 (Kampagne `tardis_paper_20261120`) verbringt 82 der 84 s Solverzeit mit Warten auf `k_ghost`.
Der Kernel bildet die PN-Spur je Ghost-Paar mit einem Warp, der die ganze Harmonischen-Matrix seiner Seite liest
(P35: ~5500 Regelpunkte × 666 Momente = 29 MB). Die Matrix wird so einmal je Paar aus dem globalen Speicher gestreamt
(GBH: ~2300 Paare, ~67 GB je Stufe). Der Speedup des adaptiven PN auf GBQ/GBH (10–15×) kam vor allem daher, dass der
adaptive Pfad die Ghost-Projektion auf dem Host mit Ordnungsstufen (3, 7, 15) und analytischem P1-Ghost rechnet.

**Änderung.** Je Seite werden die (gefilterten) Ghost-Zeilen gesammelt und `Y_s = H_sᵀ A_s` als eine cuBLAS-DGEMM auf
dem aux-Stream gebildet; `k_ghost` kopiert nur noch seine Spalte. Die Projektion selbst ist unverändert.
`M1PN_GPU_GHOST_GEMM=0` stellt die alten Spuren her, `M1PN_GPU_GHOST_WHY=1` gibt aus, warum der Device-Pfad
(nicht) gewählt wird (Verhältnis Randarbeit/Volumenarbeit, Schwelle 0.1; adaptiv und mit Interfaces immer Host).

**Lokal (RTX A2000, FP64-schwach).** GBH P35 bis t = 0.4: Solver 85.4 → 39.2 s (Ghost 63 → 21 s), Masse und
Absorptionsrate in allen Stellen gleich. Linienquelle Q1/128 P21 (FCS): 17.3 → 12.0 s, Masse 2e-13. lattice-3d 28³
P7: gleich. Auf der A2000 ist bei 224² P21 die Host-Projektion schneller (9.2 gegen 12.4 s); die Schwelle wird auf der
A100 neu bestimmt (`scripts/element_weno/tardis_perf_profile.sbatch`).

**A100 (Job 164267, `nx/tmp/tardis_perf_20261120`, Lattice wie Kampagne Block A, T = 3.2).**

| Lauf (uniform, Spiegel, FCS, gebrochene Linien + Split) | Solver alt [s] | Solver DGEMM [s] | Ghost-Warten alt → neu [s] |
|---|---:|---:|---|
| GBQ P35 | 20.26 | 2.79 | – → 1.5 |
| GBH P35 | 84.33 | 16.10 | 81.9 → 9.0 |
| GB P35 (Host-Ghost, Verhältnis < 0.1) | 94.81 | 94.48 | Host 47 s, unverändert |

Masse GBH P35: 1.29911036774 (DGEMM) gegen 1.29911036639 (alt), relativ 1e-9 (Summationsreihenfolge, Projektion
auf Toleranz). Uniform 896² P21 (t = 0.8, kein Spiegel): Host 74.5 s, Device mit DGEMM 78.9 s, Device alt 92.6 s; die
automatische Wahl (Host, Verhältnis < 0.1) bleibt dort richtig.

**Folge.** Der Abstand adaptiv–uniform schrumpft auf GBQ/GBH (P35: 2.79 gegen 1.32–1.98 s auf GBQ, 16.10 gegen
5.95–8.54 s auf GBH). Die adaptiven Läufe selbst nutzen den Host-Ghost mit Ordnungsstufen und werden nicht schneller.
Die Spektralradius-Schätzung (60 Potenziterationen des vollen Operators, einmal am Anfang) kostet auf GB P35 3.4 s.

## Profil und weitere Beschleunigung (20.11.2026, Branch `perf/solver-202611`)

**Kernel-Profil (nsys, A100, GB P35, T = 3.2, `nx/tmp/tardis_perf_20261120/nsys`).** Uniform (Device-Ghost) gegen
adaptiv (1e-4, Kachel 8, 28.8 % aktive Koeffizienten), Summe über den Lauf: Line-Solve 41.9 → 14.0 s (3.0×),
Strip-Stabilisierung 33.7 → 15.0 s (2.25×), Combine 16.2 → 6.1 s (2.65×); ideal wäre 3.5×. Die Low4-Kernel
(Momente c < 4, ein Thread je Linie und Moment) sind latenzgebunden (GB: ~2300 Threads je Achse, ~110 GB/s).
Die „GB-Anomalie“ (adaptiv nur 2× auf GB, 11× auf GBH) war keine: pro Zelle und Schritt ist die adaptive
Device-Arbeit auf allen Gittern ~2.4× kleiner; der große Faktor auf GBQ/GBH kam von der Ghost-Projektion (oben).

**Änderungen.**
1. Initialisierung: dünne Produkte in `Boundary` (D blockdiagonal in l, Normalenmatrix mit ~6 Einträgen je Zeile),
   die ungenutzte Matrix `homogeneous` nur mit `norms = true`; CUDA-Kontext auf einem Hilfsthread während des
   Schema-Aufbaus. Rundungsgleich (GBH P35: Masse 5e-11, Rate 3e-8 relativ).
2. Adaptiver Pfad: Low4- und High-Chunk-Line-Solves sowie die Strip-Stabilisierung auf Fork-Join-Streams
   (`M1PN_GPU_OVERLAP=0` seriell). Bitgleich.
3. Strip-Stabilisierung (uniform, low4, adaptiv): Zeilenzeiger statt 64-Bit-Indexprodukt je Ladevorgang
   (ncu A2000: Integer-Befehle 6.4e9 → 4.5e9, Warp-Befehle −18 %), `M1PN_GPU_STRIP_MINB` = 4/6/8 für die
   Launch-Bounds. Bitgleich.

**A100 (Job 164279, Block-A-Konfigurationen, gegen `tardis_paper_20261120`).**

| Lauf | gesamt alt → neu [s] | Init alt → neu [s] | Solver alt → neu [s] |
|---|---|---|---|
| GBQ P35 adaptiv 1e-4 | 8.5 → 6.3 | 6.4 → 4.4 | 1.65 → 1.53 |
| GBH P35 adaptiv 1e-4 | 16.7 → 14.3 | 8.3 → 6.2 | 7.32 → 7.07 |
| GB P35 adaptiv 1e-4 | 65.8 → 63.2 | 15.3 → 13.4 | 46.81 → 46.58 |
| GBQ P35 uniform | 26.9 → 7.4 | 6.3 → 4.2 | 20.26 → 2.79 |
| GBH P35 uniform | 93.6 → 23.1 | 8.1 → 6.0 | 84.46 → 16.11 |

Überlappung allein (`_serial` gegen Standard): GBQ 1.64 → 1.53 s, GBH 7.31 → 7.07 s, GB unverändert.

**Nicht geändert.** Spektralradius (60 Potenziterationen, bei 60 noch nicht konvergiert: GBH 768.7, +0.08 je
Iteration): weniger Iterationen oder ein anderer Startvektor ändern Δt und damit alle Ergebnisse; Gewinn 2–5 %.
Gemischte Genauigkeit nach ℓ: nur `k_combine` ist speichergebunden (~1.4 TB/s), Line-Solve ~0.55, Strip ~0.46 TB/s;
geschätzt 1.2–1.4× bei Umbau aller Kernel und des Speicherlayouts.

**Lokale Zeitschritte (Abschätzung, nicht umgesetzt).** Auf Tensorgittern ist der Schritt eines Elements durch
min(h_x, h_y[, h_z]) begrenzt; jedes Element, das in *einer* Achse im feinen Zentrum liegt, braucht den feinen
Schritt. Ideales LTS (Stufen 2^k) brächte als Arbeitsanteil: GB/GBH/GBQ 2D 0.941, 3D 0.981 (gleiche Grading-Muster),
also 6 % bzw. 2 % – den Aufwand (Positivität und Realisierbarkeit an Stufengrenzen) nicht wert. Lohnend nur auf
hierarchischen (Quadtree-)Gittern.

## 3D-Profil und Streifen-Stabilisierung in 3D (20.11.2026, Branch `perf/stab3d`)

A100-Profil (nsys/ncu, `tardis_perf3d_20261120`) von lattice-3d U112 P7 mit FCS, T = 3.2, Gesamtlaufzeit 97 s: Die
FCS-Tabelle (`k_lattice3d_eval_batch`) braucht 28 s, der Löser rund 60 s. Davon entfallen 15,0 s auf `k_stab_element<8,3>`, 12,9 s auf
`k_line_solve`, 8,1 s auf `k_limiter_group`, 5,7 s auf `k_combine`, 5,7 s auf `k_gather_stab` und 3,4 s auf `k_ghost`. Auf der A100 sind
alle 3D-Hauptkernel speicherbandbreitenbegrenzt (DRAM 76–80 %, SM 21–56 %), anders als auf der FP64-schwachen
A2000.

`k_stabilization_strip3<SZ>` (3D Q1): ein Thread pro (x, y, Komponente) läuft über einen z-Streifen. Das 3×3-Fenster
von a, g_0, g_1, g_2 jeder Ebene wird einmal geladen und zu den Elementtermen der drei Knoten z−1, z, z+1 addiert.
Der Element-Zwischenpuffer (Zellen × 8 × q, auf U112 P7 5,8 GB) entfällt. Die Größenklassen graduierter Gitter
verwendet der Kernel mit vorskalierten nu·K-Tabellen (`stiffness_nu`), gebrochene Linien über `strip_elem` für drei
Achsen. Die Nummerierung prüft `GpuSetup` (`strip3_ok`).

A100, P7, T = 3.2 (`tardis_stab3d_20261120`, Löserzeit; Ergebnisse gleich bis auf Rundung, Masse relativ ≤ 2e-16):

| Lauf | bisher | Streifen | |
|---|---:|---:|---:|
| G3H FCS (bisher Knotenkernel) | 123,4 s | 91,1 s | −26 % |
| G3H FCS + gebrochene Linien (bisher Knotenkernel) | 177,4 s | 148,6 s | −16 % |
| U112 ohne FCS (bisher Element + Gather) | 58,8 s | 67,4 s | +15 % |
| U168 ohne FCS (bisher Element + Gather) | 286,5 s | 348,8 s | +22 % |

Standard ist deshalb `M1PN_GPU_STRIP3=1`: Der Streifenkernel läuft nur auf graduierten und gebrochenen 3D-Gittern, auf dem
gleichmäßigen Gitter bleibt Element + Gather. `M1PN_GPU_STRIP3=0` schaltet ihn ganz ab, `=2` erzwingt ihn auch auf gleichmäßigen
Gittern. `M1PN_GPU_STRIP3_SZ` ist 8, 16 oder 32.

FCS-Tabelle 3D: Den Stack (`cudaLimitStackSize`) reserviert der Code jetzt, bevor er die Batchgröße aus dem freien
Speicher bestimmt. P15 auf U168 mit FCS bricht trotzdem weiter in `cudaDeviceSetLimit` mit „out of memory“ ab (164362, 164516,
164526): Die Löserfelder (169³ Knoten × 256 Momente, rund 10 GB pro Feld) belegen die A100 schon vorher fast ganz. Dieser
Lauf braucht mehrere GPUs oder die CPU-MPI-Rechnung.

Prüfjob 164526 (Standard nach f60aaad): G3H FCS Löser 90,5 s, U112 ohne FCS 58,7 s, Masse bitgleich zum alten Pfad; ctest ok.

## Adaptive PN in 3D: Stabilisierung als z-Streifen (21.11.2026, Branch `perf/adaptive3d`)

Auf dem gemergten Stand (709d914) brachte die dynamische adaptive PN in 3D kaum Zeitgewinn. Alle Läufe hatten FCS,
gebrochene Linien und Split, die adaptiven zusätzlich tol 3e-4 und Tile 8 (`tardis_adaptive3d_20261121`). Die
Löserzeit änderte sich bei G3Q P7 von 27,7 auf 30,7 s, bei G3Q P15 von 71,2 auf 68,3 s und bei G3H P7 von
147,5 auf 135,2 s, bei nur 28–39 % aktiven Koeffizienten. Das A100-Profil (`tardis_perf_ad3d_20261121`, G3Q P15) zeigt den Grund:
`k_stabilization_adaptive` braucht 34,2 s. Der Kernel geht über die Konnektivität und berechnet für jedes der 64
Element-Knoten-Paare die Kachel per Division. Der volle `k_stabilization_strip3` braucht dagegen 26,4 s für alle Momente.
Die übrigen adaptiven Kernel sind billig: Liniensolves 11,1 → rund 7 s, combine 10,9 → 2,3 s.

Neue Kernel, wählbar mit `M1PN_GPU_AD3`, Standard 2:
- `k_stabilization_adaptive3` (=1): strukturierte Indizes, 27-Punkte-Fenster pro Feld, Kachelaktivität nur für Knoten auf
  Kachelflächen. Bitgleich zu `k_stabilization_adaptive` (=0). Auf der A100 34,2 → 27,6 s.
- `k_stabilization_adaptive3s` (=2): die z-Streifenform von `k_stabilization_strip3` pro Kachel-Chunk. Jeder Warp läuft durch eine
  (x, y)-Spalte der Kachel über die Ebenen z0−1 … z1. Inaktive Nachbarn haben über die Ordnung ihrer Kachel den Wert 0;
  nachgeschlagen wird nur an Kachelflächen und in den Halo-Ebenen. Die Abweichungen liegen auf Rundungsniveau
  (lokal Masse ≤ 4e-16), die aktiven Anteile sind identisch. Auf der A100 **34,2 → 7,6 s**.

A100, T = 3.2, FCS, gebrochene Linien und Split (`tardis_ad3d_kernel_20261121`, `tardis_ad3d_kernel2_20261121`; Löserzeit):

| Lauf | nicht adaptiv | adaptiv, AD3=0 | AD3=1 | AD3=2 | aktiv | Faktor zu nicht adaptiv |
|---|---:|---:|---:|---:|---:|---:|
| G3Q P15 | 71,2 s | 68,3 s | 61,3 s | 42,0 s | 32,1 % | 1,70 |
| G3Q P7 | 27,7 s | 31,1 s | 28,9 s | 23,9 s | 39,4 % | 1,16 |
| G3H P7 | 147,5 s | 135,0 s | 129,2 s | 105,4 s | 28,5 % | 1,40 |

ctest und `tests/gpu_validate_adaptive.sh` (alle Fälle PASS) liefen in beiden Jobs durch. Die restliche Zeit bei G3Q P15
adaptiv: FCS-Tabelle 15,3 s, Stabilisierung 7,6 s, adaptive Liniensolves 4,6 s, aktive Flächenspuren 3,0 s, combine
2,3 s, Interface 2,0 s.

### Initialisierung und Host-Korrektur bei gebrochenen Linien (21.11.2026, `perf/adaptive3d`)

- **Zuflussdaten** (be0f1ca): `steady_data_` (ein `Incoming` mit Vektoren der Länge q pro Fläche und Randpunkt) wird für
  Bruchflächen nicht mehr ausgewertet, dort liest sie keine Stelle. G3Q P15 mit gebrochenen Linien: Init 69,9 → 32,8 s,
  RSS 101 → 35 GB. G3H P15 mit gebrochenen Linien lief vorher wegen Speichermangel nicht.
- **Host-Korrektur** (`host_correction`; liegt nach dem Warten auf die GPU auf dem kritischen Pfad; Teilzeiten mit
  `M1PN_SPLIT_TIMING=1`):
  - d673618: Die Gewichte der Flächenregel berechnet der Konstruktor einmal pro Seite, statt sie in jeder Stufe und an
    jedem Flächenknoten neu zu summieren.
  - 7dcd983: Kopien parallel, wiederverwendete Puffer, Diagnosesummen in festen Blöcken.
  - 1e2f4df: `split_apply` bildet die Bar-Zustände pro Knoten direkt, ohne die Zeilenfelder.
  - Die Lösung bleibt bitgleich; nur die Diagnosegrößen `split_mass` und Rand-Massenrate ändern sich auf Rundungsniveau.
  - Thread-Skalierung auf tardis (164639): 8/16/32 Threads → 12,4/8,2/7,2 s. Die Arbeit ist speichergebunden.

A100, FCS, gebrochene Linien und Split, T = 3.2 (`tardis_ad3d_final_20261121`; adaptiv: tol 3e-4, Tile 8). Die
Spalte „vorher“ zeigt den Stand 709d914 (164631/164625/164638):

| Lauf | Löser nicht adaptiv | Löser adaptiv (vorher) | aktiv | Faktor | gesamt nicht adaptiv / adaptiv | Init |
|---|---:|---:|---:|---:|---:|---:|
| G3Q P7 | 22,6 s | 18,2 s (30,7 s) | 39,4 % | 1,24 | 42,1 / 38,0 s | 19 s |
| G3Q P15 | 68,5 s | 32,4 s (68,3 s) | 32,1 % | 2,11 | 100,7 / 66,3 s | 32 s |
| G3H P7 | 127,5 s | 85,1 s (135,2 s) | 28,5 % | 1,50 | 195,9 / 154,1 s | 68 s |
| G3H P15 | 438,1 s | 150,7 s (189,6 s) | 22,4 % | 2,91 | 550,8 / 268,1 s | 116 s |

Rest bei G3H P15 adaptiv:
- Init 116 s: FCS-Tabelle ca. 54 s, Spektralradius 28 s (bestimmt dt), Schema 23 s.
- Löser 151 s: davon 80 s Warten auf die GPU und 15 s Host-Korrektur.
- RSS 103 GB.

## Hohe Ordnungen in 3D und Einstrahldaten pro Seite (21.11.2026, Branch `perf/init3d`)

Hohe Ordnungen auf einem A100 (`tardis_highpn3d_20261121`, Job 164641, und `tardis_init_fine3d_20261121`,
Job 164646; FCS, gebrochene Linien, T = 3.2, adaptiv tol 3e-4, Tile 8):

| Lauf | Löser nicht adaptiv | Löser adaptiv | Faktor | aktiv | mittleres N | GPU max. |
|---|---:|---:|---:|---:|---:|---:|
| G3Q P21 | 130,4 s | 41,3 s | 3,2 | 23,8 % | 6,1 | 39 GB |
| G3Q P35 | 440,6 s | 52,0 s | 8,5 | 10,3 % | 6,4 | 64 GB |
| G3F P7 (168³) | 420,4 s | 257,3 s | 1,6 | 22,5 % | 2,1 | 56 GB |
| G3F P11 (168³) | 794,3 s | 343,8 s | 2,3 | 18,6 % | 2,7 | 72 GB |

G3H P21 passt nicht auf die GPU (`cudaMalloc` out of memory bei etwa 74 GB). Die Masse von G3Q ändert sich zwischen
P21 und P35 um weniger als 0,005 %.

Bei P35 dauerte die Initialisierung 196 s, davon 117 s am Ende des Konstruktors, bei 345 GB RSS. Ursache war
`steady_data_`: ein `Incoming` mit `d` der Länge ~Q/2 für jeden der (N+5)² Punkte jeder Randfläche, bei Vakuum
alle gleich. Bei ortsunabhängiger Einstrahlung (Vakuum oder konstante Dichte ohne Atome) steht jetzt ein Eintrag
pro Seite in `steady_data_` (`steady_per_side_`, Zugriff über `steady_at`). `fill_datum` überspringt dann
Flächen ohne Datum. Die Ergebnisse bleiben bitgleich: Masse und Absorptionsrate von G3Q P15 uniform/adaptiv und
P35 adaptiv sind identisch mit den Jobs 164640/164641 (Job 164652, `tardis_init3d_20261121`).

| Lauf | Init vorher → nachher | RSS vorher → nachher | gesamt vorher → nachher |
|---|---:|---:|---:|
| G3Q P15 uniform | 31,5 → 25,3 s | 34,7 → 15,2 GB | 100,7 → 94,0 s |
| G3Q P15 adaptiv | 33,3 → 25,5 s | 34,7 → 15,2 GB | 66,3 → 58,3 s |
| G3Q P35 adaptiv | 195,7 → 62,2 s | 344,9 → 73,7 GB | 249,6 → 111,9 s |

Rest der Initialisierung bei G3Q P35: GPU-Aufbau 25 s (davon FCS-Tabelle ca. 17 s) und Spektralradius 30 s
(60 Potenziterationen mit dem vollen P35-Operator). Auf G3F dominiert die FCS-Tabelle mit 126–140 s.

### Spektralradius-Startvektoren, Stack-Reserve und GPU-Speicher pro Phase (21.11.2026, Branch `perf/init3d2`)

- Die zwei Zufallsstartvektoren der Potenziteration (Knoten × Q Ziehungen, mt19937) entstehen jetzt per
  `std::async` während des GPU-Konstruktors, mit gleicher Folge und gleicher Normierung. G3Q P35 adaptiv: Init
  62 → 58 s, Spektralradius 30 → 25 s. Bitgleich.
- Die FCS-Tabellen (`lattice3d_table`, `lattice3d_table_nodal`) setzen das Stack-Limit auf 16/24 KB pro Thread. Die
  Reserve (Limit × residente Threads) blieb bisher die ganze Rechnung belegt, etwa 4 GB auf dem A100. Das Limit wird
  nach dem Bau zurückgesetzt. Bitgleich (Jobs 164654, 164657).
- `M1PN_CTOR_TIMING=1` gibt jetzt den belegten GPU-Speicher nach jeder Phase aus (`gpu create:`, `gpu ctor:`).

Belegter GPU-Speicher nach Phasen (Job 164657, ein Schritt; 168³ = 5,36 Mio. Knoten, 264³ = 20 Mio.):

| Lauf | Tabellen | + Zustandspuffer | + Reaktion | + FCS | + Randphase A | Spitze (nvidia-smi) |
|---|---:|---:|---:|---:|---:|---:|
| G3Q P15 adaptiv | 0,5 | 7,1 | 7,3 | 7,9 | 13,4 GB | 35 GB |
| G3Q P35 adaptiv | 0,5 | 31,7 | 31,9 | 32,5 | 60,0 GB | 57 GB |
| 168³ P3 | 1,1 | 11,8 | 13,8 | 20,8 | 23,4 GB | 48 GB |
| 264³ P3 | 2,9 | 43,3 | 51,2 | OOM | | |

Befunde:
- Zustandspuffer: neben den 6 Feldern Knoten × Q (Zustand, x, r, drei Gradienten) vier Arrays pro Element mit
  4 × 8 Werten (`stab_low`, `element_limited`, `seg_ubar`, `seg_lim`), bei P3 knapp die Hälfte.
- Reaktion: FCS-Arrays pro Quadraturpunkt (Ort, Gewicht, σ_s), nach dem nodalen Bau nicht mehr gebraucht.
- FCS: die fp32-Knotentabelle (Knoten × 324) bleibt. Beim Bau liegt zusätzlich der double-Akkumulator gleicher Zeilenzahl
  (doppelte Größe) plus ein Drittel des freien Speichers für die Stapel. Daran scheitert 264³.
- Randphase A: drei Puffer Zeilen × Q (`G`, `H`, `db`) für Rand- und Bruchflächen, bei P35 27 GB. `db` (Datum)
  bleibt bei Vakuum null.

### Speicher, FCS-Kernel und Knotensummen (21.11.2026, `perf/init3d2`)

- `db` (Zeilen × Q, Datum der Randphase A) wird erst beim ersten Einstrahldatum angelegt, bei Vakuum nie. Die FCS-Arrays
  pro Quadraturpunkt (Ort, Gewicht, σ_s, q, local) werden nach dem Bau der nodalen 3D-Tabelle freigegeben. G3Q P35
  adaptiv: 60,0 → 50,8 GB nach dem Konstruktor; G3H P21 adaptiv passt jetzt auf ein A100 (Spitze 69,6 GB, Löser
  207 s, Init 121 s, aktiv 19 %, mittleres N 4,8; vorher OOM). Job 164664.
- FCS-Kernel (`k_lattice3d_eval_batch`): ncu (Job 164654) zeigte 255 Register, 12,5 % theoretische und 8 % erreichte
  Auslastung, 9 von 32 aktiven Threads pro Warp. Mit `__launch_bounds__(64, 8)` (höchstens 128 Register) ist die
  Tabelle 27 % schneller (G3Q 15,3 → 11,1 s, 168³ P3 Init 138 → 113 s), bitgleich; jetzt Standard (`M1PN_FCS3D_LB`:
  0 ohne Schranke, 2 höchstens 64 Register, langsamer). Sortieren der Punkte nach dem Abstand zur Quelle war langsamer
  (G3Q 21,7 s) und wurde verworfen (Job 164672).
- Material- und FCS-Knotensummen im Konstruktor laufen parallel über die Knoten (CSR `node_offsets_`/`node_entries_`,
  jetzt vor den nodalen Reaktionsdaten gebaut), in derselben Summationsreihenfolge wie die frühere Elementschleife.
  Die FCS-Knotenmasse braucht dabei ein explizites `std::fma`: die frühere Schleife wurde zu fma kontrahiert, die neue
  vektorisierte die Produkte ungefused (2D um 1e-8 verschoben; in 3D von der fp32-Tabelle verdeckt). 312³: CPU-Phase
  29 → 17,5 s. Bitgleich in 2D (GBQ P35, Spiegelreduktion auto/off) und 3D (Jobs 164676, 164688, 164691).

Init gegen `experiment/element-weno` (Job 164676): G3Q P7 uniform 19,1 → 14,8 s, P15 adaptiv 25,4 → 20,5 s, P35 adaptiv
62,3 → 54,3 s; 2D GBQ/GBH P35 unverändert (4–6 s).
- Verworfen (Branch `perf/fcs3d`, nicht gemergt): verschränkte Arbeits-Arrays des FCS-Kernels (Element k von Thread t
  an `work[k * count + t]`). Laut ncu-Quellprofil (Job 164654) fallen rund 37 % der Stall-Samples auf `Acc`/`Tmp`/`work`
  (Akkumulation über die Zeiten, Nullsetzen), mit 8 von 32 genutzten Bytes pro Sektor. Verschränkt bitgleich, aber
  langsamer (G3Q P7 Tabelle 11,1 → 12,9 s, 168³ P3 113 → 121 s; Job 164700): die Lanes eines Warps arbeiten auf
  verschiedenen Zeitfenstern, und die L1-Trefferquote der zusammenhängenden Arrays (94 %) geht verloren.
