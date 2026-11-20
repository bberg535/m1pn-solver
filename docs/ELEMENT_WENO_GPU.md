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
