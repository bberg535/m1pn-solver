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
| PN-Randlast (Teil A) | Host | nur Randknoten; Datum-Anteil `D B_datum` vorab für die nächste Stufe |
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

Nicht unterstützt (Abbruch mit Meldung): MPI, periodisch, Spiegel, symmetrisches Residuum, Reaktion/Filter,
M1-Target, `projection_high_order/hybrid/diagnostics`, äußere Zustände (`exterior_state`).

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

Gleiches Binary, CPU gegen `--gpu` (tardis, A100), Dichte am Ende:

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
