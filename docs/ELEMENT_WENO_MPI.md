# Element-WENO mit MPI (verteilte Zeitintegration)

Branch `feature/element-weno-mpi` (Basis `standard-20261114`). Code: `src/element_weno_mpi.{hpp,cpp}`, Hooks in
`src/element_weno.{hpp,cpp}` (`DistributedHooks`, `set_distributed`), Treiber `src/element_weno_main.cpp`, Tests
`tests/element_weno_mpi.cpp`. Ohne `mpirun` bzw. mit einem Rang ist der Solver unverändert (alle Pfade hängen an
`dist_ != nullptr`).

## Verfahren

- **Zerlegung:** Scheiben entlang der letzten Achse (y in 2D, z in 3D). Jeder Rang baut den unveränderten
  `ElementWeno` auf seiner Scheibe plus `halo` Elementschichten (Standard 2) auf jeder inneren Seite
  (`ElementWenoOptions::cut`: dort keine Randflächen). Positionen kommen aus dem globalen Elementindex
  (`element_offset`), damit die Koordinaten bitgleich zur seriellen Rechnung sind.
- **Globale Kopplungen:**
  1. Die Linienlösungen mit der konsistenten 1D-Masse entlang der Zerlegungsachse (projizierter Gradient,
     Massenlösung). Sie werden verteilt in der Reihenfolge der seriellen LDLᵀ-Faktorisierung der globalen 1D-Masse
     substituiert. Jede Operation `y_i −= l·y_j` geschieht in derselben Reihenfolge wie seriell.
     Umsetzung als Task-Pipeline: ein Thread erledigt alle MPI-Aufrufe, die übrigen Threads bearbeiten Blöcke
     ganzer Gitterlinien als OpenMP-Tasks. Die Diagonale ist in die Rückwärtssubstitution integriert.
  2. Halo-Zeilen: nach der Linienlösung des Gradienten, die Hochordnungs-Momentraten vor dem Limiter, jede
     Stufenrate.
  3. Reduktionen: Zeitschrittschranke (Minimum über eigene Knoten), Randbilanz, Spektralradius (rangunabhängiger
     Startvektor, Normen über eigene Zeilen).
- **Auswertung:** Rang 0 sammelt den Endzustand und wertet ihn mit dem unveränderten seriellen Code aus (kinetischer
  Fehler, `final.csv`, Bilanzen). `--dump-state datei` schreibt die Endkoeffizienten binär.

**Ergebnis: Der verteilte Lauf ist bitgleich zum seriellen**, für jede Rangzahl. Geprüft wurde:
- lokal mit Q1 und Q2 und ungleichen Aufteilungen;
- auf NX mit 2, 4 und 6 Rängen bei E=512, N=15 über 455 Schritte.

Halo 2 reicht. Das frühere „Halo 6“ war ein Artefakt verschobener Koordinaten.

Einschränkungen:
- 2D/3D, offenes Gebiet, keine Spiegelseiten, Standardresiduum;
- jeder Rang braucht mindestens `halo` Elemente entlang der Achse;
- Ghost-Studie und DLRA laufen nicht unter MPI.

## Bauen und Starten (NX)

Die Standard-OpenMPI (`fabrics=none`) kann zwischen Knoten nur TCP (gut 1 GB/s). Für InfiniBand gibt es die
Spack-Umgebung `m1pn-ucx` mit OpenMPI 5.0.10 und UCX 1.22, gebaut auf dem System-rdma-core; das MFEM-Paket ist davon
unberührt. Element-WENO braucht kein MFEM:

```bash
O=/home/warehouse15/bberg/spack/opt/spack/linux-zen3/openmpi-5.0.10-kfdcvl5tog3tsi5rusglshnfcauemfjq
cmake -S . -B tmp/build_ucx -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$O/bin/mpic++ -DM1PN_ELEMENT_ONLY=ON
cmake --build tmp/build_ucx -j 16
# im Slurm-Job (Knoten exklusiv, ein Rang pro Sockel, 32 Threads):
mpirun -np 6 --map-by ppr:1:package:PE=32 --bind-to core --mca pml ucx -x OMP_NUM_THREADS=32 \
       -x OMP_PLACES=cores -x OMP_PROC_BIND=close tmp/build_ucx/m1pn_element_weno <Optionen> [--dump-state f]
```

Vorlage: `nx/tmp/mpi_ucx4_20261114/run.sbatch` (3 Knoten, `--ntasks-per-node=2 --cpus-per-task=32`). ssh zwischen den
Knoten ist gesperrt; `mpirun` startet trotzdem auf allen Knoten der Allokation.

## Messung (NX, Crossing-Studie 2D, Q1, T = 0.1, feste Schrittweite)

Job 160226 (`nx/tmp/mpi_ucx6_20261114`; Commit in `commit.txt`, SHA256 der Binaries in `sha256.txt`). Ein Prozess
pro Sockel (32 Kerne), 2 Prozesse pro Server. **Alle Endzustände sind bitgleich zum Ein-Prozess-Lauf.**

| Fall | 1 Sockel | 1 Server (2 Proz.) | 2 Server (4) | 3 Server (6) |
|---|---|---|---|---|
| E=512, N=15 | 59.2 s | 45.9 s (1.29×) | 27.2 s (2.18×) | 23.1 s (2.56×) |
| E=512, N=31 | 218.5 s | 143.3 s (1.52×) | 92.7 s (2.36×) | 67.8 s (3.22×) |
| E=1024, N=15 | 510.8 s | 327.1 s (1.56×) | 200.5 s (2.55×) | 152.9 s (3.34×) |

Ein Prozess mit 64 Threads auf einem ganzen Server: 47.4 s (E=512, N=15).

**UCX-Rendezvous:** Die Pipeline sendet viele Nachrichten von einigen 10 KB aus einem Thread, der selten in MPI ist.
Oberhalb der UCX-Standardschwelle (wenige KB) läuft das Rendezvous-Protokoll und kommt nur voran, wenn beide
Seiten MPI aufrufen.

| N=31, verkürzt (T = 0.02) | Zeit |
|---|---|
| 1 Sockel | 43.6 s |
| 4 Prozesse, Standardschwelle | 49.4 s |
| 4 Prozesse, `UCX_RNDV_THRESH=4mb` | 18.5 s |
| 2 Prozesse, Standardschwelle | 39.8 s |
| 2 Prozesse, `UCX_RNDV_THRESH=4mb` | 28.4 s |

Weitere Diagnosen (`nx/tmp/mpi_diag_n31_20261114`):
- mehr Blöcke: schlechter;
- Transport `rc_x`: ohne Wirkung.

Das Programm setzt daher `UCX_RNDV_THRESH` vor `MPI_Init`, sofern nicht gesetzt (jetzt 512 KB: die
Pipeline-Nachrichten gehen eager, die MB-großen Halo-Pakete weiter per Rendezvous; mit 4 MB stieg der
Halo-Austausch bei E=1024 von 6 auf 18 s).

Bewertung:
- Die lokalen Kerne skalieren nahezu ideal.
- Begrenzend bleibt die verteilte y-Linienlösung (`t_axis_solve_max`: E=512/N=15 etwa 10 s bei 4–6 Prozessen). Sie
  ist sequenziell über die Rangkette und wird pipelined.
- Überlappen der y-Pipeline mit den x-Gradienten (Füllaufgaben niedriger Priorität, Job 160228, `mpi_ucx7`): 0–3 %
  (E=512/N=15, 6 Prozesse: 23.1 → 22.3 s; E=1024: 152.9 → 151.3 s), bitgleich. Mit Schwelle 512 KB sinkt der
  Halo-Austausch bei E=1024 wieder (4 Prozesse: 16 → 8 s).
- Grenze der bitgleichen Variante: Die y-Substitution ist eine sequenzielle Kette über alle Ränge (Reihenfolge der
  seriellen Faktorisierung). Mehr Skalierung bräuchte einen parallelen Partitionslöser (z. B. SPIKE oder Schur auf
  den Schnittstellen). Er ist mathematisch dieselbe Lösung, aber nur bis auf Rundung gleich (rangabhängig).
- Ältere Varianten (blockweise Thread-Aufteilung, je Thread eigene Kommunikation) sind per Umgebungsvariable
  erreichbar (`M1PN_MPI_CHUNK_SPLIT`, `M1PN_MPI_THREADED_PIPELINE`), aber langsamer.

## Paralleler Schnittstellenlöser (Schur, Option `M1PN_MPI_SCHUR=1`)

Job 160229 (`nx/tmp/mpi_ucx8_20261114`). Jeder Rang löst seine inneren Schichten unabhängig. Die Separatorschichten
bilden ein kleines SPD-System (eine Allreduce, Lösung auf jedem Rang), vorberechnete Spikes korrigieren das Innere.
Das Gleichungssystem ist dasselbe, die Rundung ist eine andere.

| Fall | 2 Prozesse | 4 Prozesse | 6 Prozesse | max. rel. Abweichung zum seriellen Lauf |
|---|---|---|---|---|
| E=512, N=15 | 86.2 s (bitgleich: 45.5) | 31.6 (27.6) | 28.6 (22.3) | 8e-9 … 1.3e-8 |
| E=512, N=31 | 140.6 (141.6) | 107.0 (92.1) | 103.9 (70.1) | 2.5e-7 … 3.4e-7 |
| E=1024, N=15 | 300.0 (322.9) | 181.8 (193.7) | 152.6 (151.3) | 9e-6 … 1.4e-5 |

- Die Allreduce ganzer Separatorschichten kostet 18–54 s.
- Wichtiger: Die Rundungsunterschiede wachsen im limitierten Verfahren chaotisch. Nach 455 Schritten liegen sie bei
  1e-8 … 3e-7, nach 910 Schritten bei 1e-5 (wie beim Threadwechsel im Positivitätszweig, POSITIVITY_PROJECTION.md,
  Abschnitt 3).
- Eine 1e-8-Vorgabe gegenüber dem seriellen Lauf hält über lange Läufe nur die bitgleiche Pipeline ein. Sie bleibt
  Standard.

## 3D (Zerlegung entlang z, Job 160239, `nx/tmp/mpi_3d_20261114`)

Crossing-Studie in 3D (Pulse bei z = 0.5, ohne kinetische Fehlermaße), Q1, ohne Winkelreduktion (N=7: 64,
N=15: 256 Momente). Alle Endzustände sind bitgleich zum Ein-Prozess-Lauf.

| Fall | 1 Prozess | 1 Server | 2 Server | 3 Server |
|---|---|---|---|---|
| E=64³, N=7, T=0.3 | 66.1 s | 47.6 (1.39×) | 34.0 (1.94×) | 30.1 (2.20×) |
| E=96³, N=7, T=0.2 | 200.6 s | 129.4 (1.55×) | 80.5 (2.49×) | 63.9 (3.14×) |
| E=64³, N=15, T=0.3 | 209.4 s | 148.7 (1.41×) | 103.6 (2.02×) | 82.9 (2.53×) |

## Nadelöhr y-/z-Kette (Untersuchung 14.11.2026, Kurzläufe T = 0.02, E=512)

| Variante | N=15, 4 Proz. | N=15, 6 Proz. | N=31, 4 Proz. | N=31, 6 Proz. | Ergebnis |
|---|---|---|---|---|---|
| Task-Pipeline, gleich große Blöcke | 5.87 | 4.75 | 18.2 | 13.7 | Basis |
| Trapez-Blöcke (klein am Anfang und Ende) | 5.37 | 4.55 | 19.4 | 13.9 | neutral bis −9 % |
| 48 / 192 Blöcke | 5.80 / 6.67 | 4.69 / 6.07 | 20.1 / 17.3 | 15.7 / 13.0 | uneinheitlich |
| `OMP_WAIT_POLICY=active`, `GOMP_SPINCOUNT` | −13 % | 0 | −5 % | 0 | kaum |
| feste Block-Zuordnung, Blockpuffer | 5.54 | 4.38 | 18.8 | 14.1 | wie Tasks |
| + dauerhafte Puffer (Standard) | – | 3.95 | – | – | −10 % |

Profil (Rang 0, N=15, 786 Lösungen):

| | 2 Prozesse | 6 Prozesse |
|---|---|---|
| Hauptschleife | 3.5 s | 2.2 s |
| Rechenzeit, Summe über 31 Arbeiter | 59.8 s | 12.7 s |
| ideal (Summe / 31) | 1.9 s | 0.41 s |
| Auslastung der Kette | 55 % | 19 % |

Befund:
1. Die verteilte Substitution kostet pro Wert mehr als die serielle. Seriell wird jede Linie in einem Zug im Cache
   gelöst; verteilt liegt zwischen Vorwärts- und Rückwärtsschritt die Wartezeit auf die anderen Ränge, und die
   Daten fallen aus L2/L3.
2. Mit wachsender Rangzahl dominiert die Weitergabe durch 2·P Stationen pro Block. Sie skaliert nicht.

Bitgleich ist mit dieser Kette nach heutigem Stand höchstens etwa 10–20 % mehr drin. Der parallele Schur-Löser
skaliert besser im Rechenteil, verliert aber an der Allreduce ganzer Schichten und hält 1e-8 über lange Läufe nicht.
Eine 2D-Zerlegung (x und y) würde beide Richtungen ketten und hilft nicht.
