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
