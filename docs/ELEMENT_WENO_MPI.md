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

Job 160224 (`nx/tmp/mpi_ucx5_20261114`, Commit in `commit.txt`, SHA256 der Binaries in `sha256.txt`). Ein Rang pro
Sockel (32 Kerne), 2 Ränge pro Server. Alle Endzustände sind bitgleich zum Ein-Rang-Lauf.

| Fall | 1 Rang | 2 Ränge (1 Server) | 4 Ränge (2 Server) | 6 Ränge (3 Server) |
|---|---|---|---|---|
| E=512, N=15 | 59.2 s | 43.3 s (1.37×) | 27.3 s (2.17×) | 24.6 s (2.41×) |
| E=512, N=31 | 218.5 s | 198.2 s (1.10×) | 246.8 s (0.89×) | 70.4 s (3.10×) |
| E=1024, N=15 | 510.8 s | 425.0 s (1.20×) | 195.6 s (2.61×) | 143.7 s (3.56×) |

Zum Vergleich: ein Prozess mit 64 Threads auf einem ganzen Server braucht bei E=512, N=15 47.4 s.

Bewertung:
- Die lokalen Kerne skalieren nahezu ideal: Stabilisierung, Limiter und Rand teilen sich durch die Rangzahl.
- Der Halo-Austausch kostet über InfiniBand 1.6–2 s; über TCP waren es 17–23 s.
- Begrenzend ist die verteilte y-Linienlösung (Spalte `t_axis_solve_max` in `summary.json`).
- Offen: Bei N=31 sind 2 und 4 Ränge auffällig langsam (Linienlösung 93 bzw. 183 s, 6 Ränge 33 s). Vermutet wird
  ein Protokolleffekt großer Nachrichten (UCX-Rendezvous) in der Task-Pipeline; noch nicht untersucht.
- Größere Gitter skalieren besser (E=1024: 3.56× auf 3 Servern), weil die Rechenarbeit pro Pipeline-Block mit
  der Gittergröße wächst.
