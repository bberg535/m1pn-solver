# M1PN-Solver: Element-WENO

Gekoppeltes M1PN-Modell (PN-Momente, M1-Kegel für die vier niedrigen Momente) mit elementbasiertem WENO auf
Tensor-Bernstein-Elementen Q_p, elementbasiertem Limiting, SSPRK(4,3) und momententreuer Positivitätsprojektion am Rand.
CPU (OpenMP, MPI) und GPU (CUDA, A100). Kein MFEM.

> **Stand:** Default-Branch `experiment/element-weno`, Tag `standard-20261116-ewonly` (16.11.2026). Massenkorrektur,
> Biharmonik, NN-Filter und DLRA sind entfernt; ihr letzter Stand ist der Tag `standard-20261116`.

## Programme

| Programm | Zweck |
|---|---|
| `m1pn_fast` (`src/fast_main.cpp`) | Standard: alle Fälle, MPI, `--gpu` |
| `m1pn_element_weno` (`src/element_weno_main.cpp`) | Einzelstudien: glatte periodische Genauigkeit, glatter Ausfluss, Ghost-Studie |

```bash
m1pn_fast --case crossing-study --inflow crossing-study-transport --dimension 2 --elements 128 --degree 1 \
          --pn-order 7 --time 0.32 --output DIR
m1pn_fast --case lattice-3d --dimension 3 --inflow vacuum --background 1e-10 --elements 56 --degree 1 \
          --pn-order 3 --time 3.2 --fcs analytic --output DIR            # First-Collision-Source
m1pn_fast ... --gpu                                                       # GPU (tardis)
mpirun -np 2 --map-by ppr:1:package:PE=32 --bind-to core -x OMP_NUM_THREADS=32 m1pn_fast ...   # MPI
```

Standardwerte: SSPRK(4,3), `--cfl 1 --dt 0` (CFL-Scan in `docs/ELEMENT_WENO.md`), Spiegelreduktion und
Winkelreduktion automatisch, Threads = physische Kerne eines NUMA-Knotens. `m1pn_fast --help` listet alle Optionen.

Fälle: `crossing-study`, `crossing-3d`, `line-source` und `plane-source` (auch mit `--fcs analytic`), `lattice-paul`
und `lattice-3d` (auch mit FCS), `shadow-weih`, glatte Fälle (`smooth`, auch `--periodic`) und Strahltests.
Ausgabe: `config.json`, `summary.json` (Bilanz, Laufzeiten), `history.csv`, `initial.csv`/`final.csv`
(Momente auf einem gleichmäßigen Gitter), in 3D `slices.csv`.

## Bauen und testen

```bash
bash scripts/build.sh tmp/build                  # NX: im Slurm-Job; Eigen, OpenMP, MPI (UCX-OpenMPI, m1pn-ucx)
GPU=1 bash scripts/build.sh tmp/build_gpu        # tardis: CUDA 12.6, sm_80
bash scripts/test.sh tmp/build                   # ctest + MPI-Linienlöser-Test
sbatch tests/fast_mpi_equivalence.sh             # NX: 1/2/4 Ränge bitgleich
sbatch tests/gpu_validate.sh                     # tardis: CPU gegen GPU
```

## Ordner

| Ordner | Inhalt |
|---|---|
| `src/` | Solver (`element_weno*`, Rand, Projektion, FCS `uncollided*`, Fälle `benchmark_cases.hpp`) und die beiden Programme |
| `tests/` | ctest-Programme, CLI-Test, Prüfskripte für MPI und GPU |
| `scripts/element_weno/` | Kampagnen (NX, tardis, Lattice-FCS), Auswertung und Plots |
| `scripts/positivity/`, `scripts/line_source/` | Positivitätsprojektions-Kampagnen; Line-Source-Referenz (Ganapol) |
| `nx/` | Cluster-Kurzreferenz, Pack-Job, eingefrorene Analysehilfe |
| `docs/` | Verfahren und Ergebnisse (unten), `docs/tex/` Kapitel „Numerische Tests“ |
| `results/` | versionierte Auswertungen (Tabellen, Plots, Berichte), Index in `results/README.md` |
| `tmp/`, `nx/tmp/` | Builds und Rohdaten der Kampagnen (nicht versioniert) |

## Dokumentation

| Dokument | Thema |
|---|---|
| [docs/ELEMENT_WENO.md](docs/ELEMENT_WENO.md) | Verfahren, Zeitschritt, SSPRK(4,3), Winkel- und Spiegelreduktion, CFL-Default |
| [docs/BOUNDARY_FIXES.md](docs/BOUNDARY_FIXES.md), [docs/POSITIVITY_PROJECTION.md](docs/POSITIVITY_PROJECTION.md) | Randbehandlung, Positivitätsprojektion |
| [docs/ELEMENT_HETEROGENEOUS.md](docs/ELEMENT_HETEROGENEOUS.md) | Material, Lattice, 3D, Shadow |
| [docs/FCS.md](docs/FCS.md), [docs/FCS_LATTICE.md](docs/FCS_LATTICE.md) | First-Collision-Source (Puls, Lattice 2D/3D) |
| [docs/ELEMENT_WENO_MPI.md](docs/ELEMENT_WENO_MPI.md) | MPI (Scheiben, bitgleich, Start auf NX) |
| [docs/ELEMENT_WENO_GPU.md](docs/ELEMENT_WENO_GPU.md) | GPU-Pfad (A100), Validierung |
| [docs/TARDIS_CAMPAIGN.md](docs/TARDIS_CAMPAIGN.md), [docs/TARDIS_RESULTS.md](docs/TARDIS_RESULTS.md) | Konvergenzkampagnen |
| [docs/SPEEDUP_OPTIONS.md](docs/SPEEDUP_OPTIONS.md) | gemessene Beschleunigungsoptionen |
| [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) | offene Arbeit |
