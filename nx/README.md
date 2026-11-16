# Cluster: NX und tardis

Kurzreferenz für Element-WENO-Läufe auf den Clustern (Stand 16.11.2026). Der Solver braucht kein MFEM; gebaut wird mit
`scripts/build.sh` (Standard: OpenMPI mit UCX aus der Spack-Umgebung `m1pn-ucx`, InfiniBand zwischen den Knoten).

| | NX | tardis |
|---|---|---|
| Hardware | 6 Server, je 2 × EPYC 9354 (Zen 4, 2 × 32 Kerne, 2 NUMA-Knoten), 1.5 TB | EPYC 7343 (32 Kerne) + eine A100 80 GB |
| Slurm | `--partition=long --constraint=nx` | `--partition=med --constraint=ampere --gres=gpu:1` |
| Bauen | `bash scripts/build.sh tmp/build` im Job (Login-Knoten: 1 Kern) | `GPU=1 bash scripts/build.sh tmp/build_gpu` im Job |
| Laufen | ein Rang pro Sockel, 32 Threads (`docs/ELEMENT_WENO_MPI.md`) | `m1pn_fast --gpu ...` (`docs/ELEMENT_WENO_GPU.md`) |

Hinweise:
- NFS-Zeitstempel: geänderte Quelldateien vor `cmake --build` auf dem Rechenknoten mit `touch` anfassen.
- Bandbreitenbegrenzt: am schnellsten 32 Threads je Lauf auf einem Sockel, für Durchsatz 4 × 16 gepinnt
  (`scripts/element_weno/pack_node.py`, `nx/batches/element_weno_pack.sbatch`).
- Kampagnen: `scripts/element_weno/nx_convergence.py` (NX), `tardis_convergence.py` (GPU), `lattice_fcs_study.py`;
  sie frieren Binaries und Skripte (u. a. `nx/scripts/analysis.py`, `scripts/line_source/reference.py`) in den
  Kampagnenordner unter `nx/tmp/` ein (nicht versioniert). Ausgewertete Ergebnisse gehören nach `results/`.
- Prüfskripte: `tests/fast_mpi_equivalence.sh` (NX, MPI gegen einen Rang, bitgleich), `tests/gpu_validate.sh`
  (tardis, CPU gegen GPU).

Die frühere Clusterstudie der Massenkorrektur (Variante B, 04.11.2026) ist mit Massenkorrektur und Biharmonik am
16.11.2026 entfernt worden; sie liegt in der Git-Historie (Tag `standard-20261116`).
