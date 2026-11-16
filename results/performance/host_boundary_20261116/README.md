# Host-Randpfad der GPU-Zeitschleife (tardis, 16.11.2026)

Auswertung: [docs/ELEMENT_WENO_GPU.md](../../../docs/ELEMENT_WENO_GPU.md), Abschnitt „Host-Randpfad (Block 4)“.
tardis, eine A100 80 GB, 32 Host-Threads, `m1pn_fast --gpu`, Lattice 3D Q1 56³ (P7, P15, ± FCS) und Lattice 2D Q1
224²/448² P35, T = 1, CFL 1.

| Datei | Inhalt |
|---|---|
| `timings.csv` | Löserzeit, Host-Phase A (`host_boundary_high`), Host-Phase B (`host_boundary`, inkl. Warten auf den Ghost-Kernel), Sync/Wait, Masse; vorher (Job 161073), nur Phase B und Phase A + B (Job 161081) |
| `nsys_lattice3d_P15_E56.txt` | Nsight-Systems-Kernelsumme vor der Änderung (5 Schritte, inkl. Potenziteration) |
| `binary_before.sha256`, `binary_after.sha256` | eingefrorene Binaries (`feature/block3-numerics` 9985e68 bzw. `feature/block4-host-boundary` vor dem Commit) |
| `run_before.sbatch`, `run_after.sbatch` | Jobskripte |
