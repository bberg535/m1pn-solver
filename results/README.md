# Ergebnisse (versioniert)

Ausgewertete Kampagnen: Berichte, Tabellen (CSV, TeX), Plots. Rohfelder bleiben in den Kampagnenordnern (`nx/tmp/`,
nicht versioniert). Jeder Ordner hat ein README mit Herkunft (Commit, Binary-SHA256, Job).

| Ordner | Inhalt | Doku |
|---|---|---|
| `convergence/tardis_202611/` | Konvergenzkampagnen auf der A100: paper (glatt, Rand, Crossing, Line Source, Zeit, Ablation), fcs, hetero, lattice | docs/TARDIS_RESULTS.md |
| `convergence/cfl_scan_20261116/` | CFL-Scan 0.1-1.0 (Grundlage für den Default CFL 1) | docs/ELEMENT_WENO.md |
| `heterogeneous/nx_hetero_20261111/` | Material, Lattice, 3D-Lattice, Shadow auf NX; Reaktionsmodell-Studie | docs/ELEMENT_HETEROGENEOUS.md |
| `heterogeneous/lattice_fcs_20261116/` | Lattice mit und ohne First-Collision-Source (NX, CPU) | docs/FCS_LATTICE.md |
| `heterogeneous/lattice_mc_20261116/` | Monte-Carlo-Referenz Lattice 2D/3D, alle Lattice-Läufe dagegen | docs/LATTICE_MC.md |
| `performance/host_boundary_20261116/` | Host-Randpfad der GPU-Zeitschleife vor/nach Block 4 (tardis) | docs/ELEMENT_WENO_GPU.md |
| `boundary/positivity_20261113/` | Randoptimierung und Positivitätsprojektion (Laufzeit, Genauigkeit, Skalierung) | docs/POSITIVITY_PROJECTION.md |

Ergebnisse von Massenkorrektur, Biharmonik, NN-Filter und DLRA sowie frühe Element-WENO-Vergleiche damit liegen nur
noch in der Git-Historie (bis Tag `standard-20261116`).
