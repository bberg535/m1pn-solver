# Monte-Carlo-Referenz Lattice 2D/3D (NX, 16.11.2026)

Verfahren: [docs/LATTICE_MC.md](../../../docs/LATTICE_MC.md). Programm `m1pn_lattice_mc` (`tools/lattice_mc.cpp`),
Branch `feature/block3-numerics` auf `82d4a69` (Quelle zum Zeitpunkt des Laufs noch nicht committet; eingefrorenes
Binary, SHA256 in `binary.sha256`). Job 161065, ein NX-Knoten, 64 Threads.

| Datei | Inhalt |
|---|---|
| `lattice-paul.json`, `lattice-3d.json` | Produktion: 4e10 Historien, 1000 Batches, Seed 20261116, T = 3.2, 32 Zeitpunkte; Blöcke `source`, `background` (ρ0 = 1e-10), `total` mit Standardfehlern (384 s bzw. 485 s) |
| `*_seed99.json` | unabhängige Kontrolle: 4e9 Historien, Seed 99 (Abweichung ≤ 2 Standardfehler) |
| `validation_lattice-paul_uncollided.json` | 4e9 Historien gegen die ungestoßene Analytik (32 Panels, Ordnung 6) bei t = 1, 2, 3.2 |
| `validation_lattice-3d_uncollided_panels{3,4,6}.json` | 1e8 Historien gegen die 3D-Analytik bei t = 1, Quadratur 3/4/6 Panels |
| `REPORT.md`, `comparison.csv` | alle versionierten Solver-Läufe (NX lattice_fcs, tardis lattice/hetero) gegen MC und Literatur (`scripts/element_weno/lattice_mc_compare.py`) |
