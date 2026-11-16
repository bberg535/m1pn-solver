# Ergebnisse: Lattice mit und ohne First-Collision-Source (NX, CPU), 16.11.2026

Methode: [docs/FCS_LATTICE.md](../../docs/FCS_LATTICE.md). Kampagne `nx/tmp/archive_worktrees_20261116/lattice-fcs/nx_tmp/lattice_fcs` im Haupt-Checkout
(nicht versioniert; Studienskript `scripts/element_weno/lattice_fcs_study.py`), Branch `experimental/lattice-fcs`
(`f68a994`, im Systemstand standard-20261116 enthalten). Q1, 56²–448², N = 3, 7, 15, 21 (35 auf 112², 224²), je mit und
ohne FCS, ein Sockel (32 Kerne) je Lauf.

- `REPORT.md`: Tabelle Masse / Absorption / Rate als Abweichung gegen Schotthöfer/Hauck Tab. 1 / (2π), ungestoßene
  Masse, Zeit für die FCS-Tabelle und gesamt.
- `results.csv`: Rohwerte je Lauf (`dev_*` relativ, `balance` Massenbilanz, Laufzeiten).
- `binary.sha256`: SHA256 des eingefrorenen `m1pn_fast` der Kampagne.

GPU-Gegenstück (ohne FCS, N bis 45, bis 896²): [../tardis_202611/lattice/](../tardis_202611/lattice/).
