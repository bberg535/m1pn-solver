# Ergebnisse: Konvergenzkampagnen auf tardis (A100), 15./16.11.2026

Auswertung und Interpretation: [docs/TARDIS_RESULTS.md](../../docs/TARDIS_RESULTS.md); Aufbau:
[docs/TARDIS_CAMPAIGN.md](../../docs/TARDIS_CAMPAIGN.md). Übernommen aus den (nicht versionierten) Kampagnen
`.claude/worktrees/gpu-campaign/nx/tmp/tardis_*` nach `tardis.py collect`; Rohfelder, Logs und Binaries bleiben dort.
Eingefrorene Binaries mit SHA256 im jeweiligen `manifest.json` (Pfade darin zeigen auf die Kampagnenordner).

| Ordner | Profil | Stand | Quelle (`git_head` im Manifest) | Inhalt |
|---|---|---|---|---|
| `paper/` | tardis-paper | 198 / 201 | `970c94e` (+ Kapazitätsnachtrag `bin/fast`) | glatt, Rand, Crossing, Line Source, Zeit, Ablation; `figures/`, `tables.tex` |
| `fcs/` | tardis-fcs | 60 / 60 | `970c94e` | Line Source mit FCS; `figures/`, `tables.tex` |
| `hetero/` | tardis-hetero | 61 / 61 | `970c94e` | Lattice 2D/3D, Shadow |
| `lattice/` | tardis-lattice | 12 / 18 | `46141ef` | Lattice Q1, N = 7–45, 224²–896² (ohne FCS) |

Dateien: `REPORT.md` (Statusbericht von `collect`), `results.csv` (eine Zeile je Lauf), `runtimes.md` (Wand- und
Lösungszeit je Aufgabe, GPU-Schätzung), bei paper/fcs `figures/` (PNG/PDF), `figures.json`, `tables.tex`.

Hinweise:

- `config.json` der einzelnen Läufe (nicht übernommen) enthält `"rk": 3` auch bei SSPRK(4,3) — Metadatenfehler,
  behoben in `3402f86`. Maßgeblich ist `time_integrator`; die Spalte `rk` in `results.csv` von paper/fcs stimmt (`ssprk43`, eine
  Ablation `ssprk3`); hetero/lattice haben keine `rk`-Spalte, alle Läufe dort mit SSPRK(4,3).
- `lattice/REPORT.md` trägt die Überschrift des Hetero-Profils (gleicher Berichtsgenerator). Fehlende Läufe dort:
  P7 1792², P15 1792², P21 1344², P35 896², P45 672² abgebrochen bzw. nicht gestartet; P45 224² am Timeout (1800 s)
  bei T = 2.95 von 3.2.
- Lattice-Abweichungen beziehen sich auf Schotthöfer/Hauck Tab. 1 / (2π) (kinetisch S22, 4 Stellen).

Lattice mit und ohne FCS auf NX (CPU): [../lattice_fcs_20261116/](../lattice_fcs_20261116/).
