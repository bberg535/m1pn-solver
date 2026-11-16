# Offene Arbeit nach standard-20261116 (Stand 16.11.2026)

Basis: Branch `feature/system-202611`, Tag `standard-20261116` (gepusht; Default-Branch `experiment/element-weno`
= dieser Stand + hochgeladene Projektions-PDF). Worktree `.claude/worktrees/system`, Builds `tmp/build_cpu`
(MFEM, MPI, OpenMP, Tests) und `tmp/build_gpu` (`-DM1PN_ELEMENT_GPU=ON`, nvcc `/sfw/cuda/12.6`). Bauen und Tests auf
NX per Slurm (`--partition=long --constraint=nx`; Login-Knoten hat 1 Kern; NFS-Zeitstempel: geänderte Dateien vor
`cmake --build` auf dem Knoten mit `touch` anfassen). GPU: tardis (`--constraint=ampere --gres=gpu:1`, eine A100).

## Block 1 — jetzt (Reproduzierbarkeit, Aufräumen)

1. ~~**GPU-Validierung ins Repo:**~~ erledigt 16.11.: `tests/gpu_validate.sh` + `tests/gpu_validate_cmp.py`, Job 160891
   alle 7 Fälle bestanden (Schwelle 1e-8). Ursprünglicher Auftrag: `tests/gpu_validate.sh` (Slurm-Skript für tardis) nach Vorlage
   `.claude/worktrees/system/tmp/gpuval/val.sbatch` + `cmp_final.py` (aus `gpu-campaign/nx/tmp/validate_20261115/`):
   CPU gegen `--gpu` mit demselben Binary für Crossing, Rand Q2, Line Source+FCS, Lattice (±FCS, P35), Lattice 3D;
   Schwellen wie in docs/ELEMENT_WENO_GPU.md (Abschnitt Validierung des Systemstands).
2. ~~**Ergebnisse versionieren:**~~ erledigt 16.11.: `results/tardis_202611/{paper,fcs,hetero,lattice}`,
   `results/lattice_fcs_20261116` (je README). Ursprünglicher Auftrag: aus den Kampagnen (nicht versioniert) Tabellen/Plots/Berichte nach `results/` übernehmen
   (Muster `results/positivity`): `gpu-campaign/nx/tmp/tardis_{paper,fcs,hetero}` (REPORT.md, results.csv,
   runtimes.md, figures/, tables.tex), `lattice-fcs/nx/tmp/lattice_fcs` (REPORT.md, results.csv). Keine Rohfelder.
3. ~~**tardis-Lattice-Kampagne auswerten:**~~ erledigt 16.11.: docs/TARDIS_RESULTS.md, Abschnitt Lattice-Konvergenz
   (12/18; Gitter-EOC ≈ 2, N konvergiert ab P35, Grenzwert Masse −0.02 % / Absorption +0.06 %). Ursprünglicher Auftrag: `gpu-campaign/nx/tmp/tardis_lattice` (13/18 fertig, Rest abgebrochen;
   `python3 tardis.py collect --campaign ...`), Konvergenz in N (7–45) und Gitter (224²–896²) gegen
   Schotthöfer/Hauck; in docs/TARDIS_RESULTS.md ergänzen. Beachten: `config.json` der Kampagnen hat `"rk":3` auch
   bei SSPRK(4,3) (Metadatenfehler, behoben in 3402f86; `time_integrator` stimmt).
4. **Aufräumen** (teilweise erledigt 16.11.: Worktrees gpu-a100, gpu-projection, lattice-fcs, positivity-projection,
   element-weno-fcs, element-weno-hetero3d, element-weno-ssp43 entfernt, Branches bleiben; deren `nx/tmp` und
   Nicht-Build-Inhalte von `tmp/` liegen in `<Haupt-Checkout>/nx/tmp/archive_worktrees_20261116/<name>/` (INDEX.txt).
   Offen: Rohdaten element-weno-mpi, gpu-campaign bleibt. Ursprünglicher Auftrag: mit dem Nutzer abstimmen, nichts ungefragt löschen): Worktrees/Branches, die vollständig in
   standard-20261116 enthalten sind (gpu-a100, gpu-projection, gpu-campaign, lattice-fcs, element-weno-mpi,
   positivity-projection, element-weno-fcs, hetero3d, ssp43 …); ungetrackte Logs im Haupt-Checkout.

## Block 2 — in Arbeit (Kombinierbarkeit), Branch `feature/combine-202611`, Worktree `.claude/worktrees/combine`

Erledigt (16.11.2026):
- DLRA entfernt (Nutzerentscheidung; `0cead11`, bitgleich zum Standard für alle übrigen Läufe).
- Element-WENO ist Default von `m1pn_fast`, CFL 1.0 nach CFL-Scan (`results/cfl_scan_20261116`, `c039a16`).
- GPU mit Spiegelreduktion und periodischen Gittern (`c039a16`; `tests/gpu_validate.sh` 24/24, ≤ 4e-10); Spiegel auf
  der A100 1.8× (Lattice 224² P15) bis 3.0× (Line Source 512² P7).

Offen:
- MPI im Element-WENO-Pfad von `m1pn_fast` (bisher nur ein Rang; MPI nur in `m1pn_element_weno` für Crossing und
  glatten Ausfluss). Voraussetzung für MPI + Lattice-FCS (Tabelle pro Rang) und für MPI+GPU in `m1pn_fast`.
- MPI+GPU (für einen künftigen Cluster mit mehreren GPUs; tardis hat eine A100, also nur Funktionstest).
- Lattice-FCS für `lattice-3d`: neue Quadratur (Winkelintegral über die sichtbaren Würfelflächen der Quellzelle,
  Knicke an projizierten Voxelkanten); Verfahren mit dem Nutzer abstimmen.

## Block 3 — zurückgestellt (Numerik fürs Paper)

- WENO-Aktivierung an der Rundungsgrenze: glatt Q4 256² Fehler 4.8e-8 statt ~1e-13 (min γ 0.992; ohne Stabilisierung
  1.2e-12, `--weno-linear 0` 5.2e-13). Erwähnen; eine Korrektur wäre eine Verfahrensänderung (nur mit Freigabe).
- Eigene Lattice-Referenz (Monte Carlo oder S_N, Masse/Absorption), da Abweichungen bei 0.1–0.3 % liegen und die
  Literaturtabelle nur 4 Stellen hat.
- Zeitordnung 2 mit Streuung (Strang); höhere Splitting-Ordnung nur falls nötig.

## Block 4 — zurückgestellt (Performance)

- Host-Fallbacks bei Q1 ohne FCS (bis 4e8, long double): Line Source P3 1024² 1189 s statt 154 s mit FCS;
  Abhilfe double-double auf der GPU (vom Nutzer zunächst verworfen).
- 3D: Host-Randpfad (Face-Terme) als Engpass.

Regeln: Abweichungen bis 1e-8 erlaubt, keine mathematischen Änderungen ohne Freigabe; kein Push/Merge ohne Auftrag;
keine Spack-Installationen; MFEM-Varianten nicht ändern; Experimente auf eigenen Branches; Antworten auf Deutsch.
