# Offene Arbeit (Stand 16.11.2026)

Basis: Branch `feature/combine-202611` (Worktree `.claude/worktrees/combine`), aufbauend auf Tag `standard-20261116`.
Seit 16.11.2026 nur noch Element-WENO, ohne MFEM: Massenkorrektur, Biharmonik, NN-Filter und DLRA entfernt (letzter
Stand mit ihnen: `standard-20261116`). Bauen: `bash scripts/build.sh` (NX, im Slurm-Job; UCX-OpenMPI), `GPU=1 bash
scripts/build.sh tmp/build_gpu` (tardis). Cluster-Hinweise in `nx/README.md`.

## Erledigt am 16.11.2026

- Block 1: GPU-Validierung im Repo (`tests/gpu_validate.sh`), Kampagnenergebnisse versioniert (`results/`),
  tardis-Lattice-Kampagne ausgewertet (docs/TARDIS_RESULTS.md), Aufräumen (Worktrees, Rohdaten, alter Code).
- Block 2: Element-WENO Default von `m1pn_fast` mit CFL 1 (CFL-Scan); GPU mit Spiegelreduktion und periodischen
  Gittern; Lattice-FCS für `lattice-3d` (CPU, GPU); MPI in `m1pn_fast` für alle Fälle, bitgleich zu einem Rang;
  reproduzierbare Normen in der Potenziteration (dt unabhängig von der Rangzahl); MFEM-frei, MPI über InfiniBand.

## Offen

- MPI+GPU (zurückgestellt; für einen künftigen Cluster mit mehreren GPUs).

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
keine Spack-Installationen ohne Auftrag; Experimente auf eigenen Branches; Antworten auf Deutsch.
