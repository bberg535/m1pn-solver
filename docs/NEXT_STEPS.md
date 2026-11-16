# Offene Arbeit (Stand 16.11.2026)

Basis: Default-Branch `experiment/element-weno`, Tag `standard-20261116-ewonly` (gemergt aus `feature/combine-202611`,
aufbauend auf Tag `standard-20261116`).
Seit 16.11.2026 nur noch Element-WENO, ohne MFEM: Massenkorrektur, Biharmonik, NN-Filter und DLRA entfernt (letzter
Stand mit ihnen: `standard-20261116`). Bauen: `bash scripts/build.sh` (NX, im Slurm-Job; UCX-OpenMPI), `GPU=1 bash
scripts/build.sh tmp/build_gpu` (tardis). Cluster-Hinweise in `nx/README.md`.

## Erledigt am 16.11.2026

- Block 1: GPU-Validierung im Repo (`tests/gpu_validate.sh`), Kampagnenergebnisse versioniert (`results/`),
  tardis-Lattice-Kampagne ausgewertet (docs/TARDIS_RESULTS.md), Aufräumen (Worktrees, Rohdaten, alter Code).
- Block 2 (abgeschlossen bis auf das zurückgestellte MPI+GPU): Element-WENO Default von `m1pn_fast` mit CFL 1 (CFL-Scan); GPU mit Spiegelreduktion und periodischen
  Gittern; Lattice-FCS für `lattice-3d` (CPU, GPU); MPI in `m1pn_fast` für alle Fälle, bitgleich zu einem Rang;
  reproduzierbare Normen in der Potenziteration (dt unabhängig von der Rangzahl); MFEM-frei, MPI über InfiniBand.

## Offen

- MPI+GPU (zurückgestellt; für einen künftigen Cluster mit mehreren GPUs).

## Block 3 — erledigt (16.11.2026, Branch `feature/block3-numerics`)

- WENO an der Rundungsgrenze: Ursache ist eine Rückkopplung Sensor ↔ Stabilisierung bei q = 1 an glatten Extrema
  (nicht der Sensor auf glatten Daten); kostet auch Q1/Q3 Ordnung. q = 3 beseitigt sie; Entscheidung: Standard bleibt
  q = 1 (docs/ELEMENT_WENO.md, Abschnitt „WENO-Sensor an der Rundungsgrenze“).
- Eigene Monte-Carlo-Referenz `m1pn_lattice_mc` für Lattice 2D und 3D, Fehler ~1e-5 (docs/LATTICE_MC.md,
  `results/heterogeneous/lattice_mc_20261116/`). Literatur liegt innerhalb 0.04 % von MC; 3D-Lattice weit von
  Konvergenz (Masse Q1 112³ P7 −8.9 %).
- Strang-Splitting: Ordnung 2 bestätigt, Zeitfehler bei CFL 1 2–3 Größenordnungen unter dem Ortsfehler; keine höhere
  Ordnung nötig (docs/ELEMENT_WENO.md, Abschnitt „Zeitordnung mit Reaktion“).

## Laufend

- Konvergenzkampagne `tardis-lattice-mc` (Lattice 2D bis 1344², P35; 3D bis 224³, ± FCS) gegen die MC-Referenz.

## Block 4 — teilweise erledigt (Performance)

- Host-Fallbacks bei Q1 ohne FCS (bis 4e8, long double): Line Source P3 1024² 1189 s statt 154 s mit FCS;
  Abhilfe double-double auf der GPU (vom Nutzer zunächst verworfen).
- ~~3D: Host-Randpfad (Face-Terme) als Engpass.~~ Erledigt 16.11.2026 (Branch `feature/block4-host-boundary`):
  Face-Terme bei je Face konstantem Einstrom über vorab summierte Gewichte (CPU und GPU), PN-Randlast als
  cuBLAS-DGEMM auf der GPU; Lattice 3D 56³ 12–15× schneller (docs/ELEMENT_WENO_GPU.md, „Host-Randpfad“). Nächster
  Engpass: Ghost-Kernel auf der GPU (≈ 70 % der GPU-Zeit bei P15 in 3D).
- 3D mit FCS: Ausgabe wertet die exakte ungestoßene Dichte an allen Ausgabeknoten aus (56³: 12 min); Abhilfe wäre die
  Zeittabelle bzw. eine gröbere Ausgabe.

Regeln: Abweichungen bis 1e-8 erlaubt, keine mathematischen Änderungen ohne Freigabe; kein Push/Merge ohne Auftrag;
keine Spack-Installationen ohne Auftrag; Experimente auf eigenen Branches; Antworten auf Deutsch.
