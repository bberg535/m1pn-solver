# Konvergenzkampagne auf tardis (A100), Stand 15.11.2026

Branch `experimental/gpu-campaign` (von `experimental/gpu-a100`, mit der GPU-Projektion aus `experimental/gpu-projection`).
Skript `scripts/element_weno/tardis_convergence.py`; es nutzt `nx_convergence.py` unverändert (Einfrieren, Validierung,
Auswertung, Plots) und ändert nur die Aufgabenmatrix, die Solver-Optionen der GPU-Läufe und die Einreichung.

## GPU-Pfad für alle Benchmark-Typen

- `m1pn_fast --gpu` (MFEM-Build mit `-DM1PN_ELEMENT_GPU=ON`): Zeitschleife auf der GPU, gleiche Fälle und Ausgaben wie
  ohne `--gpu`. Nicht auf der GPU: periodische Gitter (1D-Glattfall, läuft auf der CPU), Spiegelreduktion (GPU-Läufe
  mit `--mirror-reduction off`), DLRA.
- Reaktion auf der GPU (exakte lokale Lösung: Material σ_a, σ_s, Quelle, konstante Streuung/Absorption, Winkelfilter,
  Bilanz), FCS-Stoßquelle auf der GPU (Radialtabelle von ρ_u vom Host, Quadratur und Sammeln auf der GPU).
- Ghost-Projektion auf der GPU, wenn der Randaufwand groß ist (Ghost-Paare x Regelpunkte > 0.1 x Knoten x Momente;
  `M1PN_GPU_GHOST=0/1` erzwingt Host/GPU).
- Validierung CPU gegen GPU (dasselbe Binary): Abweichungen ≤ 1e-11 bis 1e-16 bei glattem Transport, Lattice 2D/3D,
  Crossing (auch mit Streuung/Filter), Line Source mit FCS bis T = 0.02; danach Rundungs-Chaos wie in
  ELEMENT_WENO_GPU.md (Line Source ohne FCS 1e-3, Q4-Lattice 1e-4; gleiche Größenordnung mit den alten generischen
  Kerneln; Fehler gegen die exakte Line-Source-Lösung auf 5-6 Stellen gleich).

## Effizienzschleife je Benchmark (tardis, cfl 0.2, nsys)

| Fall | vorher | nachher | Maßnahme |
|---|---|---|---|
| Crossing Q3 256², P7 | 125 s | 43 s | elementweise Stabilisierung (Shared Memory), Limiter in Knotengruppen (16) |
| Lattice Q4 112², P15 | 123 s | 64 s | dto. (25 Knoten, Gruppen von 32) |
| Line Source Q2 256², P15 | 88 s | 40 s | dto. (9 Knoten) |
| Glatter Transport Q2 256², P11 | 42 s | 20 s | dto. |
| Crossing Q1 1024², P15 | 210 s | 65 s | nur die vier M1-Momente ausgeben (`--sample-moments low`, die Auswertung braucht nur u0-u3) |
| Line Source mit FCS 256², P21 | 30 s | 11 s | Ghost-Projektion auf der GPU (hohe Regelpunktzahl) |
| Lattice 3D Q1 56³, P7 | 24 s | 14 s | Ghost-Projektion auf der GPU, Stabilisierung 3D |

Verbleibend: gebänderte Linienlösungen (Q2-Q4: Bandbreite p), in 3D der Host-Rand (Face-Terme).

## Aufgabenmatrix

| Kampagne | Familien | Gitter, PN |
|---|---|---|
| `tardis-paper` (201) | glatt 1D periodisch (CPU), glatter Transport am Rand, Crossing, Line Source ohne FCS, Zeitkonvergenz, Ablation | Q1 bis 1024² (2048² für P ≤ 7), Q2 bis 512², Q3 bis 256²; P3, P7, P11, P15 |
| `tardis-fcs` (60) | Line Source mit FCS | Q1 32²-1024², Q2 16²-512²; P3, P7, P11, P15, P21 |
| `tardis-hetero` (61) | Lattice 2D (Q1-Q4), Lattice 3D (Q1, Q2) | 2D bis 448² (896² für P7); P3-P15; 3D P3, P7 |

Kein Shadow-Test. Gleiche Fehlerdefinitionen wie auf NX (`nx_convergence.py collect`); FCS ist eine eigene Kampagne,
damit die Konvergenzordnungen nicht mit den Läufen ohne FCS vermischt werden. Laufzeiten: `runtimes.md` jeder Kampagne
(Wandzeit der Aufgabe, Solverzeit, Schritte, Momente).
