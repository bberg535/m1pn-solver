# Statische hp-Blöcke, Trennebenen und GPU-Optimierung (18.11.2026)

Branch `feature/multiblock` (Worktree `.claude/worktrees/multiblock`), aufbauend auf `feature/graded-gpu`
(docs/GRADED_MESH.md). Der Entwurf der ersten beiden Abschnitte stammt aus `experiment/static-order-map`.
Alle Laufzeiten: tardis, A100, lattice-paul mit FCS, t = 3,2, gegen die eigene MC-Referenz
(`tmp/lattice_mc_field/lattice-paul_field224.csv`, Auswertung `scripts/element_weno/lattice_mc_field_compare.py`).

## Prüfung des Prototyps (Entwurf)

- Raum: je Winkelgrad ℓ ein stetiger Q1-Raum auf den Elementen mit N_K ≥ ℓ; Grad 0 und 1 überall und stetig.
- Fluss an Flächen gemischter Ordnung f = ½A_n(x+y) − ½(y−x) (Spuren mit Nullen ergänzt, nur auf die überzähligen
  Grade verteilt). Nachgerechnet: semidiskret dE/dt = −Σ_F ½‖y−x‖²_F (Volumenterme ½xᵀA_nx − ½yᵀA_ny, Flussterm
  (y−x)ᵀf; gemeinsame Grade sind stetig, also y−x nur in den Schwanzgraden). Erhaltung von ρ und **J** exakt
  (Grade ≤ 1 sind gemeinsames CG). Ordnungswechsel per L²-Projektion senkt die Energie, ρ und **J** bitgleich.
- Schwächen: Sensor ‖**J**‖ absolut (nicht skalierungsinvariant, blind für Anisotropie bei **J** = 0, Quadrupol);
  gemischte Ordnungen senken die stabile Schrittweite (Schachbrett P1/P15: C = 0,62 statt 0,89), bleibt unter dem
  M1-Deckel; Masse je Grad über unregelmäßige Elementmengen ist kein Tensorprodukt mehr (keine Linienlöser, MPI-Ketten,
  GPU-Kerne), Prototyp mit festem P15 langsamer als der vorhandene Kern (0,52 s gegen 0,22/0,37 s).

## Statische Ordnungskarte (Emulation, Entwurf)

`m1pn_fast --order-map FILE` (Studienoption, CPU, Materialfälle): Gitter von Ordnungen über die Box; der
Reaktionsschritt setzt Momente mit Grad > Knotenordnung (Maximum der Nachbarelemente) auf null. Spart keine Zeit, misst
die Genauigkeit einer Karte. Karten aus dem Spektrum einer Lösung: `scripts/element_weno/order_map_from_spectrum.py`
(Schwanz jenseits n ≤ tol · max ρ_c), `scripts/element_weno/angular_spectrum_scan.py`.

Lattice-paul FCS, 364², P21, gegen MC (`nx/tmp/order_map`, Karte aus der P21-Endlösung selbst):

| Karte | Zustand | Masse/Absorption/Rate % | Feld L1 | Median rel. |
|---|---|---|---|---|
| P11 überall | 27 % | −0,26 / +0,46 / +0,49 | 3,8e-3 | 0,70 % |
| P21 überall | 100 % | −0,19 / +0,32 / +0,26 | 2,6e-3 | 0,58 % |
| tol 1e-3, 7×7 | 62 % | −0,19 / +0,32 / +0,26 | 2,6e-3 | 0,60 % |
| tol 1e-3, 14×14 | 47 % | −0,19 / +0,32 / +0,26 | 2,7e-3 | 0,67 % |
| tol 3e-3, 7×7 | 44 % | −0,19 / +0,32 / +0,26 | 2,7e-3 | 0,64 % |
| tol 3e-3, 14×14 | 32 % | −0,19 / +0,32 / +0,27 | 2,7e-3 | 0,85 % |
| tol 1e-2, 7×7 | 19 % | −0,21 / +0,35 / +0,33 | 3,1e-3 | 1,3 % |

## Trennebenen (`--break-units on`)

Auf jeder Materiallinie (ganzzahlige Koordinate im Lattice) werden die Knoten verdoppelt
(`ElementWenoOptions::breaks`). Jede Seite behält ihren stetigen Q1-Raum; die Flächen auf der Linie koppeln wie ein
Blockrand: PN-Teil ½(A_n − I) M_face (u − u_partner) exakt über die Flächenmasse (konform, gleiche Knotenpositionen),
Limiter-Barzustand knotenweise mit dem Partnerknoten (`broken_face_low`), kein C6-Ghost. Kein Knoten mittelt damit
Absorber- und Streudaten.

Wirkung (364², P21): Absorption +0,32 → +0,12 %, Rate +0,26 → +0,11 %, Feld L1 2,6e-3 → 1,9e-3.

### Zeitschrittschranke

Mit Trennebenen bestimmt die Realisierbarkeitsschranke Δt ≤ m_i/λ_i (Theorem 3.1) den Schritt statt der
Hochordnungsschranke: GB 0,00197 statt 0,00302 (`realizability_step`, `high_order_step` in summary.json), also
1623 statt 1052 Schritte. Ursache: Ein Knoten auf der Trennebene hat nur die einseitige gelumpte Masse, die
weggefallene Kantenviskosität zur anderen Seite ersetzt der Flächenterm w(ū − u_i) mit etwa gleichem Gewicht
(nachgerechnet: Flächenterm = LLF-Fluss mit λ = 1, die Schranke ist für das Verfahren scharf). Dieselbe Schranke
begrenzt das gleichmäßige 896²-Gitter an seinen Außenrandknoten. Abhilfe nur mit Änderung der Mathematik (offen,
Entscheidung des Nutzers): (a) lokale Teilschritte nur für die Trennebenenknoten (Flussregister; Theorem pro
Teilschritt), (b) Δt nach der Hochordnungsschranke mit nachträglicher Prüfung, (c) Trennebenen nur im groben Bereich.

## Mehrblock-Treiber `m1pn_blocks`

Ordnungskarte (`--order-grid`, Rechtecke gleicher Ordnung zusammengefasst) und Gitterweite je Block (`--h-grid`),
CPU und GPU (`--gpu on`, je Block ein `ElementWenoGpu` im Gleichschritt). Konforme Blockgrenzen koppeln knotenweise
(`--nodal auto`: alle Blöcke gleiche Weite) wie Trennebenen; sonst Punktkopplung über die Spur des Nachbarn.
`--break-units on`: zusätzlich Trennebenen im Inneren der Blöcke.

- Prüfung: ein Block bitgleich zu `m1pn_fast`; CPU gegen GPU 2e-15 (49 Blöcke) bzw. 6,7e-16 (Karte mit Ordnungen
  1..7 und Trennebenen); knotenweise gegen punktweise Kopplung: 2,4e-5 / 3,7e-6 / 6,3e-7 bei 8/16/32 Elementen je
  Einheit (Diskretisierungsunterschied, ~h^2,5).
- Korrektur (b24fc99): Trennebenen-Flächen sind keine Blockgrenzen (`for_each_interface_node`, GPU-Datum-Schleife);
  vorher brachen Karten mit Trennebenen ab („Interface node without neighbour block“).

Ergebnisse gegen MC (A100):

| Lauf | Zeit | Zustand | Masse/Absorption/Rate % | Feld L1 | 95 % |
|---|---|---|---|---|---|
| 364², P21 überall | 51 s | 100 % | −0,081 / +0,116 / +0,112 | 1,9e-3 | 3,2e-2 |
| 364², Karte 7×7 | 74 s | 59 % | −0,082 / +0,116 / +0,119 | 1,9e-3 | 4,8e-2 |
| 364², Karte 14×14 | 138 s | 45 % | −0,081 / +0,115 / +0,118 | 2,0e-3 | 6,3e-2 |
| 896², P21 überall | 482 s | | −0,055 / +0,096 / +0,136 | 1,1e-3 | 2,1e-2 |
| 896², P35 Zentrum 3×3 | 589 s | | −0,031 / +0,052 / +0,081 | 8,4e-4 | 3,7e-2 |

Die Karten halten die integralen Größen, sind auf der GPU aber langsamer: Alle Kernel laufen auf dem
Legacy-Default-Stream, die Blöcke werden nacheinander ausgeführt, Start- und Wartezeiten je Block und Stufe
dominieren (71 Blöcke: ~0,5 ms je Block und Stufe). Abhilfe wäre ein Stream je Block und parallele Host-Threads;
zurückgestellt, weil GB P35 (unten) genauer und schneller ist. Ordnungskarten und GB-Tensorgitter lassen sich nicht
kombinieren (Blöcke haben isotrope Weite je Block).

## GPU-Optimierung für GB P35 (18.11.2026)

GB = `--cells-per-unit 32,64,128,128,128,64,32` (896 Zellen je Achse). Ausgangsstand mit Trennebenen: 543 s.

1. **Streifen-Kernel der Stabilisierung auch mit Trennebenen** (098a1f8). nsys (`nx/tmp/nsys35`): mit Trennebenen fiel
   die Stabilisierung auf den allgemeinen Knotenkern zurück (26 ms statt 9,3 ms je Aufruf). Der Streifenkern liest das
   Element links/rechts einer Knotenlinie jetzt aus einer Tabelle (`strip_elem`; eine Knotenkopie auf der Trennebene
   hat nur eine Seite), das 3×3-Fenster bleibt. CPU gegen GPU 2e-15 (gleichmäßig) / 5e-15 (abgestuft), GPU-Validierung
   29/29. GB P35 543 → 436 s, GB P21 216 → 166 s, Ergebnisse unverändert.
2. **Volle Zeilen nur für Randknoten außerhalb der Trennebenen** (b24fc99): Die Trennebenen brauchen auf dem Host nur
   die vier niedrigen Momente; bitgleich.
3. **Thread-Platzierung:** Mit `--cpus-per-task=8` liegen die Threads ohne `OMP_PLACES=cores` teils auf Hyperthreads;
   Ghost-Zeit 38 s statt 21 s (GB P35, t = 0,4). Für Messungen: 16 Kerne, `OMP_PLACES=cores`.
4. **Ghost-Spuren auf der GPU** (siehe unten).

Messfallen: Ein gleichzeitiger CPU-Job auf tardis verdoppelte die Host-Ghost-Zeit (erste Messung „87 % Ghost“ war
dadurch verzerrt). Ghost-Projektion komplett auf der GPU (`M1PN_GPU_GHOST=1`) ist bei P35 dreimal langsamer (viele
markierte Paare werden auf dem Host neu gerechnet).

## Ghost-Spuren auf der GPU (`M1PN_GPU_GHOST_TRACES=1`, Experiment, Standard aus)

Zeitbilanz GB P35 nach Schritt 1–3: Warten am Stufenanfang + Host-Ghost + Warten am Stufenende ≈ Rechenzeit, der
Ghost liegt auf dem kritischen Pfad (die GPU wartet in Phase B auf den Host). Zwei Drittel der Ghost-Zeit sind die
Spuren ψ = H_s · a_i (Richtungen × Momente mal Zustand je Ghost-Knoten), ein Drittel die Newton-Projektionen
(im Mittel eine Iteration, keine erweiterte Regel).

Umbau: Nach der Closure sammelt die GPU auf dem Neben-Stream die Zeilen der Ghost-Knoten je Seite und rechnet
T_s = H_s A_s als cuBLAS-DGEMM (eigenes Handle auf dem Neben-Stream; H_s mit dem Ghost-Filter σ multipliziert,
(H diag σ) a = H (σ ∘ a)); das Ergebnis geht asynchron in Pinned-Speicher (Event 2). Der Host liest in Phase B die
Spalten statt `SeparableTrace::evaluate` und rechnet die Projektion mit dem unveränderten Code. Mathematisch
identisch, nur andere Rundung der Spuren (Spuren an gegen aus: 4e-12 / 1,7e-10 in den Integralen, Job 162302).
Code: `gpu::ghost_traces_setup`, `gpu::ghost_traces` (src/element_weno_gpu.cu), `trace_ghost_` in
src/element_weno_gpu.cpp. Diagnose `M1PN_GPU_DIAG_SERIAL=1` (nur Zeitmessung): GPU wartet vor Phase B, `diag_device`
= Gerätearbeit, die Phase B überlappt.

Ergebnis (Jobs 162301/162302, GB P35, t = 0,4, 16 Kerne): **kein Gewinn** (23,2 → 24,2 s ohne, 38,6 → 40,3 s mit
Trennebenen). Seriell gemessen (`M1PN_GPU_DIAG_SERIAL=1`) überdeckt Phase B 20 s Gerätearbeit, der Host braucht mit
den GPU-Spuren nur noch 4,9 s: Nach Streifenkern und Thread-Platzierung ist die **GPU** der Engpass (Massenlösung
`k_line_solve` ~75 % der Speicherbandbreite, Streifenkern ~50 %). Daher Standard aus; der Code bleibt als Experiment.

## Spiegelreduktion auf abgestuften Gittern und mit Trennebenen

lattice-paul ist spiegelsymmetrisch in x um x = 3,5 (Absorberzellen ix ↔ 6 − ix), nicht in y (Zelle (3,1)). Bisher
war die Spiegelreduktion (`--mirror-reduction auto`) für abgestufte Gitter und Trennebenen gesperrt. Jetzt:

- `fast_main`: zusätzlich zur bestehenden Prüfung (Anfangsdaten, Material, Einstrom) müssen die Gitterebenen
  symmetrisch zur Mitte liegen und die Trennebenen symmetrisch sein, keine auf der Spiegelebene. Danach obere Hälfte
  der Ebenen (Ursprung exakt die Mittelebene) und Trennebenen umnummeriert (k − n/2). Ausgabe auf dem vollen Gitter
  (`full_planes`); auf gebrochenen Gittern wird der gespiegelte Punkt um ein ulp nach unten gesetzt, damit auf einer
  Trennebene dieselbe Seite wie im vollen Gitter ausgewertet wird (`evaluate` nimmt bei Gleichstand das obere Element).
- `ElementWeno`: Spiegelseiten auf abgestuften und gebrochenen Gittern erlaubt; WENO-Erweiterung des gespiegelten
  Nachbarn mit Breitenverhältnis 1. 1D-Massen und Dirichlet-Faktoren werden schon aus den Breiten assembliert.

Prüfung: CPU Spiegel gegen voll bei gleichem Δt (8,16,16,16,16,16,8, P7): Integrale 2e-10, Feld 2e-10, auch auf
den Trennebenen 4e-10 (ohne festes Δt weichen die Läufe um 1e-6 ab: die Hochordnungsschranke kommt aus einer
Potenziteration mit Zufallsstart, auf dem Halbgebiet leicht anders). GPU (Job 162329): CPU gegen GPU mit Spiegel
bestanden (mit/ohne Trennebenen), GPU Spiegel gegen GPU voll 2e-10.

GB, t = 3,2, A100, 16 Kerne (`OMP_PLACES=cores`), gegen MC:

| Lauf | Zeit vorher | mit Spiegel | Masse/Absorption/Rate % | Feld L1 |
|---|---|---|---|---|
| GB P35, Trennebenen | 543 s → 350 s | **187 s** | −0,026 / +0,042 / +0,050 | 7,7e-4 |
| GB P35, ohne | 226 s | **125 s** | −0,046 / +0,079 / +0,076 | 8,6e-4 |
| GB P21, Trennebenen | 216 s → 164 s | **91 s** | −0,055 / +0,096 / +0,133 | 1,1e-3 |
| GB P21, ohne | 110 s | **62 s** | −0,071 / +0,127 / +0,153 | 1,2e-3 |

Zum Vergleich KiT-RT (cl 0,005): S6 269 s, +0,06 / +0,05 / −0,02 %, Feld L1 4,3e-3; S14 1270 s, +0,04 / +0,08 /
+0,01 %, 1,5e-3. Hinweis: KiT-RT nutzt keine Spiegelreduktion; fair wäre, das im Paper zu nennen.

## GPU-Läufe von Lauf zu Lauf reproduzierbar (behoben)

Befund: `tests/gpu_validate.sh`, Fall `graded_lattice_q1_P7`, zehn identische GPU-Läufe lieferten zufällig eines
von drei Ergebnissen gegen die CPU (6,3e-13, 9,8e-13, 4,2e-7; Schwelle 1e-8), schon mit dem Binary von
`feature/graded-gpu` (Job 162342, `nx/tmp/race14`). Die CPU ist für 1–16 Threads bitgleich.

Ursache: `gpu::norm` (Potenziteration der Hochordnungsschranke, daraus Δt) summierte die Blockteilsummen mit
`atomicAdd` in zufälliger Reihenfolge; Δt schwankte in der letzten Binärstelle. Behoben: Jeder Block schreibt seine
Teilsumme, der Host summiert in Blockreihenfolge. Zwölf identische Läufe bitgleich (Job 162351), alle 23 GPU-Ausgaben
der Validierung zwischen zwei Läufen bitgleich (Job 162365; auch Lattice Q2 abgestuft, vorher ~1e-12 schwankend,
docs/GRADED_MESH.md).

Verbleibend und erklärt: CPU und GPU schätzen Δt mit verschiedener Summationsreihenfolge (ein bis zwei ulp
Unterschied). `graded_lattice_q1_P7` reagiert auf ein ulp in Δt mit einer diskreten Limiter-Entscheidung: die CPU mit
dem Δt der GPU weicht selbst um 4,16e-7 ab und stimmt dann mit der GPU auf 1,2e-12 überein. Diese Empfindlichkeit
gehört zum Verfahren (CPU wie GPU), liegt weit unter dem Diskretisierungsfehler. Die Validierung gibt dem GPU-Lauf
deshalb das Δt des CPU-Laufs (`dt_selected` aus config.json); 29/29 bestanden, zweimal.
