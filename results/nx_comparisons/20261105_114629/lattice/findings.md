# Erste NX-Lattice-Vergleiche

Alle 15 räumlichen Läufe (N=28, 56, 112; PN3, 5, 7, 9, 11) wurden über `analysis.task_result` geprüft. Die eingefrorene Taskliste stimmt mit ihrem Manifest-Hash überein; die gespeicherten Eingabe-Hashes, Solver-Metadaten und Lattice-Materialintegrale stimmen ebenfalls. Gemeinsame Endzeit ist t=3.2, gemeinsames Δt=0.0005209181181832981.

Bei N=112 sinkt der Dichteunterschied zur PN11-Referenz über PN3→PN9 von L2=0.0103138 auf 0.000505495; PN11 selbst hat definitionsgemäß null Differenz. PN11 ist eine endliche Winkelordnung und keine exakte Lösung.
Für die Verfeinerung 56→112 ist die kleinste massengewichtete L2-Differenz PN5 (0.00661174), die größte PN3 (0.00700658). Das sind Same-order-Felddifferenzen nach multilinearer Interpolation auf die feineren Knoten, keine Fehler gegen eine exakte Lösung.

Der zusätzliche Lauf N=112, PN11 mit halbiertem Δt ist derzeit **pending**. Daher enthalten diese ersten Vergleiche keine zeitliche Fehlerabschätzung und begründen keine formale Konvergenzordnung.

Dichtekarten verwenden eine gemeinsame lineare Farbskala. Alle L1/L2-Werte sind Dichte-only, durch das Domänenvolumen normiert und mit den feineren Läufen gespeicherten lumped nodal masses gewichtet. Der PN11-Vergleich ist als endliche Referenz gekennzeichnet.

Dateien: `density_maps_P3_P7_P11_N112.png/.pdf`, `density_lineout_y3_5_N112.png/.pdf`, `density_difference_metrics.png/.pdf`, `density_difference_metrics.csv`, `density_lineout_y3_5_N112.csv`, `verified_runs.csv`, `provenance.json`.
