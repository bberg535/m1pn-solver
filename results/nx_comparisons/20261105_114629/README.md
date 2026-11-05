# Erste Vergleiche der NX-Studie: Standardvariante B

Stand 05.11.2026, 11:52 Uhr Europe/Berlin: **43 von 48 Aufgaben erfolgreich
abgeschlossen und erneut anhand der gespeicherten Eingabe-Hashes geprüft**.
Ein regulärer 3D-Lauf läuft; vier Aufgaben warten im Long-Array 156706 auf das
Ende des ursprünglichen Arrays 156674. Es gibt keine fehlgeschlagenen Aufgaben.
Die [Statusaufnahme](campaign_status.json) enthält die Einzelaufgaben.

| Fall | Räumliche Läufe fertig | Noch ausstehend |
|---|---:|---|
| 2D Crossing | 17/18 | 128²/P15 und dessen Δt/2-Lauf |
| Lattice | 15/15 | Δt/2-Lauf 112²/P11 |
| 3D Crossing | 11/12 | 32³/P9 läuft; dessen Δt/2-Lauf wartet |

Alle Vergleiche verwenden dieselbe Standardvariante B, 16 Threads, dieselbe
Endzeit und dieselbe Schrittweite innerhalb eines Falls. Die Daten werden
nicht mit den lokalen Läufen mit anderem Δt vermischt. Die Solverdatei,
Kampagnenaufgaben und Rohfelder bleiben unverändert.

## 2D Crossing

[Fehler nach Gitter und PN-Ordnung](crossing/crossing_error_convergence.png)
([PDF](crossing/crossing_error_convergence.pdf)),
[Dichte und signierter Fehler auf 128²](crossing/crossing_128_density_and_error.png)
([PDF](crossing/crossing_128_density_and_error.pdf)),
[Mittelschnitt](crossing/crossing_128_midline.png),
[Fehler gegenüber beobachteter Solverlaufzeit](crossing/crossing_accuracy_vs_solver_seconds.png).
Die vollständigen Zahlen stehen in [crossing_metrics.csv](crossing/crossing_metrics.csv).

Bei 128² sinkt der Dichte-L2-Fehler zur kinetischen Referenz von **0,09249
(P3)** über **0,01903 (P7)** auf **0,004543 (P11)**. Auf dem groben 32²-Gitter
liegen P11 und P15 dagegen bei etwa 0,0216–0,0217. Gitterverfeinerung senkt den
Referenzfehler nicht für jede feste PN-Ordnung: P7 ergibt 0,02581 auf 32²,
0,01697 auf 64² und 0,01903 auf 128².

64²/P15 und 128²/P11 liefern fast denselben Dichte-L2-Fehler von etwa 0,00454.
Die beobachteten Solverzeiten betragen etwa 94 beziehungsweise 140 Minuten.
Das sind Einzelmessungen der laufenden Studie, kein kontrollierter
Geschwindigkeitsvergleich mit Wiederholungen.

## Lattice

[Dichtekarten P3/P7/P11 auf 112²](lattice/density_maps_P3_P7_P11_N112.png)
([PDF](lattice/density_maps_P3_P7_P11_N112.pdf)),
[Mittelschnitte](lattice/density_lineout_y3_5_N112.png),
[Gitter- und PN-Selbstvergleiche](lattice/density_difference_metrics.png)
([PDF](lattice/density_difference_metrics.pdf)).
Siehe [Differenzmetriken](lattice/density_difference_metrics.csv),
[geprüfte Läufe](lattice/verified_runs.csv) und [Einzelbefunde](lattice/findings.md).

Alle 15 räumlichen Läufe erreichen t=3,2. Die Material- und Quellintegrale
stimmen auf allen Gittern: Fläche 49, Quelle 1, Absorption 110, Streuung 38.
Auf 112² fällt die gewichtete Dichte-L2-Abweichung zur berechneten
P11-Lösung von etwa 0,01031 (P3) auf 0,0005055 (P9).
Die Unterschiede 56→112 liegen je nach Ordnung bei etwa 0,00661–0,00701 und
sind kleiner als die Unterschiede 28→56.

Für Lattice liegt keine exakte Lösung vor. P11 ist eine endliche
Vergleichslösung. Die Gitterdifferenzen werden nach multilinearer Fortsetzung
der groben Lösung auf die feinen Knoten mit den dortigen Knotenmassen und
Normierung durch die Fläche 49 gemessen.

## 3D Crossing

[Fehler nach Gitter und PN-Ordnung](three_d/convergence.png)
([PDF](three_d/convergence.pdf)),
[Dichte- und Fehlerschnitte bei z=0,5](three_d/density_slices.png)
([PDF](three_d/density_slices.pdf)),
[Mittelschnitt y=z=0,5](three_d/lineouts.png)
([PDF](three_d/lineouts.pdf)).
Die Zahlen stehen in [metrics.csv](three_d/metrics.csv).

Auf 32³ fällt der Dichte-L2-Fehler von **0,02121 (P3)** auf
**0,009644 (P5)** und **0,008966 (P7)**. Auf 8³ verbessert eine höhere
PN-Ordnung den Referenzfehler hingegen nicht. Die Schnitte zeigen die
tatsächlichen Knotenebenen bei z=0,5 beziehungsweise y=z=0,5; die Referenz
wird an denselben Koordinaten ausgewertet.

## Aussagebereich und Herkunft

Die Fehler zum kinetischen freien Transport enthalten PN-Modell-, räumliche,
Rand- und Limitereffekte. Die L1/L2-Dichtenormen verwenden gelumpte
Knotenmassen und die Normierung durch das jeweilige Gebietsvolumen; Linf ist
das Maximum über die Knoten. Daraus folgt keine allgemeine räumliche oder
winkelbezogene Konvergenzordnung. Alle drei Δt/2-Prüfungen stehen noch aus;
eine abschließende zeitliche Fehlerbewertung ist deshalb nicht möglich.

Die Provenienz steht in [Crossing](crossing/snapshot.json),
[Lattice](lattice/provenance.json) und [3D](three_d/provenance.json).
Alle Grafiken liegen als PNG und PDF vor. Die Skripte stehen unter
`scripts/plot_crossing_snapshot.py`, `scripts/plot_lattice_snapshot.py` und
`scripts/plot_3d_snapshot.py`; mit `--campaign` und einem neuen `--output`
lassen sich weitere Momentaufnahmen ohne Änderung der Kampagnendaten erzeugen.
