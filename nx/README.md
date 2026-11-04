# Clusterstudie mit PN-Low-Order-Zeitableitung

Diese Studie überträgt den Versuchsplan aus
`M1PN_NX_NonExtrapolated_20261104/nx` auf **Standardvariante B**:
`m1pn_mass_correction`, Modell R2, SSPRK3, diagonale Masse und
PN-Viskositätsfaktor 1. Biharmonik- und KV-Koeffizient sind null.
Der Solver-Kern unter `../src` bleibt unverändert.

Die physikalischen Randflüsse bleiben erhalten. Die besondere Fortsetzung der
Randprädiktoren aus Variante C wird hier nicht verwendet. `mass_corrections=0`
bezeichnet in den Metadaten die Anzahl der alten AFC-Iterationsrunden;
`pn_mass_correction_applications=1` dokumentiert die neue Korrektur pro Stufe.

## Umfang

| Fall | Zellen je Achse | PN-Ordnungen | Endzeit | Reguläre Läufe |
|---|---|---|---|---|
| 2D Crossing | 32, 64, 128 | 3, 5, 7, 9, 11, 15 | 0,32 | 18 |
| 2D Lattice | 28, 56, 112 | 3, 5, 7, 9, 11 | 3,2 | 15 |
| 3D Crossing | 8, 16, 32 | 3, 5, 7, 9 | 0,2 | 12 |

Hinzu kommt je Fall ein Lauf mit halbierter Schrittweite bei höchster Ordnung
und feinstem Gitter: insgesamt **48 Aufgaben**. Der Vorbereitungsschritt baut
mit den vorhandenen Bibliotheken, führt CTest und 45 neue T=0-Proben aus und
friert je Fall die gemeinsame Schrittweite mit Sicherheitsfaktor 0,2 ein.
Alte Biharmonik-Schrittweiten werden nicht übernommen. Fehler und Abbrüche
werden in den Aufgabenstatus und der abschließenden Auswertung ausgewiesen.

## Start

Vom Projektordner `M1PN_MassCorrection_20261104`:

```bash
python3 nx/scripts/start_mass_correction.py --dry-run
python3 nx/scripts/start_mass_correction.py
```

Der zweite Befehl reicht Vorbereitung, abhängiges Array und abhängige
Auswertung ein. Maximal zwei Aufgaben laufen gleichzeitig mit je 16 Threads
und 64 GiB angefordertem Knotenspeicher; das interne Solverbudget beträgt
48 GiB. Vorbereitung und einzelne Aufgaben haben acht Stunden Slurm-Zeit,
der Solverprozess sieben Stunden. Ein vorhandener Launch-Nachweis verhindert
versehentliches doppeltes Einreichen.

Job-IDs und genaue Einreichungsbefehle stehen in `results/launch_*.json`;
`results/current_launch.json` zeigt auf den aktuellen Nachweis. Logdateien
liegen unter `tmp/slurm_logs`, die vollständigen Daten unter dem im Nachweis
genannten Kampagnenpfad. Die Vorbereitung muss erfolgreich enden, bevor das
Array beginnt. Die Auswertung startet nach Ende aller Array-Aufgaben, auch
wenn einzelne Läufe scheitern.

Die vorhandene MFEM/MPI-Installation ist für Zen3 gebaut; das Profil fordert
deshalb `--constraint=nx` an. Der angebotene Knoten `worldgames` hat Zen2.
Für ihn ist kein kompatibler vorhandener MFEM-Stack dokumentiert oder im
lokalen Installationsbaum gefunden worden. Es werden keine Pakete installiert.

## Einzelne Schritte und lokale Prüfung

```bash
bash nx/scripts/submit_mass_correction.sh --dry-run prepare production "$PWD/nx/tmp/my_mc_study"
bash nx/scripts/submit_mass_correction.sh prepare production "$PWD/nx/tmp/my_mc_study"
# Erst nach erfolgreicher Vorbereitung:
bash nx/scripts/submit_mass_correction.sh run "$PWD/nx/tmp/my_mc_study" 2
bash nx/scripts/submit_mass_correction.sh collect "$PWD/nx/tmp/my_mc_study" ARRAY_JOB_ID
```

`nx/scripts/smoke_local.sh /path/to/m1pn_mc_study NEW_OUTPUT` prüft die gesamte
Pipeline mit 15 kleinen Aufgaben. Die lokale Validierung steht unter
`results/`; sie ersetzt die noch laufende Produktionsstudie nicht.

Die Crossing-Referenzen sind analytische kinetische Transportmomente und
enthalten gegenüber dem endlichen PN-Modell Modellfehler. Lattice verwendet
die unveränderten Material- und Quelldefinitionen aus dem ursprünglichen
Studien-Einstiegspunkt. Erfolgreiche endliche Läufe belegen keine allgemeine
Energie- oder Gleitkomma-Realisierbarkeitsgarantie.

## Einreichung am 04.11.2026

Die angeforderte Standard-B-Studie ist eingereicht: Vorbereitung **156673**,
Produktionsarray **156674** (48 Aufgaben, maximal zwei gleichzeitig) und
Auswertung **156675**. Beim Statuscheck wartete die Vorbereitung auf verfügbare
NX-Knoten; Array und Auswertung warteten auf ihre Abhängigkeiten.
Der [Einreichungsnachweis](results/launch_20261104_161438_760701.json) enthält
die Befehle, Job-IDs, Methode und den zuletzt beobachteten Zustand.
Lokal sind vier Tests der Cluster-Pipeline und alle 15 Smoke-Aufgaben bestanden.

## Längere Aufgaben gezielt auf long verschieben

`python3 nx/scripts/submit_long_tasks.py --dry-run` vom Projektordner aus zeigt
die gezielte Verschiebung der vier noch nicht gestarteten Aufgaben 17 und 45–47.
Ohne `--dry-run` wird ein zunächst gehaltener Long-Array-Auftrag eingereicht,
der nach Ende des bisherigen Arrays startet. Danach wird die Abhängigkeit des
bestehenden Collectors um das neue Array ergänzt; nur die vier wartenden
Originalaufgaben werden entfernt und das neue Array freigegeben. Die Zahl
gleichzeitig rechnender Aufgaben bleibt dadurch insgesamt auf zwei begrenzt.

Das Long-Profil verwendet weiterhin 16 Threads, NX und 64 GiB Slurm-Speicher,
aber 24 Stunden Slurm-Zeit und 23 Stunden internes Timeout. Die eingefrorenen
Solverdateien, Aufgaben, Schrittweiten und der Studienplan bleiben erhalten.
Das Laufskript übergibt das längere Timeout als Ausführungsoption; laufende und
abschließende Statusdateien dokumentieren `effective_timeout_seconds` und
`timeout_source`. Ohne Override gelten weiterhin die sieben Stunden des Plans.

Der Vorgang wird unter `results/long_migration_*.json` protokolliert. Ein
vorhandener Verschiebungsnachweis verhindert eine doppelte Einreichung.
Bereits gestartete Aufgaben werden abgewiesen, damit vorhandene Ergebnisse
oder Teilergebnisse erhalten bleiben.
