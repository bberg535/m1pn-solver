# Reines P3 gegen gekoppeltes M1PN: Crossing auf 128²

Der lokale PN-Lauf ist abgeschlossen: **128×128 Zellen, P3, T=0.32,
4954 SSPRK3-Schritte**, `dt=6.459426725878078e-05`, acht Threads. Der reine
PN-Pfad verwendet den Standard-B-Zieloperator mit PN-Low-Order-Prädiktor und
einer lokalen Massenkorrektur. M1-Abschlussrelation, Kegellimiter und
Synchronisierung der niedrigen Momente sind abgeschaltet. Die physikalische
LLF/Onsager-Randbehandlung samt Einstrahlung bleibt enthalten.

## Ergebnis

| Dichtefehler zur kinetischen Referenz | PN ohne Kopplung | Gekoppeltes M1PN |
|---|---:|---:|
| L1, mit Knotenmassen gewichtet | 0.075444 | 0.032800 |
| L2, mit Knotenmassen gewichtet | 0.119736 | 0.092487 |
| Maximum über die Knoten | 0.487962 | 0.884386 |

Ohne M1-Kopplung entstehen deutliche negative Dichten: **min ρ = −0.251128**,
4967 von 16641 Knoten sind negativ. 6958 Knoten verletzen den M1-Kegel der
niedrigen Momente. Das ist im PN-Pfad erlaubt und wird nicht korrigiert.
Der gekoppelte Vergleich bleibt positiv (min ρ ≈ 2.464×10⁻⁵).
Die L1- und L2-Fehler sind beim gekoppelten Lauf kleiner; der maximale
Knotenfehler ist beim reinen PN-Lauf kleiner.

Die unabhängige Nachprüfung des gespeicherten Endfelds besteht mit einer
maximalen globalen Q-Bilanzabweichung von **3.502×10⁻¹⁵**. Sie bestätigt
endliche Felder und Erhaltung aller PN-Komponenten einschließlich Randbilanz,
keine Positivitäts- oder Realisierbarkeitsgarantie. M1-Abschluss-, Limiter- und
Kegelabbruchzähler sind null.

## Plots und Daten

- [Dichtekarten und signierte Fehler](plots/pn_density_and_errors.png) ([PDF](plots/pn_density_and_errors.pdf))
- [Mittelschnitt y=0.5](plots/pn_midline_y_0_5.png) ([PDF](plots/pn_midline_y_0_5.pdf), [CSV](plots/pn_midline_y_0_5.csv))
- [Alle Fehlermetriken und Normierungsfaktoren](plots/pn_comparison_metrics.csv)
- [Datenbindung und Prüfsummen](plots/provenance.json), [Validierung](validation.json), [unabhängige Gegenprüfung](audit.json), [CTest-Protokoll](ctest.log)
- [Konfiguration](config.json), [Solverdiagnostik](summary.json), [Endfeldprüfung](verification.json)

Beide Läufe haben bitweise identische Anfangsfelder und Knotenmassen sowie
dieselbe Geometrie, Endzeit, feste Schrittweite und Schrittzahl. Die Dichtekarten
verwenden gemeinsame Skalen und enthalten negative Werte. Die kinetische
Referenz ist keine exakte Lösung des endlichen PN-Systems; ihre Fehler enthalten
auch PN-Modellfehler. Die Normen sind durch das Einheitsgebiet normiert.

## Lauf und Reproduktion

Der erfolgreiche Lauf benötigte lokal 890.5 Sekunden insgesamt, davon
751.8 Sekunden im Schrittlöser. Der maximale gemeldete Speicherverbrauch
betrug **135.9 MiB**. Der NX-Vergleich lief auf anderer Hardware mit 16 Threads;
daraus lässt sich kein kontrollierter Geschwindigkeitsvorteil ableiten.
Die PN-spezifischen Untertimer sind in diesem zusätzlichen Pfad nicht
instrumentiert; ihre Nullwerte belegen keine verschwindenden Rechenkosten.

Ein erster Versuch wurde nach 4540 Schritten mit SIGTERM beendet, ohne
numerische Fehlermeldung oder Endfeld. Er wird [separat dokumentiert](interrupted_attempt.json)
und ist nicht Teil der Fehlerauswertung. Der erfolgreiche Neustart lief als
eigenständiger lokaler Prozess ([Startprotokoll](launch.json)). Die verwendete
Binary wurde vor beiden Versuchen eingefroren; [Quell- und Binary-Hashes](source_record.json)
halten auch die gleichzeitig im Arbeitsbaum vorhandenen Core-Optimierungen fest.

```bash
python3 scripts/run_local_progress.py --coupling pn-only \
  --elements 128 --pn-order 3 --time .32 \
  --dt 6.459426725878078e-05 --threads 8

python3 scripts/plot_pn_comparison.py \
  --pn-run tmp/local_PN_B_20261105_125048_939872 \
  --coupled-run nx/tmp/pn_mc_production_20261104_161438_760701/runs/crossing_N128_P3 \
  --output results/pn_comparisons/<neuer_Ordner>
```

Das rohe PN-Endfeld liegt unter
`tmp/local_PN_B_20261105_125048_939872/final.csv`.
