# Konvergenzkampagne tardis (A100): Ergebnisse, Stand 16.11.2026

Kampagnen unter `nx/tmp/tardis_{paper,fcs,hetero}` (eingefroren, `tardis.py collect`; je `REPORT.md`, `results.csv`,
`figures/`, `runtimes.md`). Aufbau und Effizienzschleife: [TARDIS_CAMPAIGN.md](TARDIS_CAMPAIGN.md). Fehler wie auf NX:
L2 der Dichte; „kinetisch" enthält den PN-Abschneidefehler, „Selbstdifferenz" vergleicht mit dem nächstfeineren Gitter.

## Status

| Kampagne | fertig | Laufzeit (Summe, eine A100) | Bemerkung |
|---|---|---|---|
| paper | 198 / 201 | 4.2 h | Ablation ohne Limiter scheitert (Kegel verlassen, erwarteter Befund); Crossing Q1 2048² P3/P7 verworfen (nicht nachgerechnet) |
| fcs | 60 / 60 | 2.2 h | |
| hetero | 61 / 61 | 0.95 h | |

2048²: erster Versuch an der festen Fallback-Liste (2^16) gescheitert, behoben in `28d07a0` (keine numerische Änderung,
Manifest-Nachtrag, altes Binary `bin/fast.pre-capacity`); zweiter Versuch am geschätzten Timeout (P7 bis T = 0.23 in
3.6 h, P3 bis T = 0.03 in 1.2 h, Host-Fallbacks); danach verworfen. Gescheiterte Versuche in `status_failed_{capacity,timeout}/`.

## Konvergenz

**Glatt, 1D periodisch (exakte PN-Lösung, CPU):** EOC Q1 2.0, Q2 3.0, Q3 4.5→3.4, Q4 ≈ 5.0 bis 128² (3.2e-12).
Q4 256²: 4.8e-8 — WENO springt an der Rundungsgrenze an (min γ 0.992). Gegenprobe: ohne Stabilisierung 1.2e-12,
`--weno-linear 0` 5.2e-13; also kein Diskretisierungsdefekt.

**Glatter Transport mit Rand (Q2, 8²-512²):** EOC 2.00 auf allen Stufen für P7, P11, P15 (512²: 7.2e-7); P3 sättigt bei
4.8e-6 (PN-Abschneidung). Laufzeit 512²: 70 s (P3), 124 s (P7), 260 s (P11), 530 s (P15).

**Crossing (kinetisch, T = 0.32):** Sättigung am PN-Fehler — P3 0.4-0.7, P7 0.09-0.17, P11 0.023, P15 0.003; für P3/P7
wächst er leicht mit der Auflösung (die Strahlartefakte von PN werden schärfer aufgelöst). Selbstdifferenzen fallen bis
zur Sättigung, P15 Q2: 1.7e-1 → 8e-3 → 1.9e-3 → 6e-4 → 4e-4. Laufzeit Q1 1024²: 287 s (P7), 468 s (P11), 738 s (P15),
930 s (P3, Host-Fallbacks).

**Line Source ohne FCS:** stagniert bei 1-3 (Strahlartefakte), Q1 1024² 1056-1397 s.

**Line Source mit FCS:** stagniert am PN-Fehler 0.35 (P3), 0.17 (P7), 0.10 (P11), 0.064 (P15), 0.032 (P21) — 10-30x
genauer als ohne FCS. Laufzeit Q1 1024²: 154 s (P3), 294 s (P7), 526 s (P11), 864 s (P15), 1640 s (P21).

**Zeitkonvergenz (Crossing und glatt, mit Streuung):** EOC 2.0 (Strang mit nichtkommutierender Reaktion; SSPRK(4,3)
allein ist dritter Ordnung).

**Ablation (Crossing Q2 256², P7):** voll / ohne WENO / SSPRK3 gleich 0.132, WENO-Gewicht 0.2: 0.129; SSPRK3 braucht
halben Zeitschritt (67 s statt 44 s); ohne Limiter Abbruch.

**Lattice 2D (gegen Schotthoefer/Hauck Tab. 1, kinetisch S22):** Abweichung fällt monoton mit dem Gitter und mit PN;
P15 Q1 448²: Masse −0.22 %, Absorption +0.43 %; P7 Q1 896²: −0.56 % / +1.08 % (350 s). Bei gleichen Freiheitsgraden
ist Q1 genauer als Q2-Q4 (Material-Unstetigkeiten). Lattice 3D (eigener Fall): Bilanz ≤ 2e-11, bis 1.4e6 Knoten in
146-382 s. Massenbilanz überall ≤ 1e-10, ρ_min > 0.

## Effizienz: verbleibender Engpass

Q1 1024² ohne FCS: bis 4e8 Host-Fallbacks (long-double-Pfad für Basis-Balken / Schließung nahe am Strahl). Folge:
Line Source P3 1024² 1189 s gegen 154 s mit FCS (0 Fallbacks), Crossing P3 930 s gegen 287 s bei P7, Crossing 2048²
P3 ≈ 3.3 s/Schritt. Nächster Schritt: diesen Pfad auf der GPU in double-double rechnen (Genauigkeit ≥ long double).
