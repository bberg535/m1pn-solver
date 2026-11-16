# Konvergenzkampagne tardis (A100): Ergebnisse, Stand 16.11.2026

Kampagnen unter `nx/tmp/tardis_{paper,fcs,hetero,lattice}` im Worktree `gpu-campaign` (eingefroren, `tardis.py collect`;
je `REPORT.md`, `results.csv`, `figures/`, `runtimes.md`; versioniert in `results/tardis_202611/`). Aufbau und Effizienzschleife: [TARDIS_CAMPAIGN.md](TARDIS_CAMPAIGN.md). Fehler wie auf NX:
L2 der Dichte; „kinetisch" enthält den PN-Abschneidefehler, „Selbstdifferenz" vergleicht mit dem nächstfeineren Gitter.

## Status

| Kampagne | fertig | Laufzeit (Summe, eine A100) | Bemerkung |
|---|---|---|---|
| paper | 198 / 201 | 4.2 h | Ablation ohne Limiter scheitert (Kegel verlassen, erwarteter Befund); Crossing Q1 2048² P3/P7 verworfen (nicht nachgerechnet) |
| fcs | 60 / 60 | 2.2 h | |
| hetero | 61 / 61 | 0.95 h | |
| lattice | 12 / 18 | 2.1 h (ohne abgebrochene Läufe) | Lattice N = 7–45, 224²–896²; Rest abgebrochen (siehe unten) |

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

## Lattice-Konvergenz in N und Gitter (Kampagne `tardis_lattice`, 16.11.2026)

GPU, Q1, SSPRK(4,3), CFL 0.2, T = 3.2, ohne FCS; Binary `46141ef` (hohe PN-Ordnungen auf der GPU). 12 / 18 fertig:
P7, P15, P21 auf 224²/448²/896²; P35 auf 224²/448²; P45 auf 448². Abgebrochen/nicht gestartet: P7 und P15 1792²,
P21 1344², P35 896², P45 672²; P45 224² am Timeout (1800 s, T = 2.95). Versioniert in
`results/tardis_202611/lattice/`. Referenz Schotthöfer/Hauck Tab. 1 / (2π): Masse 2.08016, Absorption 1.11950,
Rate 0.73164 (4 Stellen, Rundung ±0.04 % / ±0.007 % / ±0.011 %).

Abweichung Masse / Absorption / Rate:

| N | 224² | 448² | 896² | Richardson (p = 2) |
|---:|---|---|---|---|
| 7 | −0.87 / +1.65 / +1.28 % | −0.61 / +1.16 / +1.05 % | −0.56 / +1.08 / +1.02 % | −0.55 / +1.05 / +1.00 % |
| 15 | −0.57 / +1.10 / +0.73 % | −0.22 / +0.43 / +0.40 % | −0.13 / +0.28 / +0.33 % | −0.10 / +0.22 / +0.30 % |
| 21 | −0.56 / +1.06 / +0.67 % | −0.17 / +0.34 / +0.28 % | −0.067 / +0.15 / +0.20 % | −0.034 / +0.09 / +0.17 % |
| 35 | −0.55 / +1.06 / +0.65 % | −0.15 / +0.31 / +0.21 % | — | |
| 45 | (Timeout) | −0.15 / +0.30 / +0.20 % | — | |

- **Gitter:** beobachtete Ordnung aus 224/448/896 für Masse und Absorption 2.5 (P7), 2.1 (P15), 2.0 (P21); Rate
  2.1–2.6. Q1 mit Material-Unstetigkeiten also ≈ zweite Ordnung in den Integralen.
- **N:** bei 448² ändert sich die Masse von P21 nach P35 um 1.8e-4 relativ, von P35 nach P45 um 8e-6 — die
  Momentenentwicklung ist ab P35 konvergiert; der Rest bei 448² (−0.15 %) ist Ortsfehler. Bei 224² ist schon P21
  konvergiert (P21–P35: 1e-5).
- **Grenzwert:** Richardson P21 plus Winkelkorrektur P21 → P35 aus 448² (additiv angenommen) ergibt Masse ≈ −0.02 %,
  Absorption ≈ +0.06 %, Rate ≈ +0.10 % gegen die Referenz. Die Masse liegt damit innerhalb der Rundung der Tabelle;
  Absorption und Rate liegen knapp darüber, im Bereich der Modellunterschiede (S22 gegen PN-Grenzwert,
  Anfangsdichte, Einstrommodell). Eine genauere Aussage braucht eine eigene Referenz (Block 3 in NEXT_STEPS).
  Kontrolle: Referenz Masse + Absorption = 3.19965 statt 3.2, d. h. die Tabelle selbst ist nur auf ≈ 1e-4 konsistent.
- **Konsistenz CPU/GPU:** gleiche Fälle auf NX (CPU, `results/lattice_fcs_20261116`, ohne FCS) stimmen auf die
  angegebenen Stellen überein (z. B. P15 448² −0.22 % / +0.43 %, P35 224² −0.55 % / +1.06 %).
- **Bilanz:** Massenbilanz ≤ 1.1e-10 relativ, ρ_min > 0 in allen Läufen; Masse + Absorption = 3.2 − 1.3e-5
  (Ausfluss über den Vakuumrand).
- **Laufzeit (eine A100):** 896²: 332 s (P7), 847 s (P15), 1536 s (P21); 448²: 582 s (P35), 1272 s (P45). Ab P35
  dominiert der Host-Randpfad (`gpu_timing.host_boundary` + `host_boundary_high`: P35 448² 509 s von 582 s, P45 448²
  1159 s von 1272 s; 0 Host-Fallbacks). Auffällig: P35 224² (463 s) ist kaum schneller als 448² (582 s), und P45 224²
  erreicht in 1800 s nicht das Ende, während P45 448² 1272 s braucht; bei 224² entfallen 385 s auf `host_boundary`
  (448²: 261 s). Ursache nicht geklärt (ggf. parallele Aufgaben im selben Job); siehe Block 4 (3D/Host-Randpfad).

## Effizienz: verbleibender Engpass

Q1 1024² ohne FCS: bis 4e8 Host-Fallbacks (long-double-Pfad für Basis-Balken / Schließung nahe am Strahl). Folge:
Line Source P3 1024² 1189 s gegen 154 s mit FCS (0 Fallbacks), Crossing P3 930 s gegen 287 s bei P7, Crossing 2048²
P3 ≈ 3.3 s/Schritt. Nächster Schritt: diesen Pfad auf der GPU in double-double rechnen (Genauigkeit ≥ long double).
