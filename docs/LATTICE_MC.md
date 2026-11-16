# Monte-Carlo-Referenz für den Lattice (2D und 3D)

Eigene kinetische Referenz für `lattice-paul` und `lattice-3d` (Block 3, 16.11.2026). Anlass: Die Abweichungen von
Element-WENO gegen Schotthöfer/Hauck Tab. 1 liegen bei 0.1–0.3 %, die Tabelle hat nur 4 Stellen und andere
Anfangs-/Einstromdaten; für `lattice-3d` gab es bisher gar keine Referenz.

Programm `m1pn_lattice_mc` (`tools/lattice_mc.cpp`), unabhängig vom Solver. Es teilt mit ihm nur die
Mengendefinitionen (`src/benchmark_cases.hpp`) und benutzt für die Validierung die ungestoßene Analytik
(`src/uncollided_lattice*.hpp`).

## Modell

Kinetische Transportgleichung mit Geschwindigkeit 1, isotroper Streuung, zellweise konstanten σ_a, σ_s (Einheitszellen),
isotroper Quelle q0 = 1 auf der Quellzelle für t ∈ [0, T] und Vakuum außerhalb [0,7]^d (nichts tritt wieder ein).
Anfangsdichte `--background` (Standard 1e-10 wie im Solver), getrennt simuliert und addiert.
`lattice-paul`: Felder hängen von (x, y) ab, Richtungen auf S² (in z unendliche Platte), wie Solver und Analytik.
Einheiten wie `summary.json` (ρ = ∫_{S²} ψ dΩ; Literatur ×1/(2π)).

## Schätzer

Jede Historie startet mit Gewicht 1 bei t0 ~ U(0, T), x ~ U(Quellzelle), Ω ~ U(S²) und repräsentiert q0 |S| T / N.

- **Implizites Einfangen:** Das Gewicht fällt auf jedem Segment exakt mit exp(−σ_a s). Streuorte werden mit einer
  optischen Tiefe in σ_s über die Zellgrenzen hinweg gezogen; Zellgrenzen werden exakt geschnitten.
- **Tallies** zu Zeitpunkten t_k (Standard 32 äquidistante bis T): Masse ∫ρ(t_k) (Gewicht bei t_k), Gesamtabsorption
  ∫₀^{t_k}∫σ_aρ (verlorenes Gewicht, segmentweise exakt), Absorptionsrate ∫σ_aρ(t_k), Ausfluss, eingespeiste Masse;
  dieselben Größen für den ungestoßenen Anteil (vor der ersten Streuung).
- **Bilanz:** Masse + Absorption + Ausfluss = eingespeiste Masse gilt für jede Historie exakt (geprüft, ≤ 2.2e-16).
- **Statistik:** B = 1000 Batches; Mittel und Standardfehler aus den Batchmitteln.
- **Reproduzierbar:** Zählerbasierter Zufallsstrom je (Seed, Strom, Batch, Historie) (SplitMix64); Batches werden in
  fester Reihenfolge reduziert. Ergebnis bitgleich für 1 und 32 Threads (geprüft).

Laufzeit: 1e9 Historien in 19 s (2D) bzw. 24 s (3D) auf 32 Kernen (NX).

```bash
m1pn_lattice_mc --case lattice-paul --time 3.2 --histories 4e10 --batches 1000 --seed 20261116 --output mc.json
m1pn_lattice_mc --case lattice-3d ... --check-uncollided 16 --check-order 6 --check-times 1,2,3.2   # Validierung
```

## Validierung

1. **Ungestoßener Anteil gegen die Analytik** (`src/uncollided_lattice*.hpp`, unabhängiger Code: Strahlintegrale mit
   Gauß-Quadratur; Ortsintegral im MC-Programm, `--check-uncollided`):
   - 2D, 4e9 Historien, 32 Panels × 6 Punkte je Zelle: Masse und Rate bei t = 1, 2, 3.2 innerhalb |z| ≤ 1.6
     (relativ ~3e-5). Mit 16 Panels lag die Rate bei T noch bei z = 3; das war Quadraturfehler.
   - 3D: Die Quadratur konvergiert langsam (steiles Absorberprofil σ = 10 direkt neben der Quelle) monoton auf MC zu:
     Rate bei t = 1 mit 2/3/4/6 Panels 0.5565 / 0.5681 / 0.5723 / 0.5755, Richardson (4/6) ≈ 0.5780, MC 0.5778;
     Masse 0.40775 / 0.40845 / 0.40881 / 0.40891, extrapoliert ≈ 0.40895, MC 0.40898.
   - Die ungestoßene 3D-Masse bei T der FCS-Läufe (Solver-Quadratur, 14³/28³/56³: 0.4230 / 0.4299 / 0.4325) läuft
     auf den MC-Wert 0.43358 zu.
2. **Literatur:** Schotthöfer/Hauck Tab. 1 / 2π (S22, ψ0 = 0, 4 Stellen) weicht von MC um −0.033 % (Masse),
   +0.036 % (Absorption) und +0.006 % (Rate) ab, also innerhalb der Tabellenrundung bzw. ihrer eigenen Konsistenz
   (Masse + Absorption = 3.19965 statt 3.2).
3. **Bilanz** je Historie ≤ 4.4e-16, gesamt ≤ 5e-12; **Threads** 1 gegen 32 bitgleich; **Seed** 99 gegen 20261116
   innerhalb von 2 Standardfehlern.
4. ctest `lattice_mc` (`tests/lattice_mc_check.py`, 2e5 Historien): Bilanz, Thread-Unabhängigkeit, Literatur
   innerhalb 5σ.

## Ergebnisse

Referenzwerte bei T = 3.2 (4e10 Historien; `results/heterogeneous/lattice_mc_20261116/`):

| Fall | Masse | Absorption gesamt | Absorptionsrate bei T | Ausfluss | ungestoßene Masse |
|---|---:|---:|---:|---:|---:|
| `lattice-paul` | 2.080890 ± 7e-6 | 1.119098 ± 7e-6 | 0.731555 ± 1.7e-5 | 1.17e-5 | 0.780712 ± 7e-6 |
| `lattice-3d` | 0.702146 ± 6e-6 | 2.497854 ± 6e-6 | 0.993430 ± 2.0e-5 | 8.1e-8 | 0.433583 ± 5e-6 |

Die Standardfehler sind 2–3 Größenordnungen kleiner als die Abweichungen der Solver-Läufe; die Referenz begrenzt die
Aussagen nicht mehr (zuvor: 4 Stellen der Literatur).

**2D (`lattice-paul`), Auswahl** (vollständig in `REPORT.md`), Abweichung Masse / Absorption / Rate gegen MC:

| Lauf | ohne FCS | mit FCS |
|---|---|---|
| Q1 448², P7 | −0.64 / +1.20 / +1.06 % | −0.39 / +0.71 / +0.82 % |
| Q1 448², P15 | −0.25 / +0.47 / +0.41 % | −0.17 / +0.29 / +0.31 % |
| Q1 448², P21 | −0.20 / +0.38 / +0.29 % (Literatur: −0.17 / +0.34 / +0.28) | −0.14 / +0.24 / +0.21 % |
| Q1 896², P7 (A100) | −0.60 / +1.11 / +1.03 % | |

Gegen die Literatur verschieben sich alle Masse-Abweichungen um −0.033 Prozentpunkte, die Absorption um +0.036; die
bisherigen Aussagen (monotone Konvergenz in Gitter und N, FCS Faktor ≈ 1.7) bleiben. Die feinsten Läufe liegen jetzt
deutlich über der Referenzunsicherheit; der Rest ist Orts- und PN-Fehler.

**3D (`lattice-3d`), erstmals mit Referenz:** Masse / Absorption / Rate gegen MC, Q1 ohne FCS:

| Gitter | P3 | P7 |
|---|---|---|
| 28³ | −42.2 / +11.9 / +0.60 % | −42.0 / +11.8 / +0.59 % |
| 56³ | −21.3 / +6.0 / +0.47 % | −20.1 / +5.6 / +0.43 % |
| 112³ | −11.6 / +3.3 / +0.35 % | −8.9 / +2.5 / +0.37 % |

Mit FCS (P3, docs/FCS_LATTICE.md): 28³ −15.9 / −2.5 / −5.5 %, 56³ −7.6 / +0.26 / −1.2 %. Der 3D-Lattice ist auf
diesen Gittern weit von der Konvergenz entfernt: Die Masse konvergiert ohne FCS etwa mit erster Ordnung (Fehler
halbiert sich je Verdopplung), die Quellzelle hat bei 28³ nur 4³ Elemente. FCS verkleinert den Massenfehler auf 56³
um den Faktor 2.8. Bisherige 3D-Aussagen waren nur Selbstvergleiche; diese Zahlen ersetzen sie.

## Konvergenzkampagne `tardis-lattice-mc` (16.11.2026)

tardis, eine A100, Stand 4fdcb35 (Block 3 + 4 gemergt), Binary SHA256 7b920fff…, Q1, CFL 1, T = 3.2, Job 161108
(20 Läufe, 0 Fehler). Profil `tardis-lattice-mc` in `scripts/element_weno/tardis_convergence.py`; Auswertung
`results/convergence/tardis_lattice_mc_20261116/` (`lattice_mc_compare.py --campaign`; FCS-Summen aus
`summary.json` → `lattice_fcs.total_*`).

**2D** (Masse / Absorption / Rate gegen MC; in Klammern gegen Literatur):

| Lauf | mit FCS | ohne FCS | Lösezeit |
|---|---|---|---:|
| 896², P15 | −0.111 / +0.200 / +0.257 % | | 161 s |
| 1344², P15 | −0.102 / +0.187 / +0.249 % | | 663 s |
| 896², P21 | −0.070 / +0.124 / +0.153 % | −0.102 / +0.190 / +0.207 % | 298 s |
| 1344², P21 | −0.059 / +0.108 / +0.147 % | | 1112 s |
| 896², P35 | −0.044 / +0.075 / +0.074 % (Lit. −0.011 / +0.040 / +0.068) | | 660 s |

Ab 896² ändert die Gitterverfeinerung wenig (P21: −0.070 → −0.059 %); der Rest ist überwiegend PN-Fehler und fällt
mit N (P15 → P21 → P35). FCS verkleinert auch hier die Abweichung (P21 896²: −0.070 gegen −0.102 %). Gegen die
Literaturtabelle sähe der beste Lauf um den Faktor 4 genauer aus (−0.011 %), als er ist.

**3D**, Masse (Absorption, Rate) gegen MC:

| Gitter | P3 | P7 | P15 | P7 mit FCS |
|---|---|---|---|---|
| 56³ | −21.3 % | −20.1 % | −20.0 % | −7.1 % (+0.12, −1.30) |
| 112³ | −11.6 % | −8.9 % | −8.2 % | −3.0 % (+0.39, −0.18) |
| 168³ | −9.0 % | −5.8 % | – | – |
| 224³ | −8.1 % | – | – | – |

- PN-Anteil: ab 112³ wesentlich (168³: P3 −9.0 gegen P7 −5.8 %); P3 sättigt bei etwa −8 %.
- Ortskonvergenz P7: 56³ → 112³ Faktor 2.3, 112³ → 168³ Faktor 1.5 (beobachtete Ordnung ≈ 1.1 einschließlich des
  unbekannten PN-Grenzfehlers; mit einem angenommenen P7-Grenzfehler um −3 % läge sie bei 1.5–1.8). Die
  Präasymptotik-Hypothese (docs/MATH_AUDIT.md, B.8) ist damit weder bestätigt noch widerlegt; nötig wäre P15 auf 168³
  (ohne FCS, ~40 GB).
- FCS verbessert die 3D-Masse um den Faktor 2.8–3.0 bei gleichem Gitter.
- Laufzeiten (A100): 168³ P7 336 s, 224³ P3 403 s, 112³ P15 432 s; mit FCS kommt die Tabellen- und Ausgabezeit
  dazu (112³: +300 s).
