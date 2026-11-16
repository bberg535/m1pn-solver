# Ergebnisse: Randoptimierung und momententreue Positivitätsprojektion (NX, 13.11.2026)

Methode und Implementierung: [docs/POSITIVITY_PROJECTION.md](../../../docs/POSITIVITY_PROJECTION.md). Alle Läufe auf nx
(2 × EPYC 9354), Element-WENO, SSPRK3, CFL 0.9, eingefrorene Binaries mit SHA256 im jeweiligen `manifest.json`
der Kampagne (`nx/tmp/positivity_*`, nicht versioniert). Tabellen: `runtime.md`, `accuracy.md`, `scaling.md`;
Abbildungen `fig_*.png/pdf`; Rohwerte `summary.json`.

| Ordner | Inhalt | Binary |
|---|---|---|
| `runtime_final_20261113/` | Laufzeit ref (`bdd5d3f`) / optimiert C6 / Projektion 2N+1 / N+1; Crossing T=0.8, E=32–256, N=3,7,15, Q1/Q2; je ein Sockel (32 Kerne) pro Lauf | `6dd402e` |
| `crossing_20261113/` | Genauigkeit Crossing (gleiche Matrix), Skalierung in N (E=64, N bis 31) und E (N=7, E bis 1024), Kernel-Benchmark | `ac06bb7` (Projektion identisch; bench `3644eb2`) |
| `outflow_20261113/` | Genauigkeit glatter Ausfluss (glatte Winkelverteilung, exakte kinetische Referenz), T=0.6 | `8c08b98` |

## 1. Laufzeit (runtime_final)

Gegenüber `bdd5d3f`: **Gesamtlaufzeit 1.6–3.0× schneller, Randpfad 2–8×** (wächst mit dem Gitter; E=256, Q2, N=3:
547 s → 184 s). Aufpreis der Projektion gegenüber optimiertem C6 auf E=256: +2.6 … +12 %, auf E=1024 (N=7) ±0.

## 2. Genauigkeit (kinetische Referenz; enthält den PN-Modellfehler)

| Problem | N=3 | N=7 | N=15 |
|---|---|---|---|
| Crossing (Strahlen), Projektion vs. C6 | ±0.2 % | −2.8 … −6.5 % | −3.5 … −10 % (größer auf feinen Gittern) |
| Glatter Ausfluss, global / Ausflussstreifen | +0.1…+1.2 % / bis +1.5 % | +0.5…+0.8 % / +1.5…+9 % | identisch (Identität) |

Regel N+1 und 2N+1 liefern im Solver dieselbe Genauigkeit (Unterschied ≤ 0.1 %). Beim glatten Ausfluss verschlechtert
die Projektion den Streifenfehler: die Projektion ist auf der ganzen Kugel definiert; negative Keulen auf der
einlaufenden Hälfte verschieben über ξ auch die auslaufenden Momente, die C6 unverändert übernimmt.
Der Ghost wirkt nur über den Limiter (für θ = 1 ist die Randlast die PN-Hochordnungslast des unprojizierten Traces);
die Projektion vergrößert die Korrektur h = P b^H − b^L und damit die Limitierung am Ausflussrand (E=32, N=7: 711 statt
452 limitierte Randknoten), siehe docs/POSITIVITY_PROJECTION.md, Abschnitt 2b. Seit 14.11.2026 ist die Projektion Standard.

## 3. Skalierbarkeit des Optimierungsproblems

| | Ergebnis |
|---|---|
| Newton-Schritte je Randknoten (Warmstart) | Mittel 0.8–1.5, Maximum 13 (N=3) … 32 (N=31) |
| Kosten je Randknoten | ∝ Punktzahl der Regel (∝ N²); konstant ≈ 2.5–3× C6 |
| Aufpreis vs. Gitter (N=7) | E=32 +11.6 %, 64 +7.1 %, 128 +4.0 %, 256 +0.7 %, 512 +2.1 %, 1024 −0.4 % |
| Aufpreis vs. N (E=64) | N=3 +0.4 %, 7 +10.6 %, 15 +20.4 %, 31 +30.9 % |
| Ergänzte Regel (unlösbare Zustände) | 0.3–4 % der Aufrufe, Anteil sinkt mit N |
| Rückfall auf C6 (nicht konvergiert) | 7 von 1.2·10⁸ Aufrufen in allen Skalierungsläufen, gezählt |

Kernel (ein Thread): Projektion warm 0.9 µs (N=3) … 38 µs (N=31, gepeakt) bzw. 202 µs (N=31, Strahl, Regel 2N+1);
mit Regel N+1 etwa 4× billiger. Die separable Trace-Auswertung ist bei N=31 5.2× schneller als die dichte.

## 4. Reproduzierbarkeit (≤ 1e−6)

Einzelne Auswertungen und kurze Läufe: ≤ 3e−11. Lange Läufe bei N=15: bis 3e−3 punktuell am Ausflussrand. Die
unveränderte Referenz weicht beim Wechsel 32 → 16 Threads selbst um 6.5e−5 (N=15) bzw. 5.7e−6 (E=256, Q2, N=7) ab;
Details in docs/POSITIVITY_PROJECTION.md, Abschnitt 3.
