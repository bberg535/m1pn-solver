# Dynamische adaptive Gitterverfeinerung auf abgestuften Tensorgittern (`experiment/dynamic-amr`)

Branch `experiment/dynamic-amr`, aufbauend auf `experiment/dynamic-adaptive-pn` (a7b2080, dynamisches kachel-adaptives
$P_N$). Opt-in über `m1pn_fast --amr on`; ohne die Option ist der Solver bitgleich zum Ausgangsstand (geprüft: CPU,
GPU mit adaptivem $P_N$, statisches GBQ-Lattice mit FCS).

## 1. Entwurf

### 1.1 Warum achsenweise Tensor-AMR

Der Solver ist ein stetiges Tensor-Bernstein-Verfahren: Massenmatrix $M = M_x \otimes M_y$, Linienlöser je Achse,
Größenklassen abgestufter Gitter, GPU-Streifenkerne. Hängende Knoten (Quadtree) würden Tensorstruktur, Linienlöser und
GPU-Pfad aufbrechen. Stattdessen bleibt das Gitter ein **abgestuftes Tensorgitter**, dessen Ebenen sich mit der Zeit
ändern:

- Basisgitter = das angegebene Gitter (`--elements` bzw. `--cells-per-unit`), das gröbste.
- Jede Basiszelle $b$ jeder Achse $d$ trägt eine Stufe $\ell_{d,b} \in [\ell_{\min}, L]$ und wird dyadisch in
  $2^{\ell}$ gleiche Elemente geteilt. Bruchlinien (`--break-units`) und Materialgrenzen liegen auf Basisebenen und
  bleiben Elementgrenzen.
- Benachbarte Basiszellen unterscheiden sich um höchstens eine Stufe (2:1), um jede verfeinerte Zelle liegt ein Puffer
  von `--amr-buffer` Basiszellen.

Ein Neuaufbau ist damit ein gewöhnliches `ElementWeno` (+ `ElementWenoGpu`) auf neuen `planes`; alle Verfahrensteile
(WENO, Elementlimiter, Split-Breaks, Spiegel, FCS, adaptives $P_N$) bleiben unverändert.

Grenze: Verfeinerung wirkt in ganzen Streifen (Tensorprodukt). Eine Front parallel zu einer Achse verfeinert nur diese
Achse; eine runde Front verfeinert ihr umschreibendes Rechteck.

### 1.2 Zustandstransfer: Realisierbarkeit und Masse

Transfer $A \to B$ je Achse über die gemeinsame Verfeinerung $C$ (Vereinigung der Ebenen):

1. **Exakte Prolongation $A \to C$** durch Bernstein-Unterteilung (Blossoming/de Casteljau): die Funktion bleibt
   identisch, jeder neue Koeffizient ist eine Konvexkombination alter.
2. **Restriktion mit gelumpter Masse $C \to B$:**
   $u_I = \sum_j P_{BC}(j,I)\, m_j u_j / m_I$, mit $P_{BC}$ der (nichtnegativen, zeilenstochastischen) Prolongation
   $B \to C$ und $m = \int \varphi$. Wegen $\varphi^B_I = \sum_j P_{BC}(j,I) \varphi^C_j$ gilt
   $\sum_j P_{BC}(j,I) m_j = m_I$: die Zeilen sind konvexe Gewichte, und $\sum_I m_I u_I = \sum_j m_j u_j$.

Wo $B$ verfeinert, ist die Restriktion die Identität (exakte Prolongation); wo $B$ vergröbert, ist sie der positive,
konservative Mittelwert. In 2D ist der Transfer das Tensorprodukt der 1D-Transfers (achsenweise angewandt); die
gelumpte Tensor-Bernstein-Masse ist das Produkt der 1D-Massen. Folgen:

- **Masse:** $\sum_i m_i \rho_i$ bleibt bis auf Rundung erhalten (gemessen ≤ 1e-14 relativ je Neuvernetzung); der
  Rundungsrest wird als eigener Posten `amr.mass_defect` in der Massenbilanz geführt.
- **Realisierbarkeit:** jeder neue Knoten ist eine Konvexkombination alter Knoten; der M1-Kegel ist konvex, also bleibt
  jeder Knoten im Kegel. Danach läuft der bestehende Rundungs-Nudge (`admissible_rows`, im Nudge-Posten gezählt;
  in allen Läufen 0 Nudges).
- **Spiegelebenen:** am Spiegelknoten werden die unter der Spiegelung ungeraden Momente genullt (genau die Restriktion
  des gespiegelten Gesamtfelds; $\rho$ ist gerade, die Masse unberührt).
- **Adaptives $P_N$:** alle Momente werden mit derselben linearen Abbildung übertragen; das neue Schema startet mit der
  initialen Kachelanpassung auf dem übertragenen Zustand (Null-Invariante bleibt erhalten).

### 1.3 Indikator und Steuerung

Je Achse $d$ und Basiszelle $b$:
$\eta_{d,b} = \max_{i} |\delta^2_d u_i|\, w_e^2 / \max|u_0|$ über die Knoten der Basiszelle, mit $\delta^2_d$ der
zweiten dividierten Differenz der vier niedrigen Momente längs $d$ (Greville-Abszissen der Bernstein-Kontrollpunkte,
nie über Bruchlinien) und $w_e$ der Elementbreite. $\eta$ schätzt den Interpolationsfehler $\sim w^2 u''$ auf der
aktuellen Breite; Vergröbern vervierfacht ihn. Mit FCS wird der Indikator auf der Gesamtdichte (gestoßen + ungestoßen
an den Knoten) zur Zeit $t$ und $t + \Delta t_r$ (Ende des kommenden Fensters) ausgewertet, damit das Gitter der
ungestoßenen Front vorauseilt.

- Verfeinern: $\eta > $ `--amr-tol` (mehrere Stufen auf einmal, bis $\eta/4^k \le$ tol).
- Vergröbern: $4\eta < $ `--amr-hysteresis` · tol, eine Stufe je Prüfung.
- Prüfung alle $\Delta t_r$ = `--amr-interval` (Standard: Puffer × kleinste Basisbreite, Geschwindigkeit 1: die Front
  verlässt den Puffer nicht). Startgitter: Iteration auf den analytischen Anfangsdaten.
- Zeitschritt je Gitter: `cfl · step_bound()` des neuen Schemas.

## 2. Implementierung

| Datei | Inhalt |
|---|---|
| `src/amr.hpp` | 1D-Gitter, Bernstein-Unterteilung, Prolongation, gemeinsame Verfeinerung, Transfer, Tensoranwendung, Indikator, Stufen-Update |
| `src/fast_main.cpp` | Optionen `--amr*`, Startgitter, Fenster-Schleife (`amr_remesh`), Ausgabe auf dem Endgitter, Bilanzposten, `summary.json` → `amr` |
| `src/uncollided_lattice.hpp`, `src/element_weno_gpu.*` | Lattice-FCS-Zeittabelle mit Zeitfenster (`m0`), Neuaufbau auf der GPU ohne neuen Kontext |
| `src/element_weno.*` | Winkeldaten-Cache (`ElementWenoAngularCache`), `power_radius` mit Warmstart, `reflect_odd` |
| `tests/amr_transfer.cpp` | ctest `amr_transfer` |

## 3. Effizienz-Iterationen

Messsystem lokal: i5-12500 (6 Threads), RTX A2000 12 GB. Fall `lattice-paul` mit Lattice-FCS, $t = 3{,}2$, $P_{21}$,
GPU, adaptives $P_N$ (Kachel 8), Bruchlinien an den Einheitslinien, Split-Breaks, Spiegelreduktion. AMR: Basis
4/Einheit, $L = 3$ (fein 32/Einheit). Rohdaten `tmp/amr_bench/` (nicht versioniert), Auswertung `tmp/analyze_amr.py`
gegen die MC-Referenz 224².

| Stand | gesamt | davon AMR | Hauptkosten | Lösung |
|---|---:|---:|---|---|
| naiv (Neuaufbau, volle FCS-Tabelle $[0, T]$ je Gitter) | 304 s | 282 s | Tabelle ≈ 28 s je Neuvernetzung | Referenz |
| **Iter. 1:** Tabelle nur für das Zeitfenster $[t, t+\Delta t_r]$ (`m0`, CPU und GPU), Neuaufbau der Tabelle auf der GPU ohne neuen Kontext; ungestoßene Absorption fensterweise summiert | 49,6 s | 41,7 s | GPU-Setup inkl. Fenstertabellen 23 s, Spektralradius 9,7 s | bitgleich; ungestoßene Absorption 1e-12 relativ |
| **Iter. 2:** Winkeldaten-Cache über Neuaufbauten (Schema 3,2 → 0,26 s), Potenziteration mit Warmstart aus dem übertragenen Eigenvektor (20 statt 2 × 60 Schritte; 9,7 → 3,7 s), Radius bei gleicher feinster Breite nicht fallend | 41,4 s | 33,2 s | Fenstertabellen ≈ eine Volltabelle | bitgleich (Radius warm/kalt ≤ 0,33 % unter) |
| **Iter. 3:** Parameter (tol 3e-3, Puffer 1, Intervall automatisch) | 38,3 s | 30,6 s | Fenstertabellen 22 s + 2,9 s | gleich genau, 11 % weniger Knoten |

Verworfen in Iteration 3 (gemessen): Puffer 2 oder Intervall 0,5 (mehr Knoten bzw. schlechteres Feld), tol 3e-4 (mehr
Knoten, keine Genauigkeit), randgetriggerte Dilatation des adaptiven $P_N$ nach einer Neuvernetzung
(`--amr-pn-restart on`: Koeffizientenanteil 0,368 → 0,322, 3–6 % schneller, Feld-Median aber 6,1e-3 → 9,4e-3).

Die Summe der Fenstertabellen entspricht etwa einer Volltabelle auf dem mittleren Gitter, also dem, was auch das
statische Gitter zahlt: das ist die Untergrenze der FCS-Kosten (keine mathematische Änderung an der Tabelle).

## 4. Ergebnisse

### 4.1 Lattice mit FCS (GPU, $t = 3{,}2$, Abweichung gegen MC in %)

| Lauf | gesamt | Solve | Tabelle | mittl. Knoten | Masse / Abs. / Rate | Feld L1 | Median |
|---|---:|---:|---:|---:|---|---:|---:|
| GBQ statisch (8,16,32,32,32,16,8), $P_{21}$ | 25,3 s | 4,0 s | 20 s | 11 476 | −0,150 / +0,185 / +0,103 | 4,2e-3 | 1,0e-2 |
| U224 statisch, $P_{21}$ | 40,4 s | 8,5 s | 30 s | 26 796 | −0,147 / +0,181 / +0,123 | 4,0e-3 | 9,7e-3 |
| **AMR** Basis 4, $L=3$, $P_{21}$ | 38,3 s | 35,9 s | 25 s | 12 420 | −0,147 / +0,179 / +0,093 | 3,8e-3 | 6,1e-3 |
| GBH statisch (16,32,64,64,64,32,16), $P_{21}$ | 131 s | 29,6 s | 94 s | 43 660 | −0,072 / +0,110 / +0,119 | 1,8e-3 | 7,4e-3 |
| **AMR** Basis 4, $L=4$, $P_{21}$ | 166 s | 160 s | 110 s | 36 371 | −0,070 / +0,107 / +0,114 | 1,7e-3 | 5,6e-3 |
| GBQ statisch, $P_{35}$ | 35,9 s | 6,3 s | 29 s | 11 476 | −0,148 / +0,181 / +0,089 | 4,2e-3 | 1,0e-2 |
| **AMR** Basis 4, $L=3$, $P_{35}$ | 56,3 s | 51,0 s | – | 12 428 | −0,146 / +0,177 / +0,086 | 3,9e-3 | 9,4e-3 |

Bewertung: Auf gleicher feinster Stufe erreicht AMR die Größen des statischen Gitters (L4 = GBH auf 0,002–0,005
%-Punkte) und ein deutlich besseres Feld (Median 25–40 % kleiner), weil das Gitter der Front folgt statt fest am
Zentrum zu verfeinern. In der Wandzeit ist es beim Lattice **nicht schneller** als ein gut gewähltes statisches
Gitter: die Lösung füllt bis $t = 3{,}2$ das ganze Gebiet, die späten (teuren) Tabellenfenster laufen auf Gittern so groß
wie GBH, und die FCS-Tabelle dominiert. Der Vorteil liegt dort, wo die aktive Region klein bleibt oder kein statisches
Gitter vorab bekannt ist.

### 4.2 Line Source mit FCS ($P_{15}$, $\sigma_s = 1$, Pulsbreite 0,03, $t = 0{,}3$, GPU, adaptives $P_N$)

| Lauf | Knoten (mittel) | Solve | gesamt | rel. L1 gegen U256 | rel. $L_\infty$ |
|---|---:|---:|---:|---:|---:|
| U128 | 4 225 | 0,68 s | 7,5 s | 1,2e-3 | 4,1e-3 |
| U256 | 16 641 | 3,36 s | 26,9 s | – | – |
| **AMR** Basis 32, $L=3$ (fein ≙ 256) | 8 955 | 2,90 s | 19,7 s | 8,5e-5 | 6,4e-4 |

AMR folgt dem Ring (Gitter 62² → 118² auf dem Viertelgebiet in 8 Neuvernetzungen) und trifft U256 auf 1e-4 mit 54 % der
Knoten.

### 4.3 Erhaltung und Realisierbarkeit (alle Läufe)

- Transferdefekt je Neuvernetzung ≤ 1e-14 relativ (`amr.max_relative_defect`), als Bilanzposten geführt; die
  Massenbilanz schließt wie im statischen Lauf (Line Source 7e-15; Lattice ~1e-7, identisch zum statischen Gitter:
  Split-Break-Posten außerhalb des Buchs).
- 0 Kegel-Nudges in allen AMR-Läufen, minimale Kegelmarge und Dichte positiv.
- CPU gegen GPU (Lattice $P_7$, $t = 1$, $L = 2$): identische Gitterfolge, Dichte 1,1e-9 relativ.
- `--amr off`: bitgleich zum Ausgangsstand (CPU, GPU mit adaptivem $P_N$, GBQ-Lattice mit FCS).
- ctest 13/13 (neu: `amr_transfer`, 17 Prüfungen: Unterteilung, 1D-/2D-Transfer gegen die `lumped`-Massen des
  Solvers, Kegel, exakte Verfeinerung, Indikator/2:1, Fenstertabelle bitgleich).

## 5. Optionen

```
--amr on|off                 (Standard off)
--amr-levels L               feinste Stufe über dem Basisgitter (Standard 2)
--amr-min-level l            (Standard 0)
--amr-tol T                  Verfeinerungsschwelle des Indikators (Standard 3e-3)
--amr-hysteresis H           Vergröbern bei 4 eta < H tol (Standard 0.25)
--amr-buffer B               Puffer in Basiszellen (Standard 1)
--amr-interval DT            Prüfintervall (Standard: B x kleinste Basisbreite)
--amr-radius-iterations K    Warmstart-Iterationen des Schrittschätzers (0: kalt, 60)
--amr-pn-restart on|off      randgetriggerte P_N-Dilatation nach Neuvernetzung (Standard off)
M1PN_AMR_RADIUS_CHECK=1      zusätzlich kalter Radius je Neuvernetzung (Validierung)
```

Beispiel: `m1pn_fast --case lattice-paul ... --cells-per-unit 4,4,4,4,4,4,4 --break-units on --split-breaks on
--fcs analytic --gpu --adaptive-pn on --adaptive-tile 8 --amr on --amr-levels 3`.

Einschränkungen: 1D/2D (GPU 2D), seriell (kein MPI), offenes Gitter; Lattice-3D-FCS ohne Zeitfenster.

## 6. Offene Punkte

- Ganzer Gewinn bräuchte lokale (nicht streifenweise) Verfeinerung; mit Tensorstruktur ginge das nur über Blöcke
  mit eigenen Gittern (Mehrblock-Overhead, `docs/BIG_REBUILDS.md`).
- Vergröbern ist erster Ordnung (positiver Mittelwert); eine FCT-Kombination mit der $L^2$-Projektion wäre genauer,
  braucht aber Begrenzung gegen den Kegel.
- Indikator-Auswertung der ungestoßenen Dichte per Raytracing auf der CPU (L4: 5 s); die Gerätetabelle könnte das
  übernehmen.
