# Dynamisches tile-adaptives $P_N$ auf einem Einzelgitter (`experiment/dynamic-adaptive-pn`)

Branch: `experiment/dynamic-adaptive-pn` (abgezweigt von `experiment/big-rebuilds`, Commit `2273ad0`).  
Standardverhalten (`--adaptive-pn off`) ist bitidentisch unverändert; das Verfahren ist rein opt-in über `--adaptive-pn on` (CPU und GPU in `m1pn_fast`).

---

## 1. Ausgangslage: Warum die statische MC-Pilot-Karte nicht gewonnen hat

In `docs/BIG_REBUILDS.md` wurde getestet, die lokale $P_N$-Ordnung statisch aus einem vorgeschalteten Monte-Carlo-Lauf abzuleiten und über `m1pn_blocks` als Multi-Block-Gitter mit blockweisen Ordnungen zu rechnen. Dabei entstanden drei strukturelle Verluste:

1. **Pilotkosten & Rauschen:** Ein rauscharmer MC-Pilotlauf kostet signifikante Rechenzeit; statistisches Rauschen bläst lokale Blockordnungen auf.
2. **Statische Zeit-Hüllkurve:** Jeder Block zahlt von $t = 0$ bis $t = T$ die Maximalordnung $\max_{t \in [0, T]} N(x, t)$, obwohl die Front weite Teile des Gebiets erst spät erreicht oder im diffusiven Nachlauf wieder glatt wird.
3. **Multi-Block-Overhead:** 21–71 separate `ElementWenoGpu`-Kontexte erzeugen zusätzliche Kernel-Launches, künstliche gebrochene Interfaces, Host-D2H/H2D-Transfers und globale Stage-Barrieren.

Die bisherige Annahme „Online-Sensoren sind zu teuer“ gilt nur für punktweise, quadratur- oder fluxbasierte Sensoren in jedem Zeitschritt. Ein **tile-weiser, spektraler Tail-Sensor direkt auf den aktiven $P_N$-Momenten**, der nur alle `adaptive_stride` Schritte läuft, kostet dagegen **< 1,7 % der Gesamtlaufzeit** (0,03–1,0 s) und benötigt keinen Pilotlauf.

---

## 2. Mathematisches & algorithmisches Design

### 2.1 Geschachtelter Galerkin-Unterraum auf einem einzigen Tensorprodukt-Gitter

Das bestehende Einzelgitter bleibt unangetastet (keine künstlichen Block-Interfaces, keine zusätzlichen gebrochenen Linien, keine Zeitschritt-Einbuße). Die Knoten werden logisch in Kacheln (Tiles) der Größe `tb[0] × tb[1] × tb[2]` (Standard: `16` bzw. `8` Knoten pro Achse) eingeteilt.

Jede Kachel $T$ trägt eine ungerade aktive Ordnung $N_T \in [N_{\min}, N_{\max}]$ mit aktiver Momentanzahl $q_T = q(N_T)$ (inklusive $l \le 1$ überall aktiv). Es gilt strikt die Null-Invariante:
$$a_{i, c} \equiv 0 \quad \forall\, i \in T,\; c \ge q_T.$$

Da die niedrigen Momente $c \in \{0, 1, 2, 3\}$ ($\rho, \mathbf{J}$) immer überall aktiv sind, bleiben **M1-Kegel-Realisierbarkeit, Massenerhaltung, WENO-Gewichte, Element-Limiter, Split-Breaks und Spiegelrandbedingungen** exakt erhalten.

### 2.2 Exakte 1D-Band-Lösung auf aktiven Segmenten ohne Neufaktorisierung

Die bestehenden 1D-Massenmatrizen $M_d$ werden einmalig in natürlicher Reihenfolge bandfaktorisiert ($M_d = L_d D_d L_d^T$, Bandbreite $p$). Für jede aktive Moment-Gruppe und jede 1D-Gitterlinie werden zusammenhängende aktive Knotenintervalle $[k_0, k_1)$ in kompakten Segmentlisten gesammelt (`line_segs_grad[d]` mit 1 Knoten Rand-Padding für $C_d a$, `line_segs_mass[d]` ohne Padding für $M_d^{-1} r$). Die Einschränkung von $L_d, D_d$ auf $[k_0, k_1)$ invertiert $\tilde L \tilde D \tilde L^T$ (strikte SPD-Matrix, exakt an allen inneren Knoten) ohne jede Neufaktorisierung bei Ordnungswechseln.

### 2.3 Online-Spektralsensor mit Hysterese & randgetriggerter Wellenfront-Dilatation

Alle `adaptive_stride` Zeitschritte (Standard: 4, plus nach Schritt 0 und 1) misst ein leichtgewichtiger Kernel (`k_adaptive_sensor` bzw. CPU-Tile-Schleife) pro Kachel $T$ **nur auf den aktuell aktiven Momenten**:
- Maximaldichte $\rho_T = \max_{i \in T} |\rho_i|$,
- maximale Schalenamplitude $E_{T, l} = \max_{i \in T} \sqrt{\sum_{m} a_{i, (l,m)}^2}$ für $0 \le l \le N_T$,
- maximale Randstreifen-Energie $E^{\text{face}}_{T, f}$ in den äußeren 3 Knotenlagen jeder Kachelseite $f \in \{0, \dots, 2d-1\}$.

Entscheidungsregel gegen $\tau_{\text{up}} = \text{tol} \cdot \max_T \rho_T$ und $\tau_{\text{down}} = \text{hysteresis} \cdot \tau_{\text{up}}$:
- **Verfeinerung:** Übersteigt die oberste aktive Doppelschale $(N_T - 1, N_T)$ (bzw. $E_{T, 1} > 0{,}25\,\tau_{\text{up}}$ bei $N_T = 1$) den Schwellwert $\tau_{\text{up}}$, steigt $N_T$ gestuft um $+2, +4$ oder $+6$ (bis $N_{\max}$).
- **Vergröberung:** Fällt der kumulierte Tail unter $\tau_{\text{down}}$, sinkt $N_T$ glatt um höchstens $-2$ pro Anpassung; deaktivierte Koeffizienten werden in allen Stage- und Gradientenpuffern auf `0.0` gesetzt.
- **Wellenfront-Dilatation (`halo=1`):** Übersteigt $E^{\text{face}}_{T, f}$ die Schwelle auf einer Kachelseite $f$, wird die Nachbarkachel präventiv auf $\ge N_T$ angehoben (inkl. Eck-Nachbarn bei zwei aktiven Seiten), bevor die Front die Kachelgrenze überschreitet.
- **Inflow-Floor:** Kacheln an äußeren Nicht-Vakuum-Einströmungsrändern (z. B. `crossing-study`) erhalten automatisch `tile_floor[T] = max_order`, damit eintretende Strahlen ab Schritt 0 voll aufgelöst werden. Auf Vakuum-Rändern (`lattice-paul`, `line-source`) greift der Floor nicht.

---

## 3. Effizienz-Iterationen

| Iteration | Zentrale Änderungen | Wirkung (`gbq` 224² $P_{35}$ FCS GPU, Solver-Zeit) |
|---|---|---|
| **Baseline** | Uniformes $P_{35}$ auf Einzelgitter | **179,96 s** (`ghost 136,74 s`, `wait 38,19 s`) |
| **Iter. 1** | Tile-Plan (`src/adaptive_pn.hpp`), CPU/GPU-Sensor, segmentierte 1D-Liniensolves, adaptives Stab/Combine/React/Boundary; Fix gegen uninitialisierte Puffer aus `spectral_radius()` | funktionaler Einzelgitter-Prototyp, CPU/GPU konsistent |
| **Iter. 2** | `SeparableTrace::evaluate_upto`, exakter analytischer $P_1$-Fast-Path ($3|\mathbf{J}| \le \rho$), adaptiver D2H-Fetch (`gather_cols_async`), CPU-Boundary-Truncating | **25,27 s** (`7,1×`, `ghost 3,40 s`, `wait 19,99 s`) |
| **Iter. 3** | Split `low4` ($c < 4$) + Warp-aligned High-Chunks ($c_0 = 4, 36, 68, \dots$: exakte $P_7$-Ausrichtung, $P_1$-Tiles erzeugen 0 High-Work); Längensortierung der 1D-Segmente (kein Warp-Tail); 2-Warp-Liniensolve; Warp-Broadcast von `exp(-tau*(sa+ss))`; Tile-Loop-Stabilisierung auf CPU | **18,50 s** (`t16`, `9,7×`) / **16,94 s** (`t8`, `10,6×`) |
| **Iter. 4** | **Adaptive Ghost-Quadratur-Pyramide** ($N_{\text{tier}} \in \{3, 7, 15\}$ mit `rule_n = N+1` → 144/400/1296 statt 5776 Punkte für Positivitätsprojektion & Traces auf CPU+GPU); **Sparse-ELL-Broken-Line-Kernel** (`k_boundary_interface_sparse`, ersetzt dichte `cublasDgemm` auf gebrochenen Linien durch 4 FMAs/Eintrag) + per-Side `side_q_dom/brk`; per-Task-Breite `q_task` in CPU `add_boundary` + Inflow-Floor | **6,40 s** (`t16`, **28,1×**) / **5,00 s** (`t8`, **36,0×**) |

---

## 4. Messergebnisse

Messsystem: Lokal, Intel Core i5-12500 (6 Kerne / 6 OpenMP-Threads) + NVIDIA RTX A2000 12 GB (`sm_86`). MC-Referenz für `lattice-paul`: `tmp/lattice_mc_field/lattice-paul_field224.csv`.

### 4.1 `lattice-paul` 224² (`gbq`: `8,16,32,32,32,16,8`, broken, split-breaks, mirror-x, FCS, $t = 3{,}2$, GPU)

| Lauf | `solve` | Speedup | `ghost` | `wait` | `sens` | `frac` | $\bar N$ | MC `mass / abs / rate %` | L1 vs MC | med |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `gbq_p21_uni` | 20,34 s | 1,00× | 13,16 s | 4,93 s | — | 1,000 | 21,0 | `-0,148 / +0,181 / +0,102` | `4,1e-3` | `6,4e-3` |
| `gbq_p21_t16` (`tol 3e-4`) | 4,72 s | **4,31×** | 0,57 s | 3,19 s | 0,06 s | 0,447 | 10,2 | `-0,149 / +0,183 / +0,100` | `4,2e-3` | `8,5e-3` |
| `gbq_p21_t8` (`tol 3e-4`) | 3,96 s | **5,14×** | 0,32 s | 2,81 s | 0,05 s | 0,343 | 8,2 | `-0,150 / +0,185 / +0,103` | `4,2e-3` | `1,0e-2` |
| `gbq_p35_uni` | 179,96 s | 1,00× | 136,74 s | 38,19 s | — | 1,000 | 35,0 | `-0,145 / +0,176 / +0,084` | `4,1e-3` | `6,5e-3` |
| `gbq_p35_t16` (`tol 3e-4`) | 6,40 s | **28,1×** | 0,70 s | 4,47 s | 0,10 s | 0,243 | 11,8 | `-0,146 / +0,178 / +0,084` | `4,2e-3` | `8,4e-3` |
| `gbq_p35_t8` (`tol 3e-4`) | 5,00 s | **36,0×** | 0,26 s | 3,75 s | 0,08 s | 0,175 | 9,2 | `-0,148 / +0,181 / +0,089` | `4,2e-3` | `1,0e-2` |
| `gbq_p35_1e4_t8` (`tol 1e-4`) | 7,07 s | **25,5×** | 0,76 s | 4,96 s | 0,12 s | 0,278 | 12,1 | `-0,146 / +0,177 / +0,086` | `4,1e-3` | `7,2e-3` |

### 4.2 `lattice-paul` 448² (`gbh`: `16,32,64,64,64,32,16`, broken, split-breaks, mirror-x, FCS, $t = 3{,}2$, GPU)

| Lauf | `solve` | Speedup | `ghost` | `wait` | `sens` | `frac` | $\bar N$ | MC `mass / abs / rate %` | L1 vs MC | med |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `gbh_p21` (uniform) | 113,18 s | 1,00× | 49,58 s | 48,25 s | — | 1,000 | 21,0 | `-0,069 / +0,105 / +0,109` | `1,7e-3` | `3,0e-3` |
| `gbh_p21_t16` (`tol 3e-4`) | 29,17 s | **3,88×** | 0,78 s | 23,07 s | 0,43 s | 0,384 | 8,9 | `-0,070 / +0,107 / +0,112` | `1,8e-3` | `5,9e-3` |
| `gbh_p21_t8` (`tol 3e-4`) | 29,71 s | **3,81×** | 0,59 s | 23,82 s | 0,44 s | 0,316 | 7,6 | `-0,072 / +0,110 / +0,119` | `1,8e-3` | `7,4e-3` |
| `gbh_p35` (uniform) | 770,29 s | 1,00× | 578,41 s | 157,11 s | — | 1,000 | 35,0 | `-0,053 / +0,075 / +0,051` | `1,5e-3` | `2,8e-3` |
| `gbh_p35_t16` (`tol 3e-4`) | 53,13 s | **14,5×** | 0,62 s | 43,18 s | 0,88 s | 0,263 | 11,4 | `-0,055 / +0,079 / +0,057` | `1,6e-3` | `5,9e-3` |
| `gbh_p35_t8` (`tol 3e-4`) | 44,04 s | **17,5×** | 0,61 s | 35,69 s | 0,75 s | 0,205 | 9,5 | `-0,057 / +0,083 / +0,065` | `1,7e-3` | `7,3e-3` |
| `gbh_p35_1e4_t8` (`tol 1e-4`) | 58,31 s | **13,2×** | 0,75 s | 47,35 s | 1,01 s | 0,297 | 12,1 | **`-0,054 / +0,077 / +0,054`** | **`1,5e-3`** | **`3,6e-3`** |

Bemerkenswert: `gbh_p35_1e4_t8` erreicht **volle $P_{35}$-Genauigkeit** (`L1 = 1,5e-3`, MC-Abweichung auf 0,001–0,003 %-Punkte identisch zu uniformem $P_{35}$) und ist dabei **13,2× schneller als uniformes $P_{35}$** und fast **2× schneller als uniformes $P_{21}$** (`113,18 s`).

### 4.3 Reines GPU-Kernel-Timing & CPU-Laufzeiten (`gbq`, `fcs off`)

| Benchmark | Uniform | Adaptiv (Iter. 4) | Speedup | `frac` | `sens` |
|---|---:|---:|---:|---:|---:|
| `diag_p35` (`M1PN_GPU_DIAG_SERIAL=1`, GPU, $t=3{,}2$) | 181,66 s | **7,53 s** | **24,1×** | 0,279 | 0,10 s |
| `cpu_p21` (6 Threads, CPU, $t=1{,}6$) | 11,51 s | **5,92 s** | **1,94×** | 0,241 | 0,01 s |
| `cpu_p35` (6 Threads, CPU, $t=1{,}6$) | 35,49 s | **10,40 s** | **3,41×** | 0,139 | 0,03 s |

### 4.4 Problem-Unabhängigkeit (Multi-Case-Validierung)

| Testfall | Uniform | Adaptiv | Speedup | `frac` | $\bar N$ | rel. $L_1$ vs Uniform | rel. $L_\infty$ vs Uniform |
|---|---:|---:|---:|---:|---:|---:|---:|
| `crossing-study` (Beam-Inflow, 64², $P_{21}$, GPU, $t=0{,}8$) | 26,55 s | 11,03 s | **2,41×** | 0,970 | 20,5 | `1,54e-5` | `1,73e-5` |
| `line-source` (Ringschock, 128², $P_{21}$, GPU, $t=0{,}35$) | 5,14 s | 0,96 s | **5,35×** | 0,270 | 8,1 | `8,88e-4` | `1,39e-2` |
| `lattice-3d` (3D, 14³, $P_7$, CPU, $t=1{,}0$) | 0,25 s | 0,26 s | 0,98× | 0,396 | 3,4 | `2,68e-5` | `7,40e-6` |

---

## 5. CLI-Optionen & Empfehlungen für Produktivläufe (`tardis` A100 / NX)

Neue Optionen in `m1pn_fast` (Standard: `--adaptive-pn off`):
- `--adaptive-pn on|off` — aktiviert dynamisches tile-adaptives $P_N$
- `--adaptive-tol <val>` — relative Tail-Schwelle (Standard `3e-4`; `1e-4` für nahezu referenzidentische $P_{35}$-Genauigkeit)
- `--adaptive-hysteresis <val>` — Vergröberungs-Faktor (Standard `0.25`)
- `--adaptive-tile <int>` — Kachelgröße pro Achse in Knoten (Standard `16`; `8` auf 224²/448² spart weitere 15–20 % Koeffizienten)
- `--adaptive-stride <int>` — Anpassungsintervall in Zeitschritten (Standard `4`)
- `--adaptive-min-order <int>` — minimale ungerade Ordnung (Standard `1`)
- `--adaptive-halo <0|1|2>` — `0`: aus, `1`: randgetriggerte Dilatation (Standard), `2`: volle 3×3-Nachbarschaft

Beispielaufruf für das volle `GB`-Gitter (896², $P_{35}$ / $P_{45}$) auf `tardis` A100:
```bash
./tmp/build_gpu/m1pn_fast \
  --case lattice-paul --inflow vacuum --dimension 2 --elements 7 --degree 1 \
  --background 1e-10 --fcs analytic \
  --cells-per-unit 32,64,128,128,128,64,32 \
  --break-units on --split-breaks on --mirror-reduction auto \
  --pn-order 35 --time 3.2 --gpu \
  --adaptive-pn on --adaptive-tol 1e-4 --adaptive-tile 16 \
  --output tmp/gb_p35_adaptive
```

## 6. Tests

- Neue Testsuite: `tests/adaptive_pn_dynamic.cpp` (CTest `adaptive_pn_dynamic`, 12 Checks: `SeparableTrace::evaluate_upto` bitexakt `0`, Quadrupol mit $\mathbf{J}=0$, Kegel-Realisierbarkeit, `graded + broken + mirror` CPU, GPU `min=max` bitidentisch `1.1e-16` vs Uniform-GPU, CPU-vs-GPU-Konsistenz).
- Vollständige `ctest`-Suite: **12/12 Tests bestanden** (`element_weno`, `element_weno_material`, `element_weno_fixes`, `element_weno_fcs`, `uncollided_lattice`, `uncollided_lattice3d`, `kobayashi_uncollided`, `positivity_projection`, `adaptive_pn_dynamic`, `fast_cli`, `lattice_mc`, `campaign_scripts`).
