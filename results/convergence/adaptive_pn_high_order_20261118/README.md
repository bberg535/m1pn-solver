# Hochordnungs- und CFL-Studie zum adaptiven PN-Prototyp

Die Messungen beziehen sich auf den periodischen Q1-Prototyp bei Commit [c5adf8c](https://github.com/bberg535/m1pn-solver/commit/c5adf8c701d8ebce618f96e6225110917c1b7dff). Die [Studientreiber](../../../scripts/element_weno/adaptive_pn_high_order) verwenden den Solver aus diesem Commit.

## Modell kompakt

1. **Volumen.** Jedes Element $K$ trägt Kugelflächenmoden bis Grad $N_K\ge1$. Der Galerkin-Operator im Volumen folgt aus dem gewöhnlichen $P_{N_K}$-System. Alle auf beiden Seiten verfügbaren Moden sind auf der gemeinsamen Fläche stetig.
2. **Fläche.** Beide Spuren $x,y$ werden bis $N_F=\max(N_L,N_R)$ mit Nullen ergänzt. Der gemeinsame Fluss ist $\widehat f_F=\tfrac12A_n^{(N_F)}(x+y)-\tfrac12(y-x)$. Beide Seiten übernehmen ihre jeweiligen Komponenten mit entgegengesetzten Vorzeichen. Beispielsweise liefert bei $P_1/P_3$ der Grad 2 der $P_3$-Spur einen Beitrag zum Fluss von $\mathbf J$.
3. **Ordnungswechsel.** Der elementweise RMS-Wert von $\|\mathbf J_h\|$ markiert über zwei Schwellen zur Erhöhung oder Senkung von $N_K$. Der Wechsel findet zwischen vollständigen Zeitschritten statt. Eine orthogonale $L^2$-Projektion auf den neuen Raum erhält $\rho_h$ und $\mathbf J_h$ punktweise.

Im ungelimitierten, semidiskreten System trägt jede innere Fläche $-\tfrac12\int_F\|y-x\|^2\,ds$ zur Energieänderung bei. Auch die Projektion kann die Energie nur senken. Die CFL-Untersuchung unten betrifft die zusätzliche Zeitdiskretisierung mit SSPRK(4,3). [Herleitung und M1-Korrektur](../../../docs/ADAPTIVE_PN_PROTOTYPE.md)

## Bedeutung der Messgrößen

- Die geometrische CFL-Zahl ist $C=\Delta t/h$. Der Kommandozeilenparameter `m1pn_fast --cfl` skaliert stattdessen `step_bound()`. Die CSV-Dateien verwenden durchgehend die geometrische Definition.
- Die drei kinetischen Anfangsdaten heißen `broad`, `beam` und `counter`. Die letzten beiden verwenden gerichtete Exponentialverteilungen mit Konzentration 24. Bei `counter` werden zwei exakt gegenläufige Verteilungen mit gleichem Gewicht addiert. Dadurch gilt anfangs $\mathbf J=0$ bei anisotroper Winkelverteilung.
- `fixed` verwendet das angegebene $N$ überall. `warm` startet mit der vollen Ordnung und `cold` startet mit $P_1$. Der Prototyp ändert die Ordnung bei jedem Adaptionszeitpunkt um höchstens zwei. Die acht möglichen Adaptionszeitpunkte sind $jT/8$ für $j=0,\ldots,7$. Die Schwellen `lower` und `upper` beziehen sich auf den elementweisen RMS-Wert von $\|\mathbf J\|$.
- Die Fehler in $\rho$, $\mathbf J$ und $P$ sind räumliche L²-Normen zum Q1-Interpolanten der Charakteristikenreferenz. Für den Drucktensor $P$ wird die Frobeniusnorm verwendet. Der Quadraturvergleich mit den Ordnungen 24, 40 und 48 ist in `additional.csv` enthalten.
- `timings.csv` enthält die einzelnen gepaarten Zeitläufe auf einem CPU-Kern. `timed_seconds` umfasst die Zeitschritte und beim adaptiven Pfad die Sensorauswertung, den Raumwechsel und die Projektion. Initialisierung, Referenzrechnung und Energiediagnose sind ausgenommen. `cpu_seconds` misst dieselben Abschnitte als Prozess-CPU-Zeit. Die CPU-Spitzenspeicherangabe `max_rss_kib` ist die Prozess-RSS vor der Referenzrechnung. Die Zeitspalten in `physics.csv` und `additional.csv` stammen aus parallel laufenden Screening-Fällen.
- `plain` und `weno` in der Timing-Datei verwenden den bestehenden `ElementWeno`-Kern mit fester Ordnung, vollem Winkelraum und aktiviertem M1-Limiter. Bei `plain` ist die WENO-Stabilisierung deaktiviert. `native` ist ein direkter Treiber dieses Kerns für identische periodische Anfangsdaten.
- `one_step_sigma_lower` ist der Massennormfaktor einer konkret berechneten Störung nach einem SSPRK(4,3)-Schritt. Die Potenziteration auf $S^*S$ sucht eine große Verstärkung. Ein Faktor größer als eins weist einen Energieanstieg nach. Ein Faktor unter eins liefert aus dieser Rechnung keine globale Kontraktionsgarantie. `20step_factor` setzt denselben gefundenen Anfangszustand über 20 Schritte fort.

Für das lineare periodische Verfahren bei überall gleichem $N$ liefert die Fourier-Analyse eine exakte Grenze. Die betragsgrößte dimensionslose eindimensionale CG-Frequenz ist $\max_k 3|\sin\theta_k|/(2+\cos\theta_k)$. Im kartesischen Gitter kommt dazu der Faktor $\sqrt d$ und die maximale Winkelgeschwindigkeit $\alpha_N$. Das SSPRK(4,3)-Polynom ist entlang der imaginären Achse bis $|z|=\sqrt{-8+4\sqrt{10}}$ kontraktiv. Diese Grenze steht als `exact_cfl_fixed` in den CFL-Dateien. Sie gilt für feste Ordnung bei abgeschalteter WENO- und M1-Korrektur.

## Befunde

Die Auswertung enthält 85 physikalische Konfigurationen, 192 CFL-Prüfungen einschließlich Wiederholungen mit mehr Potenziterationen und 128 einzelne Zeitläufe. In den physikalischen Läufen blieb die größte Abweichung der vier integrierten niedrigen Momente unter $4\times10^{-16}$. Die Energiediagnose registrierte dort keinen Anstieg während eines Transportschritts oder einer Projektion. Die zusätzlichen Schachbrettversuche liefern explizite Störungen mit Energieanstieg.

| Schmaler 2D-Puls auf 24², $C=0.30$ | Fest im Prototyp | Adaptiv | Mittlere adaptive Zustandsgröße |
|---|---:|---:|---:|
| P7 | 0.333 s | 0.132 s | 29 % |
| P11 | 0.755 s | 0.280 s | 34 % |
| P15 | 1.335 s | 0.523 s | 40 % |
| P21 | 2.543 s | 1.401 s | 50 % |

Die Tabelle verwendet das lineare PN-Ziel, den Start bei voller Ordnung und die Schwellen 0.002 und 0.004. Bei P15 steigt der Druckfehler auf diesem Gitter von 0.002409 auf 0.002519. Mit M1-Limiter braucht der adaptive Prototyp 0.614 s. Der bestehende `ElementWeno`-Kern mit festem P15, M1-Limiter und ohne WENO-Stabilisierung braucht im selben Versuch 0.224 s und erreicht einen etwas kleineren Druckfehler.

Auf 48² bei $C=0.60$ steigt der Druckfehler für den gerichteten Puls mit den ursprünglichen Schwellen von 0.000209 auf 0.000932 und für die gegenläufigen Pulse von 0.000148 auf 0.001172. Bei Schwellen von 0.00002 und 0.00004 liegen die adaptiven P15-Druckfehler wieder innerhalb eines Prozents des festen P15. Gegenüber festem P11 im selben Prototyp sind die Laufzeitgewinne in dieser Einstellung deutlich kleiner. Beim gerichteten Puls sinkt die Zeit von 3.46 s auf 3.30 s, bei den gegenläufigen Pulsen von 3.73 s auf 3.19 s. Der Zustand mit anfänglicher voller Ordnung benötigt beim Umbau gleichzeitig den alten und den neuen Raum. Die mittlere Zustandsgröße und der Prozessspitzenspeicher unterscheiden sich entsprechend.

Im periodischen 2D-Schachbrett aus P1 und P15 auf 6² gibt es bei $C=0.62$ einen nachgewiesenen Einzelschritt-Energieanstieg. Auf 12² wächst bei $C=0.875$ die Massennorm des gefundenen Zustands in 20 Schritten um den Faktor 1.31, während die exakte Kontraktionsgrenze für festes P15 auf diesem Gitter bei 0.8897 liegt. Im 3D-Schachbrett auf 4³ zeigt bereits $C=0.42$ einen Einzelschritt-Energieanstieg. Diese Befunde beziehen sich auf ungelimitierte diskrete Operatoren mit festgehaltener elementweiser Ordnung. Der gemeinsame M1-Zeitschrittdeckel beträgt auf den getesteten Gittern $C=0.50965$ in 2D und $C=0.35393$ in 3D. Ein 2D-Gegenlauf mit aktivem M1-Limiter bestand die Stufenprüfungen bei 99 Prozent des Deckels.

## Reproduktion

Die folgenden Befehle im Repository-Wurzelverzeichnis ausführen. Benötigt werden C++17, Eigen, OpenMP, Python 3.9 oder neuer und Linux mit `taskset`. Die Treiber verwenden die CPUs aus der aktuellen Prozessaffinität. Zeitmessungen laufen einzeln auf einem Kern.

```bash
study=scripts/element_weno/adaptive_pn_high_order
EIGEN3_INCLUDE_DIR=/usr/include/eigen3 bash "$study/build.sh"
python "$study/physics_campaign.py"
python "$study/additional.py"
python "$study/stability_campaign.py"
python "$study/stability_refine.py"
python "$study/timings.py"
```

Neue Läufe schreiben ihre CSV-Dateien nach `tmp/adaptive_pn_high_order`. Die [Prüfsummen](SHA256SUMS) dokumentieren die archivierten Rohdaten und die Systemangaben. Die [Timing-Mediane](timing_medians.csv) fassen die [einzelnen Zeitmessungen](timings.csv) zusammen.
