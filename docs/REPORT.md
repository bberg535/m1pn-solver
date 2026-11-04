# Abschlussbericht: PN-Massenkorrektur im M1PN-MCL-Verfahren

Stand: 4. November 2026
Arbeitsordner: `M1PN_MassCorrection_20261104`
Feature-Branch: `experiment/m1pn-pn-mass-correction`

## Ergebnis und Empfehlung

Die Standardvariante B implementiert
\[
M_L\dot a^H=R_C^N(a)+(M_L-M_C)\dot a^L
\]
mit einem PN-Low-Order-Prädiktor, ohne konsistenten Massensolve und ohne biharmonischen Operator. Die M1PN-Kopplung, Kegelbegrenzung und SSPRK3-Stufen bleiben in der vorhandenen Solverarchitektur.

Die unabhängigen komponentenweisen Identitätsprüfungen bestehen. Auf dem periodischen uniformen 1D-Gitter besitzt B das erwartete dissipative Fourier-Symbol und die effektive Innenraumskalierung \(\gamma_{\rm eff}=1/12\). Diese günstigen Aussagen verallgemeinern sich nicht auf beliebige Gitter: Es gibt geprüfte offene und periodische Gegenbeispiele mit positiver quadratischer Arbeit. B zeigt außerdem den vorhergesagten Randfehler für kompatible affine offene Daten.

Die getrennte Variante C kopiert den fertigen Low-Order-Prädiktor an Randknoten von einem inneren Knoten. Sie beseitigt den affinen Randfehler auf den geprüften uniformen Gittern, ändert aber die globale Energiebilanz des Stabilisierungsterms nicht zu einem allgemeinen Dissipativitätsnachweis. C ist zudem auf geprüfte uniforme kartesische Tensorproduktgitter beschränkt.

**Empfehlung:** Die bestehende biharmonische Familie A/Aex bleibt die Paper-Hauptmethode. B und C werden als eigenständige experimentelle Kandidaten berichtet. B liefert beim Crossing-Beams-Test etwas kleinere globale Fehler als A, jedoch größere Fehler in der Kreuzungszone und am äußersten Randknoten; im breiteren Randstreifen ist sein Fehler kleiner. Der affine Randtest ist für B klar schlechter. C repariert den affinen Test, verbessert den Crossing-Beams-Fehler gegenüber B aber nicht. Für keine der Varianten B/C liegt ein allgemeiner quadratischer Stabilitäts- oder Dissipativitätsbeweis des vollständigen begrenzten M1PN-Verfahrens vor. Ein Ersatz der Biharmonik ist durch die Ergebnisse nicht begründet.

**Materielle Einschränkung:** Der Einströmversuch in exaktes physikalisches Vakuum bricht bei allen vier Varianten an den gespeicherten Kegelprüfungen ab. Die bestandenen automatisierten Tests ersetzen diesen fehlgeschlagenen Evolutionsbenchmark nicht.

## 1. Repository-Stand und Quellen

Die Implementierung wurde in einem neuen isolierten Arbeitsordner auf dem Feature-Branch aufgebaut. Referenzbasis ist der eingefrorene Snapshot `ce1e785`; der Standardkandidat wurde als `9900774`, die abschließende Implementierung als `e41724e` ausgewiesen. Der aktuelle Provenienzdatensatz mit Branch, Maschinen-, Compiler- und Binärhashes liegt in [results/provenance.json](../results/provenance.json). Die vollständige geänderte-Dateien-Liste liegt in [changed_files.txt](../results/changed_files.txt). Die fünf geänderten Solverdateien sind transport.cpp/.hpp, time.cpp, main.cpp und common.hpp; hinzu kommen die neuen Tests, Versuchstreiber und Dokumente.

Der lokale Solver stellt bereits konsistente und gelumpte Massenmatrizen, FE-Nachbarschaften, PN-Flussmatrizen, physikalische PN- und M1-Randbeiträge, M1-Projektion, monolithische konvexe Begrenzung und SSPRK3 bereit. B nutzt diese vorhandenen Pfade. Es wird keine parallele Solverarchitektur eingeführt.

Der im Auftrag genannte Bericht `M1PN_Paper_Paul_Literatur_20261102.pdf` und eine zugehörige TeX-Fassung waren lokal nicht auffindbar. Die zugängliche Paper-TeX-Fassung vom 29.10.2026 wurde für die FE-Matrizen, PN-/M1-Ziele, Randflüsse und bisherige Biharmonische herangezogen. Quellen, Versionsstände und die nicht verifizierbaren Gleichungsnummern sind in [LITERATURE.md](LITERATURE.md) dokumentiert.

Die Massenkorrektur-Idee ist bereits in Kuzmin–Hajduk, (3.137)–(3.138), und Moujaes–Kuzmin–Bäumer, (29)–(32), beschrieben. Die Übertragung auf die hier implementierte PN-Zeitableitung, Randlast und M1PN-MCL-Kopplung ist eine Anwendung auf das vorhandene Verfahren; eine Neuheit der Grundidee wird nicht beansprucht.

## 2. Herleitung und Abgrenzung der drei Varianten

Die vollständige algebraische Herleitung, sämtliche Flussformeln und die Voraussetzungen stehen in [MATHEMATICS.md](MATHEMATICS.md).

Es seien \(M_C=(m_{ij})\), \(M_L=\operatorname{diag}(m_i)\), \(m_i=\sum_jm_{ij}\), und \(Q=M_L-M_C\). Der PN-Low-Order-Prädiktor lautet am Knoten \(i\)
\[
 m_i\dot a_i^L=\sum_{j\ne i}\left[d^N_{ij}(a_j-a_i)-c_{ij}\cdot(\mathcal A(a_j)-\mathcal A(a_i))\right]+b_i^N,
\]
mit symmetrischer Graphviskosität \(d^N_{ij}=\lambda_N\max(|c_{ij}|,|c_{ji}|)\). Für offene Ränder enthält \(b_i^N\) den nodalen physikalischen PN-Trace und die negative, tatsächlich assemblierte numerische PN-Randlast. Die optionalen Quelllasten werden im Prädiktor und Ziel gleich behandelt.

B bildet danach genau einmal
\[
Z=Q\dot a^L,\qquad \dot a^H=M_L^{-1}(R_C^N+Z).
\]
Die Umsetzung benutzt die vorhandene konsistente Matrix zur Multiplikation, aber weder deren Faktorisierung noch einen Solve. Der Beitrag lässt sich auch über die existierenden Massekanten als \(Z_i=\sum_jm_{ij}(\dot a_i^L-\dot a_j^L)\) auswerten. Es entstehen keine neuen Graphkanten.

In der vorhandenen MCL-Zerlegung ist der neue hohe PN-Paarfluss \(m_{ij}(\dot a_i^L-\dot a_j^L)\) einmal enthalten. Die M1-Referenz \(u_i=\Pi_1a_i\), der Modellflussdefekt \(T_i=\Pi_1\mathcal A(a_i)-f(u_i)\), der \(B_{ij}=c_{ij}+c_{ji}\)-Term und die physikalische Randkorrektur \(H_{Fi}^{M1}-\Pi_1H_{Fi}^N\) bleiben bestehen. Die vorhandenen gemeinsamen symmetrischen Limiterfaktoren begrenzen weiterhin die gerichteten Bar-States. Die Synchronisierung erhält \(\Pi_1k_i=\ell_i\). Die vollständigen Herleitungen und Voraussetzungen stehen in [MATHEMATICS.md](MATHEMATICS.md).

A bezeichnet die biharmonische `biharmonic-corrected`-Variante mit \(\gamma=0.004\). Aex bezeichnet die vorhandene extrapolierte quartische Gram-Biharmonik mit \(\gamma=0.00032911848351755427\). Beide bleiben unverändert. B ist `m1pn_mass_correction`; C ist `m1pn_mass_correction_boundary_consistent`. C wendet die Koordinaten-Clamp-Abbildung auf den **vollständig assemblierten** PN-Prädiktor an und berechnet anschließend dieselbe einmalige Massenkorrektur. C verändert weder PN-/M1-Randflüsse noch M1-Referenz oder MCL.

## 3. Algebraische Identitäten und numerische Verifikation

Weil \(Q\mathbf1=0\), gilt für jede PN-Komponente
\[
\sum_i Z_i=\mathbf1^TQ\dot a^L=0.
\]
Das bleibt auch für C wahr: Die kopierte Prädiktorvariable kann selbst nicht konservativ sein, doch die abschließende symmetrische Massenkorrektur ist global konservativ.

Die vollständige M1PN-Target-Identität folgt aus den diskreten partiellen Integrationen und den tatsächlich assemblierten Randlasten:
\[
m_i\Pi_1\dot a_i^H-L_i^{M1}=\sum_j f_{ij}^{MC}+\sum_{F\ni i}f_{iF}.
\]
Für ungleiche PN- und M1-Quellresiduen ist zusätzlich \(\Pi_1r_i^N-r_i^{M1}\) einzusetzen. Die unabhängige Studie prüft die Komponenten- und Paarbeiträge getrennt und vergleicht sie mit einer eigenständigen FE-Assembly. Für ungleiche PN-/M1-Quellen ist dieser Residualunterschied zwingend; eine automatische Gleichheit unterschiedlicher Reaktionsdiskretisierungen wird nicht angenommen.

Die Target-Approximation ist nicht das konsistente PN-Verfahren selbst. Falls \(v_C=M_C^{-1}R_C^N\), gilt exakt
\[
\dot a^H-v_C=M_L^{-1}(M_L-M_C)(\dot a^L-v_C).
\]
B/C sind damit umso näher am konsistenten Ziel, je näher der Prädiktor an dessen Zeitableitung liegt. Dies ist eine Approximation und kein verborgen ausgeführter Solve.

Für die CLI wird bei B/C ohne explizite Massenangabe automatisch `--mass-mode lumped` gesetzt. Ein ausdrücklich anderes Massenmodell wird sichtbar abgewiesen. Die PN-Graphviskosität verwendet standardmäßig die maximale PN-Wellengeschwindigkeit (Skalenfaktor 1); der exponierte Skalenparameter ist auf Werte mindestens eins beschränkt. B/C sind nur mit SSPRK3 auswählbar. Kein Kandidat aktiviert zusätzlich eine Biharmonik.

Der Standard- und C-Fokuslauf prüfte Quell- und Randbeiträge, die \(D-M\)-Massenkorrektur, Komponentenerhaltung, den einmaligen Zielaufbau, die volle Target-Rekonstruktion, fehlende Faktorisierung/Solves und fehlende Biharmonik/Geisterzellen. Dazu kamen periodische und offene 1D-, 2D- und 3D-Fälle, Randexterioren, Vakuumzustand, SSPRK3-Stufen sowie für C die Clamp-Abbildung, affine Annihilation, unveränderte FE-Graphen und Ablehnung nicht unterstützter Gitter. Die finale fokussierte Suite besteht aus 190 Prüfungen und ist in [final_algebra_verified.log](../results/final_algebra_verified.log) protokolliert.

Alle 26 CTest-Tests bestanden; Protokoll: [ctest_final_verified.log](../results/ctest_final_verified.log). Die deterministischen Regressionen für Dimension 1, 2 und 3 stimmen mit dem eingefrorenen Ausgangsstand byteweise überein: Endfelder, Stufenfelder und Rand-/Bilanz-Ledger. Details: [regression.json](../results/comparison/regression.json).

## 4. Randanalyse: Gegenbeispiel, Variante C und physikalischer Einlass

Für 1D-Advektion \(a_t+ca_x=0\), \(c>0\), kompatible affine Einströmdaten und Standard-LLF-Randbehandlung reproduziert die direkte diskrete Rechnung den vorhergesagten Standardfehler: \(\dot a_0^L=0\), \(\dot a_1^L=-c\), \(Z_0/m_0=c/3\), also \(\dot a_0^H=-2c/3\) statt \(-c\). Der unabhängige Knoten-Patch-Test bestätigt den gleichen Effekt in 1D; auf uniformen offenen Tensorproduktgittern in höheren Dimensionen stimmen die Rohkorrektur und unabhängige Assembly auf Rundungsgenauigkeit überein.

C kopiert im offenen gleichmäßigen Raster den vollständig assemblierten Prädiktorwert eines benachbarten inneren Knotens **auf den Randknoten**. Für kompatible affine Daten wird der innere konstante Prädiktorwert damit auch am Rand verwendet; daraus folgt \(Q\dot a^\sharp=0\). In den 1D-, 2D- und 3D-Patch-Prüfungen waren die C-Korrekturen für affine Daten nur Rundungsrauschen und die vollständige Ableitung reproduzierte die analytische Referenz. C verlangt mindestens drei gleichmäßige kartesische Zellen je aktiver Achse; es weist nichttensorielle und tetraedrische Gitter zurück. Für periodische Gitter ohne Randfläche ist die Abbildung die Identität.

Ein unabhängiger offener P1-Test ergibt für B die Randwerte \(Z_0/m_0=c/3\) und \(\dot a_0^H=-2c/3\). Das ist eine lokale Konsistenzstörung; aus ihr allein folgt keine globale Konvergenzrate. Die beobachteten Konvergenzläufe sind separat ausgewertet.

Der affine Transportlauf mit 32 Zellen, PN-Ordnung 3 und \(t=0.05\) ergab für A, Aex und C Dichtefehler nahe Rundungsgenauigkeit (\(L^\infty\) etwa \(10^{-16}\)); für B war der Dichtefehler \(5.3112\cdot10^{-4}\) und der Impulsfehler \(1.0019\cdot10^{-4}\). Dieser Test isoliert Randkonsistenz. C verbessert ihn durch eine deklarierte Randprädiktor-Korrektur, nicht durch eine geänderte physikalische Randbedingung.

Der physikalische isotrope Einlass-/Vakuum-Pulstest konnte für keine der vier Varianten bis zum Zielzeitpunkt ausgeführt werden: A, Aex, B und C scheiterten nach konservativen Kegel-Retrys mit „Cone step rejected after 13 conservative retries“. Null-Vakuum und gezielte erste SSPRK3-Stufen bestanden. Das ist ein klarer offener Benchmark und kein Erfolg der Varianten: Es wurden keine Dichtefloors, kein Clipping und keine M1-Änderungen verwendet, um den Einlasstest künstlich zu bestehen. Die Protokolle stehen unter `tmp/boundary/vacuum_inflow_*`.

## 5. Glatte Lösungen und Konvergenz

Die glatten Eigenwellenläufe verwenden den PN-Eigenwert \(-0.8611363115940533\), \(T=0.05\), \(\Delta t=10^{-4}\), Gitter mit 16, 32 und 64 Zellen. MCL war aktiviert; in diesen Läufen trat keine Paarbegrenzung auf. Die im Ergebnisarchiv aktualisierte FE-Quadraturmessung zeigt im periodischen und offenen Fall ungefähr zweite Ordnung in den FE-gewichteten PN-\(L^2\)-Fehlern für die untersuchten Varianten.

| Rand | Gitter | PN | A | Aex | B | C |
|---|---:|---:|---:|---:|---:|---:|
| periodisch | 64 | \(6.220\cdot10^{-7}\) | \(6.227\cdot10^{-7}\) | auf periodischen Gittern nicht definiert | \(6.358\cdot10^{-7}\) | \(6.358\cdot10^{-7}\) |
| offen | 64 | \(6.193\cdot10^{-7}\) | \(6.121\cdot10^{-7}\) | \(6.164\cdot10^{-7}\) | \(7.668\cdot10^{-7}\) | \(7.459\cdot10^{-7}\) |

Die Werte sind kontinuierliche PN-L2-Fehler aus Elementquadratur. L1 und L2 werden integriert; Linf wird an Quadraturpunkten und Elementknoten abgetastet. Die Raten der letzten Verfeinerung 32→64 betragen periodisch A=2.001, PN=1.999, B=C=2.030; offen A=2.005, Aex=1.994, B=1.884, C=2.045 und PN=1.990. Es handelt sich nicht um punktweise Knotenfehler. Frühere nodale Massen-Normen zeigen für B/C auf periodischen glatten Daten eine scheinbare dritte Ordnung. Die FE-Quadratur bestätigt diese als Superkonvergenz der ausgewählten nodalen Messgröße; sie ist keine generische P1-Konvergenzordnung. Die vollständigen Werte und Raten liegen in [study_convergence.csv](../results/comparison/study_convergence.csv), die Abbildungen in [convergence_fe.png](../results/comparison/convergence_fe.png) und [convergence.png](../results/comparison/convergence.png).

Die Patch-Suite enthält Konstanten, affine, quadratische und glatte Felder, getrennte innere/Rand-/Kanten-/Ecknormen von \(M_L^{-1}Z\) sowie die Ableitungsprüfung. Die Rohdaten stehen in `results/comparison/study_patches.csv` und `results/comparison/boundary_error_strata_*.png`. Damit sind die geprüften Gitterklassen abgedeckt, aber nicht beliebig verzerrte Meshes oder ein allgemeiner Konvergenzsatz.

## 6. Crossing Beams: Genauigkeit, Kegel und Bilanz

Der Hauptvergleich verwendet PN-Ordnung 7, 64 Elemente pro Achse, denselben Crossing-Beams-Zustand, dieselben Randdaten und \(t=0.32\). Alle vier Fälle erreichten den Endzeitpunkt mit 3201 SSPRK3-Schritten und ohne verworfenen Zeitschritt. Die M1-Kegel-Marge blieb positiv; der absolute komponentenweise Gesamtbilanzrest lag bei rund \(10^{-14}\). A/Aex sind biharmonische Referenzen, B/C sind die neuen Massenkorrekturvarianten.

| Variante | relatives \(L^1\) zu kinetischer Dichte | Dichte-\(L^2\) | Dichte-\(L^\infty\) | äußerster Randknoten-\(L^\infty\) | Randstreifen-\(L^\infty\) | Kreuzungszonen-\(L^\infty\) | aktive Paarfraktion |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.051781 | 0.009747 | 0.072889 | 0.015352 | 0.021763 | 0.027020 | 0.09003 |
| Aex | 0.051915 | 0.009801 | 0.076422 | 0.015249 | 0.023356 | 0.027078 | 0.09039 |
| B | **0.050894** | **0.009447** | **0.052531** | 0.015499 | **0.015514** | 0.028754 | 0.08722 |
| C | 0.050988 | 0.009449 | 0.052536 | **0.015248** | 0.015923 | 0.028755 | 0.08794 |

„Relatives \(L^1\)“ bedeutet hier \(\|\rho_h-\rho_{kin}\|_{L^1}/\|\rho_{kin}\|_{L^1}\), mit Nenner \(0.1356543\). Die kinetische Dichte-Referenz enthält den Fehler der endlichen PN-Modellordnung und ist **keine** exakte Lösung des PN-PDE-Systems. Die separat normierten Abweichungen zur A-Lösung sind in `crossing_errors.csv` ausgewiesen. B gewinnt den globalen Dichtefehler knapp, verliert aber im Kreuzungsbereich und am äußersten Randknoten gegen A/Aex; im breiteren Randstreifen ist B besser. C ist bei globaler Dichte und Kreuzungszone nahezu gleich B, hat den kleineren äußersten Randknotenfehler, aber einen etwas größeren Randstreifenfehler. Das Bild allein reicht daher nicht als Qualitätsbeweis.

Alle vier Läufe hatten keine negativen Dichten, keinen Dichte-Clip und keine Impulsumleitung. `global_Q_balance` ist der absolute \(\ell^\infty\)-Bilanzrest von etwa \(9.2\cdot10^{-15}\), kein relativer Fehler. Das beweist nur die beobachtete Realisierbarkeit dieser Rechnungen. Die hohen PN-Momente unterscheiden sich messbar: harmonische Koeffizientenvergleiche bis Grad 7 sind in [degree_errors.csv](../results/harmonics/degree_errors.csv) und [degree_spectrum.png](../results/harmonics/degree_spectrum.png). Die Referenz dort ist die projizierte analytische kinetische Referenz; sie enthält ebenfalls endliche PN-Modellabweichung.

Alle Dichte-L1/L2-Normen in diesem Crossing-Vergleich sind nodale, mit den FE-Knotenmassen gewichtete Normen. Der Randstreifen umfasst x<.1 oder x>.9 oder y<.1 oder y>.9; die Kreuzungszone ist die Kreisscheibe mit Radius .2 um (.5,.5). Der äußerste Rand umfasst die tatsächlichen Randknoten. Der letzte der 3201 Schritte hat wegen des aufsummierten Rundungsrests nur etwa 1.9e-14 Zeitlänge.

Weitere Tabellen, Profile und Dichtemaps sind unter `results/comparison/campaign_summary.csv` sowie `results/comparison/crossing_maps/` abgelegt.

## 7. Spektren, Energie und Zeitschritt

Auf uniformen periodischen 1D-P1-Gittern besitzt der Standardoperator das unabhängige Symbol
\[
\sigma(\theta)=-\frac{ic}{h}\sin\theta\left(1+\frac{1-\cos\theta}{3}\right)-\frac{\lambda_N}{3h}(1-\cos\theta)^2.
\]
Der führende glatte Dissipationsterm ist \(-\lambda_Nh^3\partial_x^4a/12\). Daher ist die Innenraum-Zuordnung \(\beta=\gamma\lambda_Nh^3\) mit \(\gamma_{eff}=1/12\) nur asymptotisch und nur für diesen Modus-/Gitterfall plausibel. Sie behauptet keine gleiche Hochfrequenzdämpfung, keine mehrdimensionale Äquivalenz und keine allgemeine Stabilität.

Die vollständigen FE-Spektren und symmetrischen Anteile wurden auf periodischen sowie offenen Gittern ausgewertet. Diese Matrizen beschreiben den homogenen, unbegrenzten PN-Targetoperator; der nichtlineare MCL-gekoppelte Solver besitzt keine einzige solche globale Generatormatrix. Für ein uniformes periodisches 1D-Gitter war die künstliche Arbeit von B nichtpositiv. Für ein offenes Ein-Element-Beispiel war sie positiv; auf einem nichtuniformen periodischen Wechselgitter wurde ebenfalls positive Arbeit gefunden. Für das uniform offene P3-Vollgitter hatte C einen positiven maximalen symmetrischen Eigenwert des vollständigen gewichteten Generators von \(0.06451198986661\); B lag für diesen Generator nahe null, obwohl sein isolierter Korrekturanteil positive Richtungen hat. Die isolierten symmetrischen Korrekturanteile ergaben \(\mathrm{sym}Z_{max}=0.01454974918952\) für B und \(0.02763995011430\) für C; A/Aex lagen für den künstlichen Term nahe null. Die vollständigen Zahlen und Spektren stehen in [study_spectrum_summary.csv](../results/comparison/study_spectrum_summary.csv), [study_spectrum_eigenvalues.csv](../results/comparison/study_spectrum_eigenvalues.csv) und [degree_errors.csv](../results/harmonics/degree_errors.csv).

Die Gegenbeispiele widerlegen die Behauptung \(a^TZ\le0\) für beliebige Gitter. Die konservative Eigenschaft \(\sum_iZ_i=0\) ist davon unabhängig. Eine positive momentane symmetrische Rate beweist für sich allein keine Langzeitinstabilität des begrenzten nichtlinearen Verfahrens, aber sie schließt einen allgemeinen quadratischen Dissipativitätsbeweis für B/C aus. Die biharmonische Gram-Form hat für den isolierten Term den bekannten Nachweis \(a^TZ^{BH}=-\|D^{1/2}La\|^2\le0\); dieser gilt nicht automatisch für das gekoppelte vollständige Verfahren.

Der 1D-periodische SSPRK3-Zeitcontroller verwendet für B/C die diskrete PN-Fourieramplifikation. In mehreren Dimensionen und bei offenen Gittern wird eine konservative Operatornorm-Auflösungsregel verwendet. Ein separates 16-Punkt-CFL-Screening erreichte für alle Varianten die getesteten Einstellungen bis `--cfl 1`; daraus folgt kein maximal stabiler CFL-Beweis. Im Crossing-Hauptlauf war der festgesetzte Zeitschritt für alle Varianten \(10^{-4}\); relativ zur M1-Low-Order-Grenze ist dies in allen vier Fällen etwa 0.0507. Ein separates CFL-Screening mit angeforderter 1.0 ergab maximale Schritt-zu-M1-Grenz-Verhältnisse 0.12719 (A), 0.11203 (Aex), 0.37633 (B) und 0.28184 (C). Das ist eine geprüfte Konfiguration und keine Stabilitätsgrenze.

Die CFL-Screening-Verhältnisse gehören ausschließlich zu den 16 separat ausgewerteten Spektral-/Controllerfällen mit angeforderter `--cfl 1`. Sie beschreiben weder den Crossing-Hauptlauf noch eine analytische Grenze. Für diesen Hauptlauf gilt für alle Kandidaten dieselbe Zeitsteuerung und dasselbe Verhältnis zum M1-Low-Order-Limit.

Der Spektraltest trennt drei Aussagen, die nicht vermischt werden dürfen: Erhaltung folgt aus \(Q\mathbf1=0\); momentane quadratische Arbeit wird durch den symmetrischen Anteil des Operators bestimmt; und die Stabilität der vollständigen SSPRK3-MCL-Rechnung hängt zusätzlich von Nichtlinearität, Limiter und Schrittweite ab. Die hier gemessenen positiven Eigenwerte sind Gegenbeispiele zu einem allgemeinen quadratischen Dissipativitätsbeweis, aber keine Vorhersage eines konkreten Crossing-Ausfalls.

## 8. Rechenkosten, Builds und Regression

Die kontrollierte Performance-Serie lief auf einem festgelegten CPU-Kern und umfasst drei alternierende Wiederholungen je Konfiguration; andere Varianten wurden nicht gleichzeitig auf den gemessenen Kern gelegt. Kleine Konfiguration: 8² Elemente, PN3; mittlere Konfiguration: 16² Elemente, PN7; in beiden Fällen \(T=0.01\), \(\Delta t=10^{-4}\), dieselbe Crossing-Konfiguration. Die Tabelle zeigt den Median der gemessenen `solver_seconds` über drei Wiederholungen. Diese Kennzahl schließt Initialisierung und Ein-/Ausgabe aus; sie ist keine gesamte Prozesslaufzeit. Die Minima/Maxima, Zeit je SSP-Stufe, Prozesslaufzeiten, Schrittzahlen, Peak-RSS und Limiteraktivität stehen in [performance_summary.csv](../results/performance_summary.csv); vollständige Einzelversuche in [trials.json](../results/campaigns/performance_quiet/trials.json). Jede dieser Rechnungen führte 101 SSPRK3-Schritte aus, einschließlich eines abschließenden Rundungsrestschritts.

| Variante | 8² / PN3, Solversekunden | 16² / PN7, Solversekunden | PN-Prädiktor beim mittleren Fall | Korrektur beim mittleren Fall | Peak RSS mittel |
|---|---:|---:|---:|---:|---:|
| A | 0.3374 | 1.9938 | – | – | ca. 29 MiB |
| Aex | 0.3526 | 2.2905 | – | – | ca. 30 MiB |
| B | 0.4010 | 2.1960 | ca. 0.407 s | ca. 0.0259 s | ca. 28 MiB |
| C | 0.4136 | 2.1990 | ca. 0.409 s | ca. 0.0253 s | ca. 28 MiB |

B/C rufen keinen konsistenten Massensolve auf und melden im kontrollierten Lauf null Biharmonikzeit. Der Prädiktor kostet in diesen gemessenen Fällen mehr als die Massenkorrektur. B ist beim mittleren Fall rund 10 % langsamer als A und nur etwa 4 % schneller als Aex; C liegt ähnlich wie B. Das ist kein universeller Speedup. Frühe Hauptläufe wurden mit konkurrierenden CPU-Jobs ausgeführt und werden deshalb nicht für Performanceaussagen genutzt. Ein älteres B-Protokoll mit 7.802 s unter „biharmonic_seconds“ maß nur die anfängliche Nullpuffer-Einrichtung; der finale kontrollierte B-Lauf misst die Biharmonik korrekt mit null.

Die Auswertung ist ein kontrollierter Vergleich dieser zwei Konfigurationen, keine Skalierungsstudie. Die Speichermessung ist Peak-RSS des Prozesses; die Unterkomponenten „Prädiktor“ und „Korrektur“ sind instrumentierte Solverzeiten und summieren sich nicht zwingend exakt zur Gesamtzeit, da weitere Solverarbeit enthalten ist. Die mittleren Werte zeigen, dass der Verzicht auf den konsistenten Solve allein keine Laufzeitgarantie ergibt: Der PN-Prädiktor verursacht selbst messbare Kosten.

Sauberer Build und alle Tests sind in `results/build_clean_final.log` und `results/ctest_final_verified.log` protokolliert. Reproduzierbare Befehle und Parameter befinden sich in [README.md](../README.md), `scripts/` und `docs/EXPERIMENT_PLAN.json`. Die umfangreichen Rohfelder bleiben in `tmp/`; die Abschlussübersichten, Tabellen und Plots liegen unter `results/comparison/`. Die unabhängige Prüfung steht in [AUDIT.md](AUDIT.md).

Der eingefrorene Referenzstand blieb byte-identisch, wie die Hash-/Regressionprüfungen belegen. Das benachbarte ältere Arbeitsverzeichnis `M1PN_Lazy_Einstrom_20261104` änderte sich parallel durch externe Arbeiten; deshalb wird nur die Regression gegen den explizit eingefrorenen Snapshot `ce1e785` beansprucht. FCS war im eingefrorenen Ausgangsstand nicht enthalten; Änderungen in anderen Arbeitsverzeichnissen sind nicht Teil dieser Untersuchung. Es wurden keine zusätzlichen Pakete installiert und keine Slurm-Jobs gestartet.

## 9. Einschränkungen und Paper-Vorschlag

Die mathematisch gesicherten Aussagen sind die diskrete Target-Identität unter den angegebenen FE-/Randkonventionen, komponentenweise globale Konservativität der Massenkorrektur, die gemessene Fourierform für den uniformen periodischen 1D-Fall sowie die bedingte affine Annihilation von C auf den geprüften uniformen Tensorproduktgittern. Die unabhängigen Prüfungen belegen die implementierte algebraische Auswertung und Regression.

Nicht gesichert sind eine allgemeine quadratische Dissipativität von B oder C, eine allgemeine SSPRK3-Stabilitätsgrenze des vollständigen begrenzten PN-Systems, ein globaler Konvergenzsatz, Realisierbarkeit beliebiger hoher PN-Rekonstruktionen oder Robustheit gegen den gescheiterten physikalischen Vakuum-Pulseinlass. C besitzt eine bewusst eng begrenzte Gittervoraussetzung und keinen allgemeinen Energiebeweis. Die Crossing-Daten umfassen den geforderten Hauptfall; die Kennzahlen ersetzen keine vollständige Parameter-/Randbenchmark-Suite für alle PN-Ordnungen und Außenzustände.

Die Abschlussentscheidung entspricht **Variante A des Auftrags**: Die vorhandene Biharmonik einschließlich der extrapolierten Aex bleibt die Hauptmethode im Paper. B kann als separater PN-Massenkorrekturkandidat mit guter globaler Crossing-Dichtegenauigkeit und nachgewiesener Rand-/Energielimitation dargestellt werden. C ist ein gezielter Randkandidat mit erfolgreichem affinem Patchtest, jedoch ohne allgemeine Dissipationseigenschaft und ohne Genauigkeitsgewinn im Crossing-Fall. Vor einer Ersetzung wären mindestens der physikalische Einlassbenchmark, weitergehende offene Randvergleiche, unabhängige Stabilitätsgrenzen und breitere Konvergenzstudien zu lösen.

Die Berichtsstruktur trennt deshalb algebraische Identitäten, unabhängige Implementierungsprüfungen und empirische PDE-Studien. Die Identität für \(Z\) ist für jede zulässige Eingabe exakt; die konkrete Stabilitäts- und Genauigkeitsbewertung bleibt an die ausgewerteten Gitter, Randbedingungen, PN-Ordnungen und Anfangsdaten gebunden. Die separate [Audit-Zusammenfassung](AUDIT.md) dokumentiert die unabhängige Prüfung, während [MATHEMATICS.md](MATHEMATICS.md) die Formeln und Gegenbeispiele ausführt.
