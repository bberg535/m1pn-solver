# Unabhängige Nachrechnung und Qualitätskontrolle

Die Implementierung und die mathematischen/numerischen Befunde wurden in getrennten Agentenaufgaben geprüft. Der prüfende Agent war nicht Autor des Solvers oder der Herleitung und arbeitete nur lesend. Dies ist eine nachvollziehbare interne Gegenprüfung, keine externe Begutachtung oder formale Verifikation.

## Bestätigte Mathematik

- Das periodische P1-Fourier-Symbol wurde aus einer unabhängig assemblierten FE-Matrix reproduziert; größter Fehler im Skalarskript bei 32 Zellen etwa 1.1e-12.
- Das affine offene Gegenbeispiel mit mindestens zwei Elementen liefert links w0=0, w1=-c, Z0/m0=c/3 und Target=-2c/3. Ein Ein-Element-Gitter ist hierfür kein gleichwertiger Test.
- Das offene Energiegegenbeispiel liefert a^T Z=c/12>0.
- Auf vier periodischen Elementen mit Längen .45,.05,.45,.05 und a=(1,3,-1,-3) ist a^T Z=.8 bei globaler Summe null. Damit ist allgemeine quadratische Dissipativität widerlegt.
- Die vollständige M1PN-Flussidentität mit B_ij und getrennten numerischen Randflüssen stimmt algebraisch.
- Die affine Aussage für C gilt unter den dokumentierten kartesischen, konstanten und kompatiblen Voraussetzungen. Sie ist kein allgemeiner Energie- oder Evolutionsfehlerbeweis.

## Im Review gefundene und behobene Fehler

1. Der erste Prädiktor benutzte schwache antisymmetrische Transportpaare. Auf offenen Q1-Flächen ist das nicht die angeforderte starke Form. Er wurde durch gerichtete C_ij-Differenzen und den knotweisen physikalischen Normalfluss ersetzt. Der starke Prädiktor wird jetzt gegen eine unabhängige Formel geprüft.
2. Der homogene Operator musste dieselbe ausgewählte PN-Randquadratur wie die Laufzeitauswertung verwenden.
3. Die Schrittweitenabschätzung benötigte beide numerischen Randbeiträge und eine eigene Diagonalabschätzung für gelumpte Randquadratur. Die Randkopie von C benötigt zusätzlich ihre gewichtete Operatornorm. Diese Beiträge sind enthalten. Die Abschätzung bleibt eine lineare Auflösungsregel.
4. Die C-Gitterprüfung wurde um aktive Dimensionen, lokale Tensorzellen und doppelte Zellen ergänzt. Das erste affine C-Testfixture hatte keine kompatiblen Einströmdaten; der fehlgeschlagene Lauf bleibt in c_algebra_v1.log erhalten. Das korrigierte Fixture besteht mit und ohne aktiven Limiter.
5. Der Buchverweis wurde auf gedruckte S.115/PDF-S.137 berichtigt.
6. Als relativ bezeichnete absolute L2/Linf-Methodendifferenzen wurden umbenannt. Der relative L1-Fehler enthält ausdrücklich Zähler und Nenner.
7. Randfehlerplots sind jetzt nach räumlicher Dimension getrennt. Die frühere Auswertung ist unter comparison_pre_audit erhalten und kein finaler Nachweis. Fehlerfarbskalen sind um null symmetrisch, Dichteskalen sind zwischen Methoden vergleichbar.
8. Die Knotennormen wurden um kontinuierliche FE-Quadraturnormen ergänzt. Die Knotenordnungen drei/vier dürfen nicht als allgemeine P1-Konvergenzordnungen gelesen werden. Linf aus Elementquadratur und Vertices ist ausdrücklich als abgetastet bezeichnet.

## Unabhängig geprüfte Rohdaten

Alle vier P7-Hauptläufe erreichen t=.32 mit 3201 Schritten. Anfangsfelder und Knotenmassen stimmen byteweise überein. Der gemeinsame Nenner des kinetischen relativen Dichte-L1-Fehlers beträgt .1356542564592883.

| Variante | relativer kinetischer Dichte-L1-Fehler | Dichte-L2 |
|---|---:|---:|
| A | .0517813950790 | .00974728367231 |
| Aex | .0519145730873 | .00980115955405 |
| B | .0508935823135 | .00944680891882 |
| C | .0509880177507 | .00944921763869 |

Diese Fehler enthalten PN-Modellfehler. Die Methodenabweichung C gegen A hat einen anderen Nenner: .1356931226785925; ihr relativer L1-Wert beträgt .00482298594119.

Für offene vierzellige P3-Matrizen wurde aus den vollständigen exportierten Operatoren unabhängig nachgerechnet:

- C: größter Eigenwert des gewichteten symmetrischen vollständigen Generators .06451198986661158. Ein normierter maximierender Vektor liefert direkt dieselbe positive Energiearbeit. Die Eigenwerte des Generators haben dennoch keinen positiven Realteil außerhalb der Rundung.
- Symmetrischer künstlicher Z-Anteil: Maximum B=.01454974918952049, C=.02763995011430393; A/Aex nur Rundung um 6e-17. Die künstlichen Z-Operatoren bleiben global konservativ.
- Diese Aussagen unterscheiden die Arbeit von Z und die Energieänderung des vollständigen Targets.

Die vier exakten Vakuum-Einströmversuche brechen sämtlich ab. Dies bleibt eine materielle Einschränkung der eingefrorenen MCL/RK-Rechenkette. Weder der bestandene Test des Nullzustands noch die bestandenen ersten Einströmstufen beweisen eine durchgehend zulässige Vakuumtrajektorie.

Die 16 CFL-Proben testen den Sicherheitsfaktor auf dem Minimum aus M1-CFL und methodenspezifischer PN-Regel. Sie bestimmen nicht die maximal stabile physikalische CFL. Hauptlaufzeiten auf verschiedenen gleichzeitig arbeitenden CPU-Kernen werden nicht als kontrollierte Geschwindigkeitsmessung verwendet; dafür liegt eine gesonderte Serie auf CPU5 ohne parallel laufende Hauptrechnungen vor.

## Verbleibende Grenzen

Keine allgemeine quadratische Dissipativität für B/C, kein globaler nichtlinearer Stabilitäts-/Entropiesatz und keine vollständige PN-Realisierbarkeitsgarantie. Keine universelle Gamma- oder Geschwindigkeitsoptimierung. Die konkret angeforderte M1PN-PDF vom 2.November war nicht zugänglich. Der eingefrorene Snapshot bleibt die Referenz, da der ursprüngliche Arbeitsordner während der Rechnung extern weiter verändert wurde.
