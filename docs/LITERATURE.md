# Literaturprüfung: PN-Massenkorrektur

## Zugängliche Quellen und Versionen

- **Moujaes, Kuzmin & Bäumer**, *Realizability-preserving monolithic convex limiting in continuous Galerkin discretizations of the M1 model of radiative transfer*, arXiv:2509.07689v1, eingereicht am 9. September 2025. Primärquelle: [arXiv-Seite](https://arxiv.org/abs/2509.07689), [PDF v1](https://arxiv.org/pdf/2509.07689v1). Lokale Kopie: `/home/user/bberg/phd/02_literature/Masterarbeit_Quellen/Paul Paper.pdf` (28 Seiten; Titelblatt weist arXiv:2509.07689v1 aus). Die Kopie wurde direkt ausgelesen.
- **Kuzmin & Hajduk**, *Property-Preserving Numerical Schemes for Conservation Laws*, World Scientific, 2023, DOI [10.1142/13466](https://doi.org/10.1142/13466). Lokale Kopie: `/home/user/bberg/phd/02_literature/Masterarbeit_Quellen/Kuzmin Buch.pdf` (492 Seiten; Buchsatz vom 26. Juni 2023). Die Gleichungsnummern unten beziehen sich auf die gedruckten Gleichungen im Buch.
- Die im Auftrag genannte Datei `M1PN_Paper_Paul_Literatur_20261102.pdf` war in der zugänglichen Dateisuche unter `/home/user/bberg` nicht auffindbar. Deshalb konnten ihre behaupteten Abschnitte und Gleichungsnummern nicht geprüft werden.

Die Bibliografie der zugänglichen M1PN-TeX-Datei nennt das Moujaes-Paper als *Mathematics and Computers in Simulation* 246 (2026), S. 570–590. Das lokale PDF ist die Preprint-Version v1 von 2025. Beim Zitieren sollten diese Fassungen nicht stillschweigend gleichgesetzt werden; die Journalfassung wurde hier nicht separat eingesehen.

## Was die Quellen tatsächlich belegen

Im Moujaes-Paper stehen die relevanten Formeln in **§4, „Monolithic convex limiting“**, nicht in (17)–(25):

- **(29), gedruckte PDF-S. 11:** Zerlegung der Differenz zwischen semi-diskreter CG- und Low-Order-M1-Residualform in antisymmetrische Korrekturflüsse. Darin tritt $m_{ij}(\dot u_i-\dot u_j)$ als Massenlumpingkorrektur auf.
- **(30)–(32), S. 12:** Ersetzung der konsistenten Zeitableitung durch die Low-Order-Ableitung $\dot u_i^L$, Definition stabilisierter Rohflüsse und des dazugehörigen Zielverfahrens samt Bar-State-Form.
- **(33)–(34), S. 12:** Begrenzte Bar-States und voll diskreter M1-Schritt.
- **Theorem 1, §3, S. 8; Bedingung $d_{ij}\geq|c_{ij}|$:** Beweist die Realisierbarkeit der Low-Order-M1-Bar-States. Der Text leitet danach für den vorwärts-Euler-Schritt unter der CFL-Bedingung (25) die IDP-Eigenschaft her. Das ist **kein Satz** über das vorgeschlagene PN-Ziel $M_L\dot a^H=R_C^N+(M_L-M_C)\dot a^L$, dessen Randkonsistenz oder dessen Dissipation.
- Die Aussagen (29)–(34) definieren und analysieren das M1-Verfahren. Sie belegen, dass Low-Order-Zeitableitungen zur Massenlumpingkorrektur in MCL bereits verwendet werden. Sie beweisen nicht automatisch eine IDP-Eigenschaft für eine Übertragung auf den PN-Targetoperator.

Im Kuzmin–Hajduk-Buch stehen die angefragten Formeln in **Kapitel 3, §3.2.7, „Convex limiting techniques“**, gedruckte **S. 115** (PDF-S. 137; S. 116 verweist zurück):

- **(3.137):** Low-Order-Zeitableitung aus Graphviskositäts- und Flusstermen.
- **(3.138):** Korrekturfluss $f_{ij}=m_{ij}(\dot u_i^L-\dot u_j^L)+d_{ij}(u_i-u_j)$.

Damit ist das Grundprinzip „Low-Order-Zeitableitung in einer Massenlumpingkorrektur“ für FE/AFC-Verfahren eindeutig **vorveröffentlicht**. Die spezielle Anwendung auf ein PN-Galerkin-Residual mit dem im Auftrag vorgeschlagenen Rand- und M1-Kopplungsledger ist durch diese Gleichungen allein weder als neu noch als bereits publiziert erwiesen. Eine Neuheitsbehauptung braucht eine breitere Literaturrecherche.

## Abgleich mit der tatsächlich vorhandenen M1PN-TeX-Datei

Zugänglich ist [M1PN_Paper_R2_20261029.tex](/home/user/bberg/phd/05_papers/M1PN_Paper_Revision_R2_20261029/M1PN_Paper_R2_20261029.tex), Titel „M1PN: transport, local gradient-defect stabilization, and cone-only convex limiting“. Sie ist **nicht** die im Auftrag bezeichnete Datei vom 2. November. Die Gleichungen tragen manuell gesetzte Tags statt normaler `\label`-Verweise.

- **§2.2, (2.8)–(2.9):** PN-Transportgleichung und schwache Galerkin-Zielgleichung mit konsistenter Masse.
- **§3.1, (3.3)–(3.6):** $M_C,M_L$, Transportvektoren, Randmatrixidentität und Steifigkeitsmatrix.
- **§3.3, (3.10)–(3.11):** konsistente PN-Massengleichung; $Z_i$ ist genau ein ausgewählter Stabilisierungsresidual.
- **§3.4, (3.13)–(3.20):** M1-Low-Order-Referenz, Bar-States und CFL-Bedingung.
- **§3.6, (3.21):** globale Randbilanz.
- **§4, insbesondere (4.1)–(4.26):** Steifigkeits-Biharmonik und lokaler Gradient-Defekt; kein Low-Order-PN-Zeitableitungsprädiktor.
- **§5, (5.4)–(5.7):** M1/PN-Flussdefekt und exakte projizierte Korrekturidentität. Die PN-Rate $v^H$ wird dort als bereits berechnete Rate des konsistenten Massenziels (3.10) eingeführt; (5.6) enthält $m_{ij}\Pi(v_i^H-v_j^H)$. Das ist nicht die im Auftrag vorgeschlagene Approximation $M_L^{-1}[R_C^N+(M_L-M_C)\dot a^L]$.
- **§5.5, (5.17)–(5.21):** Synchronisierung und Bilanz-/Energieaussagen.
- **§7, besonders (7.1) ff.:** vorhandene numerische Beobachtungen und offene Prüfungen des dortigen lokalen Gradient-Defekt-/Cone-only-Verfahrens. Sie sind keine Messungen der vorgeschlagenen PN-Massenkorrektur.

Die TeX-Bibliografie selbst verweist für Moujaes et al. auf „especially (17)–(25), (35)–(40)“. In zugänglicher arXiv-Version v1 liegen die gesuchten MCL-Massenkorrekturformeln tatsächlich bei **(29)–(34)**; die Abweichung sollte vor einem Paper-Zitat anhand der Journalfassung geprüft werden.

## Evidenzstatus für den neuen Auftrag

| Aussage | Status nach Quellenprüfung |
|---|---|
| Massenlumpingkorrektur mit Low-Order-Zeitableitung ist ein bekanntes FE/AFC-Muster. | **Publizierte Methode:** Kuzmin–Hajduk (3.137)–(3.138); im M1-MCL-Paper außerdem (29)–(31). |
| Low-Order-M1-Bar-States sind unter den Voraussetzungen des Moujaes-Theorems realisierbar. | **Bewiesener Satz:** Theorem 1 mit $d_{ij}\geq|c_{ij}|$; der vorwärts-Euler-IDP-Schritt verlangt zusätzlich die CFL-Bedingung. |
| Das PN-Ziel mit Low-Order-PN-Prädiktor besitzt die behauptete offene Randidentität, Fourier-Symbolformel oder globale Dissipativität. | **Hier nicht durch diese Quellen bewiesen.** Muss separat algebraisch hergeleitet und/oder reproduzierbar numerisch geprüft werden. |
| Affiner Randfehler (Z_0/m_0=c/3), Target-Rate (-2c/3), γ_eff=1/12 oder Vergleichsmessungen. | **Nicht durch die zugänglichen Primärquellen belegt;** keine numerischen Messwerte aus diesen Quellen ableiten. |

## Quellenlage im aktuellen Arbeitsordner

Die Quellenangaben oben beruhen auf zugänglichen Original-PDFs und Primärlinks. Die benannte M1PN-PDF vom 2. November blieb unzugänglich. Auch die Journalfassung des M1-MCL-Artikels wurde nicht separat geöffnet. Soweit deshalb Versions- oder Gleichungsnummern von der Journalfassung abweichen könnten, ist dies ausdrücklich offen.
