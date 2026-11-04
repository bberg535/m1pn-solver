# Mathematischer Prüfbericht: PN-Massenkorrektur

Stand: 4. November 2026. Dieser Bericht enthält einen eigenständigen Beweisversuch und algebraische Gegenbeispiele. Seine Herleitungen sind nicht durch den Verfasser selbst als unabhängig geprüfte Sätze zertifiziert. Numerische Solver-Ergebnisse werden hier nicht behauptet. Die unabhängige Prüfung muss die angegebenen Voraussetzungen und Rechnungen ausdrücklich kontrollieren.

## 1. Exakte Aussage, Quantoren und Geltungsbereich

Sei für jedes betrachtete feste konforme Gitter eine nichtnegative nodale P1-/Q1-Basis gegeben. Vorausgesetzt werden dieselbe konsistente Volumen-/Randquadratur sowie

\[
m_{ij}=m_{ji},\quad m_i=\sum_jm_{ij}>0,\quad
\sum_jc_{ij}=0,\quad B_{ij}=c_{ij}+c_{ji}=B_{ji}.
\]

Setze \(Q=M_L-M_C\). Die Koeffizienten \(a_i\in\mathbb R^{(N+1)^2}\) sind beliebig; \(N\) ist eine nichtnegative ganze Zahl. Für das gekoppelte M1PN-System wird zusätzlich eine festgelegte Projektion \(u_i=\Pi_1a_i\) auf die vier ersten Momente vorausgesetzt. Für die M1-Kegelaussagen muss \(u_i\) im geschlossenen Realisierbarkeitskegel liegen. Die Algebra der Rohkorrekturen benötigt diese Kegelannahme nicht.

Die linearen PN-Flüsse lauten \(\mathcal A_r(a)=A_ra\). Für orthonormale reelle Kugelflächenfunktionen sind die \(A_r\) symmetrisch. Die skalare Graphgeschwindigkeit \(\lambda_N\) muss die benutzte Richtungs-Spektralgeschwindigkeit decken; in dimensionslosen Transporteinheiten ist \(\sup_{|n|=1}\rho(A(n))\leq1\). Eine niedrigere exakte PN-Geschwindigkeit darf nur mit entsprechender Spektralprüfung verwendet werden. Setze beispielsweise

\[
d_{ij}^N=d_{ji}^N=\lambda_N\max(|c_{ij}|,|c_{ji}|),\qquad i\ne j.
\]

Alle offenen Randflüsse sind nach außen orientiert. Definiere die tatsächlichen assemblierten Lasten \(H^N_{Fi}\), \(H^{M1}_{Fi}\) mit den jeweils eigenen äußeren Daten. Sie müssen nicht dieselben ersten Momente besitzen. Ohne Quellen sei

\[
R^N_{C,i}(a)=\sum_jc_{ji}\cdot\mathcal A(a_j)-\sum_{F\ni i}H^N_{Fi}.
\]

Der Standardprädiktor ist durch

\[
m_iw_i=\sum_{j\ne i}\left[d^N_{ij}(a_j-a_i)
-c_{ij}\cdot(\mathcal A(a_j)-\mathcal A(a_i))\right]
+\sum_{F\ni i}\left[b_{Fi}A(n_F)a_i-H^N_{Fi}\right]
\tag{M1}
\]

gegeben. Bei variabler Normalen-/Flussquadratur bezeichnet der letzte Ausdruck die entsprechenden tatsächlichen Randlasten. Das neue Ziel ist für **jedes** solche \(a\)

\[
M_L\dot a^H=R_C^N(a)+Qw,\qquad
Z_i=(Qw)_i=\sum_{j\ne i}m_{ij}(w_i-w_j).
\tag{M2}
\]

Die folgenden Erhaltungs- und Rekonstruktionsidentitäten sind endliche algebraische Aussagen unter diesen Voraussetzungen. Das Fourierresultat gilt ausschließlich für uniforme periodische eindimensionale P1-Gitter, konstante skalare Advektion und konstante Graphgeschwindigkeit. Die unten behandelte Randvariante C gilt zunächst ausschließlich auf regulären kartesischen Gittern mit mindestens drei Zellen pro aktiver Koordinatenrichtung. Aussagen zu einem allgemeinen zeitabhängigen, begrenzten M1PN-System folgen daraus nicht.

Das Prinzip der Low-Order-Zeitableitung in der Massenlumpingkorrektur ist bekannt: Kuzmin–Hajduk, (3.137)–(3.138), und Moujaes–Kuzmin–Bäumer, Preprint v1, (29)–(32). Die Anwendung auf den hier angegebenen PN-/M1-Randledger ist eine zu prüfende Übertragung; ihre Neuheit ist nicht nachgewiesen. Die genaue Quellen-/Versionsprüfung steht in [LITERATURE.md](LITERATURE.md).

## 2. Quellenbasis und Verhältnis zum konsistenten Ziel

Die zugängliche Datei `M1PN_Paper_R2_20261029.tex` definiert die FE-Matrizen in (3.3)–(3.6), die beiden LLF-Randflüsse in (3.8a)–(3.9), das konsistente PN-Ziel in (3.10), die M1-Referenz in (3.14), die Randkorrektur in (5.1) und die vollständige bisherige Identität in (5.4)–(5.7). Diese manuell gesetzten Gleichungsnummern wurden direkt eingesehen. Die im Auftrag bezeichnete PDF vom 2. November wurde nicht gefunden; ihre abweichenden Nummern werden hier nicht als geprüft zitiert.

Das ungestörte konsistente Ziel \(v_C\) erfülle \(M_Cv_C=R_C^N\). Dann gilt exakt

\[
\dot a^H-v_C=M_L^{-1}Q(w-v_C).
\tag{M3}
\]

Die neue Methode ist somit keine identische Auswertung des konsistenten PN-Verfahrens. Sie ersetzt dessen Zeitableitung innerhalb einer Massenkorrektur durch einen numerischen Prädiktor. Gleichung (M2) benötigt nur die Diagonalinversion von \(M_L\), keinen konsistenten Massensolve und keine Biharmonik. Die alte Formel (5.6) darf nicht zusätzlich mit \(m_{ij}\Pi_1(\dot a_i^H-\dot a_j^H)\) und einem zweiten Stabilisierungsfluss verwendet werden.

## 3. Erhaltung und exakte vollständige Target-Rekonstruktion

Da \(Q\mathbf1=0\) und \(Q=Q^T\), gilt komponentenweise für jedes Prädiktorfeld \(w\)

\[
\sum_iZ_i=\mathbf1^TQw=0.
\tag{M4}
\]

Alternativ folgt dies durch paarweise Aufhebung von \(z_{ij}=m_{ij}(w_i-w_j)=-z_{ji}\). Für die Gesamtbilanz ist keine konservative Prädiktorauswertung erforderlich. Eine behauptete physikalische Bilanz des Prädiktors selbst ist davon getrennt zu prüfen.

Schreibe \(F_i=f(u_i)\), \(T_i=\Pi_1\mathcal A(a_i)-F_i\), \(D_i=\sum_{j\ne i}d^{M1}_{ij}(u_j-u_i)\). Die tatsächliche M1-Referenz lässt sich mit den FE-Identitäten schreiben als

\[
L_i^{M1}=D_i-\sum_jc_{ij}\cdot F_j
+\left(\sum_jB_{ij}\right)\cdot F_i-\sum_FH^{M1}_{Fi}.
\tag{M5}
\]

Subtraktion von (M5) von der projizierten Gleichung (M2) gibt

\[
\begin{aligned}
m_i\Pi_1\dot a_i^H-L_i^{M1}
={}&\sum_jc_{ji}\cdot T_j
+\sum_jB_{ij}\cdot(F_j-F_i)-D_i\\
&+\sum_{j\ne i}m_{ij}\Pi_1(w_i-w_j)
+\sum_F(H^{M1}_{Fi}-\Pi_1H^N_{Fi}).
\end{aligned}
\tag{M6}
\]

Weil \(\sum_jc_{ij}=0\), ist

\[
\sum_jc_{ji}\cdot T_j=
\sum_{j\ne i}(c_{ji}\cdot T_j-c_{ij}\cdot T_i).
\]

Damit folgt genau die angefragte Identität mit

\[
\begin{aligned}
f^{MC}_{ij}={}&m_{ij}\Pi_1(w_i-w_j)+d^{M1}_{ij}(u_i-u_j)\\
&+c_{ji}\cdot T_j-c_{ij}\cdot T_i
+B_{ij}\cdot(F_j-F_i),\\
f_{iF}={}&H^{M1}_{Fi}-\Pi_1H^N_{Fi}.
\end{aligned}
\tag{M7}
\]

Jeder innere Term in (M7) ist bei Vertauschung von \(i,j\) antisymmetrisch. Insbesondere ist der \(B_{ij}\)-Term kein zusätzlicher physikalischer Fluss, sondern Teil der FE-Rekonstruktion. Bei Quellen/Reaktionen ist rechts zusätzlich \(\Pi_1r_i^N-r_i^{M1}\) erforderlich, sofern diese Residuen nicht nachweislich identisch projiziert sind.

Die Summation von (M2) liefert im quellenfreien Fall \(\sum_im_i\dot a_i^H=-\sum_{Fi}H^N_{Fi}\). Nach nichttrivialer Begrenzung besitzt das gekoppelte System dagegen die Bilanz der tatsächlich akzeptierten Randkorrekturen. Diese ist nicht automatisch die unbegrenzte PN-Randbilanz.

## 4. Periodischer Fourierfall und SSP-RK3

Für \(a_t+ca_x=0\), uniformes periodisches P1-Gitter mit Schrittweite \(h\), \(\lambda\geq|c|\) und \(d_{i,i\pm1}=\lambda/2\) besitzt der Modus \(a_i=\widehat a\exp(\mathrm i i\theta)\) die Symbole

\[
\widehat M_C=h(2+\cos\theta)/3,\quad
\widehat M_L=h,\quad \widehat Q=h(1-\cos\theta)/3,
\]

\[
\widehat R_C=-\mathrm i c\sin\theta,\qquad
\widehat w=\left[-\frac{\mathrm i c}{h}\sin\theta
-\frac\lambda h(1-\cos\theta)\right]\widehat a.
\]

Einsetzen in (M2) ergibt

\[
\sigma(\theta)=-\frac{\mathrm ic}{h}\sin\theta
\left(1+\frac{1-\cos\theta}{3}\right)
-\frac\lambda{3h}(1-\cos\theta)^2.
\tag{M8}
\]

Hier ist der reelle Anteil nichtpositiv. Für feste glatte Wellenzahl \(k\), \(\theta=kh\), ergibt sich formal durch Taylorentwicklung die führende künstliche Dissipation

\[
-\frac{\lambda h^3}{12}\partial_x^4a.
\tag{M9}
\]

Der Vergleich mit \(\beta=\gamma\lambda h^3\) liefert damit in genau diesem Lumping-/Fourierfall \(\gamma_{\mathrm{eff}}=1/12\). Eine konsistente Massenauswertung der Vergleichsbiharmonik verändert deren hochfrequente Dämpfung. Ebenso folgt aus (M9) weder eine mehrdimensionale isotrope Äquivalenz noch ein Satz auf nichtuniformen Gittern.

SSP-RK3 besitzt das Stabilitätspolynom \(P(z)=1+z+z^2/2+z^3/6\). Die exakte modale Bedingung dieses periodischen Skalarproblems lautet

\[
\max_{\theta\in\Theta_h}|P(\Delta t\,\sigma(\theta))|\leq1.
\tag{M10}
\]

Am Nyquistmodus ist \(\sigma(\pi)=-4\lambda/(3h)\). Mit der negativen reellen Stabilitätsradiuszahl \(r_3\), definiert durch \(P(-r_3)=-1\), ist daher \(\lambda\Delta t/h\leq3r_3/4\) notwendig, aber bei gleichzeitigem Imaginärteil nicht allgemein hinreichend. Für offene oder nichtnormale PN-Systeme genügt eine bloße Eigenwertprüfung nicht als Kontraktivitätssatz. Außerdem deckt die M1-Low-Order-CFL nicht automatisch die höheren PN-Moden ab. Die SSP-Eigenschaft überträgt nur eine tatsächlich vorhandene Vorwärts-Euler-Eigenschaft.

## 5. Entscheidende Gegenbeispiele

### Gegenbeispiel 1: affiner offener Rand, Standardvariante

Seien \(c>0\), \(\lambda=c\), mindestens zwei uniforme P1-Elemente auf einem Intervall, \(a_i=1+x_i\), exakt kompatible Einströmdaten links und physikalischer Upwind-Ausfluss rechts gegeben. Die auswärts orientierte LLF-Last links ist \(H_0=-cg\); rechts ist \(H_n=ca_n\). Die lumpierten Endpunktmassen sind \(h/2\), alle inneren Massen \(h\), und \(m_{i,i+1}=h/6\).

Aus (M1) folgt

\[
w_0=0,\qquad w_i=-c\ (1\leq i\leq n-1),\qquad w_n=-2c.
\]

Weil \(R_C\) auf diesem affinen, kompatiblen Feld genau \(-cm_i\) ergibt, gilt links

\[
\frac{Z_0}{m_0}=\frac{(h/6)(0+c)}{h/2}=\frac c3,
\qquad \dot a_0^H=-\frac{2c}{3}.
\tag{M11}
\]

Rechts ergibt sich \(Z_n/m_n=-c/3\) und \(\dot a_n^H=-4c/3\). Für mindestens drei Elemente haben die ersten inneren Nachbarn zusätzlich die Korrekturraten \(-c/6\) bzw. \(+c/6\). Für genau zwei Elemente heben sich diese beiden Effekte am einzigen inneren Knoten auf. Das Ein-Element-Gitter hat andere Zahlen und darf nicht als Prüfung von (M11) eingesetzt werden.

Damit ist eine gitterweite affine Exaktheit der Standardmassenkorrektur widerlegt. Die globale Erhaltung bleibt trotzdem gültig. Ein lokaler O(1)-Randratenfehler allein ist noch kein Beweis gegen jede globale Konvergenzrate des vollständigen zeitabhängigen Verfahrens.

### Gegenbeispiel 2: positive künstliche Arbeit bei schwachem offenen Rand

Ein P1-Element \([0,h]\), \(c=\lambda>0\), homogene schwache Einströmdaten \(g=0\), Upwind-Ausfluss, beliebige innere FE-Spur. Dann sind

\[
M_L=\frac h2I,\quad Q=\frac h6\begin{pmatrix}1&-1\\-1&1\end{pmatrix},
\quad w=\frac{2c}h\begin{pmatrix}-a_0\\a_0-a_1\end{pmatrix},
\]

\[
Z=\frac c3\begin{pmatrix}-2a_0+a_1\\2a_0-a_1\end{pmatrix}.
\]

Für \(a=(1,3/2)^T\) gilt \(Z=(-c/6,c/6)^T\), also

\[
\mathbf1^TZ=0,\qquad a^TZ=c/12>0.
\tag{M12}
\]

Der größte Eigenwert des symmetrischen Anteils des Z-Operators ist \(c(\sqrt{10}-3)/6>0\). Die innere Spur stimmt in diesem Beispiel nicht mit dem schwachen Einströmzustand überein; genau dies ist bei einer schwachen Randbehandlung zulässig. Das Beispiel beweist keine Instabilität des vollständigen Targets, widerlegt aber allgemeine negative quadratische Arbeit der künstlichen Massenkorrektur mit offenen Rändern.

### Gegenbeispiel 3: positive Arbeit ohne jeden physikalischen Rand

Betrachte vier periodische P1-Elemente mit abwechselnden Längen \(H,e,H,e\), \(H,e>0\), und \(c=\lambda=1\). Alle Lumpenmassen sind \(m=(H+e)/2\); die üblichen Graphkoeffizienten sind \(d=1/2\). Für \(a=(x,y,-x,-y)^T\) liefert direkte Assembly

\[
a^TZ=\frac{2}{3m}\{-Hx^2-ey^2+(H-e)xy\}.
\tag{M13}
\]

Bei \(H=9,e=1,x=1,y=3\) ist \(a^TZ=4/5>0\). Die gesamte Geometrie kann gleichmäßig skaliert werden, ohne dieses Vorzeichen zu verändern. Die Quadratik ist indefinit, sobald \(H/e>3+2\sqrt2\) oder umgekehrt. Damit ist die Verallgemeinerung des uniformen periodischen Dissipationsarguments auf beliebige P1-Gitter bereits in einer Raumdimension falsch.

Hier ist der periodische Galerkin-Transport schiefsymmetrisch und hat keine quadratische Arbeit. Daher wächst an diesem Zustand sogar die lumpierte Energie des vollständigen unbegrenzten Targets mit Rate \(4/5\). Das widerlegt eine allgemeine Energie-Kontraktivitätsbehauptung; ein einzelner positiver momentaner Energieanstieg beweist noch keine spektrale Langzeitinstabilität.

Ein möglicher zusätzlicher Gegencheck betrifft nichtidentisch diskretisierte Quellen: Wird \(\Pi_1r^N-r^{M1}\) aus (M6) weggelassen, ist die behauptete vollständige Target-Identität im Allgemeinen falsch.

## 6. Warum das Fourierargument nicht global genügt

Auf dem uniformen periodischen Skalargitter kommutieren die symmetrische Massendifferenz, der konstante Graphlaplaceoperator und der schiefsymmetrische Transport. Daher verschwindet die künstliche Transportarbeit und das Produkt der kommutierenden positiven Faktoren liefert die nichtpositive Dämpfung in (M8).

Auf allgemeinen Gittern ist \(QM_L^{-1}L_{PN}\) keine negative Gram-Matrix. Selbst positive semidefinite Einzelfaktoren machen den symmetrischen Anteil eines nichtkommutierenden Produkts nicht automatisch positiv semidefinit. (M13) zeigt konkret, dass diese Lücke tatsächlich relevant ist. Dies ist der schwächste Schritt eines versuchten allgemeinen Dissipationsbeweises.

Eine isolierte Biharmonik \(Z^{BH}=-L^TDL a\) mit \(D\succeq0\) erfüllt hingegen exakt \(a^TZ^{BH}=-\|D^{1/2}La\|^2\leq0\). Diese Aussage setzt die tatsächlich implementierte vollständige Adjungierte und Gewichtung voraus. Sie ist weder ein Beweis der Randkonsistenz dieser Biharmonik noch ein Energiesatz für Synchronisierung und aktive Kegelbegrenzung.

Die Synchronisierung \(k=Q_1\dot a^H+T_1\ell\) besitzt \(\Pi_1k=\ell\), sofern \(\Pi_1Q_1=0\) und \(\Pi_1T_1=I\). Auch diese algebraische Projektionsidentität ist kein Vorzeichennachweis der zusätzlichen Energiearbeit. Für die Realisierbarkeit sind die tatsächlich begrenzten gerichteten Bar-States, symmetrischen Paarfaktoren und die Stufen-CFL entscheidend; vgl. Moujaes et al., Theorem 1 und (33)–(34), sowie das lokale Manuskript (3.15)–(3.20).

## 7. Getrennte explorative Randvariante C

Erst nach vollständiger Implementierung und unabhängiger Prüfung der Standardvariante darf eine Randvariante als eigener Kandidat verglichen werden. Eine besonders einfache lokale Kandidatendefinition auf uniformen kartesischen Gittern ist:

1. Berechne überall den unveränderten Standardprädiktor \(w^L\).
2. Für Randknoten mit ganzzahligen Gitterindizes \(i_r\in\{0,\ldots,n_r\}\) setze \(\kappa_r(i)=\min(\max(i_r,1),n_r-1)\).
3. Setze \(w_i^\sharp=w^L_{\kappa(i)}\) am Rand und \(w_i^\sharp=w_i^L\) im Inneren.
4. Verwende genau \(Z^\sharp=Qw^\sharp\) im unveränderten physikalischen Residual (M2) und in (M7).

Dies erzeugt keine Zustands-Ghosts und verändert keine physikalischen Randlasten. Auf P1 in 1D sowie Q1 auf kartesischen Tensorprodukten liegt der kopierte Knoten in einer vorhandenen angrenzenden Zelle; die Massenkorrektur benutzt weiter nur die ursprünglichen Massenkanten. Eine Bezeichnung als identischer Standardprädiktor wäre dennoch falsch.

**Bedingte affine Aussage, Beweisversuch:** Für jede kompatible affine Transportlösung \(p(x)=a_*+Gx\), konstante PN-Matrizen, symmetrischen uniformen kartesischen Graph und exakte affine Quadratur ist der Standardprädiktor an jedem vollständig inneren Knoten gleich der konstanten analytischen Ableitung \(-\sum_rA_rG_r\). Die gepaarten inneren Graphbeiträge verschwinden auf affinen Feldern und der Galerkinoperator reproduziert den konstanten Gradienten. Das Kopieren dieses konstanten Wertes liefert \(w^\sharp\) konstant auf allen Knoten. Somit \(Qw^\sharp=0\), auch an Kanten und Ecken. Dies beweist unter den genannten Voraussetzungen genau die affine Annihilation der künstlichen Korrektur.

Bei glatten Daten ist formal eine O(h)-Approximation der Randableitung plausibel: Der innere Low-Order-Prädiktor besitzt dort O(h)-Konsistenz, und der Kopierabstand ist O(h). Da die Zeilennorm von \(M_L^{-1}Q\) auf regulären Gittern beschränkt bleibt, folgt zunächst nur ein O(h)-Ratenargument in der Randzone. Für dieses Argument werden gleichmäßige Glattheit und Konsistenzkonstanten vorausgesetzt; ein vollständiger Evolutionsfehlerbeweis fehlt. In 1D ist die äußerste Korrektur sogar exakt null, weil \(w_0^\sharp=w_1^\sharp\), während am ersten inneren Knoten eine O(h)-Randzonenkorrektur verbleiben kann.

**Grenzen:** Der kopierte Prädiktor ist im Allgemeinen selbst nicht physikalisch konservativ; die endgültige künstliche Massenkorrektur bleibt wegen (M4) konservativ. Es gibt keinen allgemeinen Energie-/Dissipativitätsnachweis. Der affine Beweis gilt nicht automatisch für verzerrte Gitter, gekrümmte Geometrie, inkompatible charakteristische Außendaten, variable Koeffizienten oder beliebig diskretisierte Reaktionen/Quellen. Dimensionen mit weniger als drei Zellen sind aus dieser Definition des Untersuchungskandidaten auszuschließen. Die Änderung ist eine gezielt deklarierte Randdiskretisierung, kein Nachweis einer universell besseren Methode.

Ein alternativ denkbarer affiner Gradientenabzug an den Graphkanten würde zusätzliche Rekonstruktionen und eine erneute Dissipationsanalyse benötigen. Wird er überall eingesetzt, kann er bereits die führende periodische Innendämpfung von vierter auf sechste Ordnung verändern. Er ist daher keine automatisch äquivalente oder bereits geprüfte Randreparatur.

## 8. Unabhängige Prüfschritte und offene Aussagen

Die folgenden Prüfungen sind zur Bestätigung dieser Herleitung geeignet, werden durch das Vorliegen dieses Textes aber nicht als ausgeführt erklärt:

- Direkte FE-Assembly von \(M_C,M_L,C,B\), Kontrolle jeder Zeile und jedes Randvorzeichens; Rekonstruktion von (M6) komponentenweise bei unabhängigen äußeren M1-/PN-Zuständen.
- Unabhängige Reproduktion von (M8) aus der assemblierten periodischen Matrix statt alleiniger Auswertung derselben Formel.
- Direkte Auswertung von (M11) auf mindestens zwei, besser mehreren Elementen und Vergleich mit der analytischen Ableitung \(-c\).
- Direkte Assembly der Gegenbeispiele (M12), (M13), separate Prüfung von \(\mathbf1^TZ=0\), \(a^TZ\), symmetrischem Anteil und vollständiger Targetmatrix.
- Konstanten, affine, quadratische und glatte Felder mit bekannten Ableitungen auf einer Gitterfolge; Normen von \(M_L^{-1}Z\) getrennt für Innen-, Flächen-, Kanten- und Eckknoten sowie Fehler der vollständigen Ableitung.
- Standardvariante und C getrennt, mit denselben physikalischen Randflüssen, Zeitschritten und Referenzdaten. Bei C separat die Erhaltung des endgültigen Targets und die möglicherweise fehlende Bilanz des Hilfsprädiktors dokumentieren.

Ein unabhängiger Auditor sollte insbesondere die Konventionen für \(C\), die Randquadratur und die Gleichungsnummern gegenprüfen. Keine hier vorgestellte Skalarrechnung ersetzt den vollständigen PN-Spektraltest, den MCL-Realisierbarkeitstest, die SSP-RK3-Stufenprüfung oder einen Crossing-Beams-Vergleich. Die stärkste momentan vertretbare mathematische Aussage ist die bedingte algebraische Rekonstruktion und Erhaltung; globale quadratische Dissipativität ist durch die expliziten Gegenbeispiele widerlegt. Eine Entscheidung zwischen A, B und C benötigt zusätzlich die tatsächlich ausgeführten Genauigkeits-, Rand- und Performanceexperimente.
