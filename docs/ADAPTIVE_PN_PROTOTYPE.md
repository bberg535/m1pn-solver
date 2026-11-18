# Lokale PN-Ordnung mit kontinuierlichen gemeinsamen Momenten

Der Prototyp löst quellenfreien Transport auf periodischen kartesischen Tensor-Bernstein-Q1-Gittern in 1D, 2D und 3D. Er verwendet die vorhandenen reellen Kugelflächenharmonischen, die zugehörigen Streamingmatrizen, die Levermore-Schließung sowie die vorhandenen Kegelprädikate. Das Programm `m1pn_adaptive_pn` bietet das lineare PN-Ziel und einen M1-Kegel-Limiter mit einem gemeinsamen Faktor pro Element. Rechnungen erfolgen seriell auf der CPU.

## Literaturabgleich

- Dargaville et al. verwenden räumlich variable hierarchische Winkelräume in einer stationären Subgrid-Galerkin-Methode. Sie markieren auf dem CG-Gitter mit Winkelkoeffizienten oder mit primalen und adjungierten Residuen. Ihre Rechnungen zeigen sowohl Einsparungen als auch sichtbare Übergänge in der Winkellösung bei einem Wechsel von P1 auf höhere Ordnung. Die hier getestete elementweise Markierung mit der Norm des ersten Moments und die Kopplung nur der jeweils gemeinsamen Grade sind andere Diskretisierungsentscheidungen. [Dargaville et al., JCP 397, 108846, §§4–5 und §8.2](https://doi.org/10.1016/j.jcp.2019.108846) · [Volltext](https://arxiv.org/pdf/1903.05466)
- Schween und Reville leiten die Streamingmatrizen als Projektionen der Winkeloperatoren her und behandeln ihre Eigenstruktur für Finite-Volumen- und DG-Flüsse. Der Rusanov-Fluss dieses Prototyps verwendet dieselbe Operatorstruktur mit einer gemeinsamen Einbettung beider Elementspuren. [Schween und Reville, §§3, 5–6](https://arxiv.org/abs/2402.16483)
- Für SSPRK(4,3) gilt bei festem linearen dissipativen Operator starke Stabilität für hinreichend kleine Schritte in der Massennorm. Die im Programm benutzte Schrittweite ist eine Versuchseinstellung. Der Satz umfasst weder den nichtlinearen M1-Limiter noch variables WENO innerhalb eines Schritts. [Sun und Shu, Satz 3.4](https://arxiv.org/abs/1811.10680)
- Die M1-Barzustände und das Kegelargument folgen dem invariantendomänenerhaltenden CG-Ansatz. Die neue schwache Druckkorrektur wird in diesem Prototyp mit einem konservativen Elementfaktor begrenzt. Der separate Dichte-Clip-and-Scale-Schritt des Produktionssolvers gehört zu dessen eigener Limiter-Variante. [Moujaes, Kuzmin und Bäumer](https://arxiv.org/abs/2509.07689)

## Räumliche Kopplung und Ordnungswechsel

Ein Grad teilt seine Bernstein-Freiheitsgrade über eine Fläche, wenn er auf beiden Elementen vorhanden ist. Die Masse eines Grades wird aus genau seinen aktiven Elementen assembliert. Fehlende Spuren werden im größeren Winkelraum mit Nullen ergänzt. Mit den eingebetteten Spuren $x,y$ und der orientierten Streamingmatrix $A_n$ ist der gemeinsame Fluss

$$
f = \tfrac12 A_n(x+y)-\tfrac12(y-x).
$$

Das inklusive Vorzeichen projizierte Flusspaar trägt im schwachen Residuum exakt null zu gemeinsamen Testfreiheitsgraden bei. Für die zusätzlichen Grade bleibt der Flächenbeitrag erhalten. Das Testen mit der Lösung liefert zusammen mit den Volumenrandtermen

$$
\tfrac12 x^T A_n x-\tfrac12 y^T A_n y+(y-x)^T f=-\tfrac12\|y-x\|^2.
$$

Für den Limiter ist die Druckkorrektur $G^K$ elementlokal. Der Paarbeitrag
$c_{ji}^{K,d}G_{j,d}^K-c_{ij}^{K,d}G_{i,d}^K$ ist antisymmetrisch und summiert sich wegen $\sum_jc_{ij}^{K,d}=0$ zur schwachen Divergenzkorrektur. Die Barzustände und ihre Euler-CFL hängen nur vom vorhandenen niedrigen CG-Raum ab. Die Implementierung prüft den bin64-Kegel nach jeder limitierten SSPRK-Stufe.

Der Sensor verwendet die Quadratwurzel des konsistenten Elementintegrals von $\|\mathbf J\|^2/|K|$. Die Hysterese bewegt die Ordnung in Zweierschritten. Beim Wechsel projiziert die konsistente Masse die alte Lösung orthogonal in den neuen Raum. Die niedrigen Freiheitsgrade bleiben dabei bitgleich. Auch ein neu über einer Fläche geteilter Modus erhält die gemeinsame Spur durch diese Projektion. Im Versuch wird vor dem ersten und danach vor jedem fünften vollständigen Zeitschritt markiert.

## Reproduktion

```bash
cmake -S . -B tmp/adaptive -DCMAKE_BUILD_TYPE=Release -DENABLE_MPI=OFF
cmake --build tmp/adaptive --target m1pn_adaptive_pn adaptive_pn_tests element_weno_tests
ctest --test-dir tmp/adaptive -R '^(adaptive_pn|element_weno)$' --output-on-failure
tmp/adaptive/m1pn_adaptive_pn pulse 2 24 3 .20 pn
tmp/adaptive/m1pn_adaptive_pn pulse 2 24 3 .20 pn .002 .004
tmp/adaptive/m1pn_adaptive_pn pulse 2 24 7 .20 pn
tmp/adaptive/m1pn_adaptive_pn pulse 2 24 3 .20 m1
tmp/adaptive/m1pn_adaptive_pn pulse 3 10 3 .20 pn
tmp/adaptive/m1pn_adaptive_pn quadrupole 1 48 3 .20 pn
```

Der Versuch verwendet einen gerichteten periodischen Puls oder eine räumlich variable Quadrupolverteilung bei anfangs verschwindendem Strom. Die Referenzwerte entstehen durch freie Ausbreitung entlang Charakteristiken und Winkelquadratur. Die Fehler sind räumliche L²-Normen der Differenz zum Q1-Interpolanten dieser Referenzmomente. `mean_dofs` zählt die tatsächlich benutzten Zustandsfreiheitsgrade je Schritt. Der Zähler misst weder den Spitzenverbrauch noch den Aufwand der Massensolves.

| Versuch | Ordnung | Mittlere Freiheitsgrade | Dichtefehler | Stromfehler |
|---|---|---:|---:|---:|
| 2D-Puls, 24² | Festes P1 | 2 304 | 4.7925e-2 | 4.6605e-2 |
| 2D-Puls, 24² | Festes P3 | 9 216 | 4.7438e-3 | 5.8568e-3 |
| 2D-Puls, 24² | Adaptiv P1–P3, Schwellen .055/.10 | 3 066 | 1.9302e-2 | 1.8138e-2 |
| 2D-Puls, 24² | Adaptiv P1–P3, Schwellen .002/.004 | 4 128 | 4.9879e-3 | 5.9254e-3 |
| 2D-Puls, 24² | Festes P7 | 36 864 | 5.5754e-4 | 5.4662e-4 |
| 2D-Puls, 24² | Adaptiv P1–P7, Schwellen .055/.10 | 8 923 | 1.4048e-2 | 1.4036e-2 |
| 3D-Puls, 10³ | Festes P3 | 16 000 | 3.4533e-3 | 3.3006e-3 |
| 3D-Puls, 10³ | Adaptiv P1–P3, Schwellen .055/.10 | 4 771 | 7.5467e-3 | 8.0233e-3 |
| 1D-Quadrupol, 48 Zellen | Festes P3 | 768 | 1.2827e-5 | 5.9830e-5 |
| 1D-Quadrupol, 48 Zellen | Adaptiv P1–P3, Schwellen .055/.10 | 192 | 2.6524e-2 | 3.7328e-2 |

Auf den periodischen Strukturfällen erreicht der gleichordentliche Operator den vorhandenen unstabilisierten `ElementWeno`-Operator bis auf $1.9\times10^{-15}$. Das Residuum der gemischtordentlichen Energieidentität liegt unter $7\times10^{-17}$. Die ungelimitierten niedrigen Korrekturen reproduzieren das PN-Ziel im rauen 2D-Test bis auf $3.0\times10^{-15}$. Der Limiter greift dort beim ersten Aufruf auf 15 von 16 Elementen ein. Nach 20 Schritten beträgt der Drift der vier integrierten niedrigen Momente höchstens $6.7\times10^{-16}$. Im 2D-Puls greift die M1-Korrektur bei festem P1 ein. In den dortigen P3- und adaptiven Läufen ist sie inaktiv.

Beim Standard-Schwellenpaar spart das Verfahren im 2D-Puls rund 67 Prozent der Zustandsfreiheitsgrade, bei vierfachem Dichtefehler gegenüber festem P3. Mit engeren Schwellen bleiben rund 55 Prozent Einsparung bei ungefähr fünf Prozent Mehrfehler in der Dichte. Diese Schwellen wurden am selben Puls untersucht. Der Quadrupol zeigt die strukturelle Schwäche der Markierung anhand von $\|\mathbf J\|$. Sie entfernt die anfangs vorhandene anisotrope Information auch bei ansonsten stabiler Flächenkopplung.

Der Versuch behandelt periodische Flächen, Q1, seriellen CPU-Transport und den angegebenen M1-Limiter. WENO, offene C1-Ränder, FCS, Reaktionen, MPI und GPU liegen außerhalb des Versuchs. Die Energiebilanz mit aktivem Limiter und ein Laufzeit- oder Speichergewinn gegenüber `m1pn_fast` wurden nicht nachgewiesen. Die Massensolves des Prototyps verwenden sparse Faktorisierungen und werden beim Wechsel der Ordnung neu aufgebaut.
