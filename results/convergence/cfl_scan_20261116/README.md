# CFL-Scan Element-WENO (NX, 16.11.2026)

Grundlage für den CFL-Default von `m1pn_fast` (Element-WENO, SSPRK(4,3)): CFL 1.0. Auswertung in
[docs/ELEMENT_WENO.md](../../../docs/ELEMENT_WENO.md), Abschnitt „CFL-Default von m1pn_fast“.

Jobs 160896/160902, ein Sockel (32 Kerne, EPYC 9354) je Lauf, `m1pn_fast` aus `0cead11` plus die (numerisch
neutralen) Default-Änderungen; SHA256 in `binary.sha256`. Alle Läufe mit expliziten Flags
`--stabilization element-weno --time-integrator ssprk43 --dt 0 --cfl C`.

| Fall | Flags |
|---|---|
| crossQ1 / Q2 / Q3 / Q4 | `crossing-study`, P7, T = 0.32, 128² Q1 / 64² Q2 / 43² Q3 / 32² Q4 |
| smoothQ2 | `smooth` mit `smooth-transport`-Rand, 32² Q2, P7, T = 0.15 |
| lattice | `lattice-paul`, 112² Q1, P7, T = 3.2 |
| lattice3d | `lattice-3d`, 28³ Q1, P3, T = 1.0 |
| linefcs | `line-source` mit `--fcs analytic`, σ_s = 3, 128² Q1, P7, T = 0.3 |

`results.csv`: je Fall und CFL (0.1, 0.2, 0.35, 0.5, 0.75, 0.9, 1.0) Schritte, Lösungszeit, Abweichung der Dichte
gegen CFL 0.1 (L2 und max, relativ), minimaler Kegelabstand, limitierte Elementstufen, genudgte Zustände, Masse,
`dt / high_order_step` und `dt / realizability_step`. Die Spalte `final_density_mass` von lattice3d ist nicht mit der
2D-Literaturreferenz vergleichbar.
