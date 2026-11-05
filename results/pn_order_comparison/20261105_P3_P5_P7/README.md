# PN-only standard-B comparison: P3, P5, P7

These are three existing, independently verified runs at `t=0.32`, `dt=6.459426725878078e-5`, and 4954 steps. The native grids are P3 on 128², P5 on 64², and P7 on 128². Each run was re-read with `verify_pn_run.check`, including finite-field, all-Q conservation, and required-input hash checks. No solver runs were started for this comparison.

## Artifacts

- [Density and signed-error maps (PNG)](pn_orders_density_errors.png) · [PDF](pn_orders_density_errors.pdf)
- [Common-midline comparison (PNG)](pn_orders_midline.png) · [PDF](pn_orders_midline.pdf)
- [Native-grid weighted metrics](metrics.csv)
- [Midline samples, long form](midline.csv)
- [Run and plot provenance, including source hashes and color scales](provenance.json)

Density panels share a zero-centered red-white-blue scale whose limits include all negative densities and the largest reference or computed density. Error panels share a symmetric signed scale about zero; negative density values remain visible. Each weighted norm uses that run's own lumped nodal masses and unit-domain volume. Errors against the analytic kinetic free-transport reference include finite-PN model, boundary, and spatial errors.

P3 and P7 share the 128² grid, so that pair supports a direct order comparison at fixed grid and time controls. P5 is on 64²; its difference from either 128² run mixes order and grid effects and does not isolate PN-order convergence. The line plot samples each run on its native `y=0.5` nodes and evaluates the analytic reference on the finest available x grid.
