# 3D spatial comparison snapshot

This snapshot contains all **12 completed and validated spatial 3D runs** from the production campaign: grids `N=8,16,32`, each at PN orders `P=3,5,7,9`. Every run is read back through `nx/scripts/analysis.py` and its recorded verification/input hashes are checked by the plotting script. All use final time `t=0.2`, `dt=0.0001724137931034483`, and 1160 steps. Campaign manifest SHA-256: `853525f3c904f9514d1c23e1556b87a3f1a9952c4599d1a8e8c4ff39b5abd64e`.

## Figures and data

- [Density error against PN order and grid](convergence.png) ([PDF](convergence.pdf))
- [Final density slices at z=0.5](density_slices.png) ([PDF](density_slices.pdf))
- [Density lineouts](lineouts.png) ([PDF](lineouts.pdf)); extracted values: [lineouts.csv](lineouts.csv)
- [Per-run metrics](metrics.csv)
- [Provenance, verified input hashes, and norm/reference metadata](provenance.json)

## Norms and interpretation

`density_L1` and `density_L2` are errors against the analytic kinetic free-transport density, computed with lumped nodal-mass weights and normalized by the unit domain volume. `density_Linf` is the nodal maximum error. The 3D slice plots use `z=0.5`.

This analytic kinetic reference is not an exact solution of the finite PN discretization: reported errors therefore include PN model, boundary, limiter, and spatial errors. A separate temporal-reference check is unavailable, so this is a preliminary spatial snapshot, not a pure spatial-error isolation. The order trend is not uniformly monotone: at `N=8`, the density L2 error ranges from 0.1118 (`P=3`) to 0.1166 (`P=9`), while at `N=32`, `P=7` gives 0.00897 and `P=9` 0.00906. Avoid interpreting this dataset as proof of monotone PN-order convergence.
