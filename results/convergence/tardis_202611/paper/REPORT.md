# NX Element-WENO convergence study

Completed 198 / 201 tasks; failures: 3.

Source: frozen manifest /home/warehouse15/bberg/phd/03_projects/M1PN_Solver/.claude/worktrees/gpu-campaign/nx/tmp/tardis_paper/manifest.json

## Interpretation
- Exact-PN smooth error uses the standalone L2-projected periodic test.
- Kinetic Crossing / boundary / line-source errors include PN truncation.
- Temporal rates use the finest fixed-grid time step as numerical reference.
- Spatial self differences compare each mesh to the next finer mesh, not the exact solution.
- SSPRK(4,3) transport is third order; Strang with noncommuting reactions is generally second order.
- The estimated high-order CFL is empirical, not a general stability theorem.
- Failed no-limiter ablations are diagnostic, not silently omitted.

## Files
- results.csv; results.json; failure details in status/*.json

## Failed or missing jobs

- crossing_P3_p1_E2048_std: solver returned 124
- crossing_P7_p1_E2048_std: solver returned 124
- ablation_P7_p2_E256_no_limit: solver returned 2
