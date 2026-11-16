# Heterogeneous / 3D element-WENO campaign

Completed 12 / 18 tasks; failures: 6.

- Lattice deviations are against Schotthoefer/Hauck Table 1 / (2 pi) (kinetic, S22, different initial
  density 0 and inflow model); historical path-B NX runs used an older binary (not timed here).
- Shadow stage A compares with the uncollided Beer-Lambert density of the collimated beam (radius 0.15,
  cos^2 taper to 0.2; own choice, Weih et al. do not specify it); the PN/M1 angular model and the
  background 1e-6 are part of the deviation.
- lattice-3d is an own test case; no literature reference exists.

## Failed or missing

- lattice_P7_p1_E1792_std: missing status
- lattice_P15_p1_E1792_std: missing status
- lattice_P21_p1_E1344_std: missing status
- lattice_P35_p1_E896_std: missing status
- lattice_P45_p1_E224_std: solver returned 124
- lattice_P45_p1_E672_std: missing status
