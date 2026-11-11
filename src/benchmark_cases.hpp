#pragma once
// Material sets of the heterogeneous benchmarks, shared by path B and element WENO (docs/ELEMENT_HETEROGENEOUS.md).
// Only point predicates live here; quadrature, nodal averaging and units belong to the solvers.
#include "common.hpp"
#include <cmath>
namespace m1pn::cases {
// --- 2D lattice, Moujaes/Kuzmin/Baeumer (arXiv:2509.07689v1, Sec. 5.4, (44)-(47)) on D = (0,7)^2 -----------------
// D_a = ([1,2] u [5,6]) x ([1,2] u [3,4] u [5,6]) u ([2,3] u [4,5]) x ([2,3] u [4,5]) u [3,4] x [1,2]: eleven unit
// cells, given by the cell index (floor x, floor y). Identical to Schotthoefer/Hauck (arXiv:2505.17284, (12a))
// after the shift by 3.5. |D_a| = 11, so int sigma_a = 110 and int sigma_s = 38 for sigma_a = 10, sigma_s = 1.
inline bool lattice_paul_absorber_cell(int ix, int iy) {
    return ((ix == 1 || ix == 5) && (iy == 1 || iy == 3 || iy == 5)) || ((ix == 2 || ix == 4) && (iy == 2 || iy == 4)) ||
           (ix == 3 && iy == 1);
}
inline bool lattice_paul_absorber(double x, double y) {
    return lattice_paul_absorber_cell(int(std::floor(x)), int(std::floor(y)));
}
// (45): isotropic unit source on [3,4]^2 (closed square, as in path B).
inline bool lattice_paul_source(double x, double y) { return x >= 3 && x <= 4 && y >= 3 && y <= 4; }

// --- 3D lattice extension (own test case, not a literature benchmark) on D = (0,7)^3 -----------------------------
// Cells C_ijk = [i,i+1] x [j,j+1] x [k,k+1], i, j, k in {0,...,6}. Absorbing set
//   A = { (i,j,k) in {1,...,5}^3 : i + j + k even } \ {(4,2,2)}  u  {(3,3,1)},
// i.e. a 3D checkerboard around the central source cell (3,3,3) (odd sum, not absorbing) with one absorber removed
// and one added below the source, so that no coordinate reflection or permutation maps A onto itself.
// sigma_a = 10 on A, else 0. sigma_s = 0 on A, 5 on the non-absorbing cells of the layer k = 5, 1 elsewhere.
// Source q0 = 1 on C_333. Volumes and integrals are counted independently in lattice3d_counts().
inline bool lattice3d_absorber_cell(int i, int j, int k) {
    if (i == 4 && j == 2 && k == 2)
        return false;
    if (i == 3 && j == 3 && k == 1)
        return true;
    return i >= 1 && i <= 5 && j >= 1 && j <= 5 && k >= 1 && k <= 5 && (i + j + k) % 2 == 0;
}
inline double lattice3d_sigma_a(const V3 &x) {
    return lattice3d_absorber_cell(int(std::floor(x[0])), int(std::floor(x[1])), int(std::floor(x[2]))) ? 10. : 0.;
}
inline double lattice3d_sigma_s(const V3 &x) {
    const int i = int(std::floor(x[0])), j = int(std::floor(x[1])), k = int(std::floor(x[2]));
    if (lattice3d_absorber_cell(i, j, k))
        return 0.;
    return k == 5 ? 5. : 1.;
}
inline double lattice3d_source(const V3 &x) {
    return x[0] >= 3 && x[0] <= 4 && x[1] >= 3 && x[1] <= 4 && x[2] >= 3 && x[2] <= 4 ? 1. : 0.;
}
struct Lattice3dCounts {
    int absorbing = 0, strong_scattering = 0, scattering = 0;
    double volume = 343, sigma_a = 0, sigma_s = 0, source = 1;
};
inline Lattice3dCounts lattice3d_counts() {
    Lattice3dCounts c;
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            for (int k = 0; k < 7; ++k) {
                if (lattice3d_absorber_cell(i, j, k))
                    ++c.absorbing;
                else if (k == 5)
                    ++c.strong_scattering;
                else
                    ++c.scattering;
            }
    c.sigma_a = 10. * c.absorbing;
    c.sigma_s = 5. * c.strong_scattering + 1. * c.scattering;
    return c;
}

// --- 3D shadow, Weih et al. (MNRAS 498, 3374, 2020; arXiv:2007.05718, Sec. 4.2.1) -----------------------------------
// D = (-0.75, 0.75) x (-0.25, 0.25)^2, absorbing sphere of radius R = 0.1 at the origin (published: kappa_a = 1e7),
// beam injected through x = -0.75 along +x. The beam profile is not specified in the paper; here (own choice,
// documented): a perfectly collimated beam along +x with density profile rho_b(r), r = |(y, z)|, equal to 1 for
// r <= 0.15, cos^2 taper to 0 at r = 0.2 and 0 beyond. It covers the sphere (R = 0.1) and stays away from the side
// faces y, z = +-0.25: a plane wave over the whole face runs tangentially along them and is drained there by the
// identity part of the LLF boundary flux (found in the first smoke run).
constexpr double shadow_radius = .1, shadow_x0 = -.75, shadow_beam_radius = .15, shadow_beam_taper = .05;
inline double shadow_beam_profile(double y, double z) {
    const double r = std::sqrt(y * y + z * z);
    if (r <= shadow_beam_radius)
        return 1;
    if (r >= shadow_beam_radius + shadow_beam_taper)
        return 0;
    const double c = std::cos(.5 * pi * (r - shadow_beam_radius) / shadow_beam_taper);
    return c * c;
}
inline bool shadow_inside(const V3 &x) { return x.squaredNorm() < shadow_radius * shadow_radius; }
// Length of the segment {(s, y, z) : x0 < s < x} inside the sphere (Beer-Lambert optical depth / sigma_a).
inline double shadow_chord(double x, double y, double z) {
    const double r2 = y * y + z * z, R2 = shadow_radius * shadow_radius;
    if (r2 >= R2)
        return 0;
    const double half = std::sqrt(R2 - r2), lo = -half, hi = half;
    const double a = std::max(lo, shadow_x0), b = std::min(hi, x);
    return std::max(0., b - a);
}
// Uncollided kinetic density of the collimated beam for pure absorption sigma inside the sphere, peak density rho_b:
// rho = rho_b profile(y, z) exp(-sigma chord) behind the front x < x0 + t, 0 ahead (background not included).
inline double shadow_beer_lambert(const V3 &p, double t, double sigma, double rho_b) {
    if (p[0] > shadow_x0 + t)
        return 0;
    return rho_b * shadow_beam_profile(p[1], p[2]) * std::exp(-sigma * shadow_chord(p[0], p[1], p[2]));
}
} // namespace m1pn::cases
