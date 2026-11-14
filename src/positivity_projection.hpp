#pragma once
// Moment-preserving positivity projection of a nodal PN angular trace (docs/POSITIVITY_PROJECTION.md):
//
//   min_z 1/2 sum_k w_k (z_k - y_k)^2   s.t.  z_k >= 0,  sum_k w_k m_k z_k = U,   m_k = (1, Omega_k).
//
// KKT: z_k = max(0, y_k - xi . m_k) with four multipliers xi solving F(xi) = sum_k w_k m_k z_k - U = 0.
// F is the gradient of the concave dual phi(xi) = -1/2 sum_k w_k z_k(xi)^2 - xi . U (up to a constant), so a
// semismooth Newton step d = H^-1 F, H = sum_{k active} w_k m_k m_k^T, with Armijo backtracking on phi converges
// globally whenever the problem is feasible (U in the interior of cone{m_k}). z >= 0 and complementarity hold by
// construction; the only residual to verify is the moment residual |F|.
#include "common.hpp"
#include <algorithm>

namespace m1pn {

struct PositivityProjectionOptions {
    double tolerance = 1e-12; // |F|_inf <= tolerance * (sum_k w_k |y_k| + rho_U)
    int max_iterations = 60;
    int max_backtracks = 40;
    // false: an unsolvable problem returns unconverged instead of augmenting the rule by an atom at J/|J|.
    bool augment = true;
    // Optional per-point weights of the objective 1/2 sum w c (z - y)^2 (both arrays or none; cinv = 1 / c).
    const double *c = nullptr, *cinv = nullptr;
};

struct PositivityProjectionResult {
    bool converged = false;
    bool identity = false;  // y > 0 and moments already consistent: z = y (no Newton step taken)
    bool infeasible = false; // stopped by a Farkas certificate (U not in the cone of the rule)
    int iterations = 0;     // Newton steps taken
    int backtracks = 0;     // halvings of the step over all iterations
    int active = 0;         // points with z_k > 0 at the solution
    double residual = 0;    // |F|_inf / scale at the returned z
    double scale = 0;       // sum_k w_k |y_k| + rho_U
    Four xi = Four::Zero(); // multipliers
};

// Points in structure-of-arrays form (ox, oy, oz, w of length q). y is read, z is written (may alias y).
// xi0 is the initial multiplier (warm start); the identity z = y is tested first in any case.
PositivityProjectionResult project_positive(int q, const double *ox, const double *oy, const double *oz,
                                            const double *w, const double *y, const Four &U, double *z,
                                            const PositivityProjectionOptions &options = {},
                                            const Four &xi0 = Four::Zero());

// Full boundary-node projection with the solvability treatment of section 5 of the note:
//  * U outside R1 (rho < |J|) or not finite: failure (never clipped).
//  * Vacuum U = 0: z = 0.
//  * U in R1 but outside the interior of the discrete cone {sum w m z, z >= 0} of the rule (exact test with the hull
//    facets if given: J/rho in int conv{Omega_k}; otherwise the necessary certificate |J| > rho max_k Omega_k.J^),
//    or Newton without convergence (Farkas exit): the rule is augmented by
//    one point Omega* = J/|J| with the mean weight 4 pi / q and prior value y* = 0. Then U = |J| (1, J^) +
//    (rho - |J|)(1, 0) lies in the interior of the augmented cone for every rho > |J|; on the cone boundary
//    rho = |J| the unique nonnegative representation is the atom itself. The atom's moments are returned in
//    `atom` (z excludes it); the result is still a nonnegative, moment-preserving discrete angular measure.
// Warm start of a boundary node: multipliers of its previous plain and augmented projections.
struct PositivityProjectionWarm {
    Four xi = Four::Zero(), xi_augmented = Four::Zero();
};
// Facets {Omega : n_f . Omega <= h_f} of the convex hull of the rule directions (structure of arrays). With them the
// solvability test is exact: U = rho (1, u) lies in the interior of cone{m_k} iff max_f (n_f . u - h_f) < 0.
struct PositivityFacets {
    Vec nx, ny, nz, h;
    double inradius = 0; // min_f h_f: |u| < inradius is always inside (no facet scan needed)
    int count() const { return int(h.size()); }
    double margin(const V3 &u) const; // max_f (n_f . u - h_f)
};
// Hull facets of a latitude-longitude rule: rings z_i (any order) on a common azimuth grid phi_j = 2 pi j / A,
// rotated by `frame`. Verified against all points (every point satisfies all facets up to 1e-13); returns an empty
// facet set if the verification fails.
PositivityFacets lat_long_facets(const std::vector<double> &ring_z, int azimuth, const Eigen::Matrix3d &frame);
struct PositivityProjectionOutcome {
    PositivityProjectionResult result; // of the accepted solve
    bool augmented = false, dirac = false, certificate = false, farkas = false;
    double hull_margin = std::numeric_limits<double>::quiet_NaN(); // max_f (n_f . J/rho - h_f), if facets given
    int newton_iterations = 0;         // over all solves (plain + augmented)
    V3 atom_direction = V3::Zero();
    Four atom = Four::Zero();          // w* z* (1, Omega*)
};
PositivityProjectionOutcome project_positive_moments(int q, const double *ox, const double *oy, const double *oz,
                                                     const double *w, const double *y, const Four &U, double *z,
                                                     const PositivityProjectionOptions &options = {},
                                                     const PositivityProjectionWarm *warm = nullptr,
                                                     const PositivityFacets *facets = nullptr);

// Statistics accumulated over all projected boundary nodes (merged by rate()).
struct PositivityProjectionStats {
    long calls = 0, identities = 0, failures = 0, vacuum = 0, scaled = 0;
    long capped = 0; // converged projections whose outgoing mass exceeded rho (loose tolerance near vacuum), capped
    long augmented = 0, certificates = 0, diracs = 0, farkas = 0;
    long iterations = 0, backtracks = 0;
    int max_iterations = 0;
    double max_residual = 0;      // max relative moment residual of accepted projections
    double min_active_fraction = 1;
    double seconds = 0;           // wall time of the ghost stage (trace products + projections)
    // Remaining add_boundary stages (diagnostics): face loop, per-side PN flux products, serial node reduction.
    double face_seconds = 0, product_seconds = 0, reduce_seconds = 0;
    // Diagnostics (projection_diagnostics): distance of the projected ghost to the C6 ghost, relative to rho.
    long c6_scaled = 0, hybrid_kept = 0;
    double c6_diff_sum = 0, c6_diff_max = 0, c6_scaled_diff = 0;
    std::array<double, 6> c6_diff_side{};
    void merge(const PositivityProjectionStats &o) {
        calls += o.calls, identities += o.identities, failures += o.failures, vacuum += o.vacuum, scaled += o.scaled;
        capped += o.capped;
        augmented += o.augmented, certificates += o.certificates, diracs += o.diracs, farkas += o.farkas;
        iterations += o.iterations, backtracks += o.backtracks;
        max_iterations = std::max(max_iterations, o.max_iterations);
        max_residual = std::max(max_residual, o.max_residual);
        min_active_fraction = std::min(min_active_fraction, o.min_active_fraction);
        seconds += o.seconds;
        face_seconds += o.face_seconds, product_seconds += o.product_seconds, reduce_seconds += o.reduce_seconds;
        c6_scaled += o.c6_scaled, hybrid_kept += o.hybrid_kept, c6_diff_sum += o.c6_diff_sum, c6_scaled_diff += o.c6_scaled_diff;
        c6_diff_max = std::max(c6_diff_max, o.c6_diff_max);
        for (int s = 0; s < 6; ++s)
            c6_diff_side[s] += o.c6_diff_side[s];
    }
};

} // namespace m1pn
