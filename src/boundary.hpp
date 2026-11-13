#pragma once
#include "angular.hpp"
#include <Eigen/Cholesky>
#include <functional>
#include <memory>
#include <optional>
namespace m1pn {
struct Atom {
    V3 direction;
    double mass;
};
struct Inflow {
    std::function<double(const V3 &, double, const V3 &)> density;
    std::function<std::vector<Atom>(const V3 &, double)> atoms;
    // Optional admissible M1 exterior, supplied independently of the kinetic data.
    std::function<Four(const V3 &, double)> exterior_m1;
    std::function<Four(const V3 &, double, const V3 &, const Four &)> exterior_state;
    // Set when `density` does not depend on x, t or the direction. The angular
    // quadrature of the kinetic data can then be evaluated once per normal.
    std::optional<double> constant_density;
    // Set when the inflow data do not depend on time (vacuum, isotropic, plane-wave beams). Element WENO then
    // evaluates the kinetic boundary data once per face quadrature point instead of in every stage.
    bool steady = false;
};
// Copy the interior reduced velocity only at positive interior density.
// Positive exterior density at interior vacuum requires explicit velocity data.
Four copied_velocity_exterior(double density, const Four &interior,
                              const std::optional<V3> &vacuum_velocity = std::nullopt);
struct Incoming {
    Vec d;
    Four kinetic = Four::Zero();
    Four moments = Four::Zero(); // incoming half-range moments, distinct from flux
};
struct CanonicalC1 {
    int q, r, s;
    std::vector<int> odd, even;
    Mat B, L, U, V, Rplus, Rminus, S, reflection;
    Vec sigma, root;
    Eigen::LLT<Mat> solve;
    double contraction, condition;
    explicit CanonicalC1(const Angular &pn);
    Vec incoming(const Vec &outgoing, const Vec &d) const;
};
struct Boundary {
    const Angular &pn;
    const CanonicalC1 &canonical;
    V3 n;
    Eigen::Matrix3d frame;
    Mat D, Rplus, Rminus, homogeneous, llf_homogeneous;
    // LLF flux is affine in (trace, datum): flux = llf_homogeneous*u + llf_datum*d.
    Mat llf_datum;
    double llf_norm = 0;
    std::vector<AngularPoint> rule;
    // data() consumes one angular quadrature row at a time.
    Field odd_basis;
    double norm;
    bool c0;
    // norms = false skips the two full SVDs for `norm` and `llf_norm` (only path B uses them; O(q^3) per side).
    Boundary(const Angular &, const CanonicalC1 &, const V3 &, bool c0 = false, bool norms = true);
    Incoming data(const Inflow &, const V3 &x, double time) const;
    // Atom part of data(): adds the incoming atoms (same operations and order as data()) to `out`.
    void add_atoms(const std::vector<Atom> &atoms, Incoming &out) const;
    // Checks and outward density rounding at the end of data().
    void finish_data(Incoming &out) const;
    // Full PN exterior for the LLF flux: Onsager incoming modes, interior
    // outgoing modes, and interior zero modes (equation (6.9)).
    Vec exterior(const Vec &interior, const Vec &d) const;
    Vec flux(const Vec &u, const Vec &d) const;
};
} // namespace m1pn
