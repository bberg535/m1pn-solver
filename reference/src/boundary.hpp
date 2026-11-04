#pragma once
#include "angular.hpp"
#include <Eigen/Cholesky>
#include <functional>
#include <memory>
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
};
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
    double llf_norm = 0;
    std::vector<AngularPoint> rule;
    // data() consumes one angular quadrature row at a time.
    Field odd_basis;
    double norm;
    bool c0;
    Boundary(const Angular &, const CanonicalC1 &, const V3 &, bool c0 = false);
    Incoming data(const Inflow &, const V3 &x, double time) const;
    // Full PN exterior for the LLF flux: Onsager incoming modes, interior
    // outgoing modes, and interior zero modes (equation (6.9)).
    Vec exterior(const Vec &interior, const Vec &d) const;
    Vec flux(const Vec &u, const Vec &d) const;
};
} // namespace m1pn
