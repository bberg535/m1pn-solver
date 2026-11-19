#pragma once
// Fast evaluation of nodal PN traces psi(Omega_k) = sum_c a_c Y_c(Omega_k) at the boundary hemisphere rules.
//
// The points are Omega_k = F omega_k with a tensor-product rule omega_(i,j) = (r_i cos phi_j, r_i sin phi_j, z_i)
// (sphere_rule(n, +-1); polar index i over all listed hemispheres, azimuth j) and an orthogonal face frame F.
// Spherical harmonics of degree l span an O(3)-invariant space, so Y_l(F omega) = D_l Y_l(omega) (real Wigner
// block D_l), and in the local frame Y_(l,m)(omega_(i,j)) = Theta_(l,m)(z_i) T_m(phi_j) with T_0 = 1,
// T_m = cos(m phi), T_-m = sin(m phi). Hence, per node,
//   b = D^T a                       (sum_l (2l+1) * active_l operations),
//   C(m, i) = sum_l Theta_(l,m)(z_i) b_(l,m)        (P q),
//   psi(i, j) = sum_m T_m(phi_j) C(m, i)            (P A (2N+1) = Q (2N+1)),
// instead of the dense Q q product. D and Theta are computed by quadrature (convention independent) and the
// factorization is verified against the dense harmonics at construction (relative 1e-12).
#include "common.hpp"
#include <Eigen/Core>

namespace m1pn {

class SeparableTrace {
public:
    // active: sorted global PN indices of the stored moments; hemispheres: +1 / -1 in point order.
    SeparableTrace(int order, const std::vector<int> &active, const Eigen::Matrix3d &frame, int n,
                   const std::vector<int> &hemispheres);
    int points() const { return P_ * A_; }
    // at: active x K (column k = coefficients of node k). out: points() x K, column k = psi of node k.
    void evaluate(const Mat &at, Mat &out) const;
    // Same as evaluate(), truncating at degree max_order <= N (exact when all coefficients with l > max_order are 0).
    void evaluate_upto(const Mat &at, Mat &out, int max_order) const;
    double verified_error = 0; // max |separable - dense| / max |dense| over random coefficient vectors

private:
    int N_ = 0, P_ = 0, A_ = 0;
    std::vector<int> block_start_, block_count_; // per l: rows of `at` (active list) in degree l
    std::vector<Mat> dt_;                        // per l: (2l+1) x block_count_[l], local m' rows
    std::vector<Mat> theta_;                     // per m + N: P x (N + 1 - |m|), columns l = |m| .. N
    Mat trig_;                                   // A x (2N+1), column m + N = T_m(phi_j)
};

} // namespace m1pn
