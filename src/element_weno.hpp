#pragma once
// Element-based limiting with WENO-stabilized high-order PN target.
//
// Space: tensor-product Bernstein Q_p on a uniform Cartesian mesh of [0,1]^d.
// Target: PN Galerkin with consistent mass and the CG-WENO term of
// Kuzmin, Hajduk, Vedral (arXiv:2509.04673v2), equations (6)-(7).
// Limiter: Section 6 of holo_m1_pn (clip and scale, then one element factor).
// See docs/ELEMENT_WENO.md for the formulas and the choices made here.
#include "boundary.hpp"
#include "m1.hpp"
#include <Eigen/SparseCholesky>
#include <functional>
#include <memory>
namespace m1pn {
struct ElementWenoOptions {
    int degree = 1, dimension = 1, elements = 16;
    bool periodic = true;
    bool stabilize = true; // false drops s_h^e altogether
    bool weno = true;      // false fixes gamma_e = 1
    bool limit = true;     // false applies the raw contributions
    double positivity_fraction = 1e-3; // epsilon of report (25)
    double element_theta = .999;       // theta of report Section 6.2
    double weno_q = 1, weno_r = 2, weno_epsilon = 1e-6, weno_linear = 1e-3;
    double viscosity_scale = 1;
    // Constant angular filter of nn_filter: d a_lm/dt = -sigma (l/(N+1))^4 a_lm,
    // half a step before and after each SSPRK3 step. Zero switches it off.
    double filter_strength = 0;
};
struct ElementWenoStats {
    double min_gamma = 1, min_limiter = 1, min_boundary_limiter = 1;
    double max_raw_zero_sum = 0, max_limited_zero_sum = 0;
    long limited_elements = 0, limited_boundary_nodes = 0, nudged_states = 0;
    // Wall seconds per phase of rate(), accumulated over all calls.
    double t_closure = 0, t_weno = 0, t_stabilization = 0, t_residual = 0, t_boundary = 0;
    double t_mass = 0, t_limiter = 0, t_finish = 0;
    double t_gradient_local = 0, t_gradient_gather = 0, t_gradient_solve = 0, t_stab_local = 0;
};
struct ElementWenoDebug {
    Field unlimited_rate, ho_moment_rate; // N x 4
};
class ElementWeno {
public:
    ElementWeno(const Angular &, const ElementWenoOptions &, Inflow = {});
    const Angular &pn;
    ElementWenoOptions options;
    Inflow inflow;
    int degree, dimension, elements, per_direction, nodes, cells, local_nodes;
    double h, viscosity, dt_bound;
    Vec lumped;
    // Semi-discrete rate: PN coefficients advance by the consistent-mass WENO
    // target; their degree 0 and 1 rows are replaced by the limited moments.
    Field rate(double time, const Field &a, ElementWenoStats * = nullptr,
               ElementWenoDebug * = nullptr) const;
    Field advance_ssprk3(double time, double dt, const Field &a, ElementWenoStats * = nullptr) const;
    Vec weno_gamma(const Field &a) const;
    Field angular_filter(const Field &a, double tau) const;
    Field stabilization_load(const Field &a, const Vec &gamma, ElementWenoStats * = nullptr) const; // s_h^e(phi_i, a) per element
    Field solve_mass(Field rhs) const;
    Field interpolate(const std::function<Vec(const V3 &)> &) const;
    Field project_l2(const std::function<Vec(const V3 &)> &) const;
    Vec evaluate(const Field &a, const V3 &x) const;
    V3 node_position(int node) const;
    double l2_density_error(const Field &a, const std::function<double(const V3 &)> &exact) const;

private:
    struct FaceRule {
        std::vector<int> local;
        Mat shape;       // face nodes x points
        Field points;    // points x d, element reference coordinates
        Vec weights;     // physical surface weights
    };
    struct BoundaryFace {
        int element, direction, side;
    };
    Mat mass_, stiffness_, seminorm_, viscosity_ij_;
    std::array<Mat, 3> convection_;
    std::array<std::array<Mat, 2>, 3> extension_;
    std::vector<int> connectivity_, neighbors_;
    // Node -> (element * local_nodes + local) incidence, for parallel gathers.
    std::vector<int> node_offsets_, node_entries_;
    Vec total_viscosity_;
    std::array<Incoming, 6> density_inflow_;
    bool cached_density_ = false;
    std::vector<BoundaryFace> faces_;
    std::array<FaceRule, 6> rules_;
    std::unique_ptr<CanonicalC1> canonical_;
    std::array<std::unique_ptr<Boundary>, 6> boundaries_;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> mass_1d_;
    // The same LDL^T factor in plain arrays, for row-wise multi-component solves.
    std::vector<int> factor_outer_, factor_inner_, factor_perm_;
    std::vector<double> factor_values_, factor_diagonal_;
    int stride(int direction) const;
    void element_origin(int element, V3 &origin) const;
    void add_boundary(double time, const Field &a, const Field &low, const Field &low_flux,
                      Field &high, Field &low_load, Field &node_flux_area, Vec &area,
                      Field &bar) const;
    Field gather_load(const Field &local) const;
};
// Exact PN solution for the periodic density sine wave of the accuracy study.
Vec element_weno_smooth_exact(const Angular &, const V3 &x, double t);
} // namespace m1pn
