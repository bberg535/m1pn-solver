#pragma once
#include "boundary.hpp"
#include "m1.hpp"
#include "spatial.hpp"
#include "stabilization.hpp"
#include "parallel.hpp"
#include "imex.hpp"
namespace m1pn {
struct Numerics {
    double gamma = .01, anchor = 1, mass_tolerance = 1e-11;
    std::string mass_mode = "consistent";
    int mass_corrections = 1;
    bool masked = false, c0 = false, density_vector = false;
    bool r2 = false;
    bool lazy = false;
    bool geometry_cache = false;
    std::string pn_boundary_quadrature = "lumped";
    std::string bh_pair_mode = "element", implicit_solver = "direct";
    double implicit_rtol = 1e-10, implicit_atol = 1e-14;
    int implicit_max_iterations = 300;
    std::string stabilization = "gradient-defect";
    double kv_gamma = .5;
    double pn_viscosity_scale = 1.;
    // Every ledger_stride-th step uses the audited all-Q path; between them the
    // standard mass-correction method skips the redundant pair fields of the
    // unlimited moments. 1 keeps the full path on every step.
    int ledger_stride = 1;
    Budget budget;
};
struct Diagnostics {
    double mass_residual = 0, target_recovery = 0, local_balance = 0, mass_defect = 0;
    double projection = 0, high_rate_error = 0, target_energy = 0, sync_work = 0;
    double minimum_state = std::numeric_limits<double>::infinity(), cfl_ratio = 0;
    double minimum_alpha = 1, minimum_eta = 1,
           minimum_bar = std::numeric_limits<double>::infinity();
    long active_pairs = 0, active_faces = 0;
    double density_clip_sum = 0, momentum_redirection_sum = 0, density_retention_sum = 0;
    long pair_target_passes = 0, pair_rounding_fallbacks = 0, pair_scalar_fallbacks = 0;
    long base_bar_reevaluations = 0;
    long lazy_certified_pairs = 0, lazy_fallback_pairs = 0;
    long closure_evaluations = 0;
    double implicit_residual = 0, implicit_relative_residual = 0;
    long implicit_solves = 0;
};
struct Stage {
    Field rate, target, accepted_pairs, boundary_gather, source, gather, mass_error;
    Diagnostics diagnostics;
};
struct FastMassCorrection;
struct FastMassCorrectionDeleter {
    void operator()(FastMassCorrection *) const;
};
struct Transport;
// Null unless the configuration is the standard method on planar boundary faces.
FastMassCorrection *make_fast_mass_correction(Transport &);
struct Transport {
    const Angular &pn;
    Spatial &fe;
    Numerics options;
    Runtime runtime;
    CanonicalC1 canonical;
    std::vector<std::unique_ptr<Boundary>> boundaries;
    Inflow inflow;
    Vec beta;
    std::unique_ptr<SpatialStabilization> stabilization_operator;
    std::unique_ptr<ImplicitMassSolver> implicit_solver;
    Sparse implicit_positive;
    std::vector<std::array<V3,2>> pair_geometry;
    std::vector<double> pn_low_order_edge_viscosity;
    // CSR incidence list, with edges in their original global pair order.
    std::vector<std::size_t> pair_node_offsets;
    std::vector<int> pair_node_edges;
    std::vector<signed char> pair_node_sign;
    std::vector<int> pn_boundary_map;
    double pn_boundary_map_norm = 1.;
    double generator_norm_bound = 0;
    Timings timings;
    // Stage workspace. Resized once and filled in place so the 76 MB fields
    // are not allocated again on every SSPRK stage.
    mutable std::array<Field, 3> stage_angular;
    mutable Field stage_rhs;
    mutable Field stage_stabilization_pairs;
    mutable Field stage_boundary;
    mutable Field stage_u1;
    mutable Field stage_u2;
    // Built only when the configuration matches the low-moment path; see fast_mc.cpp.
    std::unique_ptr<FastMassCorrection, FastMassCorrectionDeleter> fast_mc;
    std::uint64_t advance_count = 0;
    Transport(const Angular &, Spatial &, const Numerics &, Inflow = {}, Runtime = {});
    Stage evaluate(double time, const Field &u, bool unlimited = false,
                   const Field &integrated_source = Field());
    Stage evaluate_r2(double time, const Field &u, bool unlimited = false,
                     const Field &integrated_source = Field(), double implicit_dt = 0);
    Stage evaluate_imex(double time, double dt, const Field &u);
    Field biharmonic(const Field &, Field *w = nullptr) const;
    Field gather_pairs(const Field &) const;
    Field linear_target(const Field &) const;
    Field mass_target(const Field &rhs, double &residual, Field *predictor = nullptr) const;
    // boundary_rhs is the signed, integrated numerical PN boundary load (-H).
    // angular_flux, when set, is {A[0]*u, A[1]*u, A[2]*u}. Predictor edges then
    // use sum_d c[d] * (flux_d[j] - flux_d[i]). Null keeps fast_apply.
    Field pn_low_order_rate(const Field &u, const Field &assembled_pn_boundary_rhs = Field(),
                            const Field &source = Field(),
                            const std::array<Field, 3> *angular_flux = nullptr) const;
    Field mass_correction(const Field &rate) const;
    Field mass_correction_target(const Field &consistent_rhs, const Field &low_rate) const;
    void check_state(const Field &) const;
};
} // namespace m1pn
