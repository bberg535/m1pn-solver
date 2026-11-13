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
#include "positivity_projection.hpp"
#include "separable_trace.hpp"
#include <Eigen/SparseCholesky>
#include <functional>
#include <limits>
#include <memory>
#include <string>
namespace m1pn {
// Space-dependent isotropic reaction data (docs/ELEMENT_HETEROGENEOUS.md). sigma_a, sigma_s >= 0 in 1/length,
// source = isotropic particle density rate q0(x) >= 0 (units of rho per time; PN coefficient q0 / sqrt(4 pi)).
// Nodal values are the positive averages int sigma phi_i / int phi_i (Bernstein phi_i >= 0) by composite
// Gauss quadrature: `subcells` sub-cells per element and direction, `points` Gauss points per sub-cell and
// direction (0: degree + 4). Empty functions mean zero.
struct ElementWenoMaterial {
    std::function<double(const V3 &)> sigma_a, sigma_s, source;
    int subcells = 1, points = 0;
    std::string name;
};
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
    // Boundary/sensor corrections of docs/BOUNDARY_FIXES.md. All false reproduces commit 8efe517 bit for bit.
    bool vacuum_sign_limit = true;      // C2: accepted boundary mass rate may not exceed max(low-order rate, kinetic inflow)
    bool outward_bar_rounding = true;   // C1: directed (outward) rounding of the stable base bar, counted, <= 8 ulp, never a floor
    bool fullsphere_ghost = true;       // C6: M1 ghost = incoming kinetic moments + outgoing moments of the positive PN trace
    bool scale_invariant_weno = true;   // C5: epsilon of the WENO weights is relative to the stencil scale
    bool symmetric_residual = false;    // L3 alternative: symmetric PSD residual form of s_h (changes the operator for varying gamma)
    // Constant angular filter of nn_filter: d a_lm/dt = -sigma (l/(N+1))^4 a_lm,
    // half a step before and after each SSPRK3 step. Zero switches it off.
    double filter_strength = 0;
    // Constant isotropic scattering and absorption (unit speed):
    // d a_lm/dt = -(sigma_a + sigma_s [l >= 1]) a_lm. Integrated exactly together
    // with the filter in the same two half steps; the density is kept by
    // scattering and the M1 cone is preserved.
    double sigma_s = 0, sigma_a = 0;
    // Time integrator: false = SSPRK3 (Shu-Osher, SSP coefficient 1), true = SSPRK(4,3) (four forward Euler
    // stages of dt/2, SSP coefficient 2). Both are third order; the filter/reaction half steps are the same.
    bool ssprk43 = false;
    // Angular parity reduction (dimension < 3, docs/ELEMENT_WENO.md): drop the PN moments that are odd under
    // z -> -z and, in 1D, under y -> -y. Exact for even data, because the transport and boundary matrices of
    // the resolved directions never couple even and odd moments (checked at construction). The moments with
    // l <= 1 are always kept, so the M1 part (closure, limiter, boundary bar states) is unchanged.
    bool reduce_angular = false;
    // Domain origin + [0, length]^d (default: the unit cube).
    V3 origin = V3::Zero();
    double length = 1;
    // Optional element counts per axis (0 = `elements`). Cells stay cubic with h = length / count[0], so the
    // box is origin + [0, h count[0]] x [0, h count[1]] x [0, h count[2]] (e.g. 150 x 50 x 50 for the shadow
    // test). Non-cubic cells are not supported.
    std::array<int, 3> elements_per_axis{0, 0, 0};
    // Space-dependent reaction (adds to the constant sigma_s, sigma_a above, which must then be zero).
    std::shared_ptr<const ElementWenoMaterial> material;
    // Mirror sides, index 2 * direction + side (side 0 = low end): specular reflection instead of a physical
    // boundary. For mirror-symmetric data the scheme equals the full scheme on the mirrored domain: no flux on
    // the mirror face, moments odd under the reflection vanish on the mirror plane (Dirichlet rows of the
    // consistent mass for those components), and the WENO neighbour across the mirror is the mirrored element.
    // At most one mirror side per direction; not with periodic meshes or the symmetric residual form.
    std::array<bool, 6> mirror{};
    // M1 target instead of PN (diagnostic, docs/ELEMENT_HETEROGENEOUS.md): the high-order rate of (rho, J) is the
    // group-FE Galerkin rate -sum_d M^-1 C_d F^L_d of the nodal Levermore flux plus s_h, the closure difference
    // G = F^H - F^L is zero and the boundary keeps the low-order M1 flux (correction factor 0). Requires P1 and no
    // mirror sides (the flux components have other reflection parities than the moments).
    bool m1_target = false;
    // First-collision source (FCS, docs/FCS.md): when set, the state is the *collided* part psi_c and the
    // reaction step adds the isotropic source sigma_s(x) rho_u(x, t) of the analytic uncollided density rho_u.
    // The function fills rho_u(t) at a batch of physical points. The nodal source is the positive average
    // int sigma_s rho_u phi_i / int phi_i by Gauss quadrature (uncollided_points per element and direction,
    // 0: degree + 3), evaluated at the midpoint of each Strang half step (second order in time).
    std::function<void(double, const std::vector<V3> &, std::vector<double> &)> uncollided_density;
    int uncollided_points = 0;
    // Moment-preserving positivity projection of the C6 ghost (docs/POSITIVITY_PROJECTION.md). Requires
    // fullsphere_ghost. Instead of max(0, psi_N) on the outgoing hemisphere and a cone scaling, the nodal PN
    // trace on the full sphere is projected onto {z >= 0, sum w m z = (rho_i, J_i)} (four multipliers, semismooth
    // Newton); the ghost uses the outgoing half-range moments of z. A failed projection is counted and falls back
    // to the C6 ghost of that node, or throws with projection_strict.
    bool positivity_projection = false;
    bool projection_strict = false;
    // Exactness parameter n of sphere_rule(n, +-1) per hemisphere (0: 2N + 1, the C6 rule). Moment consistency
    // (identity for nonnegative PN traces) needs n >= N + 1.
    int projection_rule = 0;
    double projection_tolerance = 1e-12;
    bool projection_warm_start = true; // start Newton from the node's last multiplier (iteration count only)
};
struct ElementWenoStats {
    double min_gamma = 1, min_limiter = 1, min_boundary_limiter = 1;
    double max_raw_zero_sum = 0, max_limited_zero_sum = 0;
    long limited_elements = 0, limited_boundary_nodes = 0, nudged_states = 0;
    // C1/C2 counters: outward-rounded base bars; boundary nodes whose correction was reduced by the sign condition.
    long outward_bars = 0, sign_limited_nodes = 0;
    double min_sign_factor = 1;
    // Wall seconds per phase of rate(), accumulated over all calls. t_stabilization = t_gradient_solve +
    // t_stab_local. The gradient is a line operator and s_h is gathered node by node, so t_gradient_local,
    // t_gradient_gather and t_residual stay zero (kept for the summary format).
    double t_closure = 0, t_weno = 0, t_stabilization = 0, t_residual = 0, t_boundary = 0;
    double t_mass = 0, t_limiter = 0, t_finish = 0;
    double t_gradient_local = 0, t_gradient_gather = 0, t_gradient_solve = 0, t_stab_local = 0;
    // Mass ledger in rho units (sum_i m_i rho_i, m_i = lumped mass), accumulated by advance():
    //   M(T) - M(0) + boundary_outflow + absorbed - injected - nudge_mass = 0 up to rounding.
    // boundary_outflow integrates the boundary mass rate of every stage with the SSP weights; absorbed and
    // injected come from the exact reaction step (absorbed analytically from sigma_a int rho dt).
    double boundary_outflow = 0, absorbed = 0, injected = 0, nudge_mass = 0;
    double last_boundary_mass_rate = 0; // of the last rate() call (sum over boundary nodes)
    double t_reaction = 0;
    double t_fcs_density = 0, t_fcs_quadrature = 0; // FCS: rho_u evaluation and nodal source quadrature
    PositivityProjectionStats projection;         // positivity projection of the ghost (rate() calls only)
};
struct ElementWenoDebug {
    Field unlimited_rate, ho_moment_rate; // N x 4
};
class ElementWenoDlra;
class ElementWeno {
    friend class ElementWenoDlra; // factored (low-rank) evaluation of the same operators, src/element_weno_dlra.cpp
public:
    ElementWeno(const Angular &, const ElementWenoOptions &, Inflow = {});
    const Angular &pn;
    ElementWenoOptions options;
    Inflow inflow;
    int degree, dimension, elements, per_direction, nodes, cells, local_nodes;
    // Elements and nodes per axis (elements = count[0], per_direction = nodes_per_axis[0]).
    std::array<int, 3> count{1, 1, 1}, nodes_per_axis{1, 1, 1};
    // Active PN moments per node (pn.q without reduction) and their index in the full basis.
    int moments;
    std::vector<int> active;
    // Full basis <-> active moments. reduce() drops the inactive entries, expand() fills them with zero.
    Vec reduce(const Vec &full) const;
    Field expand(const Field &a) const;
    // Largest inactive part of the PN boundary datum term llf_datum d over all boundary points at `time`,
    // relative to the largest active part (0 without reduction or boundaries). Used to verify even inflow.
    double dropped_inflow(double time) const;
    double h, viscosity, dt_bound;
    Vec lumped;
    // Semi-discrete rate: PN coefficients advance by the consistent-mass WENO
    // target; their degree 0 and 1 rows are replaced by the limited moments.
    Field rate(double time, const Field &a, ElementWenoStats * = nullptr,
               ElementWenoDebug * = nullptr) const;
    Field advance_ssprk3(double time, double dt, const Field &a, ElementWenoStats * = nullptr) const;
    Field advance_ssprk43(double time, double dt, const Field &a, ElementWenoStats * = nullptr) const;
    // One step of the integrator selected by options.ssprk43.
    Field advance(double time, double dt, const Field &a, ElementWenoStats *stats = nullptr) const {
        return options.ssprk43 ? advance_ssprk43(time, dt, a, stats) : advance_ssprk3(time, dt, a, stats);
    }
    // SSP coefficient of the integrator: every stage is a forward Euler step of size dt / ssp_coefficient().
    double ssp_coefficient() const { return options.ssprk43 ? 2. : 1.; }
    // Realizability bound of a full step: ssp_coefficient() * dt_bound.
    double realizability_step() const { return ssp_coefficient() * dt_bound; }
    Vec weno_gamma(const Field &a) const;
    // Exact local reaction exp(-tau R): filter, isotropic scattering, absorption.
    Field angular_filter(const Field &a, double tau) const;
    // True when the reaction step changes anything (filter, scattering, absorption or source).
    bool reacts() const {
        return options.filter_strength != 0 || options.sigma_s != 0 || options.sigma_a != 0 || bool(options.material) ||
               bool(options.uncollided_density);
    }
    // Exact local reaction over tau (filter, absorption, scattering, source), node by node; see react() in the .cpp.
    // Adds the absorbed and injected mass to the ledger of `stats`.
    // source_time: time at which the FCS source sigma_s rho_u is evaluated (required with uncollided_density).
    Field react(const Field &a, double tau, ElementWenoStats * = nullptr,
                double source_time = std::numeric_limits<double>::quiet_NaN()) const;
    // FCS: nodal collided source int sigma_s rho_u(t) phi_i / int phi_i (density rate) and int rho_u(t) dx over
    // the computed domain (zero without uncollided_density).
    Vec collided_source(double t, double *uncollided_mass = nullptr, ElementWenoStats *stats = nullptr) const;
    // Rounding-only nudge of the low moments into the M1 cone (counted, mass recorded in the ledger).
    Field admissible_rows_public(Field x, ElementWenoStats * = nullptr) const;
    // Nodal reaction data (zero without material): averages int c phi_i / int phi_i, and the quadrature
    // integrals int c dx over the computed domain.
    Vec node_sigma_a, node_sigma_s, node_source;
    double integral_sigma_a = 0, integral_sigma_s = 0, integral_source = 0, integral_volume = 0;
    // Linear stability of the high-order target: 0.9 * c / rho with rho the spectral radius of
    // M^-1(-C.A - S_gamma), estimated by power iteration for gamma = 0 and gamma = 1 without boundary terms,
    // and c the stability limit of the integrator on the negative real axis (SSPRK3: 2.5127, SSPRK(4,3):
    // 5.1495). The largest eigenvalues of the stabilized target are dissipative (close to the negative real
    // axis); the periodic amplification test in tests/element_weno.cpp checks this. Cached per integrator.
    double high_order_step() const;
    // Step limit used by the drivers: min(realizability bound, high_order_step).
    double step_bound() const { return std::min(realizability_step(), high_order_step()); }
    // M^-1(-C.A - S_gamma) a with constant gamma and without boundary terms (the operator of high_order_step).
    Field linear_operator(const Field &x, double gamma) const;
    Field stabilization_load(const Field &a, const Vec &gamma, ElementWenoStats * = nullptr) const; // s_h^e(phi_i, a) per element
    Field solve_mass(Field rhs) const;
    // Consistent mass times x (tensor product of the 1D Bernstein masses), any number of columns, no mirror sides.
    Field apply_mass(const Field &x) const;
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
        Mat mass;        // face nodes x face nodes, sum_q w_q phi_n phi_m
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
    // LDL^T factors of the 1D mass in plain arrays, for row-wise multi-component line solves. `line[r]` is the
    // position on the grid line of factor row r. natural_: all nodes; dirichlet_[d]: without the mirror node.
    struct LineFactor {
        int n = 0;
        std::vector<int> outer, inner, perm, line;
        std::vector<double> values, diagonal;
    };
    std::array<LineFactor, 3> natural_;
    std::array<LineFactor, 3> dirichlet_;
    // FCS quadrature: physical points (element-major, fcs_per_element_ per element), sigma_s and weight at each
    // point, shape values phi_l at the per-element points, and int phi_i dx of the same rule.
    std::vector<V3> fcs_points_;
    std::vector<double> fcs_sigma_s_, fcs_weight_;
    Mat fcs_shape_;
    Vec fcs_node_mass_;
    int fcs_per_element_ = 0;
    std::array<int, 3> mirror_index_{-1, -1, -1};   // grid-line position of the mirror node per direction
    std::array<std::vector<char>, 3> reflect_odd_;  // per active moment: odd under x_d -> -x_d
    std::array<std::vector<int>, 3> mirror_nodes_;  // nodes on the mirror plane of direction d
    std::array<std::vector<int>, 3> reflected_local_; // local node index mirrored in direction d
    void solve_line(const LineFactor &, double *work, int q) const;
    // In place along direction d: natural solve, or (mirror in d) Dirichlet solve for the components that are
    // odd under the reflection; `gradient` flips the parity (d/dx_d of an even field is odd).
    void solve_lines(Field &b, int direction, bool gradient) const;
    int stride(int direction) const;
    void element_origin(int element, V3 &origin) const;
    // inflow_rate (optional, nodes): kinetic mass inflow rate int phi_i (-kinetic[0]) >= 0 per boundary node.
    // `high` is indexed by node (nodes rows) or compactly by boundary_slot_ (boundary_nodes_.size() rows).
    // DLRA hooks of add_boundary: ghost_psi(k, side, node, psi) fills the PN trace at the points of out_rules_[side]
    // (outgoing hemisphere, plus the incoming one with the positivity projection) of ghost pair k (instead of
    // out.harmonics * a.row(node)); datum (if set) receives per side the face datum loads
    // sum_points w phi_n d (boundary_slot_ rows x canonical r).
    struct BoundaryExtras {
        std::function<void(int, int, int, Vec &)> ghost_psi;
        std::array<Field, 6> *datum = nullptr;
    };
    void add_boundary(double time, const Field &a, const Field &low, const Field &low_flux,
                      Field &high, Field &low_load, Field &node_flux_area, Vec &area,
                      Field &bar, Vec *inflow_rate = nullptr, const BoundaryExtras *extras = nullptr,
                      PositivityProjectionStats *projection = nullptr) const;
    // Inputs of the limited low-order part of rate() (all node-indexed except boundary_high, which has one row per
    // boundary slot and at least the four low coefficient columns).
    struct LowOrderInputs {
        const Field *low, *low_flux, *closure, *moment_rate, *stabilization_low, *boundary_high;
        const Field *boundary_low, *flux_area, *boundary_bar;
        const Vec *boundary_area, *boundary_inflow, *gamma;
    };
    Field limited_low_load(const LowOrderInputs &, ElementWenoStats *, Field *raw_out) const;
    std::array<Sparse, 3> mass_1d_; // assembled 1D consistent masses (h m1) per axis
    // Outgoing hemisphere (omega, weight, harmonics) per face side, for the C6 ghost state. With the positivity
    // projection the incoming hemisphere follows as rows [outgoing, points) (full sphere); the C6 sums only use the
    // first `outgoing` rows. ox/oy/oz are contiguous copies of the omega columns for the projection kernel.
    struct OutRule {
        Field omega;   // points x 3
        Vec weight;
        Mat harmonics; // points x q
        int outgoing = 0;
        Vec ox, oy, oz;
        PositivityFacets facets; // hull of the full-sphere directions (positivity projection only)
    };
    std::array<OutRule, 6> out_rules_;
    // Separable trace evaluation per side (src/separable_trace.hpp; null: dense out_rules_[side].harmonics) and the
    // ghost pairs of each side.
    std::array<std::shared_ptr<const SeparableTrace>, 6> separable_;
    std::array<std::vector<int>, 6> ghost_by_side_;
    // Positivity projection warm start per ghost pair (multiplier and rule of the last accepted projection). Only a
    // starting guess: the projection is unique, so the result does not depend on it beyond the solver tolerance.
    mutable std::vector<PositivityProjectionWarm> projection_warm_;
    std::array<Field, 6> out_harmonics_; // row-major copies of out_rules_[side].harmonics (batched trace products)
    // Kinetic inflow data at a face quadrature point (one code path for the steady cache and the per-stage call).
    Incoming face_data(int side_id, const V3 &x, double time) const;
    void face_data_into(int side_id, const V3 &x, double time, Incoming &data) const;
    std::vector<Incoming> steady_data_; // per (face, point) when inflow.steady
    std::vector<int> boundary_nodes_sorted_;
    struct PairEntry {
        int j;
        double d, m, c[3], ct[3];
    };
    std::vector<PairEntry> pair_table_; // limiter pairs (i, j != i), row-major in i // ascending, for the deterministic parallel boundary corrections
    Field gather_load(const Field &local, double scale = 1.) const;
    // M^-1 C_d a along the grid lines of direction d, and the element loads of s_h for given gradients.
    Field projected_gradient(const Field &a, int direction) const;
    Field stabilization_load(const Field &a, const Vec &gamma, const std::array<Field, 3> &gradient,
                             ElementWenoStats * = nullptr) const;
    // Node-wise gathered -s_h(phi_i, a) plus (optionally) the projected low moments of the element loads.
    Field stabilization_residual(const Field &a, const Vec &gamma, const std::array<Field, 3> &gradient,
                                 Field *low, ElementWenoStats * = nullptr) const;
    // Assembled 1D convection matrix, A_d^T, and row-major element tables (K_e, C_{d,e}^T) for the row kernels.
    std::array<Sparse, 3> convection_1d_;
    std::array<Sparse, 3> flux_t_;
    std::vector<double> stiffness_rows_;
    std::array<std::vector<double>, 3> convection_t_rows_;
    std::array<Mat, 6> boundary_flux_t_; // (A_n - llf_homogeneous)^T per face side (active moments)
    std::vector<char> parity_odd_;       // odd under the reflections of the angular reduction (full index)
    std::array<Mat, 6> datum_t_;         // llf_datum^T restricted to the active moments (r x moments)
    // Boundary nodes (compact row index of the PN boundary load) and the distinct (side, node) pairs of the
    // C6 ghost state; ghost_of_face_ maps (face, face node) to such a pair.
    std::vector<int> boundary_slot_, boundary_nodes_, ghost_node_, ghost_side_, ghost_of_face_;
    mutable double spectral_radius_ = -1;
};
// PN moments of the reflected distribution psi(R_d omega): moments odd under x_d -> -x_d change sign.
Vec reflect_pn(const Angular &, const Vec &full, int direction);
// Exact PN solution for the periodic density sine wave of the accuracy study.
Vec element_weno_smooth_exact(const Angular &, const V3 &x, double t);
} // namespace m1pn
