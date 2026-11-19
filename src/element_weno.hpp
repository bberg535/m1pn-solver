#pragma once
// Element-based limiting with WENO-stabilized high-order PN target.
//
// Space: tensor-product Bernstein Q_p on a uniform Cartesian mesh of [0,1]^d.
// Target: PN Galerkin with consistent mass and the CG-WENO term of
// Kuzmin, Hajduk, Vedral (arXiv:2509.04673v2), equations (6)-(7).
// Limiter: one element factor against the M1 cone (Section 6.2 of holo_m1_pn; without the clip-and-scale stage).
// See docs/ELEMENT_WENO.md for the formulas and the choices made here.
#include "adaptive_pn.hpp"
#include "boundary.hpp"
#include "m1.hpp"
#include <random>
#include "positivity_projection.hpp"
#include "separable_trace.hpp"
#include <Eigen/SparseCholesky>
#include <functional>
#include <limits>
#include <memory>
#include <string>
namespace m1pn {
namespace uncollided {
struct GaussianPulse;
struct LatticeUncollidedTable;
struct Lattice3dUncollidedTable;
}
// Space-dependent isotropic reaction data (docs/ELEMENT_HETEROGENEOUS.md). sigma_a, sigma_s >= 0 in 1/length,
// source = isotropic particle density rate q0(x) >= 0 (units of rho per time; PN coefficient q0 / sqrt(4 pi)).
// Nodal values are the positive averages int sigma phi_i / int phi_i (Bernstein phi_i >= 0) by composite
// Gauss quadrature: `subcells` sub-cells per element and direction, `points` Gauss points per sub-cell and
// direction (0: degree + 4). Empty functions mean zero.
// Mesh-independent angular data of the scheme (boundary matrices, hemisphere rules, separable traces, adaptive ghost
// tiers), shared between schemes that differ only in the mesh (dynamic AMR rebuilds, docs/DYNAMIC_AMR.md). The first
// scheme constructed with a cache fills it; later ones with the same angular key copy it instead of rebuilding.
struct ElementWenoAngularCache;
std::shared_ptr<ElementWenoAngularCache> make_angular_cache();
long angular_cache_loads(const ElementWenoAngularCache &);
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
    double element_theta = .999;       // theta of report Section 6.2
    double weno_q = 1, weno_r = 2, weno_epsilon = 1e-6, weno_linear = 1e-3;
    // Study variants of the sensor at the rounding limit (docs/ELEMENT_WENO.md; CPU only, default off = bitwise the
    // standard): soft threshold ratio -> max(0, ratio - d) / (1 - d), and a floor of the sensor denominator
    // |u|_e -> max(|u|_e, f max_i |u_i|) with u_i the density coefficients of the element.
    double weno_deadband = 0, weno_floor = 0;
    double viscosity_scale = 1;
    // Boundary/sensor corrections of docs/BOUNDARY_FIXES.md. All false reproduces commit 8efe517 bit for bit.
    bool vacuum_sign_limit = true;      // C2: accepted boundary mass rate may not exceed max(low-order rate, kinetic inflow)
    bool outward_bar_rounding = true;   // C1: directed (outward) rounding of the stable base bar, counted, <= 8 ulp, never a floor
    bool fullsphere_ghost = true;       // C6: M1 ghost = incoming kinetic moments + outgoing moments of the positive PN trace
    bool scale_invariant_weno = true;   // C5: epsilon of the WENO weights is relative to the stencil scale
    bool symmetric_residual = false;    // L3 alternative: symmetric PSD residual form of s_h (changes the operator for varying gamma)
    // Constant angular filter (former nn_filter study): d a_lm/dt = -sigma (l/(N+1))^4 a_lm,
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
    // EXPERIMENT (experiment/big-rebuilds, docs/BIG_REBUILDS.md): SSPRK(n^2,3) of Ketcheson for ssp_n >= 3 (n = 2 is
    // SSPRK(4,3)); n^2 forward Euler stages of dt/(n^2 - n), SSP coefficient n^2 - n, real stability interval per
    // stage 1.48 (n = 3), 1.59 (n = 4) against 1.29. Requires ssprk43 (set by the driver).
    int ssp_n = 0;
    // Angular parity reduction (dimension < 3, docs/ELEMENT_WENO.md): drop the PN moments that are odd under
    // z -> -z and, in 1D, under y -> -y. Exact for even data, because the transport and boundary matrices of
    // the resolved directions never couple even and odd moments (checked at construction). The moments with
    // l <= 1 are always kept, so the M1 part (closure, limiter, boundary bar states) is unchanged.
    bool reduce_angular = false;
    // Domain origin + [0, length]^d (default: the unit cube).
    V3 origin = V3::Zero();
    // MPI subdomains: global element index of local element 0 per axis. Positions are computed from the global
    // index (origin + global * h), so a slab reproduces the serial coordinates bit for bit.
    std::array<int, 3> element_offset{};
    double length = 1;
    // Optional element counts per axis (0 = `elements`). Cells stay cubic with h = length / count[0], so the
    // box is origin + [0, h count[0]] x [0, h count[1]] x [0, h count[2]] (e.g. 150 x 50 x 50 for the shadow
    // test). Non-cubic cells are not supported.
    std::array<int, 3> elements_per_axis{0, 0, 0};
    // Graded tensor mesh (docs/GRADED_MESH.md): element boundaries per axis (count + 1 increasing coordinates,
    // absolute). Empty: the uniform mesh above (bitwise unchanged). Serial CPU only for now (no MPI, GPU, mirror
    // sides, periodic meshes); elements are boxes with the widths of their axis intervals.
    std::array<std::vector<double>, 3> planes;
    // Space-dependent reaction (adds to the constant sigma_s, sigma_a above, which must then be zero).
    std::shared_ptr<const ElementWenoMaterial> material;
    // Mirror sides, index 2 * direction + side (side 0 = low end): specular reflection instead of a physical
    // boundary. For mirror-symmetric data the scheme equals the full scheme on the mirrored domain: no flux on
    // the mirror face, moments odd under the reflection vanish on the mirror plane (Dirichlet rows of the
    // consistent mass for those components), and the WENO neighbour across the mirror is the mirrored element.
    // At most one mirror side per direction; not with periodic meshes or the symmetric residual form.
    std::array<bool, 6> mirror{};
    // MPI subdomains (src/element_weno_mpi.hpp): sides 2d + side that are internal cuts of the global mesh. No
    // boundary faces there; the halo elements beyond belong to neighbouring ranks.
    std::array<bool, 6> cut{};
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
    // Multi-block coupling (docs/STATIC_HP_BLOCKS.md): faces of an interface side couple to a neighbouring block
    // instead of the kinetic inflow. interface_trace(side, x, v) returns the neighbour's PN coefficients at x in this
    // scheme's active layout (degrees above the neighbour's order zero). Face term .5 (A_n - I)(u - v) (Rusanov),
    // limiter bar state with the neighbour's moments as exterior; no C6 ghost on these faces.
    std::array<bool, 6> interface_side{};
    // Broken lines (docs/STATIC_HP_BLOCKS.md, "Trennebenen"): per axis the element lines k (0 < k < count) at which
    // the nodes are doubled. Each side keeps its own continuous space; the faces on a broken line couple by the
    // interface term (.5 (A_n - I)(u - v), v = the trace of the element across, limiter bar state with its moments).
    // On the lattice: the material lines, so no node averages absorber and scatterer data. Serial, open mesh.
    std::array<std::vector<int>, 3> breaks;
    // EXPERIMENT (experiment/break-dt, docs/BREAK_DT.md, "a) Geteilter Schritt"): each forward Euler stage first
    // applies the element and domain-boundary terms (y = u + tau/m (E + B_dom)), then the broken-line face terms as
    // a second convex step from y: z = y + tau A/m (bar(y) + theta c/A - y) with c = T - (Bbar(y) - A y), T the
    // unsplit face term at u. theta = 1 gives exactly the unsplit stage; theta < 1 only where the cone needs it.
    // dt_bound then uses min(m / (sum 2 tv + A_dom), m / A_break) per node. Serial CPU only.
    bool split_breaks = false;
    std::function<void(int, const V3 &, Vec &)> interface_trace;
    // Non-conforming interfaces: interface_split(side, face centre, face width) = number of neighbour elements along
    // each tangential axis of this face (> 1: the face rule is applied on that many sub-faces per axis, so the face
    // integrals of the piecewise-polynomial neighbour trace stay exact). Null or 1: one rule per face.
    std::function<int(int, const V3 &, double)> interface_split;
    // Conforming interfaces (same node positions on both sides): nodal coupling as on broken lines, no point loop.
    // interface_partner(side, node) = (block, node across); interface_value(block, node, v): that node's coefficients in
    // this scheme's active layout; interface_low(block, node, n, u, flux_n): its four moments and the M1 normal flux
    // F(u) n, computed as the owner computes them (so both sides of a face use the same states).
    std::function<std::pair<int, int>(int, int, int)> interface_partner; // (side, element, node) -> (block, node)
    std::function<void(int, int, Vec &)> interface_value;
    std::function<void(int, int, const V3 &, Four &, Four &)> interface_low;
    // split_breaks at conforming block interfaces (experiment/big-rebuilds): the partner's step-1 state y and its
    // normal M1 flux F(y) n (ElementWeno::split_y_state of the partner block), read after all blocks computed y.
    std::function<void(int, int, const V3 &, Four &, Four &)> interface_split_y;
    // The same analytic pulse as an object (GPU path: the device evaluates its radial table, src/element_weno_gpu.cpp).
    std::shared_ptr<const uncollided::GaussianPulse> uncollided_pulse;
    // Lattice FCS (docs/FCS_LATTICE.md): the time table of rho_u at the FCS points; the GPU path builds it on the
    // device and fills the host copy.
    std::shared_ptr<uncollided::LatticeUncollidedTable> uncollided_lattice;
    // lattice-3d FCS (docs/FCS_LATTICE.md, section 3D): the same for the 3D table.
    std::shared_ptr<uncollided::Lattice3dUncollidedTable> uncollided_lattice3d;
    int uncollided_points = 0;
    // Moment-preserving positivity projection of the C6 ghost (docs/POSITIVITY_PROJECTION.md). Requires
    // fullsphere_ghost. Instead of max(0, psi_N) on the outgoing hemisphere and a cone scaling, the nodal PN
    // trace on the full sphere is projected onto {z >= 0, sum w m z = (rho_i, J_i)} (four multipliers, semismooth
    // Newton); the ghost uses the outgoing half-range moments of z. A failed projection is counted and falls back
    // to the C6 ghost of that node, or throws with projection_strict. Default since 2026-11-14 (more accurate
    // outgoing state than C6, docs/POSITIVITY_PROJECTION.md 2b); set false for the C6 ghost.
    bool positivity_projection = true;
    bool projection_strict = false;
    // Exactness parameter n of sphere_rule(n, +-1) per hemisphere (0: N + 1; the C6 ghost uses 2N + 1). Moment
    // consistency (identity for nonnegative PN traces) needs n >= N + 1.
    int projection_rule = 0;
    double projection_tolerance = 1e-12;
    // Weight c_in of the incoming hemisphere in the projection objective 1/2 sum w c (z - y)^2 (outgoing: c = 1).
    // 1 is the note's projection; c_in -> 0 lets the incoming half absorb the moment correction, so the outgoing trace
    // stays max(0, psi) whenever the remaining moments are realizable on the incoming half (C6 without scaling).
    double projection_incoming_weight = 1;
    // Half-range variant: project the outgoing trace alone onto {z >= 0, sum_out w m z = sum_out w m psi_N} (the
    // outgoing half moments of the PN trace). Identity (= C6) where psi_N >= 0 on the outgoing hemisphere; unsolvable
    // half moments fall back to the C6 ghost of that node (counted as failures). Uses the outgoing rule only.
    bool projection_half_range = false;
    // Angular filter of the ghost trace (C6 and projection): coefficients of degree l >= 2 are multiplied by
    // exp(-ghost_filter (l / (N + 1))^4) before the trace is evaluated; l <= 1 (rho, J) is unchanged. 0: off.
    double ghost_filter = 0;
    bool projection_diagnostics = false;
    // Hybrid: C6 ghost where admissible (a_ext <= b); the projection replaces only the C6 cone scaling cases.
    bool projection_hybrid = false; // also evaluate the C6 ghost and record its distance (slower)
    bool projection_warm_start = true; // start Newton from the node's last multiplier (iteration count only)
    // Projected exterior state also in the PN high-order boundary flux: the LLF/C1 exterior state uses
    // u~ = u + delta, delta = sum_k w_k Y(Omega_k) (z*_k - y_k) (+ atom), instead of the trace u. delta has zero
    // M1 moments (moment consistency of the rule), vanishes for identity projections, and changes the face
    // integrand by -(llf_homogeneous - (A_n + I)/2) delta. Full projection mode only.
    bool projection_high_order = false;
    // Dynamic tile-based adaptive PN order (experiment/dynamic-adaptive-pn): each spatial tile of size
    // adaptive_tile^d nodes carries an odd PN order N_T in [adaptive_min_order, pn.order] updated every
    // adaptive_stride steps by a spectral tail sensor with hysteresis and border-aware wave-front dilation.
    bool adaptive_pn = false;
    double adaptive_tol = 3e-4;
    double adaptive_hysteresis = 0.25;
    int adaptive_tile = 16;
    int adaptive_stride = 4;
    int adaptive_min_order = 1;
    int adaptive_halo = 1; // 0: off, 1: border-aware neighbor dilation, 2: full 3x3 dilation
    std::shared_ptr<ElementWenoAngularCache> angular_cache; // optional (see make_angular_cache)
    bool adaptive_restart = false;
    double adaptive_floor = 0;                // local relative PN tolerance (AdaptivePnPlan::floor), 0: global
    std::vector<int> adaptive_node_order;     // carried PN order per node (dynamic AMR), empty: none // adaptive PN on a transferred state (dynamic AMR): border-triggered initial dilation
};
struct ElementWenoStats {
    double min_gamma = 1, min_limiter = 1, min_boundary_limiter = 1;
    double max_raw_zero_sum = 0, max_limited_zero_sum = 0;
    long limited_elements = 0, limited_boundary_nodes = 0, nudged_states = 0;
    // Diagnostics: rate() calls and those in which no element and no boundary node was limited.
    long rate_calls = 0, unlimited_rate_calls = 0;
    // C1/C2 counters: outward-rounded base bars; boundary nodes whose correction was reduced by the sign condition.
    long outward_bars = 0, sign_limited_nodes = 0;
    // split_breaks: broken-line nodes whose face correction was reduced (theta < 1), and the mass that this limiting
    // removed or added (sum m_i (z - unsplit)[0], tau included; not part of the ledger otherwise).
    long split_limited = 0;
    double split_mass = 0, min_split_theta = 1;
    double min_sign_factor = 1;
    // Wall seconds per phase of rate(), accumulated over all calls. t_stabilization = t_gradient_solve +
    // t_stab_local. The gradient is a line operator and s_h is gathered node by node, so t_gradient_local,
    // t_gradient_gather and t_residual stay zero (kept for the summary format).
    double t_closure = 0, t_weno = 0, t_stabilization = 0, t_residual = 0, t_boundary = 0;
    double t_mass = 0, t_limiter = 0, t_finish = 0;
    double t_closure_setup = 0; // allocation and zeroing part of t_closure
    double t_rate = 0;    // wall time inside rate() (all phases plus anything untimed)
    double t_advance = 0; // wall time of the advance() calls (set by the driver)
    double t_finite = 0, t_ssp_stage = 0; // finiteness check of rate(); SSP stage combination + cone rounding fix
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
    // Dynamic adaptive PN diagnostics (experiment/dynamic-adaptive-pn).
    long adaptive_updates = 0;
    long adaptive_steps = 0;
    double adaptive_coeff_sum = 0;
    double adaptive_order_sum = 0;
    double adaptive_last_fraction = 1;
    double adaptive_last_mean_order = 0;
    int adaptive_last_max_order = 0;
    double t_adaptive_sensor = 0;
};
struct ElementWenoDebug {
    Field unlimited_rate, ho_moment_rate; // N x 4
};
class ElementWenoMpi;
class ElementWenoGpu;
// Hooks of a distributed (MPI) run, implemented by ElementWenoMpi. The scheme then lives on a slab of the global mesh
// plus halo layers along `axis` (the last axis); `owned` marks the nodes this rank owns.
struct DistributedHooks {
    int axis = -1;
    long node_offset = 0; // global id of local node 0 (local ids are global ids shifted along the last axis)
    long global_nodes = 0;
    virtual ~DistributedHooks() = default;
    // Global line solve with the 1D mass along `axis`, in place on the owned rows of b (same arithmetic as serial).
    virtual void solve_axis(Field &b) const = 0;
    // Halo rows of f (rows = local nodes, any column count) from their owners.
    virtual void exchange(Field &f) const = 0;
    // solve_axis while `count` independent pieces of work filler(0..count-1) run on otherwise idle threads.
    virtual void solve_axis_overlap(Field &b, int count, const std::function<void(int)> &filler) const {
        for (int i = 0; i < count; ++i)
            filler(i);
        solve_axis(b);
    }
    virtual double sum(double local) const = 0;
    virtual double min(double local) const = 0;
    // Sum of the owned values of all ranks, sequentially in rank order (ranks own contiguous layers in global order):
    // the result does not depend on the rank count.
    virtual double ordered_sum(const std::vector<double> &owned) const = 0;
};
class ElementWeno {
    friend class ElementWenoMpi;
    friend class ElementWenoGpu; // device time loop, src/element_weno_gpu.cpp (experimental)
    friend struct ElementWenoAngularCache;
public:
    // Distributed run (null: serial). owned: one flag per local node.
    void set_distributed(const DistributedHooks *hooks, std::vector<char> owned);
    const std::vector<char> &owned() const { return owned_; }
    bool is_owned(int i) const { return owned_.empty() || owned_[i]; }
    // MPI: an element is owned by the rank that owns its lowest node along the slab axis (local node 0).
    bool element_owned(int e) const { return owned_.empty() || owned_[connectivity_[std::size_t(e) * local_nodes]]; }
    int fcs_points_per_element() const { return fcs_per_element_; }
    Vec node_dt_bound; // lumped / rates per node (dt_bound is their minimum)
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
    // One SSPRK(4,3) stage of advance_ssprk43 for drivers that step several schemes in lockstep (multi-block, serial):
    // out = old a + fresh (x + dt/2 rate(t, x)), then the admissible-row nudge; `weight` is the stage weight of the
    // boundary outflow ledger. advance_ssprk43 = react(dt/2), stages (t,0,1,1/6) (t+dt/2,0,1,1/6) (t+dt,2/3,1/3,1/6)
    // (t+dt/2,0,1,1/2), react(dt/2).
    Field ssprk43_stage(double t, double dt, const Field &x, const Field &a, double old, double fresh, double weight,
                        ElementWenoStats * = nullptr) const;
    bool reacts_public() const { return reacts(); }
    // One step of the integrator selected by options.ssprk43.
    Field advance(double time, double dt, const Field &a, ElementWenoStats *stats = nullptr) const {
        if (options.ssp_n >= 3)
            return advance_ssp_n3(time, dt, a, stats);
        return options.ssprk43 ? advance_ssprk43(time, dt, a, stats) : advance_ssprk3(time, dt, a, stats);
    }
    // SSP coefficient of the integrator: every stage is a forward Euler step of size dt / ssp_coefficient().
    double ssp_coefficient() const {
        return options.ssp_n >= 3 ? double(options.ssp_n * options.ssp_n - options.ssp_n) : (options.ssprk43 ? 2. : 1.);
    }
    // SSPRK(n^2,3) stage plan: per stage whether it is the combination stage, its time (fraction of dt), and its weight
    // in u^{n+1} = u^n + dt sum_k weight_k L_k (boundary outflow ledger). ym is saved after `saved_after` stages.
    struct SspStage {
        bool combination;
        double time, weight;
    };
    struct SspPlan {
        int n = 0, stages = 0, saved_after = 0;
        double tau_fraction = 0, wa = 0, wb = 0; // tau = dt * tau_fraction; combination: wa ym + wb (x + tau L(x))
        std::vector<SspStage> stage;
    };
    static SspPlan ssp_plan(int n);
    Field advance_ssp_n3(double time, double dt, const Field &a, ElementWenoStats * = nullptr) const;
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
    // Power iteration behind high_order_step (60 iterations, random start). Dynamic AMR: warm start from the transferred
    // dominant vectors of the previous mesh (`start`, gamma = 0 and 1) with fewer iterations; `end` receives the vectors.
    double power_radius(int iterations, const std::array<Field, 2> *start, std::array<Field, 2> *end) const;
    // Step cache (driver option --step-cache): the norm of one application of linear_operator to a fixed vector
    // identifies the operator (distributed: global norm, collective); set_spectral_radius installs a stored estimate
    // of the same operator, so high_order_step() returns the same value without the power iteration.
    double operator_probe() const;
    // Euclidean norm of f in a fixed summation order: rows summed sequentially per layer along the last axis, layer sums
    // added in global layer order. Bitwise independent of the MPI rank count and the thread count (used by the
    // spectral estimate of high_order_step(), so that dt does not depend on the decomposition).
    double layered_norm(const Field &f) const;
    void set_spectral_radius(double radius) const { spectral_radius_ = radius; }
    double spectral_radius() const { high_order_step(); return spectral_radius_; }
    // Step limit used by the drivers: min(realizability bound, high_order_step).
    double step_bound() const { return std::min(realizability_step(), high_order_step()); }
    // M^-1(-C.A - S_gamma) a with constant gamma and without boundary terms (the operator of high_order_step).
    Field linear_operator(const Field &x, double gamma) const;
    Field stabilization_load(const Field &a, const Vec &gamma, ElementWenoStats * = nullptr) const; // s_h^e(phi_i, a) per element
    Field solve_mass(Field rhs) const;
    Field gradient(const Field &a, int direction) const { return projected_gradient(a, direction); } // M^-1 C_d a
    // Consistent mass times x (tensor product of the 1D Bernstein masses), any number of columns, no mirror sides.
    Field apply_mass(const Field &x) const;
    Field interpolate(const std::function<Vec(const V3 &)> &) const;
    Field project_l2(const std::function<Vec(const V3 &)> &) const;
    Vec evaluate(const Field &a, const V3 &x) const;
    // Multi-block setup: the node at position x of the element that contains `probe` (closest local node), and the
    // nodes of all interface faces as (side, element, node).
    int node_near(const V3 &x, const V3 &probe) const;
    void for_each_interface_node(const std::function<void(int, int, int)> &visit) const;
    V3 element_centre(int e) const;
    V3 node_position(int node) const;
    // Diagnostics of the ghost estimators (requires positivity_projection in full mode, i.e. the full-sphere rule):
    // for every ghost pair (boundary node, side) the outgoing half moments of
    //   0: C6 (positive part, cone scaling if a_ext > b), 1: full projection, 2: half-range projection (C6 if
    //   unsolvable), 3: hybrid (C6 if admissible, else full), 4: C6 of the filtered trace (filter strength `alpha`).
    // Cold projections, serial; independent of the scheme options except the rule.
    static constexpr int ghost_estimators = 5;
    struct GhostEstimates {
        std::vector<int> node, side;
        std::vector<std::array<Four, ghost_estimators>> m;
    };
    GhostEstimates ghost_estimates(const Field &a, double alpha) const;
    V3 side_normal(int side) const { return boundaries_[side]->n; }
    double l2_density_error(const Field &a, const std::function<double(const V3 &)> &exact) const;
    // FCS quadrature points and weights (empty without uncollided_density).
    const std::vector<V3> &fcs_points() const { return fcs_points_; }
    const std::vector<double> &fcs_weights() const { return fcs_weight_; }
    // Dynamic tile-based adaptive PN (CPU path): adapt tile orders from `a`, zeroing shrunken entries.
    // Called automatically at the start of a run and every options.adaptive_stride steps by drivers or tests.
    void adapt_pn_state(Field &a, bool initial = false, ElementWenoStats *stats = nullptr) const;
    void reset_adaptive_pn() const;
    const AdaptivePnPlan *adaptive_plan() const { return adaptive_plan_.get(); }
    // Active moment c is odd under x_d -> -x_d (dynamic AMR: such moments vanish on a mirror plane).
    bool reflect_odd(int d, int c) const { return d < 3 && !reflect_odd_[d].empty() && reflect_odd_[d][c]; }

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
    std::shared_ptr<CanonicalC1> canonical_;
    std::array<std::shared_ptr<Boundary>, 6> boundaries_;
    bool angular_cached_ = false; // angular data copied from options.angular_cache
    std::string angular_key() const;
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
    // split_breaks: the broken-line face rows summed per node apart from the domain-boundary rows.
    struct BreakAggregates {
        Field high4, low, bar, flux_area; // nodes x 4, 4, 4, 3 (projected PN face load, sum w (state - u), sum w state)
        Vec area;
    };
    void add_boundary(double time, const Field &a, const Field &low, const Field &low_flux,
                      Field &high, Field &low_load, Field &node_flux_area, Vec &area,
                      Field &bar, Vec *inflow_rate = nullptr, PositivityProjectionStats *projection = nullptr,
                      BreakAggregates *breaks = nullptr) const;
    // Inputs of the limited low-order part of rate() (all node-indexed except boundary_high, which has one row per
    // boundary slot and at least the four low coefficient columns).
    struct LowOrderInputs {
        const Field *low, *low_flux, *closure, *moment_rate, *stabilization_low, *boundary_high;
        const Field *boundary_low, *flux_area, *boundary_bar;
        const Vec *boundary_area, *boundary_inflow, *gamma;
        const Field *break_high4 = nullptr; // split_breaks: projected PN load of the broken-line rows (subtracted)
    };
    Field limited_low_load(const LowOrderInputs &, ElementWenoStats *, Field *raw_out) const;
    // split_breaks: per node the unsplit broken-line face term T (4 moments) and its area A from the last rate();
    // apply_split_breaks turns the rate rows of the broken-line nodes into the two-step update (see split_breaks).
    mutable Field split_term_;
    mutable Vec split_area_;
    std::vector<int> split_nodes_; // nodes on broken-line faces
    // split_breaks: broken-line faces, per split node its face rows (CSR, ascending face index), and persistent
    // work arrays of split_second_step (only the rows/nodes of the broken lines are written and read).
    std::vector<int> split_faces_, split_row_offsets_, split_rows_;
    mutable Field split_ylow_, split_yflux_, split_L_, split_BR_, split_FA_;
    mutable Vec split_AR_, split_IN_;
    void apply_split_breaks(const Field &x, Field &r, double tau, ElementWenoStats *stats) const;
    // Core of apply_split_breaks for split node k (split_nodes_ order): u[k] the stage state, rate[k] the low-order
    // rate of step 1 (load / m) on input, the rate of the two-step update (z - u) / tau on output. term/area: T and
    // A per node (node-indexed). Used by the CPU stage and the GPU host phase.
    void split_second_step(const std::vector<Four> &u, std::vector<Four> &rate, const Field &term, const Vec &area,
                           double tau, ElementWenoStats *stats) const;
public:
    // Split stage in two halves for multi-block drivers: split_compute_y (step 1 state y of all split nodes, published
    // for the partners) and split_apply (second step); split_second_step = both. split_y_state: y and F(y) n of a node.
    void split_compute_y(const std::vector<Four> &u, const std::vector<Four> &rate, double tau) const;
    void split_apply(const std::vector<Four> &u, std::vector<Four> &rate, const Field &term, const Vec &area, double tau,
                     ElementWenoStats *stats) const;
    void split_y_state(int node, const V3 &normal, Four &y, Four &flux_n) const;
    // apply_split_breaks in two halves (prepare: u, step-1 rate and y of the split nodes; finish: second step, rows).
    void split_prepare(const Field &x, const Field &r, double tau) const;
    void split_finish(Field &r, double tau, ElementWenoStats *stats) const;
    bool split_active() const { return split_active_; }
    // CPU stage halves (ssprk43_stage = a then b): a evaluates the rate and the step-1 states, b the second step,
    // the stage combination and the admissible rows.
    Field ssprk43_stage_a(double t, double dt, const Field &x, ElementWenoStats * = nullptr) const;
    Field ssprk43_stage_b(double t, double dt, const Field &x, const Field &a, Field r, double old, double fresh,
                          double weight, ElementWenoStats * = nullptr) const;
private:
    bool split_active_ = false;
    bool split_face(std::size_t fi) const {
        return face_break_[fi] || nodal_interface(2 * faces_[fi].direction + faces_[fi].side);
    }
    bool split_row(int r) const {
        const int side = face_row_side_[r];
        return r >= break_row_start_[side] || nodal_interface(side);
    }
    mutable std::vector<Four> split_y_of_, split_u_, split_rate_; // between the two stage halves
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
        Vec c, cinv;             // objective weights per point (empty: unweighted)
    };
    std::array<OutRule, 6> out_rules_;
    // Separable trace evaluation per side (src/separable_trace.hpp; null: dense out_rules_[side].harmonics) and the
    // ghost pairs of each side.
    std::array<std::shared_ptr<const SeparableTrace>, 6> separable_;
    // Adaptive PN: low-order quadrature pyramid (e.g. N=3, 7, 15 with rule_n = N+1) so low-order boundary nodes
    // project on 144/400/1296 points instead of the full PN rule (e.g. 5776 points at P35).
    struct AdaptiveGhostTier {
        int max_order = 0;
        int q_tier = 0;
        std::array<OutRule, 6> out_rules;
        std::array<std::shared_ptr<const SeparableTrace>, 6> separable;
    };
    std::vector<AdaptiveGhostTier> adaptive_ghost_tiers_;
    std::array<std::vector<int>, 6> ghost_by_side_;
    // Face-row layout of add_boundary (see the constructor).
    std::vector<int> face_row_, face_row_side_, boundary_row_offsets_, boundary_rows_;
    std::array<int, 7> side_row_start_{};
    // Rows of the faces on broken lines: side_row_start_[s] <= break_row_start_[s] <= side_row_start_[s + 1]
    // (domain faces first; equal to side_row_start_[s + 1] without broken lines). Their face matrix .5 (A_n - I)^T is
    // sparse (interface_sparse_t_).
    std::array<int, 6> break_row_start_{};
    // Partner node across a broken line of face `index` for its face node n (conforming: same position).
    int broken_partner_node(int index, int n) const;
    // Low-order face terms of a broken-line face (rows face_row_[index] .. + nf of L, BR, FA, AR, IN, zeroed by the
    // caller): bar states node by node with the lumped face weights and the partner's nodal state and flux (shared by
    // add_boundary and the GPU host path).
    void broken_face_low(int index, const Field &low, const Field &low_flux, Field &L, Field &BR, Field &FA, Vec &AR,
                         Vec &IN) const;
    // The same for a conforming interface face (partner data from interface_partner / interface_low).
    void interface_face_low(int index, const Field &low, const Field &low_flux, Field &L, Field &BR, Field &FA, Vec &AR,
                            Vec &IN) const;
    // V rows (face mass times the partner trace) of a broken-line or conforming interface face.
    void nodal_partner_rows(int index, const Field &a, Eigen::Ref<Field> rows, int q_cols = -1) const;
    // Populate tile_floor for non-vacuum inflow walls and per-side max active moments for boundary_high.
    void init_adaptive_inflow_floor(AdaptivePnPlan &plan) const;
    void update_adaptive_side_q(AdaptivePnPlan &plan) const;
    bool nodal_interface(int side_id) const { return options.interface_side[side_id] && bool(options.interface_partner); }
    void nodal_face_low(int index, const Field &low, const Field &low_flux,
                        const std::function<void(int, Four &, Four &)> &partner, Field &L, Field &BR, Field &FA, Vec &AR,
                        Vec &IN) const;
    std::array<Sparse, 6> interface_sparse_t_;
    // Positivity projection warm start per ghost pair (multiplier and rule of the last accepted projection). Only a
    // starting guess: the projection is unique, so the result does not depend on it beyond the solver tolerance.
    mutable std::vector<PositivityProjectionWarm> projection_warm_;
    Vec ghost_sigma_; // per active moment: ghost trace filter factor (empty: no filter)
    std::array<Field, 6> out_harmonics_; // row-major copies of out_rules_[side].harmonics (batched trace products)
    // Kinetic inflow data at a face quadrature point (one code path for the steady cache and the per-stage call).
    Incoming face_data(int side_id, const V3 &x, double time) const;
    void face_data_into(int side_id, const V3 &x, double time, Incoming &data) const;
    std::vector<Incoming> steady_data_; // per (face, point) when inflow.steady
    // Uniform faces (steady inflow without exterior callbacks, all point data bitwise equal, e.g. vacuum): the
    // low-order face terms and the PN datum load are point-independent per face node, so the point loop reduces to
    // the face weights W_r = sum_points w_p phi_n(x_p) (row r = face_row_[f] + n). Only the summation order changes.
    std::vector<char> face_uniform_;
    Vec face_weight_;
    std::vector<int> boundary_nodes_sorted_;
    struct PairEntry {
        int j;
        double d, m, c[3], ct[3];
    };
    // Element data per size class (docs/GRADED_MESH.md): one class on the uniform mesh (copies of the members
    // above), one per occurring width triple on a graded mesh. element_class_ is empty on the uniform mesh.
    struct ElementClass {
        V3 width = V3::Ones();
        double viscosity = 0;
        Mat mass, stiffness, viscosity_ij;
        std::array<Mat, 3> convection;
        std::vector<PairEntry> pairs;
        Vec total_viscosity, mass_row;
        std::vector<double> stiffness_rows;              // sum_d f_d K_d (graded: f_d = w_d / w_min, else 1)
        std::array<std::vector<double>, 3> convection_t_rows; // f_d C_d^T, stabilization only
    };
    std::vector<ElementClass> classes_;
    std::vector<int> element_class_;
    const ElementClass &klass(int e) const { return element_class_.empty() ? classes_[0] : classes_[element_class_[e]]; }
    bool graded_ = false;
    // Graded mesh: element widths per axis (global element index along the axis); per (element, 2d + side) the index
    // of the WENO extension matrix in extension_list_ (-1: none).
    std::array<std::vector<double>, 3> axis_width_;
    std::vector<Mat> extension_list_;
    std::vector<int> extension_index_;
    std::vector<double> face_scale_; // per boundary face: physical area / rule area (1 on the uniform mesh)
    // Width of element e along axis d (h on the uniform mesh) and its volume (h^d on the uniform mesh).
    double width(int e, int d) const {
        if (!graded_)
            return h;
        int id = e;
        for (int k = 0; k < d; ++k)
            id /= count[k];
        return axis_width_[d][id % count[d]];
    }
    double volume(int e) const {
        double v = 1;
        for (int d = 0; d < dimension; ++d)
            v *= width(e, d);
        return v; // only used on graded meshes (the uniform branches keep std::pow(h, d))
    }
    std::vector<PairEntry> pair_table_; // limiter pairs (i, j != i), row-major in i // ascending, for the deterministic parallel boundary corrections
    Field gather_load(const Field &local, double scale = 1.) const;
    // M^-1 C_d a along the grid lines of direction d, and the element loads of s_h for given gradients.
    Field projected_gradient(const Field &a, int direction) const;
    // One grid line of M^-1 C_d a (natural factor; the body of projected_gradient without mirror sides).
    void gradient_line(const Field &a, int direction, int line, Field &out) const;
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
    bool broken_ = false;                         // options.breaks non-empty
    std::array<std::vector<int>, 3> break_offset_; // per axis and element: number of broken lines k <= element
    std::array<std::vector<double>, 3> node_coord_; // per axis and 1D node (broken meshes)
    std::vector<char> face_break_;                  // per face: on a broken line (partner element across)
    std::array<Mat, 6> interface_t_;     // .5 (A_n - I)^T on interface sides (neighbour trace term)
    std::array<Mat, 6> exterior_t_;      // (llf_homogeneous - (A_n + I)/2)^T (projection_high_order)
    // Boundary nodes (compact row index of the PN boundary load) and the distinct (side, node) pairs of the
    // C6 ghost state; ghost_of_face_ maps (face, face node) to such a pair.
    std::vector<int> boundary_slot_, boundary_nodes_, ghost_node_, ghost_side_, ghost_of_face_;
    std::vector<char> vacuum_limit_node_; // per node: on a domain boundary face (C2 applies; not on interfaces only)
    mutable double spectral_radius_ = -1;
    const DistributedHooks *dist_ = nullptr;
    std::vector<char> owned_;
    void distributed_random(Field &x, std::mt19937 &rng, std::uniform_real_distribution<double> &u) const;
    Mat m1_reference_; // 1D reference element mass (the global line factor of the MPI layer is assembled from it)
    // Dynamic tile-based adaptive PN (CPU): tile plan and natural-order banded 1D LDL^T factors for segmented line solves.
    mutable std::unique_ptr<AdaptivePnPlan> adaptive_plan_;
    mutable bool adaptive_active_ = false;
    mutable long adaptive_step_counter_ = 0;
    struct BandFactor {
        int n = 0, p = 0;
        std::vector<double> lower, invdiag;
    };
    std::array<BandFactor, 3> band_natural_, band_dirichlet_;
    void build_band_factors();
    void solve_band_segment(const BandFactor &f, double *buf, int width, int k0, int k1) const;
    void adaptive_solve_lines(Field &b, int d, bool gradient, bool is_grad_segs) const;
    Field adaptive_projected_gradient(const Field &a, int d) const;
};
// PN moments of the reflected distribution psi(R_d omega): moments odd under x_d -> -x_d change sign.
Vec reflect_pn(const Angular &, const Vec &full, int direction);
// Exact PN solution for the periodic density sine wave of the accuracy study.
Vec element_weno_smooth_exact(const Angular &, const V3 &x, double t);
} // namespace m1pn
