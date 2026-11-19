#pragma once
// Device side of the GPU time loop (src/element_weno_gpu.cu). Plain C++ (no Eigen, no CUDA headers), so the host
// translation unit src/element_weno_gpu.cpp can include it. All fields are row-major (rows x columns) device arrays.
#include <cstddef>
#include "uncollided_lattice_core.hpp"
#include "uncollided_lattice3d_core.hpp"
namespace m1pn {
namespace gpu {
constexpr int max_local = 27; // local nodes per element (Q1/Q2 in 3D, up to Q4 in 2D)
// Constant scheme data (uploaded once).
struct Setup {
    int dimension = 0, degree = 0, nodes = 0, cells = 0, local = 0, q = 0;
    int nodes_per_axis[3] = {1, 1, 1}, count[3] = {1, 1, 1}, stride[3] = {1, 1, 1};
    double viscosity = 0;
    // Element tables (host arrays, copied to the device by create()).
    const int *connectivity = nullptr; // cells x local
    const int *neighbors = nullptr;    // cells x 2 dimension (-1: none)
    const int *node_offsets = nullptr; // nodes + 1
    const int *node_entries = nullptr; // cells * local, entry = e * local + l
    // Graded meshes (docs/GRADED_MESH.md): `classes` size classes; the per-class tables below are stacked (class c
    // at offset c * size), element_class[e] selects one (null: one class, the uniform mesh), class_viscosity[c] the
    // stabilization viscosity (null: `viscosity`). WENO extensions per (element, axis, side): ext_list (ext_count x
    // local x local) indexed by ext_index[e * 2 dimension + 2 d + side] (null: extension[d][side]).
    int classes = 1;
    bool structured = true; // node grid = element grid (false with broken lines)
    // Broken lines, 2D Q1 strip kernel: per axis d and node line k, strip_elem[d][2 k + side] = element line on the
    // left (side 0) / right (side 1) of node line k that contains it, -1 if none (a node copy on a broken line has
    // only one side). nullptr: structured mesh (element lines k - 1 and k).
    const int *strip_elem[2] = {};
    const int *element_class = nullptr;
    const double *class_viscosity = nullptr;
    int ext_count = 0;
    const double *ext_list = nullptr;
    const int *ext_index = nullptr;
    const double *stiffness_rows = nullptr;      // classes x local x local
    const double *convection_t_rows[3] = {};     // classes x local x local each
    const double *seminorm = nullptr;            // local x local, row-major
    const double *extension[3][2] = {};          // local x local, row-major
    // Banded 1D operators per axis (bandwidth = degree): convection rows (n x (2p+1), entry k-p..k+p, row-major),
    // natural-order LDL^T of the 1D mass: lower[k * p + (m - 1)] = L(k, k - m), diagonal[k] = D(k).
    const double *convection_band[3] = {}, *lower[3] = {}, *diagonal[3] = {};
    // Mirror sides (ElementWenoOptions::mirror): mirror_index[d] = grid-line position of the mirror node (-1: none);
    // dlower/ddiagonal: banded LDL^T of the 1D mass without the mirror row and column (n - 1 rows, same layout);
    // comp_odd[c]: bit d set if active moment c is odd under x_d -> -x_d; node_mirror[i]: bit d set if node i lies
    // on the mirror plane of direction d (nullptr without mirror sides); reflected[d]: local node mirrored in d.
    int mirror_index[3] = {-1, -1, -1};
    // Periodic mesh: the natural-order LDL^T of the cyclic 1D mass is banded except for its last `degree` rows,
    // which are dense: tail[d][r * n + j] = L(n - degree + r, j) (j < n - degree + r); lower holds the band rows
    // k < n - degree. convection_band then uses cyclic column offsets.
    bool periodic = false;
    const double *tail[3] = {};
    const double *dlower[3] = {}, *ddiagonal[3] = {};
    const unsigned char *comp_odd = nullptr, *node_mirror = nullptr;
    const int *reflected[3] = {};
    // A_d^T (active moments) in CSR, q x q.
    const int *flux_outer[3] = {}, *flux_inner[3] = {};
    const double *flux_value[3] = {};
    int flux_nnz[3] = {0, 0, 0};
    // Limiter pair table: classes x local x (local - 1) entries of (j, d, m, c[3], ct[3]).
    const int *pair_j = nullptr;
    const double *pair_d = nullptr, *pair_m = nullptr, *pair_c = nullptr, *pair_ct = nullptr; // c, ct: 3 per entry
    const double *total_viscosity = nullptr; // classes x local
    const double *lumped = nullptr;          // nodes
    // Options.
    bool weno = true, stabilize = true, limit = true, outward_bar_rounding = true, scale_invariant_weno = true;
    double weno_q = 1, weno_r = 2, weno_epsilon = 1e-6, weno_linear = 1e-3;
    double element_theta = .999;
    int fallback_capacity = 1 << 16;
};
// Device work arrays and counters; created by create(), freed by destroy().
struct Context;
Context *create(const Setup &);
void destroy(Context *);
// Raw device buffers of the context (row-major).
struct Buffers {
    double *a, *x, *r;                       // three state-sized buffers (state, stage buffers)
    double *low, *low_flux, *closure;        // nodes x 4, 12, 12
    double *gradient[3];                     // nodes x q
    double *stab_low;                        // cells x 4 local
    double *gamma;                           // cells
    double *moment_rate;                     // nodes x 4
    double *element_limited;                 // cells x 4 local
    double *load;                            // nodes x 4
};
Buffers buffers(Context *);
// Synchronous upload / download of a whole device array (bytes).
void upload(void *device, const void *host, std::size_t bytes);
void download(void *host, const void *device, std::size_t bytes);
void synchronize();
void synchronize(Context *); // the context stream only (other contexts keep running)
// Row gathers/scatters through a device index list (synchronous transfers of the compact rows).
// rows_soa > 0: the field is column-major with rows_soa rows (low, low_flux, closure, moment_rate, stab_low,
// element_limited); 0: row-major.
void gather_device(Context *, const double *field, int cols, const int *device_index, int count, double *host_out,
                   long long rows_soa = 0);
void scatter_device(Context *, double *field, int cols, const int *device_index, int count, const double *host_in,
                    bool add, long long rows_soa = 0);
// Pinned host memory.
void *host_alloc(std::size_t bytes);
void host_free(void *);
// Kernels of one rate evaluation. Each returns after queuing the work (stream order).
// Fallback lists: closure nodes (near-beam Levermore states, ambiguous fast cone tests out of range) and limiter
// elements (stable base bar in long double), filled by the kernels and read with fallbacks().
void closure(Context *, const double *state);
void weno_gamma(Context *, const double *state);
void gradients(Context *, const double *state);
// out = -s_h gathered; stab_low = projected element loads (when want_low). constant_gamma >= 0 replaces gamma.
void stabilization(Context *, const double *state, double *out, bool want_low, double constant_gamma);
void add_rows(Context *, double *field, const int *device_index, const double *device_rows, int count); // field[i] += rows
void mass_solve(Context *, double *field);
// mass_solve with the PN-row stage combination (rows c >= 4, as combine_pn) fused into the last direction; returns false
// (and only solves) where the fusion does not apply, then the caller runs combine_pn. M1PN_GPU_FUSE=0: never fuse.
bool mass_solve_combine(Context *, double *field, const double *x, const double *a, double old, double fresh, double tau);
// field += -sum_d A_d^T g_d per node, then moment_rate = project(field) (when want_moments).
void flux_rows(Context *, double *field, bool want_moments);
void limiter(Context *);
void gather_load(Context *);
// moment_rate = project(field - sum_d A_d^T g_d) (four low rows only; field unchanged).
void flux_moments(Context *, const double *field);
// r <- old a + fresh (x + tau (r - sum_d A_d^T g_d)) with the low rows of the rate replaced by load / lumped; then
// the admissible-row nudge.
// pn_done: the PN rows (c >= 4) were combined already by combine_pn; closure_next: also the closure of the next stage
// (input r) in the same pass.
void finish_stage(Context *, double *r, const double *x, const double *a, double old, double fresh, double tau,
                  bool pn_done = false, bool closure_next = false);
void combine_pn(Context *, double *r, const double *x, const double *a, double old, double fresh, double tau);
double norm(Context *, const double *field);
void divide(Context *, double *dst, const double *src, double divisor); // dst = src / divisor
// Counters since the last reset (synchronizes).
struct Counters {
    int error = 0;                 // nonzero: a cone requirement failed on the device (code)
    int closure_fallbacks = 0, limiter_fallbacks = 0;
    long long limited_elements = 0, nudged_bars = 0, outward_bars = 0, nudged_rows = 0;
    double min_limiter = 1, max_raw_zero_sum = 0, max_limited_zero_sum = 0, nudge_mass = 0, min_gamma = 1;
};
Counters counters(Context *);
void reset_counters(Context *);
// Asynchronous pipeline (default stream): pinned host buffers, events, a counter snapshot.
void gather_async(Context *, const double *field, int cols, const int *device_index, int count, double *pinned_out,
                  int slot, long long rows_soa = 0);
void scatter_async(Context *, double *field, int cols, const int *device_index, int count, const double *pinned_in,
                   bool add, int slot, long long rows_soa = 0);
void upload_async(void *device, const void *pinned, std::size_t bytes);
void record(Context *, int event);
void wait(Context *, int event);       // host waits for the event
void snapshot_counters(Context *);     // queue a copy of the counters into pinned memory
Counters snapshot(Context *);          // the last queued snapshot (call after waiting for a later event)
void reset_fallbacks(Context *);       // queue zeroing of the two fallback counters
// Ghost moments with the positivity projection on the device (one warp per ghost pair, src/element_weno_gpu.cu).
// Plain projections only: pairs that need the hull certificate / augmented rule / host retry are flagged and done by
// the host with project_positive_moments.
struct GhostSetup {
    int pairs = 0, max_points = 0;
    const int *node = nullptr, *side = nullptr; // per pair
    int points[6] = {}, outgoing[6] = {}, facets[6] = {};
    const double *harmonics[6] = {};             // points x q, row-major
    const double *ox[6] = {}, *oy[6] = {}, *oz[6] = {}, *w[6] = {};
    const double *fnx[6] = {}, *fny[6] = {}, *fnz[6] = {}, *fh[6] = {};
    double inradius[6] = {};
    const double *normal[6] = {};                // 3 per side
    const double *sigma = nullptr;               // ghost trace filter per moment (null: none)
    double tolerance = 1e-12;
    bool warm_start = true;
};
struct GhostStats {
    long long calls = 0, identities = 0, vacuum = 0, failures = 0, scaled = 0, capped = 0, iterations = 0,
              backtracks = 0, flagged = 0;
    int max_iterations = 0;
    double max_residual = 0, min_active_fraction = 1;
};
void ghost_setup(Context *, const GhostSetup &);
// Queue the ghost kernel for stage input `state` and the asynchronous copies of the outgoing moments (pairs x 4), the
// flag count and the flagged pair list into pinned host memory.
void ghost(Context *, const double *state, double *pinned_moments, int *pinned_flags);
GhostStats ghost_stats(Context *); // cumulative (synchronous)
// Ghost traces on the device, projections on the host (docs/STATIC_HP_BLOCKS.md): per side s the trace matrix
// harmonics_s (points_s x q, column-major, ghost filter applied) times the stage input rows of the side's ghost nodes.
struct GhostTraceSetup {
    int q = 0;
    int count[6] = {}, points[6] = {};
    const int *nodes[6] = {};        // ghost nodes of side s, in ElementWeno::ghost_by_side_ order (host)
    const double *harmonics[6] = {}; // points_s x q, column-major (host)
};
void ghost_traces_setup(Context *, const GhostTraceSetup &);
// On the aux stream after the work queued so far on the default stream: the traces psi (points_s per ghost node,
// sides in order 0..5, nodes in setup order) of `state` into pinned host memory; completion is event 2.
void ghost_traces(Context *, const double *state, double *pinned_out);
// Exact local reaction (ElementWeno::react): per node rho' = e^{-sa tau} rho + q0 tau phi1(-sa tau),
// a_k' = e^{-tau (sa + ss + f_k)} a_k (k >= 1), q0 = source + FCS collision source; then the admissible-row nudge.
struct ReactionSetup {
    const double *sigma_a = nullptr, *sigma_s = nullptr, *source = nullptr; // nodes (material) or null
    double sa = 0, ss = 0;                                                  // constant coefficients (no material)
    const double *filter = nullptr;                                         // per moment: f (l / (N+1))^4
    // FCS (2D line source): quadrature points (cells x P), weights, sigma_s, shape (local x P, row-major), node mass.
    int fcs_points = 0;
    const double *fcs_x = nullptr, *fcs_y = nullptr, *fcs_weight = nullptr, *fcs_sigma_s = nullptr;
    const double *fcs_shape = nullptr, *fcs_node_mass = nullptr;
};
void reaction_setup(Context *, const ReactionSetup &);
// table: rho_u at r = (k - 1) dr, k = 0..n (cubic interpolation as uncollided::GaussianPulse::density); null: no FCS.
void react(Context *, double *state, double tau, const double *host_table, int n, double dr, double cx, double cy);
struct Ledger {
    double absorbed = 0, injected = 0;
};
Ledger ledger(Context *); // cumulative (synchronous)
// Lattice FCS (docs/FCS_LATTICE.md): rho_u at the FCS points (cells x fcs_points, the order of ReactionSetup) on
// t_m = m dt, m = 0..M, built on the device with uncollided::lattice_density (view with host arrays; copied);
// host_out (points x (M + 1)) receives a copy. react_lattice() then interpolates it at source_time.
void lattice_table(Context *, const uncollided::LatticeView &host_view, int M, double dt, double *host_out);
// lattice-3d: the same with uncollided::lattice3d_row, one thread per point (x, y, z: host arrays of the FCS points).
void lattice3d_table(Context *, const uncollided::Lattice3dView &host_view, const double *x, const double *y,
                     const double *z, int M, double dt, double *host_out);
void react_lattice(Context *, double *state, double tau, double source_time);
// Fallback lists (device index arrays; count from counters()).
const int *closure_fallback_list(Context *);
const int *limiter_fallback_list(Context *);
// Phase A of the boundary on the device (docs/ELEMENT_WENO_GPU.md, Block 4): PN face load
//   H_r = sum_m M_side(n, m) x_{node(f, m)} B_side - DB_r   (row r = face f, face node n; B_side = (A_n - llf)^T)
// by a trace kernel, one DGEMM per boundary side (cuBLAS) and the datum part; per boundary node k the rows of
// boundary_rows in ascending order are summed and added to `out` (as the host phase A). The first four columns of the
// node sums (l <= 1) go to the host for the boundary corrections.
struct BoundaryHighSetup {
    int q = 0, rows = 0, nf = 0, nb = 0;
    const int *row_nodes = nullptr;    // rows x nf: global nodes of the face of row r
    const double *row_mass = nullptr;  // rows x nf: M_side(n, m)
    int side_start[7] = {};            // rows of side s: side_start[s] .. side_start[s + 1]
    const double *flux_t[6] = {};      // q x q column-major (Eigen) boundary_flux_t_, null for sides without boundary
    // Broken lines: rows break_start[s] .. side_start[s + 1] of side s belong to broken-line faces; their row weights
    // (nf = twice the face nodes) give M_face (u - u_partner), multiplied by iface_t[s] = .5 (A_n - I)^T.
    int break_start[6] = {};
    const double *iface_t[6] = {};
    const int *offsets = nullptr;      // nb + 1
    const int *rows_list = nullptr;    // boundary_rows_
    const int *bnodes = nullptr;       // device: boundary node of slot k
    const int *row_break = nullptr;    // split_breaks (experiment): rows x {0, 1}, 1 = broken-line row; null: none
};
void boundary_high_setup(Context *, const BoundaryHighSetup &);
// pinned_db: rows x q row-major datum part (null: zero datum). pinned_low4: nb x 4 out (async, stream order).
// pinned_low4_break (split_breaks): nb x 4 out, the l <= 1 sums of the broken-line rows only.
void boundary_high(Context *, const double *state, double *out, const double *pinned_db, double *pinned_low4,
                   double *pinned_low4_break = nullptr);
// Device scratch for uploading small index/row lists.
int *device_ints(Context *, int slot, int count);
double *device_doubles(Context *, int slot, std::size_t count);
} // namespace gpu
} // namespace m1pn
