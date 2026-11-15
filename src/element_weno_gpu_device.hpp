#pragma once
// Device side of the GPU time loop (src/element_weno_gpu.cu). Plain C++ (no Eigen, no CUDA headers), so the host
// translation unit src/element_weno_gpu.cpp can include it. All fields are row-major (rows x columns) device arrays.
#include <cstddef>
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
    const double *stiffness_rows = nullptr;      // local x local
    const double *convection_t_rows[3] = {};     // local x local each
    const double *seminorm = nullptr;            // local x local, row-major
    const double *extension[3][2] = {};          // local x local, row-major
    // Banded 1D operators per axis (bandwidth = degree): convection rows (n x (2p+1), entry k-p..k+p, row-major),
    // natural-order LDL^T of the 1D mass: lower[k * p + (m - 1)] = L(k, k - m), diagonal[k] = D(k).
    const double *convection_band[3] = {}, *lower[3] = {}, *diagonal[3] = {};
    // A_d^T (active moments) in CSR, q x q.
    const int *flux_outer[3] = {}, *flux_inner[3] = {};
    const double *flux_value[3] = {};
    int flux_nnz[3] = {0, 0, 0};
    // Limiter pair table: local x (local - 1) entries of (j, d, m, c[3], ct[3]).
    const int *pair_j = nullptr;
    const double *pair_d = nullptr, *pair_m = nullptr, *pair_c = nullptr, *pair_ct = nullptr; // c, ct: 3 per entry
    const double *total_viscosity = nullptr; // local
    const double *lumped = nullptr;          // nodes
    // Options.
    bool weno = true, stabilize = true, limit = true, outward_bar_rounding = true, scale_invariant_weno = true;
    double weno_q = 1, weno_r = 2, weno_epsilon = 1e-6, weno_linear = 1e-3;
    double positivity_fraction = 1e-3, element_theta = .999;
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
// Fallback lists (device index arrays; count from counters()).
const int *closure_fallback_list(Context *);
const int *limiter_fallback_list(Context *);
// Device scratch for uploading small index/row lists.
int *device_ints(Context *, int slot, int count);
double *device_doubles(Context *, int slot, std::size_t count);
} // namespace gpu
} // namespace m1pn
