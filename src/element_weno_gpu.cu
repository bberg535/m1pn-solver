// GPU kernels of the element WENO time loop (experimental, docs/ELEMENT_WENO_GPU.md). One thread per output entry or
// per element; every output is computed by one thread in a fixed order, so a run is deterministic. The arithmetic
// follows src/element_weno.cpp; summation orders and fused multiply-adds differ (rounding level). The exact M1 cone
// predicate is evaluated exactly (error-free transformations); paths that need long double on the host (near-beam
// Levermore states, the stable base bar) are flagged and recomputed by the host with the original code.
#include "element_weno_gpu_device.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
namespace m1pn {
namespace gpu {
namespace {
void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(e));
}
#define CK(x) check((x), #x)
template <class T> T *dalloc(std::size_t count) {
    T *p = nullptr;
    if (count)
        CK(cudaMalloc(&p, count * sizeof(T)));
    if (count)
        CK(cudaMemset(p, 0, count * sizeof(T)));
    return p;
}
template <class T> T *dcopy(const T *host, std::size_t count) {
    T *p = dalloc<T>(count);
    if (count && host)
        CK(cudaMemcpy(p, host, count * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
constexpr int MAXP = 4;
} // namespace

struct DeviceCounters {
    int error;
    int closure_fallbacks, limiter_fallbacks;
    int segments; // elements queued for the limiter factor (k_limiter_segment)
    unsigned long long limited_elements, nudged_bars, outward_bars, nudged_rows;
    unsigned long long min_limiter_bits, max_raw_bits, max_limited_bits, min_gamma_bits;
    double nudge_mass;
};
// All device pointers of the scheme, passed by value to the kernels.
struct View {
    int dimension, degree, nodes, cells, local, q;
    int npa[3], count[3], stride[3];
    double viscosity, s0, s1; // sqrt(4 pi), sqrt(4 pi / 3)
    const int *connectivity, *neighbors, *node_offsets, *node_entries;
    const double *stiffness_rows, *convection_t_rows[3], *seminorm, *extension[3][2];
    const double *stiffness_nu;       // classes x local^2: viscosity_of(c) * stiffness_rows (3D Q1 strip kernel)
    const int *klass;                 // cells: size class of an element (nullptr: one class, uniform mesh)
    int classes;
    bool structured, strip3_ok;
    const int *strip_elem[3];         // broken Q1 (2D, 3D): element line left/right of a node line (see GpuSetup)
    // 2D Q1 strip kernel tables per size class (16 per class: nu K rows, C_0^T, C_1^T rows; one nu per class), per
    // context in global memory. They were in __constant__ memory, which is shared by all contexts: with several graded
    // blocks (m1pn_blocks) the last block's classes overwrote the others' (wrong results, cone failures).
    const double *strip_kv, *strip_c0, *strip_c1, *strip_nu;
    const double *class_viscosity;    // classes (nullptr: viscosity)
    const double *ext_list;           // graded WENO extensions (ext_count x local x local)
    const int *ext_index;             // cells x 2 dimension (nullptr: extension[d][side])
    const double *convection_band[3], *lower[3], *invdiag[3];
    int mirror_index[3];                       // -1: no mirror side in d
    const double *dlower[3], *dinvdiag[3];     // Dirichlet factor of the 1D mass (mirror directions)
    const unsigned char *comp_odd, *node_mirror; // q, nodes (node_mirror == nullptr: no mirror sides)
    const int *reflected[3];                   // local
    bool periodic;
    const double *tail[3];                     // periodic: P x n dense tail rows of the cyclic LDL^T
    const int *flux_outer[3], *flux_inner[3];
    const double *flux_value[3];
    const int *pair_j;
    const double *pair_d, *pair_m, *pair_c, *pair_ct, *total_viscosity, *lumped;
    bool weno, stabilize, limit, outward_bar_rounding, scale_invariant_weno;
    double weno_q, weno_r, weno_epsilon, weno_linear, element_theta;
    int fallback_capacity;
    // Work arrays.
    double *density; // nodes: copy of column 0 of the stage input (closure kernel), read by the WENO kernel
    // Column-major (SoA, entry (row, c) at c * rows + row): low, low_flux, closure, moment_rate (rows = nodes) and
    // stab_low, element_limited (rows = cells), so that the per-element kernels read and write coalesced.
    double *low, *low_flux, *closure, *gradient[3], *stab_low, *gamma, *moment_rate, *element_limited, *load;
    const double *ell_val; // dimension x q x flux_width
    const int *ell_idx;
    int flux_width;
    bool flux_in_shared; // ELL table staged in shared memory (fits the default 48 KiB); else read from global
    int *closure_list, *limiter_list, *segment_list;
    double *seg_ubar, *seg_lim; // cells x 4 local: bar states and contributions of queued elements
    DeviceCounters *counters;
};
struct Context {
    View v;
    Buffers b;
    std::vector<void *> owned;
    int *scratch_int[8] = {};
    std::size_t scratch_int_size[8] = {};
    double *scratch_double[8] = {};
    std::size_t scratch_double_size[8] = {};
    double *norm_accumulator = nullptr;
    cudaEvent_t events[8] = {};
    // Ghost projection (ghost_setup).
    struct GhostDevice *ghost = nullptr;
    cudaStream_t aux = nullptr; // non-blocking stream of the ghost kernel (overlaps the volume kernels)
    // Context stream of all kernels and asynchronous copies (experiment/big-rebuilds): one per scheme, so that several
    // schemes (multi-block driver) run concurrently. Blocking stream: operations on the legacy default stream that
    // remain (synchronous copies, upload_async) still order with it.
    cudaStream_t stream = nullptr;
    // Fork/join streams of the adaptive path: the low-moment (c < 4) line solves and stabilization and the per-axis
    // high-chunk line solves touch disjoint coefficients (or fields) and are latency bound with few threads each, so
    // they run concurrently (M1PN_GPU_OVERLAP=0: all on `stream`).
    cudaStream_t side[2] = {};
    cudaEvent_t fork_event = nullptr, join_event[2] = {};
    struct ReactionDevice *reaction = nullptr;
    struct BoundaryHighDevice *bhigh = nullptr;
    struct GhostTraceDevice *gtrace = nullptr;
    struct AdaptiveDevice *adaptive = nullptr;
    int flux_nnz[3] = {0, 0, 0};
    DeviceCounters *pinned_counters = nullptr;
};
struct ReactionView;
struct BoundaryHighDevice;
namespace {
void react_rows_adaptive_launch(Context *ctx, const ReactionView &r, double *state, double tau, bool fcs);
bool boundary_high_adaptive_launch(Context *ctx, BoundaryHighDevice *d, const double *state, double *out,
                                   const double *pinned_db, double *pinned_low4, double *pinned_low4_break);
} // namespace

// ---------------------------------------------------------------------------------------------------------------
// Exact M1 cone predicate for the stored binary64 values (same answer as exact::cone in src/cone.hpp): the fast filter
// of the host, then the exact sign of rho^2 - |J|^2 from error-free products and a Shewchuk expansion sum after an
// exact power-of-two scaling. Components that would lose bits in that scaling set `uncertain`.
__device__ inline double dinf() { return __longlong_as_double(0x7ff0000000000000LL); }
__device__ inline void two_sum(double a, double b, double &s, double &e) {
    s = __dadd_rn(a, b);
    const double bb = __dsub_rn(s, a);
    e = __dadd_rn(__dsub_rn(a, __dsub_rn(s, bb)), __dsub_rn(b, bb));
}
__device__ __noinline__ bool cone_exact(double r, double x, double y, double z, bool &uncertain) {
    double m = fmax(fmax(fabs(r), fabs(x)), fmax(fabs(y), fabs(z)));
    int ex;
    frexp(m, &ex);
    const double v[4] = {ldexp(r, -ex), ldexp(x, -ex), ldexp(y, -ex), ldexp(z, -ex)};
    for (int k = 0; k < 4; ++k)
        if (v[k] != 0 && fabs(v[k]) < 0x1p-450)
            uncertain = true;
    double terms[8];
    for (int k = 0; k < 4; ++k) {
        const double p = __dmul_rn(v[k], v[k]);
        const double e = fma(v[k], v[k], -p);
        terms[2 * k] = k ? -p : p;
        terms[2 * k + 1] = k ? -e : e;
    }
    double h[8];
    int n = 0;
    for (int t = 0; t < 8; ++t) {
        double Q = terms[t];
        int k = 0;
        for (int i = 0; i < n; ++i) {
            double s, e;
            two_sum(Q, h[i], s, e);
            if (e != 0)
                h[k++] = e;
            Q = s;
        }
        if (Q != 0)
            h[k++] = Q;
        n = k;
    }
    return n == 0 || h[n - 1] >= 0;
}
__device__ bool cone(const double *u, bool &uncertain) {
    for (int k = 0; k < 4; ++k)
        if (!isfinite(u[k]))
            return false;
    if (u[0] < 0)
        return false;
    if (u[0] == 0)
        return u[1] == 0 && u[2] == 0 && u[3] == 0;
    const double r = u[0] * u[0], x = u[1] * u[1], y = u[2] * u[2], z = u[3] * u[3];
    const double sum = ((r + x) + y) + z, difference = ((r - x) - y) - z;
    const double error = 16 * 2.220446049250313e-16 * sum + 16 * 2.2250738585072014e-308;
    if (isfinite(sum)) {
        if (difference > error)
            return true;
        if (difference < -error)
            return false;
    }
    return cone_exact(u[0], u[1], u[2], u[3], uncertain);
}
__device__ inline void fail(DeviceCounters *c, int code) { atomicMax(&c->error, code); }
__device__ inline unsigned long long bits(double v) { return (unsigned long long)__double_as_longlong(v); }
// Statistics of nonnegative doubles by their bit patterns; the atomic only when the value improves the current one
// (hundreds of thousands of same-address atomics per kernel serialised before).
__device__ inline void stat_max(unsigned long long *addr, double value) {
    const unsigned long long b = bits(value);
    if (b > *(volatile unsigned long long *)addr)
        atomicMax(addr, b);
}
// Warp-reduced statistics: every lane of the warp must call these (no early returns before them); lane 0 updates.
__device__ inline void warp_stat_max(unsigned long long *addr, double value) {
    for (int o = 16; o > 0; o /= 2)
        value = fmax(value, __shfl_xor_sync(0xffffffffu, value, o));
    if ((threadIdx.x & 31) == 0)
        atomicMax(addr, bits(value));
}
__device__ inline void warp_stat_min(unsigned long long *addr, double value) {
    for (int o = 16; o > 0; o /= 2)
        value = fmin(value, __shfl_xor_sync(0xffffffffu, value, o));
    if ((threadIdx.x & 31) == 0)
        atomicMin(addr, bits(value));
}
__device__ inline void stat_min(unsigned long long *addr, double value) {
    const unsigned long long b = bits(value);
    if (b < *(volatile unsigned long long *)addr)
        atomicMin(addr, b);
}
__device__ double norm_up(const double *v) {
    const double scale = 0x1p600;
    const bool tiny = fmax(fabs(v[1]), fmax(fabs(v[2]), fabs(v[3]))) < 1e-290;
    const double s = tiny ? scale : 1.;
    const double n = sqrt((v[1] * s) * (v[1] * s) + (v[2] * s) * (v[2] * s) + (v[3] * s) * (v[3] * s));
    double up = n * (1 + 8 * 2.220446049250313e-16);
    up = tiny ? up / scale : up;
    return ::nextafter(up, dinf());
}
// nudge_into_cone of the host: returns true if the density was raised; fails (code 3) beyond rounding.
__device__ __noinline__ void nudge_slow(double *v, DeviceCounters *c, bool &uncertain);
__device__ inline bool nudge(double *v, DeviceCounters *c, bool &uncertain) {
    if (cone(v, uncertain))
        return false;
    nudge_slow(v, c, uncertain);
    return true;
}
__device__ __noinline__ void nudge_slow(double *v, DeviceCounters *c, bool &uncertain) {
    for (int k = 0; k < 256 && !cone(v, uncertain); ++k)
        v[0] = ::nextafter(v[0], dinf());
    if (!cone(v, uncertain) && v[0] < 1e-290) {
        v[0] = fmax(v[0], norm_up(v));
        for (int k = 0; k < 256 && !cone(v, uncertain); ++k)
            v[0] = ::nextafter(v[0], dinf());
    }
    if (!cone(v, uncertain))
        fail(c, 3);
}
__device__ double cone_exit_fraction(const double *b, const double *d) {
    const double bj = sqrt(b[1] * b[1] + b[2] * b[2] + b[3] * b[3]);
    const double c = (b[0] - bj) * (b[0] + bj);
    if (!(c > 0) || !isfinite(c))
        return -1;
    const double a = d[0] * d[0] - (d[1] * d[1] + d[2] * d[2] + d[3] * d[3]);
    const double h = b[0] * d[0] - (b[1] * d[1] + b[2] * d[2] + b[3] * d[3]);
    const double disc = h * h - a * c;
    if (!(disc >= 0) || !isfinite(disc))
        return -1;
    const double s = sqrt(disc), t = -(h + (h >= 0 ? s : -s));
    double best = -1;
    if (a != 0) {
        const double r = t / a;
        if (r > 0)
            best = r;
    }
    if (t != 0) {
        const double r = c / t;
        if (r > 0 && (best < 0 || r < best))
            best = r;
    }
    return best;
}
__device__ bool valid_at(const double *b, const double *d, double a, bool &uncertain) {
    const double p[4] = {b[0] + a * d[0], b[1] + a * d[1], b[2] + a * d[2], b[3] + a * d[3]};
    return cone(p, uncertain);
}
// segment() of src/m1.hpp (both = false).
__device__ __noinline__ double segment(const double *b, const double *d, DeviceCounters *c, bool &uncertain) {
    if (!cone(b, uncertain)) {
        fail(c, 4);
        return 0;
    }
    if (valid_at(b, d, 1, uncertain))
        return 1;
    const double guess = cone_exit_fraction(b, d);
    if (guess > 0 && guess <= 1) {
        double lo = guess * (1 - 8e-14), hi = fmin(1., guess * (1 + 8e-14));
        if (valid_at(b, d, lo, uncertain) && !valid_at(b, d, hi, uncertain)) {
            for (int k = 0; k < 8; ++k) {
                const double mid = (lo + hi) / 2;
                if (valid_at(b, d, mid, uncertain))
                    lo = mid;
                else
                    hi = mid;
            }
            return lo;
        }
    }
    double lo = 0, hi = 1;
    for (int k = 0; k < 60; ++k) {
        const double mid = (lo + hi) / 2;
        if (valid_at(b, d, mid, uncertain))
            lo = mid;
        else
            hi = mid;
    }
    for (int k = 0; k < 64 && lo > 0; ++k) {
        if (valid_at(b, d, lo, uncertain))
            return lo;
        lo = ::nextafter(lo, 0.);
    }
    for (int k = 0; k < 60 && lo > 0; ++k) {
        lo *= .5;
        if (valid_at(b, d, lo, uncertain))
            return lo;
    }
    if (!valid_at(b, d, 0, uncertain))
        fail(c, 5);
    return 0;
}
__device__ inline void project4(const View &v, const double *row, double *u) {
    u[0] = v.s0 * row[0];
    u[1] = v.s1 * row[3];
    u[2] = v.s1 * row[1];
    u[3] = v.s1 * row[2];
}

// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void closure_node(const View &v, const double *a, int i);
__global__ void k_closure(View v, const double *a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < v.nodes)
        closure_node(v, a, i);
}
__device__ __forceinline__ void closure_node(const View &v, const double *a, int i) {
    const double *row = a + std::size_t(i) * v.q;
    v.density[i] = row[0];
    double u[4];
    project4(v, row, u);
    for (int c = 0; c < 4; ++c)
        v.low[std::size_t(c) * v.nodes + i] = u[c];
    bool uncertain = false;
    const bool in = cone(u, uncertain);
    if (uncertain) {
        const int k = atomicAdd(&v.counters->closure_fallbacks, 1);
        if (k < v.fallback_capacity)
            v.closure_list[k] = i;
        return;
    }
    if (!in) {
        fail(v.counters, 1);
        return;
    }
    const bool vacuum = u[0] == 0;
    double gap = 0, K = 0, beta[3] = {0, 0, 0};
    if (!vacuum) {
        double j2 = 0;
        for (int d = 1; d < 4; ++d) {
            const double f = u[d] / u[0];
            j2 += f * f;
        }
        const double w = 1 - j2;
        if (w <= 1e-12) { // near beam: exact gap in long double on the host
            const int k = atomicAdd(&v.counters->closure_fallbacks, 1);
            if (k < v.fallback_capacity)
                v.closure_list[k] = i;
            return;
        }
        const double s = sqrt(1 + 3 * w);
        gap = 12 * w / ((s + 1) * (s + 2));
        const double f = 3 / (2 + s);
        for (int d = 0; d < 3; ++d)
            beta[d] = f * u[d + 1] / u[0];
        K = u[0] * 3 / (4 - gap);
    }
    for (int d = 0; d < v.dimension; ++d) {
        double fl[4] = {0, 0, 0, 0};
        if (!vacuum) {
            fl[0] = u[d + 1];
            for (int k = 0; k < 3; ++k)
                fl[k + 1] = K / 3 * ((k == d ? gap : 0.) + 4 * beta[d] * beta[k]);
        }
        double head[4];
        for (int j = 0; j < 4; ++j) {
            double s = 0;
            for (int e = v.flux_outer[d][j]; e < v.flux_outer[d][j + 1]; ++e)
                s += v.flux_value[d][e] * row[v.flux_inner[d][e]];
            head[j] = s;
        }
        double fh[4];
        project4(v, head, fh);
        for (int c = 0; c < 4; ++c) {
            v.low_flux[std::size_t(4 * d + c) * v.nodes + i] = fl[c];
            v.closure[std::size_t(4 * d + c) * v.nodes + i] = fh[c] - fl[c];
        }
    }
}
// Size class of element e and its stabilization viscosity (graded meshes; class 0 and `viscosity` otherwise).
__device__ __forceinline__ int klass_of(const View &v, int e) { return v.klass ? v.klass[e] : 0; }
__device__ __forceinline__ double viscosity_of(const View &v, int c) {
    return v.class_viscosity ? v.class_viscosity[c] : v.viscosity;
}
template <int N> __device__ __forceinline__ double weno_element(const View &v, int e) {
    constexpr int NA = N > 0 ? N : max_local;
    const int n = N > 0 ? N : v.local;
    if (!v.weno) {
        v.gamma[e] = 1;
        return 1;
    }
    double cand[7][NA], other[NA];
    auto norm = [&](const double *x) {
        double s = 0;
        for (int r = 0; r < n; ++r) {
            double p = 0;
            for (int c = 0; c < n; ++c)
                p += v.seminorm[r * n + c] * x[c];
            s += x[r] * p;
        }
        return sqrt(fmax(0., s));
    };
    auto power = [&](double x) { return v.weno_r == 2 ? x * x : pow(x, v.weno_r); };
    for (int l = 0; l < n; ++l)
        cand[0][l] = v.density[v.connectivity[e * n + l]];
    const double scale = norm(cand[0]);
    if (!(scale > 0)) {
        v.gamma[e] = 1;
        return 1;
    }
    int m = 0;
    for (int d = 0; d < v.dimension; ++d)
        for (int side = 0; side < 2; ++side) {
            const int next = v.neighbors[e * 2 * v.dimension + 2 * d + side];
            if (next == -1)
                continue;
            for (int l = 0; l < n; ++l) // -2: mirror side, the neighbour is this element reflected in d
                other[l] = v.density[next >= 0 ? v.connectivity[next * n + l]
                                               : v.connectivity[e * n + v.reflected[d][l]]];
            ++m;
            const double *X = v.ext_index ? v.ext_list + std::size_t(v.ext_index[e * 2 * v.dimension + 2 * d + side]) * n * n
                                          : v.extension[d][side];
            for (int r = 0; r < n; ++r) {
                double s = 0;
                for (int c = 0; c < n; ++c)
                    s += X[r * n + c] * other[c];
                cand[m][r] = s;
            }
        }
    double smooth[7], weight[7];
    smooth[0] = power(scale);
    for (int l = 1; l <= m; ++l)
        smooth[l] = power(norm(cand[l]));
    double epsilon = v.weno_epsilon;
    if (v.scale_invariant_weno) {
        double mx = smooth[0];
        for (int l = 1; l <= m; ++l)
            mx = fmax(mx, smooth[l]);
        epsilon *= mx;
    }
    double total = 0;
    for (int l = 0; l <= m; ++l) {
        weight[l] = (l ? v.weno_linear : 1 - m * v.weno_linear) / (smooth[l] + epsilon);
        total += weight[l];
    }
    for (int r = 0; r < n; ++r) {
        double star = 0;
        for (int l = 0; l <= m; ++l)
            star += (weight[l] / total) * cand[l][r];
        other[r] = cand[0][r] - star;
    }
    const double ratio = fmin(1., norm(other) / scale);
    const double g = 1 - (v.weno_q == 1 ? ratio : pow(ratio, v.weno_q));
    v.gamma[e] = g;
    return g;
}
template <int N> __global__ void k_weno(View v, const double *a) {
    (void)a;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const double g = e < v.cells ? weno_element<N>(v, e) : 1.;
    warp_stat_min(&v.counters->min_gamma_bits, fmax(g, 0.));
}
// In-place banded LDL^T solve (natural order) along the lines of direction d, one thread per (line, component).
// Loads are issued in chunks of CH so their latency overlaps.
// f == nullptr: direction first + blockIdx.y on its gradient field (independent solves in one launch).
// src != nullptr (gradients): the right-hand side C_1d src is formed on the fly in the forward sweep (sliding window).
struct FluxShared {
    const double *val;
    const int *idx;
    int q, width;
    __device__ double row(const double *g, int d, int j) const {
        const int base = (d * q + j) * width;
        double s = 0;
        for (int w = 0; w < width; ++w)
            s += val[base + w] * g[idx[base + w]];
        return s;
    }
    // Adaptive PN: only gradient columns < lim (as ElementWeno::rate, which sums col < qpad).
    __device__ double row_upto(const double *g, int d, int j, int lim) const {
        const int base = (d * q + j) * width;
        double s = 0;
        for (int w = 0; w < width; ++w) {
            const int col = idx[base + w];
            if (col < lim)
                s += val[base + w] * g[col];
        }
        return s;
    }
};
// FUSE (experiment/big-rebuilds): the last mass-solve direction also forms the stage combination of the PN rows c >= 4
// (as k_combine part 1, same operations in the same order, so bitwise equal): out = old a + fresh (x + tau rate),
// rate = solved value - sum_d A_d g_d (row c), zero on mirror planes for odd moments. Rows c < 4 stay the raw solve.
template <int P, int CH, bool FUSE = false>
__global__ void k_line_solve(View v, int first, double *field, const double *src, const double *cx = nullptr,
                             const double *ca = nullptr, double old = 0, double fresh = 1, double tau = 0) {
    const int d = first + blockIdx.y;
    double *f = field ? field : v.gradient[d];
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int n = v.npa[d], s = v.stride[d], lines = v.nodes / n;
    if (t >= std::size_t(lines) * v.q)
        return;
    const int c = int(t % v.q), line = int(t / v.q);
    const int start = (line / s) * (s * n) + line % s;
    // Mirror side in d (as ElementWeno::solve_lines): components odd under the reflection (even ones for the
    // gradient, field == nullptr) are solved with the Dirichlet factor on the line without the mirror node, whose
    // value is then zero. Reduced index k = physical index - off; nn rows.
    const int mi = v.mirror_index[d];
    const bool dirichlet = mi >= 0 && bool((v.comp_odd[c] >> d) & 1) != (field == nullptr);
    const int off = dirichlet && mi == 0 ? 1 : 0, nn = dirichlet ? n - 1 : n;
    const double *L = dirichlet ? v.dlower[d] : v.lower[d], *ID = dirichlet ? v.dinvdiag[d] : v.invdiag[d];
    const std::size_t step = std::size_t(s) * v.q;
    double *row0 = f + std::size_t(start) * v.q + c;
    double *base = row0 + std::size_t(off) * step;
    double hist[P > 0 ? P : 1];
    for (int m = 0; m < P; ++m)
        hist[m] = 0;
    const double *sbase = src ? src + std::size_t(start) * v.q + c + std::size_t(off) * step : nullptr;
    const double *Cb = v.convection_band[d] + std::size_t(off) * (2 * P + 1);
    double win[2 * P + 1]; // win[m + P] = src[k + m] (reduced index; physical k + off in [0, n))
    if (src) {
#pragma unroll
        for (int j = 0; j < 2 * P; ++j) {
            const int kk = j - P;
            win[j] = kk >= -off && kk < n - off ? sbase[std::ptrdiff_t(kk) * std::ptrdiff_t(step)] : 0.;
        }
    }
    // Software pipelining: the loads of chunk k + 1 are issued before chunk k is computed (forward: the chunk ahead is
    // not written yet; backward: the chunk below is not written yet).
    auto load_fwd = [&](int k0, double *buf) {
#pragma unroll
        for (int j = 0; j < CH; ++j) {
            if (src)
                buf[j] = k0 + j + P < n - off ? sbase[std::size_t(k0 + j + P) * step] : 0.;
            else
                buf[j] = k0 + j < nn ? base[std::size_t(k0 + j) * step] : 0.;
        }
    };
    auto fwd = [&](int k0, const double *buf) {
#pragma unroll
        for (int j = 0; j < CH; ++j) {
            const int k = k0 + j;
            if (k >= nn)
                break;
            double y;
            if (src) {
                win[2 * P] = buf[j];
                y = 0;
#pragma unroll
                for (int m = 0; m <= 2 * P; ++m)
                    y += Cb[std::size_t(k) * (2 * P + 1) + m] * win[m];
#pragma unroll
                for (int m = 0; m < 2 * P; ++m)
                    win[m] = win[m + 1];
            } else
                y = buf[j];
#pragma unroll
            for (int m = 1; m <= P; ++m)
                if (k - m >= 0)
                    y -= L[std::size_t(k) * P + m - 1] * hist[m - 1];
#pragma unroll
            for (int m = P - 1; m > 0; --m)
                hist[m] = hist[m - 1];
            if (P > 0)
                hist[0] = y;
            base[std::size_t(k) * step] = y * ID[k];
        }
    };
    double bufA[CH], bufB[CH];
    load_fwd(0, bufA);
    for (int k0 = 0; k0 < nn; k0 += 2 * CH) {
        if (k0 + CH < nn)
            load_fwd(k0 + CH, bufB);
        fwd(k0, bufA);
        if (k0 + CH >= nn)
            break;
        if (k0 + 2 * CH < nn)
            load_fwd(k0 + 2 * CH, bufA);
        fwd(k0 + CH, bufB);
    }
    for (int m = 0; m < P; ++m)
        hist[m] = 0;
    auto load_bwd = [&](int k1, double *buf) {
#pragma unroll
        for (int j = 0; j < CH; ++j)
            buf[j] = k1 - j >= 0 ? base[std::size_t(k1 - j) * step] : 0.;
    };
    auto bwd = [&](int k1, const double *buf) {
#pragma unroll
        for (int j = 0; j < CH; ++j) {
            const int k = k1 - j;
            if (k < 0)
                break;
            double x = buf[j];
#pragma unroll
            for (int m = 1; m <= P; ++m)
                if (k + m < nn)
                    x -= L[std::size_t(k + m) * P + m - 1] * hist[m - 1];
#pragma unroll
            for (int m = P - 1; m > 0; --m)
                hist[m] = hist[m - 1];
            if (P > 0)
                hist[0] = x;
            if (FUSE && c >= 4) {
                const std::size_t node = std::size_t(start) + std::size_t(k + off) * s, t = node * v.q + c;
                const FluxShared F{v.ell_val, v.ell_idx, v.q, v.flux_width};
                double rate = x;
                for (int dd = 0; dd < v.dimension; ++dd)
                    rate += -F.row(v.gradient[dd] + node * v.q, dd, c);
                if (v.node_mirror && (v.node_mirror[node] & v.comp_odd[c]))
                    rate = 0;
                base[std::size_t(k) * step] = old == 0 ? fresh * (cx[t] + tau * rate) : old * ca[t] + fresh * (cx[t] + tau * rate);
            } else
                base[std::size_t(k) * step] = x;
        }
    };
    load_bwd(nn - 1, bufA);
    for (int k1 = nn - 1; k1 >= 0; k1 -= 2 * CH) {
        if (k1 - CH >= 0)
            load_bwd(k1 - CH, bufB);
        bwd(k1, bufA);
        if (k1 - CH < 0)
            break;
        if (k1 - 2 * CH >= 0)
            load_bwd(k1 - 2 * CH, bufA);
        bwd(k1 - CH, bufB);
    }
    if (dirichlet)
        row0[std::size_t(mi) * step] = 0;
}
// Periodic direction: in-place solve with the arrow LDL^T of the cyclic 1D mass (band rows k < nb = n - P from
// v.lower, dense tail rows from v.tail), one thread per (line, component). src != nullptr: right-hand side C_1d src
// with cyclic column indices (gradients). Not pipelined (periodic runs are small accuracy studies).
template <int P> __global__ void k_line_solve_cyclic(View v, int first, double *field, const double *src) {
    const int d = first + blockIdx.y;
    double *f = field ? field : v.gradient[d];
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int n = v.npa[d], s = v.stride[d], lines = v.nodes / n;
    if (t >= std::size_t(lines) * v.q)
        return;
    const int c = int(t % v.q), line = int(t / v.q);
    const int start = (line / s) * (s * n) + line % s, nb = n - P;
    const double *L = v.lower[d], *ID = v.invdiag[d], *T = v.tail[d], *Cb = v.convection_band[d];
    const std::size_t step = std::size_t(s) * v.q;
    double *base = f + std::size_t(start) * v.q + c;
    const double *sb = src ? src + std::size_t(start) * v.q + c : nullptr;
    double hist[P], acc[P], ytail[P];
#pragma unroll
    for (int m = 0; m < P; ++m)
        hist[m] = acc[m] = ytail[m] = 0;
    for (int k = 0; k < n; ++k) {
        double y;
        if (sb) {
            y = 0;
#pragma unroll
            for (int m = 0; m <= 2 * P; ++m) {
                int j = k + m - P;
                j = j < 0 ? j + n : (j >= n ? j - n : j);
                y += Cb[std::size_t(k) * (2 * P + 1) + m] * sb[std::size_t(j) * step];
            }
        } else
            y = base[std::size_t(k) * step];
        if (k < nb) {
#pragma unroll
            for (int m = 1; m <= P; ++m)
                if (k - m >= 0)
                    y -= L[std::size_t(k) * P + m - 1] * hist[m - 1];
#pragma unroll
            for (int m = P - 1; m > 0; --m)
                hist[m] = hist[m - 1];
            hist[0] = y;
#pragma unroll
            for (int r = 0; r < P; ++r)
                acc[r] += T[std::size_t(r) * n + k] * y;
        } else {
            const int r = k - nb;
            y -= acc[r];
            for (int j = 0; j < r; ++j)
                y -= T[std::size_t(r) * n + nb + j] * ytail[j];
            ytail[r] = y;
        }
        base[std::size_t(k) * step] = y * ID[k];
    }
    double xt[P];
    for (int r = P - 1; r >= 0; --r) {
        const int k = nb + r;
        double x = base[std::size_t(k) * step];
        for (int r2 = r + 1; r2 < P; ++r2)
            x -= T[std::size_t(r2) * n + k] * xt[r2];
        xt[r] = x;
        base[std::size_t(k) * step] = x;
    }
#pragma unroll
    for (int m = 0; m < P; ++m)
        hist[m] = 0;
    for (int k = nb - 1; k >= 0; --k) {
        double x = base[std::size_t(k) * step];
#pragma unroll
        for (int m = 1; m <= P; ++m)
            if (k + m < nb)
                x -= L[std::size_t(k + m) * P + m - 1] * hist[m - 1];
#pragma unroll
        for (int r = 0; r < P; ++r)
            x -= T[std::size_t(r) * n + k] * xt[r];
#pragma unroll
        for (int m = P - 1; m > 0; --m)
            hist[m] = hist[m - 1];
        hist[0] = x;
        base[std::size_t(k) * step] = x;
    }
}
// Element-wise stabilization (any degree and dimension): one thread per (element, component) loads the element rows
// of a and g_d once and forms the loads of all local nodes; k_gather_stab sums them per node (residual = -sum).
template <int N, int DIM>
__global__ void __launch_bounds__(128) k_stab_element(View v, const double *a, double *local_out, bool want_low,
                                                      double constant_gamma) {
    // Element matrices in shared memory: viscosity * K (n x n), then C_d^T rows (n x n each).
    __shared__ double sK[N * N], sC[DIM][N * N];
    for (int k = threadIdx.x; k < N * N; k += blockDim.x) {
        sK[k] = v.viscosity * v.stiffness_rows[k];
#pragma unroll
        for (int d = 0; d < DIM; ++d)
            sC[d][k] = v.convection_t_rows[d][k];
    }
    __syncthreads();
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(v.cells) * v.q)
        return;
    const int e = int(t / v.q), c = int(t % v.q), q = v.q;
    const int *map = v.connectivity + std::size_t(e) * N;
    double av[N], gv[DIM][N];
#pragma unroll
    for (int m = 0; m < N; ++m) {
        const std::size_t r = std::size_t(map[m]) * q + c;
        av[m] = a[r];
#pragma unroll
        for (int d = 0; d < DIM; ++d)
            gv[d][m] = v.gradient[d][r];
    }
    const double g = constant_gamma >= 0 ? constant_gamma : v.gamma[e];
    const double weight = -v.viscosity * g;
    for (int l = 0; l < N; ++l) {
        double stiff = 0, conv = 0;
#pragma unroll
        for (int m = 0; m < N; ++m) {
            stiff += sK[l * N + m] * av[m];
#pragma unroll
            for (int d = 0; d < DIM; ++d)
                conv += sC[d][l * N + m] * gv[d][m];
        }
        const double term = stiff + weight * conv;
        local_out[(std::size_t(e) * N + l) * q + c] = term;
        if (want_low && c < 4) {
            const int slot = c == 0 ? 0 : (c == 1 ? 2 : (c == 2 ? 3 : 1));
            v.stab_low[std::size_t(4 * l + slot) * v.cells + e] = (c == 0 ? v.s0 : v.s1) * term;
        }
    }
}
__global__ void k_gather_stab(View v, const double *local_out, double *result) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(v.nodes) * v.q)
        return;
    const int i = int(t / v.q), c = int(t % v.q);
    double out = 0;
    for (int k = v.node_offsets[i]; k < v.node_offsets[i + 1]; ++k)
        out -= local_out[std::size_t(v.node_entries[k]) * v.q + c];
    result[t] = out;
}
// residual(i, c) = -sum_{(e, l) at i} s_h^e(phi_l, a)_c; stab_low(e, 4 l + .) = projected element loads.
__global__ void k_stabilization(View v, const double *a, double *result, bool want_low, double constant_gamma) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(v.nodes) * v.q)
        return;
    const int i = int(t / v.q), c = int(t % v.q), n = v.local, q = v.q;
    double out = 0;
    for (int k = v.node_offsets[i]; k < v.node_offsets[i + 1]; ++k) {
        const int entry = v.node_entries[k], e = entry / n, l = entry % n;
        const int *map = v.connectivity + std::size_t(e) * n;
        const int kc = klass_of(v, e);
        const double nu = viscosity_of(v, kc);
        const double *K = v.stiffness_rows + std::size_t(kc) * n * n + std::size_t(l) * n;
        double term = 0;
        for (int m = 0; m < n; ++m)
            term += (nu * K[m]) * a[std::size_t(map[m]) * q + c];
        const double g = constant_gamma >= 0 ? constant_gamma : v.gamma[e];
        const double weight = -nu * g;
        if (weight != 0)
            for (int d = 0; d < v.dimension; ++d) {
                const double *Ct = v.convection_t_rows[d] + std::size_t(kc) * n * n + std::size_t(l) * n;
                const double *G = v.gradient[d];
                for (int m = 0; m < n; ++m)
                    term += (weight * Ct[m]) * G[std::size_t(map[m]) * q + c];
            }
        if (want_low && c < 4) {
            // project(): (s0 t0, s1 t3, s1 t1, s1 t2)
            const int slot = c == 0 ? 0 : (c == 1 ? 2 : (c == 2 ? 3 : 1));
            v.stab_low[std::size_t(4 * l + slot) * v.cells + e] = (c == 0 ? v.s0 : v.s1) * term;
        }
        out -= term;
    }
    result[t] = out;
}
constexpr int max_strip_classes = 64; // graded 2D Q1: size classes held in constant memory for the strip kernel
// 2D Q1 element tables per size class c at 16 c: stiffness rows, nu K, C_d^T rows; cNu[c]: viscosity.
// (class tables: View::strip_kv, strip_c0, strip_c1, strip_nu)
// 2D Q1: the gathered residual of node i only involves its 3 x 3 neighbourhood. One thread per (x, component) walks a
// strip of SY rows in y with a sliding 3 x 3 window of a, g_0, g_1 in registers (9 loads per output); the four elements
// around a node are visited as quadrants with compile-time local indices.
template <int SY, int MINB = 4>
__global__ void __launch_bounds__(128, MINB) k_stabilization_strip(View v, const double *a, double *result, bool want_low, double constant_gamma) {
    const int q = v.q, npx = v.npa[0], npy = v.npa[1], cx = v.count[0], cy = v.count[1];
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= npx * q)
        return;
    const int ix = col / q, c = col % q;
    const int y0 = blockIdx.y * SY, y1 = min(npy, y0 + SY);
    // Row pointers (column ix, moment c) advanced by one node row per y; x neighbours at -q / +q. The element columns
    // ex of the two quadrants depend only on ix. Same loads and the same FMA order as before (fewer integer ops:
    // the 64-bit index products per load dominated the issued instructions).
    const std::size_t row = std::size_t(npx) * q;
    const bool xm = ix > 0, xp = ix + 1 < npx;
    const double *P0 = a + std::size_t(ix) * q + c, *P1 = v.gradient[0] + std::size_t(ix) * q + c,
                 *P2 = v.gradient[1] + std::size_t(ix) * q + c;
    int exq[2];
    bool exok[2];
#pragma unroll
    for (int qx = 0; qx < 2; ++qx) {
        exq[qx] = v.strip_elem[0] ? v.strip_elem[0][2 * ix + qx] : ix - 1 + qx;
        exok[qx] = exq[qx] >= 0 && exq[qx] < cx;
    }
    double W[3][3][3];
    auto load_row = [&](int y, int r) {
        if (y < 0 || y >= npy) {
#pragma unroll
            for (int f = 0; f < 3; ++f)
#pragma unroll
                for (int dx = 0; dx < 3; ++dx)
                    W[f][r][dx] = 0.;
            return;
        }
        const std::size_t off = std::size_t(y) * row;
        const double *B[3] = {P0 + off, P1 + off, P2 + off};
#pragma unroll
        for (int f = 0; f < 3; ++f) {
            W[f][r][0] = xm ? B[f][-q] : 0.;
            W[f][r][1] = B[f][0];
            W[f][r][2] = xp ? B[f][q] : 0.;
        }
    };
    load_row(y0 - 1, 0);
    load_row(y0, 1);
    for (int y = y0; y < y1; ++y) {
        load_row(y + 1, 2);
        double out = 0;
#pragma unroll
        for (int qy = 0; qy < 2; ++qy)
#pragma unroll
            for (int qx = 0; qx < 2; ++qx) {
                const int ex = exq[qx];
                const int ey = v.strip_elem[1] ? v.strip_elem[1][2 * y + qy] : y - 1 + qy;
                if (!exok[qx] || ey < 0 || ey >= cy)
                    continue;
                const int e = ex + ey * cx, l = (1 - qx) + 2 * (1 - qy);
                const int kc = 16 * klass_of(v, e);
                const double g = constant_gamma >= 0 ? constant_gamma : v.gamma[e];
                const double weight = -__ldg(v.strip_nu + kc / 16) * g;
                // nu K a + weight (C_0^T g_0 + C_1^T g_1), nu K prescaled (cKv)
                double stiff = 0, conv = 0;
#pragma unroll
                for (int my = 0; my < 2; ++my)
#pragma unroll
                    for (int mx = 0; mx < 2; ++mx) {
                        const int m = mx + 2 * my;
                        stiff += __ldg(v.strip_kv + kc + 4 * l + m) * W[0][qy + my][qx + mx];
                        conv += __ldg(v.strip_c0 + kc + 4 * l + m) * W[1][qy + my][qx + mx] +
                                __ldg(v.strip_c1 + kc + 4 * l + m) * W[2][qy + my][qx + mx];
                    }
                const double term = stiff + weight * conv;
                if (want_low && c < 4) {
                    const int slot = c == 0 ? 0 : (c == 1 ? 2 : (c == 2 ? 3 : 1));
                    v.stab_low[std::size_t(4 * l + slot) * v.cells + e] = (c == 0 ? v.s0 : v.s1) * term;
                }
                out -= term;
            }
        result[std::size_t(y) * row + std::size_t(ix) * q + c] = out;
#pragma unroll
        for (int f = 0; f < 3; ++f)
#pragma unroll
            for (int dx = 0; dx < 3; ++dx) {
                W[f][0][dx] = W[f][1][dx];
                W[f][1][dx] = W[f][2][dx];
            }
    }
}
// 3D Q1: the gathered residual of node i only involves its 3 x 3 x 3 neighbourhood. One thread per (x, y, component)
// walks a strip of SZ node planes in z. Each plane's 3 x 3 window of a, g_0, g_1, g_2 is loaded once and added to the
// element terms of the (up to) three nodes z - 1, z, z + 1 of the thread's column that it touches; a node's eight
// element terms are complete after plane z + 1. Replaces k_stab_element + k_gather_stab on 3D Q1 meshes (no element
// buffer of cells x 8 x q doubles: 5.8 GB on 112^3 at P7). Per element and plane (nu K) a + w (C_d^T g_d) with
// w = -nu gamma, summed plane by plane (rounding-level differences to the element path).
template <int SZ>
__global__ void __launch_bounds__(128) k_stabilization_strip3(View v, const double *a, double *result, bool want_low,
                                                              double constant_gamma) {
    const int q = v.q, npx = v.npa[0], npy = v.npa[1], npz = v.npa[2];
    const int cx = v.count[0], cy = v.count[1], cz = v.count[2];
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= npx * q)
        return;
    const int ix = col / q, c = col % q, iy = blockIdx.y;
    const int z0 = blockIdx.z * SZ, z1 = min(npz, z0 + SZ);
    const std::ptrdiff_t row = std::ptrdiff_t(npx) * q, plane = row * npy;
    const bool okx[3] = {ix > 0, true, ix + 1 < npx}, oky[3] = {iy > 0, true, iy + 1 < npy};
    const std::size_t base = std::size_t(iy) * row + std::size_t(ix) * q + c;
    const double *F[4] = {a + base, v.gradient[0] + base, v.gradient[1] + base, v.gradient[2] + base};
    int exq[2], eyq[2];
    bool exok[2], eyok[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        exq[s] = v.strip_elem[0] ? v.strip_elem[0][2 * ix + s] : ix - 1 + s;
        eyq[s] = v.strip_elem[1] ? v.strip_elem[1][2 * iy + s] : iy - 1 + s;
        exok[s] = exq[s] >= 0 && exq[s] < cx;
        eyok[s] = eyq[s] >= 0 && eyq[s] < cy;
    }
    auto ez_of = [&](int k, int qz) { return v.strip_elem[2] ? v.strip_elem[2][2 * k + qz] : k - 1 + qz; };
    // T[s][qx + 2 qy + 4 qz]: element terms of node k = p - 1 + s (s = 0 completes after plane p).
    double T[3][8];
#pragma unroll
    for (int s = 0; s < 3; ++s)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            T[s][j] = 0;
    for (int p = z0 - 1; p <= z1; ++p) {
        if (p >= 0 && p < npz) {
            double W[4][3][3];
            const std::ptrdiff_t off = std::ptrdiff_t(p) * plane;
#pragma unroll
            for (int f = 0; f < 4; ++f)
#pragma unroll
                for (int dy = 0; dy < 3; ++dy)
#pragma unroll
                    for (int dx = 0; dx < 3; ++dx)
                        W[f][dy][dx] = okx[dx] && oky[dy] ? F[f][off + (dy - 1) * row + (dx - 1) * q] : 0.;
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int k = p - 1 + s;
                if (k < z0 || k >= z1)
                    continue;
#pragma unroll
                for (int qz = 0; qz < 2; ++qz) {
                    const int mz = 2 - s - qz; // plane p is the element's node layer mz
                    if (mz < 0 || mz > 1)
                        continue;
                    const int ez = ez_of(k, qz);
                    if (ez < 0 || ez >= cz)
                        continue;
#pragma unroll
                    for (int qy = 0; qy < 2; ++qy)
#pragma unroll
                        for (int qx = 0; qx < 2; ++qx) {
                            if (!exok[qx] || !eyok[qy])
                                continue;
                            const int e = exq[qx] + cx * (eyq[qy] + cy * ez);
                            const int l = (1 - qx) + 2 * (1 - qy) + 4 * (1 - qz);
                            const int kc = klass_of(v, e);
                            const double g = constant_gamma >= 0 ? constant_gamma : v.gamma[e];
                            const std::size_t r = std::size_t(kc) * 64 + 8 * l + 4 * mz;
                            double t = T[s][qx + 2 * qy + 4 * qz];
#pragma unroll
                            for (int my = 0; my < 2; ++my)
#pragma unroll
                                for (int mx = 0; mx < 2; ++mx)
                                    t += __ldg(v.stiffness_nu + r + mx + 2 * my) * W[0][qy + my][qx + mx];
                            if (g != 0) {
                                double conv = 0;
#pragma unroll
                                for (int d = 0; d < 3; ++d)
#pragma unroll
                                    for (int my = 0; my < 2; ++my)
#pragma unroll
                                        for (int mx = 0; mx < 2; ++mx)
                                            conv += __ldg(v.convection_t_rows[d] + r + mx + 2 * my) * W[1 + d][qy + my][qx + mx];
                                t += (-viscosity_of(v, kc) * g) * conv;
                            }
                            T[s][qx + 2 * qy + 4 * qz] = t;
                        }
                }
            }
        }
        const int k = p - 1;
        if (k >= z0 && k < z1) {
            double out = 0;
#pragma unroll
            for (int qz = 0; qz < 2; ++qz) {
                const int ez = ez_of(k, qz);
                if (ez < 0 || ez >= cz)
                    continue;
#pragma unroll
                for (int qy = 0; qy < 2; ++qy)
#pragma unroll
                    for (int qx = 0; qx < 2; ++qx) {
                        if (!exok[qx] || !eyok[qy])
                            continue;
                        const double term = T[0][qx + 2 * qy + 4 * qz];
                        if (want_low && c < 4) {
                            const int e = exq[qx] + cx * (eyq[qy] + cy * ez);
                            const int l = (1 - qx) + 2 * (1 - qy) + 4 * (1 - qz);
                            const int slot = c == 0 ? 0 : (c == 1 ? 2 : (c == 2 ? 3 : 1));
                            v.stab_low[std::size_t(4 * l + slot) * v.cells + e] = (c == 0 ? v.s0 : v.s1) * term;
                        }
                        out -= term;
                    }
            }
            result[std::size_t(k) * plane + base] = out;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            T[0][j] = T[1][j];
            T[1][j] = T[2][j];
            T[2][j] = 0;
        }
    }
}
__global__ void k_add_rows(int q, double *field, const int *index, const double *rows, int count) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(count) * q)
        return;
    const int k = int(t / q), c = int(t % q);
    field[std::size_t(index[k]) * q + c] += rows[t];
}
// A_d^T rows in shared memory: values, inner indices, outer offsets per direction (dynamic shared size).
// A_d^T (active moments) in padded ELL form in shared memory: row j of direction d has `width` entries
// (value, index), zero-padded after the CSR entries (same summation order; the padding adds +0).

__device__ FluxShared stage_flux(const View &v, double *shared) {
    if (!v.flux_in_shared) // high PN orders (q > ~500 in 2D): read-only global loads (uniform branch)
        return FluxShared{v.ell_val, v.ell_idx, v.q, v.flux_width};
    const int total = v.dimension * v.q * v.flux_width;
    double *val = shared;
    int *idx = reinterpret_cast<int *>(shared + total);
    for (int k = threadIdx.x; k < total; k += blockDim.x) {
        val[k] = v.ell_val[k];
        idx[k] = v.ell_idx[k];
    }
    __syncthreads();
    return FluxShared{val, idx, v.q, v.flux_width};
}
// moments_only: one thread per (node, j < 4), moment_rate = project(field + flux) without writing field; otherwise
// field += flux for all entries (and moment_rate when want_moments).
// Adaptive PN (tile_q != nullptr): gradient columns < qpad of the node only, as ElementWeno::rate (the padded gradient
// segments leave nonzero entries above the node's order).
__global__ void k_flux_rows(View v, double *field, bool want_moments, bool moments_only, const int *tile_q = nullptr,
                            int tb0 = 1, int tb1 = 1, int tb2 = 1, int nt0 = 1, int nt1 = 1) {
    extern __shared__ double shared[];
    const FluxShared F = stage_flux(v, shared);
    const int q = v.q;
    const unsigned w = moments_only ? 4u : unsigned(q), t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= unsigned(v.nodes) * w)
        return;
    const int i = int(t / w), j = int(t % w);
    const std::size_t at = std::size_t(i) * q + j;
    // Same order as the host: field + (-s_0) + (-s_1) ...
    double value = field[at];
    int lim = q;
    if (tile_q) {
        const int px = i % v.npa[0], py = (i / v.npa[0]) % v.npa[1], pz = i / (v.npa[0] * v.npa[1]);
        const int qi = tile_q[((pz / tb2) * nt1 + py / tb1) * nt0 + px / tb0];
        lim = qi <= 4 ? min(4, q) : min(q, 4 + ((qi - 4 + 31) / 32) * 32);
    }
    for (int d = 0; d < v.dimension; ++d) {
        const double *g = v.gradient[d] + std::size_t(i) * q;
        value += -F.row_upto(g, d, j, lim);
    }
    if (!moments_only)
        field[at] = value;
    if ((want_moments || moments_only) && j < 4) {
        const int slot = j == 0 ? 0 : (j == 3 ? 1 : (j == 1 ? 2 : 3));
        v.moment_rate[std::size_t(slot) * v.nodes + i] = (j == 0 ? v.s0 : v.s1) * value;
    }
}
// Element limiter of limited_low_load (src/element_weno.cpp), one thread per element.
// One element of the limiter; raw / lim receive the zero sums (statistics, 0 if not applicable).
template <int N> __device__ __forceinline__ void limit_element(const View &v, int e, double &raw_out, double &lim_out) {
    constexpr int NA = N > 0 ? N : max_local;
    const int n = N > 0 ? N : v.local, dim = v.dimension;
    const int *map = v.connectivity + std::size_t(e) * n;
    const std::size_t NN = v.nodes;
    const int kc = klass_of(v, e);
    const std::size_t po = std::size_t(kc) * n * (n - 1); // pair table offset of the class
    const double *tvis = v.total_viscosity + std::size_t(kc) * n;
    double ubar[NA][4], contribution[NA][4];
    bool uncertain = false;
    long long nudged = 0;
#pragma unroll
    for (int i = 0; i < n; ++i) {
        const int ni = map[i];
        double ui[4], qi[4], Fi[12], Gi[12];
        for (int r = 0; r < 4; ++r)
            ui[r] = v.low[r * NN + ni], qi[r] = v.moment_rate[r * NN + ni];
        for (int r = 0; r < 4 * dim; ++r)
            Fi[r] = v.low_flux[r * NN + ni], Gi[r] = v.closure[r * NN + ni];
        double weighted[4] = {0, 0, 0, 0}, fi[4] = {0, 0, 0, 0};
#pragma unroll
        for (int k = 0; k < n - 1; ++k) {
            const std::size_t pk = po + i * (n - 1) + k;
            const int j = N > 0 ? (k < i ? k : k + 1) : v.pair_j[pk]; // pair table order: all j != i ascending
            const double dij = v.pair_d[pk];
            const int nj = map[j];
            double uj[4], qj[4], Fj[12], Gj[12];
            for (int r = 0; r < 4; ++r)
                uj[r] = v.low[r * NN + nj], qj[r] = v.moment_rate[r * NN + nj];
            for (int r = 0; r < 4 * dim; ++r)
                Fj[r] = v.low_flux[r * NN + nj], Gj[r] = v.closure[r * NN + nj];
            double fd[4] = {0, 0, 0, 0}, ct[4] = {0, 0, 0, 0};
            for (int d = 0; d < dim; ++d) {
                const double cij = v.pair_c[3 * pk + d], cji = v.pair_ct[3 * pk + d];
                for (int r = 0; r < 4; ++r) {
                    fd[r] += cij * (Fj[4 * d + r] - Fi[4 * d + r]);
                    ct[r] += cji * Gi[4 * d + r] - cij * Gj[4 * d + r];
                }
            }
            if (dij > 0) {
                double bar[4];
                for (int r = 0; r < 4; ++r)
                    bar[r] = .5 * (ui[r] + uj[r]) - fd[r] / (2 * dij);
                if (!cone(bar, uncertain) || uncertain) {
                    // Stable base bar (long double) on the host.
                    const int s = atomicAdd(&v.counters->limiter_fallbacks, 1);
                    if (s < v.fallback_capacity)
                        v.limiter_list[s] = e;
                    return;
                }
                for (int r = 0; r < 4; ++r) {
                    weighted[r] += dij * bar[r];
                    fi[r] += dij * (ui[r] - uj[r]);
                }
            }
            for (int r = 0; r < 4; ++r)
                fi[r] += v.pair_m[pk] * (qi[r] - qj[r]) + ct[r];
        }
        if (v.stabilize)
            for (int r = 0; r < 4; ++r)
                fi[r] -= v.stab_low[std::size_t(4 * i + r) * v.cells + e];
        for (int r = 0; r < 4; ++r) {
            ubar[i][r] = weighted[r] / tvis[i];
            contribution[i][r] = fi[r];
        }
        if (nudge(ubar[i], v.counters, uncertain))
            ++nudged;
    }
    if (uncertain) {
        const int s = atomicAdd(&v.counters->limiter_fallbacks, 1);
        if (s < v.fallback_capacity)
            v.limiter_list[s] = e;
        return;
    }
    double sum[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < n; ++i)
        for (int r = 0; r < 4; ++r)
            sum[r] += contribution[i][r];
    double raw = 0;
    for (int r = 0; r < 4; ++r)
        raw = fmax(raw, fabs(sum[r]));
    double factor = 1;
    if (v.limit) {
        bool need = false;
#pragma unroll
        for (int i = 0; i < n; ++i) {
            double t[4];
            for (int r = 0; r < 4; ++r)
                t[r] = ubar[i][r] + contribution[i][r] / (2 * tvis[i]);
            need |= !cone(t, uncertain);
        }
        if (uncertain) {
            const int s = atomicAdd(&v.counters->limiter_fallbacks, 1);
            if (s < v.fallback_capacity)
                v.limiter_list[s] = e;
            return;
        }
        if (need) { // element factor in k_limiter_segment (keeps the segment searches out of this kernel's warps)
            for (int i = 0; i < n; ++i)
                for (int r = 0; r < 4; ++r) {
                    v.seg_ubar[std::size_t(e) * 4 * n + 4 * i + r] = ubar[i][r];
                    v.seg_lim[std::size_t(e) * 4 * n + 4 * i + r] = contribution[i][r];
                }
            v.segment_list[atomicAdd(&v.counters->segments, 1)] = e;
            if (nudged)
                atomicAdd(&v.counters->nudged_bars, (unsigned long long)nudged);
            raw_out = raw;
            return;
        }
    }
    double after[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < n; ++i)
        for (int r = 0; r < 4; ++r)
            after[r] += contribution[i][r];
    double lim = 0;
    for (int r = 0; r < 4; ++r)
        lim = fmax(lim, fabs(after[r]));
#pragma unroll
    for (int i = 0; i < n; ++i) {
        const int ni = map[i];
        for (int r = 0; r < 4; ++r) {
            const double lo = 2 * tvis[i] * (ubar[i][r] - v.low[r * NN + ni]);
            v.element_limited[std::size_t(4 * i + r) * v.cells + e] = lo + contribution[i][r];
        }
    }
    if (nudged)
        atomicAdd(&v.counters->nudged_bars, (unsigned long long)nudged);
    raw_out = raw;
    lim_out = lim;
}
template <int N> __global__ void __launch_bounds__(128, 4) k_limiter(View v) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    double raw = 0, lim = 0;
    if (e < v.cells)
        limit_element<N>(v, e, raw, lim);
    warp_stat_max(&v.counters->max_raw_bits, raw);
    warp_stat_max(&v.counters->max_limited_bits, lim);
}
// One thread per (element, local node) for N = 4 or 8 (a power of two): node-local bar states and contributions,
// element sums through warp shuffles in the host's sequential order. Elements whose factor needs the segment search
// go to k_limiter_segment, elements with a stable base bar to the host (as in k_limiter).
// NG = N rounded up to a power of two (lanes i >= N of a group are idle but take part in the shuffles).
template <int N, int NG> __global__ void __launch_bounds__(256) k_limiter_group(View v) {
    const unsigned FULL = 0xffffffffu;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int e = t / NG, i = t % NG, lane = threadIdx.x & 31, base = lane & ~(NG - 1);
    const bool valid = e < v.cells && i < N;
    const int dim = v.dimension;
    const std::size_t NN = v.nodes;
    double ub[4] = {0, 0, 0, 0}, fi[4] = {0, 0, 0, 0}, ui[4] = {0, 0, 0, 0};
    bool bad = false, uncertain = false;
    const int kc = e < v.cells ? klass_of(v, e) : 0;
    const std::size_t po = std::size_t(kc) * N * (N - 1);
    const double tv = i < N ? v.total_viscosity[std::size_t(kc) * N + i] : 1.;
    if (valid) {
        const int *map = v.connectivity + std::size_t(e) * N;
        const int ni = map[i];
        double qi[4], Fi[12], Gi[12];
        for (int r = 0; r < 4; ++r)
            ui[r] = v.low[r * NN + ni], qi[r] = v.moment_rate[r * NN + ni];
        for (int r = 0; r < 4 * dim; ++r)
            Fi[r] = v.low_flux[r * NN + ni], Gi[r] = v.closure[r * NN + ni];
        double weighted[4] = {0, 0, 0, 0};
        for (int k = 0; k < N - 1; ++k) {
            const std::size_t pk = po + i * (N - 1) + k;
            const int j = k < i ? k : k + 1, nj = map[j];
            const double dij = v.pair_d[pk];
            double uj[4], fd[4] = {0, 0, 0, 0}, ct[4] = {0, 0, 0, 0};
            for (int r = 0; r < 4; ++r)
                uj[r] = v.low[r * NN + nj];
            for (int d = 0; d < dim; ++d) {
                const double cij = v.pair_c[3 * pk + d], cji = v.pair_ct[3 * pk + d];
                for (int r = 0; r < 4; ++r) {
                    fd[r] += cij * (v.low_flux[(4 * d + r) * NN + nj] - Fi[4 * d + r]);
                    ct[r] += cji * Gi[4 * d + r] - cij * v.closure[(4 * d + r) * NN + nj];
                }
            }
            if (dij > 0) {
                double bar[4];
                for (int r = 0; r < 4; ++r)
                    bar[r] = .5 * (ui[r] + uj[r]) - fd[r] / (2 * dij);
                if (!cone(bar, uncertain))
                    bad = true;
                for (int r = 0; r < 4; ++r) {
                    weighted[r] += dij * bar[r];
                    fi[r] += dij * (ui[r] - uj[r]);
                }
            }
            for (int r = 0; r < 4; ++r)
                fi[r] += v.pair_m[pk] * (qi[r] - v.moment_rate[r * NN + nj]) + ct[r];
        }
        if (v.stabilize)
            for (int r = 0; r < 4; ++r)
                fi[r] -= v.stab_low[std::size_t(4 * i + r) * v.cells + e];
        for (int r = 0; r < 4; ++r)
            ub[r] = weighted[r] / tv;
        if (!bad && nudge(ub, v.counters, uncertain))
            atomicAdd(&v.counters->nudged_bars, 1ull);
    }
    int flag = valid && (bad || uncertain);
#pragma unroll
    for (int o = 1; o < NG; o <<= 1)
        flag |= __shfl_xor_sync(FULL, flag, o);
    // Raw zero sum (statistics) in node order.
    double raw = 0;
    {
        double sum[4] = {0, 0, 0, 0};
#pragma unroll
        for (int k = 0; k < N; ++k)
            for (int r = 0; r < 4; ++r)
                sum[r] += __shfl_sync(FULL, fi[r], base + k);
        for (int r = 0; r < 4; ++r)
            raw = fmax(raw, fabs(sum[r]));
    }
    double lim[4] = {fi[0], fi[1], fi[2], fi[3]};
    int need = 0;
    if (v.limit) {
        if (valid && !flag) {
            double tt[4];
            for (int r = 0; r < 4; ++r)
                tt[r] = ub[r] + lim[r] / (2 * tv);
            need = !cone(tt, uncertain);
            if (uncertain)
                flag = 1; // only this lane knows; handled below with the group vote
        }
    }
    int vote = need | (flag << 1);
#pragma unroll
    for (int o = 1; o < NG; o <<= 1)
        vote |= __shfl_xor_sync(FULL, vote, o);
    double after = 0;
    if (valid) {
        if (vote & 2) {
            if (i == 0) {
                const int sidx = atomicAdd(&v.counters->limiter_fallbacks, 1);
                if (sidx < v.fallback_capacity)
                    v.limiter_list[sidx] = e;
            }
        } else if (vote & 1) {
            for (int r = 0; r < 4; ++r) {
                v.seg_ubar[std::size_t(e) * 4 * N + 4 * i + r] = ub[r];
                v.seg_lim[std::size_t(e) * 4 * N + 4 * i + r] = lim[r];
            }
            if (i == 0)
                v.segment_list[atomicAdd(&v.counters->segments, 1)] = e;
        } else
            for (int r = 0; r < 4; ++r)
                v.element_limited[std::size_t(4 * i + r) * v.cells + e] = 2 * tv * (ub[r] - ui[r]) + lim[r];
    }
    {
        double sum[4] = {0, 0, 0, 0};
#pragma unroll
        for (int k = 0; k < N; ++k)
            for (int r = 0; r < 4; ++r)
                sum[r] += __shfl_sync(FULL, lim[r], base + k);
        for (int r = 0; r < 4; ++r)
            after = fmax(after, fabs(sum[r]));
    }
    const bool done = valid && !vote;
    warp_stat_max(&v.counters->max_raw_bits, valid && !(vote & 2) ? raw : 0.);
    warp_stat_max(&v.counters->max_limited_bits, done ? after : 0.);
}
template <int N> __global__ void k_limiter_segment(View v) {
    constexpr int NA = N > 0 ? N : max_local;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= v.counters->segments)
        return;
    const int e = v.segment_list[idx], n = N > 0 ? N : v.local;
    const int *map = v.connectivity + std::size_t(e) * n;
    const double *tvis = v.total_viscosity + std::size_t(klass_of(v, e)) * n;
    double ubar[NA][4], lim[NA][4];
    for (int i = 0; i < n; ++i)
        for (int r = 0; r < 4; ++r) {
            ubar[i][r] = v.seg_ubar[std::size_t(e) * 4 * n + 4 * i + r];
            lim[i][r] = v.seg_lim[std::size_t(e) * 4 * n + 4 * i + r];
        }
    bool uncertain = false;
    double factor = 1;
    for (int i = 0; i < n; ++i) {
        double g[4], t[4];
        for (int r = 0; r < 4; ++r) {
            g[r] = lim[i][r] / (2 * tvis[i]);
            t[r] = ubar[i][r] + g[r];
        }
        if (cone(t, uncertain))
            continue;
        factor = fmin(factor, v.element_theta * segment(ubar[i], g, v.counters, uncertain));
    }
    if (uncertain) {
        const int s = atomicAdd(&v.counters->limiter_fallbacks, 1);
        if (s < v.fallback_capacity)
            v.limiter_list[s] = e;
        return;
    }
    double after[4] = {0, 0, 0, 0};
    for (int i = 0; i < n; ++i)
        for (int r = 0; r < 4; ++r) {
            lim[i][r] *= factor;
            after[r] += lim[i][r];
        }
    double sum = 0;
    for (int r = 0; r < 4; ++r)
        sum = fmax(sum, fabs(after[r]));
    for (int i = 0; i < n; ++i) {
        for (int r = 0; r < 4; ++r) {
            const double lo = 2 * tvis[i] * (ubar[i][r] - v.low[std::size_t(r) * v.nodes + map[i]]);
            v.element_limited[std::size_t(4 * i + r) * v.cells + e] = lo + lim[i][r];
        }
    }
    DeviceCounters *C = v.counters;
    if (factor < 1) {
        atomicAdd(&C->limited_elements, 1ull);
        stat_min(&C->min_limiter_bits, factor);
    }
    stat_max(&C->max_limited_bits, sum);
}
__global__ void k_gather_load(View v) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(v.nodes) * 4)
        return;
    const int i = int(t / 4), c = int(t % 4);
    double s = 0;
    for (int k = v.node_offsets[i]; k < v.node_offsets[i + 1]; ++k)
    {
        const int entry = v.node_entries[k], e = entry / v.local, l = entry % v.local;
        s += v.element_limited[std::size_t(4 * l + c) * v.cells + e];
    }
    v.load[t] = s;
}
// with_flux: r is the mass-solved residual without the advective term; the term -sum_d A_d^T g_d is added here for
// the PN moments (the four low rows are replaced by the limited load anyway).
// part 0: all entries; part 1: the PN rows c >= 4 only (independent of the limited load, queued before the boundary
// corrections); part 2: the four low rows only (grid over nodes x 4).
__global__ void k_combine(View v, double *r, const double *x, const double *a, double old, double fresh, double tau,
                          bool with_flux, int part) {
    extern __shared__ double shared[];
    FluxShared F{};
    if (with_flux)
        F = stage_flux(v, shared);
    const unsigned u = blockIdx.x * blockDim.x + threadIdx.x, q = v.q;
    unsigned i, c;
    if (part == 2) {
        if (u >= unsigned(v.nodes) * 4u)
            return;
        i = u / 4u, c = u % 4u;
    } else {
        if (u >= unsigned(v.nodes) * q)
            return;
        i = u / q, c = u % q;
        if (part == 1 && c < 4)
            return;
    }
    const std::size_t t = std::size_t(i) * q + c;
    double rate = r[t];
    if (with_flux && c >= 4) {
        for (int d = 0; d < v.dimension; ++d) {
            const double *g = v.gradient[d] + std::size_t(i) * q;
            rate += -F.row(g, d, c);
        }
    }
    if (c < 4) {
        // set_low(row, load / lumped): u0 = v0 / s0, u3 = v1 / s1, u1 = v2 / s1, u2 = v3 / s1
        const int slot = c == 0 ? 0 : (c == 3 ? 1 : (c == 1 ? 2 : 3));
        rate = (v.load[std::size_t(i) * 4 + slot] / v.lumped[i]) / (c == 0 ? v.s0 : v.s1);
    }
    if (v.node_mirror && (v.node_mirror[i] & v.comp_odd[c])) // odd moments vanish on the mirror planes
        rate = 0;
    r[t] = old == 0 ? fresh * (x[t] + tau * rate) : old * a[t] + fresh * (x[t] + tau * rate);
}
__device__ __forceinline__ void admissible_node(const View &v, double *r, int i);
// closure_next: also the closure of the next stage (its input is r), fused into the same pass over the rows.
// low: first the stage combination of the four low rows (rate = load / lumped), as k_combine part 2.
__global__ void k_admissible(View v, double *r, bool closure_next, bool low, const double *x, const double *a, double old,
                             double fresh, double tau) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= v.nodes)
        return;
    if (low)
        for (int c = 0; c < 4; ++c) {
            const std::size_t t = std::size_t(i) * v.q + c;
            const int slot = c == 0 ? 0 : (c == 3 ? 1 : (c == 1 ? 2 : 3));
            double rate = (v.load[std::size_t(i) * 4 + slot] / v.lumped[i]) / (c == 0 ? v.s0 : v.s1);
            if (v.node_mirror && (v.node_mirror[i] & v.comp_odd[c]))
                rate = 0;
            r[t] = old == 0 ? fresh * (x[t] + tau * rate) : old * a[t] + fresh * (x[t] + tau * rate);
        }
    admissible_node(v, r, i);
    if (closure_next)
        closure_node(v, r, i);
}
__device__ __forceinline__ void admissible_node(const View &v, double *r, int i) {
    double *row = r + std::size_t(i) * v.q;
    double u[4];
    bool uncertain = false;
    project4(v, row, u);
    if (cone(u, uncertain) && !uncertain)
        return;
    const double before = row[0];
    for (int k = 0; k < 4096; ++k) {
        project4(v, row, u);
        if (cone(u, uncertain))
            break;
        row[0] = ::nextafter(row[0], dinf());
    }
    project4(v, row, u);
    if (!cone(u, uncertain) && fabs(row[0]) < 1e-290) {
        row[0] = fmax(row[0], norm_up(u) / v.s0);
        for (int k = 0; k < 4096; ++k) {
            project4(v, row, u);
            if (cone(u, uncertain))
                break;
            row[0] = ::nextafter(row[0], dinf());
        }
    }
    project4(v, row, u);
    if (!cone(u, uncertain) || uncertain) {
        fail(v.counters, uncertain ? 7 : 6);


        return;
    }
    if (row[0] != before) {
        atomicAdd(&v.counters->nudged_rows, 1ull);
        atomicAdd(&v.counters->nudge_mass, v.lumped[i] * v.s0 * (row[0] - before));
    }
}
__global__ void k_sumsq(const double *f, std::size_t count, double *out) {
    __shared__ double part[256];
    double s = 0;
    for (std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; t < count;
         t += std::size_t(gridDim.x) * blockDim.x)
        s += f[t] * f[t];
    part[threadIdx.x] = s;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w /= 2) {
        if (threadIdx.x < w)
            part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) // one partial per block, summed on the host in block order (deterministic; an atomicAdd made
        out[blockIdx.x] = part[0]; // the power-iteration norm, hence dt, vary in the last bit from run to run)
}
__global__ void k_divide(double *dst, const double *src, std::size_t count, double divisor) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t < count)
        dst[t] = src[t] / divisor;
}
// rows_soa > 0: column-major field with that many rows.
__global__ void k_gather(const double *f, int cols, const int *index, int count, double *out, long long rows_soa) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(count) * cols)
        return;
    const int k = int(t / cols), c = int(t % cols);
    out[t] = f[rows_soa ? std::size_t(c) * rows_soa + index[k] : std::size_t(index[k]) * cols + c];
}
__global__ void k_gather_cols(const double *f, int stride_cols, int copy_cols, const int *index, int count, double *out) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(count) * copy_cols)
        return;
    const int k = int(t / copy_cols), c = int(t % copy_cols);
    out[t] = f[std::size_t(index[k]) * stride_cols + c];
}
__global__ void k_scatter(double *f, int cols, const int *index, int count, const double *in, bool add,
                          long long rows_soa) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(count) * cols)
        return;
    const int k = int(t / cols), c = int(t % cols);
    double &dst = f[rows_soa ? std::size_t(c) * rows_soa + index[k] : std::size_t(index[k]) * cols + c];
    dst = add ? dst + in[t] : in[t];
}
namespace {
inline unsigned blocks(std::size_t work, int threads = 256) { return unsigned((work + threads - 1) / threads); }
} // namespace

// ---------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------
// Positivity projection of the ghost traces (src/positivity_projection.cpp, plain rule), one warp per ghost pair.
struct GhostDeviceStats {
    unsigned long long calls, identities, vacuum, failures, scaled, capped, iterations, backtracks;
    int max_iterations, flagged;
    unsigned long long max_residual_bits, min_active_bits;
};
struct GhostView {
    int pairs, max_points, q;
    const int *node, *side;
    int points[6], outgoing[6], facets[6];
    const double *harmonics[6], *ox[6], *oy[6], *oz[6], *w[6], *fnx[6], *fny[6], *fnz[6], *fh[6], *normal[6];
    double inradius[6];
    const double *sigma;
    double tolerance;
    bool warm_start;
    double *xi;       // pairs x 4: warm start
    double *moments;  // pairs x 4: outgoing ghost moments
    int *flag_list;   // flagged pairs (host)
    GhostDeviceStats *stats;
    const double *Y;          // traces from the per-side DGEMM (nullptr: k_ghost forms them itself)
    const std::size_t *ycol;  // pairs: offset of the trace column of pair k in Y
};
struct GhostDevice {
    GhostView g;
    int warps = 4; // warps per block of k_ghost (dynamic shared memory: warps x max_points doubles)
    std::vector<void *> owned;
    // Traces as one DGEMM per side, Y_s = H_s^T A_s (points x pairs of the side): H_s is read once per call instead of
    // once per pair (P35: 29 MB per side and pair, the bottleneck of uniform high-order runs). M1PN_GPU_GHOST_GEMM=0
    // keeps the per-warp dot products.
    bool gemm = false;
    int count[6] = {};
    int *nodes[6] = {};
    std::size_t offset[6] = {};
    double *A = nullptr, *Y = nullptr;
    cublasHandle_t handle = nullptr;
};
__device__ inline double warp_sum(double v) {
#pragma unroll
    for (int o = 16; o > 0; o /= 2)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ inline double warp_min(double v) {
#pragma unroll
    for (int o = 16; o > 0; o /= 2)
        v = fmin(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ inline double warp_max(double v) {
#pragma unroll
    for (int o = 16; o > 0; o /= 2)
        v = fmax(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
struct GEval {
    double F[4], H[10], phi, min_xm; // H: 00 01 02 03 11 12 13 22 23 33
    int active;
};
// One pass over the points at multiplier xi (all lanes get the same reduced values).
__device__ GEval gevaluate(int P, const double *ox, const double *oy, const double *oz, const double *w,
                           const double *y, const double *U, const double *xi) {
    double f0 = 0, f1 = 0, f2 = 0, f3 = 0, zz = 0, mn = dinf(), h[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double act = 0;
    for (int k = threadIdx.x & 31; k < P; k += 32) {
        const double xm = xi[0] + xi[1] * ox[k] + xi[2] * oy[k] + xi[3] * oz[k];
        const double r = y[k] - xm;
        const double pos = r > 0 ? 1. : 0.;
        const double z = pos * r, wz = w[k] * z, wa = w[k] * pos;
        f0 += wz, f1 += wz * ox[k], f2 += wz * oy[k], f3 += wz * oz[k], zz += wz * z;
        h[0] += wa, h[1] += wa * ox[k], h[2] += wa * oy[k], h[3] += wa * oz[k];
        h[4] += wa * ox[k] * ox[k], h[5] += wa * ox[k] * oy[k], h[6] += wa * ox[k] * oz[k];
        h[7] += wa * oy[k] * oy[k], h[8] += wa * oy[k] * oz[k], h[9] += wa * oz[k] * oz[k];
        act += pos;
        mn = fmin(mn, xm);
    }
    GEval e;
    e.F[0] = warp_sum(f0) - U[0], e.F[1] = warp_sum(f1) - U[1], e.F[2] = warp_sum(f2) - U[2];
    e.F[3] = warp_sum(f3) - U[3];
    for (int k = 0; k < 10; ++k)
        e.H[k] = warp_sum(h[k]);
    e.phi = -.5 * warp_sum(zz) - (xi[0] * U[0] + xi[1] * U[1] + xi[2] * U[2] + xi[3] * U[3]);
    e.min_xm = warp_min(mn);
    e.active = int(warp_sum(act) + .5);
    return e;
}
__device__ inline double maxabs4(const double *v) {
    return fmax(fmax(fabs(v[0]), fabs(v[1])), fmax(fabs(v[2]), fabs(v[3])));
}
// (H + mu I) d = F by LDL^T with symmetric diagonal pivoting; false if a pivot is not positive.
__device__ bool ldlt_solve4(const double *Hp, double mu, const double *F, double *d) {
    double A[4][4];
    const int map[4][4] = {{0, 1, 2, 3}, {1, 4, 5, 6}, {2, 5, 7, 8}, {3, 6, 8, 9}};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            A[i][j] = Hp[map[i][j]] + (i == j ? mu : 0.);
    int perm[4] = {0, 1, 2, 3};
    double D[4];
    for (int k = 0; k < 4; ++k) {
        int piv = k;
        for (int i = k + 1; i < 4; ++i)
            if (fabs(A[i][i]) > fabs(A[piv][piv]))
                piv = i;
        if (piv != k) {
            for (int j = 0; j < 4; ++j) {
                const double t = A[k][j];
                A[k][j] = A[piv][j], A[piv][j] = t;
            }
            for (int i = 0; i < 4; ++i) {
                const double t = A[i][k];
                A[i][k] = A[i][piv], A[i][piv] = t;
            }
            const int t = perm[k];
            perm[k] = perm[piv], perm[piv] = t;
        }
        D[k] = A[k][k];
        if (!(D[k] > 0) || !isfinite(D[k]))
            return false;
        for (int i = k + 1; i < 4; ++i)
            A[i][k] /= D[k];
        for (int i = k + 1; i < 4; ++i)
            for (int j = k + 1; j <= i; ++j) {
                A[i][j] -= A[i][k] * A[j][k] * D[k];
                A[j][i] = A[i][j];
            }
    }
    double x[4];
    for (int i = 0; i < 4; ++i)
        x[i] = F[perm[i]];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < i; ++j)
            x[i] -= A[i][j] * x[j];
    for (int i = 0; i < 4; ++i)
        x[i] /= D[i];
    for (int i = 3; i >= 0; --i)
        for (int j = i + 1; j < 4; ++j)
            x[i] -= A[j][i] * x[j];
    for (int i = 0; i < 4; ++i)
        d[perm[i]] = x[i];
    return true;
}
struct GResult {
    bool converged, identity, infeasible;
    int iterations, backtracks, active;
    double residual, xi[4];
};
// project_positive() of src/positivity_projection.cpp (unweighted); y in shared memory, z not stored.
__device__ GResult gproject(int P, const double *ox, const double *oy, const double *oz, const double *w,
                            const double *y, const double *U, const double *xi0, double tolerance) {
    GResult out{};
    double sy = 0, ymin = dinf(), F0[4] = {0, 0, 0, 0};
    for (int k = threadIdx.x & 31; k < P; k += 32) {
        const double wy = w[k] * y[k];
        sy += fabs(wy);
        ymin = fmin(ymin, y[k]);
        F0[0] += wy, F0[1] += wy * ox[k], F0[2] += wy * oy[k], F0[3] += wy * oz[k];
    }
    sy = warp_sum(sy), ymin = warp_min(ymin);
    for (int c = 0; c < 4; ++c)
        F0[c] = warp_sum(F0[c]) - U[c];
    const double scale = sy + fabs(U[0]), tol = tolerance * scale;
    if (ymin > 0 && maxabs4(F0) <= tol) {
        out.identity = out.converged = true;
        out.active = P;
        out.residual = maxabs4(F0) / scale;
        return out;
    }
    double xi[4];
    bool finite = true;
    for (int c = 0; c < 4; ++c)
        finite &= isfinite(xi0[c]);
    for (int c = 0; c < 4; ++c)
        xi[c] = finite ? xi0[c] : 0.;
    GEval e = gevaluate(P, ox, oy, oz, w, y, U, xi);
    auto done = [&]() {
        for (int c = 0; c < 4; ++c)
            out.xi[c] = xi[c];
        out.active = e.active;
        out.residual = scale > 0 ? maxabs4(e.F) / scale : maxabs4(e.F);
        return out;
    };
    for (int it = 0; it < 60; ++it) {
        if (maxabs4(e.F) <= tol) {
            out.converged = true;
            break;
        }
        const double xu = xi[0] * U[0] + xi[1] * U[1] + xi[2] * U[2] + xi[3] * U[3];
        const double xn = sqrt(xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2] + xi[3] * xi[3]);
        if (e.min_xm >= 0 && xu < -1e-14 * xn * U[0]) {
            out.infeasible = true;
            return done();
        }
        const double trace = e.H[0] + e.H[4] + e.H[7] + e.H[9];
        double mu = 1e-14 * fmax(trace, 1e-300), d[4];
        for (int attempt = 0;; ++attempt) {
            if (ldlt_solve4(e.H, mu, e.F, d)) {
                const double df = d[0] * e.F[0] + d[1] * e.F[1] + d[2] * e.F[2] + d[3] * e.F[3];
                if (isfinite(d[0]) && isfinite(d[1]) && isfinite(d[2]) && isfinite(d[3]) && df > 0)
                    break;
            }
            if (attempt > 30)
                return done();
            mu = fmax(mu * 100, 1e-12 * fmax(trace, 1e-300));
        }
        const double slope = d[0] * e.F[0] + d[1] * e.F[1] + d[2] * e.F[2] + d[3] * e.F[3];
        double t = 1;
        GEval trial;
        bool accepted = false;
        for (int b = 0; b <= 40; ++b) {
            double cand[4];
            for (int c = 0; c < 4; ++c)
                cand[c] = xi[c] + t * d[c];
            trial = gevaluate(P, ox, oy, oz, w, y, U, cand);
            if (trial.phi >= e.phi + 1e-4 * t * slope || maxabs4(trial.F) <= tol) {
                for (int c = 0; c < 4; ++c)
                    xi[c] = cand[c];
                accepted = true;
                break;
            }
            t *= .5;
            ++out.backtracks;
        }
        ++out.iterations;
        if (!accepted || !(isfinite(xi[0]) && isfinite(xi[1]) && isfinite(xi[2]) && isfinite(xi[3])))
            return done();
        e = trial;
    }
    if (!out.converged && maxabs4(e.F) <= tol)
        out.converged = true;
    return done();
}
__global__ void __launch_bounds__(128) k_ghost(View v, GhostView g, const double *a) {
    extern __shared__ double ys[]; // max_points per warp
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int k = blockIdx.x * (blockDim.x >> 5) + warp;
    if (k >= g.pairs)
        return; // whole warps exit together
    double *y = ys + std::size_t(warp) * g.max_points;
    const int node = g.node[k], sd = g.side[k], P = g.points[sd], Pout = g.outgoing[sd], q = g.q;
    const double *ox = g.ox[sd], *oy = g.oy[sd], *oz = g.oz[sd], *w = g.w[sd];
    const double *row = a + std::size_t(node) * q;
    // PN trace at the rule points (filtered coefficients if a ghost filter is set).
    if (g.Y) {
        const double *Yk = g.Y + g.ycol[k];
        for (int j = lane; j < P; j += 32)
            y[j] = Yk[j];
    } else {
        for (int j = lane; j < P; j += 32) {
            const double *Hj = g.harmonics[sd] + std::size_t(j) * q;
            double s = 0;
            for (int c = 0; c < q; ++c)
                s += Hj[c] * (g.sigma ? g.sigma[c] * row[c] : row[c]);
            y[j] = s;
        }
    }
    __syncwarp();
    double U[4];
    for (int c = 0; c < 4; ++c)
        U[c] = v.low[std::size_t(c) * v.nodes + node];
    const double *nrm = g.normal[sd];
    GhostDeviceStats *S = g.stats;
    auto positive_part = [&](double *m) {
        double m0 = 0, m1 = 0, m2 = 0, m3 = 0;
        for (int j = lane; j < Pout; j += 32) {
            const double val = fmax(0., y[j]) * w[j];
            m0 += val, m1 += val * ox[j], m2 += val * oy[j], m3 += val * oz[j];
        }
        m[0] = warp_sum(m0), m[1] = warp_sum(m1), m[2] = warp_sum(m2), m[3] = warp_sum(m3);
    };
    auto flag = [&]() {
        if (lane == 0) {
            const int f = atomicAdd(&S->flagged, 1);
            g.flag_list[f] = k;
        }
    };
    double m[4] = {0, 0, 0, 0};
    bool projected = false;
    const double rho = U[0], jn = sqrt(U[1] * U[1] + U[2] * U[2] + U[3] * U[3]);
    const bool finiteU = isfinite(U[0]) && isfinite(U[1]) && isfinite(U[2]) && isfinite(U[3]);
    if (lane == 0)
        atomicAdd(&S->calls, 1ull);
    if (!finiteU || rho < jn) { // outside R1: failure, C6 ghost
        if (lane == 0)
            atomicAdd(&S->failures, 1ull);
        positive_part(m);
    } else if (!(rho > 0)) { // vacuum: z = 0, converged (J = 0 here)
        projected = true;
        if (lane == 0)
            atomicAdd(&S->vacuum, 1ull);
    } else {
        // Hull certificate (exact solvability test of the plain rule): such pairs go to the host.
        bool certificate = false;
        if (jn >= .999 * g.inradius[sd] * rho) {
            const double ux = U[1] / rho, uy = U[2] / rho, uz = U[3] / rho;
            double mg = -dinf();
            for (int f = lane; f < g.facets[sd]; f += 32)
                mg = fmax(mg, g.fnx[sd][f] * ux + g.fny[sd][f] * uy + g.fnz[sd][f] * uz - g.fh[sd][f]);
            certificate = warp_max(mg) > -1e-9;
        }
        if (certificate) {
            flag();
            return;
        }
        double xi0[4];
        for (int c = 0; c < 4; ++c)
            xi0[c] = g.warm_start ? g.xi[std::size_t(k) * 4 + c] : 0.;
        GResult r = gproject(P, ox, oy, oz, w, y, U, xi0, g.tolerance);
        int newton = r.iterations;
        const bool warm_nonzero = xi0[0] != 0 || xi0[1] != 0 || xi0[2] != 0 || xi0[3] != 0;
        if (!r.converged && !r.infeasible && g.warm_start && warm_nonzero) {
            const double zero[4] = {0, 0, 0, 0};
            r = gproject(P, ox, oy, oz, w, y, U, zero, g.tolerance);
            r.iterations += newton;
            newton = r.iterations;
        }
        if (!r.converged) { // Farkas or no convergence: augmented rule on the host
            flag();
            return;
        }
        projected = true;
        if (lane == 0) {
            for (int c = 0; c < 4; ++c)
                g.xi[std::size_t(k) * 4 + c] = r.xi[c];
            if (r.identity)
                atomicAdd(&S->identities, 1ull);
            atomicAdd(&S->iterations, (unsigned long long)newton);
            atomicAdd(&S->backtracks, (unsigned long long)r.backtracks);
            atomicMax(&S->max_iterations, newton);
            atomicMax(&S->max_residual_bits, bits(r.residual));
            atomicMin(&S->min_active_bits, bits(double(r.active) / P));
        }
        // Outgoing half moments of z = max(0, y - xi.m) (z = y for the identity).
        double m0 = 0, m1 = 0, m2 = 0, m3 = 0;
        for (int j = lane; j < Pout; j += 32) {
            const double xm = r.xi[0] + r.xi[1] * ox[j] + r.xi[2] * oy[j] + r.xi[3] * oz[j];
            const double z = r.identity ? y[j] : fmax(0., y[j] - xm);
            const double val = z * w[j];
            m0 += val, m1 += val * ox[j], m2 += val * oy[j], m3 += val * oz[j];
        }
        m[0] = warp_sum(m0), m[1] = warp_sum(m1), m[2] = warp_sum(m2), m[3] = warp_sum(m3);
        if (m[0] > U[0] * (1 + 64 * 2.220446049250313e-16)) {
            if (lane == 0)
                atomicAdd(&S->capped, 1ull);
            const double f = U[0] > 0 ? U[0] / m[0] : 0.;
            for (int c = 0; c < 4; ++c)
                m[c] *= f;
        }
    }
    // Vacuum outflow property: a_ext <= b of the node (cone scaling otherwise).
    const double b_node = U[0] + nrm[0] * U[1] + nrm[1] * U[2] + nrm[2] * U[3];
    const double a_ext = m[0] - (nrm[0] * m[1] + nrm[1] * m[2] + nrm[2] * m[3]);
    if (a_ext > b_node) {
        if (projected && lane == 0)
            atomicAdd(&S->scaled, 1ull);
        const double f = b_node > 0 ? b_node / a_ext : 0.;
        for (int c = 0; c < 4; ++c)
            m[c] *= f;
    }
    if (lane == 0)
        for (int c = 0; c < 4; ++c)
            g.moments[std::size_t(k) * 4 + c] = m[c];
}
inline void cublas_ok(cublasStatus_t e, const char *what) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLAS: ") + what + " failed (" + std::to_string(int(e)) + ")");
}
__global__ void k_gather_ghost_rows_sigma(int q, int count, const int *nodes, const double *state, const double *sigma,
                                          double *A) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(count) * q)
        return;
    const int j = int(t / q), c = int(t % q);
    const double v = state[std::size_t(nodes[j]) * q + c];
    A[t] = sigma ? sigma[c] * v : v;
}
void ghost_setup(Context *ctx, const GhostSetup &s) {
    GhostDevice *gd = new GhostDevice;
    GhostView &g = gd->g;
    auto own = [&](auto *p) {
        gd->owned.push_back((void *)p);
        return p;
    };
    g.pairs = s.pairs, g.max_points = s.max_points, g.q = ctx->v.q;
    {
        // Dynamic shared memory of k_ghost: fewer warps per block for large rules, then the opt-in limit.
        int device = 0, optin = 0;
        CK(cudaGetDevice(&device));
        CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        const std::size_t per_warp = std::size_t(s.max_points) * sizeof(double);
        while (gd->warps > 1 && gd->warps * per_warp > 48 * 1024)
            gd->warps /= 2;
        const std::size_t bytes = gd->warps * per_warp;
        if (bytes > std::size_t(optin))
            throw std::runtime_error("GPU ghost projection: rule too large for shared memory");
        if (bytes > 48 * 1024)
            CK(cudaFuncSetAttribute(k_ghost, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
    }
    g.node = own(dcopy(s.node, std::size_t(s.pairs)));
    g.side = own(dcopy(s.side, std::size_t(s.pairs)));
    for (int sd = 0; sd < 6; ++sd) {
        g.points[sd] = s.points[sd], g.outgoing[sd] = s.outgoing[sd], g.facets[sd] = s.facets[sd];
        g.inradius[sd] = s.inradius[sd];
        if (!s.points[sd]) {
            g.harmonics[sd] = g.ox[sd] = g.oy[sd] = g.oz[sd] = g.w[sd] = nullptr;
            g.fnx[sd] = g.fny[sd] = g.fnz[sd] = g.fh[sd] = g.normal[sd] = nullptr;
            continue;
        }
        const std::size_t P = s.points[sd], F = s.facets[sd];
        g.harmonics[sd] = own(dcopy(s.harmonics[sd], P * g.q));
        g.ox[sd] = own(dcopy(s.ox[sd], P)), g.oy[sd] = own(dcopy(s.oy[sd], P)), g.oz[sd] = own(dcopy(s.oz[sd], P));
        g.w[sd] = own(dcopy(s.w[sd], P));
        g.fnx[sd] = own(dcopy(s.fnx[sd], F)), g.fny[sd] = own(dcopy(s.fny[sd], F)), g.fnz[sd] = own(dcopy(s.fnz[sd], F));
        g.fh[sd] = own(dcopy(s.fh[sd], F));
        g.normal[sd] = own(dcopy(s.normal[sd], 3));
    }
    g.sigma = s.sigma ? own(dcopy(s.sigma, std::size_t(g.q))) : nullptr;
    g.tolerance = s.tolerance, g.warm_start = s.warm_start;
    g.xi = own(dalloc<double>(std::size_t(s.pairs) * 4));
    g.moments = own(dalloc<double>(std::size_t(s.pairs) * 4));
    g.flag_list = own(dalloc<int>(std::size_t(s.pairs) + 1));
    g.stats = own(dalloc<GhostDeviceStats>(1));
    GhostDeviceStats st{};
    const double one = 1;
    std::memcpy(&st.min_active_bits, &one, 8);
    CK(cudaMemcpy(g.stats, &st, sizeof st, cudaMemcpyHostToDevice));
    g.Y = nullptr, g.ycol = nullptr;
    ctx->ghost = gd;
    int least = 0, greatest = 0; // numerically larger = lower priority
    CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    CK(cudaStreamCreateWithPriority(&ctx->aux, cudaStreamNonBlocking, least));
    {
        const char *e = std::getenv("M1PN_GPU_GHOST_GEMM");
        gd->gemm = e ? std::atoi(e) != 0 : true;
    }
    if (gd->gemm) {
        std::array<std::vector<int>, 6> nodes;
        std::vector<std::size_t> ycol(std::size_t(s.pairs));
        std::array<std::vector<int>, 6> members;
        for (int k = 0; k < s.pairs; ++k)
            members[s.side[k]].push_back(k);
        std::size_t total = 0;
        int max_count = 0;
        for (int sd = 0; sd < 6; ++sd) {
            gd->offset[sd] = total;
            gd->count[sd] = int(members[sd].size());
            if (!gd->count[sd])
                continue;
            for (std::size_t j = 0; j < members[sd].size(); ++j) {
                nodes[sd].push_back(s.node[members[sd][j]]);
                ycol[members[sd][j]] = total + j * std::size_t(s.points[sd]);
            }
            gd->nodes[sd] = own(dcopy(nodes[sd].data(), nodes[sd].size()));
            total += std::size_t(s.points[sd]) * gd->count[sd];
            max_count = std::max(max_count, gd->count[sd]);
        }
        gd->A = own(dalloc<double>(std::max<std::size_t>(1, std::size_t(max_count) * g.q)));
        gd->Y = own(dalloc<double>(std::max<std::size_t>(1, total)));
        g.Y = gd->Y;
        g.ycol = own(dcopy(ycol.data(), ycol.size()));
        cublas_ok(cublasCreate(&gd->handle), "create (ghost)");
        cublas_ok(cublasSetStream(gd->handle, ctx->aux), "set stream (ghost)");
    }
}
// On the aux stream after the work queued so far on the default stream (event 3); completion is event 2.
void ghost(Context *ctx, const double *state, double *pinned_moments, int *pinned_flags) {
    GhostView &g = ctx->ghost->g;
    cudaStream_t st = ctx->aux;
    CK(cudaEventRecord(ctx->events[3], ctx->stream));
    CK(cudaStreamWaitEvent(st, ctx->events[3], 0));
    CK(cudaMemsetAsync(&g.stats->flagged, 0, sizeof(int), st));
    if (ctx->ghost->gemm) {
        GhostDevice *gd = ctx->ghost;
        const double one = 1, zero = 0;
        for (int sd = 0; sd < 6; ++sd) {
            const int n = gd->count[sd], p = g.points[sd];
            if (!n)
                continue;
            // A_s (q x n, column-major) = filtered state rows of the side's ghost nodes; the gather buffer is reused,
            // the DGEMM of the previous side is ordered before it on the stream.
            k_gather_ghost_rows_sigma<<<blocks(std::size_t(n) * g.q), 256, 0, st>>>(g.q, n, gd->nodes[sd], state,
                                                                                    g.sigma, gd->A);
            // harmonics[sd] is q x P column-major (row j of the rule = column j): Y_s = H^T A_s, P x n.
            cublas_ok(cublasDgemm(gd->handle, CUBLAS_OP_T, CUBLAS_OP_N, p, n, g.q, &one, g.harmonics[sd], g.q,
                                     gd->A, g.q, &zero, gd->Y + gd->offset[sd], p),
                         "dgemm (ghost)");
        }
        CK(cudaGetLastError());
    }
    const int warps = ctx->ghost->warps;
    k_ghost<<<(g.pairs + warps - 1) / warps, 32 * warps, std::size_t(warps) * g.max_points * sizeof(double), st>>>(
        ctx->v, g, state);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(pinned_moments, g.moments, std::size_t(g.pairs) * 4 * sizeof(double), cudaMemcpyDeviceToHost, st));
    CK(cudaMemcpyAsync(pinned_flags, &g.stats->flagged, sizeof(int), cudaMemcpyDeviceToHost, st));
    CK(cudaMemcpyAsync(pinned_flags + 1, g.flag_list, std::size_t(g.pairs) * sizeof(int), cudaMemcpyDeviceToHost, st));
    CK(cudaEventRecord(ctx->events[2], st));
}
GhostStats ghost_stats(Context *ctx) {
    GhostDeviceStats d;
    CK(cudaMemcpy(&d, ctx->ghost->g.stats, sizeof d, cudaMemcpyDeviceToHost));
    GhostStats o;
    o.calls = d.calls, o.identities = d.identities, o.vacuum = d.vacuum, o.failures = d.failures, o.scaled = d.scaled;
    o.capped = d.capped, o.iterations = d.iterations, o.backtracks = d.backtracks;
    o.max_iterations = d.max_iterations;
    std::memcpy(&o.max_residual, &d.max_residual_bits, 8);
    std::memcpy(&o.min_active_fraction, &d.min_active_bits, 8);
    return o;
}

// ---------------------------------------------------------------------------------------------------------------
// Local reaction and FCS collision source.
struct ReactionView {
    const double *sigma_a, *sigma_s, *source, *filter;
    double sa, ss;
    int P; // FCS points per element (0: no FCS)
    const double *fx, *fy, *fw, *fss, *fshape, *fmass;
    double *q;      // cells x P
    double *local;  // cells x local
    double *fcs;    // nodes
    double *table;  // FCS radial table
    const double *ltable; // lattice FCS: points x (lM - lm0 + 1) time table (device), times m dt, m = lm0..lM
    const float *lnode_f;  // lattice-3d nodal FCS table (FP32): nodes x (lM + 1)
    const double *lnode_d; // lattice-3d nodal FCS table (FP64): nodes x (lM + 1)
    int lM, lm0;
    double ldt;
    double *ledger; // absorbed, injected
    bool has_filter;
};
struct ReactionDevice {
    ReactionView r;
    std::vector<void *> owned;
    std::vector<void *> lattice_owned; // the lattice table and its arrays (freed on a rebuild)
    std::size_t table_size = 0;
};
__global__ void k_fcs_points(View v, ReactionView r, int n, double dr, double cx, double cy) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(v.cells) * r.P)
        return;
    const double dx = r.fx[t] - cx, dy = r.fy[t] - cy, rr = sqrt(dx * dx + dy * dy) / dr;
    double rho = 0;
    if (rr < n - 2) {
        const int i = int(rr);
        const double f = rr - i, *p = r.table + i;
        const double val = p[1] + .5 * f * (p[2] - p[0] + f * (2 * p[0] - 5 * p[1] + 4 * p[2] - p[3] +
                                                              f * (3 * (p[1] - p[2]) + p[3] - p[0])));
        rho = fmax(val, 0.);
    }
    r.q[t] = r.fw[t] * r.fss[t] * rho;
}
// Lattice FCS: rho_u at the FCS points by cubic interpolation in time (same formula as LatticeUncollidedTable::value).
__global__ void k_fcs_points_lattice(View v, ReactionView r, double time) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(v.cells) * r.P)
        return;
    double rho = 0;
    if (time > 0) {
        const double s = time / r.ldt;
        int i = int(s);
        if (i > r.lM - 2)
            i = r.lM - 2; // the host table checks the range; the driver never asks beyond final_time
        if (i < r.lm0 + (r.lm0 > 0 ? 1 : 0))
            i = r.lm0 + (r.lm0 > 0 ? 1 : 0); // never asked before the window (driver)
        const double f = s - i, *row = r.ltable + t * std::size_t(r.lM - r.lm0 + 1) - r.lm0;
        const double p1 = row[i], p2 = row[i + 1], p3 = row[i + 2], p0 = i > 0 ? row[i - 1] : 2 * p1 - p2;
        const double val = p1 + .5 * f * (p2 - p0 + f * (2 * p0 - 5 * p1 + 4 * p2 - p3 + f * (3 * (p1 - p2) + p3 - p0)));
        rho = fmax(val, 0.);
    }
    r.q[t] = r.fw[t] * r.fss[t] * rho;
}
__global__ void k_lattice_table(uncollided::LatticeView lv, const double *x, const double *y, std::size_t points, int m0,
                                int M, double dt, double *table) {
    __shared__ double s_exp[1024];
    const int W = M - m0 + 1;
    const double sS = uncollided::lattice_sigma(lv, lv.si, lv.sj);
    if (W <= 1024) {
        for (int i = threadIdx.x; i < W; i += blockDim.x)
            s_exp[i] = exp(-sS * ((m0 + i) * dt));
        __syncthreads();
    }
    const std::size_t k = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= points)
        return;
    uncollided::lattice_density_row(lv, x[k], y[k], m0, M, dt, table + k * std::size_t(W), W <= 1024 ? s_exp : nullptr);
}
__global__ void k_fcs_element(View v, ReactionView r) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(v.cells) * v.local)
        return;
    const int e = int(t / v.local), l = int(t % v.local);
    double s = 0;
    for (int k = 0; k < r.P; ++k)
        s += r.fshape[std::size_t(l) * r.P + k] * r.q[std::size_t(e) * r.P + k];
    r.local[t] = s;
}
__global__ void k_fcs_gather(View v, ReactionView r) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= v.nodes)
        return;
    double s = 0;
    for (int k = v.node_offsets[i]; k < v.node_offsets[i + 1]; ++k)
        s += r.local[v.node_entries[k]];
    r.fcs[i] = s / r.fmass[i];
}
// One thread per (node, moment) (experiment/big-rebuilds): coalesced rows, the same formulas as k_react (bitwise);
// the moment-0 thread also updates the density and the ledger (diagnostic sums, atomic as before).
__global__ void k_react_rows(View v, ReactionView r, double *a, double tau, bool fcs) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    double absorbed = 0, injected = 0;
    if (t < std::size_t(v.nodes) * v.q) {
        const int i = int(t / v.q), k = int(t % v.q);
        const double sa = r.sigma_a ? r.sigma_a[i] : r.sa, ss = r.sigma_s ? r.sigma_s[i] : r.ss;
        if (k == 0) {
            const double q0 = (r.source ? r.source[i] : 0.) + (fcs ? r.fcs[i] : 0.);
            const double c0 = 1 / v.s0, z = -sa * tau, decay0 = exp(z), phi1 = z == 0 ? 1. : expm1(z) / z;
            double *row = a + std::size_t(i) * v.q;
            const double rho = row[0] / c0;
            row[0] = decay0 * row[0] + c0 * q0 * tau * phi1;
            absorbed = v.lumped[i] * (-expm1(z) * rho + q0 * tau * (1 - phi1));
            injected = v.lumped[i] * q0 * tau;
        } else
            a[t] *= exp(-tau * (sa + ss + r.filter[k]));
    }
    absorbed = warp_sum(absorbed), injected = warp_sum(injected);
    if ((threadIdx.x & 31) == 0 && (absorbed != 0 || injected != 0)) {
        atomicAdd(&r.ledger[0], absorbed);
        atomicAdd(&r.ledger[1], injected);
    }
}
__global__ void k_react(View v, ReactionView r, double *a, double tau, bool fcs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double absorbed = 0, injected = 0;
    if (i < v.nodes) {
        const double sa = r.sigma_a ? r.sigma_a[i] : r.sa, ss = r.sigma_s ? r.sigma_s[i] : r.ss;
        const double q0 = (r.source ? r.source[i] : 0.) + (fcs ? r.fcs[i] : 0.);
        const double c0 = 1 / v.s0, z = -sa * tau, decay0 = exp(z), phi1 = z == 0 ? 1. : expm1(z) / z;
        double *row = a + std::size_t(i) * v.q;
        const double rho = row[0] / c0;
        row[0] = decay0 * row[0] + c0 * q0 * tau * phi1;
        for (int k = 1; k < v.q; ++k)
            row[k] *= exp(-tau * (sa + ss + r.filter[k]));
        absorbed = v.lumped[i] * (-expm1(z) * rho + q0 * tau * (1 - phi1));
        injected = v.lumped[i] * q0 * tau;
    }
    absorbed = warp_sum(absorbed), injected = warp_sum(injected);
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&r.ledger[0], absorbed);
        atomicAdd(&r.ledger[1], injected);
    }
}
void reaction_setup(Context *ctx, const ReactionSetup &s) {
    ReactionDevice *rd = new ReactionDevice;
    ReactionView &r = rd->r;
    auto own = [&](auto *p) {
        rd->owned.push_back((void *)p);
        return p;
    };
    const View &v = ctx->v;
    const std::size_t N = v.nodes;
    r.sigma_a = s.sigma_a ? own(dcopy(s.sigma_a, N)) : nullptr;
    r.sigma_s = s.sigma_s ? own(dcopy(s.sigma_s, N)) : nullptr;
    r.source = s.source ? own(dcopy(s.source, N)) : nullptr;
    r.sa = s.sa, r.ss = s.ss;
    r.filter = own(dcopy(s.filter, std::size_t(v.q)));
    r.has_filter = false;
    if (s.filter)
        for (int k = 0; k < v.q; ++k)
            if (s.filter[k] != 0.0)
                r.has_filter = true;
    r.P = s.fcs_points;
    if (r.P) {
        const std::size_t CP = std::size_t(v.cells) * r.P;
        r.fx = own(dcopy(s.fcs_x, CP)), r.fy = own(dcopy(s.fcs_y, CP));
        r.fw = own(dcopy(s.fcs_weight, CP)), r.fss = own(dcopy(s.fcs_sigma_s, CP));
        r.fshape = own(dcopy(s.fcs_shape, std::size_t(v.local) * r.P));
        r.fmass = own(dcopy(s.fcs_node_mass, N));
        r.q = own(dalloc<double>(CP));
        r.local = own(dalloc<double>(std::size_t(v.cells) * v.local));
        r.fcs = own(dalloc<double>(N));
    } else
        r.fx = r.fy = r.fw = r.fss = r.fshape = r.fmass = nullptr, r.q = r.local = r.fcs = nullptr;
    r.table = nullptr;
    r.ltable = nullptr, r.lnode_f = nullptr, r.lnode_d = nullptr, r.lM = 0, r.lm0 = 0, r.ldt = 0;
    r.ledger = own(dalloc<double>(2));
    ctx->reaction = rd;
}
void react(Context *ctx, double *state, double tau, const double *host_table, int n, double dr, double cx, double cy) {
    ReactionDevice *rd = ctx->reaction;
    ReactionView &r = rd->r;
    const View &v = ctx->v;
    const bool fcs = host_table != nullptr;
    if (fcs) {
        if (std::size_t(n + 1) > rd->table_size) {
            cudaFree(r.table);
            rd->table_size = std::size_t(n + 1) * 2;
            CK(cudaMalloc(&r.table, rd->table_size * sizeof(double)));
        }
        CK(cudaMemcpyAsync(r.table, host_table, std::size_t(n + 1) * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
        CK(cudaStreamSynchronize(ctx->stream));
        k_fcs_points<<<blocks(std::size_t(v.cells) * r.P), 256, 0, ctx->stream>>>(v, r, n, dr, cx, cy);
        k_fcs_element<<<blocks(std::size_t(v.cells) * v.local), 256, 0, ctx->stream>>>(v, r);
        k_fcs_gather<<<blocks(v.nodes), 256, 0, ctx->stream>>>(v, r);
    }
    if (ctx->adaptive)
        react_rows_adaptive_launch(ctx, r, state, tau, fcs);
    else
        k_react_rows<<<blocks(std::size_t(v.nodes) * v.q), 256, 0, ctx->stream>>>(v, r, state, tau, fcs);
    k_admissible<<<blocks(v.nodes), 256, 0, ctx->stream>>>(v, state, false, false, nullptr, nullptr, 0, 0, 0);
}
void lattice_table(Context *ctx, const uncollided::LatticeView &hv, int m0, int M, double dt, double *host_out) {
    ReactionDevice *rd = ctx->reaction;
    ReactionView &r = rd->r;
    const View &v = ctx->v;
    if (!r.P)
        throw std::runtime_error("lattice_table: no FCS points");
    auto own = [&](auto *p) {
        rd->owned.push_back((void *)p);
        return p;
    };
    // Rebuild (dynamic AMR windows): free the previous table and its lattice arrays.
    for (void *old : rd->lattice_owned) {
        rd->owned.erase(std::remove(rd->owned.begin(), rd->owned.end(), old), rd->owned.end());
        CK(cudaFree(old));
    }
    rd->lattice_owned.clear();
    const std::size_t owned_before = rd->owned.size();
    uncollided::LatticeView lv = hv; // device copies of the arrays
    lv.sigma_t = own(dcopy(hv.sigma_t, std::size_t(hv.nx) * hv.ny));
    lv.phi_x = own(dcopy(hv.phi_x, std::size_t(hv.phi_n))), lv.phi_w = own(dcopy(hv.phi_w, std::size_t(hv.phi_n)));
    lv.theta_x = own(dcopy(hv.theta_x, std::size_t(hv.theta_n)));
    lv.theta_w = own(dcopy(hv.theta_w, std::size_t(hv.theta_n)));
    const std::size_t points = std::size_t(v.cells) * r.P, total = points * std::size_t(M - m0 + 1);
    double *table = own(dalloc<double>(total));
    rd->lattice_owned.assign(rd->owned.begin() + owned_before, rd->owned.end());
    k_lattice_table<<<blocks(points, 128), 128, 0, ctx->stream>>>(lv, r.fx, r.fy, points, m0, M, dt, table);
    CK(cudaGetLastError());
    if (const char *e = std::getenv("M1PN_CTOR_TIMING"); e && std::atoi(e)) {
        const auto t0 = std::chrono::steady_clock::now();
        CK(cudaDeviceSynchronize());
        std::fprintf(stderr, "gpu: lattice table kernel %.3f s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    if (host_out)
        CK(cudaMemcpy(host_out, table, total * sizeof(double), cudaMemcpyDeviceToHost));
    r.ltable = table, r.lM = M, r.lm0 = m0, r.ldt = dt;
}
__global__ void k_lattice3d_table(uncollided::Lattice3dView lv, const double *x, const double *y, const double *z,
                                  std::size_t first, std::size_t count, int M, double dt, double *work,
                                  std::size_t work_stride, double *table) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= count)
        return;
    const std::size_t k = first + t;
    const double p[3] = {x[k], y[k], z[k]};
    double dist = 0; // the front does not reach x within the table: row stays zero (as the host build)
    for (int a = 0; a < 3; ++a) {
        const double e = fmax(fmax(lv.s[a] - p[a], 0.), p[a] - lv.s[a] - 1);
        dist += e * e;
    }
    if (sqrt(dist) >= M * dt) {
        for (int m = 0; m <= M; ++m)
            table[k * std::size_t(M + 1) + m] = 0;
        return;
    }
    uncollided::lattice3d_row(lv, p, dt, M, work + t * work_stride, table + k * std::size_t(M + 1));
}
void lattice3d_table(Context *ctx, const uncollided::Lattice3dView &hv, const double *hx, const double *hy,
                     const double *hz, int M, double dt, double *host_out) {
    ReactionDevice *rd = ctx->reaction;
    ReactionView &r = rd->r;
    const View &v = ctx->v;
    if (!r.P)
        throw std::runtime_error("lattice3d_table: no FCS points");
    auto own = [&](auto *p) {
        rd->owned.push_back((void *)p);
        return p;
    };
    const std::size_t points = std::size_t(v.cells) * r.P, total = points * std::size_t(M + 1);
    uncollided::Lattice3dView lv = hv; // device copies of the arrays
    lv.sigma_t = own(dcopy(hv.sigma_t, std::size_t(hv.n[0]) * hv.n[1] * hv.n[2]));
    lv.outer_x = own(dcopy(hv.outer_x, std::size_t(hv.outer_n)));
    lv.outer_w = own(dcopy(hv.outer_w, std::size_t(hv.outer_n)));
    lv.inner_x = own(dcopy(hv.inner_x, std::size_t(hv.inner_n)));
    lv.inner_w = own(dcopy(hv.inner_w, std::size_t(hv.inner_n)));
    double *dx = dcopy(hx, points), *dy = dcopy(hy, points), *dz = dcopy(hz, points);
    double *table = own(dalloc<double>(total));
    // Batches bound the work arrays (lattice3d_work_size doubles per point: 3 (M + 2) Cartesian, 10 (M + 2) polar);
    // the per-thread break lists live in local memory (polar: three lists and the stacks).
    const std::size_t batch = std::min<std::size_t>(points, 1 << 16);
    const std::size_t work_stride = uncollided::lattice3d_work_size(hv, M);
    double *work = dalloc<double>(batch * work_stride);
    CK(cudaDeviceSetLimit(cudaLimitStackSize, (hv.polar ? 16 : 24) * 1024));
    for (std::size_t first = 0; first < points; first += batch) {
        const std::size_t count = std::min(batch, points - first);
        k_lattice3d_table<<<blocks(count, 64), 64, 0, ctx->stream>>>(lv, dx, dy, dz, first, count, M, dt, work,
                                                                    work_stride, table);
        CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());
    CK(cudaFree(work));
    CK(cudaFree(dx)), CK(cudaFree(dy)), CK(cudaFree(dz));
    if (host_out)
        CK(cudaMemcpy(host_out, table, total * sizeof(double), cudaMemcpyDeviceToHost));
    r.ltable = table, r.lnode_f = nullptr, r.lnode_d = nullptr, r.lM = M, r.lm0 = 0, r.ldt = dt;
}
__global__ void k_fcs_nodes_lattice_f(View v, ReactionView r, double time) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= v.nodes)
        return;
    r.fcs[i] = uncollided::lattice3d_eval_row(r.lnode_f + std::size_t(i) * std::size_t(r.lM + 1), r.lM, r.ldt, time);
}
__global__ void k_fcs_nodes_lattice_d(View v, ReactionView r, double time) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= v.nodes)
        return;
    r.fcs[i] = uncollided::lattice3d_eval_row(r.lnode_d + std::size_t(i) * std::size_t(r.lM + 1), r.lM, r.ldt, time);
}
__global__ void k_lattice3d_eval_batch(uncollided::Lattice3dView lv, const double *px, const double *py,
                                       const double *pz, const double *weight, const double *sigma_a, const int *owned,
                                       std::size_t first, std::size_t count, int M, double dt, double final_time,
                                       double *work, std::size_t work_stride, double *batch_rows, double *batch_sum) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= count)
        return;
    const std::size_t q = first + t;
    const double p[3] = {px[q], py[q], pz[q]};
    double *row = batch_rows + t * std::size_t(M + 1);
    uncollided::lattice3d_row(lv, p, dt, M, work + t * work_stride, row);
    if (owned[q]) {
        const double w = weight[q], sa = sigma_a[q];
        const double rT = uncollided::lattice3d_eval_row(row, M, dt, final_time);
        batch_sum[3 * t + 0] = w * rT;
        batch_sum[3 * t + 1] = w * sa * rT;
        batch_sum[3 * t + 2] = sa > 0 ? w * sa * uncollided::lattice3d_time_integral_row(row, M, dt, final_time) : 0.;
    } else {
        batch_sum[3 * t + 0] = 0, batch_sum[3 * t + 1] = 0, batch_sum[3 * t + 2] = 0;
    }
}
__global__ void k_lattice3d_scatter_batch(View v, ReactionView r, const int *elem, const int *local_k,
                                          const double *weight, const int *owned, std::size_t first, std::size_t count,
                                          int W, const double *batch_rows, double *node_acc, double *mass_acc) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count * std::size_t(W))
        return;
    const std::size_t t = idx / std::size_t(W), m = idx % std::size_t(W);
    const double val = batch_rows[idx];
    if (val == 0)
        return;
    const std::size_t q = first + t;
    const int e = elem[q], lk = local_k[q];
    const std::size_t base = std::size_t(e) * r.P;
    for (int l = 0; l < v.local; ++l) {
        double c = 0;
        if (lk >= 0) {
            c = r.fshape[std::size_t(l) * r.P + lk] * r.fw[base + lk] * r.fss[base + lk];
        } else {
            for (int k = 0; k < r.P; ++k)
                c += r.fshape[std::size_t(l) * r.P + k] * r.fw[base + k] * r.fss[base + k];
        }
        if (c == 0)
            continue;
        const int node = v.connectivity[std::size_t(e) * v.local + l];
        atomicAdd(&node_acc[std::size_t(node) * std::size_t(W) + m], (c / r.fmass[node]) * val);
    }
    if (owned[q])
        atomicAdd(&mass_acc[m], weight[q] * val);
}
__global__ void k_cast_d2f(const double *src, float *dst, std::size_t n) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = float(src[i]);
}
Lattice3dNodalSummary lattice3d_table_nodal(Context *ctx, const uncollided::Lattice3dView &hv, const double *px,
                                            const double *py, const double *pz, const int *elem, const int *local_k,
                                            const double *weight, const double *sigma_a, const int *owned,
                                            std::size_t n_eval, int M, double dt, double final_time, bool fp32,
                                            double *host_mass_table) {
    ReactionDevice *rd = ctx->reaction;
    ReactionView &r = rd->r;
    const View &v = ctx->v;
    if (!r.P)
        throw std::runtime_error("lattice3d_table_nodal: no FCS points");
    auto own = [&](auto *p) {
        rd->owned.push_back((void *)p);
        return p;
    };
    const std::size_t W = std::size_t(M + 1), total_nodes = std::size_t(v.nodes) * W;
    double *node_acc = dalloc<double>(total_nodes);
    double *mass_acc = dalloc<double>(W);
    CK(cudaMemset(node_acc, 0, total_nodes * sizeof(double)));
    CK(cudaMemset(mass_acc, 0, W * sizeof(double)));
    Lattice3dNodalSummary sum{};
    if (n_eval > 0) {
        uncollided::Lattice3dView lv = hv;
        double *d_sigma_t = dcopy(hv.sigma_t, std::size_t(hv.n[0]) * hv.n[1] * hv.n[2]);
        double *d_ox = hv.outer_n ? dcopy(hv.outer_x, std::size_t(hv.outer_n)) : nullptr;
        double *d_ow = hv.outer_n ? dcopy(hv.outer_w, std::size_t(hv.outer_n)) : nullptr;
        double *d_ix = hv.inner_n ? dcopy(hv.inner_x, std::size_t(hv.inner_n)) : nullptr;
        double *d_iw = hv.inner_n ? dcopy(hv.inner_w, std::size_t(hv.inner_n)) : nullptr;
        lv.sigma_t = d_sigma_t, lv.outer_x = d_ox, lv.outer_w = d_ow, lv.inner_x = d_ix, lv.inner_w = d_iw;
        double *d_px = dcopy(px, n_eval), *d_py = dcopy(py, n_eval), *d_pz = dcopy(pz, n_eval);
        int *d_elem = dcopy(elem, n_eval), *d_lk = dcopy(local_k, n_eval), *d_own = dcopy(owned, n_eval);
        double *d_w = dcopy(weight, n_eval), *d_sa = dcopy(sigma_a, n_eval);
        const std::size_t work_stride = uncollided::lattice3d_work_size(hv, M);
        // Batch size: as many points as fit in about a third of the free device memory (work + rows per point), so
        // that one launch fills the GPU and the slow points near S overlap with the cheap ones. M1PN_FCS3D_BATCH
        // overrides (benchmarks).
        // Stack reserve first: it is taken from device memory for all resident threads, and the batch below claims a
        // third of what is free (P15 on 168^3 failed in cudaDeviceSetLimit after the batch allocation).
        CK(cudaDeviceSetLimit(cudaLimitStackSize, (hv.polar ? 16 : 24) * 1024));
        std::size_t batch;
        {
            std::size_t free_b = 0, total_b = 0;
            CK(cudaMemGetInfo(&free_b, &total_b));
            const std::size_t per_point = (work_stride + W + 3) * sizeof(double);
            batch = std::max<std::size_t>(std::size_t(1) << 14, free_b / 3 / per_point);
            batch = std::min<std::size_t>(batch, std::size_t(1) << 20);
            if (const char *e = std::getenv("M1PN_FCS3D_BATCH"))
                batch = std::max<std::size_t>(1, std::size_t(std::atoll(e)));
            batch = std::min<std::size_t>(batch, n_eval);
        }
        double *work = dalloc<double>(batch * work_stride);
        double *batch_rows = dalloc<double>(batch * W);
        double *batch_sum = dalloc<double>(batch * 3);
        std::vector<double> h_sum(batch * 3);
        for (std::size_t first = 0; first < n_eval; first += batch) {
            const std::size_t count = std::min(batch, n_eval - first);
            k_lattice3d_eval_batch<<<blocks(count, 64), 64, 0, ctx->stream>>>(
                lv, d_px, d_py, d_pz, d_w, d_sa, d_own, first, count, M, dt, final_time, work, work_stride, batch_rows,
                batch_sum);
            CK(cudaGetLastError());
            k_lattice3d_scatter_batch<<<blocks(count * W, 256), 256, 0, ctx->stream>>>(
                v, r, d_elem, d_lk, d_w, d_own, first, count, int(W), batch_rows, node_acc, mass_acc);
            CK(cudaGetLastError());
            CK(cudaMemcpyAsync(h_sum.data(), batch_sum, count * 3 * sizeof(double), cudaMemcpyDeviceToHost,
                               ctx->stream));
            CK(cudaStreamSynchronize(ctx->stream));
            for (std::size_t t = 0; t < count; ++t)
                sum.mass += h_sum[3 * t + 0], sum.abs_rate += h_sum[3 * t + 1], sum.absorbed += h_sum[3 * t + 2];
        }
        CK(cudaFree(work)), CK(cudaFree(batch_rows)), CK(cudaFree(batch_sum));
        CK(cudaFree(d_px)), CK(cudaFree(d_py)), CK(cudaFree(d_pz));
        CK(cudaFree(d_elem)), CK(cudaFree(d_lk)), CK(cudaFree(d_own)), CK(cudaFree(d_w)), CK(cudaFree(d_sa));
        CK(cudaFree(d_sigma_t));
        if (d_ox)
            CK(cudaFree(d_ox));
        if (d_ow)
            CK(cudaFree(d_ow));
        if (d_ix)
            CK(cudaFree(d_ix));
        if (d_iw)
            CK(cudaFree(d_iw));
    }
    if (host_mass_table)
        CK(cudaMemcpy(host_mass_table, mass_acc, W * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaFree(mass_acc));
    if (fp32) {
        float *node_f = own(dalloc<float>(total_nodes));
        k_cast_d2f<<<blocks(total_nodes, 256), 256, 0, ctx->stream>>>(node_acc, node_f, total_nodes);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaFree(node_acc));
        r.lnode_f = node_f, r.lnode_d = nullptr;
    } else {
        own(node_acc);
        r.lnode_d = node_acc, r.lnode_f = nullptr;
    }
    r.ltable = nullptr, r.lM = M, r.lm0 = 0, r.ldt = dt;
    return sum;
}
void react_lattice(Context *ctx, double *state, double tau, double source_time) {
    ReactionView &r = ctx->reaction->r;
    const View &v = ctx->v;
    if (r.lnode_f) {
        k_fcs_nodes_lattice_f<<<blocks(v.nodes), 256, 0, ctx->stream>>>(v, r, source_time);
    } else if (r.lnode_d) {
        k_fcs_nodes_lattice_d<<<blocks(v.nodes), 256, 0, ctx->stream>>>(v, r, source_time);
    } else {
        if (!r.ltable)
            throw std::runtime_error("react_lattice: no lattice table");
        k_fcs_points_lattice<<<blocks(std::size_t(v.cells) * r.P), 256, 0, ctx->stream>>>(v, r, source_time);
        k_fcs_element<<<blocks(std::size_t(v.cells) * v.local), 256, 0, ctx->stream>>>(v, r);
        k_fcs_gather<<<blocks(v.nodes), 256, 0, ctx->stream>>>(v, r);
    }
    if (ctx->adaptive)
        react_rows_adaptive_launch(ctx, r, state, tau, true);
    else
        k_react_rows<<<blocks(std::size_t(v.nodes) * v.q), 256, 0, ctx->stream>>>(v, r, state, tau, true);
    k_admissible<<<blocks(v.nodes), 256, 0, ctx->stream>>>(v, state, false, false, nullptr, nullptr, 0, 0, 0);
}
Ledger ledger(Context *ctx) {
    double h[2];
    CK(cudaMemcpyAsync(h, ctx->reaction->r.ledger, sizeof h, cudaMemcpyDeviceToHost, ctx->stream));
    CK(cudaStreamSynchronize(ctx->stream));
    Ledger l;
    l.absorbed = h[0], l.injected = h[1];
    return l;
}

Context *create(const Setup &s) {
    if (double(s.nodes) * s.q >= 4294967295.)
        throw std::runtime_error("GPU path: nodes x moments must stay below 2^32");
    if (s.local > max_local)
        throw std::runtime_error("GPU path: too many local nodes per element");
    if (s.degree > MAXP)
        throw std::runtime_error("GPU path: degree above 4");
    int devices = 0;
    CK(cudaGetDeviceCount(&devices));
    if (devices < 1)
        throw std::runtime_error("GPU path: no CUDA device");
    Context *ctx = new Context;
    CK(cudaStreamCreate(&ctx->stream)); // blocking: still ordered with the legacy default stream
    View &v = ctx->v;
    v.dimension = s.dimension, v.degree = s.degree, v.nodes = s.nodes, v.cells = s.cells, v.local = s.local, v.q = s.q;
    for (int d = 0; d < 3; ++d)
        v.npa[d] = s.nodes_per_axis[d], v.count[d] = s.count[d], v.stride[d] = s.stride[d];
    v.viscosity = s.viscosity;
    v.s0 = std::sqrt(4 * 3.141592653589793238462643383279502884);
    v.s1 = std::sqrt(4 * 3.141592653589793238462643383279502884 / 3);
    const int n = s.local, ne = s.cells * n;
    auto own = [&](auto *p) {
        ctx->owned.push_back((void *)p);
        return p;
    };
    v.connectivity = own(dcopy(s.connectivity, std::size_t(ne)));
    v.neighbors = own(dcopy(s.neighbors, std::size_t(s.cells) * 2 * s.dimension));
    v.node_offsets = own(dcopy(s.node_offsets, std::size_t(s.nodes) + 1));
    v.node_entries = own(dcopy(s.node_entries, std::size_t(ne)));
    const std::size_t C = std::size_t(s.classes);
    v.klass = s.element_class ? own(dcopy(s.element_class, std::size_t(s.cells))) : nullptr;
    v.classes = s.classes;
    v.structured = s.structured;
    v.strip3_ok = s.strip3_ok;
    for (int d = 0; d < 3; ++d)
        v.strip_elem[d] = s.strip_elem[d] ? own(dcopy(s.strip_elem[d], 2 * std::size_t(s.nodes_per_axis[d]))) : nullptr;
    v.class_viscosity = s.class_viscosity ? own(dcopy(s.class_viscosity, C)) : nullptr;
    v.ext_index = s.ext_index ? own(dcopy(s.ext_index, std::size_t(s.cells) * 2 * s.dimension)) : nullptr;
    v.ext_list = s.ext_index ? own(dcopy(s.ext_list, std::size_t(s.ext_count) * n * n)) : nullptr;
    v.stiffness_rows = own(dcopy(s.stiffness_rows, C * n * n));
    { // nu_c K_c per class (3D Q1 strip kernel)
        std::vector<double> kv(C * n * n);
        for (std::size_t c = 0; c < C; ++c)
            for (int k = 0; k < n * n; ++k)
                kv[c * n * n + k] = (s.class_viscosity ? s.class_viscosity[c] : s.viscosity) * s.stiffness_rows[c * n * n + k];
        v.stiffness_nu = own(dcopy(kv.data(), kv.size()));
    }
    v.seminorm = own(dcopy(s.seminorm, std::size_t(n) * n));
    for (int d = 0; d < 3; ++d) {
        v.convection_t_rows[d] = d < s.dimension ? own(dcopy(s.convection_t_rows[d], C * n * n)) : nullptr;
        for (int side = 0; side < 2; ++side)
            v.extension[d][side] = d < s.dimension ? own(dcopy(s.extension[d][side], std::size_t(n) * n)) : nullptr;
        if (d < s.dimension) {
            const int n1 = s.nodes_per_axis[d], p = s.degree;
            v.convection_band[d] = own(dcopy(s.convection_band[d], std::size_t(n1) * (2 * p + 1)));
            v.lower[d] = own(dcopy(s.lower[d], std::size_t(n1) * p));
            std::vector<double> inv(n1);
            for (int k = 0; k < n1; ++k)
                inv[k] = 1. / s.diagonal[d][k];
            v.invdiag[d] = own(dcopy(inv.data(), std::size_t(n1)));
            v.flux_outer[d] = own(dcopy(s.flux_outer[d], std::size_t(s.q) + 1));
            v.flux_inner[d] = own(dcopy(s.flux_inner[d], std::size_t(s.flux_nnz[d])));
            v.flux_value[d] = own(dcopy(s.flux_value[d], std::size_t(s.flux_nnz[d])));
            ctx->flux_nnz[d] = s.flux_nnz[d];
        } else {
            v.convection_band[d] = v.lower[d] = v.invdiag[d] = nullptr;
            v.flux_outer[d] = v.flux_inner[d] = nullptr;
            v.flux_value[d] = nullptr;
        }
        v.mirror_index[d] = d < s.dimension ? s.mirror_index[d] : -1;
        v.dlower[d] = v.dinvdiag[d] = nullptr;
        v.reflected[d] = nullptr;
        if (v.mirror_index[d] >= 0) {
            const int n1 = s.nodes_per_axis[d] - 1, p = s.degree;
            v.dlower[d] = own(dcopy(s.dlower[d], std::size_t(n1) * p));
            std::vector<double> inv(n1);
            for (int k = 0; k < n1; ++k)
                inv[k] = 1. / s.ddiagonal[d][k];
            v.dinvdiag[d] = own(dcopy(inv.data(), std::size_t(n1)));
            v.reflected[d] = own(dcopy(s.reflected[d], std::size_t(n)));
        }
    }
    v.periodic = s.periodic;
    for (int d = 0; d < 3; ++d)
        v.tail[d] = s.periodic && d < s.dimension
                        ? own(dcopy(s.tail[d], std::size_t(s.degree) * s.nodes_per_axis[d]))
                        : nullptr;
    v.comp_odd = s.node_mirror ? own(dcopy(s.comp_odd, std::size_t(s.q))) : nullptr;
    v.node_mirror = s.node_mirror ? own(dcopy(s.node_mirror, std::size_t(s.nodes))) : nullptr;
    {
        int width = 1;
        for (int d = 0; d < s.dimension; ++d)
            for (int j = 0; j < s.q; ++j)
                width = std::max(width, s.flux_outer[d][j + 1] - s.flux_outer[d][j]);
        std::vector<double> ev(std::size_t(s.dimension) * s.q * width, 0.);
        std::vector<int> ei(ev.size(), 0);
        for (int d = 0; d < s.dimension; ++d)
            for (int j = 0; j < s.q; ++j)
                for (int e = s.flux_outer[d][j], w = 0; e < s.flux_outer[d][j + 1]; ++e, ++w) {
                    ev[(std::size_t(d) * s.q + j) * width + w] = s.flux_value[d][e];
                    ei[(std::size_t(d) * s.q + j) * width + w] = s.flux_inner[d][e];
                }
        v.flux_width = width;
        v.flux_in_shared = ev.size() * (sizeof(double) + sizeof(int)) <= 48 * 1024;
        v.ell_val = own(dcopy(ev.data(), ev.size()));
        v.ell_idx = own(dcopy(ei.data(), ei.size()));
    }
    const std::size_t pairs = C * n * (n - 1);
    v.pair_j = own(dcopy(s.pair_j, pairs));
    v.pair_d = own(dcopy(s.pair_d, pairs));
    v.pair_m = own(dcopy(s.pair_m, pairs));
    v.pair_c = own(dcopy(s.pair_c, 3 * pairs));
    v.pair_ct = own(dcopy(s.pair_ct, 3 * pairs));
    v.total_viscosity = own(dcopy(s.total_viscosity, C * n));
    v.lumped = own(dcopy(s.lumped, std::size_t(s.nodes)));
    if (s.dimension == 2 && s.degree == 1 && s.classes <= max_strip_classes) {
        const std::size_t m = 16 * C;
        std::vector<double> kv(m), nu(C);
        for (std::size_t c = 0; c < C; ++c) {
            nu[c] = s.class_viscosity ? s.class_viscosity[c] : s.viscosity;
            for (int k = 0; k < 16; ++k)
                kv[16 * c + k] = nu[c] * s.stiffness_rows[16 * c + k];
        }
        v.strip_kv = own(dcopy(kv.data(), m));
        v.strip_c0 = own(dcopy(s.convection_t_rows[0], m));
        v.strip_c1 = own(dcopy(s.convection_t_rows[1], m));
        v.strip_nu = own(dcopy(nu.data(), C));
    } else
        v.strip_kv = v.strip_c0 = v.strip_c1 = v.strip_nu = nullptr;
    v.weno = s.weno, v.stabilize = s.stabilize, v.limit = s.limit, v.outward_bar_rounding = s.outward_bar_rounding;
    v.scale_invariant_weno = s.scale_invariant_weno;
    v.weno_q = s.weno_q, v.weno_r = s.weno_r, v.weno_epsilon = s.weno_epsilon, v.weno_linear = s.weno_linear;
    v.element_theta = s.element_theta;
    v.fallback_capacity = s.fallback_capacity;
    const std::size_t N = s.nodes, Q = s.q;
    Buffers &b = ctx->b;
    b.a = own(dalloc<double>(N * Q));
    b.x = own(dalloc<double>(N * Q));
    b.r = own(dalloc<double>(N * Q));
    v.density = own(dalloc<double>(N));
    v.low = b.low = own(dalloc<double>(N * 4));
    v.low_flux = b.low_flux = own(dalloc<double>(N * 12));
    v.closure = b.closure = own(dalloc<double>(N * 12));
    for (int d = 0; d < 3; ++d)
        v.gradient[d] = b.gradient[d] = d < s.dimension ? own(dalloc<double>(N * Q)) : nullptr;
    v.stab_low = b.stab_low = own(dalloc<double>(std::size_t(s.cells) * 4 * n));
    v.gamma = b.gamma = own(dalloc<double>(std::size_t(s.cells)));
    v.moment_rate = b.moment_rate = own(dalloc<double>(N * 4));
    v.element_limited = b.element_limited = own(dalloc<double>(std::size_t(s.cells) * 4 * n));
    v.load = b.load = own(dalloc<double>(N * 4));
    v.closure_list = own(dalloc<int>(std::size_t(s.fallback_capacity)));
    v.limiter_list = own(dalloc<int>(std::size_t(s.fallback_capacity)));
    v.segment_list = own(dalloc<int>(std::size_t(s.cells)));
    v.seg_ubar = own(dalloc<double>(std::size_t(s.cells) * 4 * n));
    v.seg_lim = own(dalloc<double>(std::size_t(s.cells) * 4 * n));
    v.counters = own(dalloc<DeviceCounters>(1));
    ctx->norm_accumulator = own(dalloc<double>(1024)); // per-block partials of norm() (at most 1024 blocks)
    for (auto &e : ctx->events)
        CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    CK(cudaMallocHost(&ctx->pinned_counters, sizeof(DeviceCounters)));
    reset_counters(ctx);
    return ctx;
}
// ---- Phase A of the boundary (BoundaryHighSetup) -------------------------------------------------------------------
struct BoundaryHighDevice {
    BoundaryHighSetup s;
    int *row_nodes = nullptr, *offsets = nullptr, *rows_list = nullptr;
    double *row_mass = nullptr, *flux_t[6] = {}, *iface_t[6] = {}, *G = nullptr, *H = nullptr, *db = nullptr,
           *low4 = nullptr, *low4b = nullptr;
    int *row_break = nullptr;
    cublasHandle_t handle = nullptr;
};
namespace {
void cublas_check(cublasStatus_t e, const char *what) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLAS: ") + what + " failed (" + std::to_string(int(e)) + ")");
}
__global__ void k_face_traces(int q, int rows, int nf, const int *row_nodes, const double *row_mass, const double *x,
                              double *G) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(rows) * q)
        return;
    const int r = int(t / q), c = int(t % q);
    double sum = 0;
    for (int m = 0; m < nf; ++m)
        sum += row_mass[std::size_t(r) * nf + m] * x[std::size_t(row_nodes[std::size_t(r) * nf + m]) * q + c];
    G[t] = sum;
}
__global__ void k_sub(double *a, const double *b, std::size_t n) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t < n)
        a[t] -= b[t];
}
__global__ void k_boundary_rows_add(int q, int nb, const int *offsets, const int *rows_list, const int *bnodes,
                                    const double *H, double *out, double *low4, const int *row_break, double *low4b) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(nb) * q)
        return;
    const int k = int(t / q), c = int(t % q);
    double sum = 0;
    for (int e = offsets[k]; e < offsets[k + 1]; ++e)
        sum += H[std::size_t(rows_list[e]) * q + c];
    out[std::size_t(bnodes[k]) * q + c] += sum;
    if (c < 4) {
        low4[std::size_t(k) * 4 + c] = sum;
        if (row_break) { // split_breaks: the broken-line rows apart
            double part = 0;
            for (int e = offsets[k]; e < offsets[k + 1]; ++e)
                if (row_break[rows_list[e]])
                    part += H[std::size_t(rows_list[e]) * q + c];
            low4b[std::size_t(k) * 4 + c] = part;
        }
    }
}
} // namespace
void boundary_high_setup(Context *ctx, const BoundaryHighSetup &s) {
    auto *d = new BoundaryHighDevice;
    d->s = s;
    const std::size_t rq = std::size_t(std::max(1, s.rows)) * s.q, rn = std::size_t(std::max(1, s.rows)) * s.nf;
    d->row_nodes = dcopy(s.row_nodes, rn);
    d->row_mass = dcopy(s.row_mass, rn);
    d->offsets = dcopy(s.offsets, std::size_t(s.nb) + 1);
    d->rows_list = dcopy(s.rows_list, std::size_t(std::max(1, s.offsets[s.nb])));
    for (int side = 0; side < 6; ++side) {
        if (s.flux_t[side])
            d->flux_t[side] = dcopy(s.flux_t[side], std::size_t(s.q) * s.q);
        if (s.iface_t[side])
            d->iface_t[side] = dcopy(s.iface_t[side], std::size_t(s.q) * s.q);
    }
    d->G = dalloc<double>(rq);
    d->H = dalloc<double>(rq);
    d->db = dalloc<double>(rq);
    d->low4 = dalloc<double>(std::size_t(std::max(1, s.nb)) * 4);
    if (s.row_break) {
        d->row_break = dcopy(s.row_break, std::size_t(std::max(1, s.rows)));
        d->low4b = dalloc<double>(std::size_t(std::max(1, s.nb)) * 4);
    }
    cublas_check(cublasCreate(&d->handle), "create");
    cublas_check(cublasSetStream(d->handle, ctx->stream), "set stream"); // the context stream, as all kernels here
    ctx->bhigh = d;
}
// ---------------------------------------------------------------------------------------------------------------
// Ghost traces (GhostTraceSetup): gather the ghost rows of a side (q x count, column-major = the row-major state rows)
// and one DGEMM per side, T_s = H_s A_s, on the aux stream with its own cuBLAS handle.
struct GhostTraceDevice {
    int q = 0, count[6] = {}, points[6] = {};
    int *nodes[6] = {};
    double *H[6] = {};
    double *A = nullptr, *T = nullptr; // max_count x q gather buffer, all traces
    std::size_t offset[6] = {}, total = 0;
    cublasHandle_t handle = nullptr;
};
__global__ void k_gather_ghost_rows(int q, int count, const int *nodes, const double *state, double *A) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(count) * q)
        return;
    const int j = int(t / q), c = int(t % q);
    A[t] = state[std::size_t(nodes[j]) * q + c];
}
void ghost_traces_setup(Context *ctx, const GhostTraceSetup &s) {
    auto *d = new GhostTraceDevice;
    d->q = s.q;
    int max_count = 0;
    for (int side = 0; side < 6; ++side) {
        d->count[side] = s.count[side], d->points[side] = s.points[side];
        d->offset[side] = d->total;
        if (!s.count[side])
            continue;
        d->nodes[side] = dcopy(s.nodes[side], std::size_t(s.count[side]));
        d->H[side] = dcopy(s.harmonics[side], std::size_t(s.points[side]) * s.q);
        d->total += std::size_t(s.points[side]) * s.count[side];
        max_count = std::max(max_count, s.count[side]);
    }
    CK(cudaMalloc(&d->A, sizeof(double) * std::max<std::size_t>(1, std::size_t(max_count) * s.q)));
    CK(cudaMalloc(&d->T, sizeof(double) * std::max<std::size_t>(1, d->total)));
    if (!ctx->aux) {
        int least = 0, greatest = 0;
        CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        CK(cudaStreamCreateWithPriority(&ctx->aux, cudaStreamNonBlocking, least));
    }
    cublas_check(cublasCreate(&d->handle), "create (ghost traces)");
    cublas_check(cublasSetStream(d->handle, ctx->aux), "set stream (ghost traces)");
    ctx->gtrace = d;
}
void ghost_traces(Context *ctx, const double *state, double *pinned_out) {
    GhostTraceDevice *d = ctx->gtrace;
    cudaStream_t st = ctx->aux;
    CK(cudaEventRecord(ctx->events[3], ctx->stream));
    CK(cudaStreamWaitEvent(st, ctx->events[3], 0));
    const double one = 1, zero = 0;
    for (int side = 0; side < 6; ++side) {
        const int n = d->count[side], p = d->points[side];
        if (!n)
            continue;
        // The gather buffer is reused per side: the DGEMM of the previous side is ordered before it on the stream.
        k_gather_ghost_rows<<<blocks(std::size_t(n) * d->q), 256, 0, st>>>(d->q, n, d->nodes[side], state, d->A);
        cublas_check(cublasDgemm(d->handle, CUBLAS_OP_N, CUBLAS_OP_N, p, n, d->q, &one, d->H[side], p, d->A, d->q,
                                 &zero, d->T + d->offset[side], p),
                     "dgemm (ghost traces)");
    }
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(pinned_out, d->T, d->total * sizeof(double), cudaMemcpyDeviceToHost, st));
    CK(cudaEventRecord(ctx->events[2], st));
}
void destroy_ghost_traces(Context *ctx) {
    GhostTraceDevice *d = ctx->gtrace;
    if (!d)
        return;
    for (int side = 0; side < 6; ++side)
        cudaFree(d->nodes[side]), cudaFree(d->H[side]);
    cudaFree(d->A), cudaFree(d->T);
    if (d->handle)
        cublasDestroy(d->handle);
    delete d;
    ctx->gtrace = nullptr;
}
void destroy_boundary_high(Context *ctx) {
    BoundaryHighDevice *d = ctx->bhigh;
    if (!d)
        return;
    for (void *p : {(void *)d->row_nodes, (void *)d->offsets, (void *)d->rows_list, (void *)d->row_mass, (void *)d->G,
                    (void *)d->H, (void *)d->db, (void *)d->low4, (void *)d->low4b, (void *)d->row_break})
        cudaFree(p);
    for (double *p : d->iface_t)
        cudaFree(p);
    for (double *p : d->flux_t)
        cudaFree(p);
    if (d->handle)
        cublasDestroy(d->handle);
    delete d;
    ctx->bhigh = nullptr;
}
void boundary_high(Context *ctx, const double *state, double *out, const double *pinned_db, double *pinned_low4,
                   double *pinned_low4_break) {
    BoundaryHighDevice *d = ctx->bhigh;
    const BoundaryHighSetup &s = d->s;
    if (s.rows == 0 || s.nb == 0)
        return;
    if (ctx->adaptive && boundary_high_adaptive_launch(ctx, d, state, out, pinned_db, pinned_low4, pinned_low4_break))
        return;
    const std::size_t rq = std::size_t(s.rows) * s.q;
    k_face_traces<<<blocks(rq), 256, 0, ctx->stream>>>(s.q, s.rows, s.nf, d->row_nodes, d->row_mass, state, d->G);
    CK(cudaMemsetAsync(d->H, 0, rq * sizeof(double), ctx->stream));
    const double one = 1, zero = 0;
    for (int side = 0; side < 6; ++side) {
        const int r0 = s.side_start[side], r1 = d->iface_t[side] ? s.break_start[side] : s.side_start[side + 1];
        // Row-major G (rows x q) is column-major G^T (q x rows): H^T = B^T G^T with B = flux_t (column-major q x q).
        if (d->flux_t[side] && r1 > r0)
            cublas_check(cublasDgemm(d->handle, CUBLAS_OP_T, CUBLAS_OP_N, s.q, r1 - r0, s.q, &one, d->flux_t[side], s.q,
                                     d->G + std::size_t(r0) * s.q, s.q, &zero, d->H + std::size_t(r0) * s.q, s.q),
                         "dgemm");
        const int r2 = s.side_start[side + 1];
        if (d->iface_t[side] && r2 > r1) // broken-line rows: G holds M_face (u - u_partner)
            cublas_check(cublasDgemm(d->handle, CUBLAS_OP_T, CUBLAS_OP_N, s.q, r2 - r1, s.q, &one, d->iface_t[side], s.q,
                                     d->G + std::size_t(r1) * s.q, s.q, &zero, d->H + std::size_t(r1) * s.q, s.q),
                         "dgemm");
    }
    if (pinned_db) {
        CK(cudaMemcpyAsync(d->db, pinned_db, rq * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
        k_sub<<<blocks(rq), 256, 0, ctx->stream>>>(d->H, d->db, rq);
    }
    k_boundary_rows_add<<<blocks(std::size_t(s.nb) * s.q), 256, 0, ctx->stream>>>(s.q, s.nb, d->offsets, d->rows_list, s.bnodes, d->H,
                                                                   out, d->low4, d->row_break, d->low4b);
    CK(cudaMemcpyAsync(pinned_low4, d->low4, std::size_t(s.nb) * 4 * sizeof(double), cudaMemcpyDeviceToHost, ctx->stream));
    if (pinned_low4_break && d->low4b)
        CK(cudaMemcpyAsync(pinned_low4_break, d->low4b, std::size_t(s.nb) * 4 * sizeof(double), cudaMemcpyDeviceToHost,
                           ctx->stream));
}
// ---------------------------------------------------------------------------------------------------------------
// Dynamic tile-based adaptive PN kernels (experiment/dynamic-adaptive-pn).
// ---------------------------------------------------------------------------------------------------------------
struct AdaptiveDevice {
    int num_tiles = 0, max_order = 0, L1 = 0;
    int tb[3] = {1, 1, 1}, nt[3] = {1, 1, 1};
    int *d_deg_start = nullptr;
    int *d_tile_order = nullptr;
    int *d_tile_q = nullptr;
    int *d_tile_q_prev = nullptr;
    int *d_shrunken = nullptr;
    double *d_tile_rho = nullptr;
    double *d_tile_E = nullptr;
    double *d_tile_face_E = nullptr;
    AdaptiveLineSeg *d_segs_grad[3] = {};
    int segs_grad_n[3] = {}, segs_grad_cap[3] = {};
    AdaptiveLineSeg *d_segs_mass[3] = {};
    int segs_mass_n[3] = {}, segs_mass_cap[3] = {};
    int *d_seg_slot[3] = {}, *d_seg_off[3] = {}, *d_dseg_slot[3] = {}, *d_dseg_off[3] = {};
    double *d_seg_lower[3] = {}, *d_seg_invdiag[3] = {}, *d_dseg_lower[3] = {}, *d_dseg_invdiag[3] = {};
    AdaptiveTileChunk *d_tile_chunks = nullptr;
    int tile_chunks_n = 0, tile_chunks_cap = 0;
    int global_max_q = 0;
    int side_q_dom[6] = {};
    int side_q_brk[6] = {};
    int bnd_max_q = 4;
    bool active = false;
};

namespace {
__device__ inline void atomic_max_nonneg(double *addr, double val) {
    if (val <= *addr)
        return;
    unsigned long long *uaddr = reinterpret_cast<unsigned long long *>(addr);
    atomicMax(uaddr, (unsigned long long)__double_as_longlong(val));
}

__global__ void k_adaptive_sensor(View v, const double *a, bool initial, int tb0, int tb1, int tb2,
                                  int nt0, int nt1, int max_order, const int *deg_start,
                                  const int *tile_order, double *out_rho, double *out_E, double *out_face_E) {
    extern __shared__ double smem[];
    const int L1 = max_order + 1;
    double *s_E2 = smem;          // L1
    double *s_face2 = smem + L1;  // 6
    double *s_rho = smem + L1 + 6; // 1
    const int T = blockIdx.x;
    const int tid = threadIdx.x;
    for (int k = tid; k < L1 + 7; k += blockDim.x)
        smem[k] = 0.0;
    __syncthreads();

    const int tx = T % nt0;
    const int ty = (T / nt0) % nt1;
    const int tz = T / (nt0 * nt1);
    const int x0 = tx * tb0, x1 = min(v.npa[0], x0 + tb0);
    const int y0 = ty * tb1, y1 = min(v.npa[1], y0 + tb1);
    const int z0 = tz * tb2, z1 = min(v.npa[2], z0 + tb2);
    const int wx = x1 - x0, wy = y1 - y0, wz = z1 - z0;
    const int tnodes = wx * wy * wz;
    const int ord = initial ? max_order : tile_order[T];
    const int q = v.q;
    const int margin = AdaptivePnPlan::kBorderMargin;

    for (int idx = tid; idx < tnodes; idx += blockDim.x) {
        const int lx = idx % wx;
        const int ly = (idx / wx) % wy;
        const int lz = idx / (wx * wy);
        const int ix = x0 + lx, iy = y0 + ly, iz = z0 + lz;
        const std::size_t node = std::size_t(iz) * v.stride[2] + std::size_t(iy) * v.stride[1] + ix;
        const double *row = a + node * q;
        const double rho = fabs(v.s0 * row[0]);
        atomic_max_nonneg(s_rho, rho);

        bool on_face[6] = {
            lx < margin, wx - 1 - lx < margin,
            v.dimension > 1 && ly < margin, v.dimension > 1 && wy - 1 - ly < margin,
            v.dimension > 2 && lz < margin, v.dimension > 2 && wz - 1 - lz < margin
        };
        double sq_prev = 0.0, sq_hi = 0.0;
        for (int l = 0; l <= ord; ++l) {
            const int c0 = deg_start[l], c1 = deg_start[l + 1];
            double sq = 0.0;
            for (int c = c0; c < c1; ++c) {
                const double val = row[c];
                sq += val * val;
            }
            atomic_max_nonneg(&s_E2[l], sq);
            if (l == ord - 1)
                sq_prev = sq;
            if (l == ord)
                sq_hi = sq;
        }
        const double sq_top = (ord == 1) ? sq_hi : (sq_prev + sq_hi);
        for (int f = 0; f < 2 * v.dimension; ++f)
            if (on_face[f])
                atomic_max_nonneg(&s_face2[f], sq_top);
    }
    __syncthreads();
    if (tid == 0)
        out_rho[T] = *s_rho;
    for (int l = tid; l < L1; l += blockDim.x)
        out_E[std::size_t(T) * L1 + l] = l <= ord ? sqrt(s_E2[l]) : 0.0;
    if (tid < 6)
        out_face_E[std::size_t(T) * 6 + tid] = sqrt(s_face2[tid]);
}

__global__ void k_adaptive_zero(View v, double *ba, double *bx, double *br, int tb0, int tb1, int tb2, int nt0, int nt1,
                                const int *shrunken, int count, const int *tile_q_new, const int *tile_q_old) {
    if (blockIdx.x >= count)
        return;
    const int T = shrunken[blockIdx.x];
    const int q_new = tile_q_new[T], q_old = tile_q_old[T];
    const int q_old_pad = q_old <= 4 ? 4 : min(v.q, 4 + ((q_old - 4 + 31) / 32) * 32);
    const int tx = T % nt0, ty = (T / nt0) % nt1, tz = T / (nt0 * nt1);
    const int x0 = tx * tb0, x1 = min(v.npa[0], x0 + tb0);
    const int y0 = ty * tb1, y1 = min(v.npa[1], y0 + tb1);
    const int z0 = tz * tb2, z1 = min(v.npa[2], z0 + tb2);
    const int wx = x1 - x0, wy = y1 - y0, wz = z1 - z0;
    const int tnodes = wx * wy * wz;
    const int q = v.q;
    // 1. Zero all three state buffers (a, x, r) on tile T for [q_new, q_old_pad) because SSPRK swaps them!
    const int span_u = q_old_pad - q_new;
    if (span_u > 0) {
        const std::size_t total = std::size_t(tnodes) * span_u;
        for (std::size_t k = threadIdx.x; k < total; k += blockDim.x) {
            const int idx = int(k / span_u), c = q_new + int(k % span_u);
            const int lx = idx % wx, ly = (idx / wx) % wy, lz = idx / (wx * wy);
            const std::size_t node = std::size_t(z0 + lz) * v.stride[2] + std::size_t(y0 + ly) * v.stride[1] + (x0 + lx);
            const std::size_t off = node * q + c;
            ba[off] = 0.0;
            bx[off] = 0.0;
            br[off] = 0.0;
        }
    }
    // 2. Zero dead gradient[d] on 1-node padded box of T for [q_new, q_old_pad) so no stale halo entries survive.
    const int span_g = q_old_pad - q_new;
    if (span_g > 0) {
        const int gx0 = max(0, x0 - 1), gx1 = min(v.npa[0], x1 + 1);
        const int gy0 = v.dimension > 1 ? max(0, y0 - 1) : 0, gy1 = v.dimension > 1 ? min(v.npa[1], y1 + 1) : 1;
        const int gz0 = v.dimension > 2 ? max(0, z0 - 1) : 0, gz1 = v.dimension > 2 ? min(v.npa[2], z1 + 1) : 1;
        const int gwx = gx1 - gx0, gwy = gy1 - gy0, gwz = gz1 - gz0;
        const std::size_t total = std::size_t(gwx) * gwy * gwz * span_g;
        for (std::size_t k = threadIdx.x; k < total; k += blockDim.x) {
            const int idx = int(k / span_g), c = q_new + int(k % span_g);
            const int lx = idx % gwx, ly = (idx / gwx) % gwy, lz = idx / (gwx * gwy);
            const std::size_t node = std::size_t(gz0 + lz) * v.stride[2] + std::size_t(gy0 + ly) * v.stride[1] + (gx0 + lx);
            for (int d = 0; d < v.dimension; ++d)
                v.gradient[d][node * q + c] = 0.0;
        }
    }
}

template <int P, int CH>
__global__ void k_line_solve_low4(View v, int first, double *field, const double *src) {
    const int d = first + blockIdx.y;
    double *f = field ? field : v.gradient[d];
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int n = v.npa[d], s = v.stride[d], lines = v.nodes / n;
    const int q4 = min(4, v.q);
    if (t >= std::size_t(lines) * q4)
        return;
    const int c = int(t % q4), line = int(t / q4);
    const int start = (line / s) * (s * n) + line % s;
    const int mi = v.mirror_index[d];
    const bool dirichlet = mi >= 0 && bool((v.comp_odd[c] >> d) & 1) != (field == nullptr);
    const int off = dirichlet && mi == 0 ? 1 : 0, nn = dirichlet ? n - 1 : n;
    const double *L = dirichlet ? v.dlower[d] : v.lower[d], *ID = dirichlet ? v.dinvdiag[d] : v.invdiag[d];
    const std::size_t step = std::size_t(s) * v.q;
    double *row0 = f + std::size_t(start) * v.q + c;
    double *base = row0 + std::size_t(off) * step;
    double hist[P > 0 ? P : 1];
    for (int m = 0; m < P; ++m)
        hist[m] = 0;
    const double *sbase = src ? src + std::size_t(start) * v.q + c + std::size_t(off) * step : nullptr;
    const double *Cb = v.convection_band[d] + std::size_t(off) * (2 * P + 1);
    double win[2 * P + 1];
    if (src) {
#pragma unroll
        for (int j = 0; j < 2 * P; ++j) {
            const int kk = j - P;
            win[j] = kk >= -off && kk < n - off ? sbase[std::ptrdiff_t(kk) * std::ptrdiff_t(step)] : 0.;
        }
    }
    auto load_fwd = [&](int k0, double *buf) {
#pragma unroll
        for (int j = 0; j < CH; ++j) {
            if (src)
                buf[j] = k0 + j + P < n - off ? sbase[std::size_t(k0 + j + P) * step] : 0.;
            else
                buf[j] = k0 + j < nn ? base[std::size_t(k0 + j) * step] : 0.;
        }
    };
    auto fwd = [&](int k0, const double *buf) {
#pragma unroll
        for (int j = 0; j < CH; ++j) {
            const int k = k0 + j;
            if (k >= nn)
                break;
            double y;
            if (src) {
                win[2 * P] = buf[j];
                y = 0;
#pragma unroll
                for (int m = 0; m <= 2 * P; ++m)
                    y += Cb[std::size_t(k) * (2 * P + 1) + m] * win[m];
#pragma unroll
                for (int m = 0; m < 2 * P; ++m)
                    win[m] = win[m + 1];
            } else
                y = buf[j];
#pragma unroll
            for (int m = 1; m <= P; ++m)
                if (k - m >= 0)
                    y -= L[std::size_t(k) * P + m - 1] * hist[m - 1];
#pragma unroll
            for (int m = P - 1; m > 0; --m)
                hist[m] = hist[m - 1];
            if (P > 0)
                hist[0] = y;
            base[std::size_t(k) * step] = y * ID[k];
        }
    };
    double bufA[CH], bufB[CH];
    load_fwd(0, bufA);
    for (int k0 = 0; k0 < nn; k0 += 2 * CH) {
        if (k0 + CH < nn)
            load_fwd(k0 + CH, bufB);
        fwd(k0, bufA);
        if (k0 + CH >= nn)
            break;
        if (k0 + 2 * CH < nn)
            load_fwd(k0 + 2 * CH, bufA);
        fwd(k0 + CH, bufB);
    }
    for (int m = 0; m < P; ++m)
        hist[m] = 0;
    auto load_bwd = [&](int k1, double *buf) {
#pragma unroll
        for (int j = 0; j < CH; ++j)
            buf[j] = k1 - j >= 0 ? base[std::size_t(k1 - j) * step] : 0.;
    };
    auto bwd = [&](int k1, const double *buf) {
#pragma unroll
        for (int j = 0; j < CH; ++j) {
            const int k = k1 - j;
            if (k < 0)
                break;
            double x = buf[j];
#pragma unroll
            for (int m = 1; m <= P; ++m)
                if (k + m < nn)
                    x -= L[std::size_t(k + m) * P + m - 1] * hist[m - 1];
#pragma unroll
            for (int m = P - 1; m > 0; --m)
                hist[m] = hist[m - 1];
            if (P > 0)
                hist[0] = x;
            base[std::size_t(k) * step] = x;
        }
    };
    load_bwd(nn - 1, bufA);
    for (int k1 = nn - 1; k1 >= 0; k1 -= 2 * CH) {
        if (k1 - CH >= 0)
            load_bwd(k1 - CH, bufB);
        bwd(k1, bufA);
        if (k1 - CH < 0)
            break;
        if (k1 - 2 * CH >= 0)
            load_bwd(k1 - 2 * CH, bufA);
        bwd(k1 - CH, bufB);
    }
    if (dirichlet)
        row0[std::size_t(mi) * step] = 0.0;
}

template <int SY>
__global__ void __launch_bounds__(128, 4) k_stabilization_strip_low4(
    View v, const double *a, double *result, bool want_low, double constant_gamma) {
    const int q = v.q, q4 = min(4, q), npx = v.npa[0], npy = v.npa[1], cx = v.count[0], cy = v.count[1];
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= npx * q4)
        return;
    const int ix = col / q4, c = col % q4;
    const int y0 = blockIdx.y * SY, y1 = min(npy, y0 + SY);
    // Row pointers (column ix, moment c) advanced by one node row per y; x neighbours at -q / +q. The element columns
    // ex of the two quadrants depend only on ix. Same loads and the same FMA order as before (fewer integer ops:
    // the 64-bit index products per load dominated the issued instructions).
    const std::size_t row = std::size_t(npx) * q;
    const bool xm = ix > 0, xp = ix + 1 < npx;
    const double *P0 = a + std::size_t(ix) * q + c, *P1 = v.gradient[0] + std::size_t(ix) * q + c,
                 *P2 = v.gradient[1] + std::size_t(ix) * q + c;
    int exq[2];
    bool exok[2];
#pragma unroll
    for (int qx = 0; qx < 2; ++qx) {
        exq[qx] = v.strip_elem[0] ? v.strip_elem[0][2 * ix + qx] : ix - 1 + qx;
        exok[qx] = exq[qx] >= 0 && exq[qx] < cx;
    }
    double W[3][3][3];
    auto load_row = [&](int y, int r) {
        if (y < 0 || y >= npy) {
#pragma unroll
            for (int f = 0; f < 3; ++f)
#pragma unroll
                for (int dx = 0; dx < 3; ++dx)
                    W[f][r][dx] = 0.;
            return;
        }
        const std::size_t off = std::size_t(y) * row;
        const double *B[3] = {P0 + off, P1 + off, P2 + off};
#pragma unroll
        for (int f = 0; f < 3; ++f) {
            W[f][r][0] = xm ? B[f][-q] : 0.;
            W[f][r][1] = B[f][0];
            W[f][r][2] = xp ? B[f][q] : 0.;
        }
    };
    load_row(y0 - 1, 0);
    load_row(y0, 1);
    for (int y = y0; y < y1; ++y) {
        load_row(y + 1, 2);
        double out = 0;
#pragma unroll
        for (int qy = 0; qy < 2; ++qy)
#pragma unroll
            for (int qx = 0; qx < 2; ++qx) {
                const int ex = exq[qx];
                const int ey = v.strip_elem[1] ? v.strip_elem[1][2 * y + qy] : y - 1 + qy;
                if (!exok[qx] || ey < 0 || ey >= cy)
                    continue;
                const int e = ex + ey * cx, l = (1 - qx) + 2 * (1 - qy);
                const int kc = 16 * klass_of(v, e);
                const double g = constant_gamma >= 0 ? constant_gamma : v.gamma[e];
                const double weight = -__ldg(v.strip_nu + kc / 16) * g;
                double stiff = 0, conv = 0;
#pragma unroll
                for (int my = 0; my < 2; ++my)
#pragma unroll
                    for (int mx = 0; mx < 2; ++mx) {
                        const int m = mx + 2 * my;
                        stiff += __ldg(v.strip_kv + kc + 4 * l + m) * W[0][qy + my][qx + mx];
                        conv += __ldg(v.strip_c0 + kc + 4 * l + m) * W[1][qy + my][qx + mx] +
                                __ldg(v.strip_c1 + kc + 4 * l + m) * W[2][qy + my][qx + mx];
                    }
                const double term = stiff + weight * conv;
                if (want_low) {
                    const int slot = c == 0 ? 0 : (c == 1 ? 2 : (c == 2 ? 3 : 1));
                    v.stab_low[std::size_t(4 * l + slot) * v.cells + e] = (c == 0 ? v.s0 : v.s1) * term;
                }
                out -= term;
            }
        result[std::size_t(y) * row + std::size_t(ix) * q + c] = out;
#pragma unroll
        for (int f = 0; f < 3; ++f)
#pragma unroll
            for (int dx = 0; dx < 3; ++dx) {
                W[f][0][dx] = W[f][1][dx];
                W[f][1][dx] = W[f][2][dx];
            }
    }
}

__global__ void k_stabilization_low4(View v, const double *a, double *result, bool want_low, double constant_gamma) {
    const int q = v.q, q4 = min(4, q), n = v.local;
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(v.nodes) * q4)
        return;
    const int i = int(t / q4), c = int(t % q4);
    double out = 0;
    for (int k = v.node_offsets[i]; k < v.node_offsets[i + 1]; ++k) {
        const int entry = v.node_entries[k], e = entry / n, l = entry % n;
        const int *map = v.connectivity + std::size_t(e) * n;
        const int cl = klass_of(v, e);
        const double nu = v.class_viscosity ? v.class_viscosity[cl] : v.viscosity;
        const double *K = v.stiffness_rows + (std::size_t(cl) * n + l) * n;
        double term = 0;
        for (int m = 0; m < n; ++m)
            term += K[m] * a[std::size_t(map[m]) * q + c];
        term *= nu;
        const double g = constant_gamma >= 0 ? constant_gamma : v.gamma[e];
        const double weight = -nu * g;
        if (weight != 0)
            for (int d = 0; d < v.dimension; ++d) {
                const double *C = v.convection_t_rows[d] + (std::size_t(cl) * n + l) * n;
                double cd = 0;
                for (int m = 0; m < n; ++m)
                    cd += C[m] * v.gradient[d][std::size_t(map[m]) * q + c];
                term += weight * cd;
            }
        if (want_low) {
            const int slot = c == 0 ? 0 : (c == 1 ? 2 : (c == 2 ? 3 : 1));
            v.stab_low[std::size_t(4 * l + slot) * v.cells + e] = (c == 0 ? v.s0 : v.s1) * term;
        }
        out -= term;
    }
    result[std::size_t(i) * q + c] = out;
}

__global__ void k_react_low4(View v, ReactionView r, double *a, double tau, bool fcs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double absorbed = 0.0, injected = 0.0;
    if (i < v.nodes) {
        const double sa = r.sigma_a ? r.sigma_a[i] : r.sa, ss = r.sigma_s ? r.sigma_s[i] : r.ss;
        const double q0 = (r.source ? r.source[i] : 0.) + (fcs ? r.fcs[i] : 0.);
        const double c0 = 1 / v.s0, z = -sa * tau, decay0 = exp(z), phi1 = z == 0 ? 1. : expm1(z) / z;
        double *row = a + std::size_t(i) * v.q;
        const double rho = row[0] / c0;
        row[0] = decay0 * row[0] + c0 * q0 * tau * phi1;
        const int q4 = min(4, v.q);
        if (!r.has_filter) {
            const double d1 = exp(-tau * (sa + ss));
            for (int k = 1; k < q4; ++k)
                row[k] *= d1;
        } else {
            for (int k = 1; k < q4; ++k)
                row[k] *= exp(-tau * (sa + ss + r.filter[k]));
        }
        absorbed = v.lumped[i] * (-expm1(z) * rho + q0 * tau * (1 - phi1));
        injected = v.lumped[i] * q0 * tau;
    }
    absorbed = warp_sum(absorbed), injected = warp_sum(injected);
    if ((threadIdx.x & 31) == 0 && (absorbed != 0.0 || injected != 0.0)) {
        atomicAdd(&r.ledger[0], absorbed);
        atomicAdd(&r.ledger[1], injected);
    }
}

template <int P, int CH>
__global__ void k_line_solve_adaptive(View v, int d, double *field, const double *src,
                                      const AdaptiveLineSeg *segs, int nsegs,
                                      const int *seg_slot, const int *seg_off, const double *seg_lower,
                                      const double *seg_invdiag, const int *dseg_slot, const int *dseg_off,
                                      const double *dseg_lower, const double *dseg_invdiag) {
    const int seg_id = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (seg_id >= nsegs)
        return;
    const AdaptiveLineSeg seg = segs[seg_id];
    const int c = seg.c0 + (threadIdx.x & 31);
    if (c >= v.q)
        return;
    double *f = field ? field : v.gradient[d];
    const int n = v.npa[d], s = v.stride[d], line = seg.line;
    const int start = (line / s) * (s * n) + line % s;
    const int mi = v.mirror_index[d];
    const bool dirichlet = mi >= 0 && bool((v.comp_odd[c] >> d) & 1) != (field == nullptr);
    const int off = dirichlet && mi == 0 ? 1 : 0, nn = dirichlet ? n - 1 : n;
    const int r_start = max(0, seg.k_start - off);
    const int r_end = min(nn, seg.k_end - off);
    const std::size_t step = std::size_t(s) * v.q;
    double *row0 = f + std::size_t(start) * v.q + c;
    if (r_start >= r_end) {
        if (dirichlet && seg.k_start <= mi && mi < seg.k_end)
            row0[std::size_t(mi) * step] = 0.0;
        return;
    }
    const int slot = (dirichlet ? dseg_slot : seg_slot)[r_start];
    const int base_off = (dirichlet ? dseg_off : seg_off)[slot];
    const double *L = (dirichlet ? dseg_lower : seg_lower) + std::size_t(base_off) * P;
    const double *ID = (dirichlet ? dseg_invdiag : seg_invdiag) + base_off;
    double *base = row0 + std::size_t(off) * step;
    double hist[P > 0 ? P : 1];
    for (int m = 0; m < P; ++m)
        hist[m] = 0;
    const double *sbase = src ? src + std::size_t(start) * v.q + c + std::size_t(off) * step : nullptr;
    const double *Cb = v.convection_band[d] + std::size_t(off) * (2 * P + 1);
    double win[2 * P + 1];
    if (src) {
#pragma unroll
        for (int j = 0; j < 2 * P; ++j) {
            const int kk = r_start + j - P;
            win[j] = kk >= -off && kk < n - off ? sbase[std::ptrdiff_t(kk) * std::ptrdiff_t(step)] : 0.;
        }
    }
    auto load_fwd = [&](int k0, double *buf) {
#pragma unroll
        for (int j = 0; j < CH; ++j) {
            if (src)
                buf[j] = (k0 + j < r_end && k0 + j + P < n - off) ? sbase[std::size_t(k0 + j + P) * step] : 0.;
            else
                buf[j] = k0 + j < r_end ? base[std::size_t(k0 + j) * step] : 0.;
        }
    };
    auto fwd = [&](int k0, const double *buf) {
#pragma unroll
        for (int j = 0; j < CH; ++j) {
            const int k = k0 + j;
            if (k >= r_end)
                break;
            double y;
            if (src) {
                win[2 * P] = buf[j];
                y = 0;
#pragma unroll
                for (int m = 0; m <= 2 * P; ++m)
                    y += Cb[std::size_t(k) * (2 * P + 1) + m] * win[m];
#pragma unroll
                for (int m = 0; m < 2 * P; ++m)
                    win[m] = win[m + 1];
            } else
                y = buf[j];
#pragma unroll
            for (int m = 1; m <= P; ++m)
                if (k - m >= r_start)
                    y -= L[std::size_t(k - r_start) * P + m - 1] * hist[m - 1];
#pragma unroll
            for (int m = P - 1; m > 0; --m)
                hist[m] = hist[m - 1];
            if (P > 0)
                hist[0] = y;
            base[std::size_t(k) * step] = y * ID[k - r_start];
        }
    };
    double bufA[CH], bufB[CH];
    load_fwd(r_start, bufA);
    for (int k0 = r_start; k0 < r_end; k0 += 2 * CH) {
        if (k0 + CH < r_end)
            load_fwd(k0 + CH, bufB);
        fwd(k0, bufA);
        if (k0 + CH >= r_end)
            break;
        if (k0 + 2 * CH < r_end)
            load_fwd(k0 + 2 * CH, bufA);
        fwd(k0 + CH, bufB);
    }
    for (int m = 0; m < P; ++m)
        hist[m] = 0;
    auto load_bwd = [&](int k1, double *buf) {
#pragma unroll
        for (int j = 0; j < CH; ++j)
            buf[j] = k1 - j >= r_start ? base[std::size_t(k1 - j) * step] : 0.;
    };
    auto bwd = [&](int k1, const double *buf) {
#pragma unroll
        for (int j = 0; j < CH; ++j) {
            const int k = k1 - j;
            if (k < r_start)
                break;
            double x = buf[j];
#pragma unroll
            for (int m = 1; m <= P; ++m)
                if (k + m < r_end)
                    x -= L[std::size_t(k + m - r_start) * P + m - 1] * hist[m - 1];
#pragma unroll
            for (int m = P - 1; m > 0; --m)
                hist[m] = hist[m - 1];
            if (P > 0)
                hist[0] = x;
            base[std::size_t(k) * step] = x;
        }
    };
    load_bwd(r_end - 1, bufA);
    for (int k1 = r_end - 1; k1 >= r_start; k1 -= 2 * CH) {
        if (k1 - CH >= r_start)
            load_bwd(k1 - CH, bufB);
        bwd(k1, bufA);
        if (k1 - CH < r_start)
            break;
        if (k1 - 2 * CH >= r_start)
            load_bwd(k1 - 2 * CH, bufA);
        bwd(k1 - CH, bufB);
    }
    if (dirichlet && seg.k_start <= mi && mi < seg.k_end)
        row0[std::size_t(mi) * step] = 0.0;
}

template <int MINB>
__global__ void __launch_bounds__(128, MINB) k_stabilization_strip_adaptive(
    View v, const double *a, double *result, bool want_low, double constant_gamma,
    int tb0, int tb1, const AdaptiveTileChunk *chunks, int nchunks, const int *tile_q, int nt0) {
    if (blockIdx.x >= nchunks)
        return;
    const AdaptiveTileChunk tc = chunks[blockIdx.x];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int q = v.q, npx = v.npa[0], npy = v.npa[1], cx = v.count[0], cy = v.count[1];
    const int c = tc.c0 + lane;
    if (c >= q)
        return;
    const int x0 = tc.tx * tb0, x1 = min(npx, x0 + tb0);
    const int y0 = tc.ty * tb1, y1 = min(npy, y0 + tb1);
    const bool active_c = c < tc.q_lim;
    const std::size_t row = std::size_t(npx) * q;
    for (int ix = x0 + warp; ix < x1; ix += 4) {
        if (!active_c) {
            for (int y = y0; y < y1; ++y)
                result[std::size_t(y) * row + std::size_t(ix) * q + c] = 0.0;
            continue;
        }
        // Row pointers as in k_stabilization_strip. As ElementWeno::stabilization_residual (cm_of): neighbour node
        // contributions only for moments active at that node (the padded gradient segments leave nonzero inactive
        // gradient entries); rows y0-1 .. y1 lie in the tile rows ty-1, ty, ty+1, so the activity of the three columns
        // is looked up once per column (same values as c < tile_q[(y / tb1) * nt0 + x / tb0]).
        bool act[3][3];
#pragma unroll
        for (int dx = 0; dx < 3; ++dx) {
            const int x = ix + dx - 1;
            const bool xin = x >= 0 && x < npx;
            const int tx = xin ? x / tb0 : 0;
            act[0][dx] = xin && y0 > 0 && c < tile_q[((y0 - 1) / tb1) * nt0 + tx];
            act[1][dx] = xin && c < tile_q[tc.ty * nt0 + tx];
            act[2][dx] = xin && y1 < npy && c < tile_q[(y1 / tb1) * nt0 + tx];
        }
        const double *P0 = a + std::size_t(ix) * q + c, *P1 = v.gradient[0] + std::size_t(ix) * q + c,
                     *P2 = v.gradient[1] + std::size_t(ix) * q + c;
        double W[3][3][3];
        auto load_row = [&](int y, int r) {
            const int band = y < y0 ? 0 : (y < y1 ? 1 : 2);
            if (y < 0 || y >= npy) {
#pragma unroll
                for (int f = 0; f < 3; ++f)
#pragma unroll
                    for (int dx = 0; dx < 3; ++dx)
                        W[f][r][dx] = 0.;
                return;
            }
            const std::size_t off = std::size_t(y) * row;
            const double *B[3] = {P0 + off, P1 + off, P2 + off};
#pragma unroll
            for (int f = 0; f < 3; ++f) {
                W[f][r][0] = act[band][0] ? B[f][-q] : 0.;
                W[f][r][1] = act[band][1] ? B[f][0] : 0.;
                W[f][r][2] = act[band][2] ? B[f][q] : 0.;
            }
        };
        load_row(y0 - 1, 0);
        load_row(y0, 1);
        for (int y = y0; y < y1; ++y) {
            load_row(y + 1, 2);
            double out = 0;
#pragma unroll
            for (int qy = 0; qy < 2; ++qy)
#pragma unroll
                for (int qx = 0; qx < 2; ++qx) {
                    const int ex = v.strip_elem[0] ? v.strip_elem[0][2 * ix + qx] : ix - 1 + qx;
                    const int ey = v.strip_elem[1] ? v.strip_elem[1][2 * y + qy] : y - 1 + qy;
                    if (ex < 0 || ex >= cx || ey < 0 || ey >= cy)
                        continue;
                    const int e = ex + ey * cx, l = (1 - qx) + 2 * (1 - qy);
                    const int kc = 16 * klass_of(v, e);
                    const double g = constant_gamma >= 0 ? constant_gamma : v.gamma[e];
                    const double weight = -__ldg(v.strip_nu + kc / 16) * g;
                    double stiff = 0, conv = 0;
#pragma unroll
                    for (int my = 0; my < 2; ++my)
#pragma unroll
                        for (int mx = 0; mx < 2; ++mx) {
                            const int m = mx + 2 * my;
                            stiff += __ldg(v.strip_kv + kc + 4 * l + m) * W[0][qy + my][qx + mx];
                            conv += __ldg(v.strip_c0 + kc + 4 * l + m) * W[1][qy + my][qx + mx] +
                                    __ldg(v.strip_c1 + kc + 4 * l + m) * W[2][qy + my][qx + mx];
                        }
                    const double term = stiff + weight * conv;
                    if (want_low && c < 4) {
                        const int slot = c == 0 ? 0 : (c == 1 ? 2 : (c == 2 ? 3 : 1));
                        v.stab_low[std::size_t(4 * l + slot) * v.cells + e] = (c == 0 ? v.s0 : v.s1) * term;
                    }
                    out -= term;
                }
            result[std::size_t(y) * row + std::size_t(ix) * q + c] = out;
#pragma unroll
            for (int f = 0; f < 3; ++f)
#pragma unroll
                for (int dx = 0; dx < 3; ++dx) {
                    W[f][0][dx] = W[f][1][dx];
                    W[f][1][dx] = W[f][2][dx];
                }
        }
    }
}

__global__ void k_stabilization_adaptive(View v, const double *a, double *result, bool want_low, double constant_gamma,
                                         int tb0, int tb1, int tb2, const AdaptiveTileChunk *chunks, int nchunks,
                                         const int *tile_q, int nt0, int nt1) {
    if (blockIdx.x >= nchunks)
        return;
    const AdaptiveTileChunk tc = chunks[blockIdx.x];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int q = v.q, n = v.local;
    const int c = tc.c0 + lane;
    if (c >= q)
        return;
    const int x0 = tc.tx * tb0, x1 = min(v.npa[0], x0 + tb0);
    const int y0 = tc.ty * tb1, y1 = min(v.npa[1], y0 + tb1);
    const int z0 = tc.tz * tb2, z1 = min(v.npa[2], z0 + tb2);
    const int wx = x1 - x0, wy = y1 - y0, wz = z1 - z0, tnodes = wx * wy * wz;
    const bool active_c = c < tc.q_lim;
    for (int idx = warp; idx < tnodes; idx += 4) {
        const int lx = idx % wx, ly = (idx / wx) % wy, lz = idx / (wx * wy);
        const int i = (z0 + lz) * v.stride[2] + (y0 + ly) * v.stride[1] + (x0 + lx);
        if (!active_c) {
            result[std::size_t(i) * q + c] = 0.0;
            continue;
        }
        double out = 0;
        for (int k = v.node_offsets[i]; k < v.node_offsets[i + 1]; ++k) {
            const int entry = v.node_entries[k], e = entry / n, l = entry % n;
            const int *map = v.connectivity + std::size_t(e) * n;
            const int cl = klass_of(v, e);
            const double nu = v.class_viscosity ? v.class_viscosity[cl] : v.viscosity;
            const double *K = v.stiffness_rows + (std::size_t(cl) * n + l) * n;
            // As ElementWeno::stabilization_residual (cm_of): neighbour contributions only for moments active there.
            bool act[27];
            for (int m = 0; m < n; ++m) {
                const int node = map[m], px = node % v.npa[0], py = (node / v.npa[0]) % v.npa[1];
                const int pz = node / (v.npa[0] * v.npa[1]);
                act[m] = c < tile_q[((pz / tb2) * nt1 + py / tb1) * nt0 + px / tb0];
            }
            double term = 0;
            for (int m = 0; m < n; ++m)
                if (act[m])
                    term += K[m] * a[std::size_t(map[m]) * q + c];
            term *= nu;
            const double g = constant_gamma >= 0 ? constant_gamma : v.gamma[e];
            const double weight = -nu * g;
            if (weight != 0)
                for (int d = 0; d < v.dimension; ++d) {
                    const double *C = v.convection_t_rows[d] + (std::size_t(cl) * n + l) * n;
                    double cd = 0;
                    for (int m = 0; m < n; ++m)
                        if (act[m])
                            cd += C[m] * v.gradient[d][std::size_t(map[m]) * q + c];
                    term += weight * cd;
                }
            if (want_low && c < 4) {
                const int slot = c == 0 ? 0 : (c == 1 ? 2 : (c == 2 ? 3 : 1));
                v.stab_low[std::size_t(4 * l + slot) * v.cells + e] = (c == 0 ? v.s0 : v.s1) * term;
            }
            out -= term;
        }
        result[std::size_t(i) * q + c] = out;
    }
}

__global__ void k_combine_adaptive(View v, double *r, const double *x, const double *a, double old, double fresh,
                                   double tau, int tb0, int tb1, int tb2, const AdaptiveTileChunk *chunks, int nchunks) {
    extern __shared__ double shared[];
    const FluxShared F = stage_flux(v, shared);
    if (blockIdx.x >= nchunks)
        return;
    const AdaptiveTileChunk tc = chunks[blockIdx.x];
    const int warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5, lane = threadIdx.x & 31;
    const int q = v.q, c = tc.c0 + lane;
    if (c < 4 || c >= q)
        return;
    const int x0 = tc.tx * tb0, x1 = min(v.npa[0], x0 + tb0);
    const int y0 = tc.ty * tb1, y1 = min(v.npa[1], y0 + tb1);
    const int z0 = tc.tz * tb2, z1 = min(v.npa[2], z0 + tb2);
    const int wx = x1 - x0, wy = y1 - y0, wz = z1 - z0, tnodes = wx * wy * wz;
    const bool active_c = c < tc.q_lim;
    const int qpad = tc.q_lim <= 4 ? min(4, q) : min(q, 4 + ((tc.q_lim - 4 + 31) / 32) * 32);
    for (int idx = warp; idx < tnodes; idx += nwarps) {
        const int lx = idx % wx, ly = (idx / wx) % wy, lz = idx / (wx * wy);
        const int i = (z0 + lz) * v.stride[2] + (y0 + ly) * v.stride[1] + (x0 + lx);
        const std::size_t t = std::size_t(i) * q + c;
        if (!active_c) {
            r[t] = 0.0;
            continue;
        }
        double rate = r[t];
        for (int d = 0; d < v.dimension; ++d) {
            const double *g = v.gradient[d] + std::size_t(i) * q;
            rate += -F.row_upto(g, d, c, qpad);
        }
        if (v.node_mirror && (v.node_mirror[i] & v.comp_odd[c]))
            rate = 0.0;
        r[t] = old == 0 ? fresh * (x[t] + tau * rate) : old * a[t] + fresh * (x[t] + tau * rate);
    }
}

__global__ void k_react_rows_adaptive(View v, ReactionView r, double *a, double tau, bool fcs,
                                      int tb0, int tb1, int tb2, const AdaptiveTileChunk *chunks, int nchunks) {
    (void)fcs;
    if (blockIdx.x >= nchunks)
        return;
    const AdaptiveTileChunk tc = chunks[blockIdx.x];
    const int warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5, lane = threadIdx.x & 31;
    const int q = v.q, c = tc.c0 + lane;
    const bool active_c = (c < tc.q_lim && c < q);
    const int x0 = tc.tx * tb0, x1 = min(v.npa[0], x0 + tb0);
    const int y0 = tc.ty * tb1, y1 = min(v.npa[1], y0 + tb1);
    const int z0 = tc.tz * tb2, z1 = min(v.npa[2], z0 + tb2);
    const int wx = x1 - x0, wy = y1 - y0, wz = z1 - z0, tnodes = wx * wy * wz;
    const double fc = active_c && r.has_filter ? r.filter[c] : 0.0;
    for (int idx = warp; idx < tnodes; idx += nwarps) {
        const int lx = idx % wx, ly = (idx / wx) % wy, lz = idx / (wx * wy);
        const int i = (z0 + lz) * v.stride[2] + (y0 + ly) * v.stride[1] + (x0 + lx);
        const double sa = r.sigma_a ? r.sigma_a[i] : r.sa, ss = r.sigma_s ? r.sigma_s[i] : r.ss;
        if (!r.has_filter) {
            // All 32 moments in this warp share the exact same isotropic decay factor exp(-tau * (sa + ss)):
            // compute ONE FP64 exp() per warp and broadcast via shuffle (32x fewer transcendentals!).
            double decay = lane == 0 ? exp(-tau * (sa + ss)) : 0.0;
            decay = __shfl_sync(0xffffffffu, decay, 0);
            if (active_c)
                a[std::size_t(i) * q + c] *= decay;
        } else if (active_c) {
            a[std::size_t(i) * q + c] *= exp(-tau * (sa + ss + fc));
        }
    }
}

__global__ void k_face_traces_active(int q_stride, int q_eff, int rows, int nf, const int *row_nodes,
                                     const double *row_mass, const double *x, double *G) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(rows) * q_eff)
        return;
    const int r = int(t / q_eff), c = int(t % q_eff);
    double sum = 0;
    for (int m = 0; m < nf; ++m)
        sum += row_mass[std::size_t(r) * nf + m] * x[std::size_t(row_nodes[std::size_t(r) * nf + m]) * q_stride + c];
    G[std::size_t(r) * q_stride + c] = sum;
}

__global__ void k_sub_active(int q_stride, int q_eff, int rows, double *a, const double *b) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(rows) * q_eff)
        return;
    const int r = int(t / q_eff), c = int(t % q_eff);
    const std::size_t idx = std::size_t(r) * q_stride + c;
    a[idx] -= b[idx];
}

__global__ void k_boundary_rows_add_active(int q_stride, int q_eff, int nb, const int *offsets, const int *rows_list,
                                           const int *bnodes, const double *H, double *out, double *low4,
                                           const int *row_break, double *low4b, int dim, int npa0, int npa1,
                                           int tb0, int tb1, int tb2, int nt0, int nt1, const int *tile_q) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(nb) * q_eff)
        return;
    const int k = int(t / q_eff), c = int(t % q_eff);
    const int bn = bnodes[k];
    const int ix = bn % npa0;
    const int iy = dim > 1 ? (bn / npa0) % npa1 : 0;
    const int iz = dim > 2 ? bn / (npa0 * npa1) : 0;
    const int T = ((iz / tb2) * nt1 + (iy / tb1)) * nt0 + (ix / tb0);
    if (c >= tile_q[T])
        return;
    double sum = 0;
    for (int e = offsets[k]; e < offsets[k + 1]; ++e)
        sum += H[std::size_t(rows_list[e]) * q_stride + c];
    out[std::size_t(bn) * q_stride + c] += sum;
    if (c < 4) {
        low4[std::size_t(k) * 4 + c] = sum;
        if (row_break) {
            double part = 0;
            for (int e = offsets[k]; e < offsets[k + 1]; ++e)
                if (row_break[rows_list[e]])
                    part += H[std::size_t(rows_list[e]) * q_stride + c];
            low4b[std::size_t(k) * 4 + c] = part;
        }
    }
}

void react_rows_adaptive_launch(Context *ctx, const ReactionView &r, double *state, double tau, bool fcs) {
    AdaptiveDevice *ad = ctx->adaptive;
    const View &v = ctx->v;
    if (!ad || !ad->active) {
        k_react_rows<<<blocks(std::size_t(v.nodes) * v.q), 256, 0, ctx->stream>>>(v, r, state, tau, fcs);
        return;
    }
    k_react_low4<<<blocks(v.nodes), 256, 0, ctx->stream>>>(v, r, state, tau, fcs);
    if (ad->tile_chunks_n > 0)
        k_react_rows_adaptive<<<ad->tile_chunks_n, 256, 0, ctx->stream>>>(
            v, r, state, tau, fcs, ad->tb[0], ad->tb[1], ad->tb[2], ad->d_tile_chunks, ad->tile_chunks_n);
}

// Sparse broken-line interface product: H(r, c) = 0.5 * (sgn * (G_r * A_d^T)[c] - G(r, c)) using v.ell_val / v.ell_idx.
__global__ void k_boundary_interface_sparse(View v, int r_begin, int nrows, int q_eff, int dir, double sgn,
                                            const double *G, double *H) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = std::size_t(nrows) * q_eff;
    if (t >= total)
        return;
    const int r = r_begin + int(t / q_eff);
    const int c = int(t % q_eff);
    const double *g_row = G + std::size_t(r) * v.q;
    const std::size_t off = (std::size_t(dir) * v.q + c) * v.flux_width;
    double sum = 0.0;
    for (int w = 0; w < v.flux_width; ++w) {
        const double val = v.ell_val[off + w];
        const int idx = v.ell_idx[off + w];
        if (val != 0.0 && idx < q_eff)
            sum += val * g_row[idx];
    }
    H[std::size_t(r) * v.q + c] = 0.5 * (sgn * sum - g_row[c]);
}

bool boundary_high_adaptive_launch(Context *ctx, BoundaryHighDevice *d, const double *state, double *out,
                                   const double *pinned_db, double *pinned_low4, double *pinned_low4_break) {
    AdaptiveDevice *ad = ctx->adaptive;
    if (!ad || !ad->active)
        return false;
    const BoundaryHighSetup &s = d->s;
    const View &v = ctx->v;
    const int q_bnd = std::max(4, std::min(s.q, ad->bnd_max_q));
    k_face_traces_active<<<blocks(std::size_t(s.rows) * q_bnd), 256, 0, ctx->stream>>>(
        s.q, q_bnd, s.rows, s.nf, d->row_nodes, d->row_mass, state, d->G);
    const double one = 1, zero = 0;
    for (int side = 0; side < 6; ++side) {
        const int r0 = s.side_start[side], r1 = d->iface_t[side] ? s.break_start[side] : s.side_start[side + 1];
        if (d->flux_t[side] && r1 > r0) {
            const int q_s = std::max(4, std::min(q_bnd, ad->side_q_dom[side]));
            cublas_check(cublasDgemm(d->handle, CUBLAS_OP_T, CUBLAS_OP_N, q_s, r1 - r0, q_s, &one,
                                     d->flux_t[side], s.q, d->G + std::size_t(r0) * s.q, s.q, &zero,
                                     d->H + std::size_t(r0) * s.q, s.q),
                         "dgemm (adaptive)");
        }
        const int r2 = s.side_start[side + 1];
        if (d->iface_t[side] && r2 > r1) {
            const int q_b = std::max(4, std::min(q_bnd, ad->side_q_brk[side]));
            const int dir = side / 2;
            const double sgn = (side & 1) ? 1.0 : -1.0;
            k_boundary_interface_sparse<<<blocks(std::size_t(r2 - r1) * q_b), 256, 0, ctx->stream>>>(
                v, r1, r2 - r1, q_b, dir, sgn, d->G, d->H);
        }
    }
    if (pinned_db) {
        const std::size_t rq = std::size_t(s.rows) * s.q;
        CK(cudaMemcpyAsync(d->db, pinned_db, rq * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
        k_sub_active<<<blocks(std::size_t(s.rows) * q_bnd), 256, 0, ctx->stream>>>(s.q, q_bnd, s.rows, d->H, d->db);
    }
    k_boundary_rows_add_active<<<blocks(std::size_t(s.nb) * q_bnd), 256, 0, ctx->stream>>>(
        s.q, q_bnd, s.nb, d->offsets, d->rows_list, s.bnodes, d->H, out, d->low4, d->row_break, d->low4b,
        v.dimension, v.npa[0], v.npa[1], ad->tb[0], ad->tb[1], ad->tb[2], ad->nt[0], ad->nt[1], ad->d_tile_q);
    CK(cudaMemcpyAsync(pinned_low4, d->low4, std::size_t(s.nb) * 4 * sizeof(double), cudaMemcpyDeviceToHost, ctx->stream));
    if (pinned_low4_break && d->low4b)
        CK(cudaMemcpyAsync(pinned_low4_break, d->low4b, std::size_t(s.nb) * 4 * sizeof(double), cudaMemcpyDeviceToHost,
                           ctx->stream));
    return true;
}

void destroy_adaptive(Context *ctx) {
    AdaptiveDevice *ad = ctx->adaptive;
    if (!ad)
        return;
    for (void *p : {(void *)ad->d_deg_start, (void *)ad->d_tile_order, (void *)ad->d_tile_q,
                    (void *)ad->d_tile_q_prev, (void *)ad->d_shrunken, (void *)ad->d_tile_rho,
                    (void *)ad->d_tile_E, (void *)ad->d_tile_face_E, (void *)ad->d_tile_chunks})
        cudaFree(p);
    for (int d = 0; d < 3; ++d) {
        cudaFree(ad->d_segs_grad[d]);
        cudaFree(ad->d_segs_mass[d]);
        cudaFree(ad->d_seg_slot[d]);
        cudaFree(ad->d_seg_off[d]);
        cudaFree(ad->d_seg_lower[d]);
        cudaFree(ad->d_seg_invdiag[d]);
        cudaFree(ad->d_dseg_slot[d]);
        cudaFree(ad->d_dseg_off[d]);
        cudaFree(ad->d_dseg_lower[d]);
        cudaFree(ad->d_dseg_invdiag[d]);
    }
    delete ad;
    ctx->adaptive = nullptr;
}
} // namespace

void adaptive_setup(Context *ctx, const AdaptivePnPlan &plan, const AdaptiveSegFactorTable &seg_factors) {
    destroy_adaptive(ctx);
    auto *ad = new AdaptiveDevice;
    ad->num_tiles = plan.num_tiles;
    ad->max_order = plan.max_order;
    ad->L1 = plan.max_order + 1;
    for (int d = 0; d < 3; ++d) {
        ad->tb[d] = plan.tb[d], ad->nt[d] = plan.nt[d];
        if (!seg_factors.slot[d].empty()) {
            ad->d_seg_slot[d] = dcopy(seg_factors.slot[d].data(), seg_factors.slot[d].size());
            ad->d_seg_off[d] = dcopy(seg_factors.off[d].data(), seg_factors.off[d].size());
            ad->d_seg_lower[d] = dcopy(seg_factors.lower[d].data(), seg_factors.lower[d].size());
            ad->d_seg_invdiag[d] = dcopy(seg_factors.invdiag[d].data(), seg_factors.invdiag[d].size());
        }
        if (!seg_factors.dslot[d].empty()) {
            ad->d_dseg_slot[d] = dcopy(seg_factors.dslot[d].data(), seg_factors.dslot[d].size());
            ad->d_dseg_off[d] = dcopy(seg_factors.doff[d].data(), seg_factors.doff[d].size());
            ad->d_dseg_lower[d] = dcopy(seg_factors.dlower[d].data(), seg_factors.dlower[d].size());
            ad->d_dseg_invdiag[d] = dcopy(seg_factors.dinvdiag[d].data(), seg_factors.dinvdiag[d].size());
        }
    }
    ad->d_deg_start = dcopy(plan.deg_start.data(), plan.deg_start.size());
    ad->d_tile_order = dcopy(plan.tile_order.data(), plan.tile_order.size());
    ad->d_tile_q = dcopy(plan.tile_q.data(), plan.tile_q.size());
    ad->d_tile_q_prev = dcopy(plan.tile_q_prev.data(), plan.tile_q_prev.size());
    ad->d_shrunken = dalloc<int>(std::max(1, plan.num_tiles));
    ad->d_tile_rho = dalloc<double>(std::max(1, plan.num_tiles));
    ad->d_tile_E = dalloc<double>(std::size_t(std::max(1, plan.num_tiles)) * ad->L1);
    ad->d_tile_face_E = dalloc<double>(std::size_t(std::max(1, plan.num_tiles)) * 6);
    ad->global_max_q = plan.global_max_q;
    ad->bnd_max_q = plan.bnd_max_q;
    for (int s = 0; s < 6; ++s) {
        ad->side_q_dom[s] = plan.side_q_dom[s];
        ad->side_q_brk[s] = plan.side_q_brk[s];
    }
    ad->active = false;
    ctx->adaptive = ad;
}

void adaptive_enable(Context *ctx, bool on) {
    if (ctx->adaptive)
        ctx->adaptive->active = on;
}

void adaptive_sensor(Context *ctx, const double *state, bool initial, double *host_rho, double *host_E,
                     double *host_face_E) {
    AdaptiveDevice *ad = ctx->adaptive;
    const View &v = ctx->v;
    if (initial) {
        // Clear stage scratch and gradient buffers (spectral_radius() leaves random data in b.x, b.r, gradient).
        const std::size_t bytes = sizeof(double) * std::size_t(v.nodes) * v.q;
        if (ctx->b.a != state) CK(cudaMemsetAsync(ctx->b.a, 0, bytes, ctx->stream));
        if (ctx->b.x != state) CK(cudaMemsetAsync(ctx->b.x, 0, bytes, ctx->stream));
        if (ctx->b.r != state) CK(cudaMemsetAsync(ctx->b.r, 0, bytes, ctx->stream));
        for (int d = 0; d < v.dimension; ++d)
            CK(cudaMemsetAsync(v.gradient[d], 0, bytes, ctx->stream));
    }
    const std::size_t shmem = std::size_t(ad->L1 + 7) * sizeof(double);
    k_adaptive_sensor<<<ad->num_tiles, 128, shmem, ctx->stream>>>(
        v, state, initial, ad->tb[0], ad->tb[1], ad->tb[2], ad->nt[0], ad->nt[1],
        ad->max_order, ad->d_deg_start, ad->d_tile_order, ad->d_tile_rho, ad->d_tile_E, ad->d_tile_face_E);
    CK(cudaMemcpyAsync(host_rho, ad->d_tile_rho, sizeof(double) * ad->num_tiles, cudaMemcpyDeviceToHost, ctx->stream));
    CK(cudaMemcpyAsync(host_E, ad->d_tile_E, sizeof(double) * std::size_t(ad->num_tiles) * ad->L1,
                       cudaMemcpyDeviceToHost, ctx->stream));
    CK(cudaMemcpyAsync(host_face_E, ad->d_tile_face_E, sizeof(double) * std::size_t(ad->num_tiles) * 6,
                       cudaMemcpyDeviceToHost, ctx->stream));
    CK(cudaStreamSynchronize(ctx->stream));
}

void adaptive_upload_and_zero(Context *ctx, double *state, const AdaptivePnPlan &plan,
                              const std::vector<int> &shrunken) {
    (void)state;
    AdaptiveDevice *ad = ctx->adaptive;
    const View &v = ctx->v;
    ad->global_max_q = plan.global_max_q;
    ad->bnd_max_q = plan.bnd_max_q;
    for (int s = 0; s < 6; ++s) {
        ad->side_q_dom[s] = plan.side_q_dom[s];
        ad->side_q_brk[s] = plan.side_q_brk[s];
    }
    CK(cudaMemcpyAsync(ad->d_tile_order, plan.tile_order.data(), sizeof(int) * ad->num_tiles,
                       cudaMemcpyHostToDevice, ctx->stream));
    CK(cudaMemcpyAsync(ad->d_tile_q, plan.tile_q.data(), sizeof(int) * ad->num_tiles,
                       cudaMemcpyHostToDevice, ctx->stream));
    CK(cudaMemcpyAsync(ad->d_tile_q_prev, plan.tile_q_prev.data(), sizeof(int) * ad->num_tiles,
                       cudaMemcpyHostToDevice, ctx->stream));
    if (!shrunken.empty()) {
        CK(cudaMemcpyAsync(ad->d_shrunken, shrunken.data(), sizeof(int) * shrunken.size(),
                           cudaMemcpyHostToDevice, ctx->stream));
        k_adaptive_zero<<<int(shrunken.size()), 128, 0, ctx->stream>>>(
            v, ctx->b.a, ctx->b.x, ctx->b.r, ad->tb[0], ad->tb[1], ad->tb[2], ad->nt[0], ad->nt[1],
            ad->d_shrunken, int(shrunken.size()), ad->d_tile_q, ad->d_tile_q_prev);
    }
    auto upload_vec = [&](auto *&dptr, int &cap, int &len, const auto &vec) {
        len = int(vec.size());
        if (len > cap) {
            cudaFree(dptr);
            cap = std::max(len * 2, 64);
            CK(cudaMalloc(&dptr, sizeof(vec[0]) * cap));
        }
        if (len > 0)
            CK(cudaMemcpyAsync(dptr, vec.data(), sizeof(vec[0]) * len, cudaMemcpyHostToDevice, ctx->stream));
    };
    for (int d = 0; d < v.dimension; ++d) {
        upload_vec(ad->d_segs_grad[d], ad->segs_grad_cap[d], ad->segs_grad_n[d], plan.line_segs_grad[d]);
        upload_vec(ad->d_segs_mass[d], ad->segs_mass_cap[d], ad->segs_mass_n[d], plan.line_segs_mass[d]);
    }
    upload_vec(ad->d_tile_chunks, ad->tile_chunks_cap, ad->tile_chunks_n, plan.tile_chunks);
    CK(cudaStreamSynchronize(ctx->stream)); // host vectors may be modified after return
}

void destroy(Context *ctx) {
    if (!ctx)
        return;
    if (ctx->stream)
        cudaStreamSynchronize(ctx->stream);
    destroy_adaptive(ctx);
    for (void *p : ctx->owned)
        cudaFree(p);
    if (ctx->reaction) {
        for (void *p : ctx->reaction->owned)
            cudaFree(p);
        cudaFree(ctx->reaction->r.table);
        delete ctx->reaction;
    }
    if (ctx->aux)
        cudaStreamDestroy(ctx->aux);
    for (int i = 0; i < 2; ++i) {
        if (ctx->side[i])
            cudaStreamDestroy(ctx->side[i]);
        if (ctx->join_event[i])
            cudaEventDestroy(ctx->join_event[i]);
    }
    if (ctx->fork_event)
        cudaEventDestroy(ctx->fork_event);
    destroy_boundary_high(ctx);
    destroy_ghost_traces(ctx);
    if (ctx->ghost) {
        if (ctx->ghost->handle)
            cublasDestroy(ctx->ghost->handle);
        for (void *p : ctx->ghost->owned)
            cudaFree(p);
        delete ctx->ghost;
    }
    for (auto &e : ctx->events)
        cudaEventDestroy(e);
    cudaFreeHost(ctx->pinned_counters);
    for (int k = 0; k < 8; ++k) {
        cudaFree(ctx->scratch_int[k]);
        cudaFree(ctx->scratch_double[k]);
    }
    if (ctx->stream)
        cudaStreamDestroy(ctx->stream);
    delete ctx;
}
Buffers buffers(Context *ctx) { return ctx->b; }
void upload(void *device, const void *host, std::size_t bytes) {
    CK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
}
void download(void *host, const void *device, std::size_t bytes) {
    CK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
}
void warm_up() { CK(cudaFree(nullptr)); }
void synchronize() { CK(cudaDeviceSynchronize()); }
void synchronize(Context *ctx) { CK(cudaStreamSynchronize(ctx->stream)); }
void *host_alloc(std::size_t bytes) {
    void *p = nullptr;
    CK(cudaMallocHost(&p, bytes ? bytes : 1));
    return p;
}
void host_free(void *p) { cudaFreeHost(p); }
int *device_ints(Context *ctx, int slot, int count) {
    if (std::size_t(count) > ctx->scratch_int_size[slot]) {
        cudaFree(ctx->scratch_int[slot]);
        ctx->scratch_int_size[slot] = std::size_t(count) * 2 + 16;
        CK(cudaMalloc(&ctx->scratch_int[slot], ctx->scratch_int_size[slot] * sizeof(int)));
    }
    return ctx->scratch_int[slot];
}
double *device_doubles(Context *ctx, int slot, std::size_t count) {
    if (count > ctx->scratch_double_size[slot]) {
        cudaFree(ctx->scratch_double[slot]);
        ctx->scratch_double_size[slot] = count * 2 + 16;
        CK(cudaMalloc(&ctx->scratch_double[slot], ctx->scratch_double_size[slot] * sizeof(double)));
    }
    return ctx->scratch_double[slot];
}
void closure(Context *ctx, const double *state) { k_closure<<<blocks(ctx->v.nodes), 256, 0, ctx->stream>>>(ctx->v, state); }
void weno_gamma(Context *ctx, const double *state) {
    const View &v = ctx->v;
    if (v.local == 4)
        k_weno<4><<<blocks(v.cells, 128), 128, 0, ctx->stream>>>(v, state);
    else if (v.local == 8)
        k_weno<8><<<blocks(v.cells, 128), 128, 0, ctx->stream>>>(v, state);
    else if (v.local == 9)
        k_weno<9><<<blocks(v.cells, 128), 128, 0, ctx->stream>>>(v, state);
    else
        k_weno<0><<<blocks(v.cells, 128), 128, 0, ctx->stream>>>(v, state);
}
namespace {
constexpr int LINE_CHUNK = 16;
void line_solve(Context *ctx, int d, int count, double *f, const double *src = nullptr) {
    const View &v = ctx->v;
    std::size_t work = 0;
    for (int k = d; k < d + count; ++k)
        work = std::max(work, std::size_t(v.nodes / v.npa[k]) * v.q);
    const dim3 grid(blocks(work, 64), count);
    if (v.periodic) {
        switch (v.degree) {
        case 1: k_line_solve_cyclic<1><<<grid, 64, 0, ctx->stream>>>(v, d, f, src); break;
        case 2: k_line_solve_cyclic<2><<<grid, 64, 0, ctx->stream>>>(v, d, f, src); break;
        case 3: k_line_solve_cyclic<3><<<grid, 64, 0, ctx->stream>>>(v, d, f, src); break;
        default: k_line_solve_cyclic<4><<<grid, 64, 0, ctx->stream>>>(v, d, f, src); break;
        }
        return;
    }
    // Experiment: chunk (prefetch depth) and block size of the Q1 line solve (M1PN_GPU_LINE_CHUNK 8|16|32,
    // M1PN_GPU_LINE_BLOCK 32..256); the arithmetic per thread is unchanged.
    static const int chunk = [] { const char *e = std::getenv("M1PN_GPU_LINE_CHUNK"); return e ? std::atoi(e) : 16; }();
    static const int lb = [] { const char *e = std::getenv("M1PN_GPU_LINE_BLOCK"); return e ? std::atoi(e) : 64; }();
    if (v.degree == 1 && (chunk != 16 || lb != 64)) {
        const dim3 g2(blocks(work, lb), count);
        if (chunk == 8)
            k_line_solve<1, 8><<<g2, lb, 0, ctx->stream>>>(v, d, f, src);
        else if (chunk == 32)
            k_line_solve<1, 32><<<g2, lb, 0, ctx->stream>>>(v, d, f, src);
        else
            k_line_solve<1, LINE_CHUNK><<<g2, lb, 0, ctx->stream>>>(v, d, f, src);
        return;
    }
    switch (v.degree) {
    case 1: k_line_solve<1, LINE_CHUNK><<<grid, 64, 0, ctx->stream>>>(v, d, f, src); break;
    case 2: k_line_solve<2, LINE_CHUNK><<<grid, 64, 0, ctx->stream>>>(v, d, f, src); break;
    case 3: k_line_solve<3, LINE_CHUNK><<<grid, 64, 0, ctx->stream>>>(v, d, f, src); break;
    default: k_line_solve<4, LINE_CHUNK><<<grid, 64, 0, ctx->stream>>>(v, d, f, src); break;
    }
}
void line_solve_low4(Context *ctx, int d, int count, double *f, const double *src = nullptr,
                     cudaStream_t st = nullptr) {
    const View &v = ctx->v;
    if (!st)
        st = ctx->stream;
    const int q4 = std::min(4, v.q);
    std::size_t work = 0;
    for (int k = d; k < d + count; ++k)
        work = std::max(work, std::size_t(v.nodes / v.npa[k]) * q4);
    const dim3 grid(blocks(work, 64), count);
    switch (v.degree) {
    case 1: k_line_solve_low4<1, LINE_CHUNK><<<grid, 64, 0, st>>>(v, d, f, src); break;
    case 2: k_line_solve_low4<2, LINE_CHUNK><<<grid, 64, 0, st>>>(v, d, f, src); break;
    case 3: k_line_solve_low4<3, LINE_CHUNK><<<grid, 64, 0, st>>>(v, d, f, src); break;
    default: k_line_solve_low4<4, LINE_CHUNK><<<grid, 64, 0, st>>>(v, d, f, src); break;
    }
}
void line_solve_adaptive_axis(Context *ctx, int d, double *f, const double *src,
                              const AdaptiveLineSeg *segs, int nsegs, cudaStream_t st = nullptr) {
    if (nsegs <= 0)
        return;
    if (!st)
        st = ctx->stream;
    const View &v = ctx->v;
    const AdaptiveDevice *ad = ctx->adaptive;
    const unsigned grid = (unsigned(nsegs) + 1u) / 2u; // 2 warps = 64 threads per block (register-optimal)
    switch (v.degree) {
    case 1:
        k_line_solve_adaptive<1, LINE_CHUNK><<<grid, 64, 0, st>>>(
            v, d, f, src, segs, nsegs, ad->d_seg_slot[d], ad->d_seg_off[d], ad->d_seg_lower[d], ad->d_seg_invdiag[d],
            ad->d_dseg_slot[d], ad->d_dseg_off[d], ad->d_dseg_lower[d], ad->d_dseg_invdiag[d]);
        break;
    case 2:
        k_line_solve_adaptive<2, LINE_CHUNK><<<grid, 64, 0, st>>>(
            v, d, f, src, segs, nsegs, ad->d_seg_slot[d], ad->d_seg_off[d], ad->d_seg_lower[d], ad->d_seg_invdiag[d],
            ad->d_dseg_slot[d], ad->d_dseg_off[d], ad->d_dseg_lower[d], ad->d_dseg_invdiag[d]);
        break;
    case 3:
        k_line_solve_adaptive<3, LINE_CHUNK><<<grid, 64, 0, st>>>(
            v, d, f, src, segs, nsegs, ad->d_seg_slot[d], ad->d_seg_off[d], ad->d_seg_lower[d], ad->d_seg_invdiag[d],
            ad->d_dseg_slot[d], ad->d_dseg_off[d], ad->d_dseg_lower[d], ad->d_dseg_invdiag[d]);
        break;
    default:
        k_line_solve_adaptive<4, LINE_CHUNK><<<grid, 64, 0, st>>>(
            v, d, f, src, segs, nsegs, ad->d_seg_slot[d], ad->d_seg_off[d], ad->d_seg_lower[d], ad->d_seg_invdiag[d],
            ad->d_dseg_slot[d], ad->d_dseg_off[d], ad->d_dseg_lower[d], ad->d_dseg_invdiag[d]);
        break;
    }
}
// Minimum resident blocks of the 2D strip stabilization kernels (__launch_bounds__): 4 (default, ~120 registers),
// 6 or 8 trade registers for occupancy (M1PN_GPU_STRIP_MINB; measurement on the A100).
int strip_min_blocks() {
    static const int mb = [] {
        const char *e = std::getenv("M1PN_GPU_STRIP_MINB");
        return e ? std::atoi(e) : 4;
    }();
    return mb;
}
bool overlap_enabled() {
    static const bool on = [] {
        const char *e = std::getenv("M1PN_GPU_OVERLAP");
        return e ? std::atoi(e) != 0 : true;
    }();
    return on;
}
// Side streams start after the work queued on ctx->stream; join makes ctx->stream wait for them.
void fork_streams(Context *ctx, int n) {
    if (!ctx->fork_event) {
        CK(cudaEventCreateWithFlags(&ctx->fork_event, cudaEventDisableTiming));
        for (int i = 0; i < 2; ++i) {
            CK(cudaStreamCreateWithFlags(&ctx->side[i], cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&ctx->join_event[i], cudaEventDisableTiming));
        }
    }
    CK(cudaEventRecord(ctx->fork_event, ctx->stream));
    for (int i = 0; i < n; ++i)
        CK(cudaStreamWaitEvent(ctx->side[i], ctx->fork_event, 0));
}
void join_streams(Context *ctx, int n) {
    for (int i = 0; i < n; ++i) {
        CK(cudaEventRecord(ctx->join_event[i], ctx->side[i]));
        CK(cudaStreamWaitEvent(ctx->stream, ctx->join_event[i], 0));
    }
}
} // namespace
void gradients(Context *ctx, const double *state) {
    const View &v = ctx->v;
    if (ctx->adaptive && ctx->adaptive->active && !v.periodic) {
        AdaptiveDevice *ad = ctx->adaptive;
        if (overlap_enabled() && v.dimension <= 3) {
            // low4 (all axes) on side 0; axis 0 on ctx->stream, axis 1 on side 1, axis 2 on ctx->stream after axis 0:
            // each axis writes its own gradient field, low4 only the coefficients c < 4 of all of them.
            fork_streams(ctx, 2);
            line_solve_low4(ctx, 0, v.dimension, nullptr, state, ctx->side[0]);
            for (int d = 0; d < v.dimension; ++d)
                line_solve_adaptive_axis(ctx, d, nullptr, state, ad->d_segs_grad[d], ad->segs_grad_n[d],
                                         d == 1 ? ctx->side[1] : ctx->stream);
            join_streams(ctx, 2);
            return;
        }
        line_solve_low4(ctx, 0, v.dimension, nullptr, state);
        for (int d = 0; d < v.dimension; ++d)
            line_solve_adaptive_axis(ctx, d, nullptr, state, ad->d_segs_grad[d], ad->segs_grad_n[d]);
        return;
    }
    line_solve(ctx, 0, v.dimension, nullptr, state);
}
// 3D Q1 node strip kernel on graded / broken meshes instead of the per-node connectivity kernel. On the uniform mesh
// the element kernel + gather stays faster on the A100 (U112 P7 solver 58.8 vs 67.4 s; graded G3H 123.4 -> 91.1 s,
// broken 177.4 -> 148.6 s). M1PN_GPU_STRIP3=0: never, 2: also on uniform meshes.
static bool strip3_enabled(const View &v) {
    static const int mode = [] {
        const char *e = std::getenv("M1PN_GPU_STRIP3");
        return e ? std::atoi(e) : 1;
    }();
    return mode == 2 || (mode == 1 && (v.klass || !v.structured));
}
void stabilization(Context *ctx, const double *state, double *out, bool want_low, double constant_gamma) {
    const View &v = ctx->v;
    if (ctx->adaptive && ctx->adaptive->active && constant_gamma < 0 && !v.periodic) {
        AdaptiveDevice *ad = ctx->adaptive;
        const bool use_strip = (v.dimension == 2 && v.degree == 1 && v.classes <= max_strip_classes &&
                                (v.structured || (v.strip_elem[0] && v.strip_elem[1])));
        if (use_strip) {
            const int q4 = std::min(4, v.q);
            const dim3 g4(blocks(std::size_t(v.npa[0]) * q4, 128), (v.npa[1] + 15) / 16);
            const bool ov = overlap_enabled() && ad->tile_chunks_n > 0; // low4 writes c < 4, the chunks c >= 4
            if (ov)
                fork_streams(ctx, 1);
            k_stabilization_strip_low4<16><<<g4, 128, 0, ov ? ctx->side[0] : ctx->stream>>>(v, state, out, want_low,
                                                                                            constant_gamma);
            if (ad->tile_chunks_n > 0) {
                const int mb = strip_min_blocks();
                if (mb == 8)
                    k_stabilization_strip_adaptive<8><<<ad->tile_chunks_n, 128, 0, ctx->stream>>>(
                        v, state, out, false, constant_gamma, ad->tb[0], ad->tb[1], ad->d_tile_chunks,
                        ad->tile_chunks_n, ad->d_tile_q, ad->nt[0]);
                else if (mb == 6)
                    k_stabilization_strip_adaptive<6><<<ad->tile_chunks_n, 128, 0, ctx->stream>>>(
                        v, state, out, false, constant_gamma, ad->tb[0], ad->tb[1], ad->d_tile_chunks,
                        ad->tile_chunks_n, ad->d_tile_q, ad->nt[0]);
                else
                    k_stabilization_strip_adaptive<4><<<ad->tile_chunks_n, 128, 0, ctx->stream>>>(
                        v, state, out, false, constant_gamma, ad->tb[0], ad->tb[1], ad->d_tile_chunks,
                        ad->tile_chunks_n, ad->d_tile_q, ad->nt[0]);
            }
            if (ov)
                join_streams(ctx, 1);
        } else {
            const int q4 = std::min(4, v.q);
            k_stabilization_low4<<<blocks(std::size_t(v.nodes) * q4), 256, 0, ctx->stream>>>(
                v, state, out, want_low, constant_gamma);
            if (ad->tile_chunks_n > 0)
                k_stabilization_adaptive<<<ad->tile_chunks_n, 128, 0, ctx->stream>>>(
                    v, state, out, false, constant_gamma, ad->tb[0], ad->tb[1], ad->tb[2],
                    ad->d_tile_chunks, ad->tile_chunks_n, ad->d_tile_q, ad->nt[0], ad->nt[1]);
        }
        return;
    }
    if (v.dimension == 2 && v.degree == 1 && !v.periodic && v.classes <= max_strip_classes &&
        (v.structured || (v.strip_elem[0] && v.strip_elem[1]))) {
        static const int sy = [] {
            const char *e = std::getenv("M1PN_GPU_STRIP");
            return e ? std::atoi(e) : 16;
        }();
        const dim3 grid(blocks(std::size_t(v.npa[0]) * v.q, 128), (v.npa[1] + sy - 1) / sy);
        if (sy == 8)
            k_stabilization_strip<8><<<grid, 128, 0, ctx->stream>>>(v, state, out, want_low, constant_gamma);
        else if (sy == 16 && strip_min_blocks() == 8)
            k_stabilization_strip<16, 8><<<grid, 128, 0, ctx->stream>>>(v, state, out, want_low, constant_gamma);
        else if (sy == 16 && strip_min_blocks() == 6)
            k_stabilization_strip<16, 6><<<grid, 128, 0, ctx->stream>>>(v, state, out, want_low, constant_gamma);
        else if (sy == 16)
            k_stabilization_strip<16><<<grid, 128, 0, ctx->stream>>>(v, state, out, want_low, constant_gamma);
        else if (sy == 64)
            k_stabilization_strip<64><<<grid, 128, 0, ctx->stream>>>(v, state, out, want_low, constant_gamma);
        else
            k_stabilization_strip<32><<<grid, 128, 0, ctx->stream>>>(v, state, out, want_low, constant_gamma);
    } else if (v.dimension == 3 && v.degree == 1 && !v.periodic && v.local == 8 && v.strip3_ok && strip3_enabled(v) &&
               (v.structured || (v.strip_elem[0] && v.strip_elem[1] && v.strip_elem[2]))) {
        static const int sz = [] {
            const char *e = std::getenv("M1PN_GPU_STRIP3_SZ");
            return e ? std::atoi(e) : 16;
        }();
        const dim3 grid(blocks(std::size_t(v.npa[0]) * v.q, 128), v.npa[1], (v.npa[2] + sz - 1) / sz);
        if (sz == 8)
            k_stabilization_strip3<8><<<grid, 128, 0, ctx->stream>>>(v, state, out, want_low, constant_gamma);
        else if (sz == 32)
            k_stabilization_strip3<32><<<grid, 128, 0, ctx->stream>>>(v, state, out, want_low, constant_gamma);
        else
            k_stabilization_strip3<16><<<grid, 128, 0, ctx->stream>>>(v, state, out, want_low, constant_gamma);
    } else if (v.klass || !v.structured) { // graded / broken: per node (connectivity) instead of the element kernels
        k_stabilization<<<blocks(std::size_t(v.nodes) * v.q), 256, 0, ctx->stream>>>(v, state, out, want_low, constant_gamma);
    } else {
        double *local_out = device_doubles(ctx, 7, std::size_t(v.cells) * v.local * v.q);
        const unsigned g = blocks(std::size_t(v.cells) * v.q, 128);
        const bool d3 = v.dimension == 3;
        if (v.local == 9 && !d3)
            k_stab_element<9, 2><<<g, 128, 0, ctx->stream>>>(v, state, local_out, want_low, constant_gamma);
        else if (v.local == 16 && !d3)
            k_stab_element<16, 2><<<g, 128, 0, ctx->stream>>>(v, state, local_out, want_low, constant_gamma);
        else if (v.local == 25 && !d3)
            k_stab_element<25, 2><<<g, 128, 0, ctx->stream>>>(v, state, local_out, want_low, constant_gamma);
        else if (v.local == 8 && d3)
            k_stab_element<8, 3><<<g, 128, 0, ctx->stream>>>(v, state, local_out, want_low, constant_gamma);
        else if (v.local == 27 && d3)
            k_stab_element<27, 3><<<g, 128, 0, ctx->stream>>>(v, state, local_out, want_low, constant_gamma);
        else {
            k_stabilization<<<blocks(std::size_t(v.nodes) * v.q), 256, 0, ctx->stream>>>(v, state, out, want_low, constant_gamma);
            return;
        }
        k_gather_stab<<<blocks(std::size_t(v.nodes) * v.q), 256, 0, ctx->stream>>>(v, local_out, out);
    }
}
void add_rows(Context *ctx, double *field, const int *index, const double *rows, int count) {
    if (count)
        k_add_rows<<<blocks(std::size_t(count) * ctx->v.q), 256, 0, ctx->stream>>>(ctx->v.q, field, index, rows, count);
}
void mass_solve(Context *ctx, double *field) {
    const View &v = ctx->v;
    if (ctx->adaptive && ctx->adaptive->active && !v.periodic) {
        AdaptiveDevice *ad = ctx->adaptive;
        for (int d = 0; d < v.dimension; ++d) {
            const bool ov = overlap_enabled(); // axis d: low4 (c < 4) and chunks (c >= 4) of the same field
            if (ov)
                fork_streams(ctx, 1);
            line_solve_low4(ctx, d, 1, field, nullptr, ov ? ctx->side[0] : ctx->stream);
            line_solve_adaptive_axis(ctx, d, field, nullptr, ad->d_segs_mass[d], ad->segs_mass_n[d]);
            if (ov)
                join_streams(ctx, 1);
        }
        return;
    }
    for (int d = 0; d < v.dimension; ++d)
        line_solve(ctx, d, 1, field);
}
bool mass_solve_combine(Context *ctx, double *field, const double *x, const double *a, double old, double fresh,
                        double tau) {
    const View &v = ctx->v;
    const int last = v.dimension - 1;
    // Default off: bitwise equal, but slower than mass_solve + combine_pn (P35 GB: 29.5 s against 23.0 s, job 162668;
    // the fused solve reads the flux table and the gradient rows scattered). M1PN_GPU_FUSE=1 enables it.
    static const bool off = [] {
        const char *e = std::getenv("M1PN_GPU_FUSE");
        return !(e && std::atoi(e) == 1);
    }();
    if (off || v.periodic || v.mirror_index[last] >= 0 || v.degree != 1 || (ctx->adaptive && ctx->adaptive->active)) {
        mass_solve(ctx, field);
        return false; // the caller runs combine_pn
    }
    for (int d = 0; d < last; ++d)
        line_solve(ctx, d, 1, field);
    static const int chunk = [] { const char *e = std::getenv("M1PN_GPU_LINE_CHUNK"); return e ? std::atoi(e) : 16; }();
    static const int lb = [] { const char *e = std::getenv("M1PN_GPU_LINE_BLOCK"); return e ? std::atoi(e) : 64; }();
    const dim3 grid(blocks(std::size_t(v.nodes / v.npa[last]) * v.q, lb), 1);
    if (chunk == 8)
        k_line_solve<1, 8, true><<<grid, lb, 0, ctx->stream>>>(v, last, field, nullptr, x, a, old, fresh, tau);
    else if (chunk == 32)
        k_line_solve<1, 32, true><<<grid, lb, 0, ctx->stream>>>(v, last, field, nullptr, x, a, old, fresh, tau);
    else
        k_line_solve<1, LINE_CHUNK, true><<<grid, lb, 0, ctx->stream>>>(v, last, field, nullptr, x, a, old, fresh, tau);
    CK(cudaGetLastError());
    return true;
}
namespace {
std::size_t flux_shared(Context *ctx) {
    if (!ctx->v.flux_in_shared)
        return 0;
    return std::size_t(ctx->v.dimension) * ctx->v.q * ctx->v.flux_width * (sizeof(double) + sizeof(int));
}
} // namespace
void flux_moments(Context *ctx, const double *field) {
    const View &v = ctx->v;
    const AdaptiveDevice *ad = ctx->adaptive && ctx->adaptive->active && !v.periodic ? ctx->adaptive : nullptr;
    k_flux_rows<<<blocks(std::size_t(v.nodes) * 4), 256, flux_shared(ctx), ctx->stream>>>(
        v, const_cast<double *>(field), true, true, ad ? ad->d_tile_q : nullptr, ad ? ad->tb[0] : 1, ad ? ad->tb[1] : 1,
        ad ? ad->tb[2] : 1, ad ? ad->nt[0] : 1, ad ? ad->nt[1] : 1);
}
void flux_rows(Context *ctx, double *field, bool want_moments) {
    const View &v = ctx->v;
    const AdaptiveDevice *ad = ctx->adaptive && ctx->adaptive->active && !v.periodic ? ctx->adaptive : nullptr;
    k_flux_rows<<<blocks(std::size_t(v.nodes) * v.q), 256, flux_shared(ctx), ctx->stream>>>(
        v, field, want_moments, false, ad ? ad->d_tile_q : nullptr, ad ? ad->tb[0] : 1, ad ? ad->tb[1] : 1,
        ad ? ad->tb[2] : 1, ad ? ad->nt[0] : 1, ad ? ad->nt[1] : 1);
}
void limiter(Context *ctx) {
    const View &v = ctx->v;
    const unsigned b4 = blocks(std::size_t(v.cells) * 4, 256), b8 = blocks(std::size_t(v.cells) * 8, 256),
                   b16 = blocks(std::size_t(v.cells) * 16, 256), b32 = blocks(std::size_t(v.cells) * 32, 256);
    if (v.local == 4) {
        k_limiter_group<4, 4><<<b4, 256, 0, ctx->stream>>>(v);
        k_limiter_segment<4><<<blocks(v.cells, 64), 64, 0, ctx->stream>>>(v);
    } else if (v.local == 8) {
        k_limiter_group<8, 8><<<b8, 256, 0, ctx->stream>>>(v);
        k_limiter_segment<8><<<blocks(v.cells, 64), 64, 0, ctx->stream>>>(v);
    } else if (v.local == 9) {
        k_limiter_group<9, 16><<<b16, 256, 0, ctx->stream>>>(v);
        k_limiter_segment<9><<<blocks(v.cells, 64), 64, 0, ctx->stream>>>(v);
    } else if (v.local == 16) {
        k_limiter_group<16, 16><<<b16, 256, 0, ctx->stream>>>(v);
        k_limiter_segment<16><<<blocks(v.cells, 64), 64, 0, ctx->stream>>>(v);
    } else if (v.local == 25) {
        k_limiter_group<25, 32><<<b32, 256, 0, ctx->stream>>>(v);
        k_limiter_segment<25><<<blocks(v.cells, 64), 64, 0, ctx->stream>>>(v);
    } else if (v.local == 27) {
        k_limiter_group<27, 32><<<b32, 256, 0, ctx->stream>>>(v);
        k_limiter_segment<27><<<blocks(v.cells, 64), 64, 0, ctx->stream>>>(v);
    } else {
        k_limiter<0><<<blocks(v.cells, 128), 128, 0, ctx->stream>>>(v);
        k_limiter_segment<0><<<blocks(v.cells, 64), 64, 0, ctx->stream>>>(v);
    }
}
void gather_load(Context *ctx) { k_gather_load<<<blocks(std::size_t(ctx->v.nodes) * 4), 256, 0, ctx->stream>>>(ctx->v); }
void combine_pn(Context *ctx, double *r, const double *x, const double *a, double old, double fresh, double tau) {
    const View &v = ctx->v;
    if (ctx->adaptive && ctx->adaptive->active && !v.periodic) {
        AdaptiveDevice *ad = ctx->adaptive;
        if (ad->tile_chunks_n > 0)
            k_combine_adaptive<<<ad->tile_chunks_n, 256, flux_shared(ctx), ctx->stream>>>(
                v, r, x, a, old, fresh, tau, ad->tb[0], ad->tb[1], ad->tb[2], ad->d_tile_chunks, ad->tile_chunks_n);
        return;
    }
    k_combine<<<blocks(std::size_t(v.nodes) * v.q), 256, flux_shared(ctx), ctx->stream>>>(v, r, x, a, old, fresh, tau, true, 1);
}
void finish_stage(Context *ctx, double *r, const double *x, const double *a, double old, double fresh, double tau,
                  bool pn_done, bool closure_next) {
    const View &v = ctx->v;
    if (pn_done)
        k_combine<<<blocks(std::size_t(v.nodes) * 4), 256, 0, ctx->stream>>>(v, r, x, a, old, fresh, tau, false, 2);
    else
        k_combine<<<blocks(std::size_t(v.nodes) * v.q), 256, flux_shared(ctx), ctx->stream>>>(v, r, x, a, old, fresh, tau, true, 0);
    k_admissible<<<blocks(v.nodes), 256, 0, ctx->stream>>>(v, r, closure_next, false, x, a, old, fresh, tau);
}
double norm(Context *ctx, const double *field) {
    const std::size_t count = std::size_t(ctx->v.nodes) * ctx->v.q;
    const unsigned grid = std::max(1u, std::min<unsigned>(blocks(count), 1024));
    k_sumsq<<<grid, 256, 0, ctx->stream>>>(field, count, ctx->norm_accumulator);
    double partial[1024];
    CK(cudaMemcpy(partial, ctx->norm_accumulator, grid * sizeof(double), cudaMemcpyDeviceToHost));
    double s = 0;
    for (unsigned b = 0; b < grid; ++b)
        s += partial[b];
    return std::sqrt(s);
}
void divide(Context *ctx, double *dst, const double *src, double divisor) {
    const std::size_t count = std::size_t(ctx->v.nodes) * ctx->v.q;
    k_divide<<<blocks(count), 256, 0, ctx->stream>>>(dst, src, count, divisor);
}
namespace {
Counters convert(const DeviceCounters &d);
}
Counters counters(Context *ctx) {
    CK(cudaGetLastError()); // launch failures (e.g. resource limits) since the last call
    DeviceCounters d;
    CK(cudaMemcpyAsync(&d, ctx->v.counters, sizeof d, cudaMemcpyDeviceToHost, ctx->stream)); // context stream only
    CK(cudaStreamSynchronize(ctx->stream));
    return convert(d);
}
namespace {
Counters convert(const DeviceCounters &d) {
    Counters c;
    c.error = d.error;
    c.closure_fallbacks = d.closure_fallbacks;
    c.limiter_fallbacks = d.limiter_fallbacks;
    c.limited_elements = (long long)d.limited_elements;
    c.nudged_bars = (long long)d.nudged_bars;
    c.outward_bars = (long long)d.outward_bars;
    c.nudged_rows = (long long)d.nudged_rows;
    auto as_double = [](unsigned long long b) {
        double v;
        std::memcpy(&v, &b, 8);
        return v;
    };
    c.min_limiter = as_double(d.min_limiter_bits);
    c.max_raw_zero_sum = as_double(d.max_raw_bits);
    c.max_limited_zero_sum = as_double(d.max_limited_bits);
    c.nudge_mass = d.nudge_mass;
    c.min_gamma = as_double(d.min_gamma_bits);
    return c;
}
} // namespace
void snapshot_counters(Context *ctx) {
    CK(cudaMemcpyAsync(ctx->pinned_counters, ctx->v.counters, sizeof(DeviceCounters), cudaMemcpyDeviceToHost, ctx->stream));
}
Counters snapshot(Context *ctx) {
    CK(cudaGetLastError());
    return convert(*ctx->pinned_counters);
}
void reset_fallbacks(Context *ctx) {
    CK(cudaMemsetAsync(&ctx->v.counters->closure_fallbacks, 0, 3 * sizeof(int), ctx->stream));
}
void record(Context *ctx, int event) { CK(cudaEventRecord(ctx->events[event], ctx->stream)); }
void wait(Context *ctx, int event) { CK(cudaEventSynchronize(ctx->events[event])); }
void upload_async(void *device, const void *pinned, std::size_t bytes) {
    if (bytes)
        CK(cudaMemcpyAsync(device, pinned, bytes, cudaMemcpyHostToDevice, 0));
}
void gather_async(Context *ctx, const double *field, int cols, const int *device_index, int count, double *pinned_out,
                  int slot, long long rows_soa) {
    if (!count)
        return;
    double *buf = device_doubles(ctx, slot, std::size_t(count) * cols);
    k_gather<<<blocks(std::size_t(count) * cols), 256, 0, ctx->stream>>>(field, cols, device_index, count, buf, rows_soa);
    CK(cudaMemcpyAsync(pinned_out, buf, std::size_t(count) * cols * sizeof(double), cudaMemcpyDeviceToHost, ctx->stream));
}
void gather_cols_async(Context *ctx, const double *field, int stride_cols, int copy_cols, const int *device_index,
                       int count, double *pinned_out, int slot) {
    if (!count || copy_cols <= 0)
        return;
    const std::size_t n = std::size_t(count) * copy_cols;
    double *buf = device_doubles(ctx, slot, n);
    k_gather_cols<<<blocks(n), 256, 0, ctx->stream>>>(field, stride_cols, copy_cols, device_index, count, buf);
    CK(cudaMemcpyAsync(pinned_out, buf, n * sizeof(double), cudaMemcpyDeviceToHost, ctx->stream));
}
void scatter_async(Context *ctx, double *field, int cols, const int *device_index, int count, const double *pinned_in,
                   bool add, int slot, long long rows_soa) {
    if (!count)
        return;
    double *buf = device_doubles(ctx, slot, std::size_t(count) * cols);
    CK(cudaMemcpyAsync(buf, pinned_in, std::size_t(count) * cols * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
    k_scatter<<<blocks(std::size_t(count) * cols), 256, 0, ctx->stream>>>(field, cols, device_index, count, buf, add, rows_soa);
}
void reset_counters(Context *ctx) {
    DeviceCounters d{};
    const double one = 1;
    std::memcpy(&d.min_limiter_bits, &one, 8);
    std::memcpy(&d.min_gamma_bits, &one, 8);
    CK(cudaMemcpyAsync(ctx->v.counters, &d, sizeof d, cudaMemcpyHostToDevice, ctx->stream));
    CK(cudaStreamSynchronize(ctx->stream));
}
const int *closure_fallback_list(Context *ctx) { return ctx->v.closure_list; }
const int *limiter_fallback_list(Context *ctx) { return ctx->v.limiter_list; }
// Gathers/scatters through a device index list (slot 0 for indices, slot 0/1 for rows).
void gather_device(Context *ctx, const double *field, int cols, const int *device_index, int count, double *host_out,
                   long long rows_soa) {
    if (!count)
        return;
    double *buf = device_doubles(ctx, 0, std::size_t(count) * cols);
    k_gather<<<blocks(std::size_t(count) * cols), 256, 0, ctx->stream>>>(field, cols, device_index, count, buf, rows_soa);
    CK(cudaMemcpyAsync(host_out, buf, std::size_t(count) * cols * sizeof(double), cudaMemcpyDeviceToHost, ctx->stream));
    CK(cudaStreamSynchronize(ctx->stream));
}
void scatter_device(Context *ctx, double *field, int cols, const int *device_index, int count, const double *host_in,
                    bool add, long long rows_soa) {
    if (!count)
        return;
    double *buf = device_doubles(ctx, 1, std::size_t(count) * cols);
    CK(cudaMemcpyAsync(buf, host_in, std::size_t(count) * cols * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
    CK(cudaStreamSynchronize(ctx->stream)); // host_in may be pageable and reused by the caller
    k_scatter<<<blocks(std::size_t(count) * cols), 256, 0, ctx->stream>>>(field, cols, device_index, count, buf, add, rows_soa);
}
} // namespace gpu
} // namespace m1pn
