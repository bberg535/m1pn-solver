// GPU kernels of the element WENO time loop (experimental, docs/ELEMENT_WENO_GPU.md). One thread per output entry or
// per element; every output is computed by one thread in a fixed order, so a run is deterministic. The arithmetic
// follows src/element_weno.cpp; summation orders and fused multiply-adds differ (rounding level). The exact M1 cone
// predicate is evaluated exactly (error-free transformations); paths that need long double on the host (near-beam
// Levermore states, the stable base bar) are flagged and recomputed by the host with the original code.
#include "element_weno_gpu_device.hpp"
#include <cuda_runtime.h>
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
    double weno_q, weno_r, weno_epsilon, weno_linear, positivity_fraction, element_theta;
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
    double *seg_ubar, *seg_lim; // cells x 4 local: bar states and clipped contributions of queued elements
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
    struct ReactionDevice *reaction = nullptr;
    struct BoundaryHighDevice *bhigh = nullptr;
    int flux_nnz[3] = {0, 0, 0};
    DeviceCounters *pinned_counters = nullptr;
};

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
            const double *X = v.extension[d][side];
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
template <int P, int CH> __global__ void k_line_solve(View v, int first, double *field, const double *src) {
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
        const double *K = v.stiffness_rows + std::size_t(l) * n;
        double term = 0;
        for (int m = 0; m < n; ++m)
            term += (v.viscosity * K[m]) * a[std::size_t(map[m]) * q + c];
        const double g = constant_gamma >= 0 ? constant_gamma : v.gamma[e];
        const double weight = -v.viscosity * g;
        if (weight != 0)
            for (int d = 0; d < v.dimension; ++d) {
                const double *Ct = v.convection_t_rows[d] + std::size_t(l) * n;
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
__constant__ double cK[16], cKv[16], cC0[16], cC1[16]; // 2D Q1 element tables (stiffness rows, C_d^T rows)
// 2D Q1: the gathered residual of node i only involves its 3 x 3 neighbourhood. One thread per (x, component) walks a
// strip of SY rows in y with a sliding 3 x 3 window of a, g_0, g_1 in registers (9 loads per output); the four elements
// around a node are visited as quadrants with compile-time local indices.
template <int SY>
__global__ void __launch_bounds__(128, 4) k_stabilization_strip(View v, const double *a, double *result, bool want_low, double constant_gamma) {
    const int q = v.q, npx = v.npa[0], npy = v.npa[1], cx = v.count[0], cy = v.count[1];
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= npx * q)
        return;
    const int ix = col / q, c = col % q;
    const int y0 = blockIdx.y * SY, y1 = min(npy, y0 + SY);
    const double *F[3] = {a, v.gradient[0], v.gradient[1]};
    double W[3][3][3];
    auto load_row = [&](int y, int r) {
#pragma unroll
        for (int f = 0; f < 3; ++f)
#pragma unroll
            for (int dx = 0; dx < 3; ++dx) {
                const int x = ix + dx - 1;
                W[f][r][dx] = y >= 0 && y < npy && x >= 0 && x < npx ? F[f][(std::size_t(y) * npx + x) * q + c] : 0.;
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
                const int ex = ix - 1 + qx, ey = y - 1 + qy;
                if (ex < 0 || ex >= cx || ey < 0 || ey >= cy)
                    continue;
                const int e = ex + ey * cx, l = (1 - qx) + 2 * (1 - qy);
                const double g = constant_gamma >= 0 ? constant_gamma : v.gamma[e];
                const double weight = -v.viscosity * g;
                // nu K a + weight (C_0^T g_0 + C_1^T g_1), nu K prescaled (cKv)
                double stiff = 0, conv = 0;
#pragma unroll
                for (int my = 0; my < 2; ++my)
#pragma unroll
                    for (int mx = 0; mx < 2; ++mx) {
                        const int m = mx + 2 * my;
                        stiff += cKv[4 * l + m] * W[0][qy + my][qx + mx];
                        conv += cC0[4 * l + m] * W[1][qy + my][qx + mx] + cC1[4 * l + m] * W[2][qy + my][qx + mx];
                    }
                const double term = stiff + weight * conv;
                if (want_low && c < 4) {
                    const int slot = c == 0 ? 0 : (c == 1 ? 2 : (c == 2 ? 3 : 1));
                    v.stab_low[std::size_t(4 * l + slot) * v.cells + e] = (c == 0 ? v.s0 : v.s1) * term;
                }
                out -= term;
            }
        result[(std::size_t(y) * npx + ix) * q + c] = out;
#pragma unroll
        for (int f = 0; f < 3; ++f)
#pragma unroll
            for (int dx = 0; dx < 3; ++dx) {
                W[f][0][dx] = W[f][1][dx];
                W[f][1][dx] = W[f][2][dx];
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
};
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
__global__ void k_flux_rows(View v, double *field, bool want_moments, bool moments_only) {
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
    for (int d = 0; d < v.dimension; ++d) {
        const double *g = v.gradient[d] + std::size_t(i) * q;
        value += -F.row(g, d, j);
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
            const int pk = i * (n - 1) + k;
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
            ubar[i][r] = weighted[r] / v.total_viscosity[i];
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
        double pos = 0, neg = 0, clipped[NA];
#pragma unroll
        for (int i = 0; i < n; ++i) {
            const double fmin_ = -2 * v.total_viscosity[i] * (1 - v.positivity_fraction) * ubar[i][0];
            clipped[i] = fmax(contribution[i][0], fmin_);
            (clipped[i] > 0 ? pos : neg) += clipped[i];
        }
#pragma unroll
        for (int i = 0; i < n; ++i) {
            double value = clipped[i];
            if (value > 0 && pos + neg > 0)
                value *= -neg / pos;
            else if (value < 0 && pos + neg < 0)
                value *= -pos / neg;
            contribution[i][0] = value; // contribution now holds `limited`
        }
        bool need = false;
#pragma unroll
        for (int i = 0; i < n; ++i) {
            double t[4];
            for (int r = 0; r < 4; ++r)
                t[r] = ubar[i][r] + contribution[i][r] / (2 * v.total_viscosity[i]);
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
            const double lo = 2 * v.total_viscosity[i] * (ubar[i][r] - v.low[r * NN + ni]);
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
    const double tv = i < N ? v.total_viscosity[i] : 1.;
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
            const int pk = i * (N - 1) + k, j = k < i ? k : k + 1, nj = map[j];
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
        const double clipped = fmax(fi[0], -2 * tv * (1 - v.positivity_fraction) * ub[0]);
        double pos = 0, neg = 0;
#pragma unroll
        for (int k = 0; k < N; ++k) {
            const double ck = __shfl_sync(FULL, clipped, base + k);
            (ck > 0 ? pos : neg) += ck;
        }
        double value = clipped;
        if (value > 0 && pos + neg > 0)
            value *= -neg / pos;
        else if (value < 0 && pos + neg < 0)
            value *= -pos / neg;
        lim[0] = value;
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
            g[r] = lim[i][r] / (2 * v.total_viscosity[i]);
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
            const double lo = 2 * v.total_viscosity[i] * (ubar[i][r] - v.low[std::size_t(r) * v.nodes + map[i]]);
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
    if (threadIdx.x == 0)
        atomicAdd(out, part[0]);
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
};
struct GhostDevice {
    GhostView g;
    int warps = 4; // warps per block of k_ghost (dynamic shared memory: warps x max_points doubles)
    std::vector<void *> owned;
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
    for (int j = lane; j < P; j += 32) {
        const double *Hj = g.harmonics[sd] + std::size_t(j) * q;
        double s = 0;
        for (int c = 0; c < q; ++c)
            s += Hj[c] * (g.sigma ? g.sigma[c] * row[c] : row[c]);
        y[j] = s;
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
    ctx->ghost = gd;
    int least = 0, greatest = 0; // numerically larger = lower priority
    CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    CK(cudaStreamCreateWithPriority(&ctx->aux, cudaStreamNonBlocking, least));
}
// On the aux stream after the work queued so far on the default stream (event 3); completion is event 2.
void ghost(Context *ctx, const double *state, double *pinned_moments, int *pinned_flags) {
    GhostView &g = ctx->ghost->g;
    cudaStream_t st = ctx->aux;
    CK(cudaEventRecord(ctx->events[3], 0));
    CK(cudaStreamWaitEvent(st, ctx->events[3], 0));
    CK(cudaMemsetAsync(&g.stats->flagged, 0, sizeof(int), st));
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
    const double *ltable; // lattice FCS: points x (lM + 1) time table (device)
    int lM;
    double ldt;
    double *ledger; // absorbed, injected
};
struct ReactionDevice {
    ReactionView r;
    std::vector<void *> owned;
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
        const double f = s - i, *row = r.ltable + t * std::size_t(r.lM + 1);
        const double p1 = row[i], p2 = row[i + 1], p3 = row[i + 2], p0 = i > 0 ? row[i - 1] : 2 * p1 - p2;
        const double val = p1 + .5 * f * (p2 - p0 + f * (2 * p0 - 5 * p1 + 4 * p2 - p3 + f * (3 * (p1 - p2) + p3 - p0)));
        rho = fmax(val, 0.);
    }
    r.q[t] = r.fw[t] * r.fss[t] * rho;
}
__global__ void k_lattice_table(uncollided::LatticeView lv, const double *x, const double *y, std::size_t points, int M,
                                double dt, double *table) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= points * std::size_t(M + 1))
        return;
    const std::size_t k = t / (M + 1);
    const int m = int(t % (M + 1));
    table[t] = m == 0 ? 0. : uncollided::lattice_density(lv, x[k], y[k], m * dt);
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
    r.ltable = nullptr, r.lM = 0, r.ldt = 0;
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
        CK(cudaMemcpy(r.table, host_table, std::size_t(n + 1) * sizeof(double), cudaMemcpyHostToDevice));
        k_fcs_points<<<blocks(std::size_t(v.cells) * r.P), 256>>>(v, r, n, dr, cx, cy);
        k_fcs_element<<<blocks(std::size_t(v.cells) * v.local), 256>>>(v, r);
        k_fcs_gather<<<blocks(v.nodes), 256>>>(v, r);
    }
    k_react<<<blocks(v.nodes), 256>>>(v, r, state, tau, fcs);
    k_admissible<<<blocks(v.nodes), 256>>>(v, state, false, false, nullptr, nullptr, 0, 0, 0);
}
void lattice_table(Context *ctx, const uncollided::LatticeView &hv, int M, double dt, double *host_out) {
    ReactionDevice *rd = ctx->reaction;
    ReactionView &r = rd->r;
    const View &v = ctx->v;
    if (!r.P)
        throw std::runtime_error("lattice_table: no FCS points");
    auto own = [&](auto *p) {
        rd->owned.push_back((void *)p);
        return p;
    };
    uncollided::LatticeView lv = hv; // device copies of the arrays
    lv.sigma_t = own(dcopy(hv.sigma_t, std::size_t(hv.nx) * hv.ny));
    lv.phi_x = own(dcopy(hv.phi_x, std::size_t(hv.phi_n))), lv.phi_w = own(dcopy(hv.phi_w, std::size_t(hv.phi_n)));
    lv.theta_x = own(dcopy(hv.theta_x, std::size_t(hv.theta_n)));
    lv.theta_w = own(dcopy(hv.theta_w, std::size_t(hv.theta_n)));
    const std::size_t points = std::size_t(v.cells) * r.P, total = points * std::size_t(M + 1);
    double *table = own(dalloc<double>(total));
    k_lattice_table<<<blocks(total, 128), 128>>>(lv, r.fx, r.fy, points, M, dt, table);
    CK(cudaGetLastError());
    if (host_out)
        CK(cudaMemcpy(host_out, table, total * sizeof(double), cudaMemcpyDeviceToHost));
    r.ltable = table, r.lM = M, r.ldt = dt;
}
__global__ void k_lattice3d_table(uncollided::Lattice3dView lv, const double *x, const double *y, const double *z,
                                  std::size_t first, std::size_t count, int M, double dt, double *work,
                                  double *table) {
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
    uncollided::lattice3d_row(lv, p, dt, M, work + t * 3 * std::size_t(M + 2), table + k * std::size_t(M + 1));
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
    // Batches bound the work arrays (3 (M + 2) doubles per point); the per-thread break lists live in local memory.
    const std::size_t batch = std::min<std::size_t>(points, 1 << 16);
    double *work = dalloc<double>(batch * 3 * std::size_t(M + 2));
    CK(cudaDeviceSetLimit(cudaLimitStackSize, 24 * 1024)); // two break lists of lattice3d_max_breaks doubles
    for (std::size_t first = 0; first < points; first += batch) {
        const std::size_t count = std::min(batch, points - first);
        k_lattice3d_table<<<blocks(count, 64), 64>>>(lv, dx, dy, dz, first, count, M, dt, work, table);
        CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());
    CK(cudaFree(work));
    CK(cudaFree(dx)), CK(cudaFree(dy)), CK(cudaFree(dz));
    if (host_out)
        CK(cudaMemcpy(host_out, table, total * sizeof(double), cudaMemcpyDeviceToHost));
    r.ltable = table, r.lM = M, r.ldt = dt;
}
void react_lattice(Context *ctx, double *state, double tau, double source_time) {
    ReactionView &r = ctx->reaction->r;
    const View &v = ctx->v;
    if (!r.ltable)
        throw std::runtime_error("react_lattice: no lattice table");
    k_fcs_points_lattice<<<blocks(std::size_t(v.cells) * r.P), 256>>>(v, r, source_time);
    k_fcs_element<<<blocks(std::size_t(v.cells) * v.local), 256>>>(v, r);
    k_fcs_gather<<<blocks(v.nodes), 256>>>(v, r);
    k_react<<<blocks(v.nodes), 256>>>(v, r, state, tau, true);
    k_admissible<<<blocks(v.nodes), 256>>>(v, state, false, false, nullptr, nullptr, 0, 0, 0);
}
Ledger ledger(Context *ctx) {
    double h[2];
    CK(cudaMemcpy(h, ctx->reaction->r.ledger, sizeof h, cudaMemcpyDeviceToHost));
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
    v.stiffness_rows = own(dcopy(s.stiffness_rows, std::size_t(n) * n));
    v.seminorm = own(dcopy(s.seminorm, std::size_t(n) * n));
    for (int d = 0; d < 3; ++d) {
        v.convection_t_rows[d] = d < s.dimension ? own(dcopy(s.convection_t_rows[d], std::size_t(n) * n)) : nullptr;
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
    const std::size_t pairs = std::size_t(n) * (n - 1);
    v.pair_j = own(dcopy(s.pair_j, pairs));
    v.pair_d = own(dcopy(s.pair_d, pairs));
    v.pair_m = own(dcopy(s.pair_m, pairs));
    v.pair_c = own(dcopy(s.pair_c, 3 * pairs));
    v.pair_ct = own(dcopy(s.pair_ct, 3 * pairs));
    v.total_viscosity = own(dcopy(s.total_viscosity, std::size_t(n)));
    v.lumped = own(dcopy(s.lumped, std::size_t(s.nodes)));
    if (s.dimension == 2 && s.degree == 1) {
        CK(cudaMemcpyToSymbol(cK, s.stiffness_rows, 16 * sizeof(double)));
        double kv[16];
        for (int k = 0; k < 16; ++k)
            kv[k] = s.viscosity * s.stiffness_rows[k];
        CK(cudaMemcpyToSymbol(cKv, kv, 16 * sizeof(double)));
        CK(cudaMemcpyToSymbol(cC0, s.convection_t_rows[0], 16 * sizeof(double)));
        CK(cudaMemcpyToSymbol(cC1, s.convection_t_rows[1], 16 * sizeof(double)));
    }
    v.weno = s.weno, v.stabilize = s.stabilize, v.limit = s.limit, v.outward_bar_rounding = s.outward_bar_rounding;
    v.scale_invariant_weno = s.scale_invariant_weno;
    v.weno_q = s.weno_q, v.weno_r = s.weno_r, v.weno_epsilon = s.weno_epsilon, v.weno_linear = s.weno_linear;
    v.positivity_fraction = s.positivity_fraction, v.element_theta = s.element_theta;
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
    ctx->norm_accumulator = own(dalloc<double>(1));
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
    double *row_mass = nullptr, *flux_t[6] = {}, *G = nullptr, *H = nullptr, *db = nullptr, *low4 = nullptr;
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
                                    const double *H, double *out, double *low4) {
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(nb) * q)
        return;
    const int k = int(t / q), c = int(t % q);
    double sum = 0;
    for (int e = offsets[k]; e < offsets[k + 1]; ++e)
        sum += H[std::size_t(rows_list[e]) * q + c];
    out[std::size_t(bnodes[k]) * q + c] += sum;
    if (c < 4)
        low4[std::size_t(k) * 4 + c] = sum;
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
    for (int side = 0; side < 6; ++side)
        if (s.flux_t[side])
            d->flux_t[side] = dcopy(s.flux_t[side], std::size_t(s.q) * s.q);
    d->G = dalloc<double>(rq);
    d->H = dalloc<double>(rq);
    d->db = dalloc<double>(rq);
    d->low4 = dalloc<double>(std::size_t(std::max(1, s.nb)) * 4);
    cublas_check(cublasCreate(&d->handle), "create");
    cublas_check(cublasSetStream(d->handle, 0), "set stream"); // legacy default stream, as all kernels here
    ctx->bhigh = d;
}
void destroy_boundary_high(Context *ctx) {
    BoundaryHighDevice *d = ctx->bhigh;
    if (!d)
        return;
    for (void *p : {(void *)d->row_nodes, (void *)d->offsets, (void *)d->rows_list, (void *)d->row_mass, (void *)d->G,
                    (void *)d->H, (void *)d->db, (void *)d->low4})
        cudaFree(p);
    for (double *p : d->flux_t)
        cudaFree(p);
    if (d->handle)
        cublasDestroy(d->handle);
    delete d;
    ctx->bhigh = nullptr;
}
void boundary_high(Context *ctx, const double *state, double *out, const double *pinned_db, double *pinned_low4) {
    BoundaryHighDevice *d = ctx->bhigh;
    const BoundaryHighSetup &s = d->s;
    if (s.rows == 0 || s.nb == 0)
        return;
    const std::size_t rq = std::size_t(s.rows) * s.q;
    k_face_traces<<<blocks(rq), 256>>>(s.q, s.rows, s.nf, d->row_nodes, d->row_mass, state, d->G);
    CK(cudaMemsetAsync(d->H, 0, rq * sizeof(double), 0));
    const double one = 1, zero = 0;
    for (int side = 0; side < 6; ++side) {
        const int r0 = s.side_start[side], n = s.side_start[side + 1] - r0;
        if (!d->flux_t[side] || n <= 0)
            continue;
        // Row-major G (rows x q) is column-major G^T (q x rows): H^T = B^T G^T with B = flux_t (column-major q x q).
        cublas_check(cublasDgemm(d->handle, CUBLAS_OP_T, CUBLAS_OP_N, s.q, n, s.q, &one, d->flux_t[side], s.q,
                                 d->G + std::size_t(r0) * s.q, s.q, &zero, d->H + std::size_t(r0) * s.q, s.q),
                     "dgemm");
    }
    if (pinned_db) {
        CK(cudaMemcpyAsync(d->db, pinned_db, rq * sizeof(double), cudaMemcpyHostToDevice, 0));
        k_sub<<<blocks(rq), 256>>>(d->H, d->db, rq);
    }
    k_boundary_rows_add<<<blocks(std::size_t(s.nb) * s.q), 256>>>(s.q, s.nb, d->offsets, d->rows_list, s.bnodes, d->H,
                                                                   out, d->low4);
    CK(cudaMemcpyAsync(pinned_low4, d->low4, std::size_t(s.nb) * 4 * sizeof(double), cudaMemcpyDeviceToHost, 0));
}
void destroy(Context *ctx) {
    if (!ctx)
        return;
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
    destroy_boundary_high(ctx);
    if (ctx->ghost) {
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
    delete ctx;
}
Buffers buffers(Context *ctx) { return ctx->b; }
void upload(void *device, const void *host, std::size_t bytes) {
    CK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
}
void download(void *host, const void *device, std::size_t bytes) {
    CK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
}
void synchronize() { CK(cudaDeviceSynchronize()); }
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
void closure(Context *ctx, const double *state) { k_closure<<<blocks(ctx->v.nodes), 256>>>(ctx->v, state); }
void weno_gamma(Context *ctx, const double *state) {
    const View &v = ctx->v;
    if (v.local == 4)
        k_weno<4><<<blocks(v.cells, 128), 128>>>(v, state);
    else if (v.local == 8)
        k_weno<8><<<blocks(v.cells, 128), 128>>>(v, state);
    else if (v.local == 9)
        k_weno<9><<<blocks(v.cells, 128), 128>>>(v, state);
    else
        k_weno<0><<<blocks(v.cells, 128), 128>>>(v, state);
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
        case 1: k_line_solve_cyclic<1><<<grid, 64>>>(v, d, f, src); break;
        case 2: k_line_solve_cyclic<2><<<grid, 64>>>(v, d, f, src); break;
        case 3: k_line_solve_cyclic<3><<<grid, 64>>>(v, d, f, src); break;
        default: k_line_solve_cyclic<4><<<grid, 64>>>(v, d, f, src); break;
        }
        return;
    }
    switch (v.degree) {
    case 1: k_line_solve<1, LINE_CHUNK><<<grid, 64>>>(v, d, f, src); break;
    case 2: k_line_solve<2, LINE_CHUNK><<<grid, 64>>>(v, d, f, src); break;
    case 3: k_line_solve<3, LINE_CHUNK><<<grid, 64>>>(v, d, f, src); break;
    default: k_line_solve<4, LINE_CHUNK><<<grid, 64>>>(v, d, f, src); break;
    }
}
} // namespace
void gradients(Context *ctx, const double *state) {
    const View &v = ctx->v;
    line_solve(ctx, 0, v.dimension, nullptr, state);
}
void stabilization(Context *ctx, const double *state, double *out, bool want_low, double constant_gamma) {
    const View &v = ctx->v;
    if (v.dimension == 2 && v.degree == 1 && !v.periodic) { // the strip kernel assumes the open grid
        static const int sy = [] {
            const char *e = std::getenv("M1PN_GPU_STRIP");
            return e ? std::atoi(e) : 16;
        }();
        const dim3 grid(blocks(std::size_t(v.npa[0]) * v.q, 128), (v.npa[1] + sy - 1) / sy);
        if (sy == 8)
            k_stabilization_strip<8><<<grid, 128>>>(v, state, out, want_low, constant_gamma);
        else if (sy == 16)
            k_stabilization_strip<16><<<grid, 128>>>(v, state, out, want_low, constant_gamma);
        else if (sy == 64)
            k_stabilization_strip<64><<<grid, 128>>>(v, state, out, want_low, constant_gamma);
        else
            k_stabilization_strip<32><<<grid, 128>>>(v, state, out, want_low, constant_gamma);
    } else {
        double *local_out = device_doubles(ctx, 7, std::size_t(v.cells) * v.local * v.q);
        const unsigned g = blocks(std::size_t(v.cells) * v.q, 128);
        const bool d3 = v.dimension == 3;
        if (v.local == 9 && !d3)
            k_stab_element<9, 2><<<g, 128>>>(v, state, local_out, want_low, constant_gamma);
        else if (v.local == 16 && !d3)
            k_stab_element<16, 2><<<g, 128>>>(v, state, local_out, want_low, constant_gamma);
        else if (v.local == 25 && !d3)
            k_stab_element<25, 2><<<g, 128>>>(v, state, local_out, want_low, constant_gamma);
        else if (v.local == 8 && d3)
            k_stab_element<8, 3><<<g, 128>>>(v, state, local_out, want_low, constant_gamma);
        else if (v.local == 27 && d3)
            k_stab_element<27, 3><<<g, 128>>>(v, state, local_out, want_low, constant_gamma);
        else {
            k_stabilization<<<blocks(std::size_t(v.nodes) * v.q), 256>>>(v, state, out, want_low, constant_gamma);
            return;
        }
        k_gather_stab<<<blocks(std::size_t(v.nodes) * v.q), 256>>>(v, local_out, out);
    }
}
void add_rows(Context *ctx, double *field, const int *index, const double *rows, int count) {
    if (count)
        k_add_rows<<<blocks(std::size_t(count) * ctx->v.q), 256>>>(ctx->v.q, field, index, rows, count);
}
void mass_solve(Context *ctx, double *field) {
    for (int d = 0; d < ctx->v.dimension; ++d)
        line_solve(ctx, d, 1, field);
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
    k_flux_rows<<<blocks(std::size_t(v.nodes) * 4), 256, flux_shared(ctx)>>>(v, const_cast<double *>(field), true, true);
}
void flux_rows(Context *ctx, double *field, bool want_moments) {
    const View &v = ctx->v;
    k_flux_rows<<<blocks(std::size_t(v.nodes) * v.q), 256, flux_shared(ctx)>>>(v, field, want_moments, false);
}
void limiter(Context *ctx) {
    const View &v = ctx->v;
    const unsigned b4 = blocks(std::size_t(v.cells) * 4, 256), b8 = blocks(std::size_t(v.cells) * 8, 256),
                   b16 = blocks(std::size_t(v.cells) * 16, 256), b32 = blocks(std::size_t(v.cells) * 32, 256);
    if (v.local == 4) {
        k_limiter_group<4, 4><<<b4, 256>>>(v);
        k_limiter_segment<4><<<blocks(v.cells, 64), 64>>>(v);
    } else if (v.local == 8) {
        k_limiter_group<8, 8><<<b8, 256>>>(v);
        k_limiter_segment<8><<<blocks(v.cells, 64), 64>>>(v);
    } else if (v.local == 9) {
        k_limiter_group<9, 16><<<b16, 256>>>(v);
        k_limiter_segment<9><<<blocks(v.cells, 64), 64>>>(v);
    } else if (v.local == 16) {
        k_limiter_group<16, 16><<<b16, 256>>>(v);
        k_limiter_segment<16><<<blocks(v.cells, 64), 64>>>(v);
    } else if (v.local == 25) {
        k_limiter_group<25, 32><<<b32, 256>>>(v);
        k_limiter_segment<25><<<blocks(v.cells, 64), 64>>>(v);
    } else if (v.local == 27) {
        k_limiter_group<27, 32><<<b32, 256>>>(v);
        k_limiter_segment<27><<<blocks(v.cells, 64), 64>>>(v);
    } else {
        k_limiter<0><<<blocks(v.cells, 128), 128>>>(v);
        k_limiter_segment<0><<<blocks(v.cells, 64), 64>>>(v);
    }
}
void gather_load(Context *ctx) { k_gather_load<<<blocks(std::size_t(ctx->v.nodes) * 4), 256>>>(ctx->v); }
void combine_pn(Context *ctx, double *r, const double *x, const double *a, double old, double fresh, double tau) {
    const View &v = ctx->v;
    k_combine<<<blocks(std::size_t(v.nodes) * v.q), 256, flux_shared(ctx)>>>(v, r, x, a, old, fresh, tau, true, 1);
}
void finish_stage(Context *ctx, double *r, const double *x, const double *a, double old, double fresh, double tau,
                  bool pn_done, bool closure_next) {
    const View &v = ctx->v;
    if (pn_done)
        k_combine<<<blocks(std::size_t(v.nodes) * 4), 256>>>(v, r, x, a, old, fresh, tau, false, 2);
    else
        k_combine<<<blocks(std::size_t(v.nodes) * v.q), 256, flux_shared(ctx)>>>(v, r, x, a, old, fresh, tau, true, 0);
    k_admissible<<<blocks(v.nodes), 256>>>(v, r, closure_next, false, x, a, old, fresh, tau);
}
double norm(Context *ctx, const double *field) {
    CK(cudaMemset(ctx->norm_accumulator, 0, sizeof(double)));
    const std::size_t count = std::size_t(ctx->v.nodes) * ctx->v.q;
    k_sumsq<<<std::min<unsigned>(blocks(count), 1024), 256>>>(field, count, ctx->norm_accumulator);
    double s = 0;
    CK(cudaMemcpy(&s, ctx->norm_accumulator, sizeof(double), cudaMemcpyDeviceToHost));
    return std::sqrt(s);
}
void divide(Context *ctx, double *dst, const double *src, double divisor) {
    const std::size_t count = std::size_t(ctx->v.nodes) * ctx->v.q;
    k_divide<<<blocks(count), 256>>>(dst, src, count, divisor);
}
namespace {
Counters convert(const DeviceCounters &d);
}
Counters counters(Context *ctx) {
    CK(cudaGetLastError()); // launch failures (e.g. resource limits) since the last call
    DeviceCounters d;
    CK(cudaMemcpy(&d, ctx->v.counters, sizeof d, cudaMemcpyDeviceToHost));
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
    CK(cudaMemcpyAsync(ctx->pinned_counters, ctx->v.counters, sizeof(DeviceCounters), cudaMemcpyDeviceToHost, 0));
}
Counters snapshot(Context *ctx) {
    CK(cudaGetLastError());
    return convert(*ctx->pinned_counters);
}
void reset_fallbacks(Context *ctx) {
    CK(cudaMemsetAsync(&ctx->v.counters->closure_fallbacks, 0, 3 * sizeof(int), 0));
}
void record(Context *ctx, int event) { CK(cudaEventRecord(ctx->events[event], 0)); }
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
    k_gather<<<blocks(std::size_t(count) * cols), 256>>>(field, cols, device_index, count, buf, rows_soa);
    CK(cudaMemcpyAsync(pinned_out, buf, std::size_t(count) * cols * sizeof(double), cudaMemcpyDeviceToHost, 0));
}
void scatter_async(Context *ctx, double *field, int cols, const int *device_index, int count, const double *pinned_in,
                   bool add, int slot, long long rows_soa) {
    if (!count)
        return;
    double *buf = device_doubles(ctx, slot, std::size_t(count) * cols);
    CK(cudaMemcpyAsync(buf, pinned_in, std::size_t(count) * cols * sizeof(double), cudaMemcpyHostToDevice, 0));
    k_scatter<<<blocks(std::size_t(count) * cols), 256>>>(field, cols, device_index, count, buf, add, rows_soa);
}
void reset_counters(Context *ctx) {
    DeviceCounters d{};
    const double one = 1;
    std::memcpy(&d.min_limiter_bits, &one, 8);
    std::memcpy(&d.min_gamma_bits, &one, 8);
    CK(cudaMemcpy(ctx->v.counters, &d, sizeof d, cudaMemcpyHostToDevice));
}
const int *closure_fallback_list(Context *ctx) { return ctx->v.closure_list; }
const int *limiter_fallback_list(Context *ctx) { return ctx->v.limiter_list; }
// Gathers/scatters through a device index list (slot 0 for indices, slot 0/1 for rows).
void gather_device(Context *ctx, const double *field, int cols, const int *device_index, int count, double *host_out,
                   long long rows_soa) {
    if (!count)
        return;
    double *buf = device_doubles(ctx, 0, std::size_t(count) * cols);
    k_gather<<<blocks(std::size_t(count) * cols), 256>>>(field, cols, device_index, count, buf, rows_soa);
    CK(cudaMemcpy(host_out, buf, std::size_t(count) * cols * sizeof(double), cudaMemcpyDeviceToHost));
}
void scatter_device(Context *ctx, double *field, int cols, const int *device_index, int count, const double *host_in,
                    bool add, long long rows_soa) {
    if (!count)
        return;
    double *buf = device_doubles(ctx, 1, std::size_t(count) * cols);
    CK(cudaMemcpy(buf, host_in, std::size_t(count) * cols * sizeof(double), cudaMemcpyHostToDevice));
    k_scatter<<<blocks(std::size_t(count) * cols), 256>>>(field, cols, device_index, count, buf, add, rows_soa);
}
} // namespace gpu
} // namespace m1pn
