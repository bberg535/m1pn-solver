// GPU kernels of the element WENO time loop (experimental, docs/ELEMENT_WENO_GPU.md). One thread per output entry or
// per element; every output is computed by one thread in a fixed order, so a run is deterministic. The arithmetic
// follows src/element_weno.cpp; summation orders and fused multiply-adds differ (rounding level). The exact M1 cone
// predicate is evaluated exactly (error-free transformations); paths that need long double on the host (near-beam
// Levermore states, the stable base bar) are flagged and recomputed by the host with the original code.
#include "element_weno_gpu_device.hpp"
#include <cuda_runtime.h>
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
            if (next < 0)
                continue;
            for (int l = 0; l < n; ++l)
                other[l] = v.density[v.connectivity[next * n + l]];
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
    const double *L = v.lower[d], *ID = v.invdiag[d];
    double *base = f + std::size_t(start) * v.q + c;
    const std::size_t step = std::size_t(s) * v.q;
    double hist[P > 0 ? P : 1];
    for (int m = 0; m < P; ++m)
        hist[m] = 0;
    const double *sbase = src ? src + std::size_t(start) * v.q + c : nullptr;
    const double *Cb = v.convection_band[d];
    double win[2 * P + 1]; // win[m + P] = src[k + m]
    if (src) {
#pragma unroll
        for (int j = 0; j < 2 * P; ++j) {
            const int kk = j - P;
            win[j] = kk >= 0 && kk < n ? sbase[std::size_t(kk) * step] : 0.;
        }
    }
    // Software pipelining: the loads of chunk k + 1 are issued before chunk k is computed (forward: the chunk ahead is
    // not written yet; backward: the chunk below is not written yet).
    auto load_fwd = [&](int k0, double *buf) {
#pragma unroll
        for (int j = 0; j < CH; ++j) {
            if (src)
                buf[j] = k0 + j + P < n ? sbase[std::size_t(k0 + j + P) * step] : 0.;
            else
                buf[j] = k0 + j < n ? base[std::size_t(k0 + j) * step] : 0.;
        }
    };
    auto fwd = [&](int k0, const double *buf) {
#pragma unroll
        for (int j = 0; j < CH; ++j) {
            const int k = k0 + j;
            if (k >= n)
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
    for (int k0 = 0; k0 < n; k0 += 2 * CH) {
        if (k0 + CH < n)
            load_fwd(k0 + CH, bufB);
        fwd(k0, bufA);
        if (k0 + CH >= n)
            break;
        if (k0 + 2 * CH < n)
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
                if (k + m < n)
                    x -= L[std::size_t(k + m) * P + m - 1] * hist[m - 1];
#pragma unroll
            for (int m = P - 1; m > 0; --m)
                hist[m] = hist[m - 1];
            if (P > 0)
                hist[0] = x;
            base[std::size_t(k) * step] = x;
        }
    };
    load_bwd(n - 1, bufA);
    for (int k1 = n - 1; k1 >= 0; k1 -= 2 * CH) {
        if (k1 - CH >= 0)
            load_bwd(k1 - CH, bufB);
        bwd(k1, bufA);
        if (k1 - CH < 0)
            break;
        if (k1 - 2 * CH >= 0)
            load_bwd(k1 - 2 * CH, bufA);
        bwd(k1 - CH, bufB);
    }
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
template <int N> __global__ void __launch_bounds__(256) k_limiter_group(View v) {
    const unsigned FULL = 0xffffffffu;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int e = t / N, i = t % N, lane = threadIdx.x & 31, base = lane & ~(N - 1);
    const bool valid = e < v.cells;
    const int dim = v.dimension;
    const std::size_t NN = v.nodes;
    double ub[4] = {0, 0, 0, 0}, fi[4] = {0, 0, 0, 0}, ui[4] = {0, 0, 0, 0};
    bool bad = false, uncertain = false;
    const double tv = v.total_viscosity[i];
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
    for (int o = 1; o < N; o <<= 1)
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
    for (int o = 1; o < N; o <<= 1)
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
            const double rate = (v.load[std::size_t(i) * 4 + slot] / v.lumped[i]) / (c == 0 ? v.s0 : v.s1);
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
    }
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
void destroy(Context *ctx) {
    if (!ctx)
        return;
    for (void *p : ctx->owned)
        cudaFree(p);
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
    if (v.dimension == 2 && v.degree == 1) {
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
    }
    else
        k_stabilization<<<blocks(std::size_t(v.nodes) * v.q), 256>>>(v, state, out, want_low, constant_gamma);
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
    if (v.local == 4) {
        k_limiter_group<4><<<blocks(std::size_t(v.cells) * 4, 256), 256>>>(v);
        k_limiter_segment<4><<<blocks(v.cells, 64), 64>>>(v);
    } else if (v.local == 8) {
        k_limiter<8><<<blocks(v.cells, 128), 128>>>(v);
        k_limiter_segment<8><<<blocks(v.cells, 64), 64>>>(v);
    } else if (v.local == 9) {
        k_limiter<9><<<blocks(v.cells, 128), 128>>>(v);
        k_limiter_segment<9><<<blocks(v.cells, 64), 64>>>(v);
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
Counters snapshot(Context *ctx) { return convert(*ctx->pinned_counters); }
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
