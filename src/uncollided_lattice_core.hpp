#pragma once
// Core of the lattice uncollided density (formulas in src/uncollided_lattice.hpp, docs/FCS_LATTICE.md), written once
// for host and device: plain arrays, global-namespace math, no allocation. CellLattice2D (host) and the table kernel
// in src/element_weno_gpu.cu both call lattice_density().
#include <math.h>
#if defined(__CUDACC__)
#define M1PN_HD __host__ __device__
#else
#define M1PN_HD
#endif
namespace m1pn {
namespace uncollided {
constexpr int lattice_max_breaks = 96; // window ends + (nx+1)(ny+1) corners + 8 front points, nx, ny <= 8
struct LatticeView {
    int nx = 7, ny = 7;
    const double *sigma_t = nullptr; // nx * ny, cell (i, j) at i + nx j
    int si = 3, sj = 3;              // source cell
    double q0 = 1;
    int phi_n = 0, theta_n = 0;
    const double *phi_x = nullptr, *phi_w = nullptr, *theta_x = nullptr, *theta_w = nullptr; // Gauss on [0, 1]
    bool prune = true;
};
M1PN_HD inline double lattice_sigma(const LatticeView &v, int i, int j) { return v.sigma_t[i + v.nx * j]; }
// Ray x + l d against the source cell: [l_in, l_out] (false: no hit or touch only).
M1PN_HD inline bool lattice_hit(const LatticeView &v, double x, double y, double dx, double dy, double &l_in,
                                double &l_out) {
    double lo = 0, hi = 1e300;
    const double b0[2] = {double(v.si), double(v.sj)}, p[2] = {x, y}, d[2] = {dx, dy};
    for (int a = 0; a < 2; ++a) {
        if (fabs(d[a]) < 1e-300) {
            if (p[a] < b0[a] || p[a] > b0[a] + 1)
                return false;
            continue;
        }
        double t0 = (b0[a] - p[a]) / d[a], t1 = (b0[a] + 1 - p[a]) / d[a];
        if (t0 > t1) {
            const double s = t0;
            t0 = t1, t1 = s;
        }
        lo = fmax(lo, t0), hi = fmin(hi, t1);
    }
    if (hi <= lo)
        return false;
    l_in = lo, l_out = hi;
    return true;
}
// Optical depth along x + l d for l in [0, length] (cell traversal; the path stays inside the lattice).
M1PN_HD inline double lattice_depth(const LatticeView &v, double x, double y, double dx, double dy, double length) {
    if (length <= 0)
        return 0;
    double fx = floor(x), fy = floor(y);
    if (x == fx && dx < 0)
        fx -= 1;
    if (y == fy && dy < 0)
        fy -= 1;
    int i = int(fx), j = int(fy);
    i = i < 0 ? 0 : (i > v.nx - 1 ? v.nx - 1 : i);
    j = j < 0 ? 0 : (j > v.ny - 1 ? v.ny - 1 : j);
    const int si = dx > 0 ? 1 : -1, sj = dy > 0 ? 1 : -1;
    const double inf = 1e300;
    double tx = fabs(dx) < 1e-300 ? inf : ((dx > 0 ? i + 1 : i) - x) / dx;
    double ty = fabs(dy) < 1e-300 ? inf : ((dy > 0 ? j + 1 : j) - y) / dy;
    const double ddx = fabs(dx) < 1e-300 ? inf : 1 / fabs(dx), ddy = fabs(dy) < 1e-300 ? inf : 1 / fabs(dy);
    double l = 0, T = 0;
    for (int guard = 0; guard < 4 * (v.nx + v.ny) + 8; ++guard) {
        const double next = fmin(fmin(tx, ty), length);
        T += lattice_sigma(v, i, j) * (next - l);
        l = next;
        if (l >= length)
            return T;
        if (tx <= ty)
            i += si, tx += ddx;
        else
            j += sj, ty += ddy;
        if (i < 0 || i >= v.nx || j < 0 || j >= v.ny)
            return T; // left the lattice (cannot happen before S for x inside)
    }
    return T;
}
// Fast IEEE-754 single-precision transcendentals on the GPU for the theta quadrature integrand (quadrature and table
// interpolation errors are ~2-5e-6, two orders of magnitude above float's 6e-8; geometry and sums remain binary64).
M1PN_HD inline double lattice_exp(double x) {
#if defined(__CUDA_ARCH__)
    return double(expf(float(x)));
#else
    return exp(x);
#endif
}
M1PN_HD inline double lattice_expm1(double x) {
#if defined(__CUDA_ARCH__)
    return double(expm1f(float(x)));
#else
    return expm1(x);
#endif
}
M1PN_HD inline double lattice_sin(double x) {
#if defined(__CUDA_ARCH__)
    return double(sinf(float(x)));
#else
    return sin(x);
#endif
}

// theta panel [lo, hi]: sum of w f with f the closed-form l integral.
M1PN_HD inline double lattice_panel(const LatticeView &v, double lo, double hi, bool growing, double l_in, double l_out,
                                    double T_in, double sS, double t) {
    double sum = 0;
    for (int k = 0; k < v.theta_n; ++k) {
        const double th = lo + (hi - lo) * v.theta_x[k], mu = lattice_sin(th);
        if (mu <= 0)
            continue;
        const double L = (growing ? t * mu : l_out) - l_in;
        if (L <= 0)
            continue;
        const double inv_mu = 1. / mu;
        const double e = T_in > 0 ? lattice_exp(-T_in * inv_mu) : 1.;
        const double f = sS > 0 ? e * mu / sS * -lattice_expm1(-sS * L * inv_mu) : e * L;
        sum += (hi - lo) * v.theta_w[k] * f;
    }
    return sum;
}
// theta integral of one ray (l_in, l_out, T_in) at time t: split at mu = l_in/t, l_out/t; geometric panels lo 2^k
// (boundary layer of exp(-l/mu) above a lower end lo ~ l/t).
M1PN_HD inline double lattice_theta(const LatticeView &v, double l_in, double l_out, double T_in, double sS, double t) {
    if (t <= l_in || (v.prune && T_in > 40))
        return 0;
    const double h = 1.5707963267948966;
    const double a1 = asin(fmin(1., l_in / t)), a2 = asin(fmin(1., l_out / t));
    double sum = 0;
    for (int part = 0; part < 2; ++part) {
        const double lo0 = part ? a2 : a1, hi0 = part ? h : a2;
        if (hi0 <= lo0)
            continue;
        for (double lo = lo0; lo < hi0;) {
            const double hi = lo0 > 0 && 2 * lo < hi0 ? 2 * lo : hi0;
            sum += lattice_panel(v, lo, hi, part == 0, l_in, l_out, T_in, sS, t);
            lo = hi;
        }
    }
    return sum;
}
// rho_u(x, t).
M1PN_HD inline double lattice_density(const LatticeView &v, double x, double y, double t) {
    if (t <= 0)
        return 0;
    const double pi = 3.14159265358979323846;
    const double s0 = v.si, s1 = v.si + 1, r0 = v.sj, r1 = v.sj + 1;
    const double sS = lattice_sigma(v, v.si, v.sj);
    const bool inside = x >= s0 && x <= s1 && y >= r0 && y <= r1;
    if (v.prune) {
        const double ex = fmax(fmax(s0 - x, 0.), x - s1), ey = fmax(fmax(r0 - y, 0.), y - r1);
        if (ex * ex + ey * ey >= t * t)
            return 0; // the front has not reached S
    }
    // Angular window [w0, w1] of directions that hit S.
    double w0 = 0, w1 = 2 * pi;
    if (!inside) {
        const double base = atan2(.5 * (r0 + r1) - y, .5 * (s0 + s1) - x);
        double lo = 1e9, hi = -1e9;
        const double cx[4] = {s0, s1, s1, s0}, cy[4] = {r0, r0, r1, r1};
        for (int k = 0; k < 4; ++k) {
            const double a = remainder(atan2(cy[k] - y, cx[k] - x) - base, 2 * pi);
            lo = fmin(lo, a), hi = fmax(hi, a);
        }
        w0 = base + lo, w1 = base + hi;
    }
    double br[lattice_max_breaks];
    int nb = 0;
    br[nb++] = w0, br[nb++] = w1;
    auto add = [&](double a) {
        double r = fmod(a - w0, 2 * pi); // map into [w0, w0 + 2 pi)
        if (r < 0)
            r += 2 * pi;
        a = w0 + r;
        if (a > w0 && a < w1 && nb < lattice_max_breaks)
            br[nb++] = a;
    };
    for (int i = 0; i <= v.nx; ++i)
        for (int j = 0; j <= v.ny; ++j) {
            if (double(i) == x && double(j) == y)
                continue;
            const bool corner_of_S = (i == v.si || i == v.si + 1) && (j == v.sj || j == v.sj + 1);
            if (v.prune && !corner_of_S) {
                // Only corners strictly before the entry into S change the cells crossed on [0, l_in].
                const double vx = i - x, vy = j - y, r = sqrt(vx * vx + vy * vy);
                double l_in, l_out;
                if (inside || !lattice_hit(v, x, y, vx / r, vy / r, l_in, l_out) || r >= l_in)
                    continue;
            }
            add(atan2(j - y, i - x));
        }
    // Circle of radius t around x against the four edges of S (l_in = t or l_out = t).
    const double ax[4] = {s0, s1, s1, s0}, ay[4] = {r0, r0, r1, r1};
    for (int e = 0; e < 4; ++e) {
        const double bx = ax[(e + 1) % 4], by = ay[(e + 1) % 4];
        const double ex = bx - ax[e], ey = by - ay[e], fx = ax[e] - x, fy = ay[e] - y;
        const double A = ex * ex + ey * ey, B = 2 * (fx * ex + fy * ey), C = fx * fx + fy * fy - t * t;
        const double D = B * B - 4 * A * C;
        if (D < 0)
            continue;
        const double sq = sqrt(D);
        const double roots[2] = {(-B - sq) / (2 * A), (-B + sq) / (2 * A)};
        for (int k = 0; k < 2; ++k)
            if (roots[k] > 0 && roots[k] < 1)
                add(atan2(ay[e] + roots[k] * ey - y, ax[e] + roots[k] * ex - x));
    }
    for (int a = 1; a < nb; ++a) { // insertion sort (nb <= 96)
        const double key = br[a];
        int b = a - 1;
        while (b >= 0 && br[b] > key)
            br[b + 1] = br[b], --b;
        br[b + 1] = key;
    }
    double sum = 0;
    for (int b = 0; b + 1 < nb; ++b) {
        const double a = br[b], c = br[b + 1];
        if (c - a < 1e-15)
            continue;
        for (int k = 0; k < v.phi_n; ++k) {
            // phi = a + (c - a) (3 s^2 - 2 s^3): smooths the (t - l_in)^{3/2} behaviour at the front points.
            const double sk = v.phi_x[k], phi = a + (c - a) * sk * sk * (3 - 2 * sk), jac = 6 * sk * (1 - sk);
            const double dx = cos(phi), dy = sin(phi);
            double l_in, l_out;
            if (!lattice_hit(v, x, y, dx, dy, l_in, l_out))
                continue;
            const double T_in = lattice_depth(v, x, y, dx, dy, l_in);
            sum += (c - a) * v.phi_w[k] * jac * lattice_theta(v, l_in, l_out, T_in, sS, t);
        }
    }
    return v.q0 / (2 * pi) * sum;
}

// 8-point Chebyshev antiderivative on a u-panel [u0, u1]: evaluates f at 8 Chebyshev-1 nodes, builds the degree-8
// antiderivative C[1..8] via symmetry-reduced DCT, and provides O(1) Clenshaw evaluation of int_{u0}^{alpha} f(u) du.
struct Cheb8Int {
    double c1, c2, c3, c4, c5, c6, c7, c8, pm1, half_w, mid, inv_hw;
    M1PN_HD inline void build(const double f[8], double u0, double u1) {
        constexpr double s0 = 0.98078528040323044913; // cos(pi/16)
        constexpr double s1 = 0.83146961230254523708; // cos(3pi/16)
        constexpr double s2 = 0.55557023301960222474; // cos(5pi/16)
        constexpr double s3 = 0.19509032201612826785; // cos(7pi/16)
        constexpr double p0 = 0.92387953251128675613; // cos(pi/8)
        constexpr double p1c = 0.38268343236508977173; // cos(3pi/8)
        constexpr double r2 = 0.70710678118654752440; // cos(pi/4)
        const double e0 = f[0] + f[7], e1 = f[1] + f[6], e2 = f[2] + f[5], e3 = f[3] + f[4];
        const double o0 = f[0] - f[7], o1 = f[1] - f[6], o2 = f[2] - f[5], o3 = f[3] - f[4];
        const double a0 = 0.25 * (e0 + e1 + e2 + e3);
        const double a2 = 0.25 * (p0 * (e0 - e3) + p1c * (e1 - e2));
        const double a4 = 0.25 * r2 * (e0 - e1 - e2 + e3);
        const double a6 = 0.25 * (p1c * (e0 - e3) - p0 * (e1 - e2));
        const double a1 = 0.25 * (s0 * o0 + s1 * o1 + s2 * o2 + s3 * o3);
        const double a3 = 0.25 * (s1 * o0 - s3 * o1 - s0 * o2 - s2 * o3);
        const double a5 = 0.25 * (s2 * o0 - s0 * o1 + s3 * o2 + s1 * o3);
        const double a7 = 0.25 * (s3 * o0 - s2 * o1 + s1 * o2 - s0 * o3);
        c1 = 0.5 * (a0 - a2);
        c2 = 0.25 * (a1 - a3);
        c3 = (a2 - a4) * (1. / 6.);
        c4 = 0.125 * (a3 - a5);
        c5 = 0.1 * (a4 - a6);
        c6 = (a5 - a7) * (1. / 12.);
        c7 = a6 * (1. / 14.);
        c8 = a7 * 0.0625;
        pm1 = -c1 + c2 - c3 + c4 - c5 + c6 - c7 + c8;
        half_w = 0.5 * (u1 - u0);
        mid = 0.5 * (u0 + u1);
        inv_hw = 1. / half_w;
    }
    M1PN_HD inline double integral_to(double alpha) const {
        const double s = (alpha - mid) * inv_hw;
        const double s2 = 2 * s;
        const double b8 = c8;
        const double b7 = c7 + s2 * b8;
        const double b6 = c6 + s2 * b7 - b8;
        const double b5 = c5 + s2 * b6 - b7;
        const double b4 = c4 + s2 * b5 - b6;
        const double b3 = c3 + s2 * b4 - b5;
        const double b2 = c2 + s2 * b3 - b4;
        const double b1 = c1 + s2 * b2 - b3;
        return half_w * ((s * b1 - b2) - pm1);
    }
    M1PN_HD inline double full() const { return 2 * half_w * (c1 + c3 + c5 + c7); }
};

// Accumulate one static ray (l_in, l_out, T_in) with weight w_phi into row[0..W-1] for all unsplit m in [m_first, M].
// Uses the change of variable u = cos(theta) (so sin(theta) dtheta = -du, eliminating sin/asin completely) and the
// exact t-u separation on growing part 0 + t-independent part 1 via fixed dyadic panels [1 - 2^{-p}, 1 - 2^{-(p+1)}]
// (independent of m0 and M so windowed tables match bitwise) with 8-point Chebyshev antiderivatives.
M1PN_HD inline void lattice_ray_all_times(const LatticeView &v, double l_in, double l_out, double T_in, double sS,
                                          double w_phi, int m0, int m_first, int M, double dt, const double *exp_tab,
                                          const unsigned long long *split_mask, double *row) {
    if (v.prune && T_in > 40)
        return;
    const double T_max = M * dt;
    if (T_max <= l_in)
        return;
    // Fallback for unusual sS == 0 or T_in < sS * l_in (never occurs on lattice-paul where sigma_t >= sigma_S = 1).
    if (sS <= 0 || T_in < sS * l_in - 1e-12) {
        for (int m = m_first; m <= M; ++m) {
            const int idx = m - m0;
            if ((split_mask[idx >> 6] >> (idx & 63)) & 1ULL)
                continue;
            const double t = m * dt;
            if (t > l_in)
                row[idx] += w_phi * lattice_theta(v, l_in, l_out, T_in, sS, t);
        }
        return;
    }
    constexpr double nodes[8] = {0.98078528040323044913,  0.83146961230254523708,  0.55557023301960222474,
                                 0.19509032201612826785,  -0.19509032201612826785, -0.55557023301960222474,
                                 -0.83146961230254523708, -0.98078528040323044913};
    int m1 = int(floor(l_in / dt)) + 1;
    if (m1 < m_first)
        m1 = m_first;
    while (m1 <= M && m1 * dt <= l_in)
        ++m1;
    if (m1 > M)
        return;
    int m2 = int(floor(l_out / dt)) + 1;
    if (m2 < m_first)
        m2 = m_first;
    while (m2 <= M && m2 * dt <= l_out)
        ++m2;

    const double inv_sS = 1. / sS;
    const double dT = fmax(0., T_in - sS * l_in);
    const double dL = l_out - l_in;
    const double r1_min = l_in / T_max, r2_min = fmin(1., l_out / T_max);
    const double u_stop = l_in > 0 ? sqrt(fmax(0., 1. - r1_min * r1_min)) : sqrt(fmax(0., 1. - r2_min * r2_min));

    double G_below = 0, H_below = 0, P_below = 0;

    // Sweep fixed dyadic panels [u0, u1] = [1 - 2^{-p}, 1 - 2^{-(p+1)}] bottom-up from u = 0 toward u_stop.
    double rem = 1.;
    for (int p = 0; p < 40; ++p) {
        const double u0 = 1. - rem;
        if (l_in == 0 && m2 > M)
            break;
        if (l_in > 0 && (u0 > u_stop || (m1 > M && m2 > M)))
            break;
        const double next_rem = 0.5 * rem;
        const double u1 = 1. - next_rem;
        const double mid = 0.5 * (u0 + u1), hw = 0.5 * (u1 - u0);
        double fg[8], fh[8], fp[8];
        for (int j = 0; j < 8; ++j) {
            const double u = mid + hw * nodes[j];
            const double mu = sqrt(fmax(0., 1. - u * u));
            if (mu <= 0) {
                fg[j] = fh[j] = fp[j] = 0;
                continue;
            }
            const double inv_mu = 1. / mu;
            const double eT = T_in > 0 ? lattice_exp(-T_in * inv_mu) : 1.;
            fg[j] = inv_sS * eT;
            fh[j] = inv_sS * (dT > 0 ? lattice_exp(-dT * inv_mu) : 1.);
            fp[j] = fg[j] * (-lattice_expm1(-sS * dL * inv_mu));
        }
        Cheb8Int cG, cH, cP;
        cG.build(fg, u0, u1);
        cH.build(fh, u0, u1);
        cP.build(fp, u0, u1);

        // Process m2 values whose u2(t) = sqrt(1 - (l_out/t)^2) falls in [u0, u1].
        while (m2 <= M) {
            const double t = m2 * dt;
            const double r = l_out / t;
            const double u2 = sqrt(fmax(0., 1. - r * r));
            if (u2 > u1 && p < 39)
                break;
            const int idx = m2 - m0;
            if (!((split_mask[idx >> 6] >> (idx & 63)) & 1ULL)) {
                const double G = G_below + cG.integral_to(u2);
                const double H = H_below + cH.integral_to(u2);
                const double P = P_below + cP.integral_to(u2);
                const double e_st = exp_tab ? exp_tab[idx] : exp(-sS * t);
                row[idx] += w_phi * (-G + e_st * H + P);
            }
            ++m2;
        }
        // Process m1 values whose u1(t) = sqrt(1 - (l_in/t)^2) falls in [u0, u1].
        if (l_in > 0) {
            while (m1 <= M) {
                const double t = m1 * dt;
                const double r = l_in / t;
                const double u1v = sqrt(fmax(0., 1. - r * r));
                if (u1v > u1 && p < 39)
                    break;
                const int idx = m1 - m0;
                if (!((split_mask[idx >> 6] >> (idx & 63)) & 1ULL)) {
                    const double G = G_below + cG.integral_to(u1v);
                    const double H = H_below + cH.integral_to(u1v);
                    const double e_st = exp_tab ? exp_tab[idx] : exp(-sS * t);
                    row[idx] += w_phi * (G - e_st * H);
                }
                ++m1;
            }
        }
        G_below += cG.full();
        H_below += cH.full();
        P_below += cP.full();
        rem = next_rem;
    }
    // If l_in == 0 (inside source cell), u1(t) = 1 for all m1 <= M: G(1) = 1/sS, H(1) = 1/sS.
    if (l_in == 0) {
        while (m1 <= M) {
            const int idx = m1 - m0;
            if (!((split_mask[idx >> 6] >> (idx & 63)) & 1ULL)) {
                const double e_st = exp_tab ? exp_tab[idx] : exp(-sS * (m1 * dt));
                row[idx] += w_phi * (inv_sS * (1. - e_st));
            }
            ++m1;
        }
    }
}

// Compute an entire time row rho_u(x, y, m * dt) for m = m0..M into row[0..M-m0].
// Reuses static corner breakpoints, ray geometry (l_in, l_out, T_in) and Chebyshev theta-antiderivatives across all
// unsplit times, falling back to per-time sub-interval evaluation only on the few intervals split by the front circle.
// Optional `exp_tab`: precomputed exp(-sS * (m0 + idx) * dt) for idx = 0..M-m0 (shared across points).
M1PN_HD inline void lattice_density_row(const LatticeView &v, double x, double y, int m0, int M, double dt, double *row,
                                        const double *exp_tab = nullptr) {
    const int W = M - m0 + 1;
    for (int i = 0; i < W; ++i)
        row[i] = 0.;
    // Guard against huge time tables (> 1024 entries): fall back to per-time call.
    if (W > 1024) {
        for (int m = m0; m <= M; ++m)
            row[m - m0] = m == 0 ? 0. : lattice_density(v, x, y, m * dt);
        return;
    }
    const double pi = 3.14159265358979323846;
    const double s0 = v.si, s1 = v.si + 1, r0 = v.sj, r1 = v.sj + 1;
    const double sS = lattice_sigma(v, v.si, v.sj);
    const bool inside = x >= s0 && x <= s1 && y >= r0 && y <= r1;
    double rmin2 = 0;
    if (v.prune) {
        const double ex = fmax(fmax(s0 - x, 0.), x - s1), ey = fmax(fmax(r0 - y, 0.), y - r1);
        rmin2 = ex * ex + ey * ey;
    }
    int m_first = m0 > 1 ? m0 : 1;
    while (m_first <= M && v.prune && (m_first * dt) * (m_first * dt) <= rmin2)
        ++m_first;
    if (m_first > M)
        return;

    // 1. Static angular window [w0, w1] and static breakpoints (corners of S + pruned lattice corners).
    double w0 = 0, w1 = 2 * pi;
    if (!inside) {
        const double base = atan2(.5 * (r0 + r1) - y, .5 * (s0 + s1) - x);
        double lo = 1e9, hi = -1e9;
        const double cx[4] = {s0, s1, s1, s0}, cy[4] = {r0, r0, r1, r1};
        for (int k = 0; k < 4; ++k) {
            const double a = remainder(atan2(cy[k] - y, cx[k] - x) - base, 2 * pi);
            lo = fmin(lo, a), hi = fmax(hi, a);
        }
        w0 = base + lo, w1 = base + hi;
    }
    double br[lattice_max_breaks];
    int nb = 0;
    br[nb++] = w0, br[nb++] = w1;
    auto map_angle = [&](double a) {
        double r = fmod(a - w0, 2 * pi);
        if (r < 0)
            r += 2 * pi;
        return w0 + r;
    };
    for (int i = 0; i <= v.nx; ++i)
        for (int j = 0; j <= v.ny; ++j) {
            if (double(i) == x && double(j) == y)
                continue;
            const bool corner_of_S = (i == v.si || i == v.si + 1) && (j == v.sj || j == v.sj + 1);
            if (v.prune && !corner_of_S) {
                const double vx = i - x, vy = j - y, r = sqrt(vx * vx + vy * vy);
                double l_in, l_out;
                if (inside || !lattice_hit(v, x, y, vx / r, vy / r, l_in, l_out) || r >= l_in)
                    continue;
            }
            const double a = map_angle(atan2(j - y, i - x));
            if (a > w0 && a < w1 && nb < lattice_max_breaks)
                br[nb++] = a;
        }
    for (int a = 1; a < nb; ++a) {
        const double key = br[a];
        int b = a - 1;
        while (b >= 0 && br[b] > key)
            br[b + 1] = br[b], --b;
        br[b + 1] = key;
    }

    // Precompute the 4 source-cell edge quadratics (time-independent parts A, B, C0).
    const double ax[4] = {s0, s1, s1, s0}, ay[4] = {r0, r0, r1, r1};
    double eex[4], eey[4], eA[4], eB[4], eC0[4];
    double rmax2 = 0;
    for (int e = 0; e < 4; ++e) {
        const double dx0 = ax[e] - x, dy0 = ay[e] - y;
        rmax2 = fmax(rmax2, dx0 * dx0 + dy0 * dy0);
        const double bx = ax[(e + 1) & 3], by = ay[(e + 1) & 3];
        eex[e] = bx - ax[e], eey[e] = by - ay[e];
        eA[e] = eex[e] * eex[e] + eey[e] * eey[e];
        eB[e] = 2 * (dx0 * eex[e] + dy0 * eey[e]);
        eC0[e] = dx0 * dx0 + dy0 * dy0;
    }
    const int nwords = (W + 63) >> 6;
    unsigned long long split_mask[16];

    // 2. Process each static phi-interval [a, c] independently.
    for (int b = 0; b + 1 < nb; ++b) {
        const double a = br[b], c = br[b + 1];
        if (c - a < 1e-15)
            continue;
        for (int w = 0; w < nwords; ++w)
            split_mask[w] = 0ULL;

        // Find times m where the circle of radius m*dt intersects an edge of S strictly inside (a, c).
        for (int m = m_first; m <= M; ++m) {
            const double t = m * dt;
            if (t * t >= rmax2)
                break;
            double fb[10];
            int nfb = 0;
            for (int e = 0; e < 4; ++e) {
                const double D = eB[e] * eB[e] - 4 * eA[e] * (eC0[e] - t * t);
                if (D < 0)
                    continue;
                const double sq = sqrt(D), inv2A = 0.5 / eA[e];
                const double roots[2] = {(-eB[e] - sq) * inv2A, (-eB[e] + sq) * inv2A};
                for (int k = 0; k < 2; ++k)
                    if (roots[k] > 0 && roots[k] < 1) {
                        const double ang = map_angle(atan2(ay[e] + roots[k] * eey[e] - y, ax[e] + roots[k] * eex[e] - x));
                        if (ang > a && ang < c && nfb < 8)
                            fb[nfb++] = ang;
                    }
            }
            if (nfb == 0)
                continue;
            const int idx = m - m0;
            split_mask[idx >> 6] |= (1ULL << (idx & 63));
            // Sort front breakpoints inside (a, c) and evaluate the split sub-intervals directly at time t.
            for (int i = 1; i < nfb; ++i) {
                const double key = fb[i];
                int j = i - 1;
                while (j >= 0 && fb[j] > key)
                    fb[j + 1] = fb[j], --j;
                fb[j + 1] = key;
            }
            double sub_sum = 0;
            for (int s = 0; s <= nfb; ++s) {
                const double sa = s == 0 ? a : fb[s - 1];
                const double sc = s == nfb ? c : fb[s];
                if (sc - sa < 1e-15)
                    continue;
                for (int k = 0; k < v.phi_n; ++k) {
                    const double sk = v.phi_x[k], phi = sa + (sc - sa) * sk * sk * (3 - 2 * sk), jac = 6 * sk * (1 - sk);
                    const double dx = cos(phi), dy = sin(phi);
                    double l_in, l_out;
                    if (!lattice_hit(v, x, y, dx, dy, l_in, l_out) || t <= l_in)
                        continue;
                    const double T_in = lattice_depth(v, x, y, dx, dy, l_in);
                    sub_sum += (sc - sa) * v.phi_w[k] * jac * lattice_theta(v, l_in, l_out, T_in, sS, t);
                }
            }
            row[idx] += sub_sum;
        }

        // Unsplit times: compute each of the phi_n static rays ONCE and sweep all unsplit times via Chebyshev.
        for (int k = 0; k < v.phi_n; ++k) {
            const double sk = v.phi_x[k], phi = a + (c - a) * sk * sk * (3 - 2 * sk), jac = 6 * sk * (1 - sk);
            const double dx = cos(phi), dy = sin(phi);
            double l_in, l_out;
            if (!lattice_hit(v, x, y, dx, dy, l_in, l_out))
                continue;
            const double T_in = lattice_depth(v, x, y, dx, dy, l_in);
            const double w_phi = (c - a) * v.phi_w[k] * jac;
            lattice_ray_all_times(v, l_in, l_out, T_in, sS, w_phi, m0, m_first, M, dt, exp_tab, split_mask, row);
        }
    }

    const double scale = v.q0 / (2 * pi);
    for (int m = m_first; m <= M; ++m)
        row[m - m0] *= scale;
}
} // namespace uncollided
} // namespace m1pn
