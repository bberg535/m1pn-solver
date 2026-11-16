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
// theta panel [lo, hi]: sum of w f with f the closed-form l integral.
M1PN_HD inline double lattice_panel(const LatticeView &v, double lo, double hi, bool growing, double l_in, double l_out,
                                    double T_in, double sS, double t) {
    double sum = 0;
    for (int k = 0; k < v.theta_n; ++k) {
        const double th = lo + (hi - lo) * v.theta_x[k], mu = sin(th);
        if (mu <= 0)
            continue;
        const double L = (growing ? t * mu : l_out) - l_in;
        if (L <= 0)
            continue;
        const double e = exp(-T_in / mu);
        const double f = sS > 0 ? e * mu / sS * -expm1(-sS * L / mu) : e * L;
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
} // namespace uncollided
} // namespace m1pn
