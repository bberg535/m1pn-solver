#pragma once
// Uncollided density of a constant isotropic source switched on at t = 0 in a 2D cell lattice (docs/FCS_LATTICE.md).
//
// Geometry: fields depend on (x, y) only, directions are 3D, Omega = (mu cos(phi), mu sin(phi), cos(theta)),
// mu = sin(theta). sigma_t is constant on the unit cells [i, i+1] x [j, j+1] of [0, nx] x [0, ny]; vacuum outside
// (nothing re-enters). The source q0 is isotropic on one cell S (the lattice: S = [3,4]^2, q0 = 1).
//
//   psi_u(x, Omega, t) = q0/(4 pi) int_0^t chi_S(x - s Omega) exp(-tau(x, x - s Omega)) ds.
//
// With the in-plane distance l = s mu (sin(theta) dtheta ds = dtheta dl) and the in-plane optical depth T(l) along
// the ray x - l (cos phi, sin phi):
//
//   rho_u(x, t) = q0/(2 pi) int dphi int_0^{pi/2} dtheta int_0^{t mu} chi_S(l) exp(-T(l)/mu) dl.
//
// S is convex, so a ray meets it in one segment [l_in, l_out] with sigma_S constant there; T(l_in) = T_in is the
// sum over the cells before S. The l integral is closed-form:
//
//   int_{l_in}^{min(l_out, t mu)} exp(-(T_in + sigma_S (l - l_in))/mu) dl
//     = exp(-T_in/mu) mu/sigma_S (1 - exp(-sigma_S L/mu)),   L = min(l_out, t mu) - l_in > 0.
//
// Quadrature: phi over the window in which S is seen from x, split at the angles of all lattice corners (constant
// cell sequence, smooth l_in, l_out, T_in) and at the angles where the circle of radius t meets the boundary of S
// (where l_in = t or l_out = t); theta split at mu = l_in/t and mu = l_out/t. Gauss-Legendre on every piece; the
// integrand is smooth on each piece, so the rule converges exponentially (tests/uncollided_lattice.cpp).
#include "uncollided.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace m1pn {
namespace uncollided {
struct CellLattice2D {
    int nx = 7, ny = 7;
    std::vector<double> sigma_t; // nx * ny, cell (i, j) at i + nx j
    int source_i = 3, source_j = 3;
    double q0 = 1;
    // Gauss points per piece. With pruning the phi pieces are wide: 24 points give ~2e-6 relative over random (x, t)
    // (tests/uncollided_lattice.cpp --prune-study; the unpruned 8-point rule had 1e-4 at 2x the cost).
    int phi_order = 24, theta_order = 10;
    // Pruning: only lattice corners before the entry into S are breakpoints (the others do not change the cells on
    // [0, l_in]); times before the front reaches S give zero; rays with T_in > 40 are dropped (< e^-40 q0).
    bool prune = true;

    double sigma(int i, int j) const { return sigma_t[std::size_t(i) + std::size_t(nx) * j]; }

    // Ray x + l d against the source cell: [l_in, l_out] (false: no hit or touch only).
    bool hit(double x, double y, double dx, double dy, double &l_in, double &l_out) const {
        double lo = 0, hi = 1e300;
        const double b0[2] = {double(source_i), double(source_j)}, p[2] = {x, y}, d[2] = {dx, dy};
        for (int a = 0; a < 2; ++a) {
            if (std::abs(d[a]) < 1e-300) {
                if (p[a] < b0[a] || p[a] > b0[a] + 1)
                    return false;
                continue;
            }
            double t0 = (b0[a] - p[a]) / d[a], t1 = (b0[a] + 1 - p[a]) / d[a];
            if (t0 > t1)
                std::swap(t0, t1);
            lo = std::max(lo, t0), hi = std::min(hi, t1);
        }
        if (hi <= lo)
            return false;
        l_in = lo, l_out = hi;
        return true;
    }
    // Optical depth along x + l d for l in [0, length] (cell traversal; the path stays inside the lattice).
    double depth(double x, double y, double dx, double dy, double length) const {
        if (length <= 0)
            return 0;
        // Start cell: the cell entered for small l > 0.
        auto cell = [](double p, double d, int n) {
            double f = std::floor(p);
            if (p == f && d < 0)
                f -= 1;
            return std::clamp(int(f), 0, n - 1);
        };
        int i = cell(x, dx, nx), j = cell(y, dy, ny);
        const int si = dx > 0 ? 1 : -1, sj = dy > 0 ? 1 : -1;
        const double inf = 1e300;
        double tx = std::abs(dx) < 1e-300 ? inf : ((dx > 0 ? i + 1 : i) - x) / dx;
        double ty = std::abs(dy) < 1e-300 ? inf : ((dy > 0 ? j + 1 : j) - y) / dy;
        const double ddx = std::abs(dx) < 1e-300 ? inf : 1 / std::abs(dx), ddy = std::abs(dy) < 1e-300 ? inf : 1 / std::abs(dy);
        double l = 0, T = 0;
        for (int guard = 0; guard < 4 * (nx + ny) + 8; ++guard) {
            const double next = std::min({tx, ty, length});
            T += sigma(i, j) * (next - l);
            l = next;
            if (l >= length)
                return T;
            if (tx <= ty) {
                i += si, tx += ddx;
            } else {
                j += sj, ty += ddy;
            }
            if (i < 0 || i >= nx || j < 0 || j >= ny)
                return T; // left the lattice (cannot happen before S for x inside)
        }
        return T;
    }
    // theta integral of one ray: (l_in, l_out, T_in) at time t.
    double theta_integral(double l_in, double l_out, double T_in, double sS, double t, const std::vector<double> &gx,
                          const std::vector<double> &gw) const {
        if (t <= l_in || (prune && T_in > 40))
            return 0;
        const double h = .5 * pi;
        const double a1 = std::asin(std::min(1., l_in / t)), a2 = std::asin(std::min(1., l_out / t));
        double sum = 0;
        auto panel = [&](double lo, double hi, bool growing) {
            for (std::size_t k = 0; k < gx.size(); ++k) {
                const double th = lo + (hi - lo) * gx[k], mu = std::sin(th);
                if (mu <= 0)
                    continue;
                const double L = (growing ? t * mu : l_out) - l_in;
                if (L <= 0)
                    continue;
                const double e = std::exp(-T_in / mu);
                const double f = sS > 0 ? e * mu / sS * -std::expm1(-sS * L / mu) : e * L;
                sum += (hi - lo) * gw[k] * f;
            }
        };
        // exp(-l/mu) has a boundary layer of width ~lo above a lower end lo ~ l/t: panels split geometrically at
        // lo 2^k.
        auto piece = [&](double lo0, double hi0, bool growing) {
            if (hi0 <= lo0)
                return;
            for (double lo = lo0; lo < hi0;) {
                const double hi = lo0 > 0 && 2 * lo < hi0 ? 2 * lo : hi0;
                panel(lo, hi, growing);
                lo = hi;
            }
        };
        piece(a1, a2, true);
        piece(a2, h, false);
        return sum;
    }
    // rho_u(x, t).
    double density(double x, double y, double t) const {
        if (t <= 0)
            return 0;
        static thread_local std::vector<double> gpx, gpw, gtx, gtw;
        static thread_local int gp = -1, gt = -1;
        if (gp != phi_order) {
            auto r = gauss_legendre(phi_order, 0, 1);
            gpx = r.first, gpw = r.second, gp = phi_order;
        }
        if (gt != theta_order) {
            auto r = gauss_legendre(theta_order, 0, 1);
            gtx = r.first, gtw = r.second, gt = theta_order;
        }
        const double s0 = source_i, s1 = source_i + 1, r0 = source_j, r1 = source_j + 1;
        const double sS = sigma(source_i, source_j);
        const bool inside = x >= s0 && x <= s1 && y >= r0 && y <= r1;
        if (prune) {
            const double ex = std::max({s0 - x, 0., x - s1}), ey = std::max({r0 - y, 0., y - r1});
            if (ex * ex + ey * ey >= t * t)
                return 0; // the front has not reached S
        }
        // Angular window [w0, w1] (radians) of directions d that hit S.
        double w0 = 0, w1 = 2 * pi, base = 0;
        if (!inside) {
            base = std::atan2(.5 * (r0 + r1) - y, .5 * (s0 + s1) - x);
            double lo = 1e9, hi = -1e9;
            const double cx[4] = {s0, s1, s1, s0}, cy[4] = {r0, r0, r1, r1};
            for (int k = 0; k < 4; ++k) {
                double a = std::atan2(cy[k] - y, cx[k] - x) - base;
                a = std::remainder(a, 2 * pi);
                lo = std::min(lo, a), hi = std::max(hi, a);
            }
            w0 = base + lo, w1 = base + hi;
        }
        // Breakpoints inside the window.
        static thread_local std::vector<double> br;
        br.clear();
        br.push_back(w0), br.push_back(w1);
        auto add = [&](double a) {
            double r = std::fmod(a - w0, 2 * pi); // map into [w0, w0 + 2 pi)
            if (r < 0)
                r += 2 * pi;
            a = w0 + r;
            if (a > w0 && a < w1)
                br.push_back(a);
        };
        for (int i = 0; i <= nx; ++i)
            for (int j = 0; j <= ny; ++j) {
                if (double(i) == x && double(j) == y)
                    continue;
                const bool corner_of_S = (i == source_i || i == source_i + 1) && (j == source_j || j == source_j + 1);
                if (prune && !corner_of_S) {
                    // Only corners strictly before the entry into S change the cells crossed on [0, l_in].
                    const double vx = i - x, vy = j - y, r = std::hypot(vx, vy);
                    double l_in, l_out;
                    if (inside || !hit(x, y, vx / r, vy / r, l_in, l_out) || r >= l_in)
                        continue;
                }
                add(std::atan2(j - y, i - x));
            }
        // Circle of radius t around x against the four edges of S.
        auto circle = [&](double ax, double ay, double bx, double by) {
            const double ex = bx - ax, ey = by - ay, fx = ax - x, fy = ay - y;
            const double A = ex * ex + ey * ey, B = 2 * (fx * ex + fy * ey), C = fx * fx + fy * fy - t * t;
            const double D = B * B - 4 * A * C;
            if (D < 0)
                return;
            const double sq = std::sqrt(D);
            for (double s : {(-B - sq) / (2 * A), (-B + sq) / (2 * A)})
                if (s > 0 && s < 1)
                    add(std::atan2(ay + s * ey - y, ax + s * ex - x));
        };
        circle(s0, r0, s1, r0), circle(s1, r0, s1, r1), circle(s1, r1, s0, r1), circle(s0, r1, s0, r0);
        std::sort(br.begin(), br.end());
        double sum = 0;
        for (std::size_t b = 0; b + 1 < br.size(); ++b) {
            const double a = br[b], c = br[b + 1];
            if (c - a < 1e-15)
                continue;
            for (std::size_t k = 0; k < gpx.size(); ++k) {
                // phi = a + (c - a) (3 s^2 - 2 s^3): at a breakpoint where l_in = t the phi integrand behaves like
                // (t - l_in)^{3/2}; the substitution makes it smooth in s.
                const double sk = gpx[k], phi = a + (c - a) * sk * sk * (3 - 2 * sk), jac = 6 * sk * (1 - sk);
                const double dx = std::cos(phi), dy = std::sin(phi);
                double l_in, l_out;
                if (!hit(x, y, dx, dy, l_in, l_out))
                    continue;
                const double T_in = depth(x, y, dx, dy, l_in);
                sum += (c - a) * gpw[k] * jac * theta_integral(l_in, l_out, T_in, sS, t, gtx, gtw);
            }
        }
        return q0 / (2 * pi) * sum;
    }
};
// rho_u at fixed points on the uniform time grid t_m = m dt, m = 0..M, with cubic (Catmull-Rom) interpolation in t.
struct LatticeUncollidedTable {
    CellLattice2D lattice;
    double dt = .01;
    int M = 0;
    std::vector<V3> points;
    std::vector<double> table; // points x (M + 1)
    double build_seconds = 0;
    void build(const std::vector<V3> &x, double final_time) {
        const auto start = std::chrono::steady_clock::now();
        points = x;
        M = int(std::ceil(final_time / dt)) + 2;
        table.assign(points.size() * std::size_t(M + 1), 0.);
        parallel_for(int(points.size()), [&](int k) {
            double *row = &table[std::size_t(k) * (M + 1)];
            for (int m = 1; m <= M; ++m)
                row[m] = lattice.density(points[k][0], points[k][1], m * dt);
        });
        build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    bool matches(const std::vector<V3> &x) const {
        if (x.size() != points.size() || x.empty())
            return x.size() == points.size();
        for (std::size_t k : {std::size_t(0), x.size() / 2, x.size() - 1})
            if ((x[k] - points[k]).norm() > 0)
                return false;
        return true;
    }
    double value(std::size_t k, double t) const {
        if (t <= 0)
            return 0;
        const double s = t / dt;
        int i = int(s);
        if (i >= M - 1)
            throw std::runtime_error("LatticeUncollidedTable: time beyond the table");
        const double f = s - i, *row = &table[k * std::size_t(M + 1)];
        // p0..p3 = rho(i-1..i+2); rho(-dt) mirrors as 0 (rho_u = O(t) near zero: use linear extrapolation 2p1 - p2).
        const double p1 = row[i], p2 = row[i + 1], p3 = row[i + 2], p0 = i > 0 ? row[i - 1] : 2 * p1 - p2;
        const double v = p1 + .5 * f * (p2 - p0 + f * (2 * p0 - 5 * p1 + 4 * p2 - p3 + f * (3 * (p1 - p2) + p3 - p0)));
        return std::max(v, 0.);
    }
    void density(double t, const std::vector<V3> &x, std::vector<double> &out) const {
        if (!matches(x))
            throw std::runtime_error("LatticeUncollidedTable: points differ from the tabulated ones");
        out.resize(x.size());
        parallel_for(int(x.size()), [&](int k) { out[k] = value(std::size_t(k), t); });
    }
    // int_0^t rho_u(x_k, s) ds (composite Simpson on the table, interpolated last panel).
    double time_integral(std::size_t k, double t) const {
        const double *row = &table[k * std::size_t(M + 1)];
        const int n = int(std::floor(t / dt));
        double s = 0;
        for (int m = 0; m < n; ++m) {
            const double mid = value(k, (m + .5) * dt);
            s += dt / 6 * (row[m] + 4 * mid + row[m + 1]);
        }
        const double rest = t - n * dt;
        if (rest > 0) {
            const double a = row[n], b = value(k, t), mid = value(k, n * dt + .5 * rest);
            s += rest / 6 * (a + 4 * mid + b);
        }
        return s;
    }
};
// The lattice of Moujaes/Kuzmin/Baeumer (44)-(45) / Schotthoefer/Hauck (12): sigma_t = 10 on the absorbers,
// 1 elsewhere (sigma_s = 1), source cell (3, 3).
template <class Absorber> CellLattice2D lattice_paul(Absorber absorber) {
    CellLattice2D c;
    c.sigma_t.resize(49);
    for (int j = 0; j < 7; ++j)
        for (int i = 0; i < 7; ++i)
            c.sigma_t[i + 7 * j] = absorber(i, j) ? 10. : 1.;
    return c;
}
} // namespace uncollided
} // namespace m1pn
