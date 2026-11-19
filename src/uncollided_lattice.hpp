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
#include "uncollided_lattice_core.hpp"
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
    int phi_order = 24, theta_order = 6; // theta 6 = theta 10 within the phi error (--prune-study)
    // Pruning: only lattice corners before the entry into S are breakpoints (the others do not change the cells on
    // [0, l_in]); times before the front reaches S give zero; rays with T_in > 40 are dropped (< e^-40 q0).
    bool prune = true;

    double sigma(int i, int j) const { return sigma_t[std::size_t(i) + std::size_t(nx) * j]; }
    // Plain view for lattice_density() (Gauss rules on [0, 1] from the caller).
    LatticeView view(const std::vector<double> &px, const std::vector<double> &pw, const std::vector<double> &tx,
                     const std::vector<double> &tw) const {
        if (nx > 8 || ny > 8 || int(sigma_t.size()) != nx * ny)
            throw std::runtime_error("CellLattice2D: at most 8 x 8 cells, sigma_t of size nx ny");
        LatticeView v;
        v.nx = nx, v.ny = ny, v.sigma_t = sigma_t.data(), v.si = source_i, v.sj = source_j, v.q0 = q0;
        v.phi_n = int(px.size()), v.theta_n = int(tx.size());
        v.phi_x = px.data(), v.phi_w = pw.data(), v.theta_x = tx.data(), v.theta_w = tw.data(), v.prune = prune;
        return v;
    }
    // rho_u(x, t) (src/uncollided_lattice_core.hpp).
    double density(double x, double y, double t) const {
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
        return lattice_density(view(gpx, gpw, gtx, gtw), x, y, t);
    }
};
// rho_u at fixed points on the uniform time grid t_m = m dt, m = 0..M, with cubic (Catmull-Rom) interpolation in t.
struct LatticeUncollidedTable {
    CellLattice2D lattice;
    double dt = .01;
    int M = 0;
    // Time window (dynamic AMR, docs/DYNAMIC_AMR.md): rows hold the times m dt for m = m0..M only, m0 from start_time
    // (one time before it, for the cubic stencil). start_time = 0 (m0 = 0) is the full table of [0, final_time].
    int m0 = 0;
    double start_time = 0;
    std::vector<V3> points;
    std::vector<double> table; // points x (M - m0 + 1)
    double build_seconds = 0;
    double final_time = 0; // set by the driver; build() covers [start_time, final_time]
    // Times t_m = m dt, m = 0..M, M from final_time (shared with the device build).
    int times_for(double T) const { return int(std::ceil(T / dt)) + 2; }
    int first_for(double t0) const { return std::max(0, int(std::floor(t0 / dt)) - 1); }
    int width() const { return M - m0 + 1; }
    void build(const std::vector<V3> &x, double final_time) {
        const auto start = std::chrono::steady_clock::now();
        points = x;
        M = times_for(final_time);
        m0 = first_for(start_time);
        const int W = width();
        table.assign(points.size() * std::size_t(W), 0.);
        parallel_for(int(points.size()), [&](int k) {
            double *row = &table[std::size_t(k) * W];
            for (int m = std::max(m0, 1); m <= M; ++m)
                row[m - m0] = lattice.density(points[k][0], points[k][1], m * dt);
        });
        build_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
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
        if (i - (i > 0 ? 1 : 0) < m0)
            throw std::runtime_error("LatticeUncollidedTable: time before the table window");
        const double f = s - i, *row = &table[k * std::size_t(width())] - m0;
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
    // int_0^t rho_u(x_k, s) ds (composite Simpson on the table, interpolated last panel). Needs the full table (m0 = 0).
    double time_integral(std::size_t k, double t) const {
        if (m0 != 0)
            throw std::runtime_error("LatticeUncollidedTable: time_integral from 0 needs the full table");
        const double *row = &table[k * std::size_t(width())];
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
    // int_t0^t1 rho_u(x_k, s) ds inside the window (Simpson per table panel, the same rule as time_integral).
    double window_integral(std::size_t k, double t0, double t1) const {
        double s = 0;
        for (int m = int(std::floor(t0 / dt)); m * dt < t1; ++m) {
            const double a = std::max(t0, m * dt), b = std::min(t1, (m + 1) * dt);
            if (b > a)
                s += (b - a) / 6 * (value(k, a) + 4 * value(k, .5 * (a + b)) + value(k, b));
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
