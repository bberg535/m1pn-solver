#pragma once
// Uncollided density of the lattice-3d source (constant isotropic q0 on one cell, switched on at t = 0) in the 3D cell
// lattice; formulas and quadrature in src/uncollided_lattice3d_core.hpp, docs/FCS_LATTICE.md (section 3D).
#include "uncollided.hpp"
#include "uncollided_lattice3d_core.hpp"
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace m1pn {
namespace uncollided {
struct CellLattice3D {
    int n[3] = {7, 7, 7};
    std::vector<double> sigma_t; // n0 n1 n2, cell (i, j, k) at i + n0 (j + n1 k)
    int source[3] = {3, 3, 3};
    double q0 = 1;
    // Gauss points per piece and equal sub-panels (tests/uncollided_lattice3d.cpp): without the front in S the rule is
    // exact to 1e-10; the front kinks give ~1e-5 relative (1e-4 right after arrival) with 8 sub-panels, ~5 ms per point.
    int outer_order = 6, inner_order = 6, inside_panels = 8, outside_panels = 8;
    double tau_max = 40;

    Lattice3dView view(const std::vector<double> &ox, const std::vector<double> &ow, const std::vector<double> &ix,
                       const std::vector<double> &iw) const {
        if (n[0] > 8 || n[1] > 8 || n[2] > 8 || int(sigma_t.size()) != n[0] * n[1] * n[2])
            throw std::runtime_error("CellLattice3D: at most 8^3 cells, sigma_t of size n0 n1 n2");
        Lattice3dView v;
        for (int a = 0; a < 3; ++a)
            v.n[a] = n[a], v.s[a] = source[a];
        v.sigma_t = sigma_t.data(), v.q0 = q0;
        v.outer_n = int(ox.size()), v.inner_n = int(ix.size());
        v.outer_x = ox.data(), v.outer_w = ow.data(), v.inner_x = ix.data(), v.inner_w = iw.data();
        v.inside_panels = inside_panels, v.outside_panels = outside_panels, v.tau_max = tau_max;
        if (!(lattice3d_sigma(v, source[0], source[1], source[2]) > 0))
            throw std::runtime_error("CellLattice3D: sigma_t of the source cell must be positive");
        return v;
    }
    template <class F> auto with_view(F &&f) const {
        static thread_local std::vector<double> ox, ow, ix, iw;
        static thread_local int go = -1, gi = -1;
        if (go != outer_order) {
            auto r = gauss_legendre(outer_order, 0, 1);
            ox = r.first, ow = r.second, go = outer_order;
        }
        if (gi != inner_order) {
            auto r = gauss_legendre(inner_order, 0, 1);
            ix = r.first, iw = r.second, gi = inner_order;
        }
        return f(view(ox, ow, ix, iw));
    }
    double density(const V3 &x, double t) const {
        const double p[3] = {x[0], x[1], x[2]};
        return with_view([&](const Lattice3dView &v) { return lattice3d_density(v, p, t); });
    }
    // Distance from x to the source cell (rho_u = 0 for t <= distance).
    double distance(const V3 &x) const {
        double s = 0;
        for (int a = 0; a < 3; ++a) {
            const double e = std::max({source[a] - x[a], 0., x[a] - source[a] - 1});
            s += e * e;
        }
        return std::sqrt(s);
    }
};
// The lattice-3d case (src/benchmark_cases.hpp): sigma_t = sigma_a + sigma_s per cell, source cell (3, 3, 3).
template <class SigmaT> CellLattice3D lattice3d_cells(SigmaT sigma_t) {
    CellLattice3D c;
    c.sigma_t.resize(343);
    for (int k = 0; k < 7; ++k)
        for (int j = 0; j < 7; ++j)
            for (int i = 0; i < 7; ++i)
                c.sigma_t[i + 7 * (j + 7 * k)] = sigma_t(i, j, k);
    return c;
}
// rho_u at fixed points on t_m = m dt, m = 0..M, cubic interpolation in t; same interface as LatticeUncollidedTable.
struct Lattice3dUncollidedTable {
    CellLattice3D lattice;
    double dt = .01;
    int M = 0;
    std::vector<V3> points;
    std::vector<double> table; // points x (M + 1)
    double build_seconds = 0;
    double final_time = 0;
    int times_for(double T) const { return int(std::ceil(T / dt)) + 2; }
    void build(const std::vector<V3> &x, double T) {
        const auto start = std::chrono::steady_clock::now();
        points = x;
        M = times_for(T);
        table.assign(points.size() * std::size_t(M + 1), 0.);
        parallel_for(int(points.size()), [&](int k) {
            if (lattice.distance(points[k]) >= M * dt)
                return; // the front does not reach x within the table
            thread_local std::vector<double> work;
            work.resize(3 * std::size_t(M + 2));
            const double p[3] = {points[k][0], points[k][1], points[k][2]};
            lattice.with_view([&](const Lattice3dView &v) {
                lattice3d_row(v, p, dt, M, work.data(), &table[std::size_t(k) * (M + 1)]);
                return 0;
            });
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
            throw std::runtime_error("Lattice3dUncollidedTable: time beyond the table");
        const double f = s - i, *row = &table[k * std::size_t(M + 1)];
        const double p1 = row[i], p2 = row[i + 1], p3 = row[i + 2], p0 = i > 0 ? row[i - 1] : 2 * p1 - p2;
        const double v = p1 + .5 * f * (p2 - p0 + f * (2 * p0 - 5 * p1 + 4 * p2 - p3 + f * (3 * (p1 - p2) + p3 - p0)));
        return std::max(v, 0.);
    }
    void density(double t, const std::vector<V3> &x, std::vector<double> &out) const {
        if (!matches(x))
            throw std::runtime_error("Lattice3dUncollidedTable: points differ from the tabulated ones");
        out.resize(x.size());
        parallel_for(int(x.size()), [&](int k) { out[k] = value(std::size_t(k), t); });
    }
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
} // namespace uncollided
} // namespace m1pn
