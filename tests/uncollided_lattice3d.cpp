// Tests of the 3D lattice uncollided density (src/uncollided_lattice3d*.hpp, docs/FCS_LATTICE.md section 3D).
//   uncollided_lattice3d_tests               quick checks (ctest)
//   uncollided_lattice3d_tests --full        adds the volume-integral comparison with absorbers, the quadrature study
//                                            and the balance check (minutes, threads)
#include "benchmark_cases.hpp"
#include "uncollided_lattice3d.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>
using namespace m1pn;
using uncollided::CellLattice3D;

static int failures = 0;
static void check(bool ok, const char *what, double value) {
    std::printf("%-62s %-4s %.3e\n", what, ok ? "ok" : "FAIL", value);
    std::fflush(stdout);
    if (!ok)
        ++failures;
}
static CellLattice3D lattice() {
    return uncollided::lattice3d_cells([](int i, int j, int k) {
        const V3 x(i + .5, j + .5, k + .5);
        return cases::lattice3d_sigma_a(x) + cases::lattice3d_sigma_s(x);
    });
}
static CellLattice3D homogeneous(double sigma) {
    CellLattice3D c;
    c.sigma_t.assign(343, sigma);
    return c;
}
// Independent formulation: volume integral over S of q0/(4 pi) exp(-tau(x, y)) / |x - y|^2 for |x - y| < t, tau by
// fine stepping along the segment (independent of the voxel traversal). Composite Gauss on sub^3 subcubes.
static double tau_stepping(const CellLattice3D &c, const V3 &x, const V3 &y, int steps) {
    double s = 0;
    for (int k = 0; k < steps; ++k) {
        const V3 p = x + (k + .5) / steps * (y - x);
        const int i = std::clamp(int(std::floor(p[0])), 0, 6), j = std::clamp(int(std::floor(p[1])), 0, 6),
                  l = std::clamp(int(std::floor(p[2])), 0, 6);
        s += c.sigma_t[i + 7 * (j + 7 * l)];
    }
    return s / steps * (y - x).norm();
}
// tau exactly: crossings of the segment with all planes x_a = integer, sorted; sigma of the cell at each midpoint.
static double tau_exact(const CellLattice3D &c, const V3 &x, const V3 &y) {
    std::vector<double> f = {0., 1.};
    for (int a = 0; a < 3; ++a) {
        const double d = y[a] - x[a];
        if (std::abs(d) < 1e-300)
            continue;
        for (int i = 0; i <= 7; ++i) {
            const double s = (i - x[a]) / d;
            if (s > 0 && s < 1)
                f.push_back(s);
        }
    }
    std::sort(f.begin(), f.end());
    double tau = 0;
    for (std::size_t k = 0; k + 1 < f.size(); ++k) {
        const V3 m = x + .5 * (f[k] + f[k + 1]) * (y - x);
        const int i = std::clamp(int(std::floor(m[0])), 0, 6), j = std::clamp(int(std::floor(m[1])), 0, 6),
                  l = std::clamp(int(std::floor(m[2])), 0, 6);
        tau += c.sigma_t[i + 7 * (j + 7 * l)] * (f[k + 1] - f[k]);
    }
    return tau * (y - x).norm();
}
static double volume_reference(const CellLattice3D &c, const V3 &x, double t, int sub, int order, int steps) {
    const auto g = uncollided::gauss_legendre(order, 0, 1);
    const double h = 1. / sub;
    std::vector<double> part(sub, 0.);
    parallel_for(sub, [&](int a) {
        double s = 0;
        for (int b = 0; b < sub; ++b)
            for (int e = 0; e < sub; ++e)
                for (std::size_t i = 0; i < g.first.size(); ++i)
                    for (std::size_t j = 0; j < g.first.size(); ++j)
                        for (std::size_t k = 0; k < g.first.size(); ++k) {
                            const V3 y(c.source[0] + (a + g.first[i]) * h, c.source[1] + (b + g.first[j]) * h,
                                       c.source[2] + (e + g.first[k]) * h);
                            const double r = (y - x).norm();
                            if (r >= t)
                                continue;
                            const double tau = steps < 0 ? tau_exact(c, x, y) : (steps > 0 ? tau_stepping(c, x, y, steps) : c.sigma_t[0] * r);
                            s += g.second[i] * g.second[j] * g.second[k] * h * h * h * std::exp(-tau) / (r * r);
                        }
        part[a] = s;
    });
    double s = 0;
    for (double p : part)
        s += p;
    return c.q0 / (4 * 3.14159265358979323846) * s;
}

int main(int argc, char **argv) {
    const bool full = argc > 1 && std::strcmp(argv[1], "--full") == 0;
    const CellLattice3D L = lattice();
    // 1. Source centre, t <= 0.5 (front inside S): rho_u = q0 (1 - exp(-t)) / sigma_S with sigma_S = 1.
    {
        double e = 0;
        for (double t : {.1, .3, .5})
            e = std::max(e, std::abs(L.density(V3(3.5, 3.5, 3.5), t) - (1 - std::exp(-t))));
        check(e < 1e-6, "1. source centre, t <= 0.5: rho_u = 1 - exp(-t)", e);
    }
    // 2. Homogeneous medium sigma = 1 (no kinks), points outside S, t covering all of S: smooth volume integral.
    {
        const CellLattice3D H = homogeneous(1.);
        double e = 0;
        for (const V3 &x : {V3(1.3, 2.2, 4.7), V3(5.6, 3.4, 3.5), V3(3.5, 5.5, 1.2), V3(2.7, 2.6, 2.4)}) {
            const double a = H.density(x, 6.), b = volume_reference(H, x, 6., 4, 16, 0);
            e = std::max(e, std::abs(a - b) / b);
        }
        check(e < 1e-10, "2. homogeneous sigma = 1, outside S, against volume integral", e);
    }
    // 3. Zero before the front reaches S; positive after.
    {
        const V3 x(0.7, 0.6, 3.5);
        const double d = L.distance(x);
        check(L.density(x, .99 * d) == 0 && L.density(x, d + .5) > 0, "3. zero before arrival, positive after",
              L.density(x, d + .5));
    }
    // 4. Time table (exact per ray) against direct evaluation at the table times and between them.
    {
        uncollided::Lattice3dUncollidedTable T;
        T.lattice = L;
        const std::vector<V3> pts = {V3(3.5, 3.5, 3.5), V3(2.3, 4.1, 3.3), V3(5.2, 1.4, 2.6), V3(3.4, 3.6, 0.8)};
        T.build(pts, 3.2);
        double e = 0, ei = 0;
        for (std::size_t k = 0; k < pts.size(); ++k)
            for (double t : {.5, 1., 2., 3.}) {
                e = std::max(e, std::abs(T.value(k, t) - L.density(pts[k], t)));
                ei = std::max(ei, std::abs(T.value(k, t + .005) - L.density(pts[k], t + .005)));
            }
        check(e < 1e-13, "4a. time table = direct evaluation at t_m", e);
        check(ei < 2e-5, "4b. cubic interpolation between t_m", ei);
    }
    // 5. Quadrature: default against 2x the Gauss points per piece at random points and times.
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> u(.05, 6.95), ut(.5, 3.2);
        CellLattice3D F = L;
        F.outer_order = 8, F.inner_order = 8, F.inside_panels = 16, F.outside_panels = 16;
        double e = 0, scale = 0;
        const int n = full ? 200 : 12;
        std::vector<double> err(n), val(n);
        parallel_for(n, [&](int k) {
            std::mt19937 r(7 + k);
            const V3 x(u(r), u(r), u(r));
            const double t = ut(r), a = L.density(x, t), b = F.density(x, t);
            err[k] = std::abs(a - b), val[k] = b;
        });
        for (int k = 0; k < n; ++k)
            e = std::max(e, err[k]), scale = std::max(scale, val[k]);
        check(e / scale < 2e-5, "5. default rule against 2x panels, 8 points (rel. to max)", e / scale);
    }
    if (full) {
        // 6. Volume integral with the lattice (kinks): the reference converges with sub and steps.
        // t = 6: all of S inside the front (exact tau, the reference converges with sub); t = 3: front through S
        // (the reference has a jump there and converges slowly).
        for (double t : {6., 3.})
            for (const V3 &x : {V3(1.4, 2.3, 4.6), V3(5.5, 3.2, 3.7), V3(2.6, 3.4, 5.6)}) {
                const double a = L.density(x, t);
                for (int sub : {8, 16, 32, 64}) {
                    const double b = volume_reference(L, x, t, sub, 4, -1);
                    std::printf("   6. x = (%.1f %.1f %.1f) t = %.0f: rays %.10e volume(sub %2d) %.10e rel %.2e\n", x[0],
                                x[1], x[2], t, a, sub, b, std::abs(a - b) / b);
                }
            }
        // 7. Timing per point (table to T = 3.2).
        {
            uncollided::Lattice3dUncollidedTable T;
            T.lattice = L;
            std::vector<V3> pts;
            std::mt19937 r(3);
            std::uniform_real_distribution<double> u(.05, 6.95);
            for (int k = 0; k < 4096; ++k)
                pts.emplace_back(u(r), u(r), u(r));
            T.build(pts, 3.2);
            std::printf("   7. table for 4096 random points, T = 3.2: %.2f s\n", T.build_seconds);
        }
    }
    std::printf(failures ? "FAILED (%d)\n" : "PASSED\n", failures);
    return failures ? 1 : 0;
}
