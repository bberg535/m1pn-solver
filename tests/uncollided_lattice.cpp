// Tests of the lattice uncollided density (src/uncollided_lattice.hpp, docs/FCS_LATTICE.md).
//   m1pn_uncollided_lattice_tests          quick checks (ctest)
//   m1pn_uncollided_lattice_tests --full   adds the area-integral comparison and the balance check (minutes, threads)
#include "benchmark_cases.hpp"
#include "uncollided_lattice.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <array>
#include <cstdlib>
#include <random>
using namespace m1pn;
using uncollided::CellLattice2D;

static int failures = 0;
static void check(bool ok, const char *what, double value) {
    std::printf("%-58s %-4s %.3e\n", what, ok ? "ok" : "FAIL", value);
    if (!ok)
        ++failures;
}
static CellLattice2D paul() { return uncollided::lattice_paul(cases::lattice_paul_absorber_cell); }

// Independent formulation: integral over the source cell of the kernel of an infinite line source at in-plane
// distance r, K = 1/(2 pi r) int_0^{acos(r/t)} exp(-tau/cos a) da, tau the in-plane optical depth x -> y.
// tau by fine stepping (independent of depth()).
static double tau_stepping(const CellLattice2D &c, double x, double y, double ex, double ey, int steps) {
    double s = 0;
    for (int k = 0; k < steps; ++k) {
        const double f = (k + .5) / steps, px = x + f * (ex - x), py = y + f * (ey - y);
        s += c.sigma(std::clamp(int(std::floor(px)), 0, c.nx - 1), std::clamp(int(std::floor(py)), 0, c.ny - 1));
    }
    return s / steps * std::hypot(ex - x, ey - y);
}
static double area_reference(const CellLattice2D &c, double x, double y, double t, int sub, int steps) {
    const auto g = uncollided::gauss_legendre(3, 0, 1), ga = uncollided::gauss_legendre(24, 0, 1);
    double sum = 0;
    const double h = 1. / sub;
    for (int a = 0; a < sub; ++a)
        for (int b = 0; b < sub; ++b)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    const double py = c.source_j + (b + g.first[j]) * h, px = c.source_i + (a + g.first[i]) * h;
                    const double r = std::hypot(px - x, py - y);
                    if (r >= t || r == 0)
                        continue;
                    const double tau = tau_stepping(c, x, y, px, py, steps), amax = std::acos(r / t);
                    double k = 0;
                    for (std::size_t q = 0; q < ga.first.size(); ++q)
                        k += amax * ga.second[q] * std::exp(-tau / std::cos(amax * ga.first[q]));
                    sum += h * h * g.second[i] * g.second[j] * c.q0 * k / (2 * pi * r);
                }
    return sum;
}
// int over [0,7]^2 of f(x, y) sigma-weighted, by Gauss on sub x sub pieces of every unit cell.
template <class F> static double domain_integral(const CellLattice2D &c, int sub, int order, F f) {
    const auto g = uncollided::gauss_legendre(order, 0, 1);
    const int per = sub * order, n = c.nx * per;
    std::vector<double> part(std::size_t(n), 0.);
    parallel_for(n, [&](int row) {
        const int cj = row / per, r = row % per, bj = r / order, qj = r % order;
        const double y = cj + (bj + g.first[qj]) / sub, wy = g.second[qj] / sub;
        double s = 0;
        for (int ci = 0; ci < c.nx; ++ci)
            for (int bi = 0; bi < sub; ++bi)
                for (int qi = 0; qi < order; ++qi) {
                    const double x = ci + (bi + g.first[qi]) / sub, wx = g.second[qi] / sub;
                    s += wx * wy * f(x, y, ci, cj);
                }
        part[row] = s;
    });
    double s = 0;
    for (double v : part)
        s += v;
    return s;
}

int main(int argc, char **argv) {
    const bool full = argc > 1 && !std::strcmp(argv[1], "--full");
    CellLattice2D c = paul();
    if (argc == 2 && !std::strcmp(argv[1], "--prune-study")) { // pruned breakpoints: phi order vs accuracy/time
        std::mt19937 rng(11);
        std::uniform_real_distribution<double> ux(0, 7), ut(0, 3.2);
        std::vector<std::array<double, 3>> xs(2000);
        for (auto &p : xs)
            p = {ux(rng), ux(rng), ut(rng)};
        CellLattice2D ref = c;
        ref.prune = false, ref.phi_order = 16, ref.theta_order = 20;
        std::vector<double> rv(xs.size());
        for (std::size_t k = 0; k < xs.size(); ++k)
            rv[k] = ref.density(xs[k][0], xs[k][1], xs[k][2]);
        for (bool prune : {false, true})
            for (int po : {8, 12, 16, 24})
                for (int to : {6, 8, 10}) {
                if (!prune && to != 10)
                    continue;
                CellLattice2D v = c;
                v.prune = prune, v.phi_order = po, v.theta_order = to;
                double worst = 0;
                auto s0 = std::chrono::steady_clock::now();
                for (std::size_t k = 0; k < xs.size(); ++k)
                    worst = std::max(worst, std::abs(v.density(xs[k][0], xs[k][1], xs[k][2]) - rv[k]) / (rv[k] + 1e-4));
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
                std::printf("prune %d phi %2d theta %2d: max rel err %.2e  time %.3f s\n", int(prune), po, to, worst, sec);
            }
        return 0;
    }
    if (argc == 5 && !std::strcmp(argv[1], "--orders")) { // quadrature study at one point: --orders x y t
        const double x = std::atof(argv[2]), y = std::atof(argv[3]), t = std::atof(argv[4]);
        for (int a : {8, 16, 32, 64})
            for (int b : {10, 20, 40, 80}) {
                c.phi_order = a, c.theta_order = b;
                std::printf("phi %3d theta %3d  %.15e\n", a, b, c.density(x, y, t));
            }
        return 0;
    }
    // 1. Centre of the source cell, t <= 1/2: the shell stays in S (sigma_t = 1): rho_u = 1 - exp(-t).
    for (double t : {.1, .3, .5}) {
        const double err = std::abs(c.density(3.5, 3.5, t) - (1 - std::exp(-t)));
        char s[64];
        std::snprintf(s, sizeof s, "centre, t = %.1f: |rho - (1 - e^-t)|", t);
        check(err < 1e-12, s, err);
    }
    // 2. Quadrature convergence at points in front of / behind absorbers, in S, on a lattice line.
    const double pts[][2] = {{3.5, 3.5}, {3.2, 3.9}, {2.5, 3.5}, {1.5, 2.7}, {0.3, 0.4}, {5.5, 1.5}, {3.0, 5.25}, {6.1, 3.33}};
    double worst = 0;
    for (auto &p : pts)
        for (double t : {.7, 1.9, 3.2}) {
            CellLattice2D lo = c, hi = c;
            hi.phi_order = 32, hi.theta_order = 20;
            const double a = lo.density(p[0], p[1], t), b = hi.density(p[0], p[1], t);
            const double r = std::abs(a - b) / (b + 1e-5); // relative, absolute floor 1e-14
            if (r > 1e-7)
                std::printf("   x = (%.2f, %.2f), t = %.1f: %.12e vs %.12e (%.1e)\n", p[0], p[1], t, a, b, r);
            worst = std::max(worst, r);
        }
    check(worst < 1e-5, "default vs phi 32 / theta 20 (rel., floor 1e-14)", worst);
    // 2b. Pruning (corners behind the entry, early zero, T_in cutoff) against the full breakpoint set.
    {
        std::mt19937 rng(3);
        std::uniform_real_distribution<double> ux(0, 7), ut(0, 3.2);
        CellLattice2D full = c;
        full.prune = false, full.phi_order = 16, full.theta_order = 20;
        double worst = 0, t_full = 0, t_prune = 0;
        for (int k = 0; k < 400; ++k) {
            const double x = ux(rng), y = ux(rng), t = ut(rng);
            auto s0 = std::chrono::steady_clock::now();
            const double a = full.density(x, y, t);
            auto s1 = std::chrono::steady_clock::now();
            const double b = c.density(x, y, t);
            auto s2 = std::chrono::steady_clock::now();
            t_full += std::chrono::duration<double>(s1 - s0).count(), t_prune += std::chrono::duration<double>(s2 - s1).count();
            worst = std::max(worst, std::abs(a - b) / (a + 1e-4)); // relative, absolute floor 1e-16
        }
        std::printf("   400 random (x, t): full %.3f s, pruned %.3f s (%.1fx)\n", t_full, t_prune, t_full / t_prune);
        check(worst < 1e-5, "pruned (phi 24) vs all corners (phi 16) (rel., floor 1e-16)", worst);
    }
    // 3. Zero before arrival, nonnegative.
    check(c.density(0.2, 0.2, 2.5) == 0, "corner (0.2, 0.2) before arrival (dist 3.96)", c.density(0.2, 0.2, 2.5));
    // 4. Time table: cubic interpolation against direct evaluation.
    {
        uncollided::LatticeUncollidedTable tab;
        tab.lattice = c;
        std::vector<V3> x;
        for (auto &p : pts)
            x.push_back(V3(p[0], p[1], 0));
        tab.build(x, 3.2);
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> u(0, 3.2);
        double e = 0;
        for (int k = 0; k < 200; ++k) {
            const double t = u(rng);
            for (std::size_t i = 0; i < x.size(); ++i) {
                const double ex = c.density(x[i][0], x[i][1], t);
                e = std::max(e, std::abs(tab.value(i, t) - ex));
            }
        }
        check(e < 2e-5, "table (dt 0.01, cubic) vs direct, max abs", e);
        // time integral at the centre: int_0^0.5 (1 - e^-s) ds = 0.5 - (1 - e^-0.5)
        const double ti = tab.time_integral(0, .5), ref = .5 - (1 - std::exp(-.5));
        check(std::abs(ti - ref) < 1e-6, "time integral at the centre, t = 0.5", std::abs(ti - ref));
    }
    if (full) {
        // 5. Independent area-integral formulation (outside S; 1/r kernel smooth there).
        double e = 0;
        for (auto &p : pts) {
            if (p[0] >= 3 && p[0] <= 4 && p[1] >= 3 && p[1] <= 4)
                continue;
            for (double t : {1.9, 3.2}) {
                const double a = c.density(p[0], p[1], t);
                if (a == 0 && area_reference(c, p[0], p[1], t, 8, 100) == 0)
                    continue;
                // tau by stepping is first order at the sigma jumps: 4e3 and 4e4 steps show the reference converging.
                const double b1 = area_reference(c, p[0], p[1], t, 64, 4000), b2 = area_reference(c, p[0], p[1], t, 64, 40000);
                std::printf("   x = (%.2f, %.2f), t = %.1f: angular %.10e  area(4e3) rel %.2e  area(4e4) rel %.2e\n", p[0],
                            p[1], t, a, std::abs(a - b1) / b1, std::abs(a - b2) / b2);
                if (b2 > 1e-8)
                    e = std::max(e, std::abs(a - b2) / b2);
            }
        }
        check(e < 2e-4, "angular vs area integral, 4e4 steps (relative)", e);
        // 6. Balance without outflow (t < 3): dM_u/dt = q0 |S| - int sigma_t rho_u.
        // Space quadrature: rho_u has kinks along shadow lines, so the residual falls algebraically with sub.
        for (double t : {1., 2.5}) {
            const double d = 1e-3;
            double last = 0;
            for (int sub : {8, 16, 32}) {
                auto mass = [&](double s) {
                    return domain_integral(c, sub, 4, [&](double x, double y, int, int) { return c.density(x, y, s); });
                };
                const double dm = (mass(t + d) - mass(t - d)) / (2 * d);
                const double loss = domain_integral(
                    c, sub, 4, [&](double x, double y, int i, int j) { return c.sigma(i, j) * c.density(x, y, t); });
                last = dm - (c.q0 - loss);
                std::printf("   t = %.1f, %2d x %2d x 4^2 points per cell: dM/dt %.8f  1 - int sigma rho %.8f  residual %.2e\n",
                            t, sub, sub, dm, c.q0 - loss, last);
            }
            char s[80];
            std::snprintf(s, sizeof s, "balance t = %.1f (finest quadrature)", t);
            check(std::abs(last) < 5e-5, s, std::abs(last));
        }
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "passed", failures);
    return failures ? 1 : 0;
}
