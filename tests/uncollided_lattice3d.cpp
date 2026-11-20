// Tests of the 3D lattice uncollided density (src/uncollided_lattice3d*.hpp, docs/FCS_LATTICE.md section 3D).
//   uncollided_lattice3d_tests               quick checks (ctest)
//   uncollided_lattice3d_tests --full        adds the volume-integral comparison with absorbers, the quadrature study
//                                            and the balance check (minutes, threads)
#include "benchmark_cases.hpp"
#include "uncollided_lattice3d.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
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
    // 8. Polar rule (adaptive, about the foot point) against fine Cartesian tables: max over t relative to the row
    //    maximum and the time integral. The near point (h = 0.0066 above a face of S) needs 64 Cartesian panels.
    {
        struct Case { V3 x; int panels; };
        const Case cases[] = {{V3(2.3, 4.1, 3.3), 16}, {V3(5.2, 1.4, 2.6), 16}, {V3(3.4, 3.6, 0.8), 16},
                              {V3(1.3, 2.2, 4.7), 16}, {V3(3.5, 3.5, 3.5), 16}, {V3(3.37, 3.61, 4.0066), 64}};
        double emax = 0, eint = 0;
        for (const Case &c : cases) {
            uncollided::Lattice3dUncollidedTable P, F;
            P.lattice = L, P.lattice.polar = true;
            F.lattice = L, F.lattice.outer_order = F.lattice.inner_order = 8;
            F.lattice.inside_panels = F.lattice.outside_panels = c.panels;
            P.build({c.x}, 3.2), F.build({c.x}, 3.2);
            double mx = 0, e = 0, a = 0, b = 0;
            for (int m = 0; m <= F.M; ++m) {
                const double p = P.table[m], f = F.table[m];
                mx = std::max(mx, f), e = std::max(e, std::abs(p - f)), a += p, b += f;
            }
            if (std::getenv("M1PN_TEST_VERBOSE"))
                std::printf("   8. x = (%.4f %.4f %.4f): max %.2e int %.2e\n", c.x[0], c.x[1], c.x[2], e / mx, std::abs(a - b) / b);
            emax = std::max(emax, e / mx), eint = std::max(eint, std::abs(a - b) / b);
        }
        // 8a: the largest deviation (8e-4, x = (3.4, 3.6, 0.8), foot inside the face) comes from the front circle
        // crossing the face boundary: a kink of the partial radial integral in phi at a time-dependent angle that the
        // adaptive rule does not split (docs/FCS_LATTICE.md, polar rule, open point). Time integrals stay at 1e-6.
        check(emax < 1e-3, "8a. polar rule against fine Cartesian tables (max over t, rel. to row max)", emax);
        check(eint < 2e-5, "8b. polar rule against fine Cartesian tables (time integral, relative)", eint);
    }
    // 9. Nodal FCS table (FP64 & FP32) and streaming summary against full point-major table on a 4x4x4 Q1 mesh.
    {
        const int ne = 4, nn = ne + 1, cells = ne * ne * ne, nodes = nn * nn * nn, P = 8, loc = 8;
        const double x0 = 2.0, h = 0.5, T = 3.2; // [2, 4]^3: covers S, scatterers and absorbers
        const double g1[2] = {0.5 - 0.5 / std::sqrt(3.0), 0.5 + 0.5 / std::sqrt(3.0)};
        std::vector<int> conn(cells * loc);
        std::vector<V3> pts(cells * P);
        std::vector<double> fw(cells * P, h * h * h / 8.0), fss(cells * P), fmass(nodes, 0.), shape(loc * P);
        for (int lz = 0; lz < 2; ++lz)
            for (int ly = 0; ly < 2; ++ly)
                for (int lx = 0; lx < 2; ++lx) {
                    const int l = lx + 2 * (ly + 2 * lz);
                    for (int kz = 0; kz < 2; ++kz)
                        for (int ky = 0; ky < 2; ++ky)
                            for (int kx = 0; kx < 2; ++kx) {
                                const int k = kx + 2 * (ky + 2 * kz);
                                const double bx = lx ? g1[kx] : 1 - g1[kx];
                                const double by = ly ? g1[ky] : 1 - g1[ky];
                                const double bz = lz ? g1[kz] : 1 - g1[kz];
                                shape[l * P + k] = bx * by * bz;
                            }
                }
        for (int ez = 0; ez < ne; ++ez)
            for (int ey = 0; ey < ne; ++ey)
                for (int ex = 0; ex < ne; ++ex) {
                    const int e = ex + ne * (ey + ne * ez);
                    for (int lz = 0; lz < 2; ++lz)
                        for (int ly = 0; ly < 2; ++ly)
                            for (int lx = 0; lx < 2; ++lx) {
                                const int l = lx + 2 * (ly + 2 * lz);
                                const int nd = (ex + lx) + nn * ((ey + ly) + nn * (ez + lz));
                                conn[e * loc + l] = nd;
                                for (int k = 0; k < P; ++k)
                                    fmass[nd] += shape[l * P + k] * (h * h * h / 8.0);
                            }
                    for (int kz = 0; kz < 2; ++kz)
                        for (int ky = 0; ky < 2; ++ky)
                            for (int kx = 0; kx < 2; ++kx) {
                                const int k = kx + 2 * (ky + 2 * kz);
                                const V3 p(x0 + (ex + g1[kx]) * h, x0 + (ey + g1[ky]) * h, x0 + (ez + g1[kz]) * h);
                                pts[e * P + k] = p;
                                fss[e * P + k] = cases::lattice3d_sigma_s(p);
                            }
                }
        auto sigma_a_of = [](const V3 &p) { return cases::lattice3d_sigma_a(p); };
        auto owned = [](int) { return true; };

        uncollided::Lattice3dUncollidedTable Ref, N64, N32;
        Ref.lattice = L, Ref.lattice.polar = true;
        Ref.build(pts, T);
        N64.lattice = L, N64.lattice.polar = true, N64.nodal = true, N64.nodal_fp32 = false;
        N64.build_nodal(cells, nodes, loc, P, conn, pts, fw, fss, shape.data(), fmass.data(), owned, sigma_a_of, T);
        N32.lattice = L, N32.lattice.polar = true, N32.nodal = true, N32.nodal_fp32 = true;
        N32.build_nodal(cells, nodes, loc, P, conn, pts, fw, fss, shape.data(), fmass.data(), owned, sigma_a_of, T);

        double e64 = 0, e32 = 0, ref_max = 0;
        for (double t : {0.37, 1.15, 2.43, 3.2}) {
            std::vector<double> rnode(nodes, 0.);
            for (int e = 0; e < cells; ++e)
                for (int k = 0; k < P; ++k) {
                    const double q = fw[e * P + k] * fss[e * P + k] * Ref.value(std::size_t(e * P + k), t);
                    for (int l = 0; l < loc; ++l)
                        rnode[conn[e * loc + l]] += shape[l * P + k] * q;
                }
            for (int i = 0; i < nodes; ++i) {
                rnode[i] /= fmass[i];
                ref_max = std::max(ref_max, rnode[i]);
                e64 = std::max(e64, std::abs(N64.nodal_value(i, t) - rnode[i]));
                e32 = std::max(e32, std::abs(N32.nodal_value(i, t) - rnode[i]));
            }
        }
        double r_mass = 0, r_rate = 0, r_abs = 0;
        for (std::size_t k = 0; k < pts.size(); ++k) {
            const double sa = sigma_a_of(pts[k]), r = Ref.value(k, T);
            r_mass += fw[k] * r, r_rate += fw[k] * sa * r;
            if (sa > 0)
                r_abs += fw[k] * sa * Ref.time_integral(k, T);
        }
        const double esum = std::max({std::abs(N32.summary_mass - r_mass) / r_mass,
                                      std::abs(N32.summary_abs_rate - r_rate) / r_rate,
                                      std::abs(N32.summary_absorbed - r_abs) / r_abs});
        check(e64 / ref_max < 1e-6, "9a. nodal FP64 table against point-major table (rel. to max)", e64 / ref_max);
        check(e32 / ref_max < 2e-6, "9b. nodal FP32 table against point-major table (rel. to max)", e32 / ref_max);
        check(esum < 1e-13, "9c. nodal streaming summary against point-major summary (rel.)", esum);
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
