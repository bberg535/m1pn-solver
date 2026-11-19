// Study (not a ctest): cost and accuracy of the 3D lattice uncollided time table (src/uncollided_lattice3d.hpp) as a
// function of the quadrature (sub-panels against the front kinks, Gauss points per piece). Reference: 16 panels,
// 8 points. Points are random in [0,7]^3; errors over all table times t_m relative to the global maximum, plus the
// relative error of sum_k rho_u(x_k, T) and of sum_k int_0^T rho_u (proxies for mass and absorbed mass).
//   m1pn_fcs3d_panel_study [POINTS]
#include "benchmark_cases.hpp"
#include "uncollided_lattice3d.hpp"
#include <cstdio>
#include <random>
using namespace m1pn;
using uncollided::CellLattice3D;
int main(int argc, char **argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 1024;
    const double T = 3.2;
    CellLattice3D L = uncollided::lattice3d_cells([](int i, int j, int k) {
        const V3 x(i + .5, j + .5, k + .5);
        return cases::lattice3d_sigma_a(x) + cases::lattice3d_sigma_s(x);
    });
    std::vector<V3> pts;
    std::mt19937 r(11);
    std::uniform_real_distribution<double> u(.02, 6.98);
    while (int(pts.size()) < n) {
        const V3 x(u(r), u(r), u(r));
        if (L.distance(x) < T) // points the front reaches (the others cost nothing)
            pts.push_back(x);
    }
    auto build = [&](int order, int panels, int inside) {
        uncollided::Lattice3dUncollidedTable t;
        t.lattice = L;
        t.lattice.outer_order = t.lattice.inner_order = order;
        t.lattice.outside_panels = panels, t.lattice.inside_panels = inside;
        t.build(pts, T);
        return t;
    };
    const auto ref = build(8, 16, 16);
    double vmax = 0, mref = 0, iref = 0;
    for (std::size_t k = 0; k < pts.size(); ++k) {
        for (int m = 0; m <= ref.M; ++m)
            vmax = std::max(vmax, ref.table[k * (ref.M + 1) + m]);
        mref += ref.value(k, T), iref += ref.time_integral(k, T);
    }
    std::printf("points %d (reached by the front), reference 8 points / 16 panels: %.2f s, %.3f ms per point-core\n",
                n, ref.build_seconds, 0.);
    std::printf("order panels inside | seconds  speedup | max err/max   err(rho(T) sum)  err(int rho sum)\n");
    for (int order : {6, 4})
        for (int panels : {8, 4, 2, 1}) {
            const int inside = std::max(panels, 4);
            const auto t = build(order, panels, inside);
            double e = 0, ms = 0, is = 0;
            for (std::size_t k = 0; k < pts.size(); ++k) {
                for (int m = 0; m <= t.M; ++m)
                    e = std::max(e, std::abs(t.table[k * (t.M + 1) + m] - ref.table[k * (ref.M + 1) + m]));
                ms += t.value(k, T), is += t.time_integral(k, T);
            }
            std::printf("%5d %6d %6d | %7.2f %8.1f | %.2e      %+.2e        %+.2e\n", order, panels, inside,
                        t.build_seconds, ref.build_seconds / t.build_seconds, e / vmax, ms / mref - 1, is / iref - 1);
            std::fflush(stdout);
        }
    return 0;
}
