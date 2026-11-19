// Tests of the dynamic AMR transfer (experiment/dynamic-amr, src/amr.hpp, docs/DYNAMIC_AMR.md):
// 1. Bernstein subdivision: nonnegative, rows sum to one, exact for polynomials (p = 1..3).
// 2. 1D transfer between level patterns with broken lines: convex rows, sum m_B T u = sum m_A u, exact prolongation.
// 3. 2D transfer between ElementWeno meshes (graded, broken lines, mirror side): density integral with the scheme's own
//    lumped masses conserved to rounding, every node inside the M1 cone, refine-only transfer reproduces the function.
// 4. Indicator and level update: a front is refined, a flat field coarsened, levels graded 2:1.
#include "amr.hpp"
#include "element_weno.hpp"
#include "uncollided_lattice.hpp"
#include "benchmark_cases.hpp"
#include <cmath>
#include <iostream>
#include <random>

using namespace m1pn;
namespace {
int failures = 0;
void check(bool ok, const std::string &what, double value = 0) {
    std::cout << (ok ? "ok   " : "FAIL ") << what << "  [" << value << "]\n";
    if (!ok)
        ++failures;
}
double bern(int p, int j, double x) {
    double c = 1;
    for (int k = 1; k <= j; ++k)
        c = c * (p - j + k) / k;
    return c * std::pow(x, j) * std::pow(1 - x, p - j);
}
void test_subdivision() {
    double worst = 0, neg = 0, rows = 0;
    for (int p = 1; p <= 3; ++p)
        for (auto [s, t] : {std::pair{0., .5}, {.5, 1.}, {.25, .375}, {0., 1.}}) {
            const Mat S = amr::subdivision(p, s, t);
            neg = std::min(neg, S.minCoeff());
            for (int k = 0; k <= p; ++k)
                rows = std::max(rows, std::abs(S.row(k).sum() - 1));
            // f = sum_j b_j B_j on [0, 1] equals sum_k (S b)_k B_k((x - s) / (t - s)) on [s, t].
            Vec b = Vec::Random(p + 1), c = S * b;
            for (double y : {0., .3, .7, 1.}) {
                const double x = s + (t - s) * y;
                double f = 0, g = 0;
                for (int j = 0; j <= p; ++j)
                    f += b[j] * bern(p, j, x), g += c[j] * bern(p, j, y);
                worst = std::max(worst, std::abs(f - g));
            }
        }
    check(neg >= 0, "subdivision nonnegative", neg);
    check(rows < 1e-15, "subdivision rows sum to one", rows);
    check(worst < 1e-14, "subdivision exact for polynomials", worst);
}
amr::Axis axis(int cells, std::vector<int> level, std::vector<double> breaks, int p) {
    amr::Axis a;
    for (int b = 0; b <= cells; ++b)
        a.base.push_back(2. * b / cells);
    a.level = std::move(level), a.breaks = std::move(breaks), a.p = p;
    return a;
}
void test_transfer_1d() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0, 1);
    double mass = 0, rowsum = 0, neg = 0, exact = 0;
    for (int p = 1; p <= 3; ++p) {
        const amr::Axis A = axis(8, {0, 1, 2, 2, 1, 0, 0, 1}, {1.0}, p), B = axis(8, {1, 2, 1, 0, 0, 1, 2, 2}, {1.0}, p);
        const amr::Line a = A.line(), b = B.line();
        const amr::Rows T = amr::transfer(a, b);
        const auto ma = a.lumped(), mb = b.lumped();
        std::vector<double> x(a.nodes());
        for (auto &v : x) v = u(rng);
        double sa = 0, sb = 0;
        for (int i = 0; i < a.nodes(); ++i) sa += ma[i] * x[i];
        for (int I = 0; I < T.rows(); ++I) {
            double y = 0, s = 0;
            for (int k = T.start[I]; k < T.start[I + 1]; ++k)
                y += T.w[k] * x[T.col[k]], s += T.w[k], neg = std::min(neg, T.w[k]);
            rowsum = std::max(rowsum, std::abs(s - 1));
            sb += mb[I] * y;
        }
        mass = std::max(mass, std::abs(sa - sb) / sa);
        // Refinement of A (all levels + 1): exact prolongation, the coarsening back is a convex average.
        amr::Axis F = A;
        for (int &l : F.level) ++l;
        const amr::Line f = F.line();
        const amr::Rows P = amr::transfer(a, f);
        const auto offa = a.offsets(), offf = f.offsets();
        for (int e = 0; e < f.elements(); ++e)
            for (double y : {.2, .6}) {
                const double xx = f.planes[e] + y * (f.planes[e + 1] - f.planes[e]);
                const int ea = a.element_of(xx);
                const double ya = (xx - a.planes[ea]) / (a.planes[ea + 1] - a.planes[ea]);
                double va = 0, vf = 0;
                for (int j = 0; j <= p; ++j) {
                    va += x[a.node(ea, j, offa)] * bern(p, j, ya);
                    const int n = f.node(e, j, offf);
                    double c = 0;
                    for (int k = P.start[n]; k < P.start[n + 1]; ++k) c += P.w[k] * x[P.col[k]];
                    vf += c * bern(p, j, y);
                }
                exact = std::max(exact, std::abs(va - vf));
            }
    }
    check(mass < 1e-14, "1D transfer conserves sum m u", mass);
    check(rowsum < 1e-14 && neg >= 0, "1D transfer rows convex", rowsum);
    check(exact < 1e-14, "1D refinement is exact prolongation", exact);
}
ElementWenoOptions mesh_options(const std::array<amr::Axis, 3> &ax) {
    ElementWenoOptions o;
    o.dimension = 2, o.degree = 1, o.periodic = false, o.ssprk43 = true, o.reduce_angular = true;
    for (int d = 0; d < 2; ++d) {
        const amr::Line l = ax[d].line();
        o.planes[d] = l.planes, o.breaks[d] = l.breaks;
    }
    o.elements = int(o.planes[0].size()) - 1;
    o.elements_per_axis = {o.elements, int(o.planes[1].size()) - 1, 0};
    o.length = 2;
    o.mirror[0] = true;
    return o;
}
void test_transfer_2d() {
    Angular pn(5);
    std::array<amr::Axis, 3> A{axis(8, {0, 1, 2, 2, 1, 0, 0, 0}, {1.0}, 1), axis(8, {2, 2, 1, 0, 0, 1, 2, 1}, {1.0}, 1), {}};
    std::array<amr::Axis, 3> B{axis(8, {1, 2, 1, 0, 0, 1, 1, 0}, {1.0}, 1), axis(8, {1, 1, 0, 0, 1, 2, 2, 2}, {1.0}, 1), {}};
    ElementWeno sa(pn, mesh_options(A)), sb(pn, mesh_options(B));
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> u(0, 1);
    Field a = Field::Zero(sa.nodes, sa.moments);
    for (int i = 0; i < sa.nodes; ++i) {
        Four v;
        const double rho = std::exp(-10 * u(rng)), f = .999 * u(rng);
        V3 dir(u(rng) - .5, u(rng) - .5, u(rng) - .5);
        dir *= f * rho / dir.norm();
        v << rho, dir[0], dir[1], dir[2];
        Vec full = Vec::Zero(sa.moments);
        set_low(full, v);
        for (int c = 4; c < sa.moments; ++c) full[c] = 1e-2 * rho * (u(rng) - .5);
        a.row(i) = full.transpose();
    }
    std::array<amr::Rows, 3> T;
    for (int d = 0; d < 2; ++d) T[d] = amr::transfer(A[d].line(), B[d].line());
    const Field b = amr::apply(a, sa.nodes_per_axis, 2, T);
    check(b.rows() == sb.nodes, "2D transfer node count", double(b.rows()));
    double ma = 0, mb = 0, worst = 1;
    for (int i = 0; i < sa.nodes; ++i) ma += sa.lumped[i] * project(a.row(i).transpose())[0];
    for (int i = 0; i < sb.nodes; ++i) {
        const Four v = project(b.row(i).transpose());
        mb += sb.lumped[i] * v[0];
        worst = std::min(worst, margin(v) / v[0]);
    }
    check(std::abs(ma - mb) / ma < 1e-14, "2D transfer conserves the density integral (scheme lumped mass)",
          std::abs(ma - mb) / ma);
    check(worst > -1e-15, "2D transfer keeps every node in the M1 cone (relative margin)", worst);
    // Positions: refine-only transfer reproduces point values (evaluate on both meshes).
    std::array<amr::Axis, 3> F = A;
    for (int d = 0; d < 2; ++d) for (int &l : F[d].level) ++l;
    ElementWeno sf(pn, mesh_options(F));
    for (int d = 0; d < 2; ++d) T[d] = amr::transfer(A[d].line(), F[d].line());
    const Field f = amr::apply(a, sa.nodes_per_axis, 2, T);
    double diff = 0;
    for (double x : {1.05, 1.3, 1.77, 1.99})
        for (double y : {.1, .55, 1.4, 1.93})
            diff = std::max(diff, (sa.evaluate(a, V3(x, y, 0)) - sf.evaluate(f, V3(x, y, 0))).cwiseAbs().maxCoeff());
    check(diff < 1e-13, "2D refinement reproduces the Bernstein function", diff);
}
void test_levels() {
    // Step in x at 1.1: refined near the step (with buffer and grading), coarsened where flat.
    std::array<amr::Axis, 3> ax{axis(16, std::vector<int>(16, 1), {}, 1), axis(4, std::vector<int>(4, 1), {}, 1), {}};
    amr::Settings s;
    s.max_level = 3, s.tol = 1e-3, s.buffer = 1;
    for (int it = 0; it < 4; ++it) {
        const amr::Line lx = ax[0].line(), ly = ax[1].line();
        Field u = Field::Zero(long(lx.nodes()) * ly.nodes(), 4);
        for (int j = 0; j < ly.nodes(); ++j)
            for (int i = 0; i < lx.nodes(); ++i)
                u(i + long(j) * lx.nodes(), 0) = lx.planes[std::min(i, lx.elements())] < 1.1 ? 1 : 1e-3;
        const auto eta = amr::indicator(u, 2, ax, 1.);
        amr::update_levels(ax, 2, eta, s);
    }
    const auto &lx = ax[0].level, &ly = ax[1].level;
    bool graded = true;
    for (std::size_t b = 1; b < lx.size(); ++b) graded = graded && std::abs(lx[b] - lx[b - 1]) <= 1;
    check(lx[8] == 3 && lx[9] == 3, "front base cells at the finest level", lx[8]);
    check(lx[0] == 0 && lx[15] == 0, "flat base cells coarsened to level 0", lx[0] + lx[15]);
    check(graded, "levels graded 2:1");
    check(ly[0] == 0 && ly[3] == 0, "y axis (no variation) coarsened", ly[0]);
}
void test_table_window() {
    // Windowed lattice table: same values as the full table inside the window, integrals add up.
    uncollided::LatticeUncollidedTable full, win;
    full.lattice = win.lattice = uncollided::lattice_paul(cases::lattice_paul_absorber_cell);
    const std::vector<V3> x{V3(3.5, 3.5, 0), V3(2.2, 4.7, 0), V3(1.1, 0.4, 0), V3(5.9, 3.1, 0)};
    full.build(x, 2.0);
    win.start_time = 1.13;
    win.build(x, 1.6);
    double diff = 0, integral = 0;
    for (std::size_t k = 0; k < x.size(); ++k) {
        for (double t : {1.13, 1.2, 1.37, 1.6})
            diff = std::max(diff, std::abs(full.value(k, t) - win.value(k, t)));
        const double a = full.window_integral(k, 0, 1.13) + win.window_integral(k, 1.13, 1.6), b = full.time_integral(k, 1.6);
        integral = std::max(integral, std::abs(a - b) / std::max(b, 1e-300));
    }
    check(diff == 0, "windowed FCS table bitwise equal to the full table", diff);
    check(integral < 1e-12, "window integrals add up to the time integral", integral);
    check(win.width() < full.width() / 2, "window table is shorter", win.width());
}
void test_carried_orders() {
    // Null invariant: moments above the node order vanish before the transfer -> above the mapped order after it.
    std::array<amr::Axis, 3> A{axis(8, {0, 1, 2, 2, 1, 0, 0, 0}, {1.0}, 1), axis(8, {2, 2, 1, 0, 0, 1, 2, 1}, {1.0}, 1), {}};
    std::array<amr::Axis, 3> B{axis(8, {1, 2, 1, 0, 0, 1, 1, 0}, {1.0}, 1), axis(8, {1, 1, 0, 0, 1, 2, 2, 2}, {1.0}, 1), {}};
    const amr::Line ax = A[0].line(), ay = A[1].line();
    std::array<int, 3> n{ax.nodes(), ay.nodes(), 1};
    const int q = 36; // P5 moments, order of moment c: floor(sqrt(c))
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> pick(0, 2);
    std::vector<int> order(std::size_t(n[0]) * n[1]);
    Field a = Field::Zero(long(order.size()), q);
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = 1 + 2 * pick(rng);
        for (int c = 0; c < (order[i] + 1) * (order[i] + 1); ++c) a(long(i), c) = 1 + c;
    }
    std::array<amr::Rows, 3> T;
    for (int d = 0; d < 2; ++d) T[d] = amr::transfer(A[d].line(), B[d].line());
    const Field b = amr::apply(a, n, 2, T);
    const auto ob = amr::apply_max(order, n, 2, T);
    double leak = 0;
    for (long i = 0; i < b.rows(); ++i)
        for (int c = (ob[i] + 1) * (ob[i] + 1); c < q; ++c) leak = std::max(leak, std::abs(b(i, c)));
    check(leak == 0, "carried PN orders keep the null invariant through the transfer", leak);
}
} // namespace
int main() {
    test_carried_orders();
    test_table_window();
    test_subdivision();
    test_transfer_1d();
    test_transfer_2d();
    test_levels();
    std::cout << (failures ? "FAILED" : "all passed") << "\n";
    return failures ? 1 : 0;
}
