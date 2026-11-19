// Regression tests for docs/BOUNDARY_FIXES.md: C1 (outward base-bar rounding), C2 (vacuum sign limiter),
// C5 (scale-invariant WENO sensor), L3 (analytic s_h regression and symmetric residual alternative).
// C6 (low-order consistency of the ghost state) needs the low-order control hook and lives in audit/audit_runs.cpp.
// Private members are read through `#define private public` (test-only access, no production change).
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <array>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <vector>
#define private public
#include "element_weno.hpp"
#undef private
#include "base_bar.hpp"
using namespace m1pn;
namespace {
int failures = 0;
void check(bool ok, const std::string &what, double value = 0) {
    std::cout << (ok ? "ok   " : "FAIL ") << what << "  [" << std::setprecision(6) << value << "]\n";
    if (!ok) ++failures;
}
Mat symm(const Mat &m) { return .5 * (m + m.transpose()); }
double lam_min(const Mat &m) { return Eigen::SelfAdjointEigenSolver<Mat>(symm(m)).eigenvalues().minCoeff(); }
double lam_max(const Mat &m) { return Eigen::SelfAdjointEigenSolver<Mat>(symm(m)).eigenvalues().maxCoeff(); }

// ---------------- L3: analytic 1D P1 regression on a nonuniform grid ----------------
struct Grid1D {
    std::vector<double> h;
    int nodes() const { return int(h.size()) + 1; }
    // Element matrices of P1: mass h/6[[2,1],[1,2]], C(i,k) = int phi_i phi_k' = [[-1/2,1/2],[-1/2,1/2]], K = 1/h [[1,-1],[-1,1]].
    Mat mass(const std::vector<double> &w) const {
        Mat m = Mat::Zero(nodes(), nodes());
        for (std::size_t e = 0; e < h.size(); ++e) {
            m(e, e) += w[e] * h[e] / 3; m(e + 1, e + 1) += w[e] * h[e] / 3;
            m(e, e + 1) += w[e] * h[e] / 6; m(e + 1, e) += w[e] * h[e] / 6;
        }
        return m;
    }
    Mat conv(const std::vector<double> &w) const {  // sum_e w_e C_e
        Mat c = Mat::Zero(nodes(), nodes());
        for (std::size_t e = 0; e < h.size(); ++e) {
            c(e, e) += -.5 * w[e]; c(e, e + 1) += .5 * w[e];
            c(e + 1, e) += -.5 * w[e]; c(e + 1, e + 1) += .5 * w[e];
        }
        return c;
    }
    // Original CG-WENO form: s(phi_i, phi_k) = sum_e nu_e [ (phi_i', phi_k') - gamma_e (phi_i', g_k) ], g_k = projection of phi_k'
    // onto continuous P1 in the inner product with element weights `proj_w` (all ones = unweighted L2).
    Mat original(const std::vector<double> &nu, const std::vector<double> &gamma, const std::vector<double> &proj_w) const {
        const int n = nodes();
        const Mat G = mass(proj_w).ldlt().solve(conv(proj_w));  // column k = nodal values of g_k
        Mat s = Mat::Zero(n, n);
        for (std::size_t e = 0; e < h.size(); ++e) {
            Mat Ke = Mat::Zero(n, n), Ce = Mat::Zero(n, n);
            Ke(e, e) = Ke(e + 1, e + 1) = 1 / h[e]; Ke(e, e + 1) = Ke(e + 1, e) = -1 / h[e];
            Ce(e, e) = -.5; Ce(e, e + 1) = .5; Ce(e + 1, e) = -.5; Ce(e + 1, e + 1) = .5;
            s += nu[e] * (Ke - gamma[e] * Ce.transpose() * G);   // (phi_i', g_k)_e = sum_j C_e(j,i) g_kj
        }
        return s;
    }
    // Symmetric residual form: sum_e nu_e [ (1-gamma_e)(v',u')_e + gamma_e ((I-P)v', (I-P)u')_e ].
    Mat symmetric(const std::vector<double> &nu, const std::vector<double> &gamma, const std::vector<double> &proj_w) const {
        const int n = nodes();
        const Mat G = mass(proj_w).ldlt().solve(conv(proj_w));
        Mat s = Mat::Zero(n, n);
        for (std::size_t e = 0; e < h.size(); ++e) {
            // On element e, (I-P)phi_k' = phi_k' - sum_j G(j,k) phi_j. Gram matrix of these functions over e:
            Mat Ke = Mat::Zero(n, n), Ce = Mat::Zero(n, n), Me = Mat::Zero(n, n);
            Ke(e, e) = Ke(e + 1, e + 1) = 1 / h[e]; Ke(e, e + 1) = Ke(e + 1, e) = -1 / h[e];
            Ce(e, e) = -.5; Ce(e, e + 1) = .5; Ce(e + 1, e) = -.5; Ce(e + 1, e + 1) = .5;
            Me(e, e) = Me(e + 1, e + 1) = h[e] / 3; Me(e, e + 1) = Me(e + 1, e) = h[e] / 6;
            const Mat residual_gram = Ke - Ce.transpose() * G - G.transpose() * Ce + G.transpose() * Me * G;
            s += nu[e] * ((1 - gamma[e]) * Ke + gamma[e] * residual_gram);
        }
        return s;
    }
};

void l3_analytic() {
    Grid1D g{{1., .5}};
    const std::vector<double> nu{.5, .25}, ones{1., 1.}, gam1{1., 1.};
    Vec u(3);
    u << 0, 1, 1 + 1.5 * .5;                         // gradients (1, 3/2)
    const Mat S = g.original(nu, gam1, ones);
    const double s_orig = u.dot(S * u);
    check(std::abs(s_orig + 1. / 192) < 1e-15, "L3: s_h(u,u) = -1/192 (original form, unweighted L2 projection, nu = h/2, gamma = 1)", s_orig);
    check(lam_min(S) < 0, "L3: the original form is not PSD on the nonuniform grid", lam_min(S));
    // weighted projection with weights nu_e: Pythagoras in the nu-inner product gives s = |u'|_nu^2 - |g|_nu^2 >= 0
    const Mat Sw = g.original(nu, gam1, nu);
    const double s_w = u.dot(Sw * u);
    check(s_w >= -1e-15 && lam_min(Sw) > -1e-14, "L3 alternative: nu-weighted projection is PSD for gamma = 1", s_w);
    check((Sw - Sw.transpose()).norm() < 1e-14, "L3 alternative: nu-weighted projection is symmetric for gamma = 1", (Sw - Sw.transpose()).norm());
    // symmetric residual form: symmetric and PSD for every gamma in [0,1], also on the nonuniform grid
    const Mat Ssym = g.symmetric(nu, gam1, ones);
    check((Ssym - Ssym.transpose()).norm() < 1e-14 && lam_min(Ssym) > -1e-14, "L3 alternative: symmetric residual form is symmetric PSD (gamma = 1, nonuniform)", lam_min(Ssym));
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> U(0, 1);
    Grid1D rough{{.3, 1., .5, .2, 1.2, .7}};
    std::vector<double> nu6, w6(6, 1.), gam6;
    for (double h : rough.h) { nu6.push_back(h / 2); gam6.push_back(U(rng)); }
    const Mat Sr = rough.original(nu6, gam6, w6), Ssr = rough.symmetric(nu6, gam6, w6);
    check((Sr - Sr.transpose()).norm() > 1e-3, "L3: original form is nonsymmetric for varying gamma", (Sr - Sr.transpose()).norm());
    check((Ssr - Ssr.transpose()).norm() < 1e-14 && lam_min(Ssr) > -1e-13, "L3 alternative: symmetric residual form: symmetric PSD for random gamma, random nonuniform grid", lam_min(Ssr));
    check((Ssr * Vec::Ones(7)).norm() < 1e-14, "L3 alternative: symmetric residual form annihilates constants", (Ssr * Vec::Ones(7)).norm());
    std::vector<double> gam_vertex(6, 0.);
    double worst = 1e300;
    for (int mask = 0; mask < 64; ++mask) {
        for (int e = 0; e < 6; ++e) gam_vertex[e] = (mask >> e) & 1;
        worst = std::min(worst, lam_min(rough.original(nu6, gam_vertex, w6)));
    }
    std::cout << "info  original form, nonuniform grid, min over gamma-vertices of lambda_min(sym part) = " << worst << "\n";
}

// ---------------- production-class helpers ----------------
Mat dense_S(const ElementWeno &s, const Vec &gamma) {
    Mat S(s.nodes, s.nodes);
    for (int j = 0; j < s.nodes; ++j) {
        Field a = Field::Zero(s.nodes, s.pn.q);
        a(j, 0) = 1;
        S.col(j) = s.gather_load(s.stabilization_load(a, gamma)).col(0);
    }
    return S;
}

void l3_production() {
    Angular pn(1);
    for (int p : {1, 2, 3}) {
        ElementWenoOptions o;
        o.degree = p; o.dimension = 1; o.elements = 7; o.periodic = true;
        ElementWeno orig(pn, o);
        o.symmetric_residual = true;
        ElementWeno sym(pn, o);
        const std::string tag = "p=" + std::to_string(p) + ": ";
        const Vec ones = Vec::Ones(orig.cells);
        const Mat S1 = dense_S(orig, ones), Y1 = dense_S(sym, ones);
        check((S1 - Y1).norm() < 1e-12 * S1.norm(), tag + "symmetric residual = original for gamma = 1 on the uniform mesh", (S1 - Y1).norm() / S1.norm());
        std::mt19937 rng(9 + p);
        std::uniform_real_distribution<double> U(0, 1);
        Vec g(orig.cells);
        for (int e = 0; e < orig.cells; ++e) g[e] = U(rng);
        const Mat So = dense_S(orig, g), Ys = dense_S(sym, g);
        check((So - So.transpose()).norm() > 1e-3 * S1.norm(), tag + "original form is nonsymmetric for varying gamma", (So - So.transpose()).norm() / S1.norm());
        check((Ys - Ys.transpose()).norm() < 1e-12 * S1.norm(), tag + "symmetric residual form is symmetric for varying gamma", (Ys - Ys.transpose()).norm() / S1.norm());
        check(lam_min(Ys) > -1e-12 * lam_max(S1), tag + "symmetric residual form is PSD for varying gamma", lam_min(Ys) / lam_max(S1));
        check((Ys * Vec::Ones(orig.nodes)).norm() < 1e-12 * S1.norm(), tag + "symmetric residual form annihilates constants", (Ys * Vec::Ones(orig.nodes)).norm());
    }
    // polynomial reproduction at gamma = 1 on an open mesh
    for (int p : {2, 3}) {
        ElementWenoOptions o;
        o.degree = p; o.dimension = 2; o.elements = 3; o.periodic = false; o.symmetric_residual = true;
        Angular pn1(1);
        ElementWeno s(pn1, o);
        Field f = s.project_l2([&](const V3 &x) { Vec v = Vec::Zero(pn1.q); v[0] = 1 + std::pow(x[0], p) * std::pow(x[1], p); return v; });
        const Field l = s.gather_load(s.stabilization_load(f, Vec::Ones(s.cells)));
        check(l.cwiseAbs().maxCoeff() < 1e-12 * f.cwiseAbs().maxCoeff(), "p=" + std::to_string(p) + ": symmetric residual reproduces Q_p polynomials (gamma = 1)", l.cwiseAbs().maxCoeff());
    }
}

// ---------------- C1 ----------------
void c1_base_bar() {
    // Exact stored inputs of the failing pair (audit/results/c1_probe.jsonl, T2 corner_cone_boundary_J, step 0 stage 2).
    const Four ui(0x1.2baa127db801fp-1, -0x1.2baa127db801fp-1, 0., 0.), uj(0x1.26af4dc5756dbp-8, -0x1.26af4dc5756dap-8, 0., 0.);
    const V3 edge(0x1.5555555555555p-6, -0x1.5555555555555p-7, 0.);
    const double d = 0x1.7d9f4cf754635p-6;
    bool legacy_throws = false;
    try { stable_base_bar(ui, uj, edge, d); } catch (const StepRejected &) { legacy_throws = true; }
    check(legacy_throws, "C1: legacy stable_base_bar rejects the near-beam pair (binary64 rounding of an exactly admissible bar)");
    int raised = -1;
    const Four bar = stable_base_bar_outward(ui, uj, edge, d, &raised);
    check(exact::cone(bar) && raised >= 0 && raised <= 8, "C1: outward-rounded bar is in the exact stored cone within <= 8 ulp", raised);
    // exact value in 80-digit arithmetic (audit/c1_highprecision.py): rho = 0.03515410544183704201015707, |J| = ...118858024
    check(std::abs(bar[0] - 0.03515410544183704201015707) < 2e-16 * bar[0], "C1: outward bar density agrees with the 100-digit value to 2 ulp", bar[0]);
    // Whole scheme: beam on the cone boundary in a vacuum, corner node, 7 SSPRK3 steps at the implemented step bound.
    Angular pn(3);
    ElementWenoOptions o;
    o.degree = 1; o.dimension = 2; o.elements = 8; o.periodic = false;
    for (bool fix : {false, true}) {
        o.outward_bar_rounding = fix;
        ElementWeno s(pn, o);
        Field a = Field::Zero(s.nodes, pn.q);
        Vec row = Vec::Zero(pn.q);
        set_low(row, Four(1., -1., 0., 0.));
        while (!exact::cone(project(row))) row[0] = std::nextafter(row[0], 1e300);
        a.row(0) = row.transpose();
        const int steps = int(std::ceil(.1 / s.step_bound() - 1e-12));
        bool threw = false;
        ElementWenoStats st;
        try { for (int k = 0; k < steps; ++k) a = s.advance_ssprk3(k * .1 / steps, .1 / steps, a, &st); } catch (const StepRejected &) { threw = true; }
        check(threw == !fix, std::string("C1: vacuum corner beam on the cone boundary ") + (fix ? "runs with outward rounding" : "is rejected without it (legacy)"));
        if (fix) {
            bool cone = true;
            for (int i = 0; i < s.nodes; ++i) cone = cone && exact::cone(project(a.row(i).transpose()));
            check(cone, "C1: final state in the exact cone; outward bars counted", double(st.outward_bars));
        }
    }
}

// ---------------- C2 ----------------
double mass_rate(const ElementWeno &s, const Field &rate) {
    double r = 0, scale = 0;
    for (int i = 0; i < s.nodes; ++i) { r += s.lumped[i] * rate(i, 0) * std::sqrt(4 * pi); scale += s.lumped[i] * std::abs(rate(i, 0)) * std::sqrt(4 * pi); }
    return scale > 0 ? r / scale : 0.;
}
void c2_sign_limiter() {
    Angular pn(3);
    struct Case { std::string name; int node; V3 J; };
    const std::vector<Case> cases = {{"corner, inward J", 0, V3(.5, .5, 0) / std::sqrt(2.) * 1.5}, {"corner, outward J", 0, V3(-.5, -.5, 0) / std::sqrt(2.) * 1.5},
                                      {"edge node", -2, V3(0, -.9, 0)}, {"interior node", -1, V3(.6, 0, 0)}};
    for (const auto &cs : cases)
        for (bool fix : {false, true}) {
            ElementWenoOptions o;
            o.degree = 1; o.dimension = 2; o.elements = 8; o.periodic = false; o.vacuum_sign_limit = fix;
            ElementWeno s(pn, o);
            const int node = cs.node == -1 ? 4 + 4 * s.per_direction : cs.node == -2 ? 4 : cs.node;
            Field a = Field::Zero(s.nodes, pn.q);
            Vec row = Vec::Zero(pn.q);
            set_low(row, Four(1., cs.J[0], cs.J[1], cs.J[2]));
            while (!exact::cone(project(row))) row[0] = std::nextafter(row[0], 1e300);
            a.row(node) = row.transpose();
            const int steps = int(std::ceil(.1 / (.9 * s.step_bound())));
            const double dt = .1 / steps;
            double worst = -1e300;
            bool cone = true;
            ElementWenoStats st;
            for (int k = 0; k < steps; ++k) {
                const Field r = s.rate(k * dt, a, &st);
                worst = std::max(worst, mass_rate(s, r));
                a = s.advance_ssprk3(k * dt, dt, a);
                for (int i = 0; i < s.nodes; ++i) cone = cone && exact::cone(project(a.row(i).transpose()));
            }
            // relative accepted mass rate must not be positive beyond rounding in vacuum inflow
            if (fix)
                check(worst <= 1e-12 && cone, "C2 (" + cs.name + "): accepted mass rate <= 0 in vacuum inflow, cone kept at every step", worst);
            else if (cs.name == "corner, inward J")
                check(worst > 1e-3, "C2 (" + cs.name + ", legacy): reproduces the mass gain from vacuum (counterexample)", worst);
        }
    // inflow is not suppressed: isotropic stationary state with matching kinetic data stays stationary (T1)
    ElementWenoOptions o;
    o.degree = 2; o.dimension = 2; o.elements = 4; o.periodic = false;
    Inflow in;
    in.density = [](const V3 &, double, const V3 &) { return 1. / (4 * pi); };
    in.constant_density = 1. / (4 * pi);
    ElementWeno s(pn, o, in);
    Vec iso = Vec::Zero(pn.q);
    iso[0] = 1 / std::sqrt(4 * pi);
    const Field a = s.interpolate([&](const V3 &) { return iso; });
    ElementWenoStats st;
    const Field r = s.rate(0, a, &st);
    check(r.cwiseAbs().maxCoeff() < 1e-12 && st.sign_limited_nodes == 0, "C2: matching isotropic inflow remains stationary and unlimited by the sign condition", r.cwiseAbs().maxCoeff());
    // genuine inflow: mass may increase when the kinetic inflow supplies it (no sign limit above the inflow)
    Inflow beam;
    beam.density = [](const V3 &, double, const V3 &) { return 1. / (4 * pi); };
    ElementWenoOptions ob = o;
    ElementWeno sb(pn, ob, beam);
    Field zero = Field::Zero(sb.nodes, pn.q);
    const Field rb = sb.rate(0, zero, &st);
    check(mass_rate(sb, rb) > 0 || rb.cwiseAbs().maxCoeff() > 0, "C2: kinetic inflow into an empty domain still fills it", rb.cwiseAbs().maxCoeff());
}

// ---------------- C5 ----------------
void c5_scale_invariance() {
    Angular pn(3);
    for (bool fix : {false, true}) {
        ElementWenoOptions o;
        o.degree = 2; o.dimension = 1; o.elements = 16; o.periodic = false; o.scale_invariant_weno = fix;
        ElementWeno s(pn, o);
        Field a = Field::Zero(s.nodes, pn.q);
        for (int i = 0; i < s.nodes; ++i) {
            const double x = double(i) / (s.nodes - 1);
            a(i, 0) = 1 + .5 * std::tanh((x - .5) / .03) + .2 * std::sin(9 * x);
        }
        const Vec g1 = s.weno_gamma(a);
        double worst = 0;
        for (double scale : {1e-1, 1e-3, 1e-6, 1e-12, 1e-30, 1e-100, 1e100})
            worst = std::max(worst, (s.weno_gamma(a * scale) - g1).cwiseAbs().maxCoeff());
        if (fix) check(worst < 1e-12, "C5: gamma is invariant under u -> s u for s in [1e-100, 1e100]", worst);
        else check(worst > 1e-3, "C5 (legacy, absolute epsilon): gamma depends on the amplitude", worst);
        if (fix) {
            check(s.weno_gamma(Field::Zero(s.nodes, pn.q)).minCoeff() == 1., "C5: exact vacuum has gamma = 1 by definition");
            Field half = a;
            for (int i = 0; i < s.nodes; ++i) if (i < s.nodes / 2) half.row(i).setZero();
            const Vec gh = s.weno_gamma(half * 1e-40);
            check(gh.allFinite() && gh.minCoeff() >= 0 && gh.maxCoeff() <= 1, "C5: half vacuum / half 1e-40 data gives gamma in [0,1], finite", gh.minCoeff());
            // rate homogeneity: rate(s u) = s rate(u), no inflow
            Field b = Field::Zero(s.nodes, pn.q);
            ElementWenoOptions o2 = o;
            o2.dimension = 2; o2.elements = 6;
            ElementWeno s2(pn, o2);
            b = s2.interpolate([&](const V3 &x) {
                Vec v = Vec::Zero(pn.q);
                v[0] = (.3 + std::exp(-30 * (x - V3(.5, .5, 0)).squaredNorm())) / std::sqrt(4 * pi);
                v[3] = .1 * std::exp(-30 * (x - V3(.5, .5, 0)).squaredNorm());
                return v;
            });
            const Field r1 = s2.rate(0, b);
            double dev = 0;
            for (double scale : {1e-3, 1e-9, 1e-30}) dev = std::max(dev, (s2.rate(0, b * scale) / scale - r1).cwiseAbs().maxCoeff() / r1.cwiseAbs().maxCoeff());
            check(dev < 1e-10, "C5: rate(s u)/s = rate(u) (up to rounding) for s down to 1e-30", dev);
        }
    }
}
} // namespace
int main() {
    try {
        l3_analytic();
        l3_production();
        c1_base_bar();
        c2_sign_limiter();
        c5_scale_invariance();
    } catch (const std::exception &e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    std::cout << (failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
