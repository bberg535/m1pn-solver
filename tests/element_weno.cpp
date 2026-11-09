// Structural checks of the element-based WENO variant.
#include "element_weno.hpp"
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
// A smooth realizable state whose higher coefficients are nonzero.
Vec state(const Angular &pn, const V3 &x) {
    Vec a = Vec::Zero(pn.q);
    const double rho = 1 + .3 * std::sin(2 * pi * x[0]) * std::cos(2 * pi * x[1]);
    const V3 j(.2 * std::cos(2 * pi * x[0]), .1 * std::sin(2 * pi * x[1]), 0);
    set_low(a, Four(rho, rho * j[0], rho * j[1], rho * j[2]));
    for (int k = 4; k < pn.q; ++k)
        a[k] = .05 * std::sin(2 * pi * (x[0] + k * x[1]));
    return a;
}
void structure(int degree, int dimension, bool periodic) {
    Angular pn(3);
    ElementWenoOptions o;
    o.degree = degree;
    o.dimension = dimension;
    o.elements = dimension == 1 ? 8 : 4;
    o.periodic = periodic;
    ElementWeno scheme(pn, o);
    const Field a = scheme.interpolate([&](const V3 &x) { return state(pn, x); });
    ElementWenoStats stats;
    ElementWenoDebug debug;
    const Field r = scheme.rate(0, a, &stats, &debug);
    const std::string tag = "p=" + std::to_string(degree) + " d=" + std::to_string(dimension) +
                            (periodic ? " periodic: " : " open: ");
    const double recovery = maxabs(debug.unlimited_rate - debug.ho_moment_rate);
    check(recovery < 1e-9, tag + "unlimited contributions recover PN moment rows", recovery);
    check(stats.max_raw_zero_sum < 1e-11, tag + "raw element contributions sum to zero", stats.max_raw_zero_sum);
    check(stats.max_limited_zero_sum < 1e-11, tag + "limited element contributions sum to zero", stats.max_limited_zero_sum);
    if (periodic) {
        Four total = Four::Zero();
        for (int i = 0; i < scheme.nodes; ++i)
            total += scheme.lumped[i] * project(r.row(i).transpose());
        check(total.cwiseAbs().maxCoeff() < 1e-11, tag + "limited rate conserves all four moments", total.cwiseAbs().maxCoeff());
    }
    const double dt = .9 * scheme.dt_bound;
    bool cone = true;
    const Field next = a + dt * r;
    for (int i = 0; i < scheme.nodes; ++i)
        cone = cone && exact::cone(project(next.row(i).transpose()));
    check(cone, tag + "Euler step stays in the M1 cone");
}
void polynomial(int degree) {
    Angular pn(1);
    ElementWenoOptions o;
    o.degree = degree;
    o.dimension = 2;
    o.elements = 4;
    o.periodic = false;
    ElementWeno scheme(pn, o);
    const Field a = scheme.project_l2([&](const V3 &x) {
        Vec v = Vec::Zero(pn.q);
        v[0] = 2 + std::pow(x[0], degree) + .5 * std::pow(x[1], degree);
        return v;
    });
    const Vec gamma = scheme.weno_gamma(a);
    check((1 - gamma.array()).abs().maxCoeff() < 1e-7, "p=" + std::to_string(degree) + ": WENO sensor is one on a global polynomial",
          (1 - gamma.array()).abs().maxCoeff());
    const Field load = scheme.stabilization_load(a, gamma);
    check(maxabs(load) < 1e-7, "p=" + std::to_string(degree) + ": stabilization vanishes on a global polynomial", maxabs(load));
}
void rough_cone() {
    Angular pn(3);
    ElementWenoOptions o;
    o.degree = 2;
    o.dimension = 2;
    o.elements = 6;
    o.periodic = true;
    o.weno_linear = .2; // equal weights, the sensitive choice of Kuzmin and Vedral, Section 7
    ElementWeno scheme(pn, o);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> uniform(0, 1), signed_uniform(-1, 1);
    Field a = Field::Zero(scheme.nodes, pn.q);
    for (int i = 0; i < scheme.nodes; ++i) {
        const double rho = uniform(rng) < .3 ? 1e-6 : uniform(rng);
        V3 d(signed_uniform(rng), signed_uniform(rng), signed_uniform(rng));
        d = d.normalized() * uniform(rng) * .98 * rho;
        Vec row = Vec::Zero(pn.q);
        for (int k = 4; k < pn.q; ++k)
            row[k] = .5 * signed_uniform(rng);
        set_low(row, Four(rho, d[0], d[1], d[2]));
        a.row(i) = row.transpose();
    }
    ElementWenoStats stats;
    const Field r = scheme.rate(0, a, &stats);
    const Field next = a + .9 * scheme.dt_bound * r;
    bool cone = true;
    Four total = Four::Zero();
    for (int i = 0; i < scheme.nodes; ++i) {
        cone = cone && exact::cone(project(next.row(i).transpose()));
        total += scheme.lumped[i] * project(r.row(i).transpose());
    }
    check(cone, "rough realizable data stay in the cone", stats.min_limiter);
    check(stats.limited_elements > 0, "rough data activate the limiter", double(stats.limited_elements));
    check(total.cwiseAbs().maxCoeff() < 1e-11, "rough data conserve all moments", total.cwiseAbs().maxCoeff());
    check(stats.min_gamma < .9, "rough data lower the WENO sensor with equal linear weights", stats.min_gamma);
}
} // namespace
int main() {
    try {
        for (int p = 1; p <= 4; ++p) {
            structure(p, 1, true);
            structure(p, 2, true);
            structure(p, 1, false);
        }
        structure(2, 2, false);
        for (int p = 1; p <= 4; ++p)
            polynomial(p);
        rough_cone();
    } catch (const std::exception &e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    std::cout << (failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
