// Tests of the first-collision-source split (docs/FCS.md): analytic uncollided pulses (src/uncollided.hpp) and
// their coupling to element WENO (collided source quadrature, reaction step, mass ledger).
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#define private public
#include "element_weno.hpp"
#undef private
#include "uncollided.hpp"
using namespace m1pn;
namespace {
int failures = 0;
void check(bool ok, const std::string &what, double value = 0) {
    std::cout << (ok ? "ok   " : "FAIL ") << what << "  [" << std::setprecision(6) << value << "]\n";
    if (!ok) ++failures;
}
void i0e_values() {
    double worst = 0;
    for (double z : {0., 1e-3, .1, 1., 5., 20., 29.9, 30.1, 50., 200., 600.}) {
        const double exact = std::cyl_bessel_i(0., z) * std::exp(-z);
        worst = std::max(worst, std::abs(uncollided::i0e(z) / exact - 1));
    }
    check(worst < 1e-11, "i0e against std::cyl_bessel_i (relative)", worst);
}
void plane_pulse() {
    uncollided::GaussianPulse g;
    g.dimension = 1, g.width = .05, g.mass = .7, g.sigma_t = 2, g.centre = V3(.5, 0, 0);
    const double t = .3;
    // mass: int rho_u dx = mass e^{-sigma_t t}
    const auto [x, w] = uncollided::gauss_legendre(4000, -1, 2);
    double m = 0;
    for (std::size_t k = 0; k < x.size(); ++k)
        m += w[k] * g.plane(x[k], t);
    check(std::abs(m / (g.mass * std::exp(-g.sigma_t * t)) - 1) < 1e-10, "plane pulse: uncollided mass",
          m / (g.mass * std::exp(-g.sigma_t * t)) - 1);
    // closed form against the kernel e^{-sigma t}/(2t) int_{x-t}^{x+t} rho_0
    double worst = 0;
    for (double xi : {.5, .6, .79, .8, .81, .9}) {
        const auto [y, v] = uncollided::gauss_legendre(2000, xi - t, xi + t);
        double s = 0;
        for (std::size_t k = 0; k < y.size(); ++k)
            s += v[k] * g.plane(y[k], 0);
        worst = std::max(worst, std::abs(g.plane(xi, t) - std::exp(-g.sigma_t * t) * s / (2 * t)));
    }
    check(worst < 1e-12, "plane pulse: closed form against the kernel integral", worst);
}
void line_pulse() {
    uncollided::GaussianPulse g;
    g.dimension = 2, g.width = .01, g.mass = 1, g.sigma_t = 3, g.centre = V3(.5, .5, 0);
    const double t = .2;
    // mass: int 2 pi r rho_u dr = e^{-sigma_t t}
    const auto [r, w] = uncollided::gauss_legendre(6000, 0, t + 10 * g.width);
    double m = 0;
    for (std::size_t k = 0; k < r.size(); ++k)
        m += w[k] * 2 * pi * r[k] * g.line(r[k], t);
    check(std::abs(m / std::exp(-g.sigma_t * t) - 1) < 1e-6, "line pulse: uncollided mass", m / std::exp(-g.sigma_t * t) - 1);
    // windowed integral against the full a-integral of scripts/line_source/reference.py (4000 points)
    const auto [a, wa] = uncollided::gauss_legendre(4000, 0, pi / 2);
    double worst = 0, peak = 0;
    for (double ri : {0., .05, .1, .18, .195, .2, .205, .21}) {
        double s = 0;
        for (std::size_t k = 0; k < a.size(); ++k)
            s += wa[k] * std::sin(a[k]) * g.ring(ri, t * std::sin(a[k]));
        s *= std::exp(-g.sigma_t * t);
        worst = std::max(worst, std::abs(g.line(ri, t) - s));
        peak = std::max(peak, s);
    }
    check(worst < 1e-6 * peak, "line pulse: windowed integral against the full integral (relative to peak)", worst / peak);
    // tabulated batch evaluation against line()
    std::vector<V3> pts;
    for (int k = 0; k <= 400; ++k)
        pts.push_back(g.centre + V3(.3 * k / 400., .0007 * k, 0));
    std::vector<double> out;
    g.density(t, pts, out);
    double tab = 0, top = 0;
    for (std::size_t k = 0; k < pts.size(); ++k) {
        const double exact = g.line((pts[k] - g.centre).head<2>().norm(), t);
        tab = std::max(tab, std::abs(out[k] - exact));
        top = std::max(top, exact);
    }
    check(tab < 1e-4 * top, "line pulse: tabulated density (relative to peak)", tab / top);
}
void scheme_coupling() {
    Angular pn(3);
    ElementWenoOptions o;
    o.degree = 2, o.dimension = 2, o.elements = 16, o.periodic = true, o.sigma_s = 2;
    uncollided::GaussianPulse g;
    g.dimension = 2, g.width = .06, g.mass = 1, g.sigma_t = 2, g.centre = V3(.5, .5, 0);
    o.uncollided_density = [g](double t, const std::vector<V3> &x, std::vector<double> &out) { g.density(t, x, out); };
    ElementWeno scheme(pn, o);
    const double t = .1;
    double unc = 0;
    const Vec q = scheme.collided_source(t, &unc);
    double source = 0, mass_rule = 0;
    for (int i = 0; i < scheme.nodes; ++i) {
        source += scheme.fcs_node_mass_[i] * q[i];
        mass_rule = std::max(mass_rule, std::abs(scheme.fcs_node_mass_[i] / scheme.lumped[i] - 1));
    }
    check(mass_rule < 1e-13, "FCS rule integrates the Bernstein functions exactly (int phi_i = lumped mass)", mass_rule);
    check(std::abs(source / (o.sigma_s * unc) - 1) < 1e-13, "collided source = sigma_s x uncollided mass (same rule)",
          source / (o.sigma_s * unc) - 1);
    check(std::abs(unc / std::exp(-g.sigma_t * t) - 1) < 1e-4, "uncollided mass on the grid", unc / std::exp(-g.sigma_t * t) - 1);
    bool nonneg = true;
    for (int i = 0; i < scheme.nodes; ++i)
        nonneg &= q[i] >= 0;
    check(nonneg, "nodal collided source is nonnegative");
    // Reaction step on an isotropic floor: rho_i += q_i tau exactly (sigma_a = 0); higher moments stay zero;
    // the ledger records the injected mass.
    Field a = Field::Zero(scheme.nodes, scheme.moments);
    a.col(0).setConstant(1e-3 / std::sqrt(4 * pi));
    ElementWenoStats stats;
    const double tau = .01;
    const Field b = scheme.react(a, tau, &stats, t);
    double worst = 0, other = 0, injected = 0;
    for (int i = 0; i < scheme.nodes; ++i) {
        worst = std::max(worst, std::abs((b(i, 0) - a(i, 0)) * std::sqrt(4 * pi) - q[i] * tau));
        for (int k = 1; k < scheme.moments; ++k)
            other = std::max(other, std::abs(b(i, k)));
        injected += scheme.lumped[i] * q[i] * tau;
    }
    check(worst < 1e-15, "react: density gains q tau", worst);
    check(other == 0, "react: isotropic source keeps the higher moments at zero", other);
    check(std::abs(stats.injected / injected - 1) < 1e-14, "react: ledger injected mass", stats.injected / injected - 1);
    // Without the source time the reaction refuses to run.
    bool refused = false;
    try {
        scheme.react(a, tau, &stats);
    } catch (const std::exception &) {
        refused = true;
    }
    check(refused, "react: FCS requires the source time");
}
} // namespace
int main() {
    try {
        i0e_values();
        plane_pulse();
        line_pulse();
        scheme_coupling();
    } catch (const std::exception &e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    std::cout << (failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
