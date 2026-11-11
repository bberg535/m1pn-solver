// Space-dependent reaction for element WENO (docs/ELEMENT_HETEROGENEOUS.md): exact local reaction, nodal
// averages, realizability, ledger and global particle balance.
#include "element_weno.hpp"
#include "time.hpp"
#include "benchmark_cases.hpp"
#include <iostream>
#include <memory>
#include <random>
using namespace m1pn;
namespace {
int failures = 0;
void check(bool ok, const std::string &what, double value = 0) {
    std::cout << (ok ? "ok   " : "FAIL ") << what << "  [" << value << "]\n";
    if (!ok)
        ++failures;
}
// Random realizable state with nonzero higher moments; a fraction of nodes lies exactly on the cone boundary.
Field random_state(const ElementWeno &s, const Angular &pn, unsigned seed, bool boundary = true) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0, 1), v(-1, 1);
    Field a = Field::Zero(s.nodes, s.moments);
    for (int i = 0; i < s.nodes; ++i) {
        const double rho = u(rng) < .2 ? 1e-9 : u(rng);
        V3 d(v(rng), v(rng), v(rng));
        d = d.normalized() * (boundary && u(rng) < .2 ? 1. : .95 * u(rng)) * rho;
        Vec row = Vec::Zero(pn.q);
        for (int k = 4; k < pn.q; ++k)
            row[k] = .3 * rho * v(rng);
        set_low(row, Four(rho, d[0], d[1], d[2]));
        a.row(i) = s.reduce(row).transpose();
    }
    return a;
}
std::shared_ptr<ElementWenoMaterial> blocks(double absorber, double scatter, double source) {
    auto m = std::make_shared<ElementWenoMaterial>();
    m->sigma_a = [=](const V3 &x) { return x[0] < .5 ? absorber : 0.; };
    m->sigma_s = [=](const V3 &x) { return x[0] < .5 ? 0. : scatter; };
    m->source = [=](const V3 &x) { return x[1] > .5 ? source : 0.; };
    m->name = "blocks";
    return m;
}
// The local reaction equals the independent reaction() of path B (src/time.cpp) on the nodal coefficients.
void reaction_matches_path_b() {
    Angular pn(5);
    ElementWenoOptions o;
    o.degree = 2, o.dimension = 2, o.elements = 4, o.periodic = false;
    o.material = blocks(10, 1, 2);
    ElementWeno s(pn, o);
    const Field a = random_state(s, pn, 3, false); // path B rejects states on the cone boundary
    Field source = Field::Zero(s.nodes, pn.q);
    source.col(0) = s.node_source / std::sqrt(4 * pi);
    for (double tau : {1e-4, .03, 2.}) {
        const Field mine = s.react(a, tau);
        const Field theirs = reaction(a, source, s.node_sigma_a, s.node_sigma_s, 0., tau);
        check(maxabs(mine - theirs) < 1e-13, "reaction equals path-B reaction(), tau=" + std::to_string(tau),
              maxabs(mine - theirs));
    }
}
void special_cases() {
    Angular pn(3);
    ElementWenoOptions o;
    o.degree = 1, o.dimension = 1, o.elements = 4, o.periodic = true;
    auto run = [&](double sa, double ss, double q, double tau) {
        auto m = std::make_shared<ElementWenoMaterial>();
        m->sigma_a = [=](const V3 &) { return sa; };
        m->sigma_s = [=](const V3 &) { return ss; };
        m->source = [=](const V3 &) { return q; };
        o.material = m;
        ElementWeno s(pn, o);
        Field a = Field::Zero(s.nodes, s.moments);
        for (int i = 0; i < s.nodes; ++i) {
            Vec row = Vec::Zero(pn.q);
            set_low(row, Four(2, .5, -.3, .1));
            for (int k = 4; k < pn.q; ++k) row[k] = .01 * k;
            a.row(i) = row.transpose();
        }
        ElementWenoStats st;
        return std::make_tuple(a, s.react(a, tau, &st), st);
    };
    {
        auto [a, b, st] = run(3, 0, 0, .2);
        const double expect = std::exp(-.6);
        check(std::abs(b(0, 0) - expect * a(0, 0)) < 1e-15 && std::abs(b(0, 1) - expect * a(0, 1)) < 1e-15,
              "pure absorption decays all moments by exp(-sigma_a tau)", b(0, 0) / a(0, 0) - expect);
    }
    {
        auto [a, b, st] = run(0, 4, 0, .2);
        check(b(0, 0) == a(0, 0) && std::abs(b(0, 1) - std::exp(-.8) * a(0, 1)) < 1e-15,
              "pure scattering keeps rho and damps l >= 1", b(0, 1) / a(0, 1) - std::exp(-.8));
        check(st.absorbed == 0 && st.injected == 0, "pure scattering: ledger unchanged");
    }
    {
        auto [a, b, st] = run(0, 0, 5, .2);
        check(std::abs(project(b.row(0).transpose())[0] - project(a.row(0).transpose())[0] - 1.) < 1e-14 &&
                  b(0, 1) == a(0, 1),
              "pure source adds q tau to rho only", project(b.row(0).transpose())[0] - project(a.row(0).transpose())[0]);
    }
    {
        auto [a, b, st] = run(1e-300, 0, 5, .2);
        auto [a0, b0, st0] = run(0, 0, 5, .2);
        check(maxabs(b - b0) < 1e-15, "sigma_a -> 0 limit is continuous (expm1)", maxabs(b - b0));
    }
    {
        auto [a, b, st] = run(1e7, 0, 3, .1);
        const Four u = project(b.row(0).transpose());
        check(std::abs(u[0] - 3e-7) < 1e-15 && exact::cone(u), "stiff absorption: rho -> q / sigma_a, cone kept",
              u[0] / 3e-7 - 1);
    }
    {
        auto [a, b, st] = run(2, 1, 4, .3);
        const double rho0 = project(a.row(0).transpose())[0], rho1 = project(b.row(0).transpose())[0];
        const double z = -.6, phi = std::expm1(z) / z;
        check(std::abs(rho1 - (std::exp(z) * rho0 + 4 * .3 * phi)) < 1e-14, "combined reaction: exact rho(tau)",
              rho1 - (std::exp(z) * rho0 + 4 * .3 * phi));
    }
}
// Random states and coefficients (including boundary states |J| = rho and huge sigma): cone kept, ledger closes.
void realizability_and_ledger() {
    Angular pn(7);
    ElementWenoOptions o;
    o.degree = 3, o.dimension = 2, o.elements = 5, o.periodic = false;
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> u(0, 1);
    for (int trial = 0; trial < 4; ++trial) {
        auto m = std::make_shared<ElementWenoMaterial>();
        const double scale = std::pow(10., 3 * trial - 2);
        m->sigma_a = [=](const V3 &x) { return scale * (1 + std::sin(7 * x[0]) * std::sin(5 * x[1])); };
        m->sigma_s = [=](const V3 &x) { return x[0] + x[1] > 1 ? scale : 0.; };
        m->source = [=](const V3 &x) { return x[0] < .3 ? 2. : 0.; };
        o.material = m;
        ElementWeno s(pn, o);
        const Field a = random_state(s, pn, 20 + trial);
        ElementWenoStats st;
        const double tau = .05;
        const Field b = s.react(a, tau, &st);
        bool cone = true;
        double before = 0, after = 0;
        for (int i = 0; i < s.nodes; ++i) {
            cone = cone && exact::cone(project(b.row(i).transpose()));
            before += s.lumped[i] * project(a.row(i).transpose())[0];
            after += s.lumped[i] * project(b.row(i).transpose())[0];
        }
        const double closure = after - before + st.absorbed - st.injected - st.nudge_mass;
        const std::string tag = "reaction, sigma scale " + std::to_string(scale) + ": ";
        check(cone, tag + "all nodes stay in the M1 cone", double(st.nudged_states));
        check(std::abs(closure) < 1e-13 * std::max(1., before + st.injected), tag + "ledger closes", closure);
    }
}
// Nodal averages: constants exact, linear coefficients reproduce the Bernstein Greville points
// (int t B_i / int B_i = (i+1)/(p+2)), aligned jumps exact, sub-cells converge for an unaligned disk.
void nodal_averages() {
    Angular pn(1);
    for (int p = 1; p <= 4; ++p) {
        ElementWenoOptions o;
        o.degree = p, o.dimension = 1, o.elements = 3, o.periodic = false;
        auto m = std::make_shared<ElementWenoMaterial>();
        m->sigma_a = [](const V3 &) { return 2.5; };
        m->sigma_s = [](const V3 &x) { return x[0]; };
        o.material = m;
        ElementWeno s(pn, o);
        double worst_const = 0, worst_linear = 0;
        for (int i = 0; i < s.nodes; ++i) {
            worst_const = std::max(worst_const, std::abs(s.node_sigma_a[i] - 2.5));
            // Interior node i of the global line: average of the element-local Greville points weighted by
            // the element masses (all equal for Bernstein): the exact value is computed independently here.
            double num = 0, den = 0;
            for (int e = 0; e < 3; ++e)
                for (int l = 0; l <= p; ++l)
                    if (e * p + l == i) {
                        num += (e + double(l + 1) / (p + 2)) / 3;
                        den += 1;
                    }
            worst_linear = std::max(worst_linear, std::abs(s.node_sigma_s[i] - num / den));
        }
        check(worst_const < 1e-14 && worst_linear < 1e-14,
              "p=" + std::to_string(p) + ": constant and linear coefficients averaged exactly",
              std::max(worst_const, worst_linear));
    }
    {
        Angular pn3(3);
        ElementWenoOptions o;
        o.degree = 2, o.dimension = 2, o.elements = 7, o.periodic = false, o.length = 7;
        auto m = std::make_shared<ElementWenoMaterial>();
        m->sigma_a = [](const V3 &x) { return int(std::floor(x[0]) + std::floor(x[1])) % 2 ? 10. : 0.; };
        o.material = m;
        ElementWeno s(pn3, o);
        check(std::abs(s.integral_sigma_a - 240.) < 1e-11, "element-aligned checkerboard integrated exactly",
              s.integral_sigma_a - 240.);
    }
    double previous = 1;
    for (int sub : {1, 4, 16}) {
        Angular pn1(1);
        ElementWenoOptions o;
        o.degree = 2, o.dimension = 2, o.elements = 8, o.periodic = false;
        auto m = std::make_shared<ElementWenoMaterial>();
        m->sigma_a = [](const V3 &x) { return (x - V3(.5, .5, 0)).norm() < .3 ? 1. : 0.; };
        m->subcells = sub;
        o.material = m;
        ElementWeno s(pn1, o);
        const double error = std::abs(s.integral_sigma_a - pi * .09);
        check(error < previous, "disk area converges with sub-cells: " + std::to_string(sub), error);
        previous = error;
    }
}
// Constant material through the material path equals the constant-coefficient path.
void constant_material_equals_constant_path() {
    Angular pn(5);
    ElementWenoOptions o;
    o.degree = 2, o.dimension = 2, o.elements = 6, o.periodic = true, o.filter_strength = 20;
    o.sigma_a = .7, o.sigma_s = 1.3;
    ElementWeno constant(pn, o);
    o.sigma_a = o.sigma_s = 0;
    auto m = std::make_shared<ElementWenoMaterial>();
    m->sigma_a = [](const V3 &) { return .7; };
    m->sigma_s = [](const V3 &) { return 1.3; };
    o.material = m;
    ElementWeno material(pn, o);
    Field a = constant.interpolate([&](const V3 &x) {
        Vec r = Vec::Zero(pn.q);
        set_low(r, Four(1 + .3 * std::sin(2 * pi * x[0]), .2 * std::cos(2 * pi * x[1]), .1, 0));
        for (int k = 4; k < pn.q; ++k) r[k] = .02 * std::sin(2 * pi * (x[0] + k * x[1]));
        return r;
    });
    Field b = a;
    const double dt = .5 * constant.step_bound();
    for (int n = 0; n < 4; ++n) {
        a = constant.advance(n * dt, dt, a);
        b = material.advance(n * dt, dt, b);
    }
    check(maxabs(a - b) < 1e-13, "constant material path equals constant-coefficient path (4 steps)", maxabs(a - b));
}
// Global particle balance of transport + reaction with open boundaries, both integrators.
void global_balance(bool ssprk43) {
    Angular pn(5);
    ElementWenoOptions o;
    o.degree = 2, o.dimension = 2, o.elements = 7, o.periodic = false, o.length = 7, o.ssprk43 = ssprk43;
    o.reduce_angular = true;
    auto m = std::make_shared<ElementWenoMaterial>();
    m->sigma_a = [](const V3 &x) { return (std::floor(x[0]) == 1 && std::floor(x[1]) == 1) ? 10. : 0.; };
    m->sigma_s = [](const V3 &x) { return (std::floor(x[0]) == 1 && std::floor(x[1]) == 1) ? 0. : 1.; };
    m->source = [](const V3 &x) { return x[0] >= 3 && x[0] <= 4 && x[1] >= 3 && x[1] <= 4 ? 1. : 0.; };
    o.material = m;
    ElementWeno s(pn, o);
    Field a = s.interpolate([&](const V3 &) {
        Vec r = Vec::Zero(pn.q);
        r[0] = 1e-10 / std::sqrt(4 * pi);
        return r;
    });
    auto mass = [&](const Field &f) {
        double total = 0;
        for (int i = 0; i < s.nodes; ++i) total += s.lumped[i] * project(f.row(i).transpose())[0];
        return total;
    };
    const double m0 = mass(a);
    ElementWenoStats st;
    const double dt = s.step_bound();
    bool cone = true;
    for (int n = 0; n < 40; ++n) {
        a = s.advance(n * dt, dt, a, &st);
        for (int i = 0; i < s.nodes; ++i) cone = cone && exact::cone(project(a.row(i).transpose()));
    }
    const double closure = mass(a) - m0 + st.boundary_outflow + st.absorbed - st.injected - st.nudge_mass;
    const std::string tag = std::string("balance ") + (ssprk43 ? "SSPRK(4,3)" : "SSPRK3") + ": ";
    check(std::abs(closure) < 1e-12 * st.injected, tag + "M(T) - M(0) + B + A - S - N = 0", closure / st.injected);
    check(st.injected > 0 && st.absorbed > 0 && st.boundary_outflow >= 0, tag + "source, absorption, outflow", st.injected);
    check(cone, tag + "all nodes in the M1 cone", double(st.nudged_states));
}
// Own 3D lattice extension: material integrals equal the independent cell count, balance closes, cone kept.
void lattice3d() {
    const auto counts = cases::lattice3d_counts();
    check(counts.absorbing == 62 && counts.strong_scattering + counts.scattering + counts.absorbing == 343,
          "lattice-3d: 62 absorbing cells, 343 in total", counts.absorbing);
    for (int p = 1; p <= 2; ++p) {
        Angular pn(3);
        ElementWenoOptions o;
        o.degree = p, o.dimension = 3, o.elements = 7, o.length = 7, o.periodic = false, o.ssprk43 = true;
        auto m = std::make_shared<ElementWenoMaterial>();
        m->sigma_a = cases::lattice3d_sigma_a;
        m->sigma_s = cases::lattice3d_sigma_s;
        m->source = cases::lattice3d_source;
        o.material = m;
        ElementWeno s(pn, o);
        const std::string tag = "lattice-3d p=" + std::to_string(p) + ": ";
        const double worst = std::max({std::abs(s.integral_sigma_a - counts.sigma_a), std::abs(s.integral_sigma_s - counts.sigma_s),
                                       std::abs(s.integral_source - counts.source), std::abs(s.integral_volume - 343.)});
        check(worst < 1e-10, tag + "material integrals equal the cell count (620, 429, 1, 343)", worst);
        Field a = s.interpolate([&](const V3 &) { Vec r = Vec::Zero(pn.q); r[0] = 1e-10 / std::sqrt(4 * pi); return r; });
        auto mass = [&](const Field &f) {
            double total = 0;
            for (int i = 0; i < s.nodes; ++i) total += s.lumped[i] * project(f.row(i).transpose())[0];
            return total;
        };
        const double m0 = mass(a), dt = s.step_bound();
        ElementWenoStats st;
        bool cone = true;
        for (int n = 0; n < 8; ++n) {
            a = s.advance(n * dt, dt, a, &st);
            for (int i = 0; i < s.nodes; ++i) cone = cone && exact::cone(project(a.row(i).transpose()));
        }
        const double closure = mass(a) - m0 + st.boundary_outflow + st.absorbed - st.injected - st.nudge_mass;
        check(std::abs(closure) < 1e-12 * st.injected, tag + "balance closes", closure / st.injected);
        check(cone && st.absorbed > 0, tag + "cone kept, absorption active", st.absorbed);
    }
}
} // namespace
int main() {
    try {
        reaction_matches_path_b();
        special_cases();
        realizability_and_ledger();
        nodal_averages();
        constant_material_equals_constant_path();
        global_balance(false);
        global_balance(true);
        lattice3d();
    } catch (const std::exception &e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    std::cout << (failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
