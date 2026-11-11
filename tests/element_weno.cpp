// Structural checks of the element-based WENO variant.
#include "element_weno.hpp"
#include <algorithm>
#include <array>
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

// Full SSPRK(4,3) steps at the realizability bound 2 dt_bound keep rough data in the cone.
void rough_ssprk43() {
    Angular pn(3);
    ElementWenoOptions o;
    o.degree = 2;
    o.dimension = 2;
    o.elements = 6;
    o.periodic = false;
    o.ssprk43 = true;
    ElementWeno scheme(pn, o);
    std::mt19937 rng(11);
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
    check(scheme.realizability_step() == 2 * scheme.dt_bound, "SSPRK(4,3) doubles the realizability step");
    bool cone = true;
    for (int n = 0; n < 3; ++n) {
        a = scheme.advance(.1 * n, scheme.realizability_step(), a);
        for (int i = 0; i < scheme.nodes; ++i)
            cone = cone && exact::cone(project(a.row(i).transpose()));
    }
    check(cone, "SSPRK(4,3) steps of 2 dt_bound keep rough open data in the cone");
}
// Linear stability behind high_order_step(): power iteration on the amplification operator R(dt L) of the
// periodic high-order target. At dt = high_order_step() no mode may grow; at 1.25 times the real-axis limit
// some mode must grow (otherwise the test could not detect a wrong constant).
double amplification(const ElementWeno &s, double dt, double gamma, bool ssprk43) {
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> u(-1, 1);
    Field x(s.nodes, s.pn.q);
    for (int i = 0; i < s.nodes; ++i)
        for (int c = 0; c < s.pn.q; ++c) x(i, c) = u(rng);
    x /= x.norm();
    const int iterations = 240;
    double log_sum = 0;
    auto L = [&](const Field &y) { return s.linear_operator(y, gamma); };
    for (int k = 0; k < iterations; ++k) {
        Field y;
        if (ssprk43) {
            const Field a1 = x + .5 * dt * L(x), a2 = a1 + .5 * dt * L(a1);
            const Field a3 = (2. / 3) * x + (1. / 3) * (a2 + .5 * dt * L(a2));
            y = a3 + .5 * dt * L(a3);
        } else {
            const Field a1 = x + dt * L(x);
            const Field a2 = .75 * x + .25 * (a1 + dt * L(a1));
            y = (1. / 3) * x + (2. / 3) * (a2 + dt * L(a2));
        }
        const double growth = y.norm();
        if (k >= iterations / 2) log_sum += std::log(growth);
        x = y / growth;
    }
    return std::exp(log_sum / (iterations - iterations / 2));
}
void high_order_stability(int degree, bool ssprk43) {
    Angular pn(3);
    ElementWenoOptions o;
    o.degree = degree;
    o.dimension = 2;
    o.elements = 6;
    o.periodic = true;
    o.ssprk43 = ssprk43;
    ElementWeno scheme(pn, o);
    const std::string tag = std::string(ssprk43 ? "SSPRK(4,3)" : "SSPRK3") + " p=" + std::to_string(degree) + ": ";
    const double dt = scheme.high_order_step();
    double worst = 0;
    for (double gamma : {0., 1.})
        worst = std::max(worst, amplification(scheme, dt, gamma, ssprk43));
    check(worst <= 1 + 1e-7, tag + "no growth at high_order_step (periodic)", worst);
    const double beyond = amplification(scheme, dt / .9 * 1.25, 0., ssprk43);
    check(beyond > 1.001, tag + "growth beyond the real-axis limit", beyond);
}

// Angular parity reduction: on even data the reduced scheme reproduces the full scheme (rate and steps).
Inflow in_plane_beams(double angle) {
    Inflow in;
    in.density = [](const V3 &, double, const V3 &) { return 1e-3 / (4 * pi); };
    in.constant_density = 1e-3 / (4 * pi);
    const V3 tilted = V3(std::cos(angle), 0, std::sin(angle)).normalized();
    in.atoms = [tilted](const V3 &x, double t) {
        auto p = [](const V3 &d) { return std::exp(-d.squaredNorm() / (2 * .1 * .1)); };
        return std::vector<Atom>{{V3::UnitX(), p(x - t * V3::UnitX() - V3(.3, .5, 0))},
                                 {tilted, p(x - V3(0, .5, 0))}};
    };
    return in;
}
void angular_reduction(int dimension, bool periodic, double filter, double sigma_s) {
    Angular pn(7);
    ElementWenoOptions o;
    o.degree = 2;
    o.dimension = dimension;
    o.elements = dimension == 1 ? 10 : 5;
    o.periodic = periodic;
    o.filter_strength = filter;
    o.sigma_s = sigma_s;
    o.sigma_a = sigma_s / 4;
    const Inflow in = periodic ? Inflow{} : in_plane_beams(0.);
    ElementWeno full(pn, o, in);
    o.reduce_angular = true;
    ElementWeno reduced(pn, o, in);
    const std::string tag = "reduction d=" + std::to_string(dimension) + (periodic ? " periodic" : " open") +
                            (filter > 0 || sigma_s > 0 ? " +filter/scattering" : "") + ": ";
    // 2D: z-even harmonics (N+1)(N+2)/2 plus J_z; 1D: y- and z-even sum_l (floor(l/2)+1) plus J_y, J_z.
    int expected = dimension == 2 ? (pn.order + 1) * (pn.order + 2) / 2 + 1 : 2;
    if (dimension == 1)
        for (int l = 0; l <= pn.order; ++l)
            expected += l / 2 + 1;
    check(reduced.moments == expected, tag + "expected number of active moments", reduced.moments);
    // Even data: the full state of the reduced interpolation.
    const Field even = reduced.interpolate([&](const V3 &x) { return state(pn, x); });
    const Field even_full = reduced.expand(even);
    const double scale = maxabs(even_full);
    const double rate_error = maxabs(full.rate(.05, even_full) - reduced.expand(reduced.rate(.05, even)));
    check(rate_error < 1e-11 * std::max(1., scale), tag + "rate equals the full rate on even data", rate_error);
    Field a = even_full, b = even;
    for (int n = 0; n < 3; ++n) {
        a = full.advance(.01 * n, .5 * full.step_bound(), a);
        b = reduced.advance(.01 * n, .5 * full.step_bound(), b);
    }
    const double step_error = maxabs(a - reduced.expand(b));
    check(step_error < 1e-11 * std::max(1., scale), tag + "three steps equal the full scheme", step_error);
    double odd = 0;
    for (int i = 0; i < full.nodes; ++i)
        for (int k = 0; k < pn.q; ++k)
            if (std::find(reduced.active.begin(), reduced.active.end(), k) == reduced.active.end())
                odd = std::max(odd, std::abs(a(i, k)));
    check(odd < 1e-14, tag + "full scheme keeps odd moments at zero", odd);
    if (!periodic) {
        check(reduced.dropped_inflow(.1) < 1e-12, tag + "in-plane inflow is even", reduced.dropped_inflow(.1));
        ElementWeno tilted(pn, o, in_plane_beams(.4));
        check(tilted.dropped_inflow(0) > 1e-3, tag + "out-of-plane beam is detected as odd", tilted.dropped_inflow(0));
    }
}

// Mirror sides: the quarter domain [1/2,1]^2 with mirrors at x = 1/2 and y = 1/2 reproduces the full domain
// [0,1]^2 on mirror-symmetric data (rate on every shared node and full SSPRK steps).
Vec radial_beam(const Angular &pn, const V3 &x) {
    const V3 r(x[0] - .5, x[1] - .5, 0);
    const double g = .8 * r.squaredNorm() / (.02 + r.squaredNorm()) * std::exp(-r.squaredNorm() / .1);
    Vec a = Vec::Zero(pn.q);
    a[0] = 1e-3 / std::sqrt(4 * pi); // thin background: the limiter is active
    if (g > 0)
        a += g * harmonics(pn.order, r.normalized());
    return a;
}
void mirror_quarter(bool reduce, double sigma_s, bool ssprk43) {
    Angular pn(5);
    ElementWenoOptions o;
    o.degree = 2;
    o.dimension = 2;
    o.elements = 8;
    o.periodic = false;
    o.reduce_angular = reduce;
    o.sigma_s = sigma_s;
    o.ssprk43 = ssprk43;
    ElementWeno full(pn, o);
    o.elements = 4;
    o.origin = V3(.5, .5, 0);
    o.length = .5;
    o.mirror[0] = o.mirror[2] = true;
    ElementWeno quarter(pn, o);
    const std::string tag = std::string("mirror quarter") + (reduce ? " +reduction" : "") +
                            (sigma_s > 0 ? " +scattering" : "") + (ssprk43 ? " SSPRK(4,3)" : "") + ": ";
    check(std::abs(quarter.dt_bound / full.dt_bound - 1) < 1e-14, tag + "same realizability bound",
          quarter.dt_bound / full.dt_bound - 1);
    Field a = full.interpolate([&](const V3 &x) { return radial_beam(pn, x); });
    Field b = quarter.interpolate([&](const V3 &x) { return radial_beam(pn, x); });
    const int shift = full.per_direction / 2;
    auto compare = [&](const Field &f, const Field &g) {
        double worst = 0;
        for (int j = 0; j < quarter.per_direction; ++j)
            for (int i = 0; i < quarter.per_direction; ++i)
                worst = std::max(worst, (f.row((j + shift) * full.per_direction + i + shift) -
                                         g.row(j * quarter.per_direction + i)).cwiseAbs().maxCoeff());
        return worst;
    };
    const double scale = maxabs(a);
    ElementWenoStats stats;
    const double rate_error = compare(full.rate(0, a, &stats), quarter.rate(0, b));
    check(rate_error < 1e-11 * std::max(1., scale), tag + "rate equals the full domain on shared nodes", rate_error);
    check(stats.limited_elements > 0, tag + "the limiter is active in the test", double(stats.limited_elements));
    const double dt = .5 * std::min(full.step_bound(), quarter.step_bound());
    for (int n = 0; n < 4; ++n) {
        a = full.advance(n * dt, dt, a);
        b = quarter.advance(n * dt, dt, b);
    }
    const double step_error = compare(a, b);
    check(step_error < 1e-11 * std::max(1., scale), tag + "four steps equal the full domain", step_error);
    double mass_full = 0, mass_quarter = 0;
    for (int i = 0; i < full.nodes; ++i)
        mass_full += full.lumped[i] * a(i, 0);
    for (int i = 0; i < quarter.nodes; ++i)
        mass_quarter += quarter.lumped[i] * b(i, 0);
    check(std::abs(mass_full - 4 * mass_quarter) < 1e-13 * mass_full, tag + "quarter mass is a quarter of the total",
          std::abs(mass_full - 4 * mass_quarter));
}

// Per-axis element counts (cubic cells): data that depend on x only give the same solution on a 12 x 4 (2D) or
// 6 x 2 x 3 (3D) periodic box as on the cube, node by node along x.
Vec x_wave(const Angular &pn, const V3 &x) {
    Vec a = Vec::Zero(pn.q);
    const double rho = 1 + .3 * std::sin(2 * pi * x[0] / .5);
    set_low(a, Four(rho, .2 * rho * std::cos(2 * pi * x[0] / .5), 0, 0));
    for (int k = 4; k < pn.q; ++k)
        a[k] = .02 * std::sin(2 * pi * x[0] / .5 + k);
    return a;
}
void per_axis_counts(int dimension) {
    Angular pn(3);
    ElementWenoOptions o;
    o.degree = 2;
    o.dimension = dimension;
    o.periodic = true;
    o.elements = dimension == 2 ? 12 : 6;
    o.length = .5;
    ElementWeno cube(pn, o);
    o.elements_per_axis = dimension == 2 ? std::array<int, 3>{12, 4, 0} : std::array<int, 3>{6, 2, 3};
    ElementWeno box(pn, o);
    const std::string tag = "per-axis counts d=" + std::to_string(dimension) + ": ";
    check(box.nodes * (dimension == 2 ? 3 : 6) == cube.nodes, tag + "node count", box.nodes);
    Field a = cube.interpolate([&](const V3 &x) { return x_wave(pn, x); });
    Field b = box.interpolate([&](const V3 &x) { return x_wave(pn, x); });
    auto compare = [&](const Field &f, const Field &g) {
        double worst = 0;
        for (int i = 0; i < box.nodes; ++i) {
            // Same (ix, iy, iz) index in both meshes.
            int k = i, j = 0, scale = 1;
            for (int d = 0; d < dimension; ++d) {
                j += (k % box.nodes_per_axis[d]) * scale;
                k /= box.nodes_per_axis[d];
                scale *= cube.nodes_per_axis[d];
            }
            worst = std::max(worst, (f.row(j) - g.row(i)).cwiseAbs().maxCoeff());
        }
        return worst;
    };
    check(compare(cube.rate(0, a), box.rate(0, b)) < 1e-11, tag + "rate equals the cube", compare(cube.rate(0, a), box.rate(0, b)));
    const double dt = .5 * cube.step_bound();
    for (int n = 0; n < 3; ++n) {
        a = cube.advance(n * dt, dt, a);
        b = box.advance(n * dt, dt, b);
    }
    check(compare(a, b) < 1e-11, tag + "three steps equal the cube", compare(a, b));
    check(std::abs(box.dt_bound / cube.dt_bound - 1) < 1e-13, tag + "same realizability bound", box.dt_bound / cube.dt_bound - 1);
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
        rough_ssprk43();
        angular_reduction(2, false, 0, 0);
        angular_reduction(2, true, 0, 0);
        angular_reduction(1, true, 0, 0);
        angular_reduction(2, false, 20, 2);
        mirror_quarter(false, 0, false);
        mirror_quarter(true, 0, false);
        mirror_quarter(true, 3, true);
        per_axis_counts(2);
        per_axis_counts(3);
        for (int p = 1; p <= 4; ++p) {
            high_order_stability(p, false);
            high_order_stability(p, true);
        }
        {
            Angular pn(5);
            ElementWenoOptions o;
            o.degree = 2; o.dimension = 2; o.elements = 4; o.filter_strength = 27.27;
            ElementWeno scheme(pn, o);
            const Field a = scheme.interpolate([&](const V3 &x) { return state(pn, x); });
            const Field f = scheme.angular_filter(a, .01);
            double density = 0, worst = 0; bool cone = true;
            for (int i = 0; i < scheme.nodes; ++i) {
                density = std::max(density, std::abs(f(i, 0) - a(i, 0)));
                const double lhs = f.row(i).segment(1, 3).norm(), rhs = a.row(i).segment(1, 3).norm();
                worst = std::max(worst, lhs - rhs);
                cone = cone && exact::cone(project(f.row(i).transpose()));
                const double expected = std::exp(-.01 * 27.27 * std::pow(5. / 6, 4));
                worst = std::max(worst, std::abs(f(i, 35) - expected * a(i, 35)) - 1e-15);
            }
            check(density == 0, "filter keeps the density coefficient bitwise", density);
            check(worst <= 0, "filter damps degree one and matches exp(-tau sigma (l/(N+1))^4)", worst);
            check(cone, "filtered states stay in the cone");
        }
        {
            // Scattering and absorption: exact factors per degree, cone kept.
            Angular pn(5);
            ElementWenoOptions o;
            o.degree = 2; o.dimension = 2; o.elements = 4; o.sigma_s = 2; o.sigma_a = .5;
            ElementWeno scheme(pn, o);
            const Field a = scheme.interpolate([&](const V3 &x) { return state(pn, x); });
            const double tau = .03;
            const Field f = scheme.angular_filter(a, tau);
            double worst = 0; bool cone = true;
            for (int i = 0; i < scheme.nodes; ++i) {
                worst = std::max(worst, std::abs(f(i, 0) - std::exp(-tau * .5) * a(i, 0)));
                for (int k = 1; k < pn.q; ++k)
                    worst = std::max(worst, std::abs(f(i, k) - std::exp(-tau * 2.5) * a(i, k)));
                cone = cone && exact::cone(project(f.row(i).transpose()));
            }
            check(worst < 1e-15, "scattering/absorption factors exp(-tau(sa+ss[l>0]))", worst);
            check(cone, "scattered and absorbed states stay in the cone");
            // Spatially uniform periodic state: transport vanishes, one full
            // SSPRK3 step must reproduce the exact relaxation.
            o.periodic = true; o.sigma_a = 0;
            ElementWeno uniform(pn, o);
            Vec u0 = Vec::Zero(pn.q);
            set_low(u0, Four(1, .4, -.2, 0));
            for (int k = 4; k < pn.q; ++k) u0[k] = .02 * std::cos(k);
            const Field b = uniform.interpolate([&](const V3 &) { return u0; });
            const double dt = .01;
            const Field c = uniform.advance_ssprk3(0, dt, b);
            double err = 0;
            for (int i = 0; i < uniform.nodes; ++i) {
                err = std::max(err, std::abs(c(i, 0) - b(i, 0)));
                for (int k = 1; k < pn.q; ++k)
                    err = std::max(err, std::abs(c(i, k) - std::exp(-2 * dt) * b(i, k)));
            }
            check(err < 1e-12, "uniform state: SSPRK3 step equals exact scattering relaxation", err);
            o.ssprk43 = true;
            ElementWeno uniform43(pn, o);
            const Field c43 = uniform43.advance(0, dt, b);
            double err43 = 0;
            for (int i = 0; i < uniform43.nodes; ++i) {
                err43 = std::max(err43, std::abs(c43(i, 0) - b(i, 0)));
                for (int k = 1; k < pn.q; ++k)
                    err43 = std::max(err43, std::abs(c43(i, k) - std::exp(-2 * dt) * b(i, k)));
            }
            check(err43 < 1e-12, "uniform state: SSPRK(4,3) step equals exact scattering relaxation", err43);
        }
    } catch (const std::exception &e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    std::cout << (failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
