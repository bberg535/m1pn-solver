#include "adaptive_pn.hpp"
#include "element_weno.hpp"
#include <iostream>

using namespace m1pn;
namespace {

int failures = 0;
void check(bool ok, const char *name, double value = 0.) {
    std::cout << (ok ? "ok   " : "FAIL ") << name << " [" << value << "]\n";
    failures += !ok;
}

Vec smooth(const Angular &pn, const V3 &x) {
    Vec a(pn.q);
    set_low(a, Four(1. + .2 * std::cos(2 * pi * x[0]),
                    .18 * std::sin(2 * pi * x[0]), .1 * std::cos(2 * pi * x[1]), 0.));
    for (int j = 4; j < pn.q; ++j)
        a[j] = .04 * std::sin(2 * pi * (x[0] + j * x[1] + x[2])) + .02;
    return a;
}

void uniform_recovery(int dimension) {
    constexpr int count = 4;
    int cells = 1;
    for (int d = 0; d < dimension; ++d)
        cells *= count;
    Angular pn(3);
    AdaptivePn scheme(pn, dimension, count, std::vector<int>(cells, 3));
    ElementWenoOptions o;
    o.degree = 1;
    o.dimension = dimension;
    o.elements = count;
    o.periodic = true;
    o.stabilize = false;
    ElementWeno existing(pn, o);
    const auto initial = [&](const V3 &x) { return smooth(pn, x); };
    const auto local_rate = scheme.rate(scheme.interpolate(initial));
    const Field existing_rate = existing.linear_operator(existing.interpolate(initial), 0.);
    double defect = 0.;
    for (int k = 0; k < cells; ++k) {
        const Field local = scheme.element(local_rate, k);
        for (int i = 0; i < scheme.local_nodes(); ++i) {
            const auto x = scheme.node_position(k, i);
            int node = 0, stride = 1;
            for (int d = 0; d < dimension; ++d) {
                node += static_cast<int>(std::lround(x[d] * count)) * stride;
                stride *= count;
            }
            defect = std::max(defect, maxabs(local.row(i) - existing_rate.row(node)));
        }
    }
    check(defect < 3e-13, "uniform order recovers existing ElementWeno transport", defect);
    check(scheme.dofs() == std::size_t(cells) * pn.q, "uniform degree space has the existing DOF count");
}

void mixed_and_transfer(int dimension) {
    constexpr int count = 4;
    int cells = 1;
    for (int d = 0; d < dimension; ++d)
        cells *= count;
    Angular pn(5);
    std::vector<int> orders(cells), next(cells);
    for (int k = 0; k < cells; ++k) {
        orders[k] = 1 + 2 * ((k % count + k / count) % 3);
        next[k] = 1 + 2 * ((k % count + k / count + 1) % 3);
    }
    AdaptivePn scheme(pn, dimension, count, orders), new_scheme(pn, dimension, count, next);
    auto state = scheme.interpolate([&](const V3 &x) { return smooth(pn, x); });
    AdaptivePnDiagnostics stats;
    const auto slope = scheme.rate(state, false, &stats);
    check(scheme.mixed_faces() > 0 && stats.face_dissipation > 1e-6, "mixed faces dissipate a nonzero jump",
          stats.face_dissipation);
    const double defect = std::abs(scheme.inner_product(state, slope) + stats.face_dissipation);
    check(defect < 3e-13, "mixed-face energy identity", defect);
    const double balance = maxabs(scheme.integrated_moments(slope));
    check(balance < 3e-13, "unlimited rate conserves density and all three currents", balance);

    const auto moved = new_scheme.project_from(scheme, state);
    check(maxabs(moved[0] - state[0]) == 0. && maxabs(moved[1] - state[1]) == 0.,
          "mixed projection preserves all low DOFs bit for bit");
    check(new_scheme.inner_product(moved, moved) < scheme.inner_product(state, state),
          "mixed projection decreases energy");
    const auto roundtrip = scheme.project_from(new_scheme, moved);
    const double orthogonality = std::abs(scheme.inner_product(state, roundtrip) - new_scheme.inner_product(moved, moved));
    check(orthogonality < 3e-13, "mixed L2 projection satisfies orthogonality", orthogonality);

    double max_energy_change = -1.;
    for (int step = 0; step < 20; ++step) {
        const auto after = scheme.advance(state, .025 / count);
        max_energy_change = std::max(max_energy_change,
                                     scheme.inner_product(after, after) - scheme.inner_product(state, state));
        state = after;
    }
    check(max_energy_change <= 2e-13, "unlimited SSPRK43 decreases energy on the tested steps",
          max_energy_change);
}

void limited_mixed() {
    Angular pn(3);
    AdaptivePn scheme(pn, 2, 4, {1, 3, 1, 3, 3, 3, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1});
    auto state = scheme.interpolate([&](const V3 &x) {
        Vec a = 5. * smooth(pn, x);
        const double density = 1. + .2 * std::sin(2 * pi * x[0]);
        set_low(a, Four(density, .97 * density * std::cos(2 * pi * x[1]), 0., 0.));
        return a;
    });
    AdaptivePnDiagnostics stats;
    const auto rate = scheme.rate(state, true, &stats);
    check(stats.raw_low_defect < 3e-13, "unlimited M1 contributions recover new PN target", stats.raw_low_defect);
    check(stats.element_balance_defect < 3e-13, "element corrections sum to zero", stats.element_balance_defect);
    check(stats.limited_elements > 0, "M1 limiter engages on rough mixed-order data", stats.limited_elements);
    const double rate_balance = maxabs(scheme.integrated_moments(rate));
    check(rate_balance < 3e-13, "limited rate conserves the four low moments", rate_balance);

    const Four initial_balance = scheme.integrated_moments(state);
    int limited = 0;
    for (int step = 0; step < 20; ++step)
        state = scheme.advance(state, .3 * scheme.euler_cone_step(), true, &limited);
    check(scheme.min_cone_margin(state) >= 0. && limited > 0, "all SSPRK stages pass exact cone admission",
          scheme.min_cone_margin(state));
    const double drift = maxabs(scheme.integrated_moments(state) - initial_balance);
    check(drift < 3e-13, "limited SSPRK43 conserves the four low moments", drift);
}

void sensor_blind_spot() {
    Angular pn(3);
    AdaptivePn high(pn, 1, 8, std::vector<int>(8, 3));
    const auto a = high.interpolate([&](const V3 &x) {
        Vec u = Vec::Zero(pn.q);
        u[0] = 1 / std::sqrt(4 * pi);
        u[index(2, 0)] = .8 * std::cos(2 * pi * x[0]) / std::sqrt(20 * pi);
        return u;
    });
    const auto orders = high.select_orders(a, .02, .05);
    check(orders == std::vector<int>(8, 1), "zero current coarsens a nonzero quadrupole everywhere");
    AdaptivePn low(pn, 1, 8, orders);
    const auto b = low.project_from(high, a);
    const double lost_rate = maxabs(high.rate(a)[1] - low.rate(b)[1]);
    check(lost_rate > .01, "coarsening loses a measurable initial current derivative", lost_rate);
}

} // namespace

int main() {
    for (int dimension : {1, 2, 3}) {
        uniform_recovery(dimension);
        mixed_and_transfer(dimension);
    }
    limited_mixed();
    sensor_blind_spot();
    return failures > 0;
}
