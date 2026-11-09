// The low-moment fast path must reproduce the all-Q path to rounding accuracy.
#include "time.hpp"
#include <iomanip>
#include <iostream>

using namespace m1pn;

namespace {
int checks = 0;

void close(double error, double tolerance, const std::string &name) {
    std::cout << std::setprecision(17) << "{\"check\":\"" << name << "\",\"value\":" << error
              << ",\"passed\":" << (error <= tolerance ? "true" : "false") << "}\n";
    require(std::isfinite(error) && error <= tolerance, name);
    ++checks;
}

// A sharp collimated pulse keeps the cone limiter active on the coarse mesh.
Field rough_state(const Spatial &fe, const Angular &pn) {
    Field u = Field::Zero(fe.nodes, pn.q);
    for (int i = 0; i < fe.nodes; ++i) {
        const V3 x = fe.coordinates.row(i).transpose();
        Vec row = Vec::Zero(pn.q);
        row[0] = .002 / std::sqrt(4 * pi);
        const double pulse = std::exp(-(x - V3(.4, .5, 0.)).squaredNorm() / (2 * .15 * .15));
        row += pulse * harmonics(pn.order, V3::UnitX());
        u.row(i) = row.transpose();
    }
    return u;
}

Inflow make(int kind) {
    Inflow g;
    if (kind == 1) { // constant background plus x,t-dependent collimated atoms
        const double bg = .001;
        g.density = [bg](const V3 &, double, const V3 &) { return bg / (4 * pi); };
        g.constant_density = bg / (4 * pi);
        g.atoms = [](const V3 &x, double t) {
            auto p = [](const V3 &d) { return std::exp(-d.squaredNorm() / (2 * .1 * .1)); };
            return std::vector<Atom>{{V3::UnitX(), p(x - t * V3::UnitX() - V3(.3, .5, 0.))},
                                     {V3::UnitY(), p(x - t * V3::UnitY() - V3(.5, .3, 0.))}};
        };
    } else if (kind == 2) { // angle- and time-dependent density without the constant hint
        g.density = [](const V3 &x, double t, const V3 &w) {
            return .02 * (1 + .2 * x[0] + .1 * std::sin(t)) * (1 + .2 * w[1]) / (4 * pi);
        };
    }
    return g;
}

void compare(int dimension, int order, int rk, int kind) {
    MeshSpec spec;
    spec.dimension = dimension;
    spec.elements = dimension == 3 ? 3 : 6;
    const std::string label = "D" + std::to_string(dimension) + "_P" + std::to_string(order) + "_RK" +
                              std::to_string(rk) + "_inflow" + std::to_string(kind);
    Spatial fe(spec, 1, {}, false);
    Angular pn(order);
    Numerics n;
    n.r2 = true;
    n.stabilization = "m1pn_mass_correction";
    n.mass_mode = "lumped";
    Inflow inflow = make(kind);
    Transport audited(pn, fe, n, inflow);
    n.ledger_stride = 1 << 30;
    Transport fast(pn, fe, n, inflow);
    require(bool(fast.fast_mc), "Standard method on a Cartesian mesh must select the fast path");
    fast.advance_count = 1; // never the audited attempt
    TimeControl ca(audited, spec, rk), cf(fast, spec, rk);
    Field ua = rough_state(fe, pn), uf = ua;
    const double dt = .4 * std::min(fe.cfl_step, ca.step_limit);
    double t = 0, worst = 0, worst_boundary = 0, scale = maxabs(ua);
    long active = 0;
    double worst_alpha = 0;
    for (int step = 0; step < 12; ++step) {
        Step a = advance(audited, ca, t, dt, ua), b = advance(fast, cf, t, dt, uf);
        worst = std::max(worst, maxabs(a.state - b.state));
        worst_boundary = std::max(worst_boundary, maxabs(a.integrated_boundary - b.integrated_boundary));
        require(a.stages.size() == b.stages.size(), "Stage count differs");
        for (std::size_t r = 0; r < a.stages.size(); ++r) {
            require(a.stages[r].active_pairs == b.stages[r].active_pairs &&
                        a.stages[r].active_faces == b.stages[r].active_faces,
                    "Limiter activation differs in " + label);
            active += b.stages[r].active_pairs;
            worst_alpha = std::max(worst_alpha, std::abs(a.stages[r].minimum_alpha - b.stages[r].minimum_alpha));
        }
        ua = std::move(a.state);
        uf = std::move(b.state);
        t += dt;
    }
    close(worst_alpha, 1e-9, label + "_alpha");
    close(worst, 1e-12 * std::max(1., scale), label + "_state");
    close(worst_boundary, 1e-12 * std::max(1., scale), label + "_boundary_ledger");
    // In 1D the collimated pulse stays admissible; 2D and 3D must exercise the limiter.
    require(active > 0 || dimension == 1, "Limiter stayed inactive in " + label);
}
} // namespace

int main() {
    try {
        exact::environment();
        for (int kind = 0; kind <= 2; ++kind) {
            compare(2, 3, 3, kind);
            compare(2, 7, 3, kind);
            compare(3, 3, 3, kind);
            compare(1, 3, 3, kind);
        }
        compare(2, 5, 2, 1);
        std::cout << "{\"status\":\"passed\",\"checks\":" << checks << "}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
