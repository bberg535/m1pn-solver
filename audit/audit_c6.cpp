// C6: low-order boundary consistency. For a spatially uniform state with compatible kinetic inflow (g = psi on the
// incoming hemisphere) the PN target is stationary (for PN-compatible psi); the *low-order* control (all antidiffusion
// off, hook audit_low_order) should then be stationary as well. Compares the legacy ghost (incoming half-range moments)
// with the full-sphere ghost (incoming kinetic moments + outgoing moments of the positive PN trace).
#include "audit_common.hpp"

int main() {
    struct F { std::string name; std::function<double(const V3 &)> psi; bool positive; };
    const std::vector<F> fs = {
        {"iso", [](const V3 &) { return 1.; }, true},
        {"1+0.5*wx", [](const V3 &w) { return 1 + .5 * w[0]; }, true},
        {"(1-wx)^3", [](const V3 &w) { return std::pow(1 - w[0], 3); }, true},
        {"1+0.9*wy", [](const V3 &w) { return 1 + .9 * w[1]; }, true},
        {"exp(3 wx)", [](const V3 &w) { return std::exp(3 * w[0]); }, true},
        {"exp(8 wx)", [](const V3 &w) { return std::exp(8 * w[0]); }, true},
    };
    for (int N : {3, 5, 7}) {
        Angular pn(N);
        for (const auto &f : fs) {
            for (bool ghost : {false, true}) {
                Vec c = Vec::Zero(pn.q);
                for (const auto &p : sphere_rule(N + 8)) c += p.weight * f.psi(p.omega) * harmonics(N, p.omega);
                double minpsi = 1e300;
                for (const auto &p : sphere_rule(N + 6)) minpsi = std::min(minpsi, harmonics(N, p.omega).dot(c));
                Inflow in;
                in.density = [&](const V3 &, double, const V3 &om) { return f.psi(om); };
                ElementWenoOptions o;
                o.degree = 2; o.dimension = 2; o.elements = 4; o.periodic = false;
                apply_env(o);
                o.fullsphere_ghost = ghost;
                ElementWeno s(pn, o, in);
                Field a = s.interpolate([&](const V3 &) { return c; });
                audit_low_order = true;
                const Field lo = s.rate(0, a);
                audit_low_order = false;
                const Field full = s.rate(0, a);
                double rep = 0;
                for (const auto &p : sphere_rule(N + 4)) rep = std::max(rep, std::abs(harmonics(N, p.omega).dot(c) - f.psi(p.omega)));
                J("C6_low_order_consistency").add("pn_order", N).str("psi", f.name).add("fullsphere_ghost", ghost)
                    .add("psi_representation_error", rep).add("min_psi_N", minpsi)
                    .add("max_abs_rate_density_lowOrder", lo.col(0).cwiseAbs().maxCoeff())
                    .add("max_abs_rate_moment_rows_lowOrder", std::max({lo.col(0).cwiseAbs().maxCoeff(), lo.col(1).cwiseAbs().maxCoeff(), lo.col(2).cwiseAbs().maxCoeff(), lo.col(3).cwiseAbs().maxCoeff()}))
                    .add("max_abs_rate_moment_rows_full", std::max({full.col(0).cwiseAbs().maxCoeff(), full.col(1).cwiseAbs().maxCoeff(), full.col(2).cwiseAbs().maxCoeff(), full.col(3).cwiseAbs().maxCoeff()}))
                    .add("u_norm", c.norm()).done();
            }
        }
    }
}
