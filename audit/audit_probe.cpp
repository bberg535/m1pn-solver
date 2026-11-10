// Boundary compatibility probe: does Boundary::exterior reproduce a PN-representable kinetic trace
// when the Onsager datum d is computed from the same positive kinetic function (g = psi on the incoming hemisphere)?
#include "audit_common.hpp"

int main() {
    for (int N : {3, 5, 7}) {
        Angular pn(N);
        CanonicalC1 c(pn);
        for (const auto &[nm, n] : {std::pair<std::string, V3>{"ex", V3::UnitX()}, {"-ex", -V3::UnitX()}, {"diag", V3(1, 1, 0).normalized()}}) {
            Boundary b(pn, c, n, false);
            struct F { std::string name; std::function<double(const V3 &)> psi; };
            std::vector<F> fs = {
                {"iso", [](const V3 &) { return 1.; }},
                {"linear_x:1+0.5wx", [](const V3 &w) { return 1 + .5 * w[0]; }},
                {"cubic:(1-wx)^3", [](const V3 &w) { return std::pow(1 - w[0], 3); }},
                {"linear_n:1+0.5 n.w", [&](const V3 &w) { return 1 + .5 * n.dot(w); }},
                {"linear_t:1+0.5 t.w", [&](const V3 &w) { return 1 + .5 * b.frame.col(0).dot(w); }},
                {"quad_n:1+n.w^2", [&](const V3 &w) { return 1 + std::pow(n.dot(w), 2); }}};
            for (auto &f : fs) {
                if (N < 3 && f.name.find("cubic") != std::string::npos) continue;
                Vec u = Vec::Zero(pn.q);
                for (const auto &p : sphere_rule(N + 8)) u += p.weight * f.psi(p.omega) * harmonics(N, p.omega);
                double rep = 0;
                for (const auto &p : sphere_rule(N + 4)) rep = std::max(rep, std::abs(harmonics(N, p.omega).dot(u) - f.psi(p.omega)));
                Inflow in;
                in.density = [&](const V3 &, double, const V3 &w) { return f.psi(w); };
                Incoming d = b.data(in, V3::Zero(), 0);
                const Vec ext = b.exterior(u, d.d);
                const Vec out_amp = b.Rplus.transpose() * u, in_amp_u = b.Rminus.transpose() * u, in_amp_ext = b.Rminus.transpose() * ext;
                const Mat A = pn.normal(n);
                J("PROBE_boundary_compat").add("pn_order", N).str("normal", nm).str("psi", f.name).add("psi_representation_err", rep)
                    .add("ext_minus_u_norm", (ext - u).norm()).add("u_norm", u.norm())
                    .add("incoming_amp_u", in_amp_u.norm()).add("incoming_amp_ext", in_amp_ext.norm())
                    .add("PN_boundary_integrand_norm", (.5 * (A - Mat::Identity(pn.q, pn.q)) * (u - ext)).norm())
                    .done();
            }
        }
    }
}
