#include "time.hpp"
#include "base_bar.hpp"
#include <Eigen/Eigenvalues>
#include <iomanip>
#include <iostream>
#include <random>
using namespace m1pn;
namespace {
int checks = 0;
void check(bool good, const std::string &name, double value = 0) {
    std::cout << std::setprecision(17) << "{\"check\":\"" << name << "\",\"value\":" << value
              << ",\"passed\":" << (good ? "true" : "false") << "}\n";
    require(good, name);
    ++checks;
}
void near(double error, double tolerance, const std::string &name) {
    check(std::isfinite(error) && std::abs(error) < tolerance, name, error);
}
Field stress(const Spatial &f, int q) {
    Field a(f.nodes, q);
    for (int i = 0; i < f.nodes; ++i) {
        Vec u = Vec::Zero(q);
        set_low(u, Four(1 + .1 * f.coordinates(i, 0), .2, .1, .05));
        for (int k = 4; k < q; ++k)
            u[k] = 2. * std::sin((i + 1.) * (k + 3.));
        a.row(i) = u.transpose();
    }
    return a;
}
void cone_filter() {
    std::mt19937_64 random(20261021);
    std::uint64_t count = 0;
    auto compare = [&](const Four &v) {
        require(exact::cone(v) == exact::cone_dyadic(v),
                "Filtered cone differs from exact dyadic sign");
        ++count;
    };
    for (int k = 0; k < 200000; ++k) {
        Four v;
        for (double &x : v) {
            const auto bits = random();
            std::memcpy(&x, &bits, sizeof(x));
        }
        compare(v);
    }
    for (int exponent = -1074; exponent <= 1023; ++exponent) {
        const double a = std::ldexp(1., exponent);
        compare(Four(a, a, 0, 0));
        compare(Four(a, -a, 0, 0));
        for (int k = 0; k < 8; ++k) {
            Four v(0, a, a * (double(random() % 1024) / 1024),
                   a * (double(random() % 1024) / 1024));
            v[0] = std::hypot(v[1], v[2], v[3]);
            for (int delta = -2; delta <= 2; ++delta) {
                Four p = v;
                for (int j = 0; j < std::abs(delta); ++j)
                    p[0] = std::nextafter(p[0],
                                          delta < 0 ? 0. : std::numeric_limits<double>::infinity());
                compare(p);
            }
        }
    }
    compare(Four::Zero());
    compare(Four(-0., 0., -0., 0.));
    compare(Four(-1., 0., 0., 0.));
    check(true, "filtered_cone_matches_exact_dyadic", double(count));
}
void pair_limiter() {
    const Four beam(1, 1, 0, 0), witness(.25, .25, .125, 0);
    auto r = density_pair_limit(beam, beam, witness);
    check(exact::cone(beam + r.correction) && exact::cone(beam - r.correction),
          "vector_pair_beam_admissible");
    near(r.correction[0] - .25, 1e-15, "vector_pair_retains_beam_density");
    near(r.correction[1] - .25, 1e-15, "vector_pair_retains_longitudinal_momentum");
    near(r.correction[2], 1e-15, "vector_pair_removes_transverse_beam_momentum");
    Four interior(2, .2, -.1, .3), target(.01, .02, -.01, .005);
    r = density_pair_limit(interior, interior, target);
    check(r.target_pass && (r.correction - target).norm() == 0,
          "vector_pair_feasible_target_passthrough");
    r = density_pair_limit(interior, Four(1, -.1, .15, 0), Four(.5, 2, -1, .3));
    auto swapped = density_pair_limit(Four(1, -.1, .15, 0), interior, Four(-.5, -2, 1, -.3));
    check((r.correction + swapped.correction).norm() < 2e-14,
          "vector_pair_orientation_antisymmetry");
    check(exact::cone(interior + r.correction) &&
              exact::cone(Four(1, -.1, .15, 0) - r.correction),
          "vector_pair_general_endpoints");
    const Four vacuum = Four::Zero(), rest(1, 0, 0, 0), xbeam(1, 1, 0, 0);
    auto zero_left = density_pair_limit(xbeam, rest, Four(-1, -1, .2, 0));
    check(zero_left.correction[0] == -1 && exact::cone(xbeam + zero_left.correction) &&
              exact::cone(rest - zero_left.correction),
          "vector_pair_zero_corrected_density");
    auto density_only = density_pair_limit(rest, rest, Four(3, 0, 0, 0));
    check(density_only.correction[0] == 1 && density_only.correction.tail<3>().norm() == 0,
          "vector_pair_zero_momentum_segment");
    auto both_vacuum = density_pair_limit(vacuum, vacuum, Four(.2, .1, 0, 0));
    check(both_vacuum.correction.norm() == 0 && exact::cone(both_vacuum.correction),
          "vector_pair_double_vacuum");
    auto beam_longitudinal = density_pair_limit(xbeam, xbeam, Four(.1, .8, 0, 0));
    check(beam_longitudinal.correction[0] == .1 &&
              exact::cone(xbeam + beam_longitudinal.correction) &&
              exact::cone(xbeam - beam_longitudinal.correction),
          "vector_pair_q_zero_longitudinal");
    std::mt19937_64 random(20261022);
    std::uniform_real_distribution<double> uni(-1, 1);
    auto state = [&](int k) -> Four {
        if (k % 7 == 0) return Four::Zero();
        if (k % 7 == 1) return Four(1, 1, 0, 0);
        V3 d(uni(random), uni(random), uni(random));
        if (d.norm() == 0) d = V3::UnitX();
        d.normalize();
        double rho = .1 + std::abs(uni(random)) * 2;
        double f = std::abs(uni(random)) * .999;
        return Four(rho, rho * f * d[0], rho * f * d[1], rho * f * d[2]);
    };
    for (int k = 0; k < 500; ++k) {
        Four L = state(k), R = state(k + 2), t(uni(random), uni(random), uni(random), uni(random));
        auto q = density_pair_limit(L, R, t);
        check(exact::cone(L + q.correction) && exact::cone(R - q.correction),
              "vector_pair_random_cone_" + std::to_string(k));
        near((L + q.correction + R - q.correction - L - R).norm(), 2e-14,
             "vector_pair_random_conservation_" + std::to_string(k));
    }
}
void robust_base_bar() {
    const Four left = Four::Zero();
    const Four right(1, 1 - std::ldexp(1., -52), 0, 0);
    const V3 edge = V3::UnitX();
    const Four old = .5 * (left + right) -
                     (Levermore(right).flux(edge) - Levermore(left).flux(edge)) / 2;
    check(!exact::cone(old), "old_base_bar_rounding_counterexample");
    const Four stable = stable_base_bar(left, right, edge, 1);
    check(exact::cone(stable), "stable_base_bar_exact_cone");
    for (int k : {-1000, 0, 1000}) {
        const double scale = std::ldexp(1., k);
        check(exact::squared_gap(Four(5 * scale, 3 * scale, 4 * scale, 0)) == 0,
              "exact_beam_gap_scale_" + std::to_string(k));
    }
    for (int k : {-1073, -1000, 0, 1000}) {
        const double rho = std::ldexp(1., k);
        near(double(exact::normalized_gap(Four(rho, .5 * rho, 0, 0))) - .75, 1e-15,
             "normalized_gap_scale_" + std::to_string(k));
    }
    check(exact::normalized_gap(Four::Zero()) == 0 &&
              exact::normalized_gap(Four(1, 1, 0, 0)) == 0,
          "normalized_gap_vacuum_and_axis_beam");
    check(exact::cone(stable_base_bar(Four::Zero(), Four(1, 1, 0, 0), edge, 1)),
          "stable_base_bar_vacuum_beam");
}
void angular() {
    for (int n : {1, 2, 3, 5, 7, 9, 15, 17}) {
        Angular a(n);
        const auto rule = sphere_rule(n);
        Mat y(rule.size(), a.q);
        std::array<Mat, 3> weighted = {y, y, y};
        for (std::size_t p = 0; p < rule.size(); ++p) {
            y.row(p) = std::sqrt(rule[p].weight) * harmonics(n, rule[p].omega).transpose();
            for (int d = 0; d < 3; ++d)
                weighted[d].row(p) = rule[p].omega[d] * y.row(p);
        }
        near(maxabs(y.transpose() * y - Mat::Identity(a.q, a.q)), 3e-12,
             "SH_gram_P" + std::to_string(n));
        for (int d = 0; d < 3; ++d) {
            near(maxabs(y.transpose() * weighted[d] - Mat(a.A[d])), 4e-12,
                 "analytic_vs_quadrature_P" + std::to_string(n) + "_d" + std::to_string(d));
            check(a.A[d].nonZeros() <= 6 * a.q, "sparse_linear_nnz", a.A[d].nonZeros());
        }
        Vec u = Vec::LinSpaced(a.q, 0., 1.), old = u;
        Four U(2, .1, -.3, .4);
        set_low(u, U);
        near((project(u) - U).norm(), 1e-15, "projection_right_inverse");
        if (a.q > 4)
            near((u.tail(a.q - 4) - old.tail(a.q - 4)).norm(), 1e-15, "lift_preserves_high");
        const V3 normal = V3(1, 2, -3).normalized();
        Eigen::SelfAdjointEigenSolver<Mat> eig(a.normal(normal));
        int neg = (eig.eigenvalues().array() < -1e-12).count(),
            zero = (eig.eigenvalues().array().abs() <= 1e-12).count();
        check(neg == n * (n + 1) / 2 && zero == n + 1, "normal_signature_P" + std::to_string(n));
        near(eig.eigenvalues().maxCoeff() - a.speed, 3e-13, "rotational_spectral_radius");
    }
    bool refused = false;
    try {
        Budget tiny;
        tiny.bytes = 1024;
        Angular a(100000, tiny);
    } catch (const std::exception &) {
        refused = true;
    }
    check(refused, "resource_limit_before_allocation");
}
void boundary() {
    for (int n : {1, 2, 3, 5, 7, 9, 15}) {
        Angular a(n);
        CanonicalC1 c(a);
        check(c.contraction < 1 && c.condition >= 1, "C1_SPD_contraction_P" + std::to_string(n),
              c.contraction);
        for (V3 normal : std::vector<V3>{V3::UnitZ(), V3(1, 2, -3).normalized()}) {
            Boundary b(a, c, normal);
            Vec u = Vec::Zero(a.q);
            u[0] = 1 / std::sqrt(4 * pi);
            if (n > 1)
                u[index(n - 1, 0)] = .01;
            Inflow g;
            g.density = [&](const V3 &, double, const V3 &w) { return harmonics(n, w).dot(u); };
            auto data = b.data(g, V3::Zero(), 0);
            near((b.flux(u, data.d) - a.apply(normal, u)).norm(), 3e-11, "matching_Nminus1");
            Vec v = Vec::LinSpaced(a.q, -.3, .4), wp = b.Rplus.transpose() * v,
                wm = b.Rminus.transpose() * v, h = c.incoming(wp, data.d);
            Vec z = c.U * c.sigma.asDiagonal() * (wp - h) / std::sqrt(2.),
                linv = c.L.llt().solve(data.d), sq = z + .5 * linv;
            double direct = .5 * v.dot(a.apply(normal, v)) - v.dot(b.flux(v, data.d));
            double identity = -sq.dot(c.L * sq) + .25 * data.d.dot(linv) -
                              .5 * ((wm - h).array().square() * c.sigma.array()).sum();
            near(direct - identity, 5e-11, "inhomogeneous_boundary_energy_identity");
            check(.5 * v.dot(a.apply(normal, v)) - v.dot(b.flux(v, Vec::Zero(c.r))) <= 1e-12,
                  "C1_vacuum_energy");
        }
    }
    Angular a(7);
    CanonicalC1 c(a);
    Boundary b(a, c, V3::UnitZ());
    Mat half = Mat::Zero(c.r, c.s);
    for (const auto &p : sphere_rule(12, 1)) {
        Vec y = harmonics(7, p.omega);
        for (int i = 0; i < c.r; ++i)
            for (int j = 0; j < c.s; ++j)
                half(i, j) += 2 * p.weight * y[c.odd[i]] * y[c.even[j]];
    }
    int oi = std::find(c.odd.begin(), c.odd.end(), index(2, 1)) - c.odd.begin();
    int ej = std::find(c.even.begin(), c.even.end(), index(7, 1)) - c.even.begin();
    double defect = (half - c.L * c.B)(oi, ej);
    near(defect + 9 * std::sqrt(7.) / 128, 1e-12, "P7_exact_highest_degree_defect");
    Vec u = Vec::Zero(a.q);
    u[0] = 1 / std::sqrt(4 * pi);
    u[index(7, 1)] = .01;
    Inflow matched;
    matched.density = [&](const V3 &, double, const V3 &w) { return harmonics(7, w).dot(u); };
    check((b.flux(u, b.data(matched, V3::Zero(), 0).d) - a.apply(V3::UnitZ(), u)).norm() > 1e-5,
          "positive_P7_not_equilibrium");
    Vec qin = Vec::Zero(a.q);
    for (const auto &p : sphere_rule(16, -1))
        qin += p.weight * p.omega[2] * std::pow(1 + p.omega[0], 7) * harmonics(7, p.omega);
    double exact_value = -pi * harmonics(7, V3::UnitX())[index(7, 7)] / 1024;
    near(qin[index(7, 7)] - exact_value, 1e-13, "smooth_nullmode_exact_witness");
    near(a.normal(V3::UnitZ()).col(index(7, 7)).norm(), 1e-15, "P7_nullmode");
    check(std::abs(exact_value) > 1e-3, "physical_inflow_outside_characteristic_range",
          exact_value);
    Vec tangent = harmonics(7, V3::UnitX());
    check(b.flux(tangent, Vec::Zero(c.r)).norm() > .2, "tangential_trace_model_defect");
    Inflow atom;
    atom.atoms = [](const V3 &, double) { return std::vector<Atom>{{-V3::UnitZ(), 1.}}; };
    Four injection = -project(b.flux(Vec::Zero(a.q), b.data(atom, V3::Zero(), 0).d));
    check(margin(injection) < -.04, "C1_injection_is_not_positive_M1_anchor", margin(injection));
    atom.atoms = [](const V3 &, double) {
        return std::vector<Atom>{{V3(1, 0, -1e-8).normalized(), 1.}};
    };
    check(b.data(atom, V3::Zero(), 0).d.norm() < 1e-6, "incoming_grazing_limit");
    Boundary c0(a, c, V3::UnitZ(), true);
    check((c0.flux(tangent, Vec::Zero(c.r)) - b.flux(tangent, Vec::Zero(c.r))).norm() > 1e-3,
          "C0_C1_distinct");
    for (double f : {0., .2, .7, .95}) {
        Four U(1.7, 1.7 * f, 0, 0);
        Levermore m(U);
        V3 n = V3(.2, .3, .8).normalized();
        near((m.outgoing(n) - m.outgoing(-n) - m.flux(n)).norm(), 2e-15, "M1_analytic_half_sum");
        // Independent direct quadrature, rotated so the hemisphere is smooth.
        V3 t = n.cross(V3::UnitX()).normalized(), v = n.cross(t);
        Four integral = Four::Zero();
        for (const auto &p : sphere_rule(100, 1)) {
            V3 w = p.omega[0] * t + p.omega[1] * v + p.omega[2] * n;
            double psi = m.K * std::pow(m.gap, 3) / (4 * pi * std::pow(1 - m.beta.dot(w), 4));
            integral[0] += p.weight * p.omega[2] * psi;
            integral.tail<3>() += p.weight * p.omega[2] * psi * w;
        }
        near((integral - m.outgoing(n)).norm(), 1e-9, "Levermore_halfspace_vs_quadrature");
        check(exact::cone(Four(U + m.flux(n))) && exact::cone(Four(U - m.flux(n))),
              "M1_flux_splitting");
    }
    near(Levermore(Four::Zero()).outgoing(V3::UnitX()).norm(), 1e-15, "vacuum_halfspace");
    near((Levermore(Four(1, 1, 0, 0)).outgoing(V3::UnitX()) - Four(1, 1, 0, 0)).norm(), 1e-15,
         "pure_beam_halfspace");
}
void spatial() {
    Angular a(3);
    for (int n : {1, 2, 4, 8}) {
        MeshSpec spec;
        spec.elements = n;
        Spatial fe(spec);
        near(fe.ibp_error, 2e-12, "3D_MFEM_integration_by_parts");
        near(fe.constant_error, 2e-12, "3D_MFEM_constant_derivatives");
        Vec x = fe.coordinates.col(0), kx = fe.K * x;
        double norm = (kx.array().square() / fe.mass.array()).sum();
        near(norm - 4 * n, 3e-10, "Q1_affine_4_over_h_n" + std::to_string(n));
        Vec beta = Vec::Constant(fe.nodes, .01 * std::pow(std::sqrt(3.) / n, 3) * a.speed);
        for (int i = 0; i < fe.nodes; ++i)
            if (fe.boundary[i])
                beta[i] = 0;
        near((fe.K * (beta.array() / fe.mass.array() * kx.array()).matrix()).norm(), 2e-12,
             "masked_affine_annihilation");
        if (n <= 2) {
            Numerics opt;
            Transport tr(a, fe, opt);
            Field u = stress(fe, a.q), w, z = tr.biharmonic(u, &w);
            Mat dense =
                -Mat(fe.K) * (tr.beta.array() / fe.mass.array()).matrix().asDiagonal() * Mat(fe.K);
            near(maxabs(z - dense * u), 2e-12, "biharmonic_sparse_vs_dense");
            near(z.colwise().sum().norm(), 3e-12, "biharmonic_conservation");
            Field ku = fe.K * u;
            double diss = 0;
            for (int i = 0; i < fe.nodes; ++i)
                diss += tr.beta[i] / fe.mass[i] * ku.row(i).squaredNorm();
            near((u.array() * z.array()).sum() + diss, 2e-12, "biharmonic_energy_identity");
            double residual;
            Field solved = fe.solve_mass(fe.M * u, 1e-12, residual);
            near(maxabs(solved - u), 2e-12, "consistent_mass_recovery");
        }
    }
    for (int dim : {1, 2, 3}) {
        MeshSpec s;
        s.dimension = dim;
        s.tetrahedra = dim > 1;
        Spatial f(s);
        near(f.ibp_error, 3e-12, "P1_dim" + std::to_string(dim));
    }
    MeshSpec tet;
    tet.single_tet = true;
    Spatial t(tet);
    check(t.nodes == 4 && t.pairs.size() == 6, "single_tetrahedron_full_graph");
    near(t.mass.sum() - 1. / 6, 1e-14, "tetrahedron_volume");
    bool oblique = false;
    for (const auto &f : t.faces)
        for (const auto &p : f.points)
            if (p.normal.cwiseAbs().minCoeff() > .5)
                oblique = true;
    check(oblique, "true_oblique_tetrahedron_face");
    MeshSpec periodic;
    periodic.elements = 4;
    periodic.periodic = true;
    Spatial p(periodic);
    check(p.faces.empty() && p.nodes == 64, "3D_periodic_topology");
    near(p.ibp_error, 3e-12, "periodic_skew_convection");
}
void coupling() {
    // First exercise audit priorities 1--3 in spatial 2D, then 3D.
    for (int dim : {2, 3})
        for (int n : {3, 5, 7, 9})
            for (bool mask : {false, true}) {
                MeshSpec spec;
                spec.dimension = dim;
                Spatial f(spec);
                Angular a(n);
                Numerics opt;
                opt.masked = mask;
                Inflow g;
                g.density = [](const V3 &x, double t, const V3 &w) {
                    return (1 + .2 * x[0] + .1 * t) * (1 + .2 * w[1]) / (4 * pi);
                };
                Transport tr(a, f, opt, g);
                Field u = stress(f, a.q);
                Stage un = tr.evaluate(.2, u, true), limited = tr.evaluate(.2, u);
                near(maxabs(un.rate - un.target), 3e-11, "unlimited_complete_target_recovery");
                near(limited.diagnostics.target_recovery, 3e-12, "open_surface_mass_pair_identity");
                near(limited.diagnostics.local_balance, 5e-12, "accepted_Q_balance");
                near(limited.diagnostics.high_rate_error, 1e-15, "high_rates_unchanged");
                check(limited.diagnostics.active_pairs > 0, "active_MCL",
                      limited.diagnostics.active_pairs);
                Field mu = f.M * u;
                near((mu.array() * limited.rate.array()).sum() - limited.diagnostics.target_energy -
                         limited.diagnostics.sync_work,
                     5e-12, "signed_sync_work_identity");
                TimeControl control(tr, spec, 3);
                double dt = .2 * std::min(f.cfl_step, control.step_limit);
                Step step = advance(tr, control, .2, dt, u);
                near(step.ledger_residual, 5e-12, "SSPRK3_integrated_transfer_balance");
                near(step.rate_residual, 5e-14, "SSPRK3_rate_weights");
                check(step.minimum_margin >= 0, "all_SSP_states_admissible");
            }
    {
        MeshSpec vector_spec;
        vector_spec.dimension = 2;
        vector_spec.elements = 2;
        Spatial vector_fe(vector_spec);
        Angular vector_pn(3);
        Numerics vector_options;
        vector_options.density_vector = true;
        Transport vector_transport(vector_pn, vector_fe, vector_options);
        Field vector_u = stress(vector_fe, vector_pn.q);
        Stage vector_stage = vector_transport.evaluate(.1, vector_u);
        near(vector_stage.diagnostics.local_balance, 5e-12, "vector_MCL_accepted_Q_balance");
        near(vector_stage.diagnostics.high_rate_error, 1e-15, "vector_MCL_high_rates_unchanged");
        TimeControl vector_control(vector_transport, vector_spec, 3);
        Step vector_step = advance(vector_transport, vector_control, .1,
                                   .1 * vector_control.step_limit, vector_u);
        check(vector_step.minimum_margin >= 0, "vector_MCL_all_SSP_states_admissible");
        near(vector_step.ledger_residual, 5e-12, "vector_MCL_SSPRK3_integrated_balance");
    }
    MeshSpec spec;
    spec.single_tet = true;
    Spatial f(spec, 2);
    Angular a(7);
    Numerics opt;
    opt.anchor = 2;
    Vec equilibrium = Vec::Zero(a.q);
    equilibrium[0] = 1 / std::sqrt(4 * pi);
    equilibrium[index(6, 0)] = .01;
    Inflow match;
    match.density = [&](const V3 &, double, const V3 &w) {
        return harmonics(7, w).dot(equilibrium);
    };
    Transport tr(a, f, opt, match);
    Field u(f.nodes, a.q);
    for (int i = 0; i < f.nodes; ++i)
        u.row(i) = equilibrium.transpose();
    Stage s = tr.evaluate(0, u);
    near(maxabs(s.rate), 3e-12, "strengthened_positive_P6_equilibrium");
    check(s.diagnostics.minimum_alpha == 1 && s.diagnostics.minimum_eta == 1,
          "equilibrium_all_factors_one");
    TimeControl rk2(tr, spec, 2);
    Step step = advance(tr, rk2, 0, .1 * rk2.step_limit, u);
    near(step.ledger_residual, 3e-12, "open_SSPRK2_ledger");
    Field invalid = u;
    invalid(0, 0) = -1;
    bool rejected = false;
    try {
        advance(tr, rk2, 0, .1 * rk2.step_limit, invalid);
    } catch (const StepRejected &) {
        rejected = true;
    }
    check(rejected && invalid(0, 0) == -1, "invalid_state_rejected_without_floor");
    Field src = Field::Zero(f.nodes, a.q);
    src.col(0).setConstant(.1);
    Vec absorb = Vec::LinSpaced(f.nodes, 0, 4), scatter = Vec::LinSpaced(f.nodes, 1, 3);
    Field reacted = reaction(u, src, absorb, scatter, .4, .2);
    for (int i = 0; i < f.nodes; ++i)
        check(exact::cone(project(reacted.row(i).transpose())), "variable_nodal_reaction_cone");
    double expected =
        std::exp(-(absorb[1] + scatter[1] * (1 - std::pow(.4, 6))) * .2) * u(1, index(6, 0));
    near(reacted(1, index(6, 0)) - expected, 1e-15, "zonal_degree_mapping");
}
void time_tests() {
    Angular a(7);
    const double low = 9 / (2 * (12 + 6 * std::sqrt(2.) + std::sqrt(3.)));
    const double argument = 12 * std::sqrt(3.) / 5 * a.speed;
    const double amplification =
        std::abs(1 - argument + argument * argument / 2 - argument * argument * argument / 6);
    check(.125 < low && amplification > 5.6, "Proposition_8_3_M1_pass_PN_fail", amplification);
    near(amplification - 5.626016067, 1e-8, "Proposition_8_3_numerical_value");
    near(fourier_amplification(a, 4, 3, .1, .125 / 4, 3) - amplification, 1e-11,
         "independent_periodic_symbol_witness");
    MeshSpec spec;
    spec.periodic = true;
    spec.elements = 4;
    Spatial f(spec);
    Numerics opt;
    opt.gamma = .1;
    Transport tr(a, f, opt);
    TimeControl control(tr, spec, 3);
    bool rejected = false;
    try {
        control.check(.125 / 4);
    } catch (const std::exception &) {
        rejected = true;
    }
    check(rejected, "PN_time_guard_rejects_witness");
    check(fourier_amplification(a, 4, 3, .1, control.step_limit, 3) <= 1 + 2e-13,
          "selected_step_passes_modal_control");
    // Independently apply the actual assembled operator to its two-axis Nyquist mode.
    Field mode = Field::Zero(f.nodes, a.q);
    for (int i = 0; i < f.nodes; ++i) {
        int parity =
            int(std::lround(4 * f.coordinates(i, 1))) + int(std::lround(4 * f.coordinates(i, 2)));
        mode(i, index(7, 0)) = parity % 2 ? -1 : 1;
    }
    Field v = tr.linear_target(mode);
    near(maxabs(v + argument / (.125 / 4) * mode), 3e-11, "assembled_3D_Nyquist_matches_symbol");
    // SSPRK3 sufficient contraction bound on a nonnormal dissipative test matrix.
    Mat L(3, 3);
    L << -.1, 3, 0, -3, -.2, 4, 0, -4, -.3;
    double norm = L.norm();
    double dt = .95 / norm;
    Mat r = Mat::Identity(3, 3) + dt * L + .5 * dt * dt * L * L + dt * dt * dt / 6 * L * L * L;
    Eigen::SelfAdjointEigenSolver<Mat> eig(r.transpose() * r);
    check(eig.eigenvalues().maxCoeff() <= 1, "RK3_nonnormal_sufficient_bound");
    Numerics zero;
    zero.gamma = 0;
    Transport centered(a, f, zero);
    rejected = false;
    try {
        TimeControl no_step(centered, spec, 2);
    } catch (const std::exception &) {
        rejected = true;
    }
    check(rejected, "SSPRK2_imaginary_axis_rejected");
}
} // namespace
int main(int argc, char **argv) {
#ifdef M1PN_MPI
    MPI_Init(&argc, &argv);
#endif
    int result = 0;
    try {
        exact::environment();
        require(argc == 2, "Choose a test suite");
        std::string name = argv[1];
        if (name == "angular")
            angular();
        else if (name == "boundary")
            boundary();
        else if (name == "spatial")
            spatial();
        else if (name == "coupling")
            coupling();
        else if (name == "time")
            time_tests();
        else if (name == "cone_filter")
            cone_filter();
        else if (name == "pair_limiter")
            pair_limiter();
        else if (name == "base_bar")
            robust_base_bar();
        else
            throw std::runtime_error("Unknown test suite");
        std::cout << "{\"checks_passed\":" << checks << "}\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        result = 1;
    }
#ifdef M1PN_MPI
    MPI_Finalize();
#endif
    return result;
}
