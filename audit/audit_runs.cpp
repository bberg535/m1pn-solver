// Time-stepping audits: cone margins per stage, energy budget, mass balance, T1..T8 and crossing comparison.
#include "audit_common.hpp"
#include <chrono>
#include <fstream>

static double dotf(const Field &x, const Field &y) { return (x.array() * y.array()).sum(); }

struct Mesh {
    const ElementWeno &s;
    std::vector<char> elem_bnd, node_bnd;
    explicit Mesh(const ElementWeno &sc) : s(sc), elem_bnd(sc.cells, 0), node_bnd(sc.nodes, 0) {
        for (int e = 0; e < s.cells; ++e) {
            int x = e;
            for (int d = 0; d < s.dimension; ++d, x /= s.elements)
                if (x % s.elements == 0 || x % s.elements == s.elements - 1) elem_bnd[e] = 1;
        }
        if (!s.options.periodic)
            for (int i = 0; i < s.nodes; ++i) {
                int x = i;
                for (int d = 0; d < s.dimension; ++d, x /= s.per_direction)
                    if (x % s.per_direction == 0 || x % s.per_direction == s.per_direction - 1) node_bnd[i] = 1;
            }
    }
};

static Field apply_mass(const ElementWeno &s, const Field &x) {
    const int q = x.cols();
    Field local(s.cells, s.local_nodes * q);
    for (int e = 0; e < s.cells; ++e) {
        Mat xe(s.local_nodes, q);
        for (int l = 0; l < s.local_nodes; ++l) xe.row(l) = x.row(s.connectivity_[e * s.local_nodes + l]);
        const Mat r = s.mass_ * xe;
        for (int l = 0; l < s.local_nodes; ++l)
            for (int c = 0; c < q; ++c) local(e, l * q + c) = r(l, c);
    }
    return s.gather_load(local);
}

struct Energy {
    double N2 = 0, vol = 0, stab = 0, stab_baseline = 0, bnd = 0, sync = 0, sync_elem_only = 0, sync_bnd_only = 0,
           total = 0, closure_err = 0, delta_rel = 0, delta_outside_moments = 0;
    double mass_accepted = 0, mass_target = 0, mass_expected = 0, mass_resid_rel = 0, boundary_dev_L1 = 0,
           max_bnd_rho_rate = -1e300, min_bnd_rho_rate = 1e300, target_bnd_mass = 0, accepted_bnd_mass = 0;
    double kappa_pos_defect = 0;
};

static Energy measure(const ElementWeno &s, double t, const Field &a) {
    Energy en;
    const int q = s.pn.q, n = s.nodes;
    Field low(n, 4), low_flux(n, 12), high_flux(n, 12);
    low_flux.setZero(); high_flux.setZero();
    std::array<Field, 3> af;
    for (int d = 0; d < s.dimension; ++d) af[d] = a * s.pn.A[d];
    for (int i = 0; i < n; ++i) {
        const Four u = project(a.row(i).transpose());
        low.row(i) = u.transpose();
        const Levermore lv(u);
        for (int d = 0; d < s.dimension; ++d) {
            V3 nn = V3::Zero(); nn[d] = 1;
            const Four fl = lv.flux(nn), fh = project(af[d].row(i).transpose());
            for (int c = 0; c < 4; ++c) { low_flux(i, 4 * d + c) = fl[c]; high_flux(i, 4 * d + c) = fh[c]; }
        }
    }
    const Field closure = high_flux - low_flux;
    Field loc(s.cells, s.local_nodes * q);
    for (int e = 0; e < s.cells; ++e) {
        Mat r = Mat::Zero(s.local_nodes, q);
        for (int d = 0; d < s.dimension; ++d) {
            Mat fe(s.local_nodes, q);
            for (int l = 0; l < s.local_nodes; ++l) fe.row(l) = af[d].row(s.connectivity_[e * s.local_nodes + l]);
            r.noalias() -= s.convection_[d] * fe;
        }
        for (int l = 0; l < s.local_nodes; ++l)
            for (int c = 0; c < q; ++c) loc(e, l * q + c) = r(l, c);
    }
    const Field Rvol = s.gather_load(loc);
    const Vec gamma = s.options.stabilize ? s.weno_gamma(a) : Vec(Vec::Ones(s.cells));
    Field stab = Field::Zero(n, q), stab1 = Field::Zero(n, q);
    if (s.options.stabilize) {
        stab = s.gather_load(s.stabilization_load(a, gamma));
        stab1 = s.gather_load(s.stabilization_load(a, Vec::Ones(s.cells)));
    }
    Field bh = Field::Zero(n, q), bl = Field::Zero(n, 4), fa = Field::Zero(n, 3), bbar = Field::Zero(n, 4);
    Vec area = Vec::Zero(n);
    if (!s.faces_.empty()) s.add_boundary(t, a, low, low_flux, bh, bl, fa, area, bbar);
    const Field high_rate = s.solve_mass(Field(Rvol - stab + bh));
    en.N2 = dotf(a, apply_mass(s, a));
    en.vol = dotf(a, Rvol);
    en.stab = -dotf(a, stab);
    en.stab_baseline = -dotf(a, stab1);
    en.bnd = dotf(a, bh);
    ElementWenoDebug dbg;
    ElementWenoStats st;
    const Field out = s.rate(t, a, &st, &dbg);
    const std::vector<double> theta = audit_theta;
    const Field delta = out - high_rate;
    Field d_out = delta;
    for (int c : {0, 1, 2, 3}) d_out.col(c).setZero();
    en.delta_outside_moments = d_out.cwiseAbs().maxCoeff();
    en.sync = dotf(a, apply_mass(s, delta));
    en.total = dotf(a, apply_mass(s, out));
    en.closure_err = std::abs(en.vol + en.stab + en.bnd + en.sync - en.total);
    en.delta_rel = std::sqrt(dotf(delta, apply_mass(s, delta))) / std::max(1e-300, std::sqrt(dotf(high_rate, apply_mass(s, high_rate))));
    {
        audit_skip_boundary_limit = true;
        const Field o2 = s.rate(t, a);
        audit_skip_boundary_limit = false;
        en.sync_elem_only = dotf(a, apply_mass(s, Field(o2 - high_rate)));
        audit_skip_element_limit = true;
        const Field o3 = s.rate(t, a);
        audit_skip_element_limit = false;
        en.sync_bnd_only = dotf(a, apply_mass(s, Field(o3 - high_rate)));
    }
    s.rate(t, a); // restore audit_theta of the unmodified call for completeness
    // mass balance
    const double c4 = std::sqrt(4 * pi);
    double scale = 0;
    for (int i = 0; i < n; ++i) {
        en.mass_accepted += s.lumped[i] * out(i, 0) * c4;
        en.mass_target += s.lumped[i] * c4 * (high_rate(i, 0));
    }
    for (int i = 0; i < n; ++i) {
        Four corr = project(bh.row(i).transpose()) - Four(bl(i, 0), bl(i, 1), bl(i, 2), bl(i, 3));
        double fl_term = 0;
        for (int d = 0; d < s.dimension; ++d) {
            corr -= fa(i, d) * Four(closure(i, 4 * d), closure(i, 4 * d + 1), closure(i, 4 * d + 2), closure(i, 4 * d + 3));
            fl_term += fa(i, d) * low_flux(i, 4 * d);
        }
        if (area[i] > 0) {
            const double B = bl(i, 0) + theta[i] * corr[0] - fl_term;   // accepted boundary mass rate contribution at node i
            en.mass_expected += B;
            en.max_bnd_rho_rate = std::max(en.max_bnd_rho_rate, B);
            en.min_bnd_rho_rate = std::min(en.min_bnd_rho_rate, B);
            en.boundary_dev_L1 += std::abs((1 - theta[i]) * corr[0]);
            en.target_bnd_mass += project(bh.row(i).transpose())[0];
            en.accepted_bnd_mass += B;
            scale += std::abs(bl(i, 0)) + std::abs(fl_term) + std::abs(corr[0]);
        }
    }
    // interior cancellation: periodic meshes have no boundary term at all
    en.mass_resid_rel = std::abs(en.mass_accepted - en.mass_expected) / std::max(1e-300, scale + std::abs(en.mass_accepted));
    if (s.faces_.empty()) en.mass_resid_rel = std::abs(en.mass_accepted) / std::max(1e-300, c4 * a.col(0).cwiseAbs().maxCoeff() + 1e-300);
    return en;
}

struct Agg {
    double min_margin_abs = 1e300, min_margin_rel = 1e300;
    long viol_euler = 0, viol_stage = 0, nudged = 0, max_ulps = 0, scans = 0;
    double max_raw_zero_rel = 0, max_lim_zero_rel = 0, max_raw_zero_abs = 0, max_lim_zero_abs = 0;
    double min_factor = 1, min_theta = 1;
    long el_lim_b = 0, el_tot_b = 0, el_lim_i = 0, el_tot_i = 0, node_lim_b = 0, node_tot_b = 0;
    double gamma_min_b = 1, gamma_min_i = 1, gamma_sum_b = 0, gamma_sum_i = 0;
    long gamma_cnt_b = 0, gamma_cnt_i = 0, gamma_lt99_b = 0, gamma_lt99_i = 0;
    std::vector<Energy> samples;
    double cum_vol = 0, cum_stab = 0, cum_bnd = 0, cum_sync = 0, cum_sync_el = 0, cum_sync_bn = 0, cum_total = 0;
    double max_vol = -1e300, max_stab = -1e300, max_bnd = -1e300, max_sync = -1e300, max_sync_el = -1e300, max_sync_bn = -1e300,
           max_total = -1e300, max_stab_nonlin = -1e300, max_closure_err = 0, max_mass_resid = 0, max_delta_rel = 0;
    double max_net_face = -1e300, mass_rate_max = -1e300, mass_rate_min = 1e300, mass_rate_target_max = -1e300, mass_rate_target_min = 1e300;
    double min_bnd_rho_rate = 1e300, max_bnd_rho_rate = -1e300, max_boundary_dev_L1 = 0, max_mass_target_minus_accepted = 0;
    long steps = 0;
    double dt = 0, wall = 0;
};

static void scan(const Field &x, double &mabs, double &mrel, long &viol) {
    for (int i = 0; i < x.rows(); ++i) {
        const Four u = project(x.row(i).transpose());
        if (!exact::cone(u)) ++viol;
        const double m = margin(u);
        mabs = std::min(mabs, m);
        if (u[0] > 0) mrel = std::min(mrel, m / u[0]);
    }
}

struct Driver {
    const ElementWeno &s;
    Mesh mesh;
    Agg g;
    Driver(const ElementWeno &sc) : s(sc), mesh(sc) {}
    Field R(double t, const Field &x) {
        ElementWenoStats st;
        Field r = s.rate(t, x, &st);
        g.max_raw_zero_abs = std::max(g.max_raw_zero_abs, st.max_raw_zero_sum);
        g.max_lim_zero_abs = std::max(g.max_lim_zero_abs, st.max_limited_zero_sum);
        const double sc = std::max(1e-300, *std::max_element(audit_scale.begin(), audit_scale.end()));
        g.max_raw_zero_rel = std::max(g.max_raw_zero_rel, st.max_raw_zero_sum / sc);
        g.max_lim_zero_rel = std::max(g.max_lim_zero_rel, st.max_limited_zero_sum / sc);
        g.nudged += st.nudged_states;
        for (int e = 0; e < s.cells; ++e) {
            const bool lim = audit_factor[e] < 1;
            g.min_factor = std::min(g.min_factor, audit_factor[e]);
            (mesh.elem_bnd[e] ? g.el_tot_b : g.el_tot_i) += 1;
            (mesh.elem_bnd[e] ? g.el_lim_b : g.el_lim_i) += lim;
        }
        for (int i = 0; i < s.nodes; ++i)
            if (mesh.node_bnd[i]) { ++g.node_tot_b; g.node_lim_b += audit_theta[i] < 1; g.min_theta = std::min(g.min_theta, audit_theta[i]); }
        const Vec gm = s.options.stabilize ? s.weno_gamma(x) : Vec(Vec::Ones(s.cells));
        for (int e = 0; e < s.cells; ++e) {
            if (mesh.elem_bnd[e]) { g.gamma_min_b = std::min(g.gamma_min_b, gm[e]); g.gamma_sum_b += gm[e]; ++g.gamma_cnt_b; g.gamma_lt99_b += gm[e] < .99; }
            else { g.gamma_min_i = std::min(g.gamma_min_i, gm[e]); g.gamma_sum_i += gm[e]; ++g.gamma_cnt_i; g.gamma_lt99_i += gm[e] < .99; }
        }
        return r;
    }
    Field adm(Field x) {
        for (int i = 0; i < s.nodes; ++i) {
            Four u = project(x.row(i).transpose());
            if (nudge(u, &g.max_ulps)) {
                Vec row = x.row(i).transpose();
                set_low(row, u);
                x.row(i) = row.transpose();
                ++g.nudged;
            }
        }
        return x;
    }
    Field step(double t, double dt, const Field &a, bool sample) {
        ++g.steps;
        if (sample) {
            Energy en = measure(s, t, a);
            const double w = dt / std::max(1e-300, en.N2);
            g.cum_vol += dt * en.vol; g.cum_stab += dt * en.stab; g.cum_bnd += dt * en.bnd; g.cum_sync += dt * en.sync;
            g.cum_sync_el += dt * en.sync_elem_only; g.cum_sync_bn += dt * en.sync_bnd_only; g.cum_total += dt * en.total;
            (void)w;
            const double nrm = std::max(1e-300, en.N2);
            g.max_vol = std::max(g.max_vol, en.vol / nrm); g.max_stab = std::max(g.max_stab, en.stab / nrm);
            g.max_stab_nonlin = std::max(g.max_stab_nonlin, (en.stab - en.stab_baseline) / nrm);
            g.max_bnd = std::max(g.max_bnd, en.bnd / nrm); g.max_sync = std::max(g.max_sync, en.sync / nrm);
            g.max_sync_el = std::max(g.max_sync_el, en.sync_elem_only / nrm); g.max_sync_bn = std::max(g.max_sync_bn, en.sync_bnd_only / nrm);
            g.max_total = std::max(g.max_total, en.total / nrm);
            g.max_net_face = std::max(g.max_net_face, (en.vol + en.bnd) / nrm);
            g.max_closure_err = std::max(g.max_closure_err, en.closure_err / nrm);
            g.max_mass_resid = std::max(g.max_mass_resid, en.mass_resid_rel);
            g.max_delta_rel = std::max(g.max_delta_rel, en.delta_rel);
            if (!s.faces_.empty()) {
                g.min_bnd_rho_rate = std::min(g.min_bnd_rho_rate, en.min_bnd_rho_rate);
                g.max_bnd_rho_rate = std::max(g.max_bnd_rho_rate, en.max_bnd_rho_rate);
                g.max_boundary_dev_L1 = std::max(g.max_boundary_dev_L1, en.boundary_dev_L1);
            }
            g.mass_rate_max = std::max(g.mass_rate_max, en.mass_accepted); g.mass_rate_min = std::min(g.mass_rate_min, en.mass_accepted);
            g.mass_rate_target_max = std::max(g.mass_rate_target_max, en.mass_target); g.mass_rate_target_min = std::min(g.mass_rate_target_min, en.mass_target);
            g.max_mass_target_minus_accepted = std::max(g.max_mass_target_minus_accepted, std::abs(en.mass_accepted - en.mass_target));
        }
        const Field y1 = a + dt * R(t, a);
        ++g.scans; scan(y1, g.min_margin_abs, g.min_margin_rel, g.viol_euler);
        const Field a1 = adm(y1);
        const Field y2 = a1 + dt * R(t + dt, a1);
        scan(y2, g.min_margin_abs, g.min_margin_rel, g.viol_euler);
        const Field c2 = .75 * a + .25 * y2;
        scan(c2, g.min_margin_abs, g.min_margin_rel, g.viol_stage);
        const Field a2 = adm(c2);
        const Field y3 = a2 + dt * R(t + .5 * dt, a2);
        scan(y3, g.min_margin_abs, g.min_margin_rel, g.viol_euler);
        const Field c3 = (1. / 3) * a + (2. / 3) * y3;
        scan(c3, g.min_margin_abs, g.min_margin_rel, g.viol_stage);
        return adm(c3);
    }
    Field run(Field a, double T, double cfl, int samples, double *dt_out = nullptr, double fixed_dt = 0) {
        const double lim = cfl * s.step_bound();
        const double req = fixed_dt > 0 ? fixed_dt : lim;
        if (fixed_dt > 0 && fixed_dt > s.step_bound() * (1 + 1e-12)) throw std::runtime_error("requested dt violates implemented CFL");
        const int steps = std::max(1, int(std::ceil(T / req - 1e-12)));
        const double dt = T / steps;
        g.dt = dt;
        const int stride = std::max(1, steps / std::max(1, samples));
        double t = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < steps; ++k) { a = step(t, dt, a, samples > 0 && (k % stride == 0)); t += dt; }
        g.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (dt_out) *dt_out = dt;
        return a;
    }
};

static void emit_run(const std::string &test, const std::string &label, const ElementWeno &s, const Agg &g, const Field &a_final,
                     double T, const std::map<std::string, double> &extra = {}) {
    double rho_min = 1e300, rho_max = -1e300, mass = 0;
    for (int i = 0; i < s.nodes; ++i) {
        const Four u = project(a_final.row(i).transpose());
        rho_min = std::min(rho_min, u[0]); rho_max = std::max(rho_max, u[0]);
        mass += s.lumped[i] * u[0];
    }
    J j(test);
    j.str("label", label).add("pn_order", s.pn.order).add("degree", s.degree).add("dimension", s.dimension)
        .add("elements", s.elements).add("periodic", s.options.periodic).add("T", T).add("steps", g.steps).add("dt", g.dt)
        .add("dt_bound_realizability", s.dt_bound).add("dt_high_order", s.high_order_step())
        .add("cfl_fraction_of_step_bound", g.dt / s.step_bound())
        .add("weno", s.options.weno).add("limit", s.options.limit).add("stabilize", s.options.stabilize)
        .add("audit_low_order", audit_low_order).add("weno_linear", s.options.weno_linear)
        .add("min_cone_margin_abs_all_stages", g.min_margin_abs).add("min_cone_margin_rel_all_stages", g.min_margin_rel)
        .add("cone_violations_euler_partial", double(g.viol_euler)).add("cone_violations_ssp_stage", double(g.viol_stage))
        .add("silent_nudges_ulp_repairs", double(g.nudged)).add("max_ulps_per_nudge", double(g.max_ulps))
        .add("max_raw_zero_sum_abs", g.max_raw_zero_abs).add("max_limited_zero_sum_abs", g.max_lim_zero_abs)
        .add("max_raw_zero_sum_rel", g.max_raw_zero_rel).add("max_limited_zero_sum_rel", g.max_lim_zero_rel)
        .add("min_element_factor", g.min_factor).add("min_boundary_theta", g.min_theta)
        .add("limited_frac_boundary_elements", g.el_tot_b ? double(g.el_lim_b) / g.el_tot_b : 0.)
        .add("limited_frac_interior_elements", g.el_tot_i ? double(g.el_lim_i) / g.el_tot_i : 0.)
        .add("limited_frac_boundary_nodes_theta", g.node_tot_b ? double(g.node_lim_b) / g.node_tot_b : 0.)
        .add("gamma_min_boundary_elems", g.gamma_min_b).add("gamma_min_interior_elems", g.gamma_min_i)
        .add("gamma_mean_boundary_elems", g.gamma_cnt_b ? g.gamma_sum_b / g.gamma_cnt_b : 1.)
        .add("gamma_mean_interior_elems", g.gamma_cnt_i ? g.gamma_sum_i / g.gamma_cnt_i : 1.)
        .add("frac_gamma_lt_0p99_boundary", g.gamma_cnt_b ? double(g.gamma_lt99_b) / g.gamma_cnt_b : 0.)
        .add("frac_gamma_lt_0p99_interior", g.gamma_cnt_i ? double(g.gamma_lt99_i) / g.gamma_cnt_i : 0.)
        .add("energy_rate_max_rel_volume", g.max_vol).add("energy_rate_max_rel_stabilization", g.max_stab)
        .add("energy_rate_max_rel_stabilization_nonlinear_part", g.max_stab_nonlin)
        .add("energy_rate_max_rel_boundary_term", g.max_bnd).add("energy_rate_max_rel_net_face_vol_plus_bnd", g.max_net_face)
        .add("energy_rate_max_rel_sync_full", g.max_sync).add("energy_rate_max_rel_sync_element_limiter_only", g.max_sync_el)
        .add("energy_rate_max_rel_sync_boundary_limiter_only", g.max_sync_bn).add("energy_rate_max_rel_total", g.max_total)
        .add("energy_integral_volume", g.cum_vol).add("energy_integral_stabilization", g.cum_stab)
        .add("energy_integral_boundary", g.cum_bnd).add("energy_integral_sync", g.cum_sync)
        .add("energy_integral_sync_element_only", g.cum_sync_el).add("energy_integral_sync_boundary_only", g.cum_sync_bn)
        .add("energy_integral_total", g.cum_total)
        .add("energy_budget_closure_err_rel", g.max_closure_err)
        .add("sync_defect_rel_norm_max", g.max_delta_rel)
        .add("mass_balance_resid_rel_max", g.max_mass_resid)
        .add("global_mass_rate_accepted_max", g.mass_rate_max).add("global_mass_rate_accepted_min", g.mass_rate_min)
        .add("global_mass_rate_pn_target_max", g.mass_rate_target_max).add("global_mass_rate_pn_target_min", g.mass_rate_target_min)
        .add("accepted_boundary_rho_rate_min", g.min_bnd_rho_rate).add("accepted_boundary_rho_rate_max", g.max_bnd_rho_rate)
        .add("pn_target_vs_accepted_boundary_L1_dev_max", g.max_boundary_dev_L1)
        .add("pn_target_vs_accepted_total_mass_rate_abs_dev_max", g.max_mass_target_minus_accepted)
        .add("rho_min_final", rho_min).add("rho_max_final", rho_max).add("mass_final", mass).add("wall_s", g.wall);
    for (const auto &[k, v] : extra) j.add(k, v);
    j.done();
}

static Vec iso(const Angular &pn, double rho) { Vec v = Vec::Zero(pn.q); v[0] = rho / std::sqrt(4 * pi); return v; }

// ------- test cases -------
static void T1_isotropic() {
    Angular pn(3);
    for (bool low : {false, true}) {
        ElementWenoOptions o; o.degree = 2; o.dimension = 2; o.elements = 4; o.periodic = false;
        Inflow in; in.density = [](const V3 &, double, const V3 &) { return 1. / (4 * pi); }; in.constant_density = 1. / (4 * pi);
        ElementWeno s(pn, o, in);
        audit_low_order = low;
        Field a = s.interpolate([&](const V3 &) { return iso(pn, 1.); });
        ElementWenoStats st;
        const Field r = s.rate(0, a, &st);
        const double rmax = r.cwiseAbs().maxCoeff();
        Vec trace = a.row(0).transpose();
        Boundary &b = *s.boundaries_[0];
        Incoming d = b.data(in, V3::Zero(), 0);
        const double ext_err = (b.exterior(trace, d.d) - trace).norm();
        J("T1_isotropic_stationary").add("audit_low_order", low).add("max_abs_rate_all_coeffs", rmax)
            .add("max_abs_rate_density", r.col(0).cwiseAbs().maxCoeff()).add("exterior_minus_trace_norm", ext_err)
            .add("incoming_halfrange_moment_rho", d.moments[0]).add("full_density", 1.0)
            .add("min_boundary_theta", st.min_boundary_limiter).done();
        if (!low) {
            Driver dr(s); double dtu = 0;
            Field f = dr.run(a, .05, .9, 5, &dtu);
            emit_run("T1_isotropic_run", "full", s, dr.g, f, .05, {{"max_abs_deviation_from_initial", (f - a).cwiseAbs().maxCoeff()}});
        }
        audit_low_order = false;
    }
}

static Vec psi_coeffs(const Angular &pn, const std::function<double(const V3 &)> &psi) {
    Vec a = Vec::Zero(pn.q);
    for (const auto &p : sphere_rule(pn.order + 8)) a += p.weight * psi(p.omega) * harmonics(pn.order, p.omega);
    return a;
}

static void T2_vacuum() {
    Angular pn(3);
    struct Case { std::string name; int node; V3 J; };
    for (auto cs : {Case{"corner_outflow_J", 0, V3(-.5, -.5, 0) / std::sqrt(2.) * 1.5}, Case{"corner_inflow_J", 0, V3(.5, .5, 0) / std::sqrt(2.) * 1.5},
                    Case{"corner_cone_boundary_J", 0, V3(-1, 0, 0)}, Case{"interior_node", -1, V3(.6, 0, 0)}, Case{"edge_node", -2, V3(0, -.9, 0)}}) {
        ElementWenoOptions o; o.degree = 1; o.dimension = 2; o.elements = 8; o.periodic = false;
        ElementWeno s(pn, o);
        const int node = cs.node == -1 ? 4 + 4 * s.per_direction : cs.node == -2 ? 4 : cs.node;
        Field a = Field::Zero(s.nodes, pn.q);
        Vec row = Vec::Zero(pn.q);
        set_low(row, Four(1., cs.J[0], cs.J[1], cs.J[2]));
        while (!exact::cone(project(row))) row[0] = std::nextafter(row[0], 1e300);
        a.row(node) = row.transpose();
        double m0 = s.lumped[node] * project(row)[0];
        for (bool low : {false, true}) {
            audit_low_order = low;
            Driver dr(s); double dt;
            Field f = a; std::string err;
            try { f = dr.run(a, .1, 1.0, 20, &dt); } catch (const std::exception &e) { err = e.what(); }
            emit_run("T2_vacuum_single_node", cs.name + (low ? "_lowOrder" : "_full") + (err.empty() ? "" : "_ABORTED:" + err), s, dr.g, f, .1, {{"mass_initial", m0}, {"steps_completed", double(dr.g.steps)}});
            audit_low_order = false;
        }
    }
}

static void T3_polynomial_beam() {
    Angular pn(3);
    for (double power : {3.}) {
        auto psi = [&](const V3 &om) { return std::pow(1 - om[0], power); };
        const Vec c = psi_coeffs(pn, psi);
        const Four P = project(c);
        ElementWenoOptions o; o.degree = 2; o.dimension = 2; o.elements = 4; o.periodic = false;
        Inflow in; in.density = [&](const V3 &, double, const V3 &om) { return psi(om); };
        ElementWeno s(pn, o, in);
        Field a = s.interpolate([&](const V3 &) { return c; });
        ElementWenoStats st; ElementWenoDebug dbg;
        const Field r = s.rate(0, a, &st, &dbg);
        const Energy en = measure(s, 0, a);
        // check that the PN expansion equals psi exactly (degree <= 3 polynomial)
        double rep = 0;
        for (const auto &p : sphere_rule(pn.order + 4)) rep = std::max(rep, std::abs(harmonics(3, p.omega).dot(c) - psi(p.omega)));
        J("T3_polynomial_beam_stationary").add("rho", P[0]).add("J_over_rho", P.tail<3>().norm() / P[0])
            .add("representation_error_psi", rep).add("max_abs_rate_all_coeffs_accepted", r.cwiseAbs().maxCoeff())
            .add("max_abs_rate_PN_target", s.solve_mass(Field::Zero(s.nodes, pn.q)).size() ? (dbg.ho_moment_rate).cwiseAbs().maxCoeff() : 0.)
            .add("min_boundary_theta", st.min_boundary_limiter).add("limited_boundary_nodes", double(st.limited_boundary_nodes))
            .add("mass_rate_accepted", en.mass_accepted).add("mass_rate_pn_target", en.mass_target)
            .add("mass_rate_expected_from_boundary_formula", en.mass_expected).add("mass_balance_resid_rel", en.mass_resid_rel)
            .add("energy_rate_total", en.total).add("energy_boundary_term", en.bnd).add("energy_volume_term", en.vol)
            .add("energy_sync", en.sync).add("N2", en.N2).done();
        Driver dr(s); double dt;
        Field f = dr.run(a, .1, .9, 10, &dt);
        emit_run("T3_polynomial_beam_run", "full", s, dr.g, f, .1, {{"max_abs_deviation_from_initial", (f - a).cwiseAbs().maxCoeff()}});
    }
}

// Same formula as element_weno_smooth_exact (src/element_weno.cpp), with the eigendecomposition cached.
static Vec exact_pn(const Angular &pn, const V3 &x, double t, double kappa) {
    static int cached_order = -1;
    static Eigen::SelfAdjointEigenSolver<Mat> eig;
    static Vec coefficients;
    if (cached_order != pn.order) {
        eig.compute(pn.normal(V3::UnitX()));
        Vec e0 = Vec::Zero(pn.q); e0[0] = 1;
        coefficients = eig.eigenvectors().transpose() * e0;
        cached_order = pn.order;
    }
    const double c = 1 / std::sqrt(4 * pi), s = .2 * c;
    Vec modes(pn.q);
    for (int k = 0; k < pn.q; ++k) modes[k] = coefficients[k] * std::sin(2 * pi * (x[0] - eig.eigenvalues()[k] * t));
    Vec out = s * (eig.eigenvectors() * modes);
    out[0] += c;
    Vec c0 = Vec::Zero(pn.q); c0[0] = c;
    return c0 + kappa * (out - c0);
}

static void T6_convergence() {
    const int NPN = std::getenv("T6_PN") ? std::atoi(std::getenv("T6_PN")) : 3;
    Angular pn(NPN);
    const double kappa = .5, T = .25;
    struct V { std::string name; bool weno, limit; };
    for (V v : {V{"full", true, true}, V{"no_weno", false, true}, V{"no_limit", true, false}}) {
        double prev_b = 0, prev_i = 0;
        for (int E : {4, 8, 16}) {
            ElementWenoOptions o; o.degree = 2; o.dimension = 2; o.elements = E; o.periodic = false; o.weno = v.weno; o.limit = v.limit;
            Inflow in;
            in.density = [&](const V3 &x, double t, const V3 &om) { return harmonics(pn.order, om).dot(exact_pn(pn, x, t, kappa)); };
            ElementWeno s(pn, o, in);
            Field a = s.project_l2([&](const V3 &x) { return exact_pn(pn, x, 0, kappa); });
            Driver dr(s);
            double dt;
            Field f;
            try { f = dr.run(a, T, .5, 4, &dt); } catch (const std::exception &e) { J("T6_error").str("what", e.what()).done(); continue; }
            // L2 errors, boundary-layer elements vs interior elements
            const auto [xq, wq] = gauss(5);
            double eb = 0, ei = 0, ab = 0, ai = 0, ebm = 0, eim = 0;
            for (int e = 0; e < s.cells; ++e) {
                V3 org; s.element_origin(e, org);
                for (int qx = 0; qx < 5; ++qx) for (int qy = 0; qy < 5; ++qy) {
                    V3 x = org + s.h * V3((xq[qx] + 1) / 2, (xq[qy] + 1) / 2, 0);
                    const double w = s.h * s.h * wq[qx] * wq[qy] / 4;
                    const Vec ap = s.evaluate(f, x), ae = exact_pn(pn, x, T, kappa);
                    const double dd = project(ap)[0] - project(ae)[0], dj = (project(ap) - project(ae)).tail<3>().norm();
                    const double err2 = dd * dd;
                    if (dr.mesh.elem_bnd[e]) { eb += w * err2; ab += w; ebm = std::max(ebm, std::abs(dd) + 0 * dj); }
                    else { ei += w * err2; ai += w; eim = std::max(eim, std::abs(dd)); }
                }
            }
            const double rb = std::sqrt(eb / ab), ri = ai > 0 ? std::sqrt(ei / ai) : 0;
            J j("T6_convergence");
            j.str("variant", v.name).add("elements", E).add("dt", dt).add("rms_err_density_boundary_layer", rb).add("rms_err_density_interior", ri)
                .add("max_err_boundary_layer", ebm).add("max_err_interior", eim)
                .add("order_boundary", prev_b > 0 ? std::log2(prev_b / rb) : 0.).add("order_interior", prev_i > 0 && ri > 0 ? std::log2(prev_i / ri) : 0.)
                .add("min_cone_margin_rel", dr.g.min_margin_rel).add("cone_viol", double(dr.g.viol_euler + dr.g.viol_stage))
                .add("limited_frac_interior_el", dr.g.el_tot_i ? double(dr.g.el_lim_i) / dr.g.el_tot_i : 0.)
                .add("limited_frac_boundary_el", dr.g.el_tot_b ? double(dr.g.el_lim_b) / dr.g.el_tot_b : 0.)
                .add("nudges", double(dr.g.nudged)).done();
            prev_b = rb; prev_i = ri;
        }
    }
}

static Field bump_state(const ElementWeno &s, double x0, double y0, double w, double bg, int dirx) {
    const Angular &pn = s.pn;
    const V3 dir = dirx ? V3::UnitX() : V3::UnitY();
    const Vec y = harmonics(pn.order, dir);
    return s.interpolate([&](const V3 &x) {
        Vec v = Vec::Zero(pn.q); v[0] = bg / std::sqrt(4 * pi);
        v += std::exp(-(x - V3(x0, y0, 0)).squaredNorm() / (2 * w * w)) * y;
        return v;
    });
}

static void T7_steep_near_boundary() {
    Angular pn(3);
    const double T = .08, w = .03, bg = .001;
    Inflow in; in.density = [bg](const V3 &, double, const V3 &) { return bg / (4 * pi); }; in.constant_density = bg / (4 * pi);
    auto run_case = [&](int E, double x0, Field *out_field, ElementWeno **keep) {
        ElementWenoOptions o; o.degree = 2; o.dimension = 2; o.elements = E; o.periodic = false;
        auto *s = new ElementWeno(pn, o, in);
        Field a = bump_state(*s, x0, .5, w, bg, 1);
        auto *dr = new Driver(*s);
        Field f = dr->run(a, T, .5, 4);
        *out_field = f;
        *keep = s;
        return dr;
    };
    for (const std::string where : {"near_boundary_exiting", "interior"}) {
        const double x0 = where == "interior" ? .35 : .90;
        // the interior twin is the same bump translated away from the walls; both travel +x for T (distance ~ .08)
        Field f12, f48;
        ElementWeno *s12, *s48;
        Driver *d12 = run_case(12, x0, &f12, &s12);
        Driver *d48 = run_case(48, x0, &f48, &s48);
        // error of coarse run against fine run on coarse Gauss points
        const auto [xq, wq] = gauss(4);
        double num = 0, den = 0, num_b = 0, den_b = 0;
        for (int e = 0; e < s12->cells; ++e)
            for (int qx = 0; qx < 4; ++qx) for (int qy = 0; qy < 4; ++qy) {
                V3 org; s12->element_origin(e, org);
                V3 x = org + s12->h * V3((xq[qx] + 1) / 2, (xq[qy] + 1) / 2, 0);
                const double w2 = wq[qx] * wq[qy];
                const double d = project(s12->evaluate(f12, x))[0] - project(s48->evaluate(f48, x))[0], r = project(s48->evaluate(f48, x))[0];
                num += w2 * d * d; den += w2 * r * r;
                if (x[0] > .75) { num_b += w2 * d * d; den_b += w2 * r * r; }
            }
        emit_run("T7_steep_gradient", where, *s12, d12->g, f12, T,
                 {{"rel_L2_error_vs_E48", std::sqrt(num / den)}, {"rel_L2_error_vs_E48_x_gt_0p75", den_b > 0 ? std::sqrt(num_b / den_b) : 0.},
                  {"fine_gamma_min_boundary_elems", d48->g.gamma_min_b}, {"fine_gamma_min_interior_elems", d48->g.gamma_min_i},
                  {"fine_limited_frac_interior", d48->g.el_tot_i ? double(d48->g.el_lim_i) / d48->g.el_tot_i : 0.},
                  {"fine_limited_frac_boundary", d48->g.el_tot_b ? double(d48->g.el_lim_b) / d48->g.el_tot_b : 0.}});
        delete d12; delete d48; delete s12; delete s48;
    }
}

static void T8_scaling() {
    Angular pn(3);
    ElementWenoOptions o; o.degree = 2; o.dimension = 2; o.elements = 8; o.periodic = false;
    ElementWeno s(pn, o);  // no inflow: vacuum inflow, so the problem is homogeneous of degree 1
    Field base = bump_state(s, .5, .5, .1, .05, 1);
    base += bump_state(s, .3, .6, .08, .0, 0);
    const double dt = .5 * s.step_bound();
    Field ref;
    for (double sc : {1., 1e-3, 1e-6, 1e-9, 1e-30, 1e-100, 0.}) {
        for (bool half_vacuum : {false, true}) {
            Field a = base * sc;
            if (half_vacuum) for (int i = 0; i < s.nodes; ++i) if (s.node_position(i)[0] < .5) a.row(i).setZero();
            Driver dr(s);
            Field f;
            std::string err;
            try { f = dr.run(a, 3 * dt, 1.0, 1, nullptr, dt); } catch (const std::exception &e) { err = e.what(); f = a; }
            if (sc == 1. && !half_vacuum) ref = f;
            // scaled difference to the unit-scale run, same inputs only if half_vacuum == false
            double dev = -1;
            if (!half_vacuum && sc > 0) dev = (f / sc - ref).cwiseAbs().maxCoeff() / ref.cwiseAbs().maxCoeff();
            const ElementWenoStats z;
            J("T8_scale").add("scale", sc).add("half_vacuum", half_vacuum).add("rel_dev_from_scale1_after_3_steps", dev)
                .add("cone_viol", double(dr.g.viol_euler + dr.g.viol_stage)).add("nudges", double(dr.g.nudged))
                .add("min_margin_rel", dr.g.min_margin_rel).add("gamma_min", std::min(dr.g.gamma_min_b, dr.g.gamma_min_i))
                .add("frac_limited_el", (dr.g.el_tot_i + dr.g.el_tot_b) ? double(dr.g.el_lim_i + dr.g.el_lim_b) / (dr.g.el_tot_i + dr.g.el_tot_b) : 0.)
                .add("final_all_finite", f.allFinite()).str("error", err).done();
        }
    }
}

static void crossing(int E, int p, int N, const std::string &variant, double cfl, Field *keep, Field *ref_full, ElementWeno **keep_scheme) {
    Angular pn(N);
    ElementWenoOptions o; o.degree = p; o.dimension = 2; o.elements = E; o.periodic = false;
    o.weno = variant != "no_weno"; o.limit = variant != "no_limit_diagnostic";
    const double bg = .001, w = .07;
    Inflow in;
    in.density = [bg](const V3 &, double, const V3 &) { return bg / (4 * pi); };
    in.constant_density = bg / (4 * pi);
    in.atoms = [w](const V3 &x, double t) {
        auto pp = [w](const V3 &d) { return std::exp(-d.squaredNorm() / (2 * w * w)); };
        return std::vector<Atom>{{V3::UnitX(), pp(x - t * V3::UnitX() - V3(.3, .5, 0))}, {V3::UnitY(), pp(x - t * V3::UnitY() - V3(.5, .3, 0))}};
    };
    auto *s = new ElementWeno(pn, o, in);
    Field a = s->interpolate([&](const V3 &x) {
        auto pp = [&](const V3 &d) { return std::exp(-d.squaredNorm() / (2 * w * w)); };
        Vec v = Vec::Zero(pn.q); v[0] = bg / std::sqrt(4 * pi);
        v += pp(x - V3(.3, .5, 0)) * harmonics(pn.order, V3::UnitX());
        v += pp(x - V3(.5, .3, 0)) * harmonics(pn.order, V3::UnitY());
        return v;
    });
    audit_low_order = variant == "low_order_control";
    Driver dr(*s);
    Field f;
    std::string err;
    try { f = dr.run(a, .32, cfl, 8); } catch (const std::exception &e) { err = e.what(); f = a; }
    std::map<std::string, double> extra;
    if (ref_full && ref_full->size() > 0) {
        extra["rel_L2_density_diff_to_full"] = std::sqrt(((f.col(0) - ref_full->col(0)).array().square() * s->lumped.array()).sum() / ((ref_full->col(0).array().square()) * s->lumped.array()).sum());
    }
    // density field extrema over nodes and within the beam region
    emit_run("CROSSING", variant + (err.empty() ? "" : "_ABORTED:" + err), *s, dr.g, f, .32, extra);
    audit_low_order = false;
    if (keep) *keep = f;
    if (keep_scheme) *keep_scheme = s; else delete s;
}

int main(int argc, char **argv) {
    try {
        const std::string what = argc > 1 ? argv[1] : "small";
        auto safe = [](const char *n, void (*f)()) { try { f(); } catch (const std::exception &e) { J("EXCEPTION").str("in", n).str("what", e.what()).done(); } };
        if (what == "small" || what == "all") { safe("T1", T1_isotropic); safe("T2", T2_vacuum); safe("T3", T3_polynomial_beam); safe("T8", T8_scaling); }
        if (what == "T6" || what == "all") safe("T6", T6_convergence);
        if (what == "T7" || what == "all") safe("T7", T7_steep_near_boundary);
        if (what == "crossing_small") {
            Field full; ElementWeno *s = nullptr;
            for (const std::string v : {"full", "no_weno", "low_order_control", "no_limit_diagnostic"}) {
                Field f; ElementWeno *sk = nullptr;
                crossing(16, 1, 3, v, .5, &f, v == "full" ? nullptr : &full, &sk);
                if (v == "full") { full = f; s = sk; } else delete sk;
            }
            delete s;
        }
        if (what == "crossing") {
            const int E = argc > 2 ? std::atoi(argv[2]) : 64, p = argc > 3 ? std::atoi(argv[3]) : 1, N = argc > 4 ? std::atoi(argv[4]) : 7;
            Field full; ElementWeno *s = nullptr;
            for (const std::string v : {"full", "no_weno", "low_order_control", "no_limit_diagnostic"}) {
                Field f; ElementWeno *sk = nullptr;
                crossing(E, p, N, v, .5, &f, v == "full" ? nullptr : &full, &sk);
                if (v == "full") { full = f; s = sk; } else delete sk;
            }
            delete s;
        }
    } catch (const std::exception &e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
