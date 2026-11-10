// Operator-level audits: A (PN boundary operator) and B (WENO stabilization).
#include "audit_common.hpp"

static Mat symm(const Mat &m) { return .5 * (m + m.transpose()); }
static double lam_max(const Mat &m) { return Eigen::SelfAdjointEigenSolver<Mat>(symm(m)).eigenvalues().maxCoeff(); }
static double lam_min(const Mat &m) { return Eigen::SelfAdjointEigenSolver<Mat>(symm(m)).eigenvalues().minCoeff(); }

static std::vector<std::pair<std::string, V3>> normals() {
    V3 rot(.3, -.5, .8);
    return {{"ex", V3::UnitX()}, {"ey", V3::UnitY()}, {"diag_xy", V3(1, 1, 0).normalized()},
            {"diag_xyz", V3(1, 1, 1).normalized()}, {"rotated", rot.normalized()}};
}
static Mat half_range(const Boundary &b, int N, int sign) {
    Mat H = Mat::Zero(b.pn.q, b.pn.q);
    for (const auto &p : sphere_rule(2 * N + 1, sign)) {
        const V3 w = b.frame * p.omega;
        const Vec y = harmonics(N, w);
        H.noalias() += p.weight * b.n.dot(w) * (y * y.transpose());
    }
    return H;
}
static Mat exterior_map(const Boundary &b) {
    const int q = b.pn.q;
    Mat E(q, q);
    const Vec d = Vec::Zero(b.canonical.r);
    for (int k = 0; k < q; ++k) E.col(k) = b.exterior(Vec::Unit(q, k), d);
    return E;
}

void audit_boundary_operator() {
    for (int N : {3, 5, 7}) {
        Angular pn(N);
        CanonicalC1 c(pn);
        const int q = pn.q;
        for (const auto &[name, n] : normals()) {
            Boundary b(pn, c, n, false);
            const Mat A = pn.normal(n);
            Eigen::SelfAdjointEigenSolver<Mat> es(A);
            int np = 0, nn = 0, nz = 0;
            std::vector<int> zero_idx;
            for (int k = 0; k < q; ++k) {
                const double l = es.eigenvalues()[k];
                if (l > 1e-9) ++np; else if (l < -1e-9) ++nn; else { ++nz; zero_idx.push_back(k); }
            }
            Mat Z(q, nz);
            for (int k = 0; k < nz; ++k) Z.col(k) = es.eigenvectors().col(zero_idx[k]);
            const Mat Ap = es.eigenvectors() * es.eigenvalues().cwiseMax(0.).asDiagonal() * es.eigenvectors().transpose();
            const Mat Hp = half_range(b, N, +1), Hm = half_range(b, N, -1);
            const Mat E = exterior_map(b);
            const Mat I = Mat::Identity(q, q);
            const Mat Feff = A - .5 * (A - I) * (I - E);       // implemented numerical flux, d = 0
            const Mat Q = .5 * (A - I) * (I - E) - .5 * A;      // u^T Q u = face energy rate density (volume+boundary), d = 0
            const Mat Qchar = -b.homogeneous;                    // energy rate of the characteristic (C1) flux
            const Mat ZHZ = Z.transpose() * Hp * Z;
            J j("A_boundary_operator");
            j.add("pn_order", N).str("normal", name).add("q", q).add("n_pos", np).add("n_neg", nn).add("n_zero", nz)
                .add("r_expected", N * (N + 1) / 2).add("null_expected", N + 1)
                .add("A_eq_Hp_plus_Hm", (Hp + Hm - A).norm() / A.norm())
                .add("rel_Hp_minus_Aplus", (Hp - Ap).norm() / A.norm())
                .add("Hp_sym_defect", (Hp - Hp.transpose()).norm() / Hp.norm())
                .add("lambda_min_Hp", Eigen::SelfAdjointEigenSolver<Mat>(symm(Hp)).eigenvalues().minCoeff())
                .add("Aplus_Z_norm", (Ap * Z).norm()).add("A_Z_norm", (A * Z).norm())
                .add("Hp_Z_norm", (Hp * Z).norm()).add("Hp_Z_norm_rel", (Hp * Z).norm() / std::sqrt(double(nz)))
                .add("Zt_Hp_Z_min", Eigen::SelfAdjointEigenSolver<Mat>(symm(ZHZ)).eigenvalues().minCoeff())
                .add("Zt_Hp_Z_max", Eigen::SelfAdjointEigenSolver<Mat>(symm(ZHZ)).eigenvalues().maxCoeff())
                .add("Hp_Z_equals_minus_Hm_Z", (Hp * Z + Hm * Z).norm())
                .add("exterior_copies_null", (E * Z - Z).norm())
                .add("Feff_minus_char_flux_rel", (Feff - b.homogeneous).norm() / b.homogeneous.norm())
                .add("Feff_minus_char_flux_on_null", ((Feff - b.homogeneous) * Z).norm())
                .add("Feff_minus_Ap_rel_on_outflow", ((Feff - b.homogeneous) * b.Rplus).norm())
                .add("lambda_max_Q_implemented", lam_max(Q)).add("lambda_min_Q_implemented", lam_min(Q))
                .add("lambda_max_Q_characteristic", lam_max(Qchar))
                .add("null_energy_ZtQZ_abs", (Z.transpose() * Q * Z).cwiseAbs().maxCoeff())
                .add("rho_A_spectral", es.eigenvalues().cwiseAbs().maxCoeff())
                .add("llf_norm_tabulated", b.llf_norm).add("contraction", c.contraction)
                .done();
        }
    }
}

// T4/T5: beams and null modes on one face. State = harmonic expansion of a delta in direction Omega0.
void audit_beams_nullmodes() {
    for (int N : {3, 5, 7}) {
        Angular pn(N);
        CanonicalC1 c(pn);
        const int q = pn.q;
        for (const auto &[name, n] : {std::pair<std::string, V3>{"ex", V3::UnitX()}, {"diag_xy", V3(1, 1, 0).normalized()}}) {
            Boundary b(pn, c, n, false);
            const Mat A = pn.normal(n), Hp = half_range(b, N, +1), E = exterior_map(b), I = Mat::Identity(q, q);
            const Mat Q = .5 * (A - I) * (I - E) - .5 * A, Qchar = -b.homogeneous;
            const V3 t = b.frame.col(0);
            const double mus[] = {1., .5, .1, .01, 0., -.01, -.1, -.5, -1.};
            for (double mu : mus) {
                const V3 om = mu * n + std::sqrt(std::max(0., 1 - mu * mu)) * t;
                const Vec u = harmonics(N, om);
                const Vec out = b.Rplus.transpose() * u, in = b.Rminus.transpose() * u;
                const double tot = u.squaredNorm(), zero = tot - out.squaredNorm() - in.squaredNorm();
                const Mat Ap = Eigen::SelfAdjointEigenSolver<Mat>(A).eigenvectors() *
                               Eigen::SelfAdjointEigenSolver<Mat>(A).eigenvalues().cwiseMax(0.).asDiagonal() *
                               Eigen::SelfAdjointEigenSolver<Mat>(A).eigenvectors().transpose();
                J j("T4_beam_face");
                j.add("pn_order", N).str("normal", name).add("mu", mu).add("frac_outgoing_modes", out.squaredNorm() / tot)
                    .add("frac_incoming_modes", in.squaredNorm() / tot).add("frac_null_modes", zero / tot)
                    .add("face_energy_rate_implemented_over_norm2", u.dot(Q * u) / tot)
                    .add("face_energy_rate_characteristic_over_norm2", u.dot(Qchar * u) / tot)
                    .add("kinetic_halfrange_outflow_mass", (Hp * u)[0] * std::sqrt(4 * pi))
                    .add("characteristic_outflow_mass", (Ap * u)[0] * std::sqrt(4 * pi))
                    .add("net_flux_mass_A", (A * u)[0] * std::sqrt(4 * pi)).done();
            }
            // T5: random null vectors
            Eigen::SelfAdjointEigenSolver<Mat> es(A);
            std::vector<int> zi;
            for (int k = 0; k < q; ++k) if (std::abs(es.eigenvalues()[k]) < 1e-9) zi.push_back(k);
            std::mt19937 rng(11);
            std::normal_distribution<double> g(0, 1);
            Vec z = Vec::Zero(q);
            for (int k : zi) z += g(rng) * es.eigenvectors().col(k);
            z.normalize();
            const Vec Hz = Hp * z;
            // minimum of psi_z on the sphere (is the null mode a positive angular function?)
            double minpsi = 1e300, minpsi_out = 1e300;
            for (const auto &p : sphere_rule(N + 6)) {
                const double v = harmonics(N, p.omega).dot(z);
                minpsi = std::min(minpsi, v);
                if (n.dot(p.omega) > 0) minpsi_out = std::min(minpsi_out, v);
            }
            const Four P = project(z);
            J("T5_null_mode").add("pn_order", N).str("normal", name).add("A_z", (A * z).norm()).add("Hp_z", Hz.norm())
                .add("zt_Hp_z", z.dot(Hz)).add("Hp_z_mass_row", Hz[0] * std::sqrt(4 * pi))
                .add("exterior_minus_z", (E * z - z).norm()).add("face_energy_rate_implemented", z.dot(Q * z))
                .add("char_flux_z", (b.homogeneous * z).norm()).add("Feff_z", ((A - .5 * (A - I) * (I - E)) * z).norm())
                .add("min_psi_z", minpsi).add("moments_rho", P[0]).add("moments_J_norm", P.tail<3>().norm())
                .add("n_dot_J_M1", n.dot(P.tail<3>())).done();
        }
    }
}

// ---------------- B: stabilization operator ----------------
static Mat dense_S(const ElementWeno &s, const Vec &gamma) {
    Mat S(s.nodes, s.nodes);
    for (int j = 0; j < s.nodes; ++j) {
        Field a = Field::Zero(s.nodes, s.pn.q);
        a(j, 0) = 1;
        S.col(j) = s.gather_load(s.stabilization_load(a, gamma)).col(0);
    }
    return S;
}
static Mat dense_mass(const ElementWeno &s) {
    Mat M(s.nodes, s.nodes);
    for (int j = 0; j < s.nodes; ++j) {
        Field a = Field::Zero(s.nodes, s.pn.q);
        a(j, 0) = 1;
        Field local(s.cells, s.local_nodes * s.pn.q);
        for (int e = 0; e < s.cells; ++e)
            for (int l = 0; l < s.local_nodes; ++l) {
                double v = 0;
                for (int m = 0; m < s.local_nodes; ++m) v += s.mass_(l, m) * a(s.connectivity_[e * s.local_nodes + m], 0);
                for (int cc = 0; cc < s.pn.q; ++cc) local(e, l * s.pn.q + cc) = cc == 0 ? v : 0.;
            }
        M.col(j) = s.gather_load(local).col(0);
    }
    return M;
}

void audit_stabilization() {
    Angular pn(1);
    struct Cfg { int p, dim, E; bool per; double lin; };
    std::vector<Cfg> cfgs = {{1, 1, 8, true, 1e-3}, {2, 1, 6, true, 1e-3}, {3, 1, 6, true, 1e-3}, {2, 1, 6, false, 1e-3},
                             {2, 2, 3, true, 1e-3}, {2, 2, 3, false, 1e-3}, {2, 1, 6, true, .2}};
    for (const auto &cf : cfgs) {
        ElementWenoOptions o;
        o.degree = cf.p; o.dimension = cf.dim; o.elements = cf.E; o.periodic = cf.per; o.weno_linear = cf.lin;
        ElementWeno s(pn, o);
        const Vec one = Vec::Ones(s.cells);
        const Mat S1 = dense_S(s, one), S0 = dense_S(s, Vec::Zero(s.cells));
        const double ref = S0.norm();
        std::mt19937 rng(5);
        std::uniform_real_distribution<double> U(0, 1);
        Vec gr(s.cells);
        for (int e = 0; e < s.cells; ++e) gr[e] = U(rng);
        const Mat Sr = dense_S(s, gr);
        // worst single-element pattern: gamma_e = 1 on one element, 0 elsewhere; minimum symmetric eigenvalue
        double worst_single = 0, worst_single_sym = 0;
        Vec best_pat = one;
        double worst_pat_eig = 1e300;
        const int combos = s.cells <= 12 ? (1 << s.cells) : 0;
        for (int mask = 0; mask < combos; ++mask) {
            Vec g(s.cells);
            for (int e = 0; e < s.cells; ++e) g[e] = (mask >> e) & 1;
            const Mat S = dense_S(s, g);
            const double me = lam_min(S);
            if (me < worst_pat_eig) { worst_pat_eig = me; best_pat = g; }
            worst_single_sym = std::max(worst_single_sym, (S - S.transpose()).norm());
        }
        (void)worst_single;
        // consistency: row sums, eigenvalues
        Vec ones = Vec::Ones(s.nodes);
        // polynomial reproduction (nonperiodic): all x^a y^b, a,b<=p, with gamma = 1
        double poly_err1 = -1, poly_err_weno = -1, poly_gamma_dev = -1;
        if (!cf.per) {
            poly_err1 = poly_err_weno = poly_gamma_dev = 0;
            for (int a = 0; a <= cf.p; ++a)
                for (int bb = 0; bb <= (cf.dim > 1 ? cf.p : 0); ++bb) {
                    ElementWenoOptions o2 = o;
                    Angular pn1(1);
                    Field f = s.project_l2([&](const V3 &x) {
                        Vec v = Vec::Zero(pn1.q);
                        v[0] = 1 + std::pow(x[0], a) * std::pow(x[1], bb);
                        return v;
                    });
                    const double sc = f.col(0).norm();
                    Field l1 = s.gather_load(s.stabilization_load(f, one));
                    poly_err1 = std::max(poly_err1, l1.col(0).norm() / sc);
                    const Vec g = s.weno_gamma(f);
                    poly_gamma_dev = std::max(poly_gamma_dev, (g - one).cwiseAbs().maxCoeff());
                    Field lw = s.gather_load(s.stabilization_load(f, g));
                    poly_err_weno = std::max(poly_err_weno, lw.col(0).norm() / sc);
                }
        }
        const double nrm = ref;
        J("B_stabilization_matrix").add("degree", cf.p).add("dim", cf.dim).add("elements", cf.E).add("periodic", cf.per)
            .add("weno_linear", cf.lin).add("nodes", s.nodes)
            .add("S1_row_sum_max", (S1 * ones).cwiseAbs().maxCoeff() / nrm)
            .add("Sgamma_random_row_sum_max", (Sr * ones).cwiseAbs().maxCoeff() / nrm)
            .add("S1_sym_defect_rel", (S1 - S1.transpose()).norm() / nrm)
            .add("S0_sym_defect_rel", (S0 - S0.transpose()).norm() / nrm)
            .add("S1_lambda_min_rel", lam_min(S1) / lam_max(S0))
            .add("S0_lambda_min_rel", lam_min(S0) / lam_max(S0))
            .add("Srandom_sym_defect_rel", (Sr - Sr.transpose()).norm() / nrm)
            .add("Srandom_lambda_min_sym_rel", lam_min(Sr) / lam_max(S0))
            .add("binary_pattern_worst_lambda_min_rel", combos ? worst_pat_eig / lam_max(S0) : 0.)
            .add("binary_pattern_worst_sym_defect_rel", worst_single_sym / nrm)
            .add("poly_reproduction_err_gamma1", poly_err1).add("poly_reproduction_err_weno", poly_err_weno)
            .add("poly_gamma_deviation", poly_gamma_dev).done();
    }
}

// effective operator S(gamma(a)): search for u^T S(gamma(u)) u < 0 (energy production by the stabilizer)
void audit_stabilization_effective() {
    Angular pn(1);
    for (double lin : {1e-3, .2}) {
        for (int p : {1, 2, 3}) {
            ElementWenoOptions o;
            o.degree = p; o.dimension = 1; o.elements = 8; o.periodic = true; o.weno_linear = lin;
            ElementWeno s(pn, o);
            const Vec one = Vec::Ones(s.cells);
            const Mat S1 = dense_S(s, one), S0 = dense_S(s, Vec::Zero(s.cells));
            auto f = [&](const Vec &v, double *gmin = nullptr) {
                Field a = Field::Zero(s.nodes, pn.q);
                a.col(0) = v;
                const Vec g = s.weno_gamma(a);
                if (gmin) *gmin = g.minCoeff();
                const Field l = s.gather_load(s.stabilization_load(a, g));
                const double num = v.dot(l.col(0)), den = v.dot(S0 * v);
                return den > 0 ? num / den : 1.;
            };
            std::mt19937 rng(17);
            std::normal_distribution<double> N01(0, 1);
            std::uniform_real_distribution<double> U(0, 1);
            double best = 1e300, bestg = 1;
            Vec bv;
            long evals = 0;
            for (int trial = 0; trial < 60; ++trial) {
                Vec v(s.nodes);
                for (int i = 0; i < s.nodes; ++i) {
                    const double x = double(i) / s.nodes;
                    v[i] = (trial % 3 == 0) ? 1 + .3 * N01(rng)
                         : (trial % 3 == 1) ? (x > .5 ? 1. : 1e-3) * (1 + .05 * N01(rng))
                                            : std::exp(3 * N01(rng));
                }
                double val = f(v);
                double step = .5;
                for (int it = 0; it < 250; ++it) {
                    Vec w = v;
                    const int i = int(U(rng) * s.nodes) % s.nodes;
                    w[i] *= std::exp(step * N01(rng));
                    ++evals;
                    const double nv = f(w);
                    if (nv < val) { v = w; val = nv; }
                    if (it % 80 == 79) step *= .5;
                }
                if (val < best) { best = val; bv = v; f(v, &bestg); }
            }
            const double lam_ratio_min1 = lam_min(S1) / lam_max(S0);
            J j("B_stabilization_effective_energy");
            j.add("degree", p).add("weno_linear", lin).add("evals", double(evals)).add("min_Rayleigh_uSu_over_uS0u", best)
                .add("gamma_min_at_minimizer", bestg).add("S1_lambda_min_rel", lam_ratio_min1);
            if (best < 0) {
                std::ostringstream os;
                for (int i = 0; i < bv.size(); ++i) os << (i ? "," : "") << std::setprecision(10) << bv[i];
                j.str("counterexample_density_nodes", os.str());
            }
            j.done();
        }
    }
}

// weno_gamma fallbacks and scale dependence (epsilon = 1e-6 is absolute)
void audit_weno_sensor_scaling() {
    Angular pn(1);
    ElementWenoOptions o;
    o.degree = 2; o.dimension = 1; o.elements = 16; o.periodic = false; o.weno_linear = 1e-3;
    ElementWeno s(pn, o);
    for (double scale : {1., 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-9, 1e-30}) {
        Field a = Field::Zero(s.nodes, pn.q);
        for (int i = 0; i < s.nodes; ++i) {
            const double x = double(i) / (s.nodes - 1);
            a(i, 0) = scale * (1 + .5 * std::tanh((x - .5) / .03) + .2 * std::sin(9 * x));
        }
        const Vec g = s.weno_gamma(a);
        J("T8_weno_sensor_scale").add("scale", scale).add("gamma_min", g.minCoeff()).add("gamma_mean", g.mean())
            .add("n_gamma_lt_0p99", double((g.array() < .99).count())).done();
    }
    // exactly constant / zero element -> fallback gamma = 1
    Field z = Field::Zero(s.nodes, pn.q);
    J("B_sensor_fallback_zero_density").add("gamma_min", s.weno_gamma(z).minCoeff()).done();
    Field cst = Field::Zero(s.nodes, pn.q);
    cst.col(0).setConstant(3.);
    J("B_sensor_fallback_constant_density").add("gamma_min", s.weno_gamma(cst).minCoeff()).done();
}

int main(int argc, char **argv) {
    try {
        const std::string what = argc > 1 ? argv[1] : "all";
        if (what == "all" || what == "A") { audit_boundary_operator(); audit_beams_nullmodes(); }
        if (what == "all" || what == "B") { audit_stabilization(); audit_stabilization_effective(); audit_weno_sensor_scaling(); }
    } catch (const std::exception &e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
