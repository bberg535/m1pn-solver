// Tests of the dynamical low-rank element-WENO scheme (src/element_weno_dlra.{hpp,cpp}, docs/DLRA.md).
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#define private public
#include "element_weno.hpp"
#include "element_weno_dlra.hpp"
#undef private
using namespace m1pn;
namespace {
int failures = 0;
void check(bool ok, const std::string &what, double value = 0) {
    std::cout << (ok ? "ok   " : "FAIL ") << what << "  [" << std::setprecision(6) << value << "]\n";
    if (!ok) ++failures;
}
Inflow crossing_inflow() {
    Inflow inflow;
    const double bg = 1e-3, w = .07;
    inflow.density = [bg](const V3 &, double, const V3 &) { return bg / (4 * pi); };
    inflow.constant_density = bg / (4 * pi);
    inflow.atoms = [w](const V3 &x, double t) {
        auto p = [w](const V3 &d) { return std::exp(-d.squaredNorm() / (2 * w * w)); };
        return std::vector<Atom>{{V3::UnitX(), p(x - t * V3::UnitX() - V3(0, .5, 0))},
                                 {V3::UnitY(), p(x - t * V3::UnitY() - V3(.5, 0, 0))}};
    };
    return inflow;
}
// Random realizable low moments and a random rank-k high part Y V^T.
void random_state(const ElementWeno &s, int qh, int k, Field &low, Field &Y, Mat &V, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1, 1);
    low.resize(s.nodes, 4);
    for (int i = 0; i < s.nodes; ++i) {
        const double rho = 1 + .3 * u(rng);
        Four m(rho, .3 * rho * u(rng), .3 * rho * u(rng), .3 * rho * u(rng));
        Vec c(4);
        set_low(c, m);
        low.row(i) = c.transpose();
    }
    Y.resize(s.nodes, k);
    for (int i = 0; i < s.nodes; ++i)
        for (int j = 0; j < k; ++j)
            Y(i, j) = .05 * u(rng);
    Mat R(qh, k);
    for (int i = 0; i < qh; ++i)
        for (int j = 0; j < k; ++j)
            R(i, j) = u(rng);
    Eigen::HouseholderQR<Mat> qr(R);
    V = qr.householderQ() * Mat::Identity(qh, k);
}
// The factored evaluation equals scheme.rate() on the expanded state (open mesh, inflow, C6 ghost, WENO).
void factored_rate_matches(bool reduce, int dimension, bool periodic) {
    Angular pn(5);
    ElementWenoOptions o;
    o.degree = 2, o.dimension = dimension, o.elements = dimension == 1 ? 12 : 5, o.periodic = periodic;
    o.reduce_angular = reduce;
    const ElementWeno scheme(pn, o, periodic || dimension != 2 ? Inflow{} : crossing_inflow());
    const ElementWenoDlra dlra(scheme);
    Field low, Y, Hfac, Lfac;
    Mat V;
    random_state(scheme, dlra.qh, 3, low, Y, V, 7 + reduce);
    Field full(scheme.nodes, scheme.moments);
    full.leftCols(4) = low;
    full.rightCols(dlra.qh) = Y * V.transpose();
    const Field R = scheme.rate(.1, full);
    Mat U = Mat::Random(dlra.qh, 4);
    dlra.factored_rate(.1, low, Y, V, U, Lfac, Hfac);
    const double scale_l = R.leftCols(4).cwiseAbs().maxCoeff();
    const Field Rh = R.rightCols(dlra.qh) * U;
    const double el = (R.leftCols(4) - Lfac).cwiseAbs().maxCoeff() / scale_l;
    const double eh = (Rh - Hfac).cwiseAbs().maxCoeff() / Rh.cwiseAbs().maxCoeff();
    const std::string tag = std::string(dimension == 1 ? "1D" : "2D") + (periodic ? " periodic" : " open+inflow") +
                            (reduce ? ", parity reduced" : ", full basis");
    check(el < 1e-10, "factored low rate == rate() (" + tag + ")", el);
    check(eh < 1e-10, "factored high rate == rate() (" + tag + ")", eh);
}
// react_low equals the first four columns of scheme.react (constant and space-dependent reaction).
void react_matches() {
    Angular pn(3);
    for (int material = 0; material < 2; ++material) {
        ElementWenoOptions o;
        o.degree = 2, o.dimension = 2, o.elements = 4, o.periodic = false, o.filter_strength = 2;
        if (material) {
            auto m = std::make_shared<ElementWenoMaterial>();
            m->sigma_a = [](const V3 &x) { return 1 + x[0]; };
            m->sigma_s = [](const V3 &x) { return 2 + x[1]; };
            m->source = [](const V3 &x) { return x[0] * x[1]; };
            o.material = m;
        } else
            o.sigma_s = 3, o.sigma_a = .5;
        const ElementWeno scheme(pn, o);
        const ElementWenoDlra dlra(scheme);
        Field low, Y;
        Mat V;
        random_state(scheme, dlra.qh, 2, low, Y, V, 3);
        Field full(scheme.nodes, scheme.moments);
        full.leftCols(4) = low;
        full.rightCols(dlra.qh) = Y * V.transpose();
        ElementWenoStats s1, s2;
        const Field ref = scheme.react(full, .05, &s1);
        const Field mine = dlra.react_low(low, .05, 0., &s2);
        const double e = (ref.leftCols(4) - mine).cwiseAbs().maxCoeff();
        check(e < 1e-15 && std::abs(s1.absorbed - s2.absorbed) < 1e-14 && std::abs(s1.injected - s2.injected) < 1e-14,
              std::string("react_low == react() low rows (") + (material ? "material" : "constant") + ")", e);
        // High part of the factored reaction step.
        DlraState st = dlra.compress(full);
        const DlraState rs = dlra.react(st, .05, 0., nullptr);
        const double eh = (dlra.expand(rs).rightCols(dlra.qh) - ref.rightCols(dlra.qh)).cwiseAbs().maxCoeff();
        check(eh < 1e-13, std::string("factored reaction of l >= 2 (") + (material ? "material" : "constant") + ")", eh);
    }
}
// Compression: exact for rank-k data, expand(compress(a)) == a.
void compression() {
    Angular pn(7);
    ElementWenoOptions o;
    o.degree = 2, o.dimension = 2, o.elements = 6, o.periodic = true;
    const ElementWeno scheme(pn, o);
    DlraOptions d;
    d.tolerance = 1e-6, d.initial_rank = 2; // Gram eigenvalues resolve ~1e-8 relative
    const ElementWenoDlra dlra(scheme, d);
    Field low, Y;
    Mat V;
    random_state(scheme, dlra.qh, 5, low, Y, V, 11);
    Field full(scheme.nodes, scheme.moments);
    full.leftCols(4) = low;
    full.rightCols(dlra.qh) = Y * V.transpose();
    const DlraState st = dlra.compress(full);
    const double e = (dlra.expand(st) - full).cwiseAbs().maxCoeff();
    const Mat G = st.X.transpose() * scheme.apply_mass(st.X);
    check(st.rank() == 5 && e < 1e-12, "compress: rank 5 data, exact reconstruction", e);
    check((G - Mat::Identity(5, 5)).cwiseAbs().maxCoeff() < 1e-12 && (st.W.transpose() * st.W - Mat::Identity(5, 5)).cwiseAbs().maxCoeff() < 1e-12,
          "compress: X^T M X = I, W^T W = I", (G - Mat::Identity(5, 5)).cwiseAbs().maxCoeff());
}
// Crossing beams (open mesh, inflow): DLRA with small tolerance against the full scheme, mass ledger, ranks.
void crossing_against_full() {
    Angular pn(7);
    ElementWenoOptions o;
    o.degree = 2, o.dimension = 2, o.elements = 8, o.periodic = false, o.sigma_s = 1;
    const ElementWeno scheme(pn, o, crossing_inflow());
    const double w = .07;
    Field a = scheme.interpolate([&](const V3 &x) {
        auto p = [&](const V3 &d) { return std::exp(-d.squaredNorm() / (2 * w * w)); };
        Vec s = Vec::Zero(pn.q);
        s[0] = 1e-3 / std::sqrt(4 * pi);
        s += p(x - V3(.3, .5, 0)) * harmonics(pn.order, V3::UnitX());
        s += p(x - V3(.5, .3, 0)) * harmonics(pn.order, V3::UnitY());
        return s;
    });
    const double dt = .5 * scheme.step_bound(), T = 40 * dt;
    Field ref = a;
    for (double t = 0; t < T - 1e-12; t += dt)
        ref = scheme.advance(t, dt, ref);
    double previous = 1e300;
    for (double tol : {1e-2, 1e-3, 1e-5}) {
        DlraOptions d;
        d.tolerance = tol;
        const ElementWenoDlra dlra(scheme, d);
        DlraState st = dlra.compress(a);
        ElementWenoStats stats;
        DlraStats ds;
        double mass0 = 0, mass1 = 0;
        for (int i = 0; i < scheme.nodes; ++i)
            mass0 += scheme.lumped[i] * project(st.low.row(i).transpose())[0];
        for (double t = 0; t < T - 1e-12; t += dt)
            st = dlra.advance(t, dt, st, &stats, &ds);
        for (int i = 0; i < scheme.nodes; ++i)
            mass1 += scheme.lumped[i] * project(st.low.row(i).transpose())[0];
        const double balance = mass1 - mass0 + stats.boundary_outflow + stats.absorbed - stats.injected - stats.nudge_mass;
        const Field full = dlra.expand(st);
        const double e = (full.col(0) - ref.col(0)).cwiseAbs().maxCoeff() / ref.col(0).cwiseAbs().maxCoeff();
        std::cout << "     tol " << tol << ": density error vs full " << e << ", ranks " << ds.min_rank << ".."
                  << ds.max_rank << " (q_H " << dlra.qh << "), balance " << balance << "\n";
        check(std::abs(balance) < 1e-12 * std::max(1., mass0), "crossing DLRA: mass ledger closes (tol " + std::to_string(tol) + ")", balance);
        check(e < previous * 1.05 || e < 1e-6, "crossing DLRA: error does not grow as tol decreases", e);
        previous = e;
        if (tol == 1e-5)
            check(e < 5e-3, "crossing DLRA tol 1e-5: density within 5e-3 of the full scheme", e);
    }
}
// Time-step dependence of the DLRA-vs-full difference at a tight tolerance (BUG projection error).
void crossing_dt_study() {
    Angular pn(7);
    ElementWenoOptions o;
    o.degree = 2, o.dimension = 2, o.elements = 8, o.periodic = false, o.sigma_s = 1;
    const ElementWeno scheme(pn, o, crossing_inflow());
    const double w = .07;
    Field a = scheme.interpolate([&](const V3 &x) {
        auto p = [&](const V3 &d) { return std::exp(-d.squaredNorm() / (2 * w * w)); };
        Vec s = Vec::Zero(pn.q);
        s[0] = 1e-3 / std::sqrt(4 * pi);
        s += p(x - V3(.3, .5, 0)) * harmonics(pn.order, V3::UnitX());
        s += p(x - V3(.5, .3, 0)) * harmonics(pn.order, V3::UnitY());
        return s;
    });
    const double T = 20 * .5 * scheme.step_bound();
    std::vector<double> errors;
    for (int n : {20, 40, 80}) {
        const double dt = T / n;
        Field ref = a;
        DlraOptions d;
        d.tolerance = 1e-6;
        const ElementWenoDlra dlra(scheme, d);
        DlraState st = dlra.compress(a);
        for (int k = 0; k < n; ++k) {
            ref = scheme.advance(k * dt, dt, ref);
            st = dlra.advance(k * dt, dt, st);
        }
        const double e = (dlra.expand(st).col(0) - ref.col(0)).cwiseAbs().maxCoeff() / ref.col(0).cwiseAbs().maxCoeff();
        std::cout << "     steps " << n << ": density difference DLRA(1e-6) vs full " << e << "\n";
        errors.push_back(e);
    }
    check(errors[2] < errors[0], "crossing DLRA: difference to the full scheme decreases with dt",
          std::log2(errors[1] / errors[2]));
}
// The precomputed (fast) S step equals the stage-wise re-evaluation when gamma is constant (WENO off), and differs
// only by the frozen gamma of the l >= 2 Galerkin rate otherwise.
void fast_stage_matches() {
    Angular pn(7);
    for (int weno = 0; weno < 2; ++weno) {
        ElementWenoOptions o;
        o.degree = 2, o.dimension = 2, o.elements = 8, o.periodic = false, o.sigma_s = 1, o.weno = weno;
        const ElementWeno scheme(pn, o, crossing_inflow());
        const double w = .07;
        const Field a = scheme.interpolate([&](const V3 &x) {
            auto p = [&](const V3 &d) { return std::exp(-d.squaredNorm() / (2 * w * w)); };
            Vec s = Vec::Zero(pn.q);
            s[0] = 1e-3 / std::sqrt(4 * pi);
            s += p(x - V3(.3, .5, 0)) * harmonics(pn.order, V3::UnitX());
            s += p(x - V3(.5, .3, 0)) * harmonics(pn.order, V3::UnitY());
            return s;
        });
        const double dt = .5 * scheme.step_bound();
        DlraOptions d;
        d.tolerance = 1e-4;
        d.frozen_gamma = true;
        const ElementWenoDlra fast(scheme, d);
        d.frozen_gamma = false;
        const ElementWenoDlra exact(scheme, d);
        DlraState x = fast.compress(a), y = exact.compress(a);
        for (int k = 0; k < 10; ++k) {
            x = fast.advance(k * dt, dt, x);
            y = exact.advance(k * dt, dt, y);
        }
        const Field fx = fast.expand(x), fy = exact.expand(y);
        const double e = (fx - fy).cwiseAbs().maxCoeff() / fy.cwiseAbs().maxCoeff();
        if (!weno)
            check(e < 1e-11, "fast S step == exact S step for constant gamma (10 steps)", e);
        else
            check(e < 1e-3, "fast S step (frozen gamma) close to exact S step with WENO (10 steps)", e);
    }
}
} // namespace
int main() {
    try {
        factored_rate_matches(false, 2, false);
        factored_rate_matches(true, 2, false);
        factored_rate_matches(false, 2, true);
        factored_rate_matches(false, 1, false);
        react_matches();
        compression();
        crossing_against_full();
        crossing_dt_study();
        fast_stage_matches();
    } catch (const std::exception &e) {
        std::cout << "FAIL exception: " << e.what() << "\n";
        return 1;
    }
    std::cout << (failures ? "FAILED " : "passed ") << failures << "\n";
    return failures ? 1 : 0;
}
