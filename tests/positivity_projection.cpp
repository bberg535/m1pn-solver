// Unit tests of the moment-preserving positivity projection (src/positivity_projection.hpp).
// Reference: Dykstra's alternating projections in the W-weighted inner product (independent of the KKT form).
#include "angular.hpp"
#include "positivity_projection.hpp"
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
using namespace m1pn;
namespace {
int failures = 0, checks = 0;
void check(bool ok, const std::string &what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << "\n";
    }
}
struct Rule {
    Vec ox, oy, oz, w;
    Mat harm; // points x q
};
Rule full_rule(int order, int n) {
    auto points = sphere_rule(n, +1);
    const auto lower = sphere_rule(n, -1);
    points.insert(points.end(), lower.begin(), lower.end());
    Rule r;
    const int q = int(points.size());
    r.ox.resize(q), r.oy.resize(q), r.oz.resize(q), r.w.resize(q);
    r.harm.resize(q, (order + 1) * (order + 1));
    for (int k = 0; k < q; ++k) {
        r.ox[k] = points[k].omega[0], r.oy[k] = points[k].omega[1], r.oz[k] = points[k].omega[2];
        r.w[k] = points[k].weight;
        r.harm.row(k) = harmonics(order, points[k].omega).transpose();
    }
    return r;
}
// PN coefficients of f by the rule (exact for polynomial f of degree <= N).
Vec coefficients(const Rule &r, const std::function<double(const V3 &)> &f) {
    Vec c = Vec::Zero(r.harm.cols());
    for (int k = 0; k < r.w.size(); ++k)
        c += r.w[k] * f(V3(r.ox[k], r.oy[k], r.oz[k])) * r.harm.row(k).transpose();
    return c;
}
Four moments(const Rule &r, const Vec &z) {
    Four m = Four::Zero();
    for (int k = 0; k < z.size(); ++k)
        m += r.w[k] * z[k] * Four(1, r.ox[k], r.oy[k], r.oz[k]);
    return m;
}
// Dykstra: W-orthogonal projection onto {B z = U} and onto {z >= 0}, alternating with corrections.
Vec dykstra(const Rule &r, const Vec &y, const Four &U, int iterations) {
    const int q = y.size();
    Mat M(4, q); // rows m^T
    for (int k = 0; k < q; ++k)
        M.col(k) = Four(1, r.ox[k], r.oy[k], r.oz[k]);
    // Affine projection in the W inner product: z - M^T (M W M^T)^-1 (M W z - U).
    Mat G = M * r.w.asDiagonal() * M.transpose();
    Eigen::LDLT<Mat> ldlt(G);
    Vec x = y, p = Vec::Zero(q), qv = Vec::Zero(q);
    for (int it = 0; it < iterations; ++it) {
        Vec a = x + p;
        Vec ya = a - M.transpose() * ldlt.solve(M * (r.w.asDiagonal() * a) - U);
        p = a - ya;
        Vec b = ya + qv;
        x = b.cwiseMax(0.);
        qv = b - x;
    }
    return x;
}
double objective(const Rule &r, const Vec &z, const Vec &y) {
    return .5 * (r.w.array() * (z - y).array().square()).sum();
}
} // namespace

int main() {
    std::cout << std::setprecision(6);
    for (int order : {3, 7}) {
        const Rule r = full_rule(order, 2 * order + 1);
        const int q = r.w.size();
        auto run = [&](const Vec &c, PositivityProjectionResult &res) {
            Vec y = r.harm * c;
            const Four U(std::sqrt(4 * pi) * c[0], std::sqrt(4 * pi / 3) * c[3], std::sqrt(4 * pi / 3) * c[1],
                         std::sqrt(4 * pi / 3) * c[2]);
            Vec z(q);
            res = project_positive(q, r.ox.data(), r.oy.data(), r.oz.data(), r.w.data(), y.data(), U, z.data());
            return std::make_tuple(y, U, z);
        };
        const std::string tag = "P" + std::to_string(order) + ": ";
        // 1. Identity for a nonnegative polynomial trace.
        {
            PositivityProjectionResult res;
            auto [y, U, z] = run(coefficients(r, [](const V3 &w) { return 1 + .5 * w[0] + .3 * w[1] * w[2]; }), res);
            check(res.converged && res.identity && res.iterations == 0, tag + "identity flagged");
            check((z - y).cwiseAbs().maxCoeff() == 0, tag + "identity: z == y exactly");
            check((moments(r, y) - U).cwiseAbs().maxCoeff() < 1e-13, tag + "rule reproduces the PN moments");
        }
        // 2. Truncated exp(3 w_x) and exp(-4 (1 - w.e)) have negative lobes: compare with Dykstra.
        for (double kappa : {3., 8.}) {
            const V3 e = V3(1, 2, -1).normalized();
            PositivityProjectionResult res;
            auto [y, U, z] = run(coefficients(r, [&](const V3 &w) { return std::exp(kappa * (w.dot(e) - 1)); }), res);
            const std::string t = tag + "kappa " + std::to_string(int(kappa)) + ": ";
            if (kappa > 3)
                check(y.minCoeff() < 0, t + "PN trace has negative lobes");
            check(res.converged, t + "converged");
            check(z.minCoeff() >= 0, t + "z >= 0");
            check((moments(r, z) - U).cwiseAbs().maxCoeff() <= 1e-11 * U[0], t + "moments preserved");
            if (res.converged) {
                const Vec ref = dykstra(r, y, U, 20000);
                const double gap = objective(r, z, y) - objective(r, ref, y);
                const double dist = std::sqrt((r.w.array() * (z - ref).array().square()).sum() / r.w.sum());
                std::cout << t << "iterations " << res.iterations << " backtracks " << res.backtracks << " active "
                          << res.active << "/" << q << " objective gap to Dykstra " << gap << " W-distance " << dist
                          << " (Dykstra moment residual " << (moments(r, ref) - U).cwiseAbs().maxCoeff() << ")\n";
                check(gap <= 1e-9 * objective(r, z, y), t + "objective not above the Dykstra reference");
                check(dist <= 1e-5 * std::sqrt((r.w.array() * y.array().square()).sum() / r.w.sum()),
                      t + "solution agrees with Dykstra");
            }
        }
        // 3. Beam + background: |J|/rho -> 1. Feasible iff U is in the discrete cone of the rule.
        for (double bg : {1e-1, 1e-2, 1e-3, 1e-4}) {
            for (const V3 &e : {V3(1, 0, 0), V3(0, 0, 1), V3(1, 1, 0).normalized()}) {
                Vec c = harmonics(order, e);
                c[0] += bg * std::sqrt(4 * pi);
                PositivityProjectionResult res;
                auto [y, U, z] = run(c, res);
                Vec za(q);
                const auto o = project_positive_moments(q, r.ox.data(), r.oy.data(), r.oz.data(), r.w.data(),
                                                        y.data(), U, za.data());
                const Four total = moments(r, za) + o.atom;
                const std::string t = tag + "beam bg " + std::to_string(bg) + ": ";
                check(o.result.converged, t + "wrapper converged");
                check(za.minCoeff() >= 0 && o.atom[0] >= 0, t + "wrapper z >= 0");
                check((total - U).cwiseAbs().maxCoeff() <= 1e-11 * U[0], t + "wrapper moments preserved");
                if (res.converged)
                    check(!o.augmented && (za - z).cwiseAbs().maxCoeff() == 0, t + "wrapper = plain when feasible");
                double best = 0; // max_k Omega_k . e: the discrete cone contains (1, s e) only for s <= ~best
                for (int k = 0; k < q; ++k)
                    best = std::max(best, V3(r.ox[k], r.oy[k], r.oz[k]).dot(e));
                std::printf("%sbeam bg %.0e e=(%.2f,%.2f,%.2f): |J|/rho %.8f, max Omega.e %.8f, converged %d, it %d, "
                            "residual %.2e | wrapper: certificate %d augmented %d atom mass %.3e it %d\n",
                            tag.c_str(), bg, e[0], e[1], e[2], U.tail<3>().norm() / U[0], best, res.converged,
                            res.iterations, res.residual, o.certificate, o.augmented, o.atom[0] / U[0], o.newton_iterations);
                // The certificate is sufficient for infeasibility: it may never fire on a solvable problem.
                check(!(o.certificate && res.converged), t + "certificate only on infeasible problems");
            }
        }
    }
    // Hull facets: the exact solvability test agrees with Newton on random strongly anisotropic states.
    for (int order : {3, 7}) {
        const int n = 2 * order + 1;
        const Rule r = full_rule(order, n);
        const int q = r.w.size();
        std::vector<double> rings;
        for (int h : {+1, -1}) {
            const auto local = sphere_rule(n, h);
            for (std::size_t k = 0; k < local.size(); k += 2 * n + 4)
                rings.push_back(local[k].omega[2]);
        }
        const Eigen::Matrix3d frame = Eigen::AngleAxisd(.7, V3(1, 2, 3).normalized()).toRotationMatrix();
        const PositivityFacets facets = lat_long_facets(rings, 2 * n + 4, frame);
        check(facets.count() == int(rings.size() - 1) * (2 * n + 4) + 2, "P" + std::to_string(order) + ": hull facets built");
        Rule rr = r; // rotated rule
        for (int k = 0; k < q; ++k) {
            const V3 o = frame * V3(r.ox[k], r.oy[k], r.oz[k]);
            rr.ox[k] = o[0], rr.oy[k] = o[1], rr.oz[k] = o[2];
        }
        std::mt19937 gen(11);
        std::uniform_real_distribution<double> uni(-1, 1);
        int agree = 0, total = 0, inside = 0;
        for (int trial = 0; trial < 400; ++trial) {
            V3 e(uni(gen), uni(gen), uni(gen));
            e.normalize();
            const double s = 1 - std::pow(10., -1 - 4 * (uni(gen) + 1) / 2); // |J|/rho in (0.9, 0.99999)
            const Four U(1, s * e[0], s * e[1], s * e[2]);
            Vec y = Vec::Constant(q, .05), z(q);
            const double m = facets.margin(V3(U.tail<3>()));
            PositivityProjectionOptions opt;
            opt.max_iterations = 200;
            const auto res = project_positive(q, rr.ox.data(), rr.oy.data(), rr.oz.data(), rr.w.data(), y.data(), U,
                                              z.data(), opt);
            if (std::abs(m) < 1e-6)
                continue; // too close to the boundary for a clean Newton verdict
            ++total;
            inside += m < 0;
            agree += (m < 0) == res.converged;
        }
        std::cout << "P" << order << ": hull test vs Newton: " << agree << "/" << total << " agree (" << inside
                  << " inside)\n";
        check(agree == total && inside > 0 && inside < total, "P" + std::to_string(order) + ": hull test exact");
    }
    // Cone boundary (pure beam) and outside R1.
    {
        const Rule r = full_rule(3, 7);
        const int q = r.w.size();
        Vec y = Vec::Zero(q), z(q);
        const auto o = project_positive_moments(q, r.ox.data(), r.oy.data(), r.oz.data(), r.w.data(), y.data(),
                                                Four(2, 0, 2, 0), z.data());
        check(o.result.converged && o.dirac && (o.atom - Four(2, 0, 2, 0)).norm() < 1e-15, "cone boundary: Dirac atom");
        const auto bad = project_positive_moments(q, r.ox.data(), r.oy.data(), r.oz.data(), r.w.data(), y.data(),
                                                  Four(1, 0, 1.01, 0), z.data());
        check(!bad.result.converged, "outside R1: failure, no clipping");
        const auto vac = project_positive_moments(q, r.ox.data(), r.oy.data(), r.oz.data(), r.w.data(), y.data(),
                                                  Four::Zero(), z.data());
        check(vac.result.converged && z.isZero(0), "vacuum: z = 0");
    }
    std::cout << checks - failures << "/" << checks << " checks passed\n";
    return failures ? 1 : 0;
}
