// Scalability benchmark of the boundary positivity projection (docs/POSITIVITY_PROJECTION.md), independent of the
// solver: per boundary node cost of the PN trace (dense / separable), the C6 positive part and the projection
// (cold / warm), Newton iterations, solvability (augmented share), and the quadrature convergence of the projected
// outgoing moments. One JSON object per (N, rule, state class) on stdout.
//
// Usage: m1pn_positivity_projection_bench [--orders 3,7,15] [--states 2000] [--repeat 3] [--seed 1]
#include "angular.hpp"
#include "positivity_projection.hpp"
#include "separable_trace.hpp"
#include <cstdio>
#include <functional>
#include <memory>
#include <iostream>
#include <random>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace m1pn;
namespace {
struct FullRule {
    int n = 0, q = 0, outgoing = 0;
    Vec ox, oy, oz, w;
    Mat harm; // q x (N+1)^2 (dense, global frame)
    std::unique_ptr<SeparableTrace> separable;
    PositivityFacets facets;
};
FullRule make_rule(int order, int n, const Eigen::Matrix3d &frame) {
    FullRule r;
    r.n = n;
    auto pts = sphere_rule(n, +1);
    r.outgoing = int(pts.size());
    const auto lower = sphere_rule(n, -1);
    pts.insert(pts.end(), lower.begin(), lower.end());
    r.q = int(pts.size());
    r.ox.resize(r.q), r.oy.resize(r.q), r.oz.resize(r.q), r.w.resize(r.q);
    const int qf = (order + 1) * (order + 1);
    r.harm.resize(r.q, qf);
    for (int k = 0; k < r.q; ++k) {
        const V3 o = frame * pts[k].omega;
        r.ox[k] = o[0], r.oy[k] = o[1], r.oz[k] = o[2], r.w[k] = pts[k].weight;
        r.harm.row(k) = harmonics(order, o).transpose();
    }
    std::vector<int> active(qf);
    for (int c = 0; c < qf; ++c)
        active[c] = c;
    r.separable = std::make_unique<SeparableTrace>(order, active, frame, n, std::vector<int>{+1, -1});
    std::vector<double> rings;
    for (int h : {+1, -1}) {
        const auto local = sphere_rule(n, h);
        for (std::size_t k = 0; k < local.size(); k += 2 * n + 4)
            rings.push_back(local[k].omega[2]);
    }
    r.facets = lat_long_facets(rings, 2 * n + 4, frame);
    return r;
}
// PN coefficients of a nonnegative angular density: background + optional smooth lobe + optional beams.
struct StateGen {
    int order;
    std::mt19937 gen;
    Vec smooth_lobe(const V3 &e, double kappa) {
        // Projection of exp(kappa (w.e - 1)) with a fine rule (exact up to the truncation of the exponential).
        static thread_local std::vector<AngularPoint> fine;
        if (fine.empty())
            fine = sphere_rule(2 * order + 40);
        Vec c = Vec::Zero((order + 1) * (order + 1));
        for (const auto &p : fine)
            c += p.weight * std::exp(kappa * (p.omega.dot(e) - 1)) * harmonics(order, p.omega);
        return c;
    }
    V3 direction() {
        std::normal_distribution<double> n;
        V3 v(n(gen), n(gen), n(gen));
        return v.normalized();
    }
    Vec make(const std::string &cls) {
        std::uniform_real_distribution<double> u(0, 1);
        Vec c = Vec::Zero((order + 1) * (order + 1));
        if (cls == "smooth") { // positive, mildly anisotropic: identity expected
            c[0] = std::sqrt(4 * pi) * .3;
            c += smooth_lobe(direction(), 4 * u(gen));
        } else if (cls == "peaked") { // strongly forward peaked, negative PN lobes
            c[0] = std::sqrt(4 * pi) * std::pow(10., -3 + 2 * u(gen));
            c += smooth_lobe(direction(), 5 + 35 * u(gen));
        } else { // beams on a small background (crossing beams)
            c[0] = std::sqrt(4 * pi) * std::pow(10., -4 + 3 * u(gen));
            const int beams = 1 + int(2 * u(gen));
            for (int b = 0; b < beams; ++b)
                c += (.2 + u(gen)) * harmonics(order, direction());
        }
        return c;
    }
};
Four low(const Vec &c) {
    return Four(std::sqrt(4 * pi) * c[0], std::sqrt(4 * pi / 3) * c[3], std::sqrt(4 * pi / 3) * c[1],
                std::sqrt(4 * pi / 3) * c[2]);
}
Four outgoing_moments(const FullRule &r, const double *z, const Four &atom, const V3 &atom_dir, const V3 &normal) {
    Four m = Four::Zero();
    for (int k = 0; k < r.outgoing; ++k)
        m += r.w[k] * z[k] * Four(1, r.ox[k], r.oy[k], r.oz[k]);
    if (atom[0] > 0 && atom_dir.dot(normal) > 0)
        m += atom;
    return m;
}
std::vector<int> parse_list(const std::string &s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        out.push_back(std::stoi(item));
    return out;
}
} // namespace

int main(int argc, char **argv) {
#ifdef _OPENMP
    omp_set_num_threads(1); // per-node kernel cost, single thread
#endif
    std::vector<int> orders{3, 5, 7, 11, 15, 21, 31};
    int states = 2000, repeat = 3;
    unsigned seed = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (key == "--orders") orders = parse_list(next());
        else if (key == "--states") states = std::stoi(next());
        else if (key == "--repeat") repeat = std::stoi(next());
        else if (key == "--seed") seed = unsigned(std::stoul(next()));
        else { std::cerr << "unknown option " << key << "\n"; return 2; }
    }
    // A face with normal +x: local pole -> +x.
    Eigen::Matrix3d frame;
    frame << 0, 0, 1, 0, 1, 0, -1, 0, 0;
    const V3 normal = frame.col(2);
    for (int N : orders) {
        const int qf = (N + 1) * (N + 1);
        // Reference rule for the quadrature convergence of the projected outgoing moments.
        const FullRule fine = make_rule(N, 4 * N + 7, frame);
        for (int n : {N + 1, 2 * N + 1}) {
            const FullRule r = make_rule(N, n, frame);
            for (const std::string cls : {"smooth", "peaked", "beam"}) {
                StateGen sg{N, std::mt19937(seed + 977 * N + (cls == "peaked") * 13 + (cls == "beam") * 29)};
                Mat C(qf, states);
                for (int s = 0; s < states; ++s)
                    C.col(s) = sg.make(cls);
                // Traces.
                Mat dense, sep;
                double t_dense = 1e300, t_sep = 1e300;
                for (int rep = 0; rep < repeat; ++rep) {
                    double t0 = seconds();
                    dense.noalias() = r.harm * C;
                    t_dense = std::min(t_dense, seconds() - t0);
                    t0 = seconds();
                    Mat block;
                    sep.resize(r.q, states);
                    for (int s = 0; s < states; s += 32) {
                        const int k = std::min(32, states - s);
                        r.separable->evaluate(C.middleCols(s, k), block);
                        sep.middleCols(s, k) = block;
                    }
                    t_sep = std::min(t_sep, seconds() - t0);
                }
                const double trace_diff = (dense - sep).cwiseAbs().maxCoeff() / dense.cwiseAbs().maxCoeff();
                // C6 positive part (outgoing hemisphere only).
                double t_c6 = 1e300;
                std::vector<Four> c6(states);
                for (int rep = 0; rep < repeat; ++rep) {
                    const double t0 = seconds();
                    for (int s = 0; s < states; ++s) {
                        Four m = Four::Zero();
                        const double *y = sep.col(s).data();
                        for (int k = 0; k < r.outgoing; ++k) {
                            const double v = std::max(0., y[k]) * r.w[k];
                            m[0] += v, m[1] += v * r.ox[k], m[2] += v * r.oy[k], m[3] += v * r.oz[k];
                        }
                        c6[s] = m;
                    }
                    t_c6 = std::min(t_c6, seconds() - t0);
                }
                // Projection, cold.
                Vec z(r.q);
                PositivityProjectionOptions opt;
                std::vector<PositivityProjectionWarm> warm(states);
                std::vector<Four> outm(states);
                long it_cold = 0, aug = 0, ident = 0, fail = 0;
                int it_max = 0;
                double res_max = 0, t_cold = 1e300;
                for (int rep = 0; rep < repeat; ++rep) {
                    it_cold = aug = ident = fail = 0, it_max = 0, res_max = 0;
                    const double t0 = seconds();
                    for (int s = 0; s < states; ++s) {
                        const Four U = low(C.col(s));
                        const auto o = project_positive_moments(r.q, r.ox.data(), r.oy.data(), r.oz.data(),
                                                                r.w.data(), sep.col(s).data(), U, z.data(), opt,
                                                                nullptr, &r.facets);
                        it_cold += o.newton_iterations, it_max = std::max(it_max, o.newton_iterations);
                        aug += o.augmented, ident += o.result.identity, fail += !o.result.converged;
                        res_max = std::max(res_max, o.result.residual);
                        if (o.result.converged) {
                            warm[s].xi = o.augmented ? warm[s].xi : o.result.xi;
                            warm[s].xi_augmented = o.augmented ? o.result.xi : warm[s].xi_augmented;
                        }
                        outm[s] = outgoing_moments(r, z.data(), o.atom, o.atom_direction, normal);
                    }
                    t_cold = std::min(t_cold, seconds() - t0);
                }
                // Projection, warm: a 1% perturbation of each state (one time step later), started from xi above.
                std::normal_distribution<double> nd;
                std::mt19937 pg(seed + 5);
                Mat Cp = C;
                for (int s = 0; s < states; ++s)
                    Cp.col(s) += .01 * C.col(s).norm() / std::sqrt(double(qf)) *
                                 Vec::NullaryExpr(qf, [&]() { return nd(pg); });
                Mat sepp(r.q, states), block;
                for (int s = 0; s < states; s += 32) {
                    const int k = std::min(32, states - s);
                    r.separable->evaluate(Cp.middleCols(s, k), block);
                    sepp.middleCols(s, k) = block;
                }
                long it_warm = 0, it_cold_p = 0;
                double t_warm = 1e300, t_cold_p = 1e300;
                for (int rep = 0; rep < repeat; ++rep) {
                    it_warm = 0;
                    double t0 = seconds();
                    for (int s = 0; s < states; ++s) {
                        const Four U = low(Cp.col(s));
                        if (U[0] < U.tail<3>().norm())
                            continue;
                        const auto o = project_positive_moments(r.q, r.ox.data(), r.oy.data(), r.oz.data(),
                                                                r.w.data(), sepp.col(s).data(), U, z.data(), opt,
                                                                &warm[s], &r.facets);
                        it_warm += o.newton_iterations;
                    }
                    t_warm = std::min(t_warm, seconds() - t0);
                    it_cold_p = 0;
                    t0 = seconds();
                    for (int s = 0; s < states; ++s) {
                        const Four U = low(Cp.col(s));
                        if (U[0] < U.tail<3>().norm())
                            continue;
                        const auto o = project_positive_moments(r.q, r.ox.data(), r.oy.data(), r.oz.data(),
                                                                r.w.data(), sepp.col(s).data(), U, z.data(), opt,
                                                                nullptr, &r.facets);
                        it_cold_p += o.newton_iterations;
                    }
                    t_cold_p = std::min(t_cold_p, seconds() - t0);
                }
                // Quadrature convergence: projected outgoing moments vs the fine rule; C6 vs projection difference.
                double conv = 0, c6_vs_proj = 0, c6_conv = 0;
                {
                    Vec zf(fine.q), yf(fine.q);
                    const int sample = std::min(states, 300);
                    for (int s = 0; s < sample; ++s) {
                        const Four U = low(C.col(s));
                        yf = fine.harm * C.col(s);
                        const auto o = project_positive_moments(fine.q, fine.ox.data(), fine.oy.data(), fine.oz.data(),
                                                                fine.w.data(), yf.data(), U, zf.data(), opt, nullptr,
                                                                &fine.facets);
                        if (!o.result.converged)
                            continue;
                        const Four mf = outgoing_moments(fine, zf.data(), o.atom, o.atom_direction, normal);
                        conv = std::max(conv, (outm[s] - mf).cwiseAbs().maxCoeff() / U[0]);
                        c6_vs_proj = std::max(c6_vs_proj, (outm[s] - c6[s]).cwiseAbs().maxCoeff() / U[0]);
                        // C6 on the fine rule (its own quadrature convergence).
                        Four m = Four::Zero();
                        for (int k = 0; k < fine.outgoing; ++k) {
                            const double v = std::max(0., yf[k]) * fine.w[k];
                            m += v * Four(1, fine.ox[k], fine.oy[k], fine.oz[k]);
                        }
                        c6_conv = std::max(c6_conv, (c6[s] - m).cwiseAbs().maxCoeff() / U[0]);
                    }
                }
                const double us = 1e6 / states;
                std::printf("{\"N\":%d,\"q\":%d,\"rule\":%d,\"rule_label\":\"%s\",\"points\":%d,\"class\":\"%s\","
                            "\"states\":%d,\"us_trace_dense\":%.4g,\"us_trace_separable\":%.4g,"
                            "\"trace_rel_diff\":%.3g,\"us_c6\":%.4g,\"us_projection_cold\":%.4g,"
                            "\"us_projection_warm\":%.4g,\"us_projection_cold_perturbed\":%.4g,"
                            "\"iterations_cold_mean\":%.4g,\"iterations_cold_max\":%d,\"iterations_warm_mean\":%.4g,"
                            "\"iterations_cold_perturbed_mean\":%.4g,\"identity_share\":%.4g,\"augmented_share\":%.4g,"
                            "\"failures\":%ld,\"max_residual\":%.3g,\"max_outgoing_error_vs_fine_rule\":%.3g,"
                            "\"max_c6_error_vs_fine_rule\":%.3g,\"max_c6_minus_projection\":%.3g}\n",
                            N, qf, n, n == N + 1 ? "N+1" : "2N+1", r.q, cls.c_str(), states, t_dense * us,
                            t_sep * us, trace_diff, t_c6 * us, t_cold * us, t_warm * us, t_cold_p * us,
                            double(it_cold) / states, it_max, double(it_warm) / states, double(it_cold_p) / states,
                            double(ident) / states, double(aug) / states, fail, res_max, conv, c6_conv, c6_vs_proj);
                std::fflush(stdout);
            }
        }
    }
    return 0;
}
