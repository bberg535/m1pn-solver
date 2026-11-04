#pragma once
#include "cone.hpp"
namespace m1pn {
inline double segment(const Four &b, const Four &d, bool both = false) {
    cone_require(exact::cone(b), "Low-order base bar outside the stored binary64 cone");
    auto valid = [&](double a) {
        Four p = b + a * d, m = b - a * d;
        return exact::cone(p) && (!both || exact::cone(m));
    };
    if (valid(1))
        return 1;
    double lo = 0, hi = 1;
    for (int k = 0; k < 60; ++k) {
        double mid = (lo + hi) / 2;
        if (valid(mid))
            lo = mid;
        else
            hi = mid;
    }
    // Move the common scalar inward, never modify one endpoint independently.
    for (int k = 0; k < 64 && lo > 0; ++k) {
        if (valid(lo))
            return lo;
        lo = std::nextafter(lo, 0.);
    }
    for (int k = 0; k < 60 && lo > 0; ++k) {
        lo *= .5;
        if (valid(lo))
            return lo;
    }
    cone_require(valid(0), "Limiter endpoint rounding check failed");
    return 0;
}

// Density-prioritized conservative correction for a pair of realizable M1
// states. `target` is the requested pair flux p: the updated states are
// L+p and R-p. The result is accepted only when those stored binary64 states
// pass the exact dyadic cone predicate.
struct PairLimitResult {
    Four correction = Four::Zero();
    bool target_pass = false, density_clipped = false;
    bool momentum_limited = false, rounding_fallback = false, scalar_fallback = false;
    double density_retention = 1, momentum_theta = 1;
};
inline PairLimitResult density_pair_limit(const Four &L, const Four &R, const Four &target) {
    cone_require(exact::cone(L) && exact::cone(R), "Invalid pair limiter bases");
    require(target.allFinite(), "Nonfinite pair limiter target");
    PairLimitResult out;
    if (exact::cone(L + target) && exact::cone(R - target)) {
        out.correction = target;
        out.target_pass = true;
        return out;
    }
    const double ah = target[0];
    const double a = std::max(-L[0], std::min(R[0], ah));
    out.density_clipped = a != ah;
    out.density_retention = ah == 0 ? 1 : std::abs(a / ah);
    const double rl = L[0] + a, rr = R[0] - a, S = L[0] + R[0];
    if (!(std::isfinite(S) && S >= 0 && std::isfinite(rl) && std::isfinite(rr))) {
        out.scalar_fallback = true;
    } else if (S == 0) {
        out.correction = Four::Zero();
        out.rounding_fallback = target[1] != 0 || target[2] != 0 || target[3] != 0 || a != 0;
        return out;
    } else {
        V3 v = (L.tail<3>() + R.tail<3>()) / S;
        // This anchor is a mathematical construction; never repair its stored
        // endpoints. If rounding moves either outside the exact cone, use the
        // conservative scalar fallback below.
        Four anchor;
        anchor[0] = a;
        if ((L.array() == R.array()).all()) {
            anchor.tail<3>() = a * (L.tail<3>() / L[0]);
        } else {
            for (int k = 1; k < 4; ++k)
                anchor[k] = static_cast<double>(static_cast<long double>(rl) * v[k - 1] -
                                                static_cast<long double>(L[k]));
        }
        Four anchorL = L + anchor, anchorR = R - anchor;
        if (!v.allFinite() || !exact::cone(anchorL) || !exact::cone(anchorR)) {
            out.rounding_fallback = true;
            out.scalar_fallback = true;
        } else {
            V3 db = target.tail<3>() - anchor.tail<3>();
            const long double A = db.cast<long double>().squaredNorm();
            const long double z = v.cast<long double>().dot(db.cast<long double>());
            long double q = 1 - v.cast<long double>().squaredNorm();
            if (q < 0) { // Computed v is just outside its exact mathematical ball.
                out.rounding_fallback = out.scalar_fallback = true;
            } else {
                long double theta = 1;
                if (A > 0) {
                    if (q == 0) theta = 0;
                    else {
                        const long double D = std::sqrt(z * z + A * q);
                        const long double tl = z >= 0 ? q / (D + z) : (D - z) / A;
                        const long double tr = z <= 0 ? q / (D - z) : (D + z) / A;
                        theta = std::min({1.L, rl * tl, rr * tr});
                    }
                }
                double t = std::max(0., std::min(1., static_cast<double>(theta)));
                out.momentum_theta = t;
                out.momentum_limited = t < 1;
                auto valid = [&](double x) {
                    Four p = anchor;
                    p.tail<3>() += x * db;
                    return exact::cone(L + p) && exact::cone(R - p);
                };
                for (int k = 0; k < 64 && !valid(t) && t > 0; ++k) {
                    t = std::nextafter(t, 0.);
                    out.rounding_fallback = true;
                }
                if (valid(t)) {
                    out.correction = anchor;
                    out.correction.tail<3>() += t * db;
                    out.momentum_theta = t;
                    return out;
                }
                out.rounding_fallback = out.scalar_fallback = true;
            }
        }
    }
    // Guaranteed conservative fallback from the supplied admissible bases.
    auto scalar_valid = [&](double t) {
        return exact::cone(L + t * target) && exact::cone(R - t * target);
    };
    double lo = 0, hi = 1;
    if (scalar_valid(1)) lo = 1;
    else {
        for (int k = 0; k < 60; ++k) {
            double mid = (lo + hi) / 2;
            if (scalar_valid(mid)) lo = mid; else hi = mid;
        }
        for (int k = 0; k < 64 && lo > 0 && !scalar_valid(lo); ++k)
            lo = std::nextafter(lo, 0.);
    }
    double theta = lo;
    out.correction = theta * target;
    out.momentum_theta = theta;
    out.momentum_limited = true;
    out.density_retention = target[0] == 0 ? 1 : std::abs(out.correction[0] / target[0]);
    out.density_clipped = out.correction[0] != target[0];
    cone_require(exact::cone(L + out.correction) && exact::cone(R - out.correction),
                 "Pair limiter scalar fallback failed");
    return out;
}
struct Levermore {
    Four u;
    V3 beta = V3::Zero();
    double gap = 0, K = 0;
    bool vacuum = false, beam = false;
    explicit Levermore(const Four &state) : u(state) {
        cone_require(exact::cone(u), "Levermore state outside stored physical cone");
        vacuum = u[0] == 0;
        if (vacuum)
            return;
        const long double rho = u[0];
        long double j2 = 0;
        for (int d = 1; d < 4; ++d) {
            long double f = static_cast<long double>(u[d]) / rho;
            j2 += f * f;
        }
        long double v = 1 - j2;
        const long double near_beam = 64 * std::numeric_limits<long double>::epsilon();
        if (v <= near_beam)
            v = exact::normalized_gap(u);
        if (exact::squared_gap(u) == 0) {
            beam = true;
            return;
        }
        const double s = std::sqrt(1 + 3 * double(v));
        gap = 12 * double(v) / ((s + 1) * (s + 2));
        beta = (3 / (2 + s)) * u.tail<3>() / u[0];
        K = u[0] * 3 / (4 - gap);
        require(gap > 0 && std::isfinite(K), "Unrepresentable Levermore parameters");
    }
    Four flux(const V3 &n) const {
        Four g = Four::Zero();
        if (vacuum)
            return g;
        g[0] = n.dot(u.tail<3>());
        if (beam)
            g.tail<3>() = (g[0] / u[0]) * u.tail<3>();
        else
            g.tail<3>() = K / 3 * (gap * n + 4 * beta.dot(n) * beta);
        return g;
    }
    // Analytic Levermore hemisphere, adapted from the audited B4 repair.
    // Small-side formula avoids quadrature failure near a beam.
    Four small_outgoing(const V3 &n) const {
        const double bn = beta.dot(n), r = std::hypot(std::sqrt(gap), bn), z = bn / r;
        const double t = bn < 0 ? gap / (r * (r - bn)) : 1 + z, w = r * t * t / 4;
        Four g;
        g[0] = K * w * t * (3 - z) / 3;
        g.tail<3>() = beta * g[0] + (K * w * (2 - z) * gap / (3 * r)) * n;
        require(g.allFinite(), "Nonfinite Levermore hemisphere");
        return g;
    }
    Four outgoing(const V3 &n) const {
        require(std::abs(n.norm() - 1) < 2e-12, "Half-space normal must be unit");
        if (vacuum)
            return Four::Zero();
        if (beam) {
            double mu = n.dot(u.tail<3>()) / u[0];
            return mu > 0 ? Four(mu * u) : Four::Zero();
        }
        if (beta.dot(n) <= 0)
            return small_outgoing(n);
        return flux(n) + small_outgoing(-n);
    }
};
} // namespace m1pn
