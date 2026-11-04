#pragma once
#include "cone.hpp"

namespace m1pn::lazy_m1 {

// The Levermore Eddington factor is enclosed by rational functions of
// x=|J/rho|^2.  The small outward expansion protects the decision from
// endpoint rounding; the final dyadic cone predicate remains authoritative.
struct ChiInterval { double lo = 0, hi = 1; };

inline ChiInterval chi_interval(const Four &u) {
    if (!(u[0] > 0)) return {};
    const long double rho = u[0];
    long double x = 0;
    for (int k = 1; k < 4; ++k) {
        const long double f = static_cast<long double>(u[k]) / rho;
        x += f * f;
    }
    x = std::max(0.L, std::min(1.L, x));
    const long double upper = std::min(1.L, (3 + 4*x)/(9 - 2*x));
    const long double t1 = 2 - 3*x/4, t2 = 2.5L - 1.5L*x;
    const long double t = std::min(t1,t2);
    const long double lower = std::max({1.L/3, x, (3 + 4*x)/(5 + 2*t)});
    const long double pad = 64 * std::numeric_limits<double>::epsilon();
    return {double(std::max(1.L/3, lower-pad)), double(std::min(1.L, upper+pad))};
}

inline Four pressure_times(const Four &u, double chi, const V3 &c) {
    if (u[0] == 0) return Four::Zero();
    const double jnorm = u.tail<3>().norm();
    if (jnorm == 0) return Four(u.tail<3>().dot(c), u[0]*(1-chi)*c[0]/2,
                                u[0]*(1-chi)*c[1]/2, u[0]*(1-chi)*c[2]/2);
    const V3 e = u.tail<3>() / jnorm;
    const V3 pc = u[0] * ((1-chi)/2*c + ((3*chi-1)/2)*e.dot(c)*e);
    return Four(u.tail<3>().dot(c), pc[0], pc[1], pc[2]);
}

inline bool strictly_interior(const Four &w) {
    if (!exact::cone(w)) return false;
    const double scale = std::max(1., std::abs(w[0]));
    return margin(w) > 64 * std::numeric_limits<double>::epsilon() * scale;
}

// H contains all terms independent of the M1 closure.  Since P(U,chi)c is
// affine in chi, checking both interval endpoints certifies the real closure.
inline bool certify_full_pair(const Four &ui, const Four &uj, const V3 &cij,
                              const V3 &cji, double d, const Four &H) {
    if (!(d > 0)) return false;
    const ChiInterval ai = chi_interval(ui), aj = chi_interval(uj);
    for (double chi_i : {ai.lo, ai.hi}) {
        const Four wi = ui + (H + pressure_times(ui,chi_i,cij-cji))/(2*d);
        if (!strictly_interior(wi)) return false;
    }
    for (double chi_j : {aj.lo, aj.hi}) {
        const Four wj = uj + (-H + pressure_times(uj,chi_j,cji-cij))/(2*d);
        if (!strictly_interior(wj)) return false;
    }
    return true;
}

} // namespace m1pn::lazy_m1
