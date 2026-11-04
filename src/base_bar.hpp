#pragma once
#include "cone.hpp"
namespace m1pn {
// Algebraically identical reevaluation of the M1 interior base bar. This is
// used when the ordinary flux subtraction fails exact stored-cone admission.
// It neither changes either input state nor raises the bar density by a floor.
inline Four stable_base_bar(const Four &left, const Four &right, const V3 &edge, double d) {
    cone_require(exact::cone(left) && exact::cone(right), "Invalid base-bar input state");
    const Four speed(d, edge[0], edge[1], edge[2]);
    cone_require(d > 0 && exact::cone(speed), "Base-bar graph speed below physical speed");
    using L3 = Eigen::Matrix<long double, 3, 1>;
    using L4 = Eigen::Matrix<long double, 4, 1>;
    const L3 t = edge.cast<long double>() / static_cast<long double>(d);
    const long double qt = exact::normalized_gap(speed);
    auto split = [&](const Four &u, int sign) -> L4 {
        if (u[0] == 0)
            return L4::Zero();
        const long double rho = u[0], q = exact::normalized_gap(u);
        const L3 w = u.tail<3>().cast<long double>() / rho;
        const long double mu = w.dot(t), s = std::sqrt(1 + 3 * q);
        const long double c = q / (1 + s), k = 3 / (2 + s);
        // 1 +/- w.t = (q + qt + |w +/- t|^2)/2. All three
        // summands are nonnegative, including near the vanishing split.
        const long double tau = (q + qt + (w + sign * t).squaredNorm()) / 2;
        L4 value;
        value[0] = rho * tau;
        value.tail<3>() = rho * (tau * w + sign * c * (t - k * mu * w));
        return value;
    };
    const Four bar = ((split(left, 1) + split(right, -1)) / 2).cast<double>();
    cone_require(exact::cone(bar), "Stable M1 base bar outside stored binary64 cone");
    return bar;
}
} // namespace m1pn
