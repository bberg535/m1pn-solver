// Exact dyadic cone admission for the ACTUALLY STORED binary64 values.
// No floor, clipping, epsilon cone, or entropy claim. GNU C++17 is required.
#pragma once
#include <cstring>
#include <cfenv>
#include "common.hpp"
namespace m1pn::exact {
using Big = std::array<std::uint64_t, 68>;
using U128 = unsigned __int128;
inline void add_word(Big &a, int p, std::uint64_t v) {
    while (v) {
        require(p < int(a.size()), "dyadic cone accumulator overflow");
        const auto old = a[p];
        a[p] += v;
        v = a[p] < old ? 1 : 0;
        ++p;
    }
}
inline void add_square(Big &a, double v, unsigned factor = 1) {
    std::uint64_t bits;
    std::memcpy(&bits, &v, 8);
    const unsigned be = (bits >> 52) & 2047;
    std::uint64_t m = bits & ((std::uint64_t(1) << 52) - 1);
    if (be)
        m |= std::uint64_t(1) << 52;
    const int e = be ? int(be) - 1075 : -1074;
    U128 product = U128(m) * m * factor;
    const int shift = 2 * (e + 1074), p = shift / 64, b = shift % 64;
    for (int j = 0; j < 2; ++j) {
        const std::uint64_t word = std::uint64_t(product);
        product >>= 64;
        add_word(a, p + j, word << b);
        if (b)
            add_word(a, p + j + 1, word >> (64 - b));
    }
}
inline bool cone_dyadic(const Four &v, unsigned density_factor = 1) {
    for (double x : v)
        if (!std::isfinite(x))
            return false;
    if (v[0] < 0)
        return false;
    Big lhs{}, rhs{};
    add_square(lhs, v[0], density_factor);
    for (int k = 1; k < 4; ++k)
        add_square(rhs, v[k]);
    for (int p = int(lhs.size()) - 1; p >= 0; --p) {
        if (lhs[p] > rhs[p])
            return true;
        if (lhs[p] < rhs[p])
            return false;
    }
    return true;
}
inline bool cone(const Four &v, unsigned density_factor = 1) {
    // A sign filter, not an epsilon cone: ambiguous signs use the unchanged
    // exact integer predicate. For finite binary64 squares, 16*epsilon*sum
    // bounds the roundoff of four products, three differences and the sum;
    // 16*min_normal covers gradual-underflow errors (very conservatively).
    if (density_factor == 1) {
        for (double x : v)
            if (!std::isfinite(x))
                return false;
        if (v[0] < 0)
            return false;
        if (v[0] == 0)
            return v[1] == 0 && v[2] == 0 && v[3] == 0;
        const double r = v[0] * v[0], x = v[1] * v[1], y = v[2] * v[2], z = v[3] * v[3];
        const double sum = ((r + x) + y) + z, difference = ((r - x) - y) - z;
        const double error = 16 * std::numeric_limits<double>::epsilon() * sum +
                             16 * std::numeric_limits<double>::min();
        if (std::isfinite(sum)) {
            if (difference > error)
                return true;
            if (difference < -error)
                return false;
        }
    }
    return cone_dyadic(v, density_factor);
}
// Evaluate rho^2-|J|^2 without subtracting nearly equal floating squares.
// The integer difference is exact for the stored binary64 inputs; only its
// final conversion to extended precision is rounded. No negative gap is clipped.
inline long double squared_gap(const Four &v) {
    static_assert(std::numeric_limits<long double>::max_exponent > 4352 &&
                      std::numeric_limits<long double>::min_exponent < -2148,
                  "Exact squared-gap conversion requires extended exponent range");
    cone_require(cone(v), "Squared gap requires an admissible stored state");
    Big lhs{}, rhs{}, difference{};
    add_square(lhs, v[0]);
    for (int k = 1; k < 4; ++k)
        add_square(rhs, v[k]);
    std::uint64_t borrow = 0;
    for (std::size_t k = 0; k < lhs.size(); ++k) {
        const std::uint64_t word = lhs[k] - rhs[k];
        const bool first = lhs[k] < rhs[k];
        difference[k] = word - borrow;
        borrow = first || word < borrow;
    }
    require(borrow == 0, "Negative exact squared gap");
    long double value = 0;
    for (int k = int(difference.size()) - 1; k >= 0; --k)
        value = std::ldexp(value, 64) + difference[k];
    value = std::ldexp(value, -2148);
    require(std::isfinite(value), "Squared gap exceeds extended precision range");
    return value;
}
inline long double normalized_gap(const Four &v) {
    if (v[0] == 0) {
        cone_require(cone(v), "Invalid vacuum state");
        return 0;
    }
    const long double rho = v[0];
    return squared_gap(v) / (rho * rho);
}
inline void environment() {
    require(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
            "strict contract requires IEEE binary64");
    require(std::fegetround() == FE_TONEAREST, "strict contract requires nearest rounding");
    volatile double tiny = std::numeric_limits<double>::denorm_min(), one = 1.;
    require(tiny * one == tiny && tiny > 0., "strict contract requires gradual underflow");
}
} // namespace m1pn::exact
