#pragma once
// Analytic uncollided densities for the first-collision-source (FCS) split (docs/FCS.md).
//
// psi = psi_u + psi_c with
//   d_t psi_u + Omega.grad psi_u + sigma_t psi_u = 0,                         psi_u(0) = isotropic pulse,
//   d_t psi_c + Omega.grad psi_c + sigma_t psi_c = sigma_s/(4 pi) (rho_u + rho_c),   psi_c(0) = floor,
// (Kusch & Stammer, arXiv:2111.07160, Eq. (7); Hauck & McClarren, MMS 11 (2013)). For a homogeneous infinite
// medium the uncollided density of an isotropic Gaussian pulse is known in closed form or as a 1D integral
// (Bennett & McClarren, arXiv:2205.15783; Garrett & Hauck 2013):
//   plane (1D slab, fields depend on x only):  rho_u = e^{-sigma_t t} / (2t) int_{x-t}^{x+t} rho_0(y) dy,
//   line  (2D, fields depend on x, y):         rho_u = e^{-sigma_t t} int_0^{pi/2} sin(a) ring(r, t sin a) da,
// with ring(r, s) the mean of the normalised 2D Gaussian over the circle of radius s at distance r from its centre.
#include "common.hpp"
#include "parallel.hpp"
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
namespace m1pn {
namespace uncollided {
// exp(-z) I0(z), z >= 0 (series below 30, asymptotic expansion above; relative error < 1e-12).
inline double i0e(double z) {
    if (z < 30) {
        const double q = .25 * z * z;
        double term = 1, sum = 1;
        for (int k = 1; k < 200; ++k) {
            term *= q / (double(k) * k);
            sum += term;
            if (term < 1e-17 * sum)
                break;
        }
        return sum * std::exp(-z);
    }
    // e^{-z} I0(z) ~ (2 pi z)^{-1/2} sum_k ((2k-1)!!)^2 / (k! 8^k z^k)
    double term = 1, sum = 1;
    for (int k = 1; k < 30; ++k) {
        const double next = term * (2. * k - 1) * (2. * k - 1) / (8. * k * z);
        if (next > term)
            break;
        term = next;
        sum += term;
        if (term < 1e-17)
            break;
    }
    return sum / std::sqrt(2 * pi * z);
}
// Gauss-Legendre nodes and weights on [lo, hi].
inline std::pair<std::vector<double>, std::vector<double>> gauss_legendre(int n, double lo, double hi) {
    std::vector<double> x(n), w(n);
    for (int i = 0; i < (n + 1) / 2; ++i) {
        double z = std::cos(pi * (i + .75) / (n + .5)), dp = 0;
        for (int it = 0; it < 100; ++it) {
            double p0 = 1, p1 = z;
            for (int k = 2; k <= n; ++k) {
                const double p2 = ((2. * k - 1) * z * p1 - (k - 1.) * p0) / k;
                p0 = p1, p1 = p2;
            }
            if (n == 1)
                p1 = z, p0 = 1;
            dp = n * (z * p1 - p0) / (z * z - 1);
            const double dz = p1 / dp;
            z -= dz;
            if (std::abs(dz) < 1e-16)
                break;
        }
        const double wi = 2 / ((1 - z * z) * dp * dp);
        x[i] = -z, x[n - 1 - i] = z, w[i] = w[n - 1 - i] = wi;
    }
    for (int i = 0; i < n; ++i)
        x[i] = lo + .5 * (hi - lo) * (x[i] + 1), w[i] *= .5 * (hi - lo);
    return {x, w};
}
// Isotropic Gaussian pulse rho_0(x) = mass * N(x; centre, width^2 I_d), uncollided in a homogeneous infinite
// medium with total cross section sigma_t. dimension 1: plane pulse (slab); dimension 2: line pulse.
struct GaussianPulse {
    int dimension = 2;
    V3 centre = V3::Constant(.5);
    double width = .01, mass = 1, sigma_t = 0;
    // Plane pulse: closed form.
    double plane(double x, double t) const {
        const double y = x - centre[0], s2 = std::sqrt(2.) * width;
        if (t <= 0)
            return mass * std::exp(-y * y / (2 * width * width)) / (std::sqrt(2 * pi) * width);
        return std::exp(-sigma_t * t) * mass / (4 * t) * (std::erf((y + t) / s2) - std::erf((y - t) / s2));
    }
    // ring(r, s) of the normalised 2D Gaussian.
    double ring(double r, double s) const {
        const double w2 = width * width;
        return std::exp(-(r - s) * (r - s) / (2 * w2)) * i0e(r * s / w2) / (2 * pi * w2);
    }
    // Line pulse at radius r: theta integral restricted to the window |t sin(a) - r| <= 10 width, split into
    // `panels` panels of 8 Gauss-Legendre points (the integrand is smooth in a; a panel covers at most 2.5 widths).
    int panels = 6;
    double line(double r, double t) const {
        if (t <= 0)
            return mass * std::exp(-r * r / (2 * width * width)) / (2 * pi * width * width);
        const double lo = std::max(0., r - 10 * width), hi = std::min(t, r + 10 * width);
        if (lo >= hi)
            return 0;
        const double a0 = std::asin(std::min(1., lo / t)), a1 = std::asin(std::min(1., hi / t));
        static const auto rule = gauss_legendre(8, 0, 1);
        double sum = 0;
        for (int k = 0; k < panels; ++k) {
            const double pa = a0 + (a1 - a0) * k / panels, pb = a0 + (a1 - a0) * (k + 1) / panels;
            for (std::size_t q = 0; q < rule.first.size(); ++q) {
                const double a = pa + (pb - pa) * rule.first[q];
                sum += (pb - pa) * rule.second[q] * std::sin(a) * ring(r, t * std::sin(a));
            }
        }
        return std::exp(-sigma_t * t) * mass * sum;
    }
    // Batch evaluation at the points x (physical coordinates). The line pulse is tabulated in r on a grid of
    // width / 16 with cubic (Catmull-Rom) interpolation, clipped at zero (relative error ~1e-5 of the peak), because
    // it is evaluated at every quadrature point in every reaction half step.
    void density(double t, const std::vector<V3> &x, std::vector<double> &out) const {
        out.resize(x.size());
        if (dimension == 1) {
            for (std::size_t k = 0; k < x.size(); ++k)
                out[k] = plane(x[k][0], t);
            return;
        }
        // rho_u vanishes beyond r = t + 10 width (window of line()); the table covers [0, t + 10 width].
        const double dr = width / 8, reach = std::max(t, 0.) + 10 * width;
        const int n = int(std::ceil(reach / dr)) + 3;
        // table[k + 1] = rho_u(k dr); table[0] mirrors table[2] (rho_u is even in r).
        std::vector<double> table(n + 1);
        parallel_for(n, [&](int k) { table[k + 1] = line(k * dr, t); });
        table[0] = table[2];
        const double cx = centre[0], cy = centre[1], inv = 1 / dr;
        parallel_for(int(x.size()), [&](int k) {
            const double dx = x[k][0] - cx, dy = x[k][1] - cy, r = std::sqrt(dx * dx + dy * dy) * inv;
            if (r >= n - 2) {
                out[k] = 0;
                return;
            }
            const int i = int(r);
            const double f = r - i, *p = &table[i]; // p[0..3] = rho_u at (i-1, i, i+1, i+2) dr
            const double v = p[1] + .5 * f * (p[2] - p[0] + f * (2 * p[0] - 5 * p[1] + 4 * p[2] - p[3] +
                                                                f * (3 * (p[1] - p[2]) + p[3] - p[0])));
            out[k] = std::max(v, 0.);
        });
    }
};
} // namespace uncollided
} // namespace m1pn
