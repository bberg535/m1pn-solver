#include "angular.hpp"
#include <Eigen/Eigenvalues>
#include <complex>
#include <map>
namespace m1pn {
std::pair<Vec, Vec> gauss(int n) {
    require(n > 0, "positive Gauss order required");
    Vec x(n), w(n);
    for (int i = 0; i < (n + 1) / 2; ++i) {
        double z = std::cos(pi * (i + .75) / (n + .5)), der = 0.;
        bool converged = false;
        auto eval = [&](double a) {
            double p = 1, prev = 0;
            for (int k = 1; k <= n; ++k) {
                double next = ((2 * k - 1) * a * p - (k - 1) * prev) / k;
                prev = p;
                p = next;
            }
            der = n * (a * p - prev) / (a * a - 1);
            return p;
        };
        for (int k = 0; k < 64; ++k) {
            double dz = eval(z) / der;
            z -= dz;
            if (std::abs(dz) < 4e-16) {
                converged = true;
                break;
            }
        }
        require(converged, "Gauss recurrence did not converge");
        eval(z);
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = w[n - 1 - i] = 2 / ((1 - z * z) * der * der);
    }
    return {x, w};
}
Vec harmonics(int n, const V3 &omega) {
    require(std::abs(omega.norm() - 1) < 2e-12, "SH evaluation requires a unit direction");
    Vec y = Vec::Zero((n + 1) * (n + 1));
    const double z = std::clamp(omega[2], -1., 1.), r = std::sqrt(std::max(0., 1 - z * z));
    const double phi = std::atan2(omega[1], omega[0]);
    // Fully normalized associated Legendre recurrence: no factorial overflow.
    double diagonal = 1 / std::sqrt(4 * pi);
    for (int m = 0; m <= n; ++m) {
        if (m)
            diagonal *= r * std::sqrt((2. * m + 1) / (2. * m));
        double prev = 0, p = diagonal;
        for (int l = m; l <= n; ++l) {
            if (l > m) {
                const double a = std::sqrt((4. * l * l - 1) / (double(l) * l - m * m));
                const double b = l == m + 1
                                     ? 0
                                     : std::sqrt((2. * l + 1) * ((l - 1.) * (l - 1.) - m * m) /
                                                 ((2. * l - 3) * (double(l) * l - m * m)));
                const double next = a * z * p - b * prev;
                prev = p;
                p = next;
            }
            if (m == 0)
                y[index(l, 0)] = p;
            else {
                y[index(l, m)] = std::sqrt(2.) * p * std::cos(m * phi);
                y[index(l, -m)] = std::sqrt(2.) * p * std::sin(m * phi);
            }
        }
    }
    require(y.allFinite(), "SH recurrence overflow");
    return y;
}
std::vector<AngularPoint> sphere_rule(int n, int hemisphere) {
    const int polar = n + 2, azimuth = 2 * n + 4;
    const auto [x, w] = gauss(polar);
    std::vector<AngularPoint> points;
    points.reserve(polar * azimuth);
    for (int i = 0; i < polar; ++i)
        for (int j = 0; j < azimuth; ++j) {
            const double z = hemisphere ? (x[i] + hemisphere) * .5 : x[i], r = std::sqrt(1 - z * z),
                         a = 2 * pi * j / azimuth;
            points.push_back({V3(r * std::cos(a), r * std::sin(a), z),
                              w[i] * 2 * pi / azimuth * (hemisphere ? .5 : 1.)});
        }
    return points;
}
Angular::Angular(int n, const Budget &budget) : order(n), q(0) {
    require(n >= 1 && double(n + 1.) * (n + 1.) <= std::numeric_limits<int>::max(),
            "PN order/index overflow");
    q = int((n + 1.) * (n + 1.));
    // Includes temporary canonical C1 matrices and eigensolver workspace.
    budget.check(32.L * q * q * sizeof(double) + 256.L * q, "PN and canonical C1 workspace");
    using Z = std::complex<double>;
    const Z I(0, 1);
    std::vector<std::vector<std::pair<int, Z>>> change(q);
    for (int l = 0; l <= n; ++l) {
        change[index(l, 0)].push_back({index(l, 0), 1.});
        for (int m = 1; m <= l; ++m) {
            int p = index(l, m), k = index(l, -m);
            double sign = m % 2 ? -1 : 1;
            change[p] = {{p, sign / std::sqrt(2.)}, {k, sign / (I * std::sqrt(2.))}};
            change[k] = {{p, 1 / std::sqrt(2.)}, {k, -1. / (I * std::sqrt(2.))}};
        }
    }
    std::array<std::map<std::pair<int, int>, Z>, 3> entries;
    auto add = [&](int d, int i, int j, Z val) {
        for (auto [a, ca] : change[i])
            for (auto [b, cb] : change[j])
                entries[d][{a, b}] += std::conj(ca) * val * cb;
    };
    for (int l = 0; l <= n; ++l)
        for (int m = -l; m <= l; ++m) {
            int k = index(l, m);
            if (l < n) {
                int j = index(l + 1, m);
                double a = std::sqrt(((l + 1.) * (l + 1.) - m * m) / ((2. * l + 1) * (2. * l + 3)));
                add(2, j, k, a);
                add(2, k, j, a);
                double p = -std::sqrt((l + m + 1.) * (l + m + 2.) / ((2. * l + 1) * (2. * l + 3)));
                j = index(l + 1, m + 1);
                add(0, j, k, p / 2);
                add(0, k, j, p / 2);
                add(1, j, k, p / (2. * I));
                add(1, k, j, -p / (2. * I));
            }
            if (l > 0 && std::abs(m + 1) <= l - 1) {
                double p = std::sqrt((l - m) * double(l - m - 1) / ((2. * l - 1) * (2. * l + 1)));
                int j = index(l - 1, m + 1);
                add(0, j, k, p / 2);
                add(0, k, j, p / 2);
                add(1, j, k, p / (2. * I));
                add(1, k, j, -p / (2. * I));
            }
        }
    for (int d = 0; d < 3; ++d) {
        std::vector<Triplet> t;
        for (auto [ij, v] : entries[d]) {
            require(std::abs(v.imag()) < 2e-14, "nonreal SH operator");
            if (std::abs(v.real()) > 1e-15)
                t.emplace_back(ij.first, ij.second, v.real());
        }
        A[d].resize(q, q);
        A[d].setFromTriplets(t.begin(), t.end());
        Sparse defect = A[d] - Sparse(A[d].transpose());
        require(defect.norm() < 1e-13, "asymmetric SH recurrence");
    }
    Mat jac = Mat::Zero(n + 1, n + 1);
    for (int l = 0; l < n; ++l)
        jac(l, l + 1) = jac(l + 1, l) = (l + 1) / std::sqrt((2. * l + 1) * (2. * l + 3));
    Eigen::SelfAdjointEigenSolver<Mat> eig(jac);
    require(eig.info() == Eigen::Success, "PN speed eigensolve failed");
    speed = eig.eigenvalues().maxCoeff();
}
Mat Angular::normal(const V3 &n) const {
    Mat b = Mat::Zero(q, q);
    for (int d = 0; d < 3; ++d)
        b += n[d] * Mat(A[d]);
    return b;
}
Vec Angular::apply(const V3 &n, const Vec &u) const {
    Vec b = Vec::Zero(q);
    for (int d = 0; d < 3; ++d)
        if (n[d] != 0)
            b.noalias() += n[d] * (A[d] * u);
    return b;
}
Mat Angular::rotation(const Eigen::Matrix3d &r) const {
    Mat d = Mat::Zero(q, q);
    require((r.transpose() * r - Eigen::Matrix3d::Identity()).norm() < 1e-12 && r.determinant() > 0,
            "invalid rotation");
    // Rotations preserve degree; exploiting this avoids Q^2 work per sample.
    for (const auto &p : sphere_rule(order)) {
        Vec y = harmonics(order, p.omega), yr = harmonics(order, r * p.omega);
        for (int l = 0; l <= order; ++l)
            d.block(l * l, l * l, 2 * l + 1, 2 * l + 1).noalias() +=
                p.weight * yr.segment(l * l, 2 * l + 1) * y.segment(l * l, 2 * l + 1).transpose();
    }
    require(maxabs(d.transpose() * d - Mat::Identity(q, q)) < 2e-11,
            "SH rotation orthogonality failed");
    return d;
}
} // namespace m1pn
