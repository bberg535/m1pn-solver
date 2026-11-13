#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SparseCore>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace m1pn {
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using Field = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Sparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<double>;
using V3 = Eigen::Vector3d;
using Four = Eigen::Vector4d;
constexpr double pi = 3.141592653589793238462643383279502884;
struct StepRejected : std::runtime_error {
    using std::runtime_error::runtime_error;
};
inline void cone_require(bool ok, const std::string &what) {
    if (!ok)
        throw StepRejected(what);
}
inline void require(bool ok, const std::string &what) {
    if (!ok)
        throw std::runtime_error(what);
}
// Literal overloads: no std::string is constructed on the (hot) success path.
inline void cone_require(bool ok, const char *what) {
    if (!ok)
        throw StepRejected(what);
}
inline void require(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}
inline double seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
template <class Derived> inline double maxabs(const Eigen::MatrixBase<Derived> &a) {
    return a.size() ? a.eval().cwiseAbs().maxCoeff() : 0.;
}
inline int index(int l, int m) {
    return l * l + l + m;
}
template <class Derived> inline Four project(const Eigen::MatrixBase<Derived> &u) {
    return Four(std::sqrt(4 * pi) * u[0], std::sqrt(4 * pi / 3) * u[3],
                std::sqrt(4 * pi / 3) * u[1], std::sqrt(4 * pi / 3) * u[2]);
}
inline void set_low(Eigen::Ref<Vec> u, const Four &v) {
    u[0] = v[0] / std::sqrt(4 * pi);
    u[3] = v[1] / std::sqrt(4 * pi / 3);
    u[1] = v[2] / std::sqrt(4 * pi / 3);
    u[2] = v[3] / std::sqrt(4 * pi / 3);
}
inline double margin(const Four &u) {
    // std::hypot(x,y,z) is ~10x slower here; only the scale-safe branch needs it.
    const double s = u[1] * u[1] + u[2] * u[2] + u[3] * u[3];
    if (s > 1e-280 && s < 1e280)
        return u[0] - std::sqrt(s);
    return u[0] - std::hypot(u[1], u[2], u[3]);
}
struct Budget {
    std::size_t bytes = std::size_t(1024) * 1024 * 1024;
    void check(long double count, const std::string &purpose) const {
        require(std::isfinite(count) && count >= 0 && count <= bytes,
                "Memory budget exceeded before allocation: " + purpose);
    }
};
struct Timings {
    double mass = 0, biharmonic = 0, boundary = 0, mcl = 0, communication = 0;
    double pn_predictor = 0, pn_mass_correction = 0;
    // Diagnostic phase totals. Updated only while M1PN_PHASE_LOG=1.
    double phase_check = 0, phase_advection = 0, phase_angular = 0, phase_spatial = 0;
    double phase_closure = 0, phase_gather = 0, phase_finalize = 0, phase_evaluate = 0;
    double phase_fast_angular = 0;
    int phase_angular_mismatch = 0;
};
} // namespace m1pn
