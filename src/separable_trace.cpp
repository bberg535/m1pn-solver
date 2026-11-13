#include "separable_trace.hpp"
#include "angular.hpp"
#include <random>

namespace m1pn {

SeparableTrace::SeparableTrace(int order, const std::vector<int> &active, const Eigen::Matrix3d &frame, int n,
                               const std::vector<int> &hemispheres)
    : N_(order) {
    const int qf = (N_ + 1) * (N_ + 1);
    A_ = 2 * n + 4;
    require(N_ < A_ / 2, "SeparableTrace: azimuthal rule too coarse for the PN order");
    require(std::is_sorted(active.begin(), active.end()), "SeparableTrace: active indices must be sorted");
    // Local points of all hemispheres in rule order.
    std::vector<AngularPoint> local;
    for (int h : hemispheres) {
        const auto pts = sphere_rule(n, h);
        local.insert(local.end(), pts.begin(), pts.end());
    }
    P_ = int(local.size()) / A_;
    require(P_ * A_ == int(local.size()), "SeparableTrace: rule is not a tensor product");
    // Degree blocks of the active list.
    block_start_.assign(N_ + 1, 0);
    block_count_.assign(N_ + 1, 0);
    for (std::size_t c = 0; c < active.size(); ++c) {
        const int l = int(std::sqrt(double(active[c]) + .5));
        if (block_count_[l]++ == 0)
            block_start_[l] = int(c);
    }
    // Azimuthal basis T_m(phi_j), phi_j = 2 pi j / A (as in sphere_rule).
    trig_.resize(A_, 2 * N_ + 1);
    for (int j = 0; j < A_; ++j) {
        const double phi = 2 * pi * j / A_;
        for (int m = -N_; m <= N_; ++m)
            trig_(j, m + N_) = m == 0 ? 1. : m > 0 ? std::cos(m * phi) : std::sin(-m * phi);
    }
    // Theta(i, (l, m)) by discrete azimuthal projection (exact: trig degree <= 2N < A).
    Mat theta_full(P_, qf);
    for (int i = 0; i < P_; ++i) {
        Vec acc = Vec::Zero(qf);
        for (int j = 0; j < A_; ++j) {
            const Vec y = harmonics(N_, local[std::size_t(i) * A_ + j].omega);
            for (int l = 0; l <= N_; ++l)
                for (int m = -l; m <= l; ++m)
                    acc[index(l, m)] += y[index(l, m)] * trig_(j, m + N_) * (m == 0 ? 1. : 2.) / A_;
        }
        theta_full.row(i) = acc.transpose();
    }
    theta_.resize(2 * N_ + 1);
    for (int m = -N_; m <= N_; ++m) {
        const int am = std::abs(m);
        Mat &t = theta_[m + N_];
        t.resize(P_, N_ + 1 - am);
        for (int l = am; l <= N_; ++l)
            t.col(l - am) = theta_full.col(index(l, m));
    }
    // Wigner blocks D_(c, c') = int Y_c(F w) Y_c'(w) dw (exact for degree 2N with sphere_rule(N + 1)).
    Mat D = Mat::Zero(qf, qf);
    for (const auto &p : sphere_rule(N_ + 1)) {
        const Vec yg = harmonics(N_, frame * p.omega), yl = harmonics(N_, p.omega);
        for (int l = 0; l <= N_; ++l) {
            const int s = l * l, w = 2 * l + 1;
            D.block(s, s, w, w).noalias() += p.weight * yg.segment(s, w) * yl.segment(s, w).transpose();
        }
    }
    dt_.resize(N_ + 1);
    for (int l = 0; l <= N_; ++l) {
        dt_[l].resize(2 * l + 1, block_count_[l]);
        for (int c = 0; c < block_count_[l]; ++c) {
            const int g = active[block_start_[l] + c];
            dt_[l].col(c) = D.block(g, l * l, 1, 2 * l + 1).transpose();
        }
    }
    // Verification against the dense evaluation at random coefficients.
    const int K = 3;
    std::mt19937 gen(7);
    std::normal_distribution<double> normal;
    Mat at(active.size(), K);
    for (int c = 0; c < at.size(); ++c)
        at.data()[c] = normal(gen);
    Mat fast;
    evaluate(at, fast);
    double err = 0, size = 0;
    for (int k = 0; k < points(); ++k) {
        const Vec y = harmonics(N_, frame * local[k].omega);
        for (int col = 0; col < K; ++col) {
            double dense = 0;
            for (std::size_t c = 0; c < active.size(); ++c)
                dense += y[active[c]] * at(c, col);
            err = std::max(err, std::abs(dense - fast(k, col)));
            size = std::max(size, std::abs(dense));
        }
    }
    verified_error = err / std::max(size, 1e-300);
    require(verified_error < 1e-12, "SeparableTrace: factorization does not reproduce the dense traces (" +
                                        std::to_string(verified_error) + ")");
}

void SeparableTrace::evaluate(const Mat &at, Mat &out) const {
    const int K = int(at.cols());
    // b = D^T a in the local full basis, row-major (row index(l, m') contiguous over the K nodes).
    thread_local Field b, C;
    thread_local Mat R;
    b.resize((N_ + 1) * (N_ + 1), K);
    for (int l = 0; l <= N_; ++l) {
        const int w = 2 * l + 1, cnt = block_count_[l], s0 = block_start_[l];
        for (int r = 0; r < w; ++r) {
            double *row = b.row(l * l + r).data();
            for (int k = 0; k < K; ++k)
                row[k] = 0;
            for (int c = 0; c < cnt; ++c) {
                const double d = dt_[l](r, c);
                if (d == 0)
                    continue;
                for (int k = 0; k < K; ++k)
                    row[k] += d * at(s0 + c, k);
            }
        }
    }
    // C(m + N, i * K + k) = sum_l Theta_(l,m)(z_i) b_(l,m),k (row-major, contiguous over k).
    C.resize(2 * N_ + 1, std::size_t(P_) * K);
    for (int m = -N_; m <= N_; ++m) {
        const int am = std::abs(m);
        const Mat &t = theta_[m + N_];
        double *crow = C.row(m + N_).data();
        for (int i = 0; i < P_; ++i) {
            double *dst = crow + std::size_t(i) * K;
            for (int k = 0; k < K; ++k)
                dst[k] = 0;
            for (int l = am; l <= N_; ++l) {
                const double th = t(i, l - am);
                const double *src = b.row(index(l, m)).data();
                for (int k = 0; k < K; ++k)
                    dst[k] += th * src[k];
            }
        }
    }
    // R(j, i * K + k) = psi at (i, j) of node k: one GEMM A x (2N+1) x (P K).
    R.noalias() = trig_ * C;
    out.resize(points(), K);
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < P_; ++i)
            out.col(k).segment(std::size_t(i) * A_, A_) = R.col(std::size_t(i) * K + k);
}

} // namespace m1pn
