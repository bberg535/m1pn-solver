#pragma once
#include "common.hpp"
namespace m1pn {
struct AngularPoint {
    V3 omega;
    double weight;
};
std::pair<Vec, Vec> gauss(int n);
Vec harmonics(int order, const V3 &omega);
std::vector<AngularPoint> sphere_rule(int order, int hemisphere = 0);
struct Angular {
    int order, q;
    double speed;
    std::array<Sparse, 3> A;
    explicit Angular(int n, const Budget &budget = {});
    Mat normal(const V3 &n) const;
    Vec apply(const V3 &direction, const Vec &u) const;
    Mat rotation(const Eigen::Matrix3d &r) const;
};
} // namespace m1pn
