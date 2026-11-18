#pragma once
#include "angular.hpp"
#include <memory>

namespace m1pn {

struct AdaptivePnDiagnostics {
    double face_dissipation = 0;
    double raw_low_defect = 0;
    double element_balance_defect = 0;
    int limited_elements = 0;
};

// Periodic tensor Bernstein Q1 prototype in one, two or three spatial dimensions.
// Each state block contains the active spatial DOFs of one angular degree.
class AdaptivePn {
public:
    using State = std::vector<Field>;

    AdaptivePn(const Angular &, int dimension, int elements_per_axis, std::vector<int> orders);
    ~AdaptivePn();
    AdaptivePn(const AdaptivePn &) = delete;
    AdaptivePn &operator=(const AdaptivePn &) = delete;

    const std::vector<int> &orders() const { return orders_; }
    int cells() const { return static_cast<int>(orders_.size()); }
    int local_nodes() const { return corners_; }
    std::size_t dofs() const;
    int mixed_faces() const;
    double euler_cone_step() const { return euler_cone_step_; }
    V3 node_position(int cell, int local_node) const;
    Field element(const State &, int cell) const;

    State interpolate(const std::function<Vec(const V3 &)> &) const;
    State project_from(const AdaptivePn &old, const State &) const;
    std::vector<int> select_orders(const State &, double lower, double upper) const;
    State rate(const State &, bool limit = false, AdaptivePnDiagnostics * = nullptr) const;
    State advance(const State &, double dt, bool limit = false, int *limited_elements = nullptr) const;

    double inner_product(const State &, const State &) const;
    Four integrated_moments(const State &) const;
    double min_cone_margin(const State &) const;

private:
    struct DegreeSpace;
    const Angular &angular_;
    int dimension_, count_, corners_;
    double h_, lump_, viscosity_sum_ = 0, euler_cone_step_;
    std::vector<int> orders_;
    std::vector<std::array<int, 3>> neighbor_;
    Mat mass_, viscosity_;
    std::array<Mat, 3> convection_, face_mass_;
    std::array<std::vector<int>, 3> face_nodes_;
    std::vector<std::array<Sparse, 3>> flux_;
    std::vector<std::unique_ptr<DegreeSpace>> degrees_;

    State zero_state() const;
    void check_shape(const State &) const;
    void add_element(State &, int cell, const Field &) const;
    void solve_mass(State &) const;
    Field low_moments(const State &) const;
    void limit_low(const State &, const std::vector<std::array<Field, 3>> &high_flux, State &,
                   AdaptivePnDiagnostics *) const;
};

} // namespace m1pn
