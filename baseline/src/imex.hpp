#pragma once
#include "spatial.hpp"
namespace m1pn {
struct ImplicitResult {
    Field rate, predictor;
    double residual = 0, relative_residual = 0;
    int solves = 0;
};
class ImplicitMassSolver {
    const Spatial &spatial_;
    Sparse positive_A_;
    double cached_dt_ = -1;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> direct_factor_, defect_factor_;
    bool direct_ready_ = false, defect_ready_ = false;
    long factorization_count_ = 0;
    double norm_lumped_inverse(const Field &) const;
public:
    ImplicitMassSolver(const Spatial &, const Sparse &positive_A);
    ImplicitResult solve(const Field &b, double dt, const std::string &mode,
                         double rtol, double atol, int max_iterations);
    long factorizations() const { return factorization_count_; }
};
} // namespace m1pn
