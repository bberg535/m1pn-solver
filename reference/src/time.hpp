#pragma once
#include "transport.hpp"
namespace m1pn {
struct TimeControl {
    int order;
    double step_limit, operator_bound = 0, contraction = 1;
    std::string method;
    TimeControl(Transport &, const MeshSpec &, int rk, int norm_max_dofs = 512);
    void check(double dt) const;
};
double fourier_amplification(const Angular &, int cells, int dimension, double gamma, double dt,
                             int rk, bool gradient_defect = false, int correction_degree = -1);
struct Step {
    Field state, integrated_pairs, integrated_boundary, integrated_source;
    double rate_residual = 0, ledger_residual = 0, minimum_margin = 0;
    std::vector<Diagnostics> stages;
};
Step advance(Transport &, const TimeControl &, double time, double dt, const Field &);
Field reaction(const Field &, const Field &source, const Vec &absorption, const Vec &scattering,
               double anisotropy, double tau);
} // namespace m1pn
