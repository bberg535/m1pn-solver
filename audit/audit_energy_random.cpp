// Discrete face energy with the face masses actually used (consistent Gauss face rules):
// (a^T Rvol + a^T B^H) / ||a||_M^2 for random PN fields with vacuum inflow (d = 0). Must be <= 0 up to roundoff.
#include "audit_common.hpp"

int main() {
    std::mt19937 rng(2026);
    std::normal_distribution<double> g(0, 1);
    for (int N : {3, 5, 7}) {
        for (int p : {1, 2, 3}) {
            Angular pn(N);
            ElementWenoOptions o; o.degree = p; o.dimension = 2; o.elements = 3; o.periodic = false;
            apply_env(o);
            ElementWeno s(pn, o);  // no inflow: d = 0
            double worst = -1e300, worst_face = -1e300, best = 1e300;
            for (int trial = 0; trial < 200; ++trial) {
                Field a(s.nodes, pn.q);
                const double smooth = trial % 2 ? 1. : 0.;  // alternate between smooth and noisy fields
                for (int i = 0; i < s.nodes; ++i)
                    for (int c = 0; c < pn.q; ++c) a(i, c) = g(rng) * (smooth ? std::exp(-.3 * c) : 1.);
                std::array<Field, 3> af;
                for (int d = 0; d < 2; ++d) af[d] = a * pn.A[d];
                Field loc(s.cells, s.local_nodes * pn.q);
                for (int e = 0; e < s.cells; ++e) {
                    Mat r = Mat::Zero(s.local_nodes, pn.q);
                    for (int d = 0; d < 2; ++d) {
                        Mat fe(s.local_nodes, pn.q);
                        for (int l = 0; l < s.local_nodes; ++l) fe.row(l) = af[d].row(s.connectivity_[e * s.local_nodes + l]);
                        r.noalias() -= s.convection_[d] * fe;
                    }
                    for (int l = 0; l < s.local_nodes; ++l)
                        for (int c = 0; c < pn.q; ++c) loc(e, l * pn.q + c) = r(l, c);
                }
                const Field Rvol = s.gather_load(loc);
                Field low = Field::Zero(s.nodes, 4), lf = Field::Zero(s.nodes, 12), bh = Field::Zero(s.nodes, pn.q), bl = Field::Zero(s.nodes, 4),
                      fa = Field::Zero(s.nodes, 3), bb = Field::Zero(s.nodes, 4);
                Vec area = Vec::Zero(s.nodes);
                s.add_boundary(0, a, low, lf, bh, bl, fa, area, bb);
                Field Ma(s.nodes, pn.q);
                {
                    Field l2(s.cells, s.local_nodes * pn.q);
                    for (int e = 0; e < s.cells; ++e) {
                        Mat xe(s.local_nodes, pn.q);
                        for (int l = 0; l < s.local_nodes; ++l) xe.row(l) = a.row(s.connectivity_[e * s.local_nodes + l]);
                        const Mat r = s.mass_ * xe;
                        for (int l = 0; l < s.local_nodes; ++l)
                            for (int c = 0; c < pn.q; ++c) l2(e, l * pn.q + c) = r(l, c);
                    }
                    Ma = s.gather_load(l2);
                }
                const double N2 = (a.array() * Ma.array()).sum();
                const double face = (a.array() * (Rvol + bh).array()).sum();
                // boundary trace energy sum_f int |u_h|^2 using the same face quadrature
                double trace2 = 0;
                for (const auto &f : s.faces_) {
                    const auto &rule = s.rules_[2 * f.direction + f.side];
                    for (int q = 0; q < rule.weights.size(); ++q) {
                        Vec tr = Vec::Zero(pn.q);
                        for (std::size_t n = 0; n < rule.local.size(); ++n) tr += rule.shape(n, q) * a.row(s.connectivity_[f.element * s.local_nodes + rule.local[n]]).transpose();
                        trace2 += rule.weights[q] * tr.squaredNorm();
                    }
                }
                worst = std::max(worst, face / N2);
                worst_face = std::max(worst_face, face / trace2);
                best = std::min(best, face / trace2);
            }
            J("A_discrete_face_energy_random").add("pn_order", N).add("degree", p).add("max_rate_over_N2", worst)
                .add("max_rate_over_trace_L2", worst_face).add("min_rate_over_trace_L2", best).done();
        }
    }
}
