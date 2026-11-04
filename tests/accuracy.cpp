#include "time.hpp"
#include <Eigen/Eigenvalues>
#include <iostream>

using namespace m1pn;

static double weighted_error(const Spatial &fe, const Field &a, const Field &b) {
    double sum = 0.;
    for (int i = 0; i < fe.nodes; ++i)
        sum += fe.mass[i] * (a.row(i) - b.row(i)).squaredNorm();
    return std::sqrt(sum);
}

static double weighted_density_error(const Spatial &fe, const Field &a, const Field &b) {
    double sum = 0.;
    for (int i = 0; i < fe.nodes; ++i) {
        const double difference = std::sqrt(4 * pi) * (a(i, 0) - b(i, 0));
        sum += fe.mass[i] * difference * difference;
    }
    return std::sqrt(sum);
}

int main() {
    try {
        constexpr double final_time = .05, target_dt = .0005, amplitude = .001;
        const std::vector<std::string> modes{"consistent", "lumped", "lumped_afc"};
        std::vector<Field> consistent_solutions(3);
        std::vector<double> lumped_errors(3), afc_errors(3);
        std::cout << "mode,cells,steps,eigenvalue,amplitude,kv_gamma,weighted_pn_l2,"
                     "physical_density_l2,deviation_from_consistent,limiter_activity,"
                     "max_balance_residual\n";
        for (const std::string &mode : modes) {
            int resolution_index = 0;
            for (int cells : {16, 32, 64}) {
                MeshSpec mesh;
                mesh.dimension = 1;
                mesh.elements = cells;
                mesh.periodic = true;
                Spatial fe(mesh, 1., {}, mode == "consistent");
                Angular pn(3);
                Numerics options;
                options.r2 = true;
                options.mass_mode = mode;
                options.stabilization = "gradient-defect";
                options.kv_gamma = .03;
                Transport transport(pn, fe, options);
                TimeControl control(transport, mesh, 3);

                Eigen::SelfAdjointEigenSolver<Mat> eig(Mat(pn.A[0]));
                require(eig.info() == Eigen::Success, "P3 streaming eigensystem failed");
                const Vec eigenvalues = eig.eigenvalues();
                const Mat eigenvectors = eig.eigenvectors();
                int wave = 0;
                for (int k = 1; k < pn.q; ++k)
                    if (std::abs(eigenvalues[k]) > std::abs(eigenvalues[wave])) wave = k;
                Field state(fe.nodes, pn.q);
                for (int i = 0; i < fe.nodes; ++i) {
                    const double phase = 2 * pi * fe.coordinates(i, 0);
                    state.row(i).setZero();
                    state(i, 0) = 1. / std::sqrt(4 * pi);
                    for (int k = 0; k < pn.q; ++k)
                        state(i, k) += amplitude * std::cos(phase) * eigenvectors(k, wave);
                }
                transport.check_state(state);
                const int steps = static_cast<int>(std::ceil(final_time / target_dt));
                const double dt = final_time / steps;
                control.check(dt);
                Field reference(fe.nodes, pn.q);
                for (int i = 0; i < fe.nodes; ++i) {
                    const double phase = 2 * pi * fe.coordinates(i, 0) -
                                         2 * pi * eigenvalues[wave] * final_time;
                    reference.row(i).setZero();
                    reference(i, 0) = 1. / std::sqrt(4 * pi);
                    for (int k = 0; k < pn.q; ++k)
                        reference(i, k) += amplitude * std::cos(phase) * eigenvectors(k, wave);
                }
                std::vector<long> activity;
                long active_pairs = 0;
                double balance = 0.;
                for (int step = 0; step < steps; ++step) {
                    Step result = advance(transport, control, step * dt, dt, state);
                    for (const Diagnostics &diagnostics : result.stages)
                        active_pairs += diagnostics.active_pairs;
                    balance = std::max(balance, result.ledger_residual);
                    state = std::move(result.state);
                }
                const double pn_error = weighted_error(fe, state, reference);
                const double density_error = weighted_density_error(fe, state, reference);
                double deviation = 0.;
                if (mode == "consistent") consistent_solutions[resolution_index] = state;
                else deviation = weighted_error(fe, state, consistent_solutions[resolution_index]);
                if (mode == "lumped") lumped_errors[resolution_index] = pn_error;
                if (mode == "lumped_afc") {
                    afc_errors[resolution_index] = pn_error;
                    require(afc_errors[resolution_index] < lumped_errors[resolution_index],
                            "lumped_afc did not improve on lumped PN error");
                }
                std::cout.precision(12);
                std::cout << mode << ',' << cells << ',' << steps << ',' << eigenvalues[wave]
                          << ',' << amplitude << ",0.03," << pn_error << ',' << density_error
                          << ',' << deviation << ',' << active_pairs << ','
                          << balance << '\n';
                ++resolution_index;
            }
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
