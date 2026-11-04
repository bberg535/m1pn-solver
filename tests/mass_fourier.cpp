#include "time.hpp"
#include <Eigen/Eigenvalues>
#include <complex>
#include <iomanip>
#include <iostream>

using namespace m1pn;

namespace {
double rk_amplification(std::complex<double> z, int rk) {
    std::complex<double> value = 1. + z + .5 * z * z;
    if (rk == 3) value += z * z * z / 6.;
    return std::abs(value);
}

void verify_case(int dim, int cells, const std::string &mass_mode, int corrections,
                 const std::string &stabilization) {
    MeshSpec mesh;
    mesh.dimension = dim;
    mesh.elements = cells;
    mesh.periodic = true;
    const bool kv = stabilization == "gradient-defect";
    const bool r2 = true;
    Numerics numerics;
    numerics.r2 = r2;
    numerics.gamma = .07;
    numerics.kv_gamma = numerics.gamma;
    numerics.stabilization = stabilization;
    numerics.mass_mode = mass_mode;
    numerics.mass_corrections = std::max(1, corrections);
    Spatial spatial(mesh, 1., {}, false);
    Angular pn(1);
    Transport transport(pn, spatial, numerics);

    // Assemble the complete real generator from its action. Its complex
    // spectrum contains every discrete Fourier mode and angular eigenbranch.
    const int total = spatial.nodes * pn.q;
    Mat generator(total, total);
    for (int j = 0; j < total; ++j) {
        Field unit = Field::Zero(spatial.nodes, pn.q);
        unit.data()[j] = 1.;
        const Field rate = transport.linear_target(unit);
        generator.col(j) = Eigen::Map<const Vec>(rate.data(), total);
    }
    Eigen::ComplexEigenSolver<Mat> eig(generator, false);
    require(eig.info() == Eigen::Success, "Dense periodic generator eigensolve failed");

    const int degree = mass_mode == "lumped" ? 0 : corrections;
    // Deliberately unstable probe: avoids vacuous max-amplification=1 checks.
    const double dt = 8. / cells;
    for (int rk : {2, 3}) {
        double observed = 1.;
        for (const auto &lambda : eig.eigenvalues())
            observed = std::max(observed, rk_amplification(dt * lambda, rk));
        const double predicted = fourier_amplification(
            pn, cells, dim, numerics.gamma, dt, rk, kv, degree);
        const double error = std::abs(observed - predicted);
        std::cout << std::setprecision(17)
                  << "{\"dimension\":" << dim << ",\"cells\":" << cells
                  << ",\"mass_mode\":\"" << mass_mode << "\",\"corrections\":"
                  << (mass_mode == "lumped" ? 0 : corrections)
                  << ",\"stabilization\":\"" << stabilization << "\",\"rk\":" << rk
                  << ",\"dt\":" << dt << ",\"dense_amplification\":" << observed
                  << ",\"fourier_amplification\":" << predicted
                  << ",\"absolute_error\":" << error << "}\n";
        require(error <= 2e-9 * std::max(1., observed),
                "Assembled periodic spectrum disagrees with Fourier amplification");
    }
}
} // namespace

int main() {
    try {
        for (const auto &[dim, cells] :
             {std::pair<int, int>{1, 8}, {2, 4}, {3, 3}}) {
            for (const auto &stabilization :
                 {std::string("gradient-defect"), std::string("biharmonic-original"),
                  std::string("biharmonic-corrected")}) {
                verify_case(dim, cells, "lumped", 0, stabilization);
                verify_case(dim, cells, "lumped_afc", 1, stabilization);
                verify_case(dim, cells, "lumped_afc", 3, stabilization);
            }
        }
    } catch (const std::exception &error) {
        std::cerr << "mass Fourier validation failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
