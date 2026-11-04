#include "time.hpp"
#include <iomanip>
#include <iostream>
#include <sys/resource.h>
using namespace m1pn;
int main(int argc, char **argv) {
#ifdef M1PN_MPI
    MPI_Init(&argc, &argv);
#endif
    int result = 0;
    try {
        const int cells = argc > 1 ? std::stoi(argv[1]) : 4;
        std::cout << "N,Q,nodes,pairs,setup_seconds,sparse_target_seconds,dense_target_seconds,max_"
                     "difference,peak_rss_kib\n";
        for (int n : {3, 5, 7, 9, 15}) {
            double start = seconds();
            Angular a(n);
            MeshSpec spec;
            spec.elements = cells;
            Spatial f(spec);
            Transport tr(a, f, Numerics{});
            Field u(f.nodes, a.q);
            for (int i = 0; i < f.nodes; ++i)
                for (int k = 0; k < a.q; ++k)
                    u(i, k) = std::sin((i + 1.) * (k + 1.));
            Mat mass(f.M), stiffness(f.K);
            std::array<Mat, 3> convection = {Mat(f.C[0]), Mat(f.C[1]), Mat(f.C[2])};
            std::array<Mat, 3> angular = {Mat(a.A[0]), Mat(a.A[1]), Mat(a.A[2])};
            Eigen::LLT<Mat> factor(mass);
            auto reference = [&]() {
                Mat w = stiffness * u;
                for (int i = 0; i < f.nodes; ++i)
                    w.row(i) *= -tr.beta[i] / f.mass[i];
                Mat rhs = stiffness * w;
                for (int d = 0; d < 3; ++d)
                    rhs.noalias() += convection[d].transpose() * u * angular[d].transpose();
                for (const auto &face : f.faces)
                    for (const auto &p : face.points) {
                        Vec trace = Vec::Zero(a.q);
                        for (std::size_t j = 0; j < face.nodes.size(); ++j)
                            trace += p.shape[j] * u.row(face.nodes[j]).transpose();
                        Vec flux = tr.boundaries[p.normal_id]->homogeneous * trace;
                        for (std::size_t j = 0; j < face.nodes.size(); ++j)
                            rhs.row(face.nodes[j]) -= p.weight * p.shape[j] * flux.transpose();
                    }
                return Mat(factor.solve(rhs));
            };
            double setup = seconds() - start;
            Field optimized = tr.linear_target(u);
            Mat dense = reference();
            const double error = maxabs(optimized - dense);
            require(error < 1e-9, "Sparse and dense target differ");
            double accumulator = 0;
            constexpr int repeats = 5;
            start = seconds();
            for (int repeat = 0; repeat < repeats; ++repeat)
                accumulator += tr.linear_target(u)(0, 0);
            double sparse_time = (seconds() - start) / repeats;
            start = seconds();
            for (int repeat = 0; repeat < repeats; ++repeat)
                accumulator += reference()(0, 0);
            double dense_time = (seconds() - start) / repeats;
            require(std::isfinite(accumulator), "Nonfinite benchmark output");
            struct rusage usage{};
            getrusage(RUSAGE_SELF, &usage);
            std::cout << std::setprecision(17) << n << "," << a.q << "," << f.nodes << ","
                      << f.pairs.size() << "," << setup << "," << sparse_time << "," << dense_time
                      << "," << error << "," << usage.ru_maxrss << "\n";
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        result = 1;
    }
#ifdef M1PN_MPI
    MPI_Finalize();
#endif
    return result;
}
