#include "transport.hpp"
#include <fstream>
#include <iostream>
#include <sys/resource.h>

using namespace m1pn;

static long peak_rss_kib() {
    std::ifstream in("/proc/self/status");
    std::string key;
    while (in >> key) {
        if (key == "VmHWM:") {
            long value = 0;
            std::string unit;
            in >> value >> unit;
            return value;
        }
        std::string rest;
        std::getline(in, rest);
    }
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

int main(int argc, char **argv) {
    try {
        require(argc == 6, "usage: kernel_benchmark mode cells dimension pnorder repeats");
        const std::string mode = argv[1];
        require(mode == "consistent" || mode == "lumped" || mode == "lumped_afc",
                "mode must be consistent, lumped, or lumped_afc");
        const int cells = std::stoi(argv[2]), dim = std::stoi(argv[3]);
        const int order = std::stoi(argv[4]), repeats = std::stoi(argv[5]);
        require(cells > 0 && dim >= 1 && dim <= 3 && order >= 0 && repeats > 0,
                "invalid benchmark parameters");
        const double setup_start = seconds();
        MeshSpec mesh;
        mesh.dimension = dim;
        mesh.elements = cells;
        mesh.periodic = true;
        Spatial fe(mesh, 1., {}, mode == "consistent");
        Angular pn(order);
        Numerics options;
        options.r2 = true;
        options.mass_mode = mode;
        Transport transport(pn, fe, options);
        const double setup_seconds = seconds() - setup_start;

        Field rhs(fe.nodes, pn.q);
        for (int i = 0; i < fe.nodes; ++i) {
            const double x = fe.coordinates(i, 0);
            for (int k = 0; k < pn.q; ++k)
                rhs(i, k) = std::sin((k + 1) * .173 + 2 * pi * x) +
                            .25 * std::cos((k + 2) * .119 + 4 * pi * x);
        }
        double residual = 0.;
        Field answer = transport.mass_target(rhs, residual);
        double checksum = answer.sum();
        double kernel_seconds = 0.;
        for (int r = 0; r < repeats; ++r) {
            const double start = seconds();
            answer = transport.mass_target(rhs, residual);
            kernel_seconds += seconds() - start;
            checksum += answer.sum();
        }
        std::cout.precision(17);
        std::cout << "{\"mode\":\"" << mode << "\",\"setup_seconds\":" << setup_seconds
                  << ",\"kernel_seconds\":" << kernel_seconds / repeats
                  << ",\"peak_rss_kib\":" << peak_rss_kib()
                  << ",\"mass_solve_calls\":" << fe.mass_solve_calls
                  << ",\"factorized\":" << (fe.factorized ? "true" : "false")
                  << ",\"nodes\":" << fe.nodes << ",\"Q\":" << pn.q
                  << ",\"checksum\":" << checksum << "}\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
