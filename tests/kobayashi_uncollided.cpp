// Kobayashi problems 1i and 2i (pure absorber): the steady uncollided density of src/uncollided_box3d.hpp is the whole
// solution there; compare with the analytic reference (Kyoto, as tabulated in Brown, Chang, Hanebutte, UCRL-JC-135163,
// Tables III and V; 3 digits; line C of problem 1 lies at y = 55, the table heading there says y = 95, but its first
// value equals line A at y = 55). Also prints the quadrature convergence (order / panels) at the detector points.
//   kobayashi_uncollided_tests [--study]
#include "uncollided_box3d.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace m1pn;
using namespace m1pn::uncollided;
namespace {
struct Ref {
    int problem;
    char line;
    double x, y, z, value;
};
// clang-format off
const Ref refs[] = {
    {1,'A',5,5,5,5.96e0},{1,'A',5,15,5,1.37e0},{1,'A',5,25,5,5.01e-1},{1,'A',5,35,5,2.52e-1},{1,'A',5,45,5,1.50e-1},
    {1,'A',5,55,5,5.95e-2},{1,'A',5,65,5,1.53e-2},{1,'A',5,75,5,4.18e-3},{1,'A',5,85,5,1.19e-3},{1,'A',5,95,5,3.47e-4},
    {1,'B',5,5,5,5.96e0},{1,'B',15,15,15,4.71e-1},{1,'B',25,25,25,1.70e-1},{1,'B',35,35,35,8.68e-2},{1,'B',45,45,45,5.25e-2},
    {1,'B',55,55,55,1.33e-2},{1,'B',65,65,65,1.46e-3},{1,'B',75,75,75,1.75e-4},{1,'B',85,85,85,2.25e-5},{1,'B',95,95,95,3.01e-6},
    {1,'C',5,55,5,5.95e-2},{1,'C',15,55,5,5.50e-2},{1,'C',25,55,5,4.81e-2},{1,'C',35,55,5,3.97e-2},{1,'C',45,55,5,3.16e-2},
    {1,'C',55,55,5,2.35e-2},{1,'C',65,55,5,5.84e-3},{1,'C',75,55,5,1.57e-3},{1,'C',85,55,5,4.53e-4},{1,'C',95,55,5,1.37e-4},
    {2,'A',5,5,5,5.96e0},{2,'A',5,15,5,1.37e0},{2,'A',5,25,5,5.01e-1},{2,'A',5,35,5,2.52e-1},{2,'A',5,45,5,1.50e-1},
    {2,'A',5,55,5,9.92e-2},{2,'A',5,65,5,7.02e-2},{2,'A',5,75,5,5.22e-2},{2,'A',5,85,5,4.03e-2},{2,'A',5,95,5,3.21e-2},
    {2,'B',5,95,5,3.21e-2},{2,'B',15,95,5,1.71e-3},{2,'B',25,95,5,1.41e-4},{2,'B',35,95,5,3.27e-5},{2,'B',45,95,5,1.09e-5},
    {2,'B',55,95,5,4.14e-6},
};
// clang-format on
} // namespace
int main(int argc, char **argv) {
    const bool study = argc > 1 && std::strcmp(argv[1], "--study") == 0;
    int failures = 0;
    for (int problem = 1; problem <= 2; ++problem) {
        Box3dGeometry g = kobayashi::geometry(problem);
        Box3dGeometry fine = g;
        fine.order = 10, fine.outside_panels = 4, fine.inside_panels = 8;
        double worst = 0, worst_q = 0;
        const auto start = std::chrono::steady_clock::now();
        for (const Ref &r : refs) {
            if (r.problem != problem)
                continue;
            const double p[3] = {r.x, r.y, r.z};
            const double v = g.density(p), vf = study ? fine.density(p) : v;
            // The reference has 3 digits: |v - ref| <= 0.5 units of its last digit (plus 1 % of a unit for ties).
            const double rel = v / r.value - 1, unit = std::pow(10., std::floor(std::log10(r.value)) - 2);
            const double digits = std::fabs(v - r.value) / unit;
            worst = std::max(worst, digits), worst_q = std::max(worst_q, std::fabs(v / vf - 1));
            std::printf("%di %c (%4.0f %4.0f %4.0f)  ours %.6e  Kyoto %.2e  rel %+.2e  (%.2f units of the last digit)\n", problem,
                        r.line, r.x, r.y, r.z, v, r.value, rel, digits);
            if (study)
                std::printf("      fine quadrature %.6e  rel diff %.1e\n", vf, v / vf - 1);
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("problem %di: max deviation from Kyoto %.2f units of the last digit%s (%.2f s)\n", problem, worst,
                    study ? (" , quadrature " + std::to_string(worst_q)).c_str() : "", s);
        if (worst > .51) { // within the rounding of the 3-digit reference
            std::printf("FAILED problem %di\n", problem);
            ++failures;
        }
    }
    std::printf(failures ? "FAILED\n" : "PASSED\n");
    return failures;
}
