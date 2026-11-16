// WENO activation at the rounding limit (docs/ELEMENT_WENO.md, "Rundungsgrenze"): 1D periodic smooth PN solution
// (element_weno_smooth_exact), the setting of the tardis campaign (P3, T = 0.1, CFL 0.15). Not a ctest; prints JSON lines.
//
// Per (p, n):
//   data_t0 / data_T : gamma of the L2 projection of the exact solution (no time stepping)
//   noise            : gamma of the projection with relative coefficient noise of size `noise` (sensitivity to rounding)
//   run              : time stepping to T; L2 error, min gamma over all stages (stats) and after each step, the
//                      element of the minimum and its position.
// 1 - gamma = ratio^q with ratio = |u - u*|_e / |u|_e (src/element_weno.cpp, weno_gamma). Options as in the driver:
//   --degrees 1,2,3,4 --elements 32,64,128,256,512 --time .1 --cfl .15 --pn-order 3 --weno-q 1 --weno-linear 1e-3
//   --noise 1e-16,1e-14 --no-run
#include "element_weno.hpp"
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
using namespace m1pn;
namespace {
std::vector<double> list(const std::string &s) {
    std::vector<double> v;
    std::stringstream in(s);
    for (std::string item; std::getline(in, item, ',');)
        if (!item.empty())
            v.push_back(std::stod(item));
    return v;
}
struct GammaSummary {
    double min = 1, sum_defect = 0;
    int argmin = -1, active = 0; // elements with 1 - gamma > 1e-6
};
GammaSummary summarize(const Vec &g) {
    GammaSummary s;
    for (int e = 0; e < g.size(); ++e) {
        s.sum_defect += 1 - g[e];
        if (1 - g[e] > 1e-6)
            ++s.active;
        if (g[e] < s.min)
            s.min = g[e], s.argmin = e;
    }
    return s;
}
} // namespace
int main(int argc, char **argv) {
    try {
        std::vector<double> degrees{1, 2, 3, 4}, elements{32, 64, 128, 256, 512}, noises{1e-16, 1e-15, 1e-14};
        double T = .1, cfl = .15, weno_q = 1, weno_linear = 1e-3, deadband = 0, floor_ = 0;
        int pn_order = 3;
        bool run = true, weno = true, limit = true, stabilize = true;
        int trace = 0; // print every `trace` steps: t, min gamma, its position, L2 error, max density coefficient error
        for (int i = 1; i < argc; ++i) {
            const std::string k = argv[i];
            auto next = [&]() -> std::string {
                require(i + 1 < argc, "missing value for " + k);
                return argv[++i];
            };
            if (k == "--degrees") degrees = list(next());
            else if (k == "--elements") elements = list(next());
            else if (k == "--noise") noises = list(next());
            else if (k == "--time") T = std::stod(next());
            else if (k == "--cfl") cfl = std::stod(next());
            else if (k == "--pn-order") pn_order = std::stoi(next());
            else if (k == "--weno-q") weno_q = std::stod(next());
            else if (k == "--weno-linear") weno_linear = std::stod(next());
            else if (k == "--no-run") run = false;
            else if (k == "--no-weno") weno = false;
            else if (k == "--weno-deadband") deadband = std::stod(next());
            else if (k == "--weno-floor") floor_ = std::stod(next());
            else if (k == "--no-limit") limit = false;
            else if (k == "--no-stabilization") stabilize = false;
            else if (k == "--trace") trace = std::stoi(next());
            else throw std::runtime_error("unknown option " + k);
        }
        Angular pn(pn_order);
        std::cout << std::setprecision(6);
        for (double pd : degrees)
            for (double nd : elements) {
                ElementWenoOptions o; // as m1pn_element_weno --problem smooth (periodic, SSPRK(4,3), defaults)
                o.degree = int(pd), o.dimension = 1, o.elements = int(nd), o.periodic = true;
                o.weno_q = weno_q, o.weno_linear = weno_linear;
                o.weno = weno, o.limit = limit, o.stabilize = stabilize;
                o.weno_deadband = deadband, o.weno_floor = floor_;
                ElementWeno s(pn, o, Inflow{});
                auto exact = [&](double t) {
                    return s.project_l2([&](const V3 &x) { return element_weno_smooth_exact(pn, x, t); });
                };
                const Field a0 = exact(0), aT = exact(T);
                const GammaSummary g0 = summarize(s.weno_gamma(a0)), gT = summarize(s.weno_gamma(aT));
                std::ostringstream line;
                line << std::setprecision(6) << "{\"p\":" << o.degree << ",\"n\":" << o.elements << ",\"q\":" << weno_q
                     << ",\"linear\":" << weno_linear << ",\"deadband\":" << deadband
                     << ",\"floor\":" << floor_ << ",\"data_t0\":{\"min_gamma\":" << g0.min
                     << ",\"active\":" << g0.active << ",\"x\":" << (g0.argmin + .5) / o.elements << "}"
                     << ",\"data_T\":{\"min_gamma\":" << gT.min << ",\"active\":" << gT.active << "}";
                std::mt19937_64 rng(12345);
                std::normal_distribution<double> normal;
                line << ",\"noise\":[";
                for (std::size_t k = 0; k < noises.size(); ++k) {
                    Field b = a0;
                    for (int i = 0; i < b.rows(); ++i)
                        for (int c = 0; c < b.cols(); ++c)
                            b(i, c) *= 1 + noises[k] * normal(rng);
                    const GammaSummary gn = summarize(s.weno_gamma(b));
                    line << (k ? "," : "") << "{\"rel\":" << noises[k] << ",\"min_gamma\":" << gn.min
                         << ",\"active\":" << gn.active << "}";
                }
                line << "]";
                if (run) {
                    const int steps = std::max(1, int(std::ceil(T / (cfl * s.step_bound()) - 1e-12)));
                    const double dt = T / steps;
                    Field a = a0;
                    ElementWenoStats stats;
                    double t = 0, min_step = 1, x_min = -1;
                    int step_min = -1;
                    for (int k = 0; k < steps; ++k) {
                        a = s.advance(t, dt, a, &stats);
                        t += dt;
                        const GammaSummary g = summarize(s.weno_gamma(a));
                        if (g.min < min_step)
                            min_step = g.min, step_min = k + 1, x_min = (g.argmin + .5) / o.elements;
                        if (trace > 0 && ((k + 1) % trace == 0 || k + 1 == steps)) {
                            const Field ex = exact(t);
                            const double e2 = s.l2_density_error(
                                a, [&](const V3 &x) { return project(element_weno_smooth_exact(pn, x, t))[0]; });
                            std::cout << "{\"trace\":1,\"p\":" << o.degree << ",\"n\":" << o.elements << ",\"step\":" << k + 1
                                      << ",\"t\":" << t << ",\"min_gamma\":" << g.min << ",\"active\":" << g.active
                                      << ",\"x\":" << (g.argmin + .5) / o.elements << ",\"l2_error\":" << e2
                                      << ",\"max_coef_error\":" << (a - ex).cwiseAbs().maxCoeff() << "}" << std::endl;
                        }
                    }
                    const double err = s.l2_density_error(
                        a, [&](const V3 &x) { return project(element_weno_smooth_exact(pn, x, t))[0]; });
                    // Error of the exact projection itself (data error floor) for reference.
                    const double err0 = s.l2_density_error(
                        aT, [&](const V3 &x) { return project(element_weno_smooth_exact(pn, x, T))[0]; });
                    line << ",\"run\":{\"steps\":" << steps << ",\"dt\":" << dt << ",\"l2_error\":" << err
                         << ",\"projection_error\":" << err0 << ",\"min_gamma_stages\":" << stats.min_gamma
                         << ",\"min_gamma_steps\":" << min_step << ",\"step_of_min\":" << step_min
                         << ",\"x_of_min\":" << x_min << "}";
                }
                line << "}";
                std::cout << line.str() << std::endl;
            }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "weno_roundoff_study: " << e.what() << "\n";
        return 1;
    }
}
