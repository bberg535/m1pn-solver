// Stand-alone driver for the element-based WENO variant.
#include "element_weno.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace m1pn;
namespace {
struct Args {
    int pn = 3, degree = 2, dimension = 1, elements = 32;
    double time = .1, cfl = .9, dt = 0, width = .07, background = .001, weno_linear = 1e-3, weno_q = 1;
    int sample_grid = 0;
    double filter_strength = 0;
    std::string problem = "smooth", output = "tmp/element_weno";
    bool weno = true, limit = true, stabilize = true, legacy_boundary = false, symmetric_residual = false, ssprk43 = false;
};
Args parse(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> std::string {
            require(i + 1 < argc, "Missing value for " + key);
            return argv[++i];
        };
        if (key == "--pn-order") a.pn = std::stoi(next());
        else if (key == "--degree") a.degree = std::stoi(next());
        else if (key == "--dimension") a.dimension = std::stoi(next());
        else if (key == "--elements") a.elements = std::stoi(next());
        else if (key == "--time") a.time = std::stod(next());
        else if (key == "--cfl") a.cfl = std::stod(next());
        else if (key == "--dt") a.dt = std::stod(next());
        else if (key == "--width") a.width = std::stod(next());
        else if (key == "--background") a.background = std::stod(next());
        else if (key == "--weno-linear") a.weno_linear = std::stod(next());
        else if (key == "--weno-q") a.weno_q = std::stod(next());
        else if (key == "--sample-grid") a.sample_grid = std::stoi(next());
        else if (key == "--filter-strength") a.filter_strength = std::stod(next());
        else if (key == "--problem") a.problem = next();
        else if (key == "--output") a.output = next();
        else if (key == "--no-weno") a.weno = false;
        else if (key == "--no-limit") a.limit = false;
        else if (key == "--ssprk43") a.ssprk43 = true; // SSPRK(4,3) instead of SSPRK3
        else if (key == "--no-stabilization") a.stabilize = false;
        else if (key == "--legacy-boundary") a.legacy_boundary = true; // commit 8efe517 behaviour (docs/BOUNDARY_FIXES.md)
        else if (key == "--symmetric-residual") a.symmetric_residual = true;
        else throw std::runtime_error("Unknown option " + key);
    }
    require(a.problem == "smooth" || a.problem == "crossing-study", "Problem must be smooth or crossing-study");
    return a;
}
} // namespace
int main(int argc, char **argv) {
    try {
        const Args args = parse(argc, argv);
        Angular pn(args.pn);
        ElementWenoOptions options;
        options.degree = args.degree;
        options.dimension = args.dimension;
        options.elements = args.elements;
        options.weno = args.weno;
        options.limit = args.limit;
        options.stabilize = args.stabilize;
        options.weno_linear = args.weno_linear;
        options.weno_q = args.weno_q;
        if (args.legacy_boundary)
            options.vacuum_sign_limit = options.outward_bar_rounding = options.fullsphere_ghost = options.scale_invariant_weno = false;
        options.symmetric_residual = args.symmetric_residual;
        options.filter_strength = args.filter_strength;
        options.ssprk43 = args.ssprk43;
        const bool smooth = args.problem == "smooth";
        options.periodic = smooth;
        Inflow inflow;
        if (!smooth) {
            require(args.dimension == 2, "crossing-study is 2D");
            const double bg = args.background, w = args.width;
            inflow.density = [bg](const V3 &, double, const V3 &) { return bg / (4 * pi); };
            inflow.constant_density = bg / (4 * pi);
            inflow.atoms = [w](const V3 &x, double t) {
                auto p = [w](const V3 &d) { return std::exp(-d.squaredNorm() / (2 * w * w)); };
                return std::vector<Atom>{{V3::UnitX(), p(x - t * V3::UnitX() - V3(.3, .5, 0))},
                                         {V3::UnitY(), p(x - t * V3::UnitY() - V3(.5, .3, 0))}};
            };
        }
        ElementWeno scheme(pn, options, inflow);
        Field a;
        if (smooth) {
            a = scheme.project_l2([&](const V3 &x) { return element_weno_smooth_exact(pn, x, 0); });
        } else {
            a = scheme.interpolate([&](const V3 &x) {
                auto p = [&](const V3 &d) { return std::exp(-d.squaredNorm() / (2 * args.width * args.width)); };
                Vec s = Vec::Zero(pn.q);
                s[0] = args.background / std::sqrt(4 * pi);
                s += p(x - V3(.3, .5, 0)) * harmonics(pn.order, V3::UnitX());
                s += p(x - V3(.5, .3, 0)) * harmonics(pn.order, V3::UnitY());
                return s;
            });
        }
        const double step_limit = args.cfl * scheme.step_bound();
        const double requested = args.dt > 0 ? std::min(args.dt, step_limit) : step_limit;
        const int steps = std::max(1, int(std::ceil(args.time / requested - 1e-12)));
        const double dt = args.time / steps;
        ElementWenoStats stats;
        const double start = seconds();
        double t = 0;
        for (int n = 0; n < steps; ++n) {
            a = scheme.advance(t, dt, a, &stats);
            t += dt;
        }
        const double wall = seconds() - start;
        double error = -1;
        if (smooth) {
            const double final_time = t;
            error = scheme.l2_density_error(a, [&](const V3 &x) {
                return project(element_weno_smooth_exact(pn, x, final_time))[0];
            });
        }
        std::filesystem::create_directories(args.output);
        {
            std::ofstream csv(args.output + "/final.csv");
            csv << "x,y,z,density,jx,jy,jz\n" << std::setprecision(15);
            const int n = (args.sample_grid > 0 ? args.sample_grid : args.elements * args.degree) + 1, ny = args.dimension > 1 ? n : 1, nz = args.dimension > 2 ? n : 1;
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < n; ++i) {
                        V3 x(double(i) / (n - 1), args.dimension > 1 ? double(j) / (n - 1) : 0,
                             args.dimension > 2 ? double(k) / (n - 1) : 0);
                        const Four u = project(scheme.evaluate(a, x));
                        csv << x[0] << ',' << x[1] << ',' << x[2] << ',' << u[0] << ',' << u[1] << ',' << u[2] << ',' << u[3] << '\n';
                    }
        }
        double min_coefficient_density = std::numeric_limits<double>::infinity();
        double min_coefficient_margin = min_coefficient_density;
        for (int i = 0; i < scheme.nodes; ++i) {
            const Four u = project(a.row(i).transpose());
            min_coefficient_density = std::min(min_coefficient_density, u[0]);
            min_coefficient_margin = std::min(min_coefficient_margin, margin(u));
        }
        double total = 0;
        for (int i = 0; i < scheme.nodes; ++i)
            total += scheme.lumped[i] * project(a.row(i).transpose())[0];
        std::ostringstream summary;
        summary << std::setprecision(12) << "{\"problem\":\"" << args.problem << "\",\"pn_order\":" << args.pn
                << ",\"degree\":" << args.degree << ",\"elements\":" << args.elements
                << ",\"dimension\":" << args.dimension << ",\"steps\":" << steps << ",\"dt\":" << dt
                << ",\"dt_bound\":" << scheme.dt_bound << ",\"high_order_step\":" << scheme.high_order_step() << ",\"wall_seconds\":" << wall
                << ",\"l2_density_error\":" << error << ",\"final_mass\":" << total
                << ",\"min_coefficient_density\":" << min_coefficient_density
                << ",\"min_coefficient_margin\":" << min_coefficient_margin
                << ",\"min_gamma\":" << stats.min_gamma << ",\"min_limiter\":" << stats.min_limiter
                << ",\"min_boundary_limiter\":" << stats.min_boundary_limiter
                << ",\"limited_elements\":" << stats.limited_elements
                << ",\"limited_boundary_nodes\":" << stats.limited_boundary_nodes
                << ",\"nudged_states\":" << stats.nudged_states
                << ",\"max_raw_zero_sum\":" << stats.max_raw_zero_sum
                << ",\"max_limited_zero_sum\":" << stats.max_limited_zero_sum
                << ",\"seconds\":{\"closure\":" << stats.t_closure << ",\"weno\":" << stats.t_weno
                << ",\"stabilization\":" << stats.t_stabilization << ",\"residual\":" << stats.t_residual
                << ",\"boundary\":" << stats.t_boundary << ",\"mass\":" << stats.t_mass
                << ",\"limiter\":" << stats.t_limiter << ",\"gradient_local\":" << stats.t_gradient_local
                << ",\"gradient_gather\":" << stats.t_gradient_gather << ",\"gradient_solve\":" << stats.t_gradient_solve
                << ",\"stab_local\":" << stats.t_stab_local << ",\"finish\":" << stats.t_finish << "}}";
        std::ofstream(args.output + "/summary.json") << summary.str() << "\n";
        std::cout << summary.str() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
