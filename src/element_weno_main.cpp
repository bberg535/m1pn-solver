// Stand-alone driver for the element-based WENO variant.
#include "element_weno.hpp"
#include "hardware.hpp"
#include <filesystem>
#include <malloc.h>
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
    bool weno = true, limit = true, stabilize = true, legacy_boundary = false, symmetric_residual = false, ssprk43 = false,
         reduce_angular = true;
    bool positivity_projection = false, projection_strict = false, projection_warm_start = true;
    int projection_rule = 0;
    // smooth-outflow (docs/POSITIVITY_PROJECTION.md): angular concentration, direction (degrees), pulse width,
    // reference angular rule and sample points per axis of the error.
    double kappa = 5, angle = 30, outflow_width = .1;
    int reference_rule = 48, error_samples = 256;
    double projection_tolerance = 1e-12;
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
        else if (key == "--no-angular-reduction") a.reduce_angular = false; // both problems are even in 1D/2D
        else if (key == "--no-stabilization") a.stabilize = false;
        else if (key == "--legacy-boundary") a.legacy_boundary = true; // commit 8efe517 behaviour (docs/BOUNDARY_FIXES.md)
        else if (key == "--symmetric-residual") a.symmetric_residual = true;
        else if (key == "--positivity-projection") a.positivity_projection = true; // docs/POSITIVITY_PROJECTION.md
        else if (key == "--projection-strict") a.projection_strict = true;
        else if (key == "--projection-cold-start") a.projection_warm_start = false;
        else if (key == "--kappa") a.kappa = std::stod(next());
        else if (key == "--angle") a.angle = std::stod(next());
        else if (key == "--outflow-width") a.outflow_width = std::stod(next());
        else if (key == "--reference-rule") a.reference_rule = std::stoi(next());
        else if (key == "--error-samples") a.error_samples = std::stoi(next());
        else if (key == "--projection-rule") a.projection_rule = std::stoi(next());
        else if (key == "--projection-tolerance") a.projection_tolerance = std::stod(next());
        else throw std::runtime_error("Unknown option " + key);
    }
    require(a.problem == "smooth" || a.problem == "crossing-study" || a.problem == "smooth-outflow",
            "Problem must be smooth, crossing-study or smooth-outflow");
    return a;
}
} // namespace
int main(int argc, char **argv) {
    // Large fields are reallocated every stage; keep them in the heap (as m1pn_fast does).
    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, 1 << 30);
#ifdef _OPENMP
    // Physical cores of one NUMA node unless OMP_NUM_THREADS is set (src/hardware.hpp).
    omp_set_num_threads(m1pn::default_threads().threads);
#endif
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
        options.reduce_angular = args.reduce_angular && args.dimension < 3;
        options.positivity_projection = args.positivity_projection;
        options.projection_strict = args.projection_strict;
        options.projection_warm_start = args.projection_warm_start;
        options.projection_rule = args.projection_rule;
        options.projection_tolerance = args.projection_tolerance;
        const bool smooth = args.problem == "smooth";
        options.periodic = smooth;
        Inflow inflow;
        // smooth-outflow: psi_0 = bg / 4 pi + G(x - c) f(Omega), f = kappa / (2 pi (1 - e^-2kappa)) exp(kappa (Omega.e - 1))
        // (unit mass, smooth in angle), free transport: psi(x, Omega, t) = psi_0(x - t Omega, Omega), also the inflow.
        const bool outflow = args.problem == "smooth-outflow";
        const double kappa = args.kappa, ow = args.outflow_width, bgo = args.background;
        const V3 odir(std::cos(args.angle * pi / 180), std::sin(args.angle * pi / 180), 0), ocenter(.4, .4, 0);
        auto fdist = [kappa, odir](const V3 &o) {
            return kappa / (2 * pi * (1 - std::exp(-2 * kappa))) * std::exp(kappa * (o.dot(odir) - 1));
        };
        auto gauss2 = [ow](const V3 &d) { return std::exp(-(d[0] * d[0] + d[1] * d[1]) / (2 * ow * ow)); };
        if (outflow) {
            require(args.dimension == 2, "smooth-outflow is 2D");
            inflow.density = [=](const V3 &x, double t, const V3 &o) {
                return bgo / (4 * pi) + gauss2(x - t * o - ocenter) * fdist(o);
            };
        } else if (!smooth) {
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
        for (int k = 0; k <= 8; ++k) // both problems have data even in the unresolved directions; verify the inflow
            require(scheme.dropped_inflow(args.time * k / 8.) <= 1e-12, "Inflow is not even; use --no-angular-reduction");
        Field a;
        if (smooth) {
            a = scheme.project_l2([&](const V3 &x) { return element_weno_smooth_exact(pn, x, 0); });
        } else if (outflow) {
            Vec F = Vec::Zero(pn.q); // PN coefficients of f (fine rule; f is smooth)
            for (const auto &p : sphere_rule(2 * args.pn + 40))
                F += p.weight * fdist(p.omega) * harmonics(pn.order, p.omega);
            a = scheme.interpolate([&](const V3 &x) {
                Vec s = gauss2(x - ocenter) * F;
                s[0] += bgo / std::sqrt(4 * pi);
                return s;
            });
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
        // crossing-study: free transport, the kinetic density is bg + sum of the two shifted Gaussians (no
        // scattering). Its L2 distance contains the PN model error; the mass in the box measures the outflow.
        double kinetic_error = -1, kinetic_error_outflow = -1, exact_mass = -1;
        if (outflow) {
            // Exact density by angular quadrature (points with negligible f dropped), sampled L2 error on the cell
            // centres of an M x M grid (whole box and the outflow strip max(x, y) > 0.8), exact mass via erf.
            const double T = t;
            std::vector<AngularPoint> rule;
            double fmax = 0;
            for (const auto &p : sphere_rule(args.reference_rule))
                fmax = std::max(fmax, fdist(p.omega));
            for (const auto &p : sphere_rule(args.reference_rule))
                if (fdist(p.omega) > 1e-16 * fmax)
                    rule.push_back({p.omega, p.weight * fdist(p.omega)});
            auto rho = [&](const V3 &x) {
                double r = bgo;
                for (const auto &p : rule)
                    r += p.weight * gauss2(x - T * p.omega - ocenter);
                return r;
            };
            const int M = args.error_samples;
            std::vector<double> all(M, 0.), strip(M, 0.);
#pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < M; ++j)
                for (int i = 0; i < M; ++i) {
                    const V3 x((i + .5) / M, (j + .5) / M, 0);
                    const double d = project(scheme.evaluate(a, x))[0] - rho(x);
                    all[j] += d * d / (double(M) * M);
                    if (std::max(x[0], x[1]) > .8)
                        strip[j] += d * d / (double(M) * M);
                }
            double sa = 0, ss = 0;
            for (int j = 0; j < M; ++j)
                sa += all[j], ss += strip[j];
            kinetic_error = std::sqrt(sa);
            kinetic_error_outflow = std::sqrt(ss);
            auto line = [ow](double c) {
                return ow * std::sqrt(pi / 2) *
                       (std::erf((1 - c) / (std::sqrt(2.) * ow)) - std::erf(-c / (std::sqrt(2.) * ow)));
            };
            exact_mass = bgo;
            for (const auto &p : rule)
                exact_mass += p.weight * line(ocenter[0] + T * p.omega[0]) * line(ocenter[1] + T * p.omega[1]);
        } else if (!smooth) {
            const double w = args.width, bg = args.background, T = t;
            const V3 c1(.3 + T, .5, 0), c2(.5, .3 + T, 0);
            auto rho = [&](const V3 &x) {
                auto p = [w](const V3 &d) { return std::exp(-(d[0] * d[0] + d[1] * d[1]) / (2 * w * w)); };
                return bg + p(x - c1) + p(x - c2);
            };
            kinetic_error = scheme.l2_density_error(a, rho);
            auto line = [w](double c) {
                return w * std::sqrt(pi / 2) * (std::erf((1 - c) / (std::sqrt(2.) * w)) - std::erf(-c / (std::sqrt(2.) * w)));
            };
            exact_mass = bg + line(c1[0]) * line(c1[1]) + line(c2[0]) * line(c2[1]);
            // Error restricted to the outflow strip max(x, y) > 1 - 4 width (where the ghost acts).
            const double edge = 1 - 4 * w;
            const double inner = std::pow(scheme.l2_density_error(a, [&](const V3 &x) {
                return std::max(x[0], x[1]) > edge ? rho(x) : project(scheme.evaluate(a, x))[0];
            }), 2);
            kinetic_error_outflow = std::sqrt(std::max(0., inner));
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
                << ",\"threads\":" << m1pn::default_threads().threads << ",\"thread_policy\":\""
                << m1pn::default_threads().reason << "\""
                << ",\"dt_bound\":" << scheme.dt_bound << ",\"high_order_step\":" << scheme.high_order_step() << ",\"wall_seconds\":" << wall
                << ",\"l2_density_error\":" << error << ",\"final_mass\":" << total
                << ",\"kinetic_l2_error\":" << kinetic_error << ",\"kinetic_l2_error_outflow_strip\":" << kinetic_error_outflow
                << ",\"exact_mass\":" << exact_mass << ",\"boundary_outflow\":" << stats.boundary_outflow
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
                << ",\"stab_local\":" << stats.t_stab_local << ",\"finish\":" << stats.t_finish
                << ",\"ghost\":" << stats.projection.seconds << "}";
        if (args.positivity_projection) {
            const auto &p = stats.projection;
            summary << ",\"projection\":{\"rule\":" << args.projection_rule << ",\"calls\":" << p.calls
                    << ",\"identities\":" << p.identities << ",\"failures\":" << p.failures
                    << ",\"vacuum\":" << p.vacuum << ",\"scaled\":" << p.scaled
                    << ",\"augmented\":" << p.augmented << ",\"certificates\":" << p.certificates
                    << ",\"diracs\":" << p.diracs << ",\"farkas\":" << p.farkas
                    << ",\"warm_start\":" << (args.projection_warm_start ? "true" : "false")
                    << ",\"iterations\":" << p.iterations << ",\"backtracks\":" << p.backtracks
                    << ",\"max_iterations\":" << p.max_iterations << ",\"max_residual\":" << p.max_residual
                    << ",\"min_active_fraction\":" << p.min_active_fraction << "}";
        }
        summary << "}";
        std::ofstream(args.output + "/summary.json") << summary.str() << "\n";
        std::cout << summary.str() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
