// Stand-alone driver for the element-based WENO variant.
#include "element_weno.hpp"
#include "element_weno_mpi.hpp"
#ifdef M1PN_ELEMENT_GPU
#include "element_weno_gpu.hpp"
#endif
#include <cstdlib>
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
    double time = .1, cfl = .9, dt = 0, width = .07, background = .001, weno_linear = 1e-3, weno_q = 3;
    int sample_grid = 0;
    double filter_strength = 0;
    std::string problem = "smooth", output = "tmp/element_weno";
    bool weno = true, limit = true, stabilize = true, legacy_boundary = false, symmetric_residual = false, ssprk43 = true,
         reduce_angular = true;
    bool positivity_projection = true, projection_strict = false, projection_warm_start = true;
    int projection_rule = 0;
    // smooth-outflow (docs/POSITIVITY_PROJECTION.md): angular concentration, direction (degrees), pulse width,
    // reference angular rule and sample points per axis of the error.
    double kappa = 5, angle = 30, outflow_width = .1;
    int reference_rule = 48, error_samples = 256;
    double projection_tolerance = 1e-12, projection_incoming_weight = 1;
    bool projection_half_range = false;
    double ghost_filter = 0;
    bool projection_diagnostics = false, projection_hybrid = false, projection_high_order = false;
    int ghost_study = 0; // checkpoints of the ghost-estimator study (0: off)
    double ghost_study_filter = 16;
    int mpi_halo = 2, mpi_chunks = 0; // MPI slab decomposition (src/element_weno_mpi.hpp)
    std::string step_cache;           // file of stored spectral radii (keyed by configuration and an operator probe)
    bool gpu = false;
    std::string dump_state;           // binary dump of the final coefficients (rows, cols, row-major doubles)
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
        else if (key == "--mpi-halo") a.mpi_halo = std::stoi(next());
        else if (key == "--mpi-chunks") a.mpi_chunks = std::stoi(next());
        else if (key == "--dump-state") a.dump_state = next();
        else if (key == "--step-cache") a.step_cache = next();
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
        else if (key == "--gpu") a.gpu = true; // experimental device time loop (src/element_weno_gpu.cpp)
        else if (key == "--ssprk43") a.ssprk43 = true; // SSPRK(4,3) (default since 2026-11-15, docs/SPEEDUP_OPTIONS.md)
        else if (key == "--ssprk3") a.ssprk43 = false; // Shu-Osher SSPRK3 (former default)
        else if (key == "--no-angular-reduction") a.reduce_angular = false; // both problems are even in 1D/2D
        else if (key == "--no-stabilization") a.stabilize = false;
        else if (key == "--legacy-boundary") a.legacy_boundary = true; // commit 8efe517 behaviour (docs/BOUNDARY_FIXES.md)
        else if (key == "--symmetric-residual") a.symmetric_residual = true;
        else if (key == "--positivity-projection") a.positivity_projection = true; // default, docs/POSITIVITY_PROJECTION.md
        else if (key == "--c6-ghost") a.positivity_projection = false;          // previous default ghost (C6)
        else if (key == "--projection-strict") a.projection_strict = true;
        else if (key == "--projection-cold-start") a.projection_warm_start = false;
        else if (key == "--projection-incoming-weight") a.projection_incoming_weight = std::stod(next());
        else if (key == "--ghost-filter") a.ghost_filter = std::stod(next());
        else if (key == "--projection-diagnostics") a.projection_diagnostics = true;
        else if (key == "--projection-high-order") a.projection_high_order = true; // z* also in the PN boundary flux
        else if (key == "--ghost-study") a.ghost_study = std::stoi(next());
        else if (key == "--ghost-study-filter") a.ghost_study_filter = std::stod(next());
        else if (key == "--projection-mode") {
            const std::string m = next();
            require(m == "full" || m == "half" || m == "hybrid", "--projection-mode must be full, half or hybrid");
            a.projection_half_range = m == "half";
            a.projection_hybrid = m == "hybrid";
        }
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
    if (a.legacy_boundary) a.positivity_projection = false; // the projection replaces the C6 ghost, which legacy disables
    return a;
}
} // namespace
int main(int argc, char **argv) {
    // Task priorities (MPI line pipeline before filler work); read by the OpenMP runtime at its start.
    setenv("OMP_MAX_TASK_PRIORITY", "1", 0);
    int mpi_rank = 0, mpi_size = 1;
#ifdef M1PN_MPI
    {
        int provided = 0;
        // UCX: the line pipeline sends many messages of some 10 KB from a thread that is rarely inside MPI; the
        // rendezvous protocol (default above a few KB) then stalls (NX, N=31, 4 ranks: 49 s -> 18.5 s with 4 MB).
        setenv("UCX_RNDV_THRESH", "512kb", 0); // pipeline messages eager, MB-sized halo messages rendezvous; keeps a user value
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided); // MULTIPLE: threaded line pipeline
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    }
    struct Finalize {
        ~Finalize() { MPI_Finalize(); }
    } finalize;
#endif
    // Large fields are reallocated every stage; keep them in the heap (as m1pn_fast does).
    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, 1 << 30);
#ifdef _OPENMP
    // Physical cores of one NUMA node unless OMP_NUM_THREADS is set (src/hardware.hpp).
    omp_set_num_threads(m1pn::default_threads().threads);
#endif
    try {
        const double program_start = seconds();
        const Args args = parse(argc, argv);
        const double zc = args.dimension == 3 ? .5 : 0.; // crossing-study pulse centre height (3D: mid-plane)
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
        options.projection_incoming_weight = args.projection_incoming_weight;
        options.projection_half_range = args.projection_half_range;
        options.ghost_filter = args.ghost_filter;
        options.projection_diagnostics = args.projection_diagnostics;
        options.projection_hybrid = args.projection_hybrid;
        options.projection_high_order = args.projection_high_order;
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
            require(args.dimension >= 2, "crossing-study is 2D or 3D (3D: pulses at z = 0.5, no kinetic metrics)");
            const double bg = args.background, w = args.width;
            inflow.density = [bg](const V3 &, double, const V3 &) { return bg / (4 * pi); };
            inflow.constant_density = bg / (4 * pi);
            inflow.atoms = [w, zc](const V3 &x, double t) {
                auto p = [w](const V3 &d) { return std::exp(-d.squaredNorm() / (2 * w * w)); };
                return std::vector<Atom>{{V3::UnitX(), p(x - t * V3::UnitX() - V3(.3, .5, zc))},
                                         {V3::UnitY(), p(x - t * V3::UnitY() - V3(.5, .3, zc))}};
            };
        }
        // MPI: every rank integrates its slab (plus halo) of the global mesh; rank 0 evaluates the gathered final
        // state with the unchanged serial code below (global scheme `post`).
        std::unique_ptr<ElementWeno> holder;
#ifdef M1PN_MPI
        std::unique_ptr<ElementWenoMpi> mpi;
        SlabPartition part;
        if (mpi_size > 1) {
            require(args.dimension >= 2 && !smooth && args.ghost_study == 0,
                    "MPI runs: 2D/3D open problems (crossing-study, smooth-outflow), no ghost study");
            const int D = args.dimension - 1;
            part = slab_partition(args.elements, mpi_size, mpi_rank, D, args.mpi_halo);
            holder = std::make_unique<ElementWeno>(pn, slab_options(options, part), inflow);
            mpi = std::make_unique<ElementWenoMpi>(*holder, MPI_COMM_WORLD, part, args.mpi_chunks);
        } else
#endif
            holder = std::make_unique<ElementWeno>(pn, options, inflow);
        ElementWeno *sp = holder.get();
#ifdef M1PN_ELEMENT_GPU
        std::unique_ptr<ElementWenoGpu> device;
        double gpu_radius_seconds = 0;
        if (args.gpu) {
            require(mpi_size == 1 && args.ghost_study == 0, "--gpu: one process, no ghost study");
            device = std::make_unique<ElementWenoGpu>(*sp);
            const double r0 = seconds();
            sp->set_spectral_radius(device->spectral_radius());
            gpu_radius_seconds = seconds() - r0;
        }
#else
        require(!args.gpu, "--gpu: built without M1PN_ELEMENT_GPU");
#endif
        for (int k = 0; k <= 8; ++k) // both problems have data even in the unresolved directions; verify the inflow
            require(sp->dropped_inflow(args.time * k / 8.) <= 1e-12, "Inflow is not even; use --no-angular-reduction");
        Field a;
        if (smooth) {
            a = sp->project_l2([&](const V3 &x) { return element_weno_smooth_exact(pn, x, 0); });
        } else if (outflow) {
            Vec F = Vec::Zero(pn.q); // PN coefficients of f (fine rule; f is smooth)
            for (const auto &p : sphere_rule(2 * args.pn + 40))
                F += p.weight * fdist(p.omega) * harmonics(pn.order, p.omega);
            a = sp->interpolate([&](const V3 &x) {
                Vec s = gauss2(x - ocenter) * F;
                s[0] += bgo / std::sqrt(4 * pi);
                return s;
            });
        } else {
            const Vec hx = harmonics(pn.order, V3::UnitX()), hy = harmonics(pn.order, V3::UnitY()); // once, not per node
            a = sp->interpolate([&](const V3 &x) {
                auto p = [&](const V3 &d) { return std::exp(-d.squaredNorm() / (2 * args.width * args.width)); };
                Vec s = Vec::Zero(pn.q);
                s[0] = args.background / std::sqrt(4 * pi);
                s += p(x - V3(.3, .5, zc)) * hx;
                s += p(x - V3(.5, .3, zc)) * hy;
                return s;
            });
        }
        if (!args.step_cache.empty()) {
            // The power iteration of high_order_step() costs ~120 operator applications (N=31: 25 s). A stored radius
            // is reused when the configuration and the probe (one application, bitwise) match; the step is then the
            // one the iteration would give for this rank count.
            std::ostringstream k;
            k << "v1 pn=" << args.pn << " degree=" << args.degree << " dim=" << args.dimension << " E=" << args.elements
              << " ranks=" << mpi_size << " ssprk43=" << args.ssprk43 << " probe=" << std::hexfloat << sp->operator_probe();
            const std::string key = k.str();
            double radius = -1;
            if (mpi_rank == 0) {
                std::ifstream in(args.step_cache);
                for (std::string line; std::getline(in, line);)
                    if (line.size() > key.size() && line.compare(0, key.size(), key) == 0 && line[key.size()] == ' ')
                        radius = std::strtod(line.c_str() + key.size() + 1, nullptr);
            }
#ifdef M1PN_MPI
            if (mpi_size > 1)
                MPI_Bcast(&radius, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
            if (radius > 0)
                sp->set_spectral_radius(radius);
            else {
                radius = sp->spectral_radius();
                if (mpi_rank == 0) {
                    std::ofstream out(args.step_cache, std::ios::app);
                    out << key << ' ' << std::hexfloat << radius << '\n';
                }
            }
        }
        const double step_limit = args.cfl * sp->step_bound();
        const double requested = args.dt > 0 ? std::min(args.dt, step_limit) : step_limit;
        const int steps = std::max(1, int(std::ceil(args.time / requested - 1e-12)));
        const double dt = args.time / steps;
        // Ghost-estimator study: outgoing half moments of the estimators vs the exact kinetic ones at the boundary nodes.
        std::vector<AngularPoint> study_rule;
        if (args.ghost_study > 0) {
            require(!smooth && args.positivity_projection, "--ghost-study needs an open problem and --positivity-projection");
            for (const auto &p : sphere_rule(args.reference_rule))
                if (outflow ? fdist(p.omega) > 1e-16 : true)
                    study_rule.push_back(p);
        }
        auto exact_half = [&](const V3 &x, const V3 &n, double tt) {
            Four m = Four::Zero();
            if (outflow) {
                for (const auto &p : study_rule)
                    if (p.omega.dot(n) > 0) {
                        const double v = p.weight * (bgo / (4 * pi) + gauss2(x - tt * p.omega - ocenter) * fdist(p.omega));
                        m += v * Four(1, p.omega[0], p.omega[1], p.omega[2]);
                    }
                // Background on the part of the hemisphere outside the pruned rule is negligible (f > 1e-16 kept all).
            } else {
                const double bg = args.background, w = args.width;
                m += Four(bg / 2, bg / 4 * n[0], bg / 4 * n[1], bg / 4 * n[2]);
                auto p = [w](const V3 &d) { return std::exp(-(d[0] * d[0] + d[1] * d[1]) / (2 * w * w)); };
                const V3 ex = V3::UnitX(), ey = V3::UnitY();
                if (ex.dot(n) > 0)
                    m += p(x - tt * ex - V3(.3, .5, zc)) * Four(1, 1, 0, 0);
                if (ey.dot(n) > 0)
                    m += p(x - tt * ey - V3(.5, .3, zc)) * Four(1, 0, 1, 0);
            }
            return m;
        };
        constexpr int E = ElementWeno::ghost_estimators;
        std::array<std::array<double, 4>, E> study_err{}; // |d rho|, |d n.J|, |dJ|, |d (rho - n.J)|
        double study_scale = 0, study_points = 0;
        auto run_study = [&](const Field &state, double tt) {
            const auto g = sp->ghost_estimates(state, args.ghost_study_filter);
            for (std::size_t k = 0; k < g.node.size(); ++k) {
                const V3 x = sp->node_position(g.node[k]), n = sp->side_normal(g.side[k]);
                const Four ex = exact_half(x, n, tt);
                study_scale += ex[0];
                study_points += 1;
                for (int e = 0; e < E; ++e) {
                    const Four d = g.m[k][e] - ex;
                    study_err[e][0] += std::abs(d[0]);
                    study_err[e][1] += std::abs(n.dot(d.tail<3>()));
                    study_err[e][2] += d.tail<3>().norm();
                    study_err[e][3] += std::abs(d[0] - n.dot(d.tail<3>()));
                }
            }
        };
        ElementWenoStats stats;
        const double start = seconds();
        double setup_seconds = start - program_start, rebuild_seconds = 0;
        double t = 0;
        double study_seconds = 0;
#ifdef M1PN_ELEMENT_GPU
        if (device) {
            const double step_start = seconds();
            a = device->run(t, dt, steps, a, &stats);
            stats.t_advance += seconds() - step_start;
            for (int n = 0; n < steps; ++n)
                t += dt;
        }
#endif
        for (int n = 0; n < (args.gpu ? 0 : steps); ++n) {
            const double step_start = seconds();
            a = sp->advance(t, dt, a, &stats);
            stats.t_advance += seconds() - step_start;
            t += dt;
            if (args.ghost_study > 0 && (n + 1) % std::max(1, steps / args.ghost_study) == 0) {
                const double s0 = seconds();
                run_study(a, t);
                study_seconds += seconds() - s0;
            }
        }
        double wall = seconds() - start - study_seconds;
        const double post_start = seconds();
        std::string mpi_json;
        // The step bound used by this run (MPI: the distributed estimate, cached in the slab scheme; the global scheme
        // rebuilt on rank 0 for the evaluation would repeat the whole power iteration serially).
        const double high_order_step = sp->high_order_step();
#ifdef M1PN_MPI
        if (mpi) {
            wall = mpi->max(wall);
            const double ts = mpi->max(mpi->t_solve), tx = mpi->max(mpi->t_exchange), tw = mpi->max(mpi->t_wait);
            std::ostringstream m;
            m << ",\"mpi\":{\"ranks\":" << mpi_size << ",\"halo\":" << args.mpi_halo << ",\"local_nodes_rank0\":"
              << sp->nodes << ",\"t_axis_solve_max\":" << ts << ",\"t_exchange_max\":" << tx
              << ",\"t_pipeline_wait_max\":" << tw << ",\"profile_rank0\":{\"solves\":" << mpi->n_solves
              << ",\"setup\":" << mpi->t_setup << ",\"loop\":" << mpi->t_loop << ",\"worker_compute_sum\":" << mpi->t_exec
              << ",\"threads\":" << omp_get_max_threads() << "}}";
            mpi_json = m.str();
            setup_seconds = mpi->max(setup_seconds);
            Field global = mpi->gather(a);
            mpi.reset(); // detaches from the local scheme
            if (mpi_rank != 0)
                return 0;
            const double r0 = seconds();
            holder = std::make_unique<ElementWeno>(pn, options, inflow); // serial global scheme for the evaluation
            rebuild_seconds = seconds() - r0;
            sp = holder.get();
            a = std::move(global);
        }
#endif
        if (!args.dump_state.empty()) {
            std::ofstream f(args.dump_state, std::ios::binary);
            const long rows = a.rows(), cols = a.cols();
            f.write(reinterpret_cast<const char *>(&rows), sizeof rows);
            f.write(reinterpret_cast<const char *>(&cols), sizeof cols);
            f.write(reinterpret_cast<const char *>(a.data()), std::streamsize(sizeof(double) * rows * cols));
        }
        double error = -1;
        if (smooth) {
            const double final_time = t;
            error = sp->l2_density_error(a, [&](const V3 &x) {
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
                    const double d = project(sp->evaluate(a, x))[0] - rho(x);
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
        } else if (!smooth && args.dimension == 2) {
            const double w = args.width, bg = args.background, T = t;
            const V3 c1(.3 + T, .5, 0), c2(.5, .3 + T, 0);
            auto rho = [&](const V3 &x) {
                auto p = [w](const V3 &d) { return std::exp(-(d[0] * d[0] + d[1] * d[1]) / (2 * w * w)); };
                return bg + p(x - c1) + p(x - c2);
            };
            kinetic_error = sp->l2_density_error(a, rho);
            auto line = [w](double c) {
                return w * std::sqrt(pi / 2) * (std::erf((1 - c) / (std::sqrt(2.) * w)) - std::erf(-c / (std::sqrt(2.) * w)));
            };
            exact_mass = bg + line(c1[0]) * line(c1[1]) + line(c2[0]) * line(c2[1]);
            // Error restricted to the outflow strip max(x, y) > 1 - 4 width (where the ghost acts).
            const double edge = 1 - 4 * w;
            const double inner = std::pow(sp->l2_density_error(a, [&](const V3 &x) {
                return std::max(x[0], x[1]) > edge ? rho(x) : project(sp->evaluate(a, x))[0];
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
                        const Four u = project(sp->evaluate(a, x));
                        csv << x[0] << ',' << x[1] << ',' << x[2] << ',' << u[0] << ',' << u[1] << ',' << u[2] << ',' << u[3] << '\n';
                    }
        }
        double min_coefficient_density = std::numeric_limits<double>::infinity();
        double min_coefficient_margin = min_coefficient_density;
        for (int i = 0; i < sp->nodes; ++i) {
            const Four u = project(a.row(i).transpose());
            min_coefficient_density = std::min(min_coefficient_density, u[0]);
            min_coefficient_margin = std::min(min_coefficient_margin, margin(u));
        }
        double total = 0;
        for (int i = 0; i < sp->nodes; ++i)
            total += sp->lumped[i] * project(a.row(i).transpose())[0];
        std::ostringstream summary;
        summary << std::setprecision(12) << "{\"problem\":\"" << args.problem << "\",\"pn_order\":" << args.pn
                << ",\"degree\":" << args.degree << ",\"elements\":" << args.elements
                << ",\"dimension\":" << args.dimension << ",\"steps\":" << steps << ",\"dt\":" << dt
                << ",\"threads\":" << m1pn::default_threads().threads << ",\"thread_policy\":\""
                << m1pn::default_threads().reason << "\""
                << ",\"dt_bound\":" << sp->dt_bound << ",\"high_order_step\":" << high_order_step << ",\"wall_seconds\":" << wall
                << ",\"ghost_filter\":" << args.ghost_filter << ",\"l2_density_error\":" << error << ",\"final_mass\":" << total
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
                << ",\"seconds\":{\"closure\":" << stats.t_closure << ",\"closure_setup\":" << stats.t_closure_setup << ",\"advance\":" << stats.t_advance << ",\"rate\":" << stats.t_rate << ",\"finite\":" << stats.t_finite << ",\"ssp_stage\":" << stats.t_ssp_stage << ",\"weno\":" << stats.t_weno
                << ",\"stabilization\":" << stats.t_stabilization << ",\"residual\":" << stats.t_residual
                << ",\"boundary\":" << stats.t_boundary << ",\"mass\":" << stats.t_mass
                << ",\"limiter\":" << stats.t_limiter << ",\"gradient_local\":" << stats.t_gradient_local
                << ",\"gradient_gather\":" << stats.t_gradient_gather << ",\"gradient_solve\":" << stats.t_gradient_solve
                << ",\"stab_local\":" << stats.t_stab_local << ",\"finish\":" << stats.t_finish
                << ",\"ghost\":" << stats.projection.seconds << ",\"boundary_faces\":" << stats.projection.face_seconds
                << ",\"boundary_products\":" << stats.projection.product_seconds
                << ",\"boundary_reduce\":" << stats.projection.reduce_seconds << "}";
        if (args.positivity_projection) {
            const auto &p = stats.projection;
            summary << ",\"projection\":{\"rule\":" << args.projection_rule << ",\"calls\":" << p.calls
                    << ",\"identities\":" << p.identities << ",\"failures\":" << p.failures
                    << ",\"vacuum\":" << p.vacuum << ",\"scaled\":" << p.scaled << ",\"capped\":" << p.capped
                    << ",\"augmented\":" << p.augmented << ",\"certificates\":" << p.certificates
                    << ",\"diracs\":" << p.diracs << ",\"farkas\":" << p.farkas
                    << ",\"warm_start\":" << (args.projection_warm_start ? "true" : "false")
                    << ",\"incoming_weight\":" << args.projection_incoming_weight
                    << ",\"mode\":\"" << (args.projection_half_range ? "half" : args.projection_hybrid ? "hybrid" : "full")
                    << "\",\"hybrid_kept\":" << p.hybrid_kept
                    << ",\"iterations\":" << p.iterations << ",\"backtracks\":" << p.backtracks
                    << ",\"max_iterations\":" << p.max_iterations << ",\"max_residual\":" << p.max_residual
                    << ",\"min_active_fraction\":" << p.min_active_fraction;
            if (args.projection_diagnostics) {
                summary << ",\"c6_scaled\":" << p.c6_scaled << ",\"c6_diff_mean\":" << p.c6_diff_sum / std::max(1L, p.calls)
                        << ",\"c6_diff_max\":" << p.c6_diff_max << ",\"c6_scaled_diff_share\":"
                        << p.c6_scaled_diff / std::max(1e-300, p.c6_diff_sum) << ",\"c6_diff_side\":[";
                for (int s = 0; s < 6; ++s)
                    summary << (s ? "," : "") << p.c6_diff_side[s] / std::max(1e-300, p.c6_diff_sum);
                summary << "]";
            }
            summary << "}";
        }
        if (args.ghost_study > 0) {
            static const char *names[E] = {"c6", "full", "half", "hybrid", "c6_filtered"};
            summary << ",\"ghost_study\":{\"checkpoints\":" << args.ghost_study << ",\"pairs\":" << study_points
                    << ",\"filter\":" << args.ghost_study_filter << ",\"exact_outgoing_mass\":" << study_scale;
            for (int e = 0; e < E; ++e)
                summary << ",\"" << names[e] << "\":{\"rho\":" << study_err[e][0] / study_scale << ",\"nJ\":"
                        << study_err[e][1] / study_scale << ",\"J\":" << study_err[e][2] / study_scale
                        << ",\"rho_minus_nJ\":" << study_err[e][3] / study_scale << "}";
            summary << "}";
        }
        summary << ",\"phases\":{\"setup\":" << setup_seconds << ",\"post\":" << seconds() - post_start
                << ",\"post_rebuild\":" << rebuild_seconds << "}";
#ifdef M1PN_ELEMENT_GPU
        if (device)
            summary << ",\"gpu\":{\"spectral_radius_seconds\":" << gpu_radius_seconds << ",\"host_boundary\":"
                    << device->t_boundary << ",\"host_boundary_high\":" << device->t_boundary_high << ",\"host_face_data\":" << device->t_face_data << ",\"host_correction\":" << device->t_correction << ",\"host_fallback\":"
                    << device->t_fallback << ",\"wait\":" << device->t_wait << ",\"sync\":" << device->t_sync
                    << ",\"closure_fallbacks\":" << device->closure_fallbacks << ",\"limiter_fallbacks\":"
                    << device->limiter_fallbacks << ",\"ghost_flagged\":" << device->ghost_flagged << ",\"wait_ghost\":"
                    << device->t_wait_ghost << "}";
#endif
        summary << mpi_json << "}";
        std::ofstream(args.output + "/summary.json") << summary.str() << "\n";
        std::cout << summary.str() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "error (rank " << mpi_rank << "): " << e.what() << "\n";
#ifdef M1PN_MPI
        if (mpi_size > 1)
            MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
    return 0;
}
