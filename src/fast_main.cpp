// m1pn_fast: the standard M1PN program, element WENO on tensor Bernstein elements (docs/ELEMENT_WENO.md), without
// MFEM. Cases on generated Cartesian hex meshes (crossing, line/plane source with optional first-collision source,
// lattice-paul, lattice-3d, shadow-weih, kobayashi, smooth and beam test cases), MPI slabs (docs/ELEMENT_WENO_MPI.md), GPU time
// loop with --gpu (docs/ELEMENT_WENO_GPU.md). The former MFEM paths (mass correction, biharmonic, NN filter) were
// removed on 2026-11-16; last state with them: tag standard-20261116.
#include "angular.hpp"
#include "boundary.hpp"
#include "uncollided.hpp"
#include "uncollided_lattice.hpp"
#include "uncollided_lattice3d.hpp"
#include "uncollided_box3d.hpp"
#ifdef M1PN_ELEMENT_GPU
#include "element_weno_gpu.hpp"
#endif
#include "element_weno.hpp"
#include "element_weno_mpi.hpp"
#include "hardware.hpp"
#include "benchmark_cases.hpp"
#include "parallel.hpp"
#include <memory>
#include <filesystem>
#include <malloc.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <sys/resource.h>
#include <omp.h>
using namespace m1pn;
namespace {
// Default CFL (dt = cfl * step_bound). CFL scan of 2026-11-16 (docs/ELEMENT_WENO.md): stable up to 1 in all tested
// cases (Q1-Q4 crossing, smooth, lattice 2D/3D, line source + FCS); the time error at CFL 1 stays far below the
// spatial error, and CFL 1 is ~5x faster than 0.2. Accuracy studies at round-off level pass a smaller --cfl.
constexpr double kDefaultCfl = 1.0;
struct Mesh {
    int dimension = 3, elements = 2;
    bool periodic = false;
};
struct Options {
    std::string filter_mode = "off";
    std::string angular_reduction = "auto"; // auto|on|off (parity reduction in 1D/2D)
    std::string mirror_reduction = "auto";
    double shadow_sigma = 1e7, beam_density = 1;
    int material_subcells = 0, history_stride = 0;
    int break_every = 0; // broken lines at every k-th element line of each axis (docs/STATIC_HP_BLOCKS.md)
    // EXPERIMENT (experiment/break-dt, docs/BREAK_DT.md): "high-order" selects dt = cfl * high_order_step and drops
    // the a-priori realizability bound (Theorem 3.1 then no longer guarantees the stage states).
    std::string dt_bound = "both";
    // EXPERIMENT: with --break-units on, only these unit lines (coordinates, both axes) are broken; empty: all.
    std::vector<int> break_at;
    // EXPERIMENT: graded mesh with broken unit lines: the element on each side of a broken line gets `break_widen`
    // times the width (that many cells merged), so the one-sided lumped mass of the line nodes grows.
    int break_widen = 1;
    bool split_breaks = false; // EXPERIMENT: two convex steps per stage at broken-line nodes (docs/BREAK_DT.md)
    bool break_units = false; // broken lines on the unit lines of the box (lattices: the material lines)
    std::vector<int> cells_per_unit; // graded mesh: elements per unit interval of each axis (docs/GRADED_MESH.md)
    bool skip_fields = false;
    double filter_strength = 0;
    Mesh mesh;
    Budget budget;
    int order = 3, rk = 4, max_steps = 100000, threads = 1;
    bool threads_explicit = false;
    std::string thread_policy = "OpenMP maximum";
    double final_time = .01, dt = 0, cfl = kDefaultCfl, sigma_a = 0, sigma_s = 0;
    double pulse_width = .07, background = .001;
    double plane_width = .05; // plane-source Gaussian width (peak one); default exp(-200 (x - 1/2)^2)
    int degree = 2;
    double weno_linear = 1e-3, weno_q = 1, weno_deadband = 0, weno_floor = 0;
    bool m1_target = false;               // M1 instead of PN target (P1, --mirror-reduction off)
    std::string sample_moments = "auto";  // moments written to initial/final.csv
    bool gpu = false;                     // time loop on the GPU (src/element_weno_gpu.cpp)
    std::string fcs = "off";              // first-collision source (docs/FCS.md, docs/FCS_LATTICE.md)
    bool weno = true, element_limit = true;
    std::string problem = "smooth", inflow = "vacuum", output = "tmp/run";
};
Options parse(int argc, char **argv) {
    Options o;
#ifdef _OPENMP
    o.threads = omp_get_max_threads();
#endif
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help") {
            std::cout
                << "m1pn_fast --pn-order N --elements n --dimension 1|2|3 [--periodic] --degree p --case C --inflow I\n"
                   " --time T --dt DT --cfl C (default 1) --time-integrator ssprk43|ssprk3 (default ssprk43)\n"
                   " --case smooth|stress|isotropic|vacuum|pure-beam|beam|crossing|plane-source|line-source|\n"
                   "   smooth-beam|vacuum-beam|oblique-beam|crossing-three|crossing-study|crossing-3d|\n"
                   "   lattice-paul (2D, (0,7)^2)|lattice-3d (3D, (0,7)^3)|shadow-weih (3D, --inflow shadow-beam)|\n"
                   "   kobayashi-1i|1ii|2i|2ii (3D, cm, --elements n per 200 cm, n divisible by 20; docs/KOBAYASHI.md)\n"
                   " --inflow vacuum|isotropic|time-dependent|beam|crossing|matching|smooth-transport|crossing-transport|\n"
                   "   crossing-three-transport|crossing-study-transport|crossing-3d-transport|shadow-beam\n"
                   " --pulse-width W --plane-width W --background B --sigma-a A --sigma-s S --memory-mib M --threads T --max-steps K\n"
                   " --weno on|off --weno-linear W --weno-q Q --weno-deadband D --weno-floor F (study, CPU) --element-limit on|off --filter off|constant --filter-strength S\n"
                   " --angular-reduction auto|on|off (1D/2D: drop PN moments odd in the unresolved directions)\n"
                   " --mirror-reduction auto|off (half/quarter/octant domain for mirror-symmetric cases)\n"
                   " --shadow-sigma S --beam-density B --material-subcells K --history-stride S --skip-fields\n"
                   " --cells-per-unit n0,n1,... (graded mesh, 1D/2D, serial CPU: every axis is split into as many equal\n"
                   "   segments as entries, entry k = elements in segment k; e.g. lattice: 7 unit segments)\n"
                   " --sample-moments auto|all|low --fcs off|analytic --m1-target on|off --gpu --output DIR\n"
                   " compatibility: --stabilization element-weno, --model r2, --mesh-type hex are accepted\n"
                   "threads: physical cores of one NUMA node unless --threads or OMP_NUM_THREADS is set\n";
            std::exit(0);
        }
        if (key == "--periodic") {
            o.mesh.periodic = true;
            continue;
        }
        if (key == "--gpu") {
            o.gpu = true;
            continue;
        }
        if (key == "--skip-fields") {
            o.skip_fields = true;
            continue;
        }
        if (key == "--deterministic") // accepted for compatibility (element WENO is deterministic)
            continue;
        require(i + 1 < argc, "Missing value for " + key);
        std::string v = argv[++i];
        auto real = [&]() {
            std::size_t used = 0;
            double x = std::stod(v, &used);
            require(used == v.size() && std::isfinite(x), "Invalid number for " + key);
            return x;
        };
        auto integer = [&]() {
            double x = real();
            require(x == std::floor(x) && x >= 0 && x <= std::numeric_limits<int>::max(),
                    "Invalid integer for " + key);
            return int(x);
        };
        if (key == "--filter") {
            require(v == "off" || v == "constant", "--filter must be off or constant");
            o.filter_mode = v;
        } else if (key == "--filter-strength")
            o.filter_strength = real();
        else if (key == "--pn-order")
            o.order = integer();
        else if (key == "--elements")
            o.mesh.elements = integer();
        else if (key == "--dimension")
            o.mesh.dimension = integer();
        else if (key == "--mesh-type")
            require(v == "hex", "Only --mesh-type hex (generated Cartesian mesh)");
        else if (key == "--model")
            require(v == "r2", "Only --model r2");
        else if (key == "--stabilization")
            require(v == "element-weno", "Only --stabilization element-weno (mass correction and biharmonic were "
                                         "removed; tag standard-20261116)");
        else if (key == "--time-integrator") {
            require(v == "ssprk3" || v == "ssprk43", "--time-integrator must be ssprk3 or ssprk43");
            o.rk = v == "ssprk3" ? 3 : 4;
        } else if (key == "--case")
            o.problem = v;
        else if (key == "--inflow")
            o.inflow = v;
        else if (key == "--output")
            o.output = v;
        else if (key == "--time")
            o.final_time = real();
        else if (key == "--dt")
            o.dt = real();
        else if (key == "--dt-bound")
            o.dt_bound = v; // both | high-order (experiment)
        else if (key == "--split-breaks") // experiment: on|off
            o.split_breaks = v == "on";
        else if (key == "--break-widen") // experiment: merge that many cells next to each broken line
            o.break_widen = std::stoi(v);
        else if (key == "--break-at") { // experiment: comma list of unit lines
            std::stringstream list(v);
            std::string item;
            while (std::getline(list, item, ','))
                o.break_at.push_back(std::stoi(item));
        }
        else if (key == "--break-units")
            o.break_units = v == "on";
        else if (key == "--break-every")
            o.break_every = integer();
        else if (key == "--cfl")
            o.cfl = real();
        else if (key == "--sigma-a")
            o.sigma_a = real();
        else if (key == "--sigma-s")
            o.sigma_s = real();
        else if (key == "--memory-mib") {
            double mib = real();
            require(mib > 0 && mib < 1e7, "Invalid memory budget");
            o.budget.bytes = std::size_t(mib * 1024 * 1024);
        } else if (key == "--max-steps")
            o.max_steps = integer();
        else if (key == "--pulse-width")
            o.pulse_width = real();
        else if (key == "--plane-width")
            o.plane_width = real();
        else if (key == "--background")
            o.background = real();
        else if (key == "--threads") {
            o.threads = integer();
            o.threads_explicit = true;
            o.thread_policy = "--threads";
        } else if (key == "--degree")
            o.degree = integer();
        else if (key == "--weno-linear")
            o.weno_linear = real();
        else if (key == "--shadow-sigma") {
            o.shadow_sigma = std::stod(v);
            require(o.shadow_sigma >= 0 && std::isfinite(o.shadow_sigma), "Invalid --shadow-sigma");
        } else if (key == "--beam-density") {
            o.beam_density = std::stod(v);
            require(o.beam_density >= 0 && std::isfinite(o.beam_density), "Invalid --beam-density");
        } else if (key == "--cells-per-unit") {
            std::stringstream list(v);
            for (std::string item; std::getline(list, item, ',');) {
                o.cells_per_unit.push_back(std::stoi(item));
                require(o.cells_per_unit.back() >= 1, "Invalid --cells-per-unit");
            }
        } else if (key == "--material-subcells") {
            o.material_subcells = std::stoi(v);
            require(o.material_subcells >= 0, "Invalid --material-subcells");
        } else if (key == "--history-stride")
            o.history_stride = std::stoi(v);
        else if (key == "--mirror-reduction") {
            require(v == "auto" || v == "off", "Invalid --mirror-reduction");
            o.mirror_reduction = v;
        } else if (key == "--angular-reduction") {
            require(v == "auto" || v == "on" || v == "off", "Invalid --angular-reduction");
            o.angular_reduction = v;
        } else if (key == "--weno-q")
            o.weno_q = real();
        else if (key == "--weno-deadband")
            o.weno_deadband = real();
        else if (key == "--weno-floor")
            o.weno_floor = real();
        else if (key == "--sample-moments") {
            require(v == "auto" || v == "all" || v == "low", "--sample-moments must be auto, all or low");
            o.sample_moments = v;
        } else if (key == "--fcs") {
            require(v == "off" || v == "analytic", "--fcs must be off or analytic");
            o.fcs = v;
        } else if (key == "--m1-target") {
            require(v == "on" || v == "off", "--m1-target must be on or off");
            o.m1_target = v == "on";
        } else if (key == "--weno" || key == "--element-limit") {
            require(v == "on" || v == "off", key + " must be on or off");
            (key == "--weno" ? o.weno : o.element_limit) = v == "on";
        } else
            throw std::runtime_error("Unknown option: " + key);
    }
    require(o.filter_strength >= 0, "Negative filter strength");
    require(o.degree >= 1 && o.degree <= 12, "degree must lie in [1,12]");
    require(o.weno_linear > 0 && o.weno_linear < .5 && o.weno_q >= 1, "Invalid WENO weights");
    require(o.weno_deadband >= 0 && o.weno_deadband < 1 && o.weno_floor >= 0, "Invalid --weno-deadband/--weno-floor");
    require(!o.gpu || (o.weno_deadband == 0 && o.weno_floor == 0), "--weno-deadband/--weno-floor: CPU only");
    require(o.final_time >= 0 && o.dt >= 0 && o.cfl > 0 && o.cfl <= 1 && o.threads > 0 && o.max_steps > 0,
            "Invalid time/thread configuration");
    require(o.sigma_a >= 0 && o.sigma_s >= 0, "Invalid reaction coefficients");
    require(o.pulse_width > 0 && o.plane_width > 0 && o.background >= 0, "Invalid pulse width/background");
    if (o.problem == "line-source")
        require(o.mesh.dimension == 2 && o.inflow == "vacuum", "line-source is 2D with vacuum inflow");
    return o;
}
Vec matching(int n) {
    Vec u = Vec::Zero((n + 1) * (n + 1));
    u[0] = 1 / std::sqrt(4 * pi);
    if (n > 1)
        u[index(n - 1, 0)] = .01;
    return u;
}
Inflow make_inflow(const Options &o, int dimension) {
    Inflow g;
    g.steady = o.inflow == "vacuum" || o.inflow == "isotropic" || o.inflow == "shadow-beam";
    if (o.inflow == "vacuum")
        return g;
    if (o.inflow == "isotropic")
        g.density = [](const V3 &, double, const V3 &) { return 1 / (4 * pi); }, g.constant_density = 1 / (4 * pi);
    else if (o.inflow == "time-dependent")
        g.density = [](const V3 &x, double t, const V3 &w) {
            return (1 + .2 * x[0] + .1 * std::sin(t)) * (1 + .2 * w[1]) / (4 * pi);
        };
    else if (o.inflow == "smooth-transport")
        g.density = [](const V3 &x, double t, const V3 &w) {
            return (1 + .2 * std::sin(2 * pi * (x[0] - w[0] * t))) / (4 * pi);
        };
    else if (o.inflow == "crossing-transport") {
        g.density = [](const V3 &, double, const V3 &) { return .01 / (4 * pi); };
        g.constant_density = .01 / (4 * pi);
        g.atoms = [z = dimension == 2 ? 0. : .5](const V3 &x, double t) {
            return std::vector<Atom>{{V3::UnitX(),std::exp(-50*(x-t*V3::UnitX()-V3(.3,.5,z)).squaredNorm())},{V3::UnitY(),std::exp(-50*(x-t*V3::UnitY()-V3(.5,.3,z)).squaredNorm())}};
        };
    } else if (o.inflow == "crossing-three-transport") {
        const double bg=o.background, w=o.pulse_width;
        g.density = [bg](const V3 &, double, const V3 &) { return bg / (4 * pi); };
        g.constant_density = bg / (4 * pi);
        g.atoms = [z = dimension == 2 ? 0. : .5, w](const V3 &x, double t) {
            auto pulse=[w](const V3 &d){return std::exp(-d.squaredNorm()/(2*w*w));};
            return std::vector<Atom>{
                {V3::UnitX(), pulse(x-t*V3::UnitX()-V3(.32,.5,z))},
                {V3::UnitY(), pulse(x-t*V3::UnitY()-V3(.5,.32,z))},
                {-V3::UnitX(), pulse(x+t*V3::UnitX()-V3(.68,.5,z))}};
        };
    } else if (o.inflow == "crossing-study-transport" || o.inflow == "crossing-3d-transport") {
        const bool three=o.inflow=="crossing-3d-transport"; const double bg=o.background,w=o.pulse_width;
        g.density=[bg](const V3&,double,const V3&){return bg/(4*pi);}; g.constant_density=bg/(4*pi);
        g.atoms=[three,w](const V3&x,double t){auto p=[w](const V3&d){return std::exp(-d.squaredNorm()/(2*w*w));};
            std::vector<Atom> a{{V3::UnitX(),p(x-t*V3::UnitX()-V3(.3,.5,three?.5:0))},{V3::UnitY(),p(x-t*V3::UnitY()-V3(.5,.3,three?.5:0))}};
            if(three)a.push_back({V3::UnitZ(),p(x-t*V3::UnitZ()-V3(.5,.5,.3))}); return a;};
    } else if (o.inflow == "shadow-beam") {
        // Collimated beam along +x with the radial profile of cases::shadow_beam_profile (peak beam_density); it
        // enters only through x = -0.75 (w.n < 0 there, w.n = 0 on the side faces, where the profile is 0);
        // isotropic background density o.background elsewhere.
        const double bg = o.background, rho_b = o.beam_density;
        g.density = [bg](const V3 &, double, const V3 &) { return bg / (4 * pi); };
        g.constant_density = bg / (4 * pi);
        g.atoms = [rho_b](const V3 &x, double) {
            return std::vector<Atom>{{V3::UnitX(), rho_b * cases::shadow_beam_profile(x[1], x[2])}};
        };
    } else if (o.inflow == "matching") {
        Vec u = matching(o.order);
        g.density = [u, n = o.order](const V3 &, double, const V3 &w) {
            return harmonics(n, w).dot(u);
        };
    } else if (o.inflow == "beam" || o.inflow == "crossing")
        g.atoms = [cross = o.inflow == "crossing"](const V3 &x, double t) {
            std::vector<Atom> a = {
                {V3(1, .1, .05).normalized(),
                 (1 + .1 * std::sin(t)) *
                     std::exp(-30 * ((x[1] - .4) * (x[1] - .4) + (x[2] - .5) * (x[2] - .5)))}};
            if (cross)
                a.push_back(
                    {V3(.1, 1, -.05).normalized(),
                     std::exp(-30 * ((x[0] - .6) * (x[0] - .6) + (x[2] - .5) * (x[2] - .5)))});
            return a;
        };
    else
        throw std::runtime_error("Unknown inflow: " + o.inflow);
    return g;
}
Vec initial_point(const Options &o, const V3 &x, int i, int dimension, const Angular &pn) {
    {
        Vec a = Vec::Zero(pn.q);
        if (o.problem == "vacuum") {
        } else if (o.problem == "isotropic")
            a = matching(pn.order);
        else if (o.problem == "smooth")
            a[0] = (1 + .2 * std::sin(2 * pi * x[0])) / std::sqrt(4 * pi);
        else if (o.problem == "smooth-beam" || o.problem == "vacuum-beam" ||
                 o.problem == "oblique-beam") {
            double rho = 1 + .2 * std::sin(2 * pi * x[0]);
            if (o.problem == "vacuum-beam")
                rho = std::pow(std::max(std::cos(2 * pi * (x[0] - .35)), 0.), 6);
            const V3 direction = o.problem == "oblique-beam" ? V3(3. / 5, 4. / 5, 0)
                                                              : V3::UnitX();
            a = rho * harmonics(pn.order, direction);
            set_low(a, Four(rho, rho * direction[0], rho * direction[1], rho * direction[2]));
        }
        else if (o.problem == "stress") {
            Four v(1 + .2 * x[0] + .1 * x[1] + .05 * x[2], .2, .1, .05);
            set_low(a, v);
            for (int k = 4; k < pn.q; ++k)
                a[k] = .15 * std::sin((i + 1.) * (k + 3.) * 1.618);
        } else if (o.problem == "pure-beam") {
            // This fixed physical beam mass has an exactly admissible rounded lift.
            // Mass one does not: its rounded rho is below its rounded Jx in this basis.
            a = 1.5 * harmonics(pn.order, V3::UnitX());
            set_low(a, Four(1.5, 1.5, 0, 0));
        } else if (o.problem == "beam" || o.problem == "crossing") {
            const double z = dimension == 2 && o.problem == "crossing" ? 0. : .5;
            a[0] = .01 / std::sqrt(4 * pi);
            a += std::exp(-50 * (x - V3(.3, .5, z)).squaredNorm()) *
                 harmonics(pn.order, V3::UnitX());
            if (o.problem == "crossing")
                a += std::exp(-50 * (x - V3(.5, .3, z)).squaredNorm()) *
                     harmonics(pn.order, V3::UnitY());
        } else if (o.problem == "crossing-study" || o.problem == "crossing-3d") {
            const bool three=o.problem=="crossing-3d"; const double z=three?.5:0.;
            auto p=[&](const V3&d){return std::exp(-d.squaredNorm()/(2*o.pulse_width*o.pulse_width));};
            a[0]=o.background/std::sqrt(4*pi);
            a += p(x-V3(.3,.5,z))*harmonics(pn.order,V3::UnitX());
            a += p(x-V3(.5,.3,z))*harmonics(pn.order,V3::UnitY());
            if(three)a += p(x-V3(.5,.5,.3))*harmonics(pn.order,V3::UnitZ());
        } else if (o.problem == "crossing-three") {
            const double z = dimension == 2 ? 0. : .5;
            auto pulse=[&](const V3 &d){return std::exp(-d.squaredNorm()/(2*o.pulse_width*o.pulse_width));};
            a[0] = o.background / std::sqrt(4 * pi);
            a += pulse(x-V3(.32,.5,z)) * harmonics(pn.order, V3::UnitX());
            a += pulse(x-V3(.5,.32,z)) * harmonics(pn.order, V3::UnitY());
            a += pulse(x-V3(.68,.5,z)) * harmonics(pn.order, -V3::UnitX());
        } else if (o.problem == "plane-source")
            a[0] = (o.plane_width == .05 ? std::exp(-200 * (x[0] - .5) * (x[0] - .5))
                                         : std::exp(-(x[0] - .5) * (x[0] - .5) / (2 * o.plane_width * o.plane_width))) /
                   std::sqrt(4 * pi);
        else if (o.problem == "line-source") {
            // Garrett-Hauck line source: unit-mass Gaussian of width W plus floor B.
            const double r2 = (x[0] - .5) * (x[0] - .5) + (x[1] - .5) * (x[1] - .5);
            const double w2 = o.pulse_width * o.pulse_width;
            a[0] = (o.background + std::exp(-r2 / (2 * w2)) / (2 * pi * w2)) / std::sqrt(4 * pi);
        }
        else if (o.problem == "shadow" || o.problem == "lattice" || o.problem == "lattice-paul")
            a[0] = (o.problem=="lattice-paul"?o.background:.01) / std::sqrt(4 * pi);
        else if (o.problem == "lattice-3d" || o.problem == "shadow-weih" || o.problem.rfind("kobayashi-", 0) == 0)
            a[0] = o.background / std::sqrt(4 * pi); // isotropic background, no initial beam
        else
            throw std::runtime_error("Unknown case: " + o.problem);
        return a;
    }
}
std::string beam_metadata(const Options &o,int dimension) {
    const bool n2=o.problem=="crossing-study"||o.inflow=="crossing-study-transport";
    const bool n3=o.problem=="crossing-3d"||o.inflow=="crossing-3d-transport";
    const bool l3=o.problem=="crossing-three"||o.inflow=="crossing-three-transport";
    const bool l2=o.problem=="crossing"||o.inflow=="crossing-transport";
    const int count=(n3||l3)?3:((n2||l2)?2:0); std::ostringstream out; out<<std::setprecision(17)<<",\"beam_count\":"<<count;
    if(count){const bool modern=n2||n3;const double z=dimension==2?0.:.5,w=modern||l3?o.pulse_width:std::sqrt(1./100.),bg=modern||l3?o.background:.01;
        const double cx=l3?.32:.3, cy=l3?.32:.3;
        out<<",\"pulse_width\":"<<w<<",\"background\":"<<bg<<",\"beam_geometry\":{\"background\":"<<bg<<",\"amplitude\":1,\"width_exponent\":"<<-1/(2*w*w)<<",\"beams\":["
           <<"{\"center\":["<<cx<<",0.5,"<<z<<"],\"direction\":[1,0,0]},{\"center\":[0.5,"<<cy<<","<<z<<"],\"direction\":[0,1,0]}";
        if(n3)out<<",{\"center\":[0.5,0.5,0.3],\"direction\":[0,0,1]}";else if(l3)out<<",{\"center\":[0.68,0.5,"<<z<<"],\"direction\":[-1,0,0]}";out<<"]}";
    }return out.str();
}
// Element-based limiting with CG-WENO target on tensor Bernstein elements.
// Writes the same field files as the Q1 path; the Bernstein solution is
// evaluated on a uniform sample grid (nodes of Q1 with elements*degree cells).
int run_element_weno(const Options &o, const Angular &pn, double begin, const Runtime &world) {
    const int dim = o.mesh.dimension;
    // MPI (docs/ELEMENT_WENO_MPI.md): slabs along the last axis with two halo element layers (ElementWenoMpi, bitwise
    // equal to the serial run); sums over owned nodes/elements are reduced; rank 0 writes all files from the
    // gathered state with a global scheme (`post`).
    const int ranks = world.size, rank = world.rank;
    const bool distributed = ranks > 1;
#ifndef M1PN_MPI
    require(!distributed, "element-weno with several ranks needs an MPI build");
#endif
    if (distributed) {
        require(dim >= 2 && !o.mesh.periodic, "element-weno MPI: 2D/3D open meshes (slabs along the last axis)");
        require(!o.gpu, "element-weno MPI: no --gpu (MPI+GPU is not implemented)");
    }
    require(o.problem != "lattice" && o.problem != "shadow",
            "use lattice-paul, lattice-3d or shadow-weih for heterogeneous material");
    if (o.problem == "crossing-study" || o.inflow == "crossing-study-transport")
        require(dim == 2, "crossing-study is 2D");
    // ---- Case geometry and material (src/benchmark_cases.hpp; docs/ELEMENT_HETEROGENEOUS.md) -------------------
    // Box origin + h (count_x, count_y, count_z) with cubic cells; `--elements n` is the count per unit-cube axis,
    // per length-7 axis (lattices) or per 0.5 (shadow, whose x axis then has 3n elements).
    V3 origin = V3::Zero();
    std::array<int, 3> counts{o.mesh.elements, dim > 1 ? o.mesh.elements : 1, dim > 2 ? o.mesh.elements : 1};
    double length_x = 1;
    std::shared_ptr<ElementWenoMaterial> material;
    std::string expected_integrals = "null";
    const bool lattice2d = o.problem == "lattice-paul", lattice3d = o.problem == "lattice-3d",
               shadow = o.problem == "shadow-weih";
    // Kobayashi benchmarks (docs/KOBAYASHI.md): problem 1 or 2, case i (pure absorber) or ii (scattering ratio 1/2), on
    // the full mirrored box in cm; the mirror reduction computes the benchmark octant with its reflecting faces.
    int kobayashi_problem = 0;
    bool kobayashi_scattering = false;
    if (o.problem.rfind("kobayashi-", 0) == 0) {
        const std::string v = o.problem.substr(10);
        require(v == "1i" || v == "1ii" || v == "2i" || v == "2ii", "Unknown Kobayashi case (kobayashi-1i|1ii|2i|2ii)");
        kobayashi_problem = v[0] - '0', kobayashi_scattering = v.size() == 3;
    }
    if (lattice2d || lattice3d || shadow || kobayashi_problem)
        require(o.sigma_s == 0 && o.sigma_a == 0, "This case owns its material coefficients");
    if (lattice2d) {
        require(dim == 2 && o.mesh.elements % 7 == 0, "lattice-paul is 2D with elements divisible by 7");
        require(o.inflow == "vacuum" && std::abs(o.background - 1e-10) <= 1e-16,
                "lattice-paul: vacuum inflow and Paul's initial density --background 1e-10 ((47))");
        length_x = 7;
        material = std::make_shared<ElementWenoMaterial>();
        material->sigma_a = [](const V3 &x) { return cases::lattice_paul_absorber(x[0], x[1]) ? 10. : 0.; };
        material->sigma_s = [](const V3 &x) { return cases::lattice_paul_absorber(x[0], x[1]) ? 0. : 1.; };
        material->source = [](const V3 &x) { return cases::lattice_paul_source(x[0], x[1]) ? 1. : 0.; };
        material->name = "lattice-paul (44)-(45)";
        expected_integrals = "{\"volume\":49,\"source\":1,\"sigma_a\":110,\"sigma_s\":38}";
    } else if (lattice3d) {
        require(dim == 3 && o.mesh.elements % 7 == 0, "lattice-3d is 3D with elements divisible by 7");
        require(o.inflow == "vacuum", "lattice-3d uses vacuum inflow");
        length_x = 7;
        material = std::make_shared<ElementWenoMaterial>();
        material->sigma_a = cases::lattice3d_sigma_a;
        material->sigma_s = cases::lattice3d_sigma_s;
        material->source = cases::lattice3d_source;
        material->name = "lattice-3d extension";
        const auto c = cases::lattice3d_counts();
        std::ostringstream e;
        e << std::setprecision(17) << "{\"volume\":" << c.volume << ",\"source\":" << c.source << ",\"sigma_a\":" << c.sigma_a
          << ",\"sigma_s\":" << c.sigma_s << ",\"absorbing_cells\":" << c.absorbing
          << ",\"strong_scattering_cells\":" << c.strong_scattering << ",\"scattering_cells\":" << c.scattering << "}";
        expected_integrals = e.str();
    } else if (shadow) {
        require(dim == 3, "shadow-weih is 3D");
        require(o.inflow == "shadow-beam", "shadow-weih uses --inflow shadow-beam");
        origin = V3(cases::shadow_x0, -.25, -.25);
        counts = {3 * o.mesh.elements, o.mesh.elements, o.mesh.elements};
        length_x = 1.5;
        material = std::make_shared<ElementWenoMaterial>();
        const double sigma = o.shadow_sigma;
        material->sigma_a = [sigma](const V3 &x) { return cases::shadow_inside(x) ? sigma : 0.; };
        material->name = "shadow-weih sphere";
        std::ostringstream e;
        e << std::setprecision(17) << "{\"volume\":0.375,\"source\":0,\"sigma_a\":" << sigma * 4 * pi / 3 * std::pow(.1, 3)
          << ",\"sigma_s\":0}";
        expected_integrals = e.str();
    }
    if (kobayashi_problem) {
        require(dim == 3 && o.mesh.elements % 20 == 0, "kobayashi is 3D with elements divisible by 20 (h divides 10 cm)");
        require(o.inflow == "vacuum", "kobayashi uses vacuum inflow (reflecting faces by the mirror reduction)");
        const auto half = uncollided::kobayashi::box_half(kobayashi_problem);
        const double hk = 200. / o.mesh.elements; // cm
        for (int d = 0; d < 3; ++d)
            counts[d] = int(std::lround(2 * half[d] / hk)), origin[d] = -half[d];
        length_x = 2 * half[0];
        material = std::make_shared<ElementWenoMaterial>();
        const int kp = kobayashi_problem;
        const double ratio = kobayashi_scattering ? .5 : 0.;
        using uncollided::kobayashi::Region;
        material->sigma_a = [kp, ratio](const V3 &x) {
            return (1 - ratio) * uncollided::kobayashi::sigma_t(uncollided::kobayashi::region(kp, x));
        };
        material->sigma_s = [kp, ratio](const V3 &x) {
            return ratio * uncollided::kobayashi::sigma_t(uncollided::kobayashi::region(kp, x));
        };
        material->source = [kp](const V3 &x) {
            return uncollided::kobayashi::region(kp, x) == Region::source ? 1. : 0.;
        };
        material->name = "kobayashi-" + std::to_string(kp) + (kobayashi_scattering ? "ii" : "i");
    }
    if (material)
        material->subcells = o.material_subcells > 0 ? o.material_subcells : (shadow ? 4 : 1);
    ElementWenoOptions eo;
    eo.degree = o.degree;
    eo.dimension = dim;
    eo.elements = counts[0];
    eo.elements_per_axis = {counts[0], dim > 1 ? counts[1] : 0, dim > 2 ? counts[2] : 0};
    eo.origin = origin;
    if (!o.cells_per_unit.empty()) {
        // Graded mesh: every axis of the box is split into equal segments, one per entry, with the given element
        // counts (for the lattices: unit segments, so material and source boundaries stay on element boundaries).
        require(dim <= 2 && !distributed && !o.mesh.periodic && (!o.gpu || dim == 2),
                "--cells-per-unit: 1D/2D (GPU: 2D), serial, open mesh (docs/GRADED_MESH.md)");
        const std::array<int, 3> uniform_counts = counts; // the loop overwrites counts[d]; extents use the originals
        for (int d = 0; d < dim; ++d) {
            const double extent_d = length_x / uniform_counts[0] * uniform_counts[d];
            const double segment = extent_d / double(o.cells_per_unit.size());
            auto &P = eo.planes[d];
            P.assign(1, origin[d]);
            for (std::size_t u = 0; u < o.cells_per_unit.size(); ++u)
                for (int k = 1; k <= o.cells_per_unit[u]; ++k)
                    P.push_back(origin[d] + segment * (double(u) + double(k) / o.cells_per_unit[u]));
            if (o.break_widen > 1 && o.break_units)
                for (std::size_t u = 0; u + 1 < o.cells_per_unit.size(); ++u) {
                    if (!o.break_at.empty() &&
                        std::find(o.break_at.begin(), o.break_at.end(), int(u) + 1) == o.break_at.end())
                        continue;
                    const double line = origin[d] + segment * double(u + 1);
                    const auto at = std::find_if(P.begin(), P.end(), [&](double p) { return std::abs(p - line) < 1e-12; });
                    require(at != P.end(), "--break-widen: unit line not a plane");
                    const long k = at - P.begin();
                    // drop the planes k - 1 .. k - (w - 1) and k + 1 .. k + (w - 1)
                    std::vector<double> Q;
                    for (long m = 0; m < long(P.size()); ++m)
                        if (m == k || std::abs(m - k) >= o.break_widen)
                            Q.push_back(P[m]);
                    P = std::move(Q);
                }
            counts[d] = int(P.size()) - 1;
        }
        eo.elements = counts[0];
        eo.elements_per_axis = {counts[0], dim > 1 ? counts[1] : 0, dim > 2 ? counts[2] : 0};
        eo.length = length_x;
    }
    eo.length = length_x;
    eo.material = material;
    if (o.break_every > 0 || o.break_units) {
        require(!distributed && !o.mesh.periodic && !(o.break_every > 0 && o.break_units),
                "--break-every / --break-units: serial, open mesh, one of the two (docs/STATIC_HP_BLOCKS.md)");
        require(o.break_units || o.cells_per_unit.empty(), "--break-every needs a uniform mesh (graded: --break-units)");
        for (int d = 0; d < dim; ++d) {
            if (o.break_every > 0)
                for (int k = o.break_every; k < counts[d]; k += o.break_every)
                    eo.breaks[d].push_back(k);
            else if (!o.cells_per_unit.empty()) { // graded: the segment lines (experiment: only those in --break-at)
                const auto &P = eo.planes[d];
                const double segment = (P.back() - P.front()) / double(o.cells_per_unit.size());
                for (std::size_t u = 0; u + 1 < o.cells_per_unit.size(); ++u) {
                    if (!o.break_at.empty() &&
                        std::find(o.break_at.begin(), o.break_at.end(), int(u) + 1) == o.break_at.end())
                        continue;
                    const double line = P.front() + segment * double(u + 1);
                    const auto at = std::find_if(P.begin(), P.end(), [&](double p) { return std::abs(p - line) < 1e-12; });
                    require(at != P.end(), "--break-units: unit line not a plane");
                    eo.breaks[d].push_back(int(at - P.begin()));
                }
            }
            else { // uniform: element lines at the integer coordinates of the box
                const double hd = length_x / counts[0];
                for (int k = 1; k < counts[d]; ++k) {
                    const double xk = origin[d] + k * hd;
                    if (std::abs(xk - std::round(xk)) < 1e-9 * std::max(1., std::abs(xk)))
                        eo.breaks[d].push_back(k);
                }
            }
        }
    }
    eo.periodic = o.mesh.periodic;
    eo.split_breaks = o.split_breaks;
    eo.weno = o.weno;
    eo.limit = o.element_limit;
    eo.weno_linear = o.weno_linear;
    eo.m1_target = o.m1_target;
    // ---- First-collision source (docs/FCS.md) ----------------------------------------------------------------------
    // The grid carries the collided part psi_c (initially the isotropic floor), the Gaussian pulse is uncollided and
    // known analytically in the homogeneous infinite medium; it feeds sigma_s rho_u into psi_c.
    const bool fcs = o.fcs == "analytic";
    uncollided::GaussianPulse pulse;
    // Lattice (docs/FCS_LATTICE.md): the whole source q0 = 1 on [3,4]^2 is uncollided (ray tracing through the
    // cells, tabulated in time at the FCS quadrature points); the grid gets sigma_s rho_u only.
    // lattice-3d: the same with the 3D ray tracing over the visible faces of the source cell
    // (src/uncollided_lattice3d_core.hpp); CPU only.
    const bool fcs_lattice = fcs && (lattice2d || lattice3d);
    auto lattice_table = std::make_shared<uncollided::LatticeUncollidedTable>();
    auto lattice3_table = std::make_shared<uncollided::Lattice3dUncollidedTable>();
    // Table access for the summary (2D or 3D table).
    auto table_value = [&](std::size_t k, double t) {
        return lattice3d ? lattice3_table->value(k, t) : lattice_table->value(k, t);
    };
    auto table_integral = [&](std::size_t k, double t) {
        return lattice3d ? lattice3_table->time_integral(k, t) : lattice_table->time_integral(k, t);
    };
    std::function<void(double, const std::vector<V3> &, std::vector<double> &)> sample_uncollided;
    if (fcs_lattice && lattice3d) {
        lattice3_table->lattice = uncollided::lattice3d_cells([](int i, int j, int k) {
            const V3 c(i + .5, j + .5, k + .5);
            return cases::lattice3d_sigma_a(c) + cases::lattice3d_sigma_s(c);
        });
        material->source = [](const V3 &) { return 0.; };
        material->name += ", source uncollided (FCS)";
        const std::string from = "\"source\":1,";
        const auto at = expected_integrals.find(from);
        require(at != std::string::npos, "lattice-3d FCS: unexpected material integrals");
        expected_integrals.replace(at, from.size(), "\"source\":0,");
        lattice3_table->final_time = o.final_time;
        eo.uncollided_lattice3d = lattice3_table;
        eo.uncollided_density = [lattice3_table](double t, const std::vector<V3> &x, std::vector<double> &out) {
            if (lattice3_table->points.empty())
                lattice3_table->build(x, lattice3_table->final_time);
            lattice3_table->density(t, x, out);
        };
        eo.uncollided_points = o.degree + 1;
        sample_uncollided = [lattice3_table](double t, const std::vector<V3> &x, std::vector<double> &out) {
            out.resize(x.size());
            parallel_for(int(x.size()), [&](int k) { out[k] = lattice3_table->lattice.density(x[k], t); });
        };
    } else if (fcs_lattice) {
        lattice_table->lattice = uncollided::lattice_paul(cases::lattice_paul_absorber_cell);
        material->source = [](const V3 &) { return 0.; };
        material->name += ", source uncollided (FCS)";
        expected_integrals = "{\"volume\":49,\"source\":0,\"sigma_a\":110,\"sigma_s\":38}";
        lattice_table->final_time = o.final_time;
        eo.uncollided_lattice = lattice_table;
        eo.uncollided_density = [lattice_table](double t, const std::vector<V3> &x, std::vector<double> &out) {
            if (lattice_table->points.empty())
                lattice_table->build(x, lattice_table->final_time);
            lattice_table->density(t, x, out);
        };
        eo.uncollided_points = o.degree + 1; // exact for int phi_i; the table cost scales with the point count
        sample_uncollided = [lattice_table](double t, const std::vector<V3> &x, std::vector<double> &out) {
            out.resize(x.size());
            parallel_for(int(x.size()), [&](int k) { out[k] = lattice_table->lattice.density(x[k][0], x[k][1], t); });
        };
    } else if (fcs && kobayashi_problem) {
        // Kobayashi: the whole source is uncollided; the benchmark is steady, so the collided part gets the steady
        // uncollided density (t -> infinity, src/uncollided_box3d.hpp) from t = 0 on: same steady state, no time table.
        auto box = std::make_shared<const uncollided::Box3dGeometry>(uncollided::kobayashi::geometry(kobayashi_problem));
        material->source = [](const V3 &) { return 0.; };
        material->name += ", source uncollided (FCS, steady)";
        auto cache = std::make_shared<std::pair<std::vector<V3>, std::vector<double>>>();
        eo.uncollided_density = [box, cache](double, const std::vector<V3> &x, std::vector<double> &out) {
            const bool same = cache->first.size() == x.size() &&
                              (x.empty() || ((cache->first.front() - x.front()).norm() == 0 &&
                                             (cache->first.back() - x.back()).norm() == 0));
            if (!same) {
                cache->first = x;
                cache->second.resize(x.size());
                parallel_for(int(x.size()), [&](int k) { cache->second[k] = box->density(x[k]); });
            }
            out = cache->second;
        };
        eo.uncollided_points = o.degree + 1;
        sample_uncollided = [box](double, const std::vector<V3> &x, std::vector<double> &out) {
            out.resize(x.size());
            parallel_for(int(x.size()), [&](int k) { out[k] = box->density(x[k]); });
        };
    } else if (fcs) {
        require(o.problem == "line-source" || o.problem == "plane-source", "--fcs analytic needs line-source or plane-source");
        require(!material && o.inflow == "vacuum", "--fcs analytic needs a homogeneous medium and vacuum inflow");
        pulse.sigma_t = o.sigma_a + o.sigma_s;
        if (o.problem == "line-source") {
            require(dim == 2, "--fcs analytic line-source is 2D");
            pulse.dimension = 2, pulse.width = o.pulse_width, pulse.mass = 1, pulse.centre = V3(.5, .5, 0);
        } else {
            require(dim == 1, "--fcs analytic plane-source is 1D");
            // initial_point(): rho_0 = exp(-(x - 1/2)^2 / (2 w^2)), peak one (default w = 1/20).
            pulse.dimension = 1, pulse.width = o.plane_width, pulse.mass = std::sqrt(2 * pi) * o.plane_width,
            pulse.centre = V3(.5, 0, 0);
        }
        // The analytic pulse assumes an infinite medium: its support (t + 6 width, Gaussian tail < 1e-8) must stay
        // inside the box.
        require(o.final_time + 6 * pulse.width <= .5 * length_x,
                "--fcs analytic: the uncollided front reaches the boundary before the final time");
        eo.uncollided_density = [pulse](double t, const std::vector<V3> &x, std::vector<double> &out) {
            pulse.density(t, x, out);
        };
        eo.uncollided_pulse = std::make_shared<const uncollided::GaussianPulse>(pulse);
        sample_uncollided = [pulse](double t, const std::vector<V3> &x, std::vector<double> &out) {
            pulse.density(t, x, out);
        };
    }
    eo.weno_q = o.weno_q;
    eo.weno_deadband = o.weno_deadband, eo.weno_floor = o.weno_floor;
    eo.filter_strength = o.filter_mode == "constant" ? o.filter_strength : 0.;
    eo.sigma_s = o.sigma_s;
    eo.sigma_a = o.sigma_a;
    eo.ssprk43 = o.rk == 4;
    const double h = length_x / counts[0];
    V3 extent = V3::Zero(), center = V3::Zero();
    for (int d = 0; d < dim; ++d) extent[d] = h * counts[d], center[d] = origin[d] + .5 * extent[d];
    const Inflow inflow = make_inflow(o, dim);
    // ---- Mirror reduction per direction about the box centre ----------------------------------------------------
    // Exact for mirror-symmetric problems (tests/element_weno.cpp, mirror_quarter). Required per direction d:
    // even element count, open mesh, initial data symmetric at every node of the full grid (reflected PN moments),
    // material sigma_a, sigma_s, q0 symmetric at every point of the material quadrature (or a dense sample), and
    // inflow data symmetric: density(Rx, t, R w) = density(x, t, w) and atoms(Rx, t) = R atoms(x, t) on all faces.
    std::array<bool, 3> mirror_dir{};
    std::array<std::string, 3> mirror_reason{"", "", ""};
    const std::array<std::vector<double>, 3> full_planes = eo.planes; // graded: planes of the full box (samples)
    double mirror_asymmetry = 0;
    if (distributed && o.mirror_reduction == "auto")
        mirror_reason = {"MPI", "MPI", "MPI"}; // mirror sides and slabs are not combined
    // Graded meshes and unit broken lines (docs/STATIC_HP_BLOCKS.md, "Spiegelreduktion"): additionally the planes and
    // the broken lines must be symmetric about the centre, and no broken line may lie on the mirror plane.
    if (o.break_every > 0)
        mirror_reason = {"broken lines (--break-every)", "broken lines (--break-every)", "broken lines (--break-every)"};
    if (o.mirror_reduction == "auto" && !o.mesh.periodic && !distributed && o.break_every == 0) {
        const int pd = o.degree;
        std::array<int, 3> per{1, 1, 1};
        long total = 1;
        for (int d = 0; d < dim; ++d) per[d] = counts[d] * pd + 1, total *= per[d];
        auto node_x = [&](long i) {
            V3 x = V3::Zero();
            long k = i;
            for (int d = 0; d < dim; ++d) {
                x[d] = origin[d] + extent[d] * double(k % per[d]) / (per[d] - 1);
                k /= per[d];
            }
            return x;
        };
        // Material sample: composite quadrature points of the material rule (capped; above the cap a deterministic
        // pseudo-random sample of 2e6 points is used and recorded).
        std::vector<V3> material_points;
        std::string material_check = "none";
        if (material) {
            const auto [gx, gw] = gauss(o.degree + 4);
            const int sub = material->subcells, pts = int(gx.size()) * sub;
            long elements_total = 1;
            for (int d = 0; d < dim; ++d) elements_total *= counts[d];
            long points_total = elements_total;
            for (int d = 0; d < dim; ++d) points_total *= pts;
            auto point = [&](long e, long qp) {
                V3 x = V3::Zero();
                for (int d = 0; d < dim; ++d) {
                    const int ie = int(e % counts[d]), iq = int(qp % pts);
                    e /= counts[d], qp /= pts;
                    x[d] = origin[d] + h * (ie + (iq / int(gx.size()) + (gx[iq % gx.size()] + 1) / 2) / sub);
                }
                return x;
            };
            long per_element = points_total / elements_total;
            if (points_total <= 20000000) {
                material_check = "full_quadrature";
                for (long e = 0; e < elements_total; ++e)
                    for (long qp = 0; qp < per_element; ++qp) material_points.push_back(point(e, qp));
            } else {
                material_check = "sampled_2e6";
                std::mt19937_64 rng(2026);
                std::uniform_int_distribution<long> pe(0, elements_total - 1), pq(0, per_element - 1);
                for (int k = 0; k < 2000000; ++k) material_points.push_back(point(pe(rng), pq(rng)));
            }
        }
        const std::vector<double> times{0., .25 * o.final_time, .5 * o.final_time, o.final_time};
        for (int d = 0; d < dim; ++d) {
            auto reflect = [&](V3 x) { x[d] = 2 * center[d] - x[d]; return x; };
            if (counts[d] % 2) { mirror_reason[d] = "odd element count"; continue; }
            if (!eo.planes[d].empty()) {
                const auto &P = eo.planes[d];
                const double c = P[counts[d] / 2], tol = 1e-12 * std::max(1., std::abs(c));
                bool even = true;
                for (int k = 0; k <= counts[d]; ++k)
                    even = even && std::abs(P[k] + P[counts[d] - k] - 2 * c) <= tol;
                if (!even || std::abs(c - center[d]) > tol) { mirror_reason[d] = "graded planes not symmetric"; continue; }
            }
            if (!eo.breaks[d].empty()) {
                const auto &K = eo.breaks[d];
                bool even = !std::binary_search(K.begin(), K.end(), counts[d] / 2);
                for (int k : K)
                    even = even && std::binary_search(K.begin(), K.end(), counts[d] - k);
                if (!even) { mirror_reason[d] = "broken lines not symmetric or on the mirror plane"; continue; }
            }
            double asym = 0, scale = 0;
            for (long i = 0; i < total; ++i) {
                const V3 x = node_x(i);
                const Vec u = initial_point(o, x, int(i), dim, pn);
                scale = std::max(scale, u.cwiseAbs().maxCoeff());
                asym = std::max(asym, (reflect_pn(pn, initial_point(o, reflect(x), int(i), dim, pn), d) - u).cwiseAbs().maxCoeff());
            }
            if (asym > 1e-14 * std::max(1., scale)) { mirror_reason[d] = "initial data"; mirror_asymmetry = std::max(mirror_asymmetry, asym); continue; }
            bool symmetric = true;
            if (material)
                for (const V3 &x : material_points) {
                    const V3 y = reflect(x);
                    auto same = [&](const std::function<double(const V3 &)> &f) { return !f || f(x) == f(y); };
                    if (!same(material->sigma_a) || !same(material->sigma_s) || !same(material->source)) { symmetric = false; break; }
                }
            if (!symmetric) { mirror_reason[d] = "material"; continue; }
            // Inflow on every face: 7^(dim-1) points per face, 4 times, 26 directions.
            std::vector<V3> directions;
            for (int a = -1; a <= 1; ++a) for (int b = -1; b <= 1; ++b) for (int c = -1; c <= 1; ++c)
                if (a || b || c) directions.push_back(V3(a, b, c).normalized());
            for (int f = 0; f < 2 * dim && symmetric; ++f) {
                const int fd = f / 2;
                const int samples = dim == 1 ? 1 : (dim == 2 ? 7 : 49);
                for (int s = 0; s < samples && symmetric; ++s) {
                    V3 x = origin;
                    for (int e = 0, k = s; e < dim; ++e) {
                        if (e == fd) { x[e] = f % 2 ? origin[e] + extent[e] : origin[e]; continue; }
                        x[e] = origin[e] + extent[e] * (k % 7 + .5) / 7;
                        k /= 7;
                    }
                    const V3 y = reflect(x);
                    for (double t : times) {
                        if (inflow.density)
                            for (const V3 &w : directions) {
                                V3 rw = w;
                                rw[d] = -rw[d];
                                if (std::abs(inflow.density(x, t, w) - inflow.density(y, t, rw)) > 1e-14) { symmetric = false; break; }
                            }
                        if (inflow.atoms && symmetric) {
                            auto ax = inflow.atoms(x, t), ay = inflow.atoms(y, t);
                            if (ax.size() != ay.size()) { symmetric = false; break; }
                            for (auto &atom : ay) atom.direction[d] = -atom.direction[d];
                            for (const auto &atom : ax) {
                                bool found = false;
                                for (const auto &other : ay)
                                    found = found || ((atom.direction - other.direction).norm() < 1e-14 &&
                                                      std::abs(atom.mass - other.mass) <= 1e-14 * std::max(1., std::abs(atom.mass)));
                                if (!found) { symmetric = false; break; }
                            }
                        }
                        if (inflow.exterior_m1 || inflow.exterior_state) symmetric = false; // not checked: refuse
                    }
                }
            }
            if (!symmetric) { mirror_reason[d] = "inflow"; continue; }
            mirror_dir[d] = true;
            mirror_reason[d] = material ? "symmetric (" + material_check + ")" : "symmetric";
        }
        for (int d = 0; d < dim; ++d)
            if (mirror_dir[d]) {
                const int half = counts[d] / 2;
                counts[d] = half;
                origin[d] = center[d];
                if (!eo.planes[d].empty()) { // graded: the upper half of the planes, starting exactly at the origin
                    std::vector<double> P(eo.planes[d].begin() + half, eo.planes[d].end());
                    origin[d] = P.front();
                    eo.planes[d] = std::move(P);
                }
                if (!eo.breaks[d].empty()) { // broken lines of the upper half, renumbered
                    std::vector<int> K;
                    for (int k : eo.breaks[d])
                        if (k > half)
                            K.push_back(k - half);
                    eo.breaks[d] = std::move(K);
                }
                eo.mirror[2 * d] = true;
            }
        eo.origin = origin;
        eo.elements = counts[0];
        eo.elements_per_axis = {counts[0], dim > 1 ? counts[1] : 0, dim > 2 ? counts[2] : 0};
        eo.length = h * counts[0];
    }
    const bool mirrored = mirror_dir[0] || mirror_dir[1] || mirror_dir[2];
    double domain_factor = 1; // full mass / computed mass
    for (int d = 0; d < dim; ++d) if (mirror_dir[d]) domain_factor *= 2;
    // ---- Angular parity reduction (1D/2D) --------------------------------------------------------------------------
    eo.reduce_angular = dim < 3 && o.angular_reduction != "off";
    require(!(o.angular_reduction == "on" && dim == 3), "--angular-reduction on needs dimension 1 or 2");
    std::unique_ptr<ElementWeno> scheme_ptr;
#ifdef M1PN_MPI
    std::unique_ptr<ElementWenoMpi> mpi;
    SlabPartition part;
    if (distributed)
        part = slab_partition(counts[dim - 1], ranks, rank, dim - 1, 2);
#endif
    auto build_scheme = [&]() {
#ifdef M1PN_MPI
        mpi.reset(); // detaches from the old scheme first
        if (distributed) {
            scheme_ptr = std::make_unique<ElementWeno>(pn, slab_options(eo, part), inflow);
            mpi = std::make_unique<ElementWenoMpi>(*scheme_ptr, MPI_COMM_WORLD, part, 0);
            return;
        }
#endif
        scheme_ptr = std::make_unique<ElementWeno>(pn, eo, inflow);
    };
    build_scheme();
    auto reduce_sum = [&](double v) {
#ifdef M1PN_MPI
        if (mpi)
            return mpi->sum(v);
#endif
        return v;
    };
    auto reduce_min = [&](double v) {
#ifdef M1PN_MPI
        if (mpi)
            return mpi->min(v);
#endif
        return v;
    };
    auto reduce_max = [&](double v) {
#ifdef M1PN_MPI
        if (mpi)
            return mpi->max(v);
#endif
        return v;
    };
    std::vector<Vec> initial(scheme_ptr->nodes);
    for (int i = 0; i < scheme_ptr->nodes; ++i) {
        initial[i] = initial_point(o, scheme_ptr->node_position(i), i, dim, pn);
        if (fcs) { // collided part: the floor only (the pulse is uncollided)
            initial[i].setZero();
            initial[i][0] = o.background / std::sqrt(4 * pi);
        }
        cone_require(exact::cone(project(initial[i])), "Initial Bernstein coefficient outside the M1 cone");
    }
    double odd_initial = 0, odd_inflow = 0;
    if (eo.reduce_angular) {
        double scale = 0, dropped = 0;
        std::vector<char> kept(pn.q, 0);
        for (int k : scheme_ptr->active) kept[k] = 1;
        for (const Vec &u : initial)
            for (int k = 0; k < pn.q; ++k) (kept[k] ? scale : dropped) = std::max(kept[k] ? scale : dropped, std::abs(u[k]));
        scale = reduce_max(scale), dropped = reduce_max(dropped); // the decision is global
        odd_initial = scale > 0 ? dropped / scale : dropped;
        for (int k = 0; k <= 8; ++k)
            odd_inflow = std::max(odd_inflow, scheme_ptr->dropped_inflow(o.final_time * k / 8.));
        odd_inflow = reduce_max(odd_inflow);
        const bool even = odd_initial <= 1e-13 && odd_inflow <= 1e-12;
        require(even || o.angular_reduction != "on",
                "--angular-reduction on: initial or inflow data are not even in the unresolved directions");
        if (!even) {
            if (rank == 0)
                std::cerr << "note: angular reduction off (odd data: initial " << odd_initial << ", inflow " << odd_inflow << ")\n";
            eo.reduce_angular = false;
            build_scheme();
        }
    }
    const ElementWeno &scheme = *scheme_ptr;
    Field a(scheme.nodes, scheme.moments);
    for (int i = 0; i < scheme.nodes; ++i)
        a.row(i) = scheme.reduce(initial[i]).transpose();
    std::vector<Vec>().swap(initial);
    // MPI: global scheme on rank 0 for the sample output, the slices and the material integrals.
    std::unique_ptr<ElementWeno> post;
    if (distributed && rank == 0)
        post = std::make_unique<ElementWeno>(pn, eo, inflow);
    const ElementWeno &out_scheme = post ? *post : scheme;
    auto gathered = [&](const Field &f) -> Field { // owned rows of all ranks on rank 0 (global order)
#ifdef M1PN_MPI
        if (mpi)
            return mpi->gather(f);
#endif
        return f;
    };
#ifdef M1PN_ELEMENT_GPU
    std::unique_ptr<ElementWenoGpu> device;
    double gpu_radius_seconds = 0;
    if (o.gpu) {
        device = std::make_unique<ElementWenoGpu>(scheme);
        const double r0 = seconds();
        scheme.set_spectral_radius(device->spectral_radius());
        gpu_radius_seconds = seconds() - r0;
    }
#else
    require(!o.gpu, "--gpu: built without M1PN_ELEMENT_GPU");
#endif
    require(o.dt_bound == "both" || o.dt_bound == "high-order", "--dt-bound both|high-order");
    const bool high_order_only = o.dt_bound == "high-order";
    const double selected = o.dt > 0 ? o.dt : o.cfl * (high_order_only ? scheme.high_order_step() : scheme.step_bound());
    require(high_order_only || selected <= scheme.realizability_step() * (1 + 1e-12),
            "Requested dt violates the element realizability bound");
    if (high_order_only && rank == 0)
        std::cerr << "EXPERIMENT --dt-bound high-order: dt " << selected << ", realizability step "
                  << scheme.realizability_step() << " (ratio " << selected / scheme.realizability_step() << ")\n";
    if (selected > scheme.high_order_step() && rank == 0)
        std::cerr << "warning: dt exceeds the estimated high-order stability step " << scheme.high_order_step() << "\n";
    // ---- Uniform sample grid on the full box (Q1 nodes of count*degree cells per axis) -----------------------------
    std::array<int, 3> cells_axis{1, 1, 1}, points_axis{1, 1, 1};
    long samples = 1;
    for (int d = 0; d < dim; ++d) {
        cells_axis[d] = (mirror_dir[d] ? 2 : 1) * counts[d] * o.degree;
        points_axis[d] = cells_axis[d] + (o.mesh.periodic ? 0 : 1);
        samples *= points_axis[d];
    }
    const double hs = h / o.degree;
    V3 full_origin = origin;
    for (int d = 0; d < dim; ++d) if (mirror_dir[d]) full_origin[d] = center[d] - .5 * extent[d];
    std::vector<V3> where(samples);
    Vec sample_mass(samples);
    // Graded mesh: the samples are the Q1 nodes of the element planes subdivided by the degree, with trapezoid
    // weights (the uniform branch is unchanged).
    std::array<std::vector<double>, 3> axis_coord;
    if (!full_planes[0].empty())
        for (int d = 0; d < dim; ++d) {
            const auto &P = full_planes[d];
            for (std::size_t i = 0; i + 1 < P.size(); ++i)
                for (int k = 0; k < o.degree; ++k)
                    axis_coord[d].push_back(P[i] + (P[i + 1] - P[i]) * k / o.degree);
            axis_coord[d].push_back(P.back());
            require(int(axis_coord[d].size()) == points_axis[d], "graded sample grid");
        }
    for (long s = 0; s < samples; ++s) {
        V3 x = V3::Zero();
        double m = 1;
        long k = s;
        for (int d = 0; d < dim; ++d) {
            const int index = int(k % points_axis[d]);
            k /= points_axis[d];
            if (!axis_coord[d].empty()) {
                const auto &c = axis_coord[d];
                x[d] = c[index];
                m *= .5 * ((index + 1 < int(c.size()) ? c[index + 1] : c[index]) - (index > 0 ? c[index - 1] : c[index]));
                continue;
            }
            x[d] = full_origin[d] + hs * index;
            const bool edge = !o.mesh.periodic && (index == 0 || index == cells_axis[d]);
            m *= (edge ? .5 : 1.) * hs;
        }
        where[s] = x;
        sample_mass[s] = m;
    }
    // With FCS the written u0 is the total density (collided + uncollided) and rho_uncollided is appended; the
    // uncollided current is not added (u1..u3 are the collided moments).
    auto write_samples = [&](const std::string &file, const Field &state, double time) {
        std::ofstream f(file);
        require(bool(f), "Cannot write field");
        const int written = state.cols() == out_scheme.moments ? pn.q : 4;
        f << std::setprecision(17) << "node,x,y,z";
        for (int k = 0; k < written; ++k) f << ",u" << k;
        if (fcs) f << ",rho_uncollided";
        f << "\n";
        std::vector<double> unc;
        if (fcs) sample_uncollided(time, where, unc);
        Field values(samples, pn.q);
        parallel_for(int(samples), [&](int s) {
            // Mirror reduction: evaluate at the reflected point and reflect the moments back.
            V3 x = where[s];
            std::array<bool, 3> flip{};
            for (int d = 0; d < dim; ++d)
                if (mirror_dir[d] && x[d] < center[d]) {
                    x[d] = 2 * center[d] - x[d], flip[d] = true;
                    // Broken lines: on a line the value is two-sided and evaluate() takes the upper element; the
                    // reflected point must take the lower one to match the full mesh (one ulp, only broken meshes).
                    if (!eo.breaks[d].empty())
                        x[d] = std::nextafter(x[d], -std::numeric_limits<double>::infinity());
                }
            const Vec active_values = out_scheme.evaluate(state, x);
            Vec u = Vec::Zero(pn.q);
            for (int k = 0; k < active_values.size(); ++k) u[out_scheme.active[k]] = active_values[k];
            for (int d = 0; d < dim; ++d)
                if (flip[d]) u = reflect_pn(pn, u, d);
            values.row(s) = u.transpose();
        });
        for (long s = 0; s < samples; ++s) {
            f << s << "," << where[s][0] << "," << where[s][1] << "," << where[s][2];
            if (fcs) values(s, 0) += unc[s] / std::sqrt(4 * pi);
            for (int k = 0; k < written; ++k) f << "," << values(s, k);
            if (fcs) f << "," << unc[s];
            f << "\n";
        }
    };
    if (!o.skip_fields) {
        const bool all = o.sample_moments != "low";
        const Field g = gathered(all ? a : Field(a.leftCols(4)));
        if (rank == 0)
            write_samples(o.output + "/initial.csv", g, 0.);
    }
    auto mass_of = [&](const Field &state) { // owned nodes, reduced (all ranks call it)
        double m = 0;
        for (int i = 0; i < scheme.nodes; ++i)
            if (scheme.is_owned(i)) m += scheme.lumped[i] * project(state.row(i).transpose())[0];
        return reduce_sum(m) * domain_factor;
    };
    auto absorption_rate = [&](const Field &state) { // int sigma_a rho dx with nodal sigma_a and lumped mass
        double r = 0;
        for (int i = 0; i < scheme.nodes; ++i)
            if (scheme.is_owned(i))
                r += scheme.lumped[i] * (scheme.node_sigma_a[i] + o.sigma_a) * project(state.row(i).transpose())[0];
        return reduce_sum(r) * domain_factor;
    };
    const double initial_mass = mass_of(a);
    if (rank == 0) {
        if (!o.skip_fields) {
            std::ofstream masses(o.output + "/nodal_mass.csv");
            masses << std::setprecision(17) << "node,mass\n";
            for (long s = 0; s < samples; ++s) masses << s << "," << sample_mass[s] << "\n";
        }
        std::ofstream config(o.output + "/config.json");
        config << std::setprecision(17) << "{\n\"method\":\"element-weno\",\"model\":\"r2\",\"coupling\":\"m1pn\""
               << ",\"stabilization\":\"element-weno\",\"pn_target\":\"consistent_mass_cg_weno\""
               << ",\"basis\":\"tensor_bernstein\",\"degree\":" << o.degree
               << ",\"m1_target\":" << (o.m1_target ? "true" : "false")
               << ",\"fcs\":" << std::quoted(o.fcs)
               << ",\"weno\":" << (o.weno ? "true" : "false") << ",\"weno_linear\":" << o.weno_linear
               << ",\"weno_q\":" << o.weno_q << ",\"weno_deadband\":" << o.weno_deadband
               << ",\"weno_floor\":" << o.weno_floor << ",\"element_limit\":" << (o.element_limit ? "true" : "false")
               << ",\"filter_mode\":" << std::quoted(o.filter_mode) << ",\"filter_constant\":" << o.filter_strength
               << ",\"sigma_s\":" << o.sigma_s << ",\"sigma_a\":" << o.sigma_a
               << ",\"reaction_integration\":\"exact_exponential_half_steps_around_ssprk\""
               << ",\"reaction_model\":" << (material ? "\"nodal_average_int_sigma_phi_over_int_phi\"" : "\"constant\"")
               << ",\"material\":" << std::quoted(material ? material->name : std::string("none"))
               << ",\"material_subcells\":" << (material ? material->subcells : 0)
               << ",\"material_integrals\":{\"volume\":" << out_scheme.integral_volume * domain_factor
               << ",\"source\":" << out_scheme.integral_source * domain_factor
               << ",\"sigma_a\":" << out_scheme.integral_sigma_a * domain_factor
               << ",\"sigma_s\":" << out_scheme.integral_sigma_s * domain_factor << "}"
               << ",\"material_expected\":" << expected_integrals
               << ",\"domain_origin\":[" << full_origin[0] << "," << full_origin[1] << "," << full_origin[2] << "]"
               << ",\"domain_extent\":[" << extent[0] << "," << extent[1] << "," << extent[2] << "]"
               << ",\"cell_size\":" << h << ",\"graded\":" << (o.cells_per_unit.empty() ? "false" : "true")
               << ",\"pn_order\":" << pn.order << ",\"Q\":" << pn.q << ",\"dimension\":" << dim
               << ",\"mirror_reduction\":" << (mirrored ? "true" : "false")
               << ",\"mirror_directions\":[" << mirror_dir[0] << "," << mirror_dir[1] << "," << mirror_dir[2] << "]"
               << ",\"mirror_reasons\":[" << std::quoted(mirror_reason[0]) << "," << std::quoted(mirror_reason[1]) << ","
               << std::quoted(mirror_reason[2]) << "]"
               << ",\"mirror_reduction_mode\":" << std::quoted(o.mirror_reduction)
               << ",\"mirror_asymmetry\":" << mirror_asymmetry
               << ",\"computed_elements_per_axis\":" << counts[0]
               << ",\"computed_counts\":[" << counts[0] << "," << counts[1] << "," << counts[2] << "]"
               << ",\"angular_reduction\":" << (eo.reduce_angular ? "true" : "false")
               << ",\"angular_reduction_mode\":" << std::quoted(o.angular_reduction)
               << ",\"active_moments\":" << out_scheme.moments << ",\"odd_initial_fraction\":" << odd_initial
               << ",\"odd_inflow_fraction\":" << odd_inflow
               << ",\"elements_per_axis\":" << o.mesh.elements << ",\"coefficients\":" << out_scheme.nodes
               << ",\"spatial_dofs_full_domain\":" << long(std::llround(out_scheme.nodes * domain_factor))
               << ",\"mpi_ranks\":" << ranks << ",\"mpi_halo\":" << (distributed ? 2 : 0)
               << ",\"sample_points\":" << samples << ",\"sample_cells_per_axis\":" << cells_axis[0]
               << ",\"periodic\":" << (o.mesh.periodic ? "true" : "false") << ",\"rk\":" << o.rk
               << ",\"time_integrator\":" << (eo.ssprk43 ? "\"ssprk43\"" : "\"ssprk3\"")
               << ",\"case\":" << std::quoted(o.problem) << ",\"inflow\":" << std::quoted(o.inflow)
               << ",\"background\":" << o.background << ",\"shadow_sigma\":" << (shadow ? o.shadow_sigma : 0.)
               << ",\"final_time\":" << o.final_time << ",\"dt_requested\":" << o.dt << ",\"dt_selected\":" << selected
               << ",\"thread_policy\":" << std::quoted(o.thread_policy)
               << ",\"cfl_safety\":" << o.cfl << ",\"element_dt_bound\":" << scheme.dt_bound
               << ",\"realizability_step\":" << scheme.realizability_step()
               << ",\"high_order_step\":" << scheme.high_order_step()
               << ",\"gpu\":" << (o.gpu ? "true" : "false")

               << ",\"threads\":" << o.threads << beam_metadata(o, dim)
               << ",\"compiler\":" << std::quoted(__VERSION__) << "\n}\n";
    }
    auto progress = [&](const char *status, double time, int count) {
        if (rank != 0)
            return;
        const std::string tmp = o.output + "/progress.json.tmp", dst = o.output + "/progress.json";
        std::ofstream f(tmp);
        f << std::setprecision(17) << "{\"status\":" << std::quoted(status) << ",\"time\":" << time
          << ",\"steps\":" << count << ",\"dt\":" << selected << ",\"wall_seconds\":" << seconds() - begin << "}\n";
        f.close();
        std::filesystem::rename(tmp, dst);
    };
    ElementWenoStats stats;
    double t = 0, solver_seconds = 0, min_density = std::numeric_limits<double>::infinity(), min_margin = min_density;
    double min_margin_steps = min_density;
    int steps = 0;
    const double initialization = seconds() - begin;
    progress("running", t, steps);
    std::ofstream history;
    if (rank == 0) {
        history.open(o.output + "/history.csv");
        history << std::setprecision(17) << "step,time,mass,absorption_rate,injected,absorbed,boundary_outflow,nudge_mass,min_margin\n";
    }
    // Ledger: injected, absorbed and nudge_mass are per rank over owned nodes (summed here); boundary_outflow is
    // reduced inside the scheme already.
    auto record = [&](int step) { // all ranks (reductions); rank 0 writes
        double mm = std::numeric_limits<double>::infinity();
        for (int i = 0; i < scheme.nodes; ++i)
            if (scheme.is_owned(i)) mm = std::min(mm, margin(project(a.row(i).transpose())));
        mm = reduce_min(mm);
        min_margin_steps = std::min(min_margin_steps, mm);
        const double mass = mass_of(a), rate = absorption_rate(a);
        const double inj = reduce_sum(stats.injected), absd = reduce_sum(stats.absorbed), nud = reduce_sum(stats.nudge_mass);
        if (rank == 0)
            history << step << "," << t << "," << mass << "," << rate << "," << inj * domain_factor << ","
                    << absd * domain_factor << "," << stats.boundary_outflow * domain_factor << "," << nud * domain_factor
                    << "," << mm << "\n";
    };
    record(0);
#ifdef M1PN_ELEMENT_GPU
    if (device)
        device->begin(a);
#endif
    while (t < o.final_time) {
        require(steps < o.max_steps, "Maximum step count exceeded");
        double dt = std::min(selected, o.final_time - t);
        const double start = seconds();
#ifdef M1PN_ELEMENT_GPU
        if (device) {
            const double after = t + dt, next = after < o.final_time ? after : std::numeric_limits<double>::quiet_NaN();
            device->step(t, dt, next, stats);
        } else
#endif
            a = scheme.advance(t, dt, a, &stats);
        solver_seconds += seconds() - start;
        t = (o.final_time - (t + dt) <= 8 * std::numeric_limits<double>::epsilon() * o.final_time) ? o.final_time : t + dt;
        ++steps;
        if (steps % 10 == 0) progress("running", t, steps);
        if (o.history_stride > 0 && steps % o.history_stride == 0) {
#ifdef M1PN_ELEMENT_GPU
            if (device)
                a = device->state();
#endif
            record(steps);
        }
    }
#ifdef M1PN_ELEMENT_GPU
    if (device)
        a = device->end(stats);
#endif
    if (o.history_stride <= 0 || steps % o.history_stride) record(steps);
    for (int i = 0; i < scheme.nodes; ++i) {
        if (!scheme.is_owned(i))
            continue;
        const Four u = project(a.row(i).transpose());
        min_density = std::min(min_density, u[0]);
        min_margin = std::min(min_margin, margin(u));
    }
    min_density = reduce_min(min_density), min_margin = reduce_min(min_margin);
    solver_seconds = reduce_max(solver_seconds);
    // Final state on rank 0 (global order) for the fields and the 3D slices.
    const bool all_moments = o.sample_moments != "low";
    const Field final_global = (!o.skip_fields || dim == 3) ? gathered(all_moments || dim == 3 ? a : Field(a.leftCols(4)))
                                                           : Field();
    if (!o.skip_fields && rank == 0)
        write_samples(o.output + "/final.csv", all_moments ? final_global : Field(final_global.leftCols(4)), t);
    if (dim == 3 && rank == 0) {
        // Mid-plane slices through the box centre (density and |J|), written even with --skip-fields: the full 3D
        // sample field is too large for P11 on fine meshes. plane = normal axis, (i, j) = in-plane sample indices.
        std::ofstream sl(o.output + "/slices.csv");
        sl << std::setprecision(17) << "plane,i,j,x,y,z,rho,jnorm\n";
        for (int plane = 0; plane < 3; ++plane) {
            const int u = plane == 0 ? 1 : 0, v = plane == 2 ? 1 : 2;
            std::vector<std::array<double, 5>> rowsv(std::size_t(points_axis[u]) * points_axis[v]);
            parallel_for(points_axis[u] * points_axis[v], [&](int k) {
                const int i = k % points_axis[u], j = k / points_axis[u];
                V3 x = full_origin;
                x[plane] = center[plane];
                x[u] = full_origin[u] + hs * i;
                x[v] = full_origin[v] + hs * j;
                V3 y = x;
                for (int d = 0; d < dim; ++d)
                    if (mirror_dir[d] && y[d] < center[d]) y[d] = 2 * center[d] - y[d];
                const Vec values = out_scheme.evaluate(final_global, y);
                Vec full = Vec::Zero(pn.q);
                for (int c = 0; c < values.size(); ++c) full[out_scheme.active[c]] = values[c];
                const Four m = project(full);
                rowsv[k] = {x[0], x[1], x[2], m[0], m.tail<3>().norm()};
            });
            for (int k = 0; k < points_axis[u] * points_axis[v]; ++k)
                sl << plane << "," << k % points_axis[u] << "," << k / points_axis[u] << "," << rowsv[k][0] << ","
                   << rowsv[k][1] << "," << rowsv[k][2] << "," << rowsv[k][3] << "," << rowsv[k][4] << "\n";
        }
    }
    if (kobayashi_problem && rank == 0) {
        // Benchmark detector points (octant, cm): collided density of the final state (Bernstein evaluation) plus the
        // steady uncollided density at the point itself (with FCS); without FCS the grid carries everything.
        const auto box = uncollided::kobayashi::geometry(kobayashi_problem);
        std::ofstream df(o.output + "/detectors.csv");
        df << std::setprecision(17) << "line,x,y,z,rho_grid,rho_uncollided,rho\n";
        for (const auto &d : uncollided::kobayashi::detectors(kobayashi_problem)) {
            V3 y(d.x, d.y, d.z);
            for (int k = 0; k < dim; ++k)
                if (mirror_dir[k] && y[k] < center[k]) y[k] = 2 * center[k] - y[k];
            const Vec values = out_scheme.evaluate(final_global, y);
            Vec full = Vec::Zero(pn.q);
            for (int c = 0; c < values.size(); ++c) full[out_scheme.active[c]] = values[c];
            const double grid = project(full)[0], unc = fcs ? box.density(V3(d.x, d.y, d.z)) : 0.;
            df << d.line << "," << d.x << "," << d.y << "," << d.z << "," << grid << "," << unc << "," << grid + unc << "\n";
        }
    }
    progress("completed", t, steps);
    const double final_mass = mass_of(a);
    const double injected = reduce_sum(stats.injected) * domain_factor, absorbed = reduce_sum(stats.absorbed) * domain_factor,
                 outflow = stats.boundary_outflow * domain_factor, nudge = reduce_sum(stats.nudge_mass) * domain_factor;
    const double balance = final_mass - initial_mass + outflow + absorbed - injected - nudge;
    double uncollided_final = 0; // int rho_u(T) over the computed domain by the FCS quadrature
    if (fcs) scheme.collided_source(t, &uncollided_final); // owned elements
    uncollided_final = reduce_sum(uncollided_final);
    // Lattice FCS: uncollided mass, absorption rate and absorbed total (FCS quadrature, time integral of the table),
    // and the totals that compare with the kinetic reference.
    std::string lattice_fcs_json;
    if (fcs_lattice) {
        const auto &xs = scheme.fcs_points();
        const auto &ws = scheme.fcs_weights();
        const int P = scheme.fcs_points_per_element();
        double mu = 0, ru = 0, au = 0, table_points = 0;
        for (std::size_t k = 0; k < xs.size(); ++k) {
            if (!scheme.element_owned(int(k / P)))
                continue; // MPI: owned elements only
            table_points += 1;
            const double sa = material->sigma_a(xs[k]), r = table_value(k, t);
            mu += ws[k] * r, ru += ws[k] * sa * r;
            if (sa > 0)
                au += ws[k] * sa * table_integral(k, t);
        }
        mu = reduce_sum(mu) * domain_factor, ru = reduce_sum(ru) * domain_factor, au = reduce_sum(au) * domain_factor;
        table_points = reduce_sum(table_points);
        const double rc = absorption_rate(a);
        std::ostringstream j;
        j << std::setprecision(17) << ",\"lattice_fcs\":{\"uncollided_mass_final\":" << mu
          << ",\"uncollided_absorption_rate_final\":" << ru << ",\"uncollided_absorbed\":" << au
          << ",\"total_mass_final\":" << final_mass + mu << ",\"total_absorption_rate_final\":" << rc + ru
          << ",\"total_absorbed\":" << absorbed + au << ",\"table_points\":" << table_points
          << ",\"table_times\":" << (lattice3d ? lattice3_table->M : lattice_table->M) + 1
          << ",\"table_seconds\":" << reduce_max(lattice3d ? lattice3_table->build_seconds : lattice_table->build_seconds) << "}";
        lattice_fcs_json = j.str();
    }
    const double balance_scale = std::max({std::abs(initial_mass), injected, absorbed, std::abs(outflow), 1e-300});
    const double absorption_final = absorption_rate(a);
    // Diagnostics over ranks (element counts include the halo elements of every rank).
    stats.min_gamma = reduce_min(stats.min_gamma), stats.min_limiter = reduce_min(stats.min_limiter);
    stats.max_limited_zero_sum = reduce_max(stats.max_limited_zero_sum);
    stats.limited_elements = long(reduce_sum(double(stats.limited_elements)));
    stats.limited_boundary_nodes = long(reduce_sum(double(stats.limited_boundary_nodes)));
    stats.nudged_states = long(reduce_sum(double(stats.nudged_states)));
    if (rank != 0)
        return 0;
    struct rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    std::ofstream result(o.output + "/summary.json");
    result << std::setprecision(17) << "{\"status\":\"completed\",\"method\":\"element-weno\",\"steps\":" << steps
           << ",\"time\":" << t << ",\"total_seconds\":" << seconds() - begin << ",\"solver_seconds\":" << solver_seconds
#ifdef M1PN_ELEMENT_GPU
               << (device ? ",\"gpu_timing\":{\"spectral_radius\":" + std::to_string(gpu_radius_seconds) +
                                ",\"host_boundary\":" + std::to_string(device->t_boundary) +
                                ",\"host_ghost\":" + std::to_string(device->t_ghost) +
                                ",\"diag_device\":" + std::to_string(device->t_diag_device) +
                                ",\"host_fallback\":" + std::to_string(device->t_fallback) +
                                ",\"host_boundary_high\":" + std::to_string(device->t_boundary_high) +
                                ",\"host_face_data\":" + std::to_string(device->t_face_data) +
                                ",\"host_correction\":" + std::to_string(device->t_correction) +
                                ",\"wait\":" + std::to_string(device->t_wait) + ",\"sync\":" + std::to_string(device->t_sync) +
                                ",\"fallbacks\":" + std::to_string(device->closure_fallbacks + device->limiter_fallbacks) + "}"
                          : std::string())
#endif
           
           << ",\"initialization_seconds\":" << initialization << ",\"peak_rss_kib_rank0\":" << usage.ru_maxrss
           << ",\"minimum_coefficient_density\":" << min_density << ",\"minimum_cone_margin\":" << min_margin
           << ",\"minimum_cone_margin_recorded_steps\":" << min_margin_steps
           << ",\"final_density_mass\":" << final_mass << ",\"initial_density_mass\":" << initial_mass
           << ",\"ledger\":{\"injected\":" << injected << ",\"absorbed\":" << absorbed << ",\"boundary_outflow\":" << outflow
           << ",\"nudge_mass\":" << nudge << ",\"balance_residual\":" << balance
           << ",\"balance_relative\":" << balance / balance_scale << "}"
           << ",\"absorption_rate_final\":" << absorption_final
           << ",\"fcs\":" << std::quoted(o.fcs) << ",\"uncollided_mass_final\":" << uncollided_final * domain_factor
           << ",\"uncollided_mass_exact\":" << (fcs && !fcs_lattice && !kobayashi_problem ? pulse.mass * std::exp(-pulse.sigma_t * t) : 0.)
           << lattice_fcs_json
           << ",\"min_weno_gamma\":" << stats.min_gamma
           << ",\"min_element_factor\":" << stats.min_limiter << ",\"limited_element_stages\":" << stats.limited_elements << ",\"rate_calls\":" << stats.rate_calls << ",\"unlimited_rate_calls\":" << stats.unlimited_rate_calls << ",\"cells\":" << out_scheme.cells
           << ",\"mpi_ranks\":" << ranks
           << ",\"limited_boundary_nodes\":" << stats.limited_boundary_nodes << ",\"nudged_states\":" << stats.nudged_states
           << ",\"split_limited\":" << stats.split_limited << ",\"split_mass\":" << stats.split_mass
           << ",\"min_split_theta\":" << stats.min_split_theta
           << ",\"max_limited_zero_sum\":" << stats.max_limited_zero_sum
           << ",\"phase_seconds\":{\"closure\":" << stats.t_closure << ",\"weno\":" << stats.t_weno
           << ",\"stabilization\":" << stats.t_stabilization << ",\"gradient_solve\":" << stats.t_gradient_solve
           << ",\"stab_local\":" << stats.t_stab_local << ",\"residual\":" << stats.t_residual
           << ",\"boundary\":" << stats.t_boundary << ",\"mass\":" << stats.t_mass << ",\"limiter\":" << stats.t_limiter
           << ",\"finish\":" << stats.t_finish << ",\"reaction\":" << stats.t_reaction
           << ",\"fcs_density\":" << stats.t_fcs_density << ",\"fcs_quadrature\":" << stats.t_fcs_quadrature << "}"
           << ",\"projection\":{\"calls\":" << stats.projection.calls << ",\"identities\":" << stats.projection.identities
           << ",\"augmented\":" << stats.projection.augmented << ",\"diracs\":" << stats.projection.diracs
           << ",\"certificates\":" << stats.projection.certificates << ",\"failures\":" << stats.projection.failures
           << ",\"iterations\":" << stats.projection.iterations << ",\"backtracks\":" << stats.projection.backtracks
           << ",\"max_iterations\":" << stats.projection.max_iterations
           << ",\"trace_thread_seconds\":" << stats.projection.product_seconds
           << ",\"solve_thread_seconds\":" << stats.projection.reduce_seconds << "}}\n";
    std::cout << "Completed " << steps << " steps, P" << pn.order << ", element-weno p=" << o.degree << ", "
              << out_scheme.nodes << " coefficients" << (distributed ? ", " + std::to_string(ranks) + " MPI ranks" : std::string()) << "\n";
    return 0;
}
} // namespace
int main(int argc, char **argv) {
    // Large fields are reallocated every stage; keep them in the heap and reuse their pages.
    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, 1 << 30);
#ifdef M1PN_MPI
    int provided;
    // Threaded line pipeline (MULTIPLE) and eager pipeline messages, as m1pn_element_weno; a user value of
    // UCX_RNDV_THRESH is kept.
    setenv("UCX_RNDV_THRESH", "512kb", 0);
    setenv("OMP_MAX_TASK_PRIORITY", "1", 0);
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
#endif
    Runtime world;
    try {
        exact::environment();
        const double begin = seconds();
        Options o = parse(argc, argv);
        // Bandwidth bound: default to the physical cores of one NUMA node (src/hardware.hpp).
        if (!o.threads_explicit) {
            const ThreadDefault automatic = default_threads();
            o.threads = automatic.threads;
            o.thread_policy = automatic.reason;
        }
#ifdef _OPENMP
        omp_set_num_threads(o.threads);
#else
        require(o.threads == 1, "This build has no OpenMP support");
#endif
        if (world.rank == 0) {
            require(!std::filesystem::exists(o.output), "Output directory already exists: " + o.output);
            std::filesystem::create_directories(o.output);
        }
        Angular pn(o.order, o.budget);
        const int code = run_element_weno(o, pn, begin, world);
#ifdef M1PN_MPI
        MPI_Finalize();
#endif
        return code;
    } catch (const std::exception &e) {
        std::cerr << "rank " << world.rank << ": " << e.what() << "\n";
#ifdef M1PN_MPI
        if (world.size > 1)
            MPI_Abort(MPI_COMM_WORLD, 2);
        else
            MPI_Finalize();
#endif
        return 2;
    }
}
