// Generated isolated driver: scripts/prepare_driver.py
#include "time.hpp"
#include "cuda_path.hpp"
#include "pn_uncoupled.hpp"
#include "filter.hpp"
#include "element_weno.hpp"
#include <memory>
#include <filesystem>
#include <malloc.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/resource.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace m1pn;
namespace {
struct Options {
    std::string filter_mode = "off", filter_model;
    std::string angular_reduction = "auto"; // element-weno: auto|on|off (parity reduction in 1D/2D)
    std::string mirror_reduction = "auto";  // element-weno: auto|off (half/quarter domain for mirror-symmetric cases)
    double filter_strength = 0;
    MeshSpec mesh;
    Numerics num;
    int order = 3, rk = 3, max_steps = 100000, norm_max = 512, threads = 1, checkpoint_stride = 0;
    double final_time = .01, dt = 0, cfl = .8, sigma_a = 0, sigma_s = 0, anisotropy = 0, source = 0;
    double pulse_width = .07, background = .001;
    // line-source: generated square [0.5-L/2, 0.5+L/2]^2, source at its centre.
    double domain_length = 1;
    bool jump = false, deterministic = false, mass_mode_explicit = false, pn_only_self_test = false;
    bool cuda_explicit = false, cuda_on = false;
    bool rk_explicit = false;
    // --stabilization element-weno (Bernstein Q_p, element-based limiting)
    int degree = 2;
    double weno_linear = 1e-3, weno_q = 1;
    bool weno = true, element_limit = true;
    std::string problem = "smooth", inflow = "vacuum", output = std::string(PN_AFC_ROOT) + "/tmp/run", pair_limiter = "scalar", coupling = "m1pn";
};
Options parse(int argc, char **argv) {
    Options o;
    o.num.ledger_stride = 64;
    o.num.r2 = true;
#ifdef M1PN_FAST_DEFAULTS
    // m1pn_fast: production settings of the 512-element runs are the default.
    o.rk = 2;
    o.cfl = 1;
    o.num.stabilization = "m1pn_mass_correction";
    o.num.mass_mode = "lumped";
    o.num.kv_gamma = 0;
#ifdef _OPENMP
    o.threads = omp_get_max_threads();
#endif
#endif
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help") {
            std::cout
                << "m1pn --pn-order N --elements n --dimension 1|2|3 --mesh-type "
                   "hex|tet|single-tet\n"
                   " --periodic | --mesh FILE --biharmonic-gamma G --biharmonic-boundary "
                   "natural|masked\n"
                   " --boundary onsager-c1|characteristic-c0 --time-integrator ssprk2|ssprk3|ssprk43 (ssprk43: element-weno only)\n"
                   " --case "
                   "smooth|stress|isotropic|vacuum|pure-beam|beam|crossing|plane-source|line-source|shadow|"
                   "lattice|lattice-paul|smooth-beam|vacuum-beam|oblique-beam|crossing-three|crossing-study|crossing-3d\n"
                   " --model r2|legacy --stabilization gradient-defect|biharmonic-original|biharmonic-corrected\n"
                   " --stabilization m1pn_ghost_local_projected_p4 --ghost-gamma G (open Cartesian Q1 only)\n"
                   " --stabilization m1pn_mass_correction --pn-viscosity-scale L (SSPRK3, lumped mass)\n"
                   " --coupling m1pn|pn-only (pn-only is restricted to crossing-study with standard B)\n"
                   " --kv-gamma G --pair-limiter scalar|density-vector (legacy only)\n"
                   " --inflow vacuum|isotropic|time-dependent|beam|crossing|matching|"
                   "smooth-transport|crossing-transport|crossing-three-transport|crossing-study-transport|crossing-3d-transport\n"
                   " --time T --dt DT --cfl C --anchor-scale 1|2 --mass-tolerance TOL\n"
                   " --mass-mode consistent|lumped|lumped_afc --mass-corrections R\n"
                   " --pulse-width W --background B --checkpoint-stride S\n"
                   " --domain-length L (line-source only: square of side L centred at (0.5,0.5))\n"
                   " --sigma-a A --sigma-s S --anisotropy g --source-density Q --material-jump\n"
                   " --memory-mib M --pn-norm-max-dofs K --threads T --ledger-stride S --deterministic --output "
                   "DIR\n"
                   " --cuda on|off --mcl-mode eager|lazy\n"
                   " --filter off|constant|nn --filter-strength S --filter-model FILE\n"
                   " --stabilization element-weno --degree p --weno on|off --weno-linear W --weno-q Q\n"
                   "   --element-limit on|off (uniform Cartesian hex mesh, SSPRK3, constant filter only)\n"
                   "   --angular-reduction auto|on|off (1D/2D: drop PN moments odd in the unresolved directions;\n"
                   "     auto = on when initial and inflow data are even, otherwise off)\n"
                   "   --mirror-reduction auto|off (open mesh, vacuum inflow, even elements, initial data mirror-\n"
                   "     symmetric about x_d = 1/2 in every direction: compute on [1/2,1]^d with mirror sides)\n"
#ifdef M1PN_FAST_DEFAULTS
                   "defaults: --stabilization m1pn_mass_correction --time-integrator ssprk2 --cfl 1 --dt 0\n"
                   "  --mass-mode lumped --kv-gamma 0 --threads <OpenMP maximum>\n"
#endif
                   ;
            std::exit(0);
        }
        if (key == "--periodic") {
            o.mesh.periodic = true;
            continue;
        }
        if (key == "--material-jump") {
            o.jump = true;
            continue;
        }
        if (key == "--deterministic") {
            o.deterministic = true;
            continue;
        }
        if (key == "--pn-only-self-test") {
            o.pn_only_self_test = true;
            continue;
        }
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
        if (key == "--filter") o.filter_mode = v;
        else if (key == "--filter-strength") o.filter_strength = real();
        else if (key == "--filter-model") o.filter_model = v;
        else if (key == "--pn-order")
            o.order = integer();
        else if (key == "--elements")
            o.mesh.elements = integer();
        else if (key == "--dimension")
            o.mesh.dimension = integer();
        else if (key == "--mesh")
            o.mesh.file = v;
        else if (key == "--mesh-type") {
            require(v == "hex" || v == "tet" || v == "single-tet", "Invalid mesh type");
            o.mesh.tetrahedra = v == "tet";
            o.mesh.single_tet = v == "single-tet";
        } else if (key == "--biharmonic-gamma" || key == "--ghost-gamma")
            o.num.gamma = real();
        else if (key == "--biharmonic-boundary") {
            require(v == "natural" || v == "masked", "Invalid biharmonic boundary");
            o.num.masked = v == "masked";
        } else if (key == "--boundary") {
            require(v == "onsager-c1" || v == "characteristic-c0", "Invalid boundary model");
            o.num.c0 = v == "characteristic-c0";
        } else if (key == "--time-integrator") {
            require(v == "ssprk2" || v == "ssprk3" || v == "ssprk43", "Invalid time integrator");
            o.rk = v == "ssprk2" ? 2 : (v == "ssprk3" ? 3 : 4); // 4 = SSPRK(4,3), element-weno only
            o.rk_explicit = true;
        } else if (key == "--model") {
            require(v == "r2" || v == "legacy", "Invalid model");
            o.num.r2 = v == "r2";
        } else if (key == "--stabilization") {
            require(v == "gradient-defect" || v == "biharmonic-original" ||
                    v == "biharmonic-corrected" || v == "m1pn_ghost_local_projected_p4" ||
                    v == "m1pn_mass_correction" || v == "element-weno", "Invalid stabilization");
            o.num.stabilization = v;
        } else if (key == "--pn-viscosity-scale")
            o.num.pn_viscosity_scale = real();
        else if (key == "--kv-gamma")
            o.num.kv_gamma = real();
        else if (key == "--method") {
            require(v == "m1pn-r2" || v == "m1pn-biharmonic-c1" ||
                    v == "m1pn_ghost_local_projected_p4", "Invalid method");
            o.num.r2 = v != "m1pn-biharmonic-c1";
            if (v == "m1pn_ghost_local_projected_p4") o.num.stabilization = v;
        }
        else if (key == "--case")
            o.problem = v;
        else if (key == "--coupling") {
            require(v == "m1pn" || v == "pn-only", "Invalid coupling mode");
            o.coupling = v;
        }
        else if (key == "--pair-limiter") {
            require(v == "scalar" || v == "density-vector", "Invalid pair limiter");
            o.pair_limiter = v;
            o.num.density_vector = v == "density-vector";
        }
        else if (key == "--inflow")
            o.inflow = v;
        else if (key == "--output")
            o.output = v;
        else if (key == "--time")
            o.final_time = real();
        else if (key == "--dt")
            o.dt = real();
        else if (key == "--cfl")
            o.cfl = real();
        else if (key == "--anchor-scale")
            o.num.anchor = real();
        else if (key == "--mass-tolerance")
            o.num.mass_tolerance = real();
        else if (key == "--mass-mode") {
            require(v == "consistent" || v == "lumped" || v == "lumped_afc", "Invalid mass mode");
            o.num.mass_mode = v;
            o.mass_mode_explicit = true;
        } else if (key == "--mass-corrections")
            o.num.mass_corrections = integer();
        else if (key == "--sigma-a")
            o.sigma_a = real();
        else if (key == "--sigma-s")
            o.sigma_s = real();
        else if (key == "--anisotropy")
            o.anisotropy = real();
        else if (key == "--source-density")
            o.source = real();
        else if (key == "--memory-mib") {
            double mib = real();
            require(mib > 0 && mib < 1e7, "Invalid memory budget");
            o.num.budget.bytes = std::size_t(mib * 1024 * 1024);
        } else if (key == "--pn-norm-max-dofs")
            o.norm_max = integer();
        else if (key == "--max-steps")
            o.max_steps = integer();
        else if (key == "--checkpoint-stride")
            o.checkpoint_stride = integer();
        else if (key == "--pulse-width")
            o.pulse_width = real();
        else if (key == "--background")
            o.background = real();
        else if (key == "--domain-length")
            o.domain_length = real();
        else if (key == "--threads")
            o.threads = integer();
        else if (key == "--ledger-stride") {
            o.num.ledger_stride = integer();
            require(o.num.ledger_stride >= 1, "--ledger-stride must be at least 1");
        }
        else if (key == "--cuda") {
            require(v == "on" || v == "off", "--cuda must be on or off");
            o.cuda_explicit = true;
            o.cuda_on = v == "on";
        } else if (key == "--degree") {
            o.degree = integer();
        } else if (key == "--weno-linear") {
            o.weno_linear = real();
        } else if (key == "--mirror-reduction") {
            require(v == "auto" || v == "off", "Invalid --mirror-reduction");
            o.mirror_reduction = v;
        } else if (key == "--angular-reduction") {
            require(v == "auto" || v == "on" || v == "off", "Invalid --angular-reduction");
            o.angular_reduction = v;
        } else if (key == "--weno-q") {
            o.weno_q = real();
        } else if (key == "--weno" || key == "--element-limit") {
            require(v == "on" || v == "off", key + " must be on or off");
            (key == "--weno" ? o.weno : o.element_limit) = v == "on";
        } else if (key == "--mcl-mode") {
            require(v == "eager" || v == "lazy", "Invalid MCL mode");
            o.num.lazy = v == "lazy";
        } else
            throw std::runtime_error("Unknown option: " + key);
    }
    require(o.filter_mode == "off" || o.filter_mode == "constant" || o.filter_mode == "nn",
            "Invalid --filter mode");
    require(o.filter_strength >= 0, "Negative filter strength");
    const bool element_weno = o.num.stabilization == "element-weno";
    if (o.filter_mode != "off") {
        require(o.mesh.dimension == 2 && o.num.r2, "The filter requires 2D R2");
        require(!o.cuda_on, "Filter experiment requires CPU evolution");
        require(!element_weno || o.filter_mode == "constant",
                "element-weno supports only the constant filter");
    }
    if (element_weno) {
        if (!o.rk_explicit) o.rk = 3;
        require(o.rk == 3 || o.rk == 4, "element-weno uses SSPRK3 or SSPRK(4,3)");
        require(o.coupling == "m1pn", "element-weno has no pn-only coupling");
        require(o.degree >= 1 && o.degree <= 12, "element-weno degree must lie in [1,12]");
        require(o.weno_linear > 0 && o.weno_linear < .5 && o.weno_q >= 1, "Invalid WENO weights");
    }
    if (o.filter_model.empty())
        o.filter_model = std::string(NN_FILTER_ROOT) + "/models/model_N" + std::to_string(o.order) + ".txt";
    require(o.final_time >= 0 && o.dt >= 0 && o.cfl > 0 && o.cfl <= 1 && o.threads > 0 &&
                o.max_steps > 0,
            "Invalid time/thread configuration");
    require(o.sigma_a >= 0 && o.sigma_s >= 0 && std::abs(o.anisotropy) <= 1 && o.source >= 0 &&
                o.num.mass_tolerance > 0,
            "Invalid reaction/tolerance configuration");
    require(o.num.anchor == 1 || o.num.anchor == 2, "Anchor scale must be 1 or 2");
    require(o.num.kv_gamma >= 0 && o.num.gamma >= 0, "Negative stabilization coefficient");
    require(o.num.mass_corrections >= 1 && o.num.mass_corrections <= 32,
            "Mass corrections must lie in [1,32]");
    if (o.num.stabilization == "m1pn_mass_correction") {
        require(o.num.r2, "PN mass correction requires --model r2");
        require(o.rk == 2 || o.rk == 3, "PN mass correction requires --time-integrator ssprk2|ssprk3");
        require(!o.mass_mode_explicit || o.num.mass_mode == "lumped",
                "PN mass correction requires --mass-mode lumped; conflicting explicit mass mode rejected");
        o.num.mass_mode = "lumped";
        require(o.num.pn_viscosity_scale >= 1 && std::isfinite(o.num.pn_viscosity_scale),
                "PN viscosity scale must be finite and at least one");
    }
    if (o.coupling == "pn-only") {
        require(o.num.r2 && o.num.stabilization == "m1pn_mass_correction",
                "PN-only coupling requires R2 with standard m1pn_mass_correction");
        require(o.rk == 3, "PN-only coupling requires SSPRK3");
        require(o.problem == "crossing-study" && o.mesh.dimension == 2 &&
                    o.inflow == "crossing-study-transport",
                "PN-only coupling is currently restricted to 2D crossing-study transport");
        require(o.sigma_a == 0 && o.sigma_s == 0 && o.source == 0 && o.anisotropy == 0 && !o.jump,
                "PN-only coupling requires zero material, source and jump terms");
    }
    require(!o.pn_only_self_test || o.coupling == "pn-only",
            "--pn-only-self-test requires --coupling pn-only");
    require(o.num.r2 || o.num.mass_mode == "consistent", "Lumped modes require --model r2");
    require(o.pulse_width > 0 && o.background >= 0, "Invalid pulse width/background");
    require(o.domain_length > 0 && std::isfinite(o.domain_length), "Invalid domain length");
    if (o.problem == "line-source")
        require(o.mesh.dimension == 2 && o.inflow == "vacuum", "line-source is 2D with vacuum inflow");
    if (o.domain_length != 1)
        require(o.problem == "line-source" && o.mesh.file.empty() && !o.mesh.periodic && !element_weno,
                "--domain-length needs line-source on a generated open mesh (not element-weno)");
    require(o.num.r2 || o.num.stabilization != "m1pn_ghost_local_projected_p4",
            "Local projected ghost stabilization requires --model r2");
    if (o.num.r2) {
        require(!o.num.density_vector, "R2 uses only the common scalar cone limiter");
        require(!o.num.masked, "R2 uses unmasked physical vertex stars");
        require(!o.num.c0, "R2 requires the full Onsager exterior construction");
    }
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
            a[0] = std::exp(-200 * (x[0] - .5) * (x[0] - .5)) / std::sqrt(4 * pi);
        else if (o.problem == "line-source") {
            // Garrett-Hauck line source: unit-mass Gaussian of width W plus floor B.
            const double r2 = (x[0] - .5) * (x[0] - .5) + (x[1] - .5) * (x[1] - .5);
            const double w2 = o.pulse_width * o.pulse_width;
            a[0] = (o.background + std::exp(-r2 / (2 * w2)) / (2 * pi * w2)) / std::sqrt(4 * pi);
        }
        else if (o.problem == "shadow" || o.problem == "lattice" || o.problem == "lattice-paul")
            a[0] = (o.problem=="lattice-paul"?o.background:.01) / std::sqrt(4 * pi);
        else
            throw std::runtime_error("Unknown case: " + o.problem);
        return a;
    }
}
Field initial(const Options &o, const Spatial &f, const Angular &pn) {
    Field u = Field::Zero(f.nodes, pn.q);
    for (int i = 0; i < f.nodes; ++i)
        u.row(i) = initial_point(o, f.coordinates.row(i).transpose(), i, f.dimension, pn).transpose();
    return u;
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
void write_field(const std::string &file, const Spatial &fe, const Field &u) {
    std::ofstream f(file);
    require(bool(f), "Cannot write field");
    f << std::setprecision(17) << "node,x,y,z";
    for (int k = 0; k < u.cols(); ++k)
        f << ",u" << k;
    f << "\n";
    for (int i = 0; i < u.rows(); ++i) {
        f << i;
        for (int d = 0; d < 3; ++d)
            f << "," << fe.coordinates(i, d);
        for (int k = 0; k < u.cols(); ++k)
            f << "," << u(i, k);
        f << "\n";
    }
}
void write_checkpoint(const std::string &dir,const Spatial &fe,const Field &u,int step,double time) {
    std::ostringstream name; name << "checkpoint_step_" << std::setw(6) << std::setfill('0') << step;
    const std::string stem=dir+"/"+name.str(), csv=stem+".csv", meta=stem+".json";
    const std::string csv_tmp=csv+".tmp", meta_tmp=meta+".tmp";
    write_field(csv_tmp,fe,u);
    std::filesystem::rename(csv_tmp,csv);
    { std::ofstream f(meta_tmp); require(bool(f),"Cannot write checkpoint metadata");
      f << std::setprecision(17) << "{\"step\":" << step << ",\"time\":" << time << "}\n"; }
    std::filesystem::rename(meta_tmp,meta);
}
// Element-based limiting with CG-WENO target on tensor Bernstein elements.
// Writes the same field files as the Q1 path; the Bernstein solution is
// evaluated on a uniform sample grid (nodes of Q1 with elements*degree cells).
int run_element_weno(const Options &o, const Angular &pn, double begin) {
    const int dim = o.mesh.dimension;
    require(o.mesh.file.empty() && !o.mesh.tetrahedra && !o.mesh.single_tet,
            "element-weno needs the generated Cartesian hex mesh");
    require(o.problem != "lattice" && o.problem != "lattice-paul" && o.problem != "shadow" &&
                o.source == 0 && !o.jump && o.anisotropy == 0,
            "element-weno supports only constant isotropic scattering and absorption, no sources");
    if (o.problem == "crossing-study" || o.inflow == "crossing-study-transport")
        require(dim == 2, "crossing-study is 2D");
    ElementWenoOptions eo;
    eo.degree = o.degree;
    eo.dimension = dim;
    eo.elements = o.mesh.elements;
    eo.periodic = o.mesh.periodic;
    eo.weno = o.weno;
    eo.limit = o.element_limit;
    eo.weno_linear = o.weno_linear;
    eo.weno_q = o.weno_q;
    eo.filter_strength = o.filter_mode == "constant" ? o.filter_strength : 0.;
    eo.sigma_s = o.sigma_s;
    eo.sigma_a = o.sigma_a;
    eo.ssprk43 = o.rk == 4;
    // Mirror reduction: the solution is symmetric about x_d = 1/2 in every direction when the initial data are
    // (checked at every node of the full grid, with the reflected PN moments) and the inflow is vacuum. Then
    // [1/2,1]^d with mirror sides at x_d = 1/2 gives the full solution (tests/element_weno.cpp, mirror_quarter).
    bool mirrored = false;
    double mirror_asymmetry = 0;
    if (o.mirror_reduction == "auto" && !o.mesh.periodic && o.inflow == "vacuum" && o.mesh.elements % 2 == 0 &&
        o.sigma_s >= 0) {
        const int per = o.mesh.elements * o.degree + 1;
        int total = 1;
        for (int d = 0; d < dim; ++d) total *= per;
        double scale = 0;
        for (int i = 0; i < total; ++i) {
            V3 x = V3::Zero();
            for (int d = 0, k = i; d < dim; ++d, k /= per) x[d] = double(k % per) / (per - 1);
            const Vec u = initial_point(o, x, i, dim, pn);
            scale = std::max(scale, u.cwiseAbs().maxCoeff());
            for (int d = 0; d < dim; ++d) {
                V3 y = x;
                y[d] = 1 - x[d];
                mirror_asymmetry = std::max(mirror_asymmetry,
                                            (reflect_pn(pn, initial_point(o, y, i, dim, pn), d) - u).cwiseAbs().maxCoeff());
            }
        }
        mirrored = mirror_asymmetry <= 1e-14 * std::max(1., scale);
        if (mirrored) {
            eo.elements = o.mesh.elements / 2;
            eo.length = .5;
            for (int d = 0; d < dim; ++d) {
                eo.origin[d] = .5;
                eo.mirror[2 * d] = true;
            }
        } else
            std::cerr << "note: mirror reduction off (initial data asymmetry " << mirror_asymmetry << ")\n";
    }
    double domain_factor = 1; // full mass / computed mass
    for (int d = 0; d < dim && mirrored; ++d) domain_factor *= 2;
    // Angular parity reduction: exact when the initial and inflow data are even in the unresolved directions.
    // auto checks both (initial coefficients of every node, inflow datum term at nine times in [0, T]).
    eo.reduce_angular = dim < 3 && o.angular_reduction != "off";
    require(!(o.angular_reduction == "on" && dim == 3), "--angular-reduction on needs dimension 1 or 2");
    auto scheme_ptr = std::make_unique<ElementWeno>(pn, eo, make_inflow(o, dim));
    std::vector<Vec> initial(scheme_ptr->nodes);
    for (int i = 0; i < scheme_ptr->nodes; ++i) {
        initial[i] = initial_point(o, scheme_ptr->node_position(i), i, dim, pn);
        cone_require(exact::cone(project(initial[i])), "Initial Bernstein coefficient outside the M1 cone");
    }
    double odd_initial = 0, odd_inflow = 0;
    if (eo.reduce_angular) {
        double scale = 0, dropped = 0;
        std::vector<char> kept(pn.q, 0);
        for (int k : scheme_ptr->active) kept[k] = 1;
        for (const Vec &u : initial)
            for (int k = 0; k < pn.q; ++k) (kept[k] ? scale : dropped) = std::max(kept[k] ? scale : dropped, std::abs(u[k]));
        odd_initial = scale > 0 ? dropped / scale : dropped;
        for (int k = 0; k <= 8; ++k)
            odd_inflow = std::max(odd_inflow, scheme_ptr->dropped_inflow(o.final_time * k / 8.));
        const bool even = odd_initial <= 1e-13 && odd_inflow <= 1e-12;
        require(even || o.angular_reduction != "on",
                "--angular-reduction on: initial or inflow data are not even in the unresolved directions");
        if (!even) {
            std::cerr << "note: angular reduction off (odd data: initial " << odd_initial << ", inflow " << odd_inflow << ")\n";
            eo.reduce_angular = false;
            scheme_ptr = std::make_unique<ElementWeno>(pn, eo, make_inflow(o, dim));
        }
    }
    const ElementWeno &scheme = *scheme_ptr;
    Field a(scheme.nodes, scheme.moments);
    for (int i = 0; i < scheme.nodes; ++i)
        a.row(i) = scheme.reduce(initial[i]).transpose();
    std::vector<Vec>().swap(initial);
    const double selected = o.dt > 0 ? o.dt : o.cfl * scheme.step_bound();
    require(selected <= scheme.realizability_step() * (1 + 1e-12), "Requested dt violates the element realizability bound");
    if (selected > scheme.high_order_step())
        std::cerr << "warning: dt exceeds the estimated high-order stability step " << scheme.high_order_step() << "\n";
    // Uniform sample grid (Q1 nodes of elements*degree cells per axis).
    const int cells_axis = o.mesh.elements * o.degree, points_axis = cells_axis + (o.mesh.periodic ? 0 : 1);
    int samples = 1;
    for (int d = 0; d < dim; ++d) samples *= points_axis;
    std::vector<V3> where(samples);
    Vec sample_mass(samples);
    for (int s = 0; s < samples; ++s) {
        V3 x = V3::Zero();
        double m = 1;
        int k = s;
        for (int d = 0; d < dim; ++d, k /= points_axis) {
            const int index = k % points_axis;
            x[d] = double(index) / cells_axis;
            const bool edge = !o.mesh.periodic && (index == 0 || index == cells_axis);
            m *= (edge ? .5 : 1.) / cells_axis;
        }
        where[s] = x;
        sample_mass[s] = m;
    }
    auto write_samples = [&](const std::string &file, const Field &state) {
        std::ofstream f(file);
        require(bool(f), "Cannot write field");
        f << std::setprecision(17) << "node,x,y,z";
        for (int k = 0; k < pn.q; ++k) f << ",u" << k;
        f << "\n";
        for (int s = 0; s < samples; ++s) {
            // Mirror reduction: evaluate at the reflected point and reflect the moments back.
            V3 x = where[s];
            std::array<bool, 3> flip{};
            for (int d = 0; d < dim && mirrored; ++d)
                if (x[d] < .5) x[d] = 1 - x[d], flip[d] = true;
            const Vec active_values = scheme.evaluate(state, x);
            Vec u = Vec::Zero(pn.q);
            for (int k = 0; k < scheme.moments; ++k) u[scheme.active[k]] = active_values[k];
            for (int d = 0; d < dim; ++d)
                if (flip[d]) u = reflect_pn(pn, u, d);
            f << s << "," << where[s][0] << "," << where[s][1] << "," << where[s][2];
            for (int k = 0; k < pn.q; ++k) f << "," << u[k];
            f << "\n";
        }
    };
    write_samples(o.output + "/initial.csv", a);
    {
        std::ofstream masses(o.output + "/nodal_mass.csv");
        masses << std::setprecision(17) << "node,mass\n";
        for (int s = 0; s < samples; ++s) masses << s << "," << sample_mass[s] << "\n";
        std::ofstream config(o.output + "/config.json");
        config << std::setprecision(17) << "{\n\"method\":\"element-weno\",\"model\":\"r2\",\"coupling\":\"m1pn\""
               << ",\"stabilization\":\"element-weno\",\"pn_target\":\"consistent_mass_cg_weno\""
               << ",\"basis\":\"tensor_bernstein\",\"degree\":" << o.degree
               << ",\"weno\":" << (o.weno ? "true" : "false") << ",\"weno_linear\":" << o.weno_linear
               << ",\"weno_q\":" << o.weno_q << ",\"element_limit\":" << (o.element_limit ? "true" : "false")
               << ",\"filter_mode\":" << std::quoted(o.filter_mode) << ",\"filter_constant\":" << o.filter_strength
               << ",\"sigma_s\":" << o.sigma_s << ",\"sigma_a\":" << o.sigma_a
               << ",\"reaction_integration\":\"exact_exponential_half_steps_around_ssprk\""
               << ",\"pn_order\":" << pn.order << ",\"Q\":" << pn.q << ",\"dimension\":" << dim
               << ",\"mirror_reduction\":" << (mirrored ? "true" : "false")
               << ",\"mirror_reduction_mode\":" << std::quoted(o.mirror_reduction)
               << ",\"mirror_asymmetry\":" << mirror_asymmetry << ",\"computed_elements_per_axis\":" << eo.elements
               << ",\"angular_reduction\":" << (eo.reduce_angular ? "true" : "false")
               << ",\"angular_reduction_mode\":" << std::quoted(o.angular_reduction)
               << ",\"active_moments\":" << scheme.moments << ",\"odd_initial_fraction\":" << odd_initial
               << ",\"odd_inflow_fraction\":" << odd_inflow
               << ",\"elements_per_axis\":" << o.mesh.elements << ",\"coefficients\":" << scheme.nodes
               << ",\"sample_points\":" << samples << ",\"sample_cells_per_axis\":" << cells_axis
               << ",\"periodic\":" << (o.mesh.periodic ? "true" : "false") << ",\"rk\":3"
               << ",\"time_integrator\":" << (eo.ssprk43 ? "\"ssprk43\"" : "\"ssprk3\"")
               << ",\"case\":" << std::quoted(o.problem) << ",\"inflow\":" << std::quoted(o.inflow)
               << ",\"final_time\":" << o.final_time << ",\"dt_requested\":" << o.dt << ",\"dt_selected\":" << selected
               << ",\"cfl_safety\":" << o.cfl << ",\"element_dt_bound\":" << scheme.dt_bound
               << ",\"realizability_step\":" << scheme.realizability_step()
               << ",\"high_order_step\":" << scheme.high_order_step()
               << ",\"threads\":" << o.threads << beam_metadata(o, dim)
               << ",\"compiler\":" << std::quoted(__VERSION__) << "\n}\n";
    }
    auto progress = [&](const char *status, double time, int count) {
        const std::string tmp = o.output + "/progress.json.tmp", dst = o.output + "/progress.json";
        std::ofstream f(tmp);
        f << std::setprecision(17) << "{\"status\":" << std::quoted(status) << ",\"time\":" << time
          << ",\"steps\":" << count << ",\"dt\":" << selected << ",\"wall_seconds\":" << seconds() - begin << "}\n";
        f.close();
        std::filesystem::rename(tmp, dst);
    };
    ElementWenoStats stats;
    double t = 0, solver_seconds = 0, min_density = std::numeric_limits<double>::infinity(), min_margin = min_density;
    int steps = 0;
    const double initialization = seconds() - begin;
    progress("running", t, steps);
    while (t < o.final_time) {
        require(steps < o.max_steps, "Maximum step count exceeded");
        double dt = std::min(selected, o.final_time - t);
        const double start = seconds();
        a = scheme.advance(t, dt, a, &stats);
        solver_seconds += seconds() - start;
        t = (o.final_time - (t + dt) <= 8 * std::numeric_limits<double>::epsilon() * o.final_time) ? o.final_time : t + dt;
        ++steps;
        if (steps % 10 == 0) progress("running", t, steps);
    }
    for (int i = 0; i < scheme.nodes; ++i) {
        const Four u = project(a.row(i).transpose());
        min_density = std::min(min_density, u[0]);
        min_margin = std::min(min_margin, margin(u));
    }
    write_samples(o.output + "/final.csv", a);
    progress("completed", t, steps);
    double mass0 = 0;
    for (int i = 0; i < scheme.nodes; ++i) mass0 += scheme.lumped[i] * project(a.row(i).transpose())[0];
    mass0 *= domain_factor;
    struct rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    std::ofstream result(o.output + "/summary.json");
    result << std::setprecision(17) << "{\"status\":\"completed\",\"method\":\"element-weno\",\"steps\":" << steps
           << ",\"time\":" << t << ",\"total_seconds\":" << seconds() - begin << ",\"solver_seconds\":" << solver_seconds
           << ",\"initialization_seconds\":" << initialization << ",\"peak_rss_kib_rank0\":" << usage.ru_maxrss
           << ",\"minimum_coefficient_density\":" << min_density << ",\"minimum_cone_margin\":" << min_margin
           << ",\"final_density_mass\":" << mass0 << ",\"min_weno_gamma\":" << stats.min_gamma
           << ",\"min_element_factor\":" << stats.min_limiter << ",\"limited_element_stages\":" << stats.limited_elements
           << ",\"limited_boundary_nodes\":" << stats.limited_boundary_nodes << ",\"nudged_states\":" << stats.nudged_states
           << ",\"max_limited_zero_sum\":" << stats.max_limited_zero_sum
           << ",\"phase_seconds\":{\"closure\":" << stats.t_closure << ",\"weno\":" << stats.t_weno
           << ",\"stabilization\":" << stats.t_stabilization << ",\"residual\":" << stats.t_residual
           << ",\"boundary\":" << stats.t_boundary << ",\"mass\":" << stats.t_mass << ",\"limiter\":" << stats.t_limiter
           << ",\"finish\":" << stats.t_finish << "}}\n";
    std::cout << "Completed " << steps << " steps, P" << pn.order << ", element-weno p=" << o.degree << ", "
              << scheme.nodes << " coefficients\n";
    return 0;
}
} // namespace
int main(int argc, char **argv) {
    // Large fields are reallocated every step; keep them in the heap and reuse
    // their pages instead of an mmap/munmap and page-fault cycle per allocation.
    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, 1 << 30);
#ifdef M1PN_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#endif
    Runtime world;
    try {
        exact::environment();
        double begin = seconds();
        Options o = parse(argc, argv);
        nnfilter::Filter filter(o.order, o.filter_mode, o.filter_strength, o.filter_model);
        if (o.cuda_explicit)
            cuda_set_requested(o.cuda_on);
        world.deterministic = o.deterministic;
#ifdef _OPENMP
        omp_set_num_threads(o.threads);
#else
        require(o.threads == 1, "This build has no OpenMP support");
#endif
        if (world.rank == 0) {
            require(!std::filesystem::exists(o.output),
                    "Output directory already exists: " + o.output);
            std::filesystem::create_directories(o.output);
        }
        Angular pn(o.order, o.num.budget);
        if (o.num.stabilization == "element-weno") {
            require(world.size == 1, "element-weno runs on a single MPI rank");
            const int code = run_element_weno(o, pn, begin);
#ifdef M1PN_MPI
            MPI_Finalize();
#endif
            return code;
        }
        const bool lattice=o.problem=="lattice-paul";
        if(lattice){
            require(o.mesh.elements>0 && o.mesh.elements%7==0, "lattice-paul requires elements-per-axis divisible by 7");
            const long double cells=static_cast<long double>(o.mesh.elements)*o.mesh.elements;
            const long double nodes=static_cast<long double>(o.mesh.elements+1)*(o.mesh.elements+1);
            require(cells*6<std::numeric_limits<int>::max() && nodes<std::numeric_limits<int>::max(), "Mesh index overflow");
            o.num.budget.check(cells*4096.L+nodes*8192.L, "Mesh/FE preallocation estimate");
            require(o.mesh.dimension==2 && !o.mesh.periodic && o.mesh.file.empty() && !o.mesh.tetrahedra && !o.mesh.single_tet,
                    "lattice-paul requires a generated nonperiodic 2D quadrilateral mesh");
            require(o.inflow=="vacuum", "lattice-paul requires vacuum inflow");
            require(o.sigma_a==0 && o.sigma_s==0 && o.source==0 && !o.jump && o.anisotropy==0,
                    "lattice-paul owns its material and source coefficients");
            require(std::abs(o.background - 1e-10) <= 1e-16,
                    "lattice-paul requires the Paul initial density --background 1e-10");
            auto scaled=make_mesh(o.mesh);
            for(int i=0;i<scaled->GetNV();++i){double *v=scaled->GetVertex(i);v[0]*=7;v[1]*=7;}
            o.mesh.file=o.output+"/input.mesh";
            if(world.rank==0){std::ofstream mf(o.mesh.file); require(bool(mf),"Cannot write scaled lattice mesh"); scaled->Print(mf); mf.close();}
            world.barrier();
        }
        if(o.domain_length!=1){
            auto scaled=make_mesh(o.mesh); const double L=o.domain_length;
            for(int i=0;i<scaled->GetNV();++i){double *v=scaled->GetVertex(i);v[0]=.5+L*(v[0]-.5);v[1]=.5+L*(v[1]-.5);}
            o.mesh.file=o.output+"/input.mesh";
            if(world.rank==0){std::ofstream mf(o.mesh.file); require(bool(mf),"Cannot write scaled line-source mesh"); scaled->Print(mf); mf.close();}
            world.barrier();
        }
        if(o.problem=="crossing-study" || o.inflow=="crossing-study-transport") require(o.mesh.dimension==2,"crossing-study is 2D");
        if(o.problem=="crossing-3d" || o.inflow=="crossing-3d-transport") require(o.mesh.dimension==3,"crossing-3d is 3D");
        double assembly_start = seconds();
        Spatial fe(o.mesh, o.num.anchor, o.num.budget, o.num.mass_mode == "consistent");
        double assembly = seconds() - assembly_start;
        Transport tr(pn, fe, o.num, make_inflow(o, fe.dimension), world);
        TimeControl control(tr, o.mesh, o.rk, o.norm_max);
        const bool pn_only = o.coupling == "pn-only";
        Field u = initial(o, fe, pn), u0 = u;
        if (pn_only) pn_only_require_finite(u, fe, pn);
        else tr.check_state(u);
        if (o.pn_only_self_test) {
            const Stage coupled_oracle = tr.evaluate(.0123, u);
            const PNOnlyStage uncoupled_oracle = evaluate_pn_only(tr, .0123, u);
            const double oracle_target_difference =
                maxabs(coupled_oracle.target - uncoupled_oracle.stage.target);
            require(oracle_target_difference < 2e-12,
                    "PN-only target differs from the standard-B coupled oracle on admissible state");
            Field nonrealizable = Field::Zero(fe.nodes, pn.q);
            for (int i = 0; i < fe.nodes; ++i) {
                Vec row = nonrealizable.row(i).transpose();
                set_low(row, Four(.25, .5, 0., 0.));
                for (int k = 4; k < pn.q; ++k)
                    row[k] = .001 * std::sin((i + 1.) * (k + 2.));
                nonrealizable.row(i) = row.transpose();
            }
            const PNOnlyStage probe = evaluate_pn_only(tr, 0., nonrealizable);
            require(probe.state.outside_m1_cone_nodes == fe.nodes,
                    "PN-only self-test state did not lie outside the M1 cone");
            require(probe.stage.rate.allFinite() && probe.maximum_target_recovery < 2e-10,
                    "PN-only nonrealizable derivative/oracle check failed");
            if (world.rank == 0)
                std::cout << "{\"status\":\"passed\",\"suite\":\"pn_only\",\"oracle_target_difference\":"
                          << std::setprecision(17) << oracle_target_difference
                          << ",\"outside_cone_nodes\":"
                          << probe.state.outside_m1_cone_nodes << ",\"target_recovery\":"
                          << std::setprecision(17) << probe.maximum_target_recovery << "}\n";
#ifdef M1PN_MPI
            MPI_Finalize();
#endif
            return 0;
        }
        Vec absorption = Vec::Constant(fe.nodes, o.sigma_a),
            scattering = Vec::Constant(fe.nodes, o.sigma_s);
        Field src = Field::Zero(fe.nodes, pn.q);
        src.col(0).setConstant(o.source / std::sqrt(4 * pi));
        Vec source_density=Vec::Constant(fe.nodes,o.source);
        if(lattice){
            absorption.setZero(); scattering.setZero(); source_density.setZero();
            mfem::Array<int> vdofs; mfem::Vector shape;
            for(int e=0;e<fe.mesh->GetNE();++e){
                const auto *el=fe.space->GetFE(e); auto *T=fe.mesh->GetElementTransformation(e);
                fe.space->GetElementDofs(e,vdofs); shape.SetSize(el->GetDof());
                const auto &ir=mfem::IntRules.Get(el->GetGeomType(),6);
                for(int q=0;q<ir.GetNPoints();++q){const auto &ip=ir.IntPoint(q); mfem::Vector x(fe.dimension); T->SetIntPoint(&ip); T->Transform(ip,x); el->CalcShape(ip,shape);
                    const int ix=int(std::floor(x[0])),iy=int(std::floor(x[1]));
                    const bool absorber=((ix==1||ix==5)&&(iy==1||iy==3||iy==5))||((ix==2||ix==4)&&(iy==2||iy==4))||(ix==3&&iy==1);
                    const double w=ip.weight*T->Weight(); const bool srcbox=x[0]>=3&&x[0]<=4&&x[1]>=3&&x[1]<=4;
                    for(int j=0;j<el->GetDof();++j){int n=vdofs[j]; absorption[n]+=w*shape[j]*(absorber?10:0); scattering[n]+=w*shape[j]*(absorber?0:1); source_density[n]+=w*shape[j]*(srcbox?1:0);}
                }
            }
            absorption.array()/=fe.mass.array(); scattering.array()/=fe.mass.array(); source_density.array()/=fe.mass.array();
            src.col(0)=source_density/std::sqrt(4*pi);
            if(world.rank==0){std::ofstream mat(o.output+"/material.csv"); mat<<std::setprecision(17)<<"node,sigma_a,sigma_s,source_density\n";for(int i=0;i<fe.nodes;++i)mat<<i<<","<<absorption[i]<<","<<scattering[i]<<","<<source_density[i]<<"\n";}
        }
        for (int i = 0; i < fe.nodes; ++i) {
            const V3 x = fe.coordinates.row(i).transpose();
            if (o.jump && x[0] > .5) {
                absorption[i] *= 4;
                scattering[i] *= 2;
            }
            if (o.problem == "shadow" && (x - V3(.5, .5, .5)).norm() < .2)
                absorption[i] += 10;
            if (o.problem == "lattice" && int(x[0] * 5) % 2 == 0 && int(x[1] * 5) % 2 == 0)
                absorption[i] += 5;

        }
        const bool splitting =
            absorption.maxCoeff() > 0 || scattering.maxCoeff() > 0 || source_density.maxCoeff() > 0;
        double selected = o.dt > 0 ? o.dt : o.cfl * std::min(fe.cfl_step, control.step_limit);
        if (o.final_time > 0) {
            control.check(std::min(selected, o.final_time));
            require(std::min(selected, o.final_time) <= fe.cfl_step,
                    "Requested dt violates M1 CFL");
        }
        std::ofstream stage_log, step_log;
        if (world.rank == 0) {
            write_field(o.output + "/initial.csv", fe, u);
            std::ofstream config(o.output + "/config.json");
            config << std::setprecision(17)
                   << "{\n\"method\":" << std::quoted(o.num.r2 ?
                       (o.num.stabilization == "m1pn_ghost_local_projected_p4" ?
                        "m1pn_ghost_local_projected_p4" :
                        (o.num.stabilization == "m1pn_mass_correction" ?
                         "m1pn_mass_correction" : "m1pn-r2")) : "m1pn-biharmonic-c1")
                   << ",\"model\":\"r2\""
                   << ",\"coupling\":" << std::quoted(o.coupling)
                   << ",\"coupling_mode\":" << std::quoted(o.coupling)
                   << ",\"m1_coupling_enabled\":" << (pn_only ? "false" : "true")
                   << ",\"m1_closure_enabled\":" << (pn_only ? "false" : "true")
                   << ",\"pair_limiter_enabled\":" << (pn_only ? "false" : "true")
                   << ",\"cone_rejection_enabled\":" << (pn_only ? "false" : "true")
                   << ",\"pn_target\":\"standard_B_mass_corrected\""
                   << ",\"filter_mode\":" << std::quoted(o.filter_mode)
                   << ",\"filter_constant\":" << o.filter_strength
                   << ",\"filter_model\":" << std::quoted(o.filter_model)
                   << ",\"filter_order\":4,\"filter_density_coefficient\":0"
                   << ",\"filter_integration\":\"symmetric_exponential_midpoint_splitting\""
                   << ",\"filter_features\":\"upstream_2d_spatial_sample_normalized_native_fe_derivatives\""
                   << ",\"pn_order\":" << pn.order
                   << ",\"Q\":" << pn.q << ",\"dimension\":" << fe.dimension
                   << ",\"elements_per_axis\":" << o.mesh.elements << ",\"nodes\":" << fe.nodes
                   << ",\"pairs\":" << fe.pairs.size() << ",\"faces\":" << fe.faces.size()
                   << ",\"mesh_file\":" << std::quoted(o.mesh.file) << ",\"mesh_type\":"
                   << std::quoted(o.mesh.single_tet   ? "single-tet"
                                  : o.mesh.tetrahedra ? "tet"
                                                      : "hex")
                   << ",\"periodic\":" << (o.mesh.periodic ? "true" : "false")
                   << ",\"gamma\":" << o.num.gamma << ",\"stabilization\":"
                   << std::quoted(o.num.r2 ? o.num.stabilization :
                                  (o.num.masked ? "biharmonic-masked" : "biharmonic-natural"))
                   << ",\"kv_gamma\":" << o.num.kv_gamma
                   << ",\"pn_viscosity_scale\":" << o.num.pn_viscosity_scale
                   << ",\"pn_mass_correction_applications\":"
                   << (o.num.stabilization == "m1pn_mass_correction" ? 1 : 0)
                   << ",\"uses_pn_low_order_predictor\":"
                   << (o.num.stabilization == "m1pn_mass_correction" ? "true" : "false")
                   << ",\"boundary_flux\":" << std::quoted(o.num.r2 ? "LLF_lambda_1" : "characteristic_PN_kinetic_M1")
                   << ",\"m1_exterior_policy\":" << std::quoted(o.num.r2 ? "incoming_half_range_moments" : "kinetic_inflow")
                   << ",\"m1_exterior_rounding\":" << std::quoted(o.num.r2 ? "outward_density_ulp_until_exact_cone" : "unused")
                   << ",\"length_scale\":\"maximum_element_diameter_in_node_star\",\"boundary\":"
                   << std::quoted(o.num.c0 ? "characteristic-c0" : "onsager-c1")
                   << ",\"anchor_scale\":" << o.num.anchor << ",\"rk\":" << o.rk
                   << ",\"case\":" << std::quoted(o.problem) << ",\"geometry_policy\":" << std::quoted(lattice?"lattice_paul_floor_cells_domain_0_7":(o.domain_length!=1?"centred_square_side_domain_length":"unit_domain")) << ",\"domain_length\":" << o.domain_length << ",\"material_average_policy\":" << std::quoted(lattice?"positive_fe_quadrature_average_order6":"nodal_constant") << ",\"lattice_material_policy\":" << std::quoted(lattice?"absorber_sigma_a_10_else_sigma_s_1_source_unit_on_3_4_square":"none") << ",\"lattice_integrated_expected\":" << (lattice?"{\"domain_mass\":49,\"source_mass\":1,\"sigma_a_mass\":110,\"sigma_s_mass\":38}":"null")
                   << ",\"pair_limiter\":" << std::quoted(o.num.r2 ? "cone-only" : o.pair_limiter)
                   << ",\"inflow\":" << std::quoted(o.inflow) << ",\"final_time\":" << o.final_time
                   << ",\"time_accumulation\":\"kahan_compensated_accepted_dt_with_8ulp_endpoint_snap\""
                   << ",\"dt_requested\":" << o.dt << ",\"dt_selected\":" << selected
                   << ",\"checkpoint_stride\":" << o.checkpoint_stride
                   << ",\"cfl_safety\":" << o.cfl << ",\"low_cfl_limit\":" << fe.cfl_step
                   << ",\"pn_step_limit\":" << control.step_limit
                   << ",\"pn_time_control\":" << std::quoted(control.method)
                   << ",\"pn_operator_norm_bound\":" << control.operator_bound
                   << ",\"pn_norm_max_dofs\":" << o.norm_max
                   << ",\"mass_mode\":" << std::quoted(o.num.mass_mode)
                   << ",\"mass_corrections\":" << (o.num.mass_mode == "lumped_afc" ? o.num.mass_corrections : 0)
                   << ",\"mass_factorized\":" << (fe.factorized ? "true" : "false")
                   << ",\"mass_solver\":" << std::quoted(o.num.mass_mode == "consistent" ?
                       "Eigen_SimplicialLDLT_consistent" :
                       (o.num.mass_mode == "lumped" ? "lumped_diagonal" : "local_diagonal_and_sparse_edge_correction"))
                   << ",\"mass_tolerance\":"
                   << o.num.mass_tolerance
                   << ",\"mass_equivalence_lower_bound\":" << fe.mass_equivalence
                   << ",\"sigma_a\":" << o.sigma_a << ",\"sigma_s\":" << o.sigma_s
                   << ",\"anisotropy\":" << o.anisotropy << ",\"source_density\":" << (lattice?1.0:o.source)
                   << ",\"material_jump\":" << (o.jump ? "true" : "false")
                   << ",\"reaction_discretization\":\"nodal_HG_exponential_Strang\",\"memory_"
                      "budget_bytes\":"
                   << o.num.budget.bytes << ",\"max_steps\":" << o.max_steps
                   << ",\"threads\":" << o.threads
                   << ",\"cuda\":" << (cuda_requested() ? "true" : "false")
                   << ",\"ledger_stride\":" << o.num.ledger_stride
                   << ",\"fast_mass_correction_path\":" << (tr.fast_mc ? "true" : "false")
                   << ",\"mpi_ranks\":" << world.size
                   << ",\"parallel_layout\":\"replicated_mesh_state_partitioned_pairs_faces\","
                      "\"deterministic\":"
                   << (o.deterministic ? "true" : "false")
                   << beam_metadata(o,fe.dimension)
                   << ",\"compiler\":" << std::quoted(__VERSION__)
                   << ",\"mfem_version\":" << MFEM_VERSION << "\n}\n";
            std::ofstream mesh_file(o.output + "/mesh.mesh");
            fe.mesh->Print(mesh_file);
            std::ofstream masses(o.output + "/nodal_mass.csv");
            masses << std::setprecision(17) << "node,mass\n";
            for (int i = 0; i < fe.nodes; ++i)
                masses << i << "," << fe.mass[i] << "\n";
            stage_log.open(o.output + "/stages.csv");
            stage_log << "step,stage,mass_residual,target_recovery,local_balance,projection,high_"
                         "rate_error,target_energy,sync_work,coupled_energy,min_alpha,min_eta,min_"
                         "bar,active_pairs,active_faces,min_euler_state,cfl_ratio,density_clip_sum,"
                         "momentum_redirection_sum,density_retention_sum,pair_target_passes,"
                         "pair_rounding_fallbacks,pair_scalar_fallbacks,base_bar_reevaluations\n";
            step_log.open(o.output + "/steps.csv");
            step_log << "step,time,dt,rate_residual,ledger_residual,min_margin\n";
            stage_log << std::setprecision(17);
            step_log << std::setprecision(17);
        }
        double initialization = seconds() - begin, solver_seconds = 0, t = 0, time_compensation = 0, max_balance = 0,
               min_margin = std::numeric_limits<double>::infinity();
        double max_mass_residual = 0, max_sync_work = 0, min_sync_work = 0;
        long active_pair_total = 0, active_face_total = 0, stage_total = 0;
        long closure_evaluations_total = 0, lazy_certified_total = 0, lazy_fallback_total = 0;
        double density_clip_total = 0, momentum_redirection_total = 0, density_retention_total = 0;
        double minimum_density_observed = std::numeric_limits<double>::infinity();
        double minimum_cone_margin_observed = std::numeric_limits<double>::infinity();
        long negative_density_node_stage_observations = 0;
        long outside_m1_cone_node_stage_observations = 0;
        double pn_only_target_recovery_max = 0;
        double pn_only_rate_residual_max = 0;
        auto observe_pn_state = [&](const PNOnlyStateDiagnostics &obs) {
            minimum_density_observed = std::min(minimum_density_observed, obs.minimum_density);
            minimum_cone_margin_observed = std::min(minimum_cone_margin_observed,
                                                     obs.minimum_cone_margin);
            negative_density_node_stage_observations += obs.negative_density_nodes;
            outside_m1_cone_node_stage_observations += obs.outside_m1_cone_nodes;
        };
        long pair_target_pass_total = 0, pair_rounding_fallback_total = 0,
             pair_scalar_fallback_total = 0, base_bar_reevaluation_total = 0;
        int steps = 0;
        Field boundary = Field::Zero(fe.nodes, pn.q), reaction_ledger = boundary, filter_ledger = boundary;
        int rejected = 0;
        auto progress = [&](const char *status, double time, int count, double step_dt) {
            if (world.rank != 0) return;
            const std::string tmp=o.output+"/progress.json.tmp", dst=o.output+"/progress.json";
            std::ofstream f(tmp);
            f << std::setprecision(17) << "{\"status\":" << std::quoted(status)
              << ",\"time\":" << time << ",\"steps\":" << count
              << ",\"dt\":" << step_dt << ",\"wall_seconds\":" << seconds()-begin << "}\n";
            f.close(); std::filesystem::rename(tmp,dst);
        };
        progress("running",t,steps,selected);
        std::ofstream rejection_log;
        if (world.rank == 0) {
            rejection_log.open(o.output + "/rejections.csv");
            rejection_log << "step,attempt,dt,reason\n";
        }
        Field filtered, trial_filter;
        while (t < o.final_time) {
            require(steps < o.max_steps, "Maximum step count exceeded");
            double dt = std::min(selected, o.final_time - t);
            Step step;
            PNOnlyStep pn_only_step;
            Field candidate, trial_reaction;
            bool accepted = false;
            for (int attempt = 0; attempt <= 12; ++attempt) {
                try {
                    const bool filtering = filter.active();
                    if (filtering) {
                        nnfilter::reset_ledger(trial_filter, fe.nodes, pn.q);
                        filter.apply_into(filtered,pn,fe,u,absorption,scattering,src,dt/2,trial_filter);
                        if (!pn_only) nnfilter::check_state_parallel(filtered);
                    }
                    const Field &before_reaction = filtering ? filtered : u;
                    if (splitting) {
                        trial_reaction = Field::Zero(fe.nodes, pn.q);
                        candidate = reaction(before_reaction, src, absorption, scattering, o.anisotropy, dt / 2);
                        trial_reaction += fe.mass.asDiagonal() * (candidate - before_reaction);
                    }
                    const Field &stage_input = splitting ? candidate : before_reaction;
                    const double step_start = seconds();
                    if (pn_only) {
                        pn_only_step = advance_pn_only(tr, control, t, dt, stage_input);
                        step = pn_only_step.step;
                    } else {
                        step = advance(tr, control, t, dt, stage_input);
                    }
                    solver_seconds += seconds() - step_start;
                    candidate = std::move(step.state);
                    if (splitting) {
                        Field after =
                            reaction(candidate, src, absorption, scattering, o.anisotropy, dt / 2);
                        trial_reaction += fe.mass.asDiagonal() * (after - candidate);
                        candidate = std::move(after);
                    }
                    if (filtering)
                        filter.apply_into(candidate,pn,fe,candidate,absorption,scattering,src,dt/2,trial_filter);
                    if (pn_only) pn_only_require_finite(candidate, fe, pn);
                    else if (filtering) nnfilter::check_state_parallel(candidate);
                    else if (splitting) tr.check_state(candidate);
                    accepted = true;
                    break;
                } catch (const StepRejected &error) {
                    ++rejected;
                    if (world.rank == 0)
                        rejection_log << steps + 1 << "," << attempt << "," << std::setprecision(17)
                                      << dt << "," << std::quoted(error.what()) << "\n";
                    dt *= .5;
                    require(t + dt > t, "Step rejection reached time resolution");
                }
            }
            if (!accepted) {
                Field saved_boundary = boundary;
                world.sum(saved_boundary);
                if (world.rank == 0) {
                    write_field(o.output + "/rollback_state.csv", fe, u);
                    write_field(o.output + "/rollback_boundary.csv", fe, saved_boundary);
                    write_field(o.output + "/rollback_reaction.csv", fe, reaction_ledger);
                    std::ofstream failure(o.output + "/rejection.json");
                    failure << std::setprecision(17)
                            << "{\"status\":\"rejected\",\"last_accepted_time\":" << t
                            << ",\"accepted_steps\":" << steps
                            << ",\"rejected_attempts\":" << rejected << "}\n";
                    rejection_log.flush();
                    stage_log.flush();
                    step_log.flush();
                }
                progress("rejected",t,steps,dt);
                // All checkpoint streams have closed/flushed before any rank can abort.
                world.barrier();
                throw std::runtime_error("Cone step rejected after 13 conservative retries; state "
                                         "and ledgers unchanged");
            }
            // Commit only an entirely accepted Strang/SSP step.
            u = std::move(candidate);
            if (pn_only) {
                for (const auto &obs : pn_only_step.observations) observe_pn_state(obs);
                pn_only_target_recovery_max = std::max(pn_only_target_recovery_max,
                                                        pn_only_step.maximum_target_recovery);
                pn_only_rate_residual_max = std::max(pn_only_rate_residual_max,
                                                      step.rate_residual);
            }
            if (splitting) reaction_ledger += trial_reaction;
            if (filter.active()) nnfilter::add_ledger(filter_ledger, trial_filter);
            boundary += step.integrated_boundary;
            selected = std::min(selected, dt);
            const double corrected_dt = dt - time_compensation;
            const double next_time = t + corrected_dt;
            time_compensation = (next_time - t) - corrected_dt;
            t = next_time;
            const double endpoint_scale = std::max({std::abs(o.final_time), std::abs(dt), 1e-300});
            const double endpoint_tolerance = 8 * std::numeric_limits<double>::epsilon() * endpoint_scale;
            if (std::abs(o.final_time - t) <= endpoint_tolerance) {
                t = o.final_time;
                time_compensation = 0;
            }
            ++steps;
            max_balance = std::max(max_balance, step.ledger_residual);
            min_margin = std::min(min_margin, step.minimum_margin);
            if (pn_only || splitting) {
                double smallest = std::numeric_limits<double>::infinity();
#pragma omp parallel for schedule(static) reduction(min : smallest)
                for (int i = 0; i < fe.nodes; ++i)
                    smallest = std::min(smallest, margin(project(u.row(i).transpose())));
                min_margin = std::min(min_margin, smallest);
            }
            for (const auto &d : step.stages) {
                max_mass_residual = std::max(max_mass_residual, d.mass_residual);
                max_sync_work = std::max(max_sync_work, d.sync_work);
                min_sync_work = std::min(min_sync_work, d.sync_work);
                active_pair_total += d.active_pairs;
                active_face_total += d.active_faces;
                density_clip_total += d.density_clip_sum;
                momentum_redirection_total += d.momentum_redirection_sum;
                density_retention_total += d.density_retention_sum;
                pair_target_pass_total += d.pair_target_passes;
                pair_rounding_fallback_total += d.pair_rounding_fallbacks;
                pair_scalar_fallback_total += d.pair_scalar_fallbacks;
                base_bar_reevaluation_total += d.base_bar_reevaluations;
                closure_evaluations_total += d.closure_evaluations;
                lazy_certified_total += d.lazy_certified_pairs;
                lazy_fallback_total += d.lazy_fallback_pairs;
                ++stage_total;
                min_margin = std::min(min_margin, d.minimum_state);
            }
            if (world.rank == 0) {
                step_log << steps << "," << t << "," << dt << "," << step.rate_residual << ","
                         << step.ledger_residual << "," << step.minimum_margin << "\n";
                for (std::size_t r = 0; r < step.stages.size(); ++r) {
                    const auto &d = step.stages[r];
                    stage_log << steps << "," << r << "," << d.mass_residual << ","
                              << d.target_recovery << "," << d.local_balance << "," << d.projection
                              << "," << d.high_rate_error << "," << d.target_energy << ","
                              << d.sync_work << "," << d.target_energy + d.sync_work << ","
                              << d.minimum_alpha << "," << d.minimum_eta << "," << d.minimum_bar
                              << "," << d.active_pairs << "," << d.active_faces << ","
                              << d.minimum_state << "," << d.cfl_ratio << ","
                              << d.density_clip_sum << "," << d.momentum_redirection_sum
                              << "," << d.density_retention_sum << "," << d.pair_target_passes
                              << "," << d.pair_rounding_fallbacks << ","
                              << d.pair_scalar_fallbacks << "," << d.base_bar_reevaluations << "\n";
                }
                if(steps%10==0){ step_log.flush(); stage_log.flush(); rejection_log.flush(); progress("running",t,steps,dt); }
                if(o.checkpoint_stride>0 && steps%o.checkpoint_stride==0){
                    step_log.flush(); stage_log.flush();
                    write_checkpoint(o.output,fe,u,steps,t);
                }
            }
        }
        world.sum(boundary);
        Vec global = (fe.mass.asDiagonal() * (u - u0) + boundary - reaction_ledger - filter_ledger)
                         .colwise()
                         .sum()
                         .transpose();
        if (world.rank == 0) {
            progress("completed",t,steps,selected);
            write_field(o.output + "/final.csv", fe, u);
            write_field(o.output + "/boundary_integral.csv", fe, boundary);
            write_field(o.output + "/reaction_integral.csv", fe, reaction_ledger);
            write_field(o.output + "/filter_integral.csv", fe, filter_ledger);
            const Vec final_strength = filter.strength(pn,fe,u,absorption,scattering,src);
            std::ofstream sf(o.output + "/filter_strength.csv");
            sf << std::setprecision(17) << "node,sigma_f\n";
            for (int i=0; i<fe.nodes; ++i) sf << i << "," << final_strength[i] << "\n";
            struct rusage usage{};
            getrusage(RUSAGE_SELF, &usage);
            std::ofstream result(o.output + "/summary.json");
            if (steps == 0) {
                min_margin = std::numeric_limits<double>::infinity();
                for (int i = 0; i < fe.nodes; ++i)
                    min_margin = std::min(min_margin, margin(project(u.row(i).transpose())));
            }
            const auto final_pn_diagnostics = pn_only_state_diagnostics(u, fe, pn);
            if (!pn_only) {
                minimum_density_observed = std::min(
                    pn_only_state_diagnostics(u0, fe, pn).minimum_density,
                    final_pn_diagnostics.minimum_density);
                minimum_cone_margin_observed = min_margin;
            } else if (!std::isfinite(minimum_density_observed)) {
                minimum_density_observed = final_pn_diagnostics.minimum_density;
                minimum_cone_margin_observed = final_pn_diagnostics.minimum_cone_margin;
            }
            result << std::setprecision(17) << "{\"status\":\"completed\",\"steps\":" << steps
                   << ",\"time\":" << t << ",\"total_seconds\":" << seconds() - begin
                   << ",\"filter_seconds_including_retries\":" << filter.elapsed
                   << ",\"filter_evaluations_including_retries\":" << filter.evaluations
                   << ",\"filter_density_ledger_max\":" << maxabs(filter_ledger.col(0))
                   << ",\"final_filter_strength_min\":" << final_strength.minCoeff()
                   << ",\"final_filter_strength_max\":" << final_strength.maxCoeff()
                   << ",\"solver_seconds\":" << solver_seconds
                   << ",\"assembly_seconds\":" << assembly
                   << ",\"initialization_seconds\":" << initialization
                   << ",\"mass_seconds\":" << tr.timings.mass
                   << ",\"pn_predictor_seconds\":" << tr.timings.pn_predictor
                   << ",\"pn_mass_correction_seconds\":" << tr.timings.pn_mass_correction
                   << ",\"biharmonic_seconds\":" << tr.timings.biharmonic
                   << ",\"boundary_seconds\":" << tr.timings.boundary
                   << ",\"mcl_seconds\":" << tr.timings.mcl
                   << ",\"communication_seconds\":" << tr.timings.communication
                   << ",\"peak_rss_kib_rank0\":" << usage.ru_maxrss
                   << ",\"rejected_attempts\":" << rejected
                   << ",\"mass_solve_calls\":" << fe.mass_solve_calls
                   << ",\"m1_closure_evaluations\":" << (pn_only ? 0 : closure_evaluations_total)
                   << ",\"pair_limiter_evaluations\":" << (pn_only ? 0 : active_pair_total)
                   << ",\"cone_rejections\":" << (pn_only ? 0 : rejected)
                   << ",\"minimum_density_observed\":" << minimum_density_observed
                   << ",\"minimum_cone_margin_observed\":" << minimum_cone_margin_observed
                   << ",\"negative_density_node_stage_observations\":"
                   << negative_density_node_stage_observations
                   << ",\"outside_m1_cone_node_stage_observations\":"
                   << outside_m1_cone_node_stage_observations
                   << ",\"final_negative_density_nodes\":"
                   << final_pn_diagnostics.negative_density_nodes
                   << ",\"final_outside_m1_cone_nodes\":"
                   << final_pn_diagnostics.outside_m1_cone_nodes
                   << ",\"pn_only_target_recovery_max\":" << pn_only_target_recovery_max
                   << ",\"pn_only_rate_residual_max\":" << pn_only_rate_residual_max
                   << ",\"mass_iterations\":" << (o.num.mass_mode == "lumped_afc" ? o.num.mass_corrections : 0)
                   << ",\"mass_iterations_note\":" << std::quoted(o.num.mass_mode == "consistent" ?
                       "direct_factorization" : "fixed_local_corrections_per_stage") << ","
                      "\"minimum_cone_margin\":"
                   << min_margin << ",\"maximum_step_balance\":" << max_balance
                   << ",\"maximum_mass_residual\":" << max_mass_residual
                   << ",\"active_pair_evaluations\":" << active_pair_total
                   << ",\"active_face_evaluations\":" << active_face_total
                   << ",\"density_clip_sum\":" << density_clip_total
                   << ",\"momentum_redirection_sum\":" << momentum_redirection_total
                   << ",\"density_retention_sum\":" << density_retention_total
                   << ",\"pair_target_passes\":" << pair_target_pass_total
                   << ",\"pair_rounding_fallbacks\":" << pair_rounding_fallback_total
                   << ",\"pair_scalar_fallbacks\":" << pair_scalar_fallback_total
                   << ",\"base_bar_reevaluations\":" << base_bar_reevaluation_total
                   << ",\"lazy_certified_pairs\":" << lazy_certified_total
                   << ",\"lazy_fallback_pairs\":" << lazy_fallback_total
                   << ",\"mcl_mode\":" << std::quoted(o.num.lazy ? "lazy" : "eager")
                   << ",\"active_pair_fraction\":"
                   << (stage_total ? double(active_pair_total) / (stage_total * fe.pairs.size())
                                   : 0.)
                   << ",\"sync_work_min_including_zero\":" << min_sync_work
                   << ",\"sync_work_max_including_zero\":" << max_sync_work
                   << ",\"global_Q_balance\":" << global.lpNorm<Eigen::Infinity>() << "}\n";
            std::cout << "Completed " << steps << " steps, P" << pn.order << ", " << fe.nodes
                      << " nodes, Q balance " << global.lpNorm<Eigen::Infinity>() << "\n";
        }
#ifdef M1PN_MPI
        MPI_Finalize();
#endif
        return 0;
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
