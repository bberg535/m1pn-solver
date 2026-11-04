#include "time.hpp"
#include <filesystem>
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
    MeshSpec mesh;
    Numerics num;
    int order = 3, rk = 3, max_steps = 100000, norm_max = 512, threads = 1, checkpoint_stride = 0;
    double final_time = .01, dt = 0, cfl = .8, sigma_a = 0, sigma_s = 0, anisotropy = 0, source = 0;
    double pulse_width = .07, background = .001;
    bool jump = false, deterministic = false, mass_mode_explicit = false;
    std::string exterior_policy = "kinetic";
    double exterior_density = 1, pulse_duration = .06;
    std::optional<V3> vacuum_velocity;
    std::string problem = "smooth", inflow = "vacuum", output = std::string(PN_AFC_ROOT) + "/tmp/run", pair_limiter = "scalar";
};
Options parse(int argc, char **argv) {
    Options o;
    o.num.r2 = true;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help") {
            std::cout
                << "m1pn --pn-order N --elements n --dimension 1|2|3 --mesh-type "
                   "hex|tet|single-tet\n"
                   " --periodic | --mesh FILE --biharmonic-gamma G --biharmonic-boundary "
                   "natural|masked\n"
                   " --boundary onsager-c1|characteristic-c0 --time-integrator ssprk2|ssprk3\n"
                   " --case "
                   "smooth|stress|isotropic|vacuum|pure-beam|beam|crossing|plane-source|shadow|"
                   "lattice|smooth-beam|vacuum-beam|oblique-beam|crossing-three\n"
                   " --model r2|legacy --stabilization gradient-defect|biharmonic-original|biharmonic-corrected\n"
                   " --stabilization m1pn_ghost_local_projected_p4 --ghost-gamma G (open Cartesian Q1 only)\n"
                   " --stabilization m1pn_mass_correction --pn-viscosity-scale L (SSPRK3, lumped mass)\n"
                   " --stabilization m1pn_mass_correction_boundary_consistent (uniform open Cartesian P1/Q1)\n"
                   " --kv-gamma G --pair-limiter scalar|density-vector (legacy only)\n"
                   " --inflow vacuum|isotropic|time-dependent|beam|crossing|matching|"
                   "smooth-transport|crossing-transport|crossing-three-transport\n"
                   " --time T --dt DT --cfl C --anchor-scale 1|2 --mass-tolerance TOL\n"
                   " --mass-mode consistent|lumped|lumped_afc --mass-corrections R\n"
                   " --mcl-mode eager|lazy --geometry-cache on|off\n"
                   " --time-integrator imex-euler --implicit-solver direct|defect|pcg\n"
                   " --implicit-rtol R --implicit-atol A --implicit-max-iterations K\n"
                   " --pn-boundary-quadrature lumped|consistent --bh-pair-mode element|canonical|factorized\n"
                   " --m1-exterior kinetic|isotropic|copied --exterior-density R --vacuum-velocity x,y,z\n"
                   " --inflow pulse|affine --case affine --pulse-duration T\n"
                   " --pulse-width W --background B --checkpoint-stride S\n"
                   " --sigma-a A --sigma-s S --anisotropy g --source-density Q --material-jump\n"
                   " --memory-mib M --pn-norm-max-dofs K --threads T --deterministic --output "
                   "DIR\n";
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
        if (key == "--pn-order")
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
            require(v == "ssprk2" || v == "ssprk3" || v == "imex-euler", "Invalid time integrator");
            o.rk = v == "imex-euler" ? 1 : (v == "ssprk2" ? 2 : 3);
        } else if (key == "--model") {
            require(v == "r2" || v == "legacy", "Invalid model");
            o.num.r2 = v == "r2";
        } else if (key == "--stabilization") {
            require(v == "gradient-defect" || v == "biharmonic-original" ||
                    v == "biharmonic-corrected" || v == "m1pn_ghost_local_projected_p4" ||
                    v == "m1pn_mass_correction" || v == "m1pn_mass_correction_boundary_consistent", "Invalid stabilization");
            o.num.stabilization = v;
        } else if (key == "--kv-gamma")
            o.num.kv_gamma = real();
        else if (key == "--pn-viscosity-scale") o.num.pn_viscosity_scale=real();
        else if (key == "--method") {
            require(v == "m1pn-r2" || v == "m1pn-biharmonic-c1" ||
                    v == "m1pn_ghost_local_projected_p4", "Invalid method");
            o.num.r2 = v != "m1pn-biharmonic-c1";
            if (v == "m1pn_ghost_local_projected_p4") o.num.stabilization = v;
        }
        else if (key == "--case")
            o.problem = v;
        else if (key == "--pair-limiter") {
            require(v == "scalar" || v == "density-vector", "Invalid pair limiter");
            o.pair_limiter = v;
            o.num.density_vector = v == "density-vector";
        }
        else if (key == "--mcl-mode") {
            require(v == "eager" || v == "lazy", "Invalid MCL mode"); o.num.lazy = v == "lazy";
        } else if (key == "--geometry-cache") {
            require(v == "on" || v == "off", "Invalid geometry cache mode"); o.num.geometry_cache = v == "on";
        } else if (key == "--implicit-solver") {
            require(v == "direct" || v == "defect" || v == "pcg", "Invalid implicit solver"); o.num.implicit_solver = v;
        } else if (key == "--implicit-rtol") o.num.implicit_rtol = real();
        else if (key == "--implicit-atol") o.num.implicit_atol = real();
        else if (key == "--implicit-max-iterations") o.num.implicit_max_iterations = integer();
        else if (key == "--pn-boundary-quadrature") {
            require(v == "lumped" || v == "consistent", "Invalid PN boundary quadrature"); o.num.pn_boundary_quadrature = v;
        } else if (key == "--bh-pair-mode") {
            require(v == "element" || v == "canonical" || v == "factorized", "Invalid BH pair mode"); o.num.bh_pair_mode = v;
        } else if (key == "--m1-exterior") {
            require(v == "kinetic" || v == "isotropic" || v == "copied", "Invalid exterior policy"); o.exterior_policy = v;
        } else if (key == "--exterior-density") o.exterior_density = real();
        else if (key == "--pulse-duration") o.pulse_duration = real();
        else if (key == "--vacuum-velocity") {
            V3 vel; std::istringstream input(v); char c1, c2;
            require(bool(input >> vel[0] >> c1 >> vel[1] >> c2 >> vel[2]) && c1 == ',' && c2 == ',' && input.peek() == EOF, "Velocity requires x,y,z");
            require(vel.allFinite() && vel.norm() <= 1, "Invalid vacuum exterior velocity"); o.vacuum_velocity = vel;
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
        else if (key == "--threads")
            o.threads = integer();
        else
            throw std::runtime_error("Unknown option: " + key);
    }
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
    if(o.num.stabilization=="m1pn_mass_correction" || o.num.stabilization=="m1pn_mass_correction_boundary_consistent") {
        require(o.num.r2,"PN mass correction requires --model r2");
        require(o.rk==3,"PN mass correction requires --time-integrator ssprk3");
        require(!o.mass_mode_explicit || o.num.mass_mode=="lumped",
                "PN mass correction requires --mass-mode lumped; conflicting explicit mass mode rejected");
        o.num.mass_mode="lumped";
        require(o.num.pn_viscosity_scale>=1&&std::isfinite(o.num.pn_viscosity_scale),"PN viscosity scale must be finite and at least one");
    }
    require(o.num.r2 || o.num.mass_mode == "consistent", "Lumped modes require --model r2");
    require(o.num.r2 || !(o.num.lazy || o.num.geometry_cache), "Lazy and geometry cache require --model r2");
    require(o.rk != 1 || o.num.mass_mode == "consistent", "IMEX uses --implicit-solver; select --mass-mode consistent");
    require(o.pulse_width > 0 && o.background >= 0, "Invalid pulse width/background");
    require(o.exterior_density >= 0 && o.pulse_duration > 0, "Invalid exterior/pulse data");
    require(o.num.implicit_rtol > 0 && o.num.implicit_atol >= 0 && o.num.implicit_max_iterations > 0, "Invalid implicit residual policy");
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
    if (o.exterior_policy != "kinetic") {
        g.exterior_state = [o](const V3 &x, double t, const V3 &, const Four &inside) {
            double rho = o.exterior_density;
            if (o.inflow == "pulse" && !(x[0] < 1e-12 && t < o.pulse_duration)) rho = 0;
            if (o.exterior_policy == "copied") return copied_velocity_exterior(rho, inside, o.vacuum_velocity);
            return Four(rho, 0, 0, 0);
        };
    }
    if (o.inflow == "vacuum") return g;
    if (o.inflow == "pulse")
        g.density = [o](const V3 &x, double t, const V3 &) {
            return x[0] < 1e-12 && t < o.pulse_duration ? o.exterior_density / (4*pi) : 0.;
        };
    else if (o.inflow == "affine")
        g.density = [dimension](const V3 &x, double t, const V3 &w) {
            V3 r(1, dimension > 1 ? .5 : 0., dimension > 2 ? -.3 : 0.);
            return (1 + .12*r.dot(x) - .12*t*r.dot(w))/(4*pi);
        };
    else if (o.inflow == "isotropic")
        g.density = [](const V3 &, double, const V3 &) { return 1 / (4 * pi); };
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
        g.atoms = [z = dimension == 2 ? 0. : .5](const V3 &x, double t) {
            return std::vector<Atom>{
                {V3::UnitX(), std::exp(-50 * (x - t * V3::UnitX() - V3(.3, .5, z)).squaredNorm())},
                {V3::UnitY(), std::exp(-50 * (x - t * V3::UnitY() - V3(.5, .3, z)).squaredNorm())}};
        };
    } else if (o.inflow == "crossing-three-transport") {
        const double bg=o.background, w=o.pulse_width;
        g.density = [bg](const V3 &, double, const V3 &) { return bg / (4 * pi); };
        g.atoms = [z = dimension == 2 ? 0. : .5, w](const V3 &x, double t) {
            auto pulse=[w](const V3 &d){return std::exp(-d.squaredNorm()/(2*w*w));};
            return std::vector<Atom>{
                {V3::UnitX(), pulse(x-t*V3::UnitX()-V3(.32,.5,z))},
                {V3::UnitY(), pulse(x-t*V3::UnitY()-V3(.5,.32,z))},
                {-V3::UnitX(), pulse(x+t*V3::UnitX()-V3(.68,.5,z))}};
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
Field initial(const Options &o, const Spatial &f, const Angular &pn) {
    Field u = Field::Zero(f.nodes, pn.q);
    for (int i = 0; i < f.nodes; ++i) {
        V3 x = f.coordinates.row(i).transpose();
        Vec a = Vec::Zero(pn.q);
        if (o.problem == "vacuum") {
        } else if (o.problem == "affine") {
            V3 r(1, f.dimension > 1 ? .5 : 0., f.dimension > 2 ? -.3 : 0.);
            set_low(a, Four(1 + .12*r.dot(x), 0, 0, 0));
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
            const double z = f.dimension == 2 && o.problem == "crossing" ? 0. : .5;
            a[0] = .01 / std::sqrt(4 * pi);
            a += std::exp(-50 * (x - V3(.3, .5, z)).squaredNorm()) *
                 harmonics(pn.order, V3::UnitX());
            if (o.problem == "crossing")
                a += std::exp(-50 * (x - V3(.5, .3, z)).squaredNorm()) *
                     harmonics(pn.order, V3::UnitY());
        } else if (o.problem == "crossing-three") {
            const double z = f.dimension == 2 ? 0. : .5;
            auto pulse=[&](const V3 &d){return std::exp(-d.squaredNorm()/(2*o.pulse_width*o.pulse_width));};
            a[0] = o.background / std::sqrt(4 * pi);
            a += pulse(x-V3(.32,.5,z)) * harmonics(pn.order, V3::UnitX());
            a += pulse(x-V3(.5,.32,z)) * harmonics(pn.order, V3::UnitY());
            a += pulse(x-V3(.68,.5,z)) * harmonics(pn.order, -V3::UnitX());
        } else if (o.problem == "plane-source")
            a[0] = std::exp(-200 * (x[0] - .5) * (x[0] - .5)) / std::sqrt(4 * pi);
        else if (o.problem == "shadow" || o.problem == "lattice")
            a[0] = .01 / std::sqrt(4 * pi);
        else
            throw std::runtime_error("Unknown case: " + o.problem);
        u.row(i) = a.transpose();
    }
    return u;
}
std::string beam_metadata(const Options &o,int dimension) {
    const bool three=o.problem=="crossing-three" || o.inflow=="crossing-three-transport";
    const bool two=o.problem=="crossing" || o.inflow=="crossing-transport";
    std::ostringstream out;
    out << std::setprecision(17);
    out << ",\"beam_count\":" << (three?3:(two?2:0));
    if(three || two) {
        const double z=dimension==2?0.:.5;
        const double bg=three?o.background:.01, width=three?o.pulse_width:std::sqrt(1./100.);
        out << ",\"pulse_width\":" << width << ",\"background\":" << bg
            << ",\"beam_geometry\":{\"background\":" << bg << ",\"amplitude\":1,\"width_exponent\":" << (three?-1/(2*width*width):-50.) << ",\"beams\":["
            << "{\"center\":[" << (three?.32:.3) << ",0.5," << z << "],\"direction\":[1,0,0]},"
            << "{\"center\":[0.5," << (three?.32:.3) << "," << z << "],\"direction\":[0,1,0]}";
        if(three) out << ",{\"center\":[0.68,0.5," << z << "],\"direction\":[-1,0,0]}";
        out << "]}";
    }
    return out.str();
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
} // namespace
int main(int argc, char **argv) {
#ifdef M1PN_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#endif
    Runtime world;
    try {
        exact::environment();
        double begin = seconds();
        Options o = parse(argc, argv);
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
        double assembly_start = seconds();
        Spatial fe(o.mesh, o.num.anchor, o.num.budget, o.num.mass_mode == "consistent");
        double assembly = seconds() - assembly_start;
        Transport tr(pn, fe, o.num, make_inflow(o, fe.dimension), world);
        TimeControl control(tr, o.mesh, o.rk, o.norm_max);
        Field u = initial(o, fe, pn), u0 = u;
        tr.check_state(u);
        Vec absorption = Vec::Constant(fe.nodes, o.sigma_a),
            scattering = Vec::Constant(fe.nodes, o.sigma_s);
        Field src = Field::Zero(fe.nodes, pn.q);
        src.col(0).setConstant(o.source / std::sqrt(4 * pi));
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
            absorption.maxCoeff() > 0 || scattering.maxCoeff() > 0 || o.source > 0;
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
                        (o.num.stabilization == "m1pn_mass_correction" || o.num.stabilization == "m1pn_mass_correction_boundary_consistent") ? o.num.stabilization : "m1pn-r2") : "m1pn-biharmonic-c1")
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
                   << ((o.num.stabilization=="m1pn_mass_correction" || o.num.stabilization=="m1pn_mass_correction_boundary_consistent")?1:0)
                   << ",\"boundary_flux\":" << std::quoted(o.num.r2 ? "LLF_lambda_1" : "characteristic_PN_kinetic_M1")
                   << ",\"m1_exterior_policy\":" << std::quoted(o.exterior_policy)
                   << ",\"m1_exterior_rounding\":" << std::quoted(o.num.r2 ? (o.exterior_policy == "kinetic" ? "outward_density_ulp_until_exact_cone" : "reject_invalid_prescribed_state") : "unused")
                   << ",\"length_scale\":\"maximum_element_diameter_in_node_star\",\"boundary\":"
                   << std::quoted(o.num.c0 ? "characteristic-c0" : "onsager-c1")
                   << ",\"anchor_scale\":" << o.num.anchor << ",\"rk\":" << o.rk
                   << ",\"case\":" << std::quoted(o.problem)
                   << ",\"pair_limiter\":" << std::quoted(o.num.r2 ? "cone-only" : o.pair_limiter)
                   << ",\"inflow\":" << std::quoted(o.inflow) << ",\"final_time\":" << o.final_time
                   << ",\"dt_requested\":" << o.dt << ",\"dt_selected\":" << selected
                   << ",\"checkpoint_stride\":" << o.checkpoint_stride
                   << ",\"cfl_safety\":" << o.cfl << ",\"low_cfl_limit\":" << fe.cfl_step
                   << ",\"pn_step_limit\":" << control.step_limit
                   << ",\"pn_time_control\":" << std::quoted(control.method)
                   << ",\"pn_operator_norm_bound\":" << control.operator_bound
                   << ",\"pn_norm_max_dofs\":" << o.norm_max
                   << ",\"mcl_mode\":" << std::quoted(o.num.lazy ? "lazy" : "eager")
                   << ",\"geometry_cache\":" << (o.num.geometry_cache ? "true" : "false")
                   << ",\"implicit_solver\":" << std::quoted(o.num.implicit_solver)
                   << ",\"implicit_rtol\":" << o.num.implicit_rtol
                   << ",\"implicit_atol\":" << o.num.implicit_atol
                   << ",\"implicit_max_iterations\":" << o.num.implicit_max_iterations
                   << ",\"pn_boundary_quadrature\":" << std::quoted(o.num.pn_boundary_quadrature)
                   << ",\"bh_pair_mode\":" << std::quoted(o.num.bh_pair_mode)
                   << ",\"mass_mode\":" << std::quoted(o.num.mass_mode)
                   << ",\"mass_corrections\":" << (o.num.mass_mode == "lumped_afc" ? o.num.mass_corrections : 0)
                   << ",\"mass_factorized\":" << (fe.factorized ? "true" : "false")
                   << ",\"mass_solver\":" << std::quoted(o.num.mass_mode == "consistent" ?
                       "Eigen_SimplicialLDLT_consistent" : "local_diagonal_and_sparse_edge_correction")
                   << ",\"mass_tolerance\":"
                   << o.num.mass_tolerance
                   << ",\"mass_equivalence_lower_bound\":" << fe.mass_equivalence
                   << ",\"sigma_a\":" << o.sigma_a << ",\"sigma_s\":" << o.sigma_s
                   << ",\"anisotropy\":" << o.anisotropy << ",\"source_density\":" << o.source
                   << ",\"material_jump\":" << (o.jump ? "true" : "false")
                   << ",\"reaction_discretization\":\"nodal_HG_exponential_Strang\",\"memory_"
                      "budget_bytes\":"
                   << o.num.budget.bytes << ",\"max_steps\":" << o.max_steps
                   << ",\"threads\":" << o.threads << ",\"mpi_ranks\":" << world.size
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
                         "pair_rounding_fallbacks,pair_scalar_fallbacks,base_bar_reevaluations,lazy_certified_pairs,lazy_fallback_pairs,closure_evaluations,implicit_residual,implicit_relative_residual,implicit_solves\n";
            step_log.open(o.output + "/steps.csv");
            step_log << "step,time,dt,rate_residual,ledger_residual,min_margin\n";
            stage_log << std::setprecision(17);
            step_log << std::setprecision(17);
        }
        double initialization = seconds() - begin, solver_seconds = 0, t = 0, max_balance = 0,
               min_margin = std::numeric_limits<double>::infinity();
        double max_mass_residual = 0, max_sync_work = 0, min_sync_work = 0;
        long active_pair_total = 0, active_face_total = 0, stage_total = 0;
        double density_clip_total = 0, momentum_redirection_total = 0, density_retention_total = 0;
        long pair_target_pass_total = 0, pair_rounding_fallback_total = 0,
             pair_scalar_fallback_total = 0, base_bar_reevaluation_total = 0;
        long lazy_certified_total=0, lazy_fallback_total=0, closure_total=0, implicit_solve_total=0;
        double maximum_implicit_residual=0, maximum_implicit_relative_residual=0;
        int steps = 0;
        Field boundary = Field::Zero(fe.nodes, pn.q), reaction_ledger = boundary;
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
        while (t < o.final_time) {
            require(steps < o.max_steps, "Maximum step count exceeded");
            double dt = std::min(selected, o.final_time - t);
            Step step;
            Field candidate, trial_reaction;
            bool accepted = false;
            for (int attempt = 0; attempt <= 12; ++attempt) {
                try {
                    candidate = u;
                    trial_reaction = Field::Zero(fe.nodes, pn.q);
                    if (splitting) {
                        candidate = reaction(u, src, absorption, scattering, o.anisotropy, dt / 2);
                        trial_reaction += fe.mass.asDiagonal() * (candidate - u);
                    }
                    double solve_begin = seconds();
                    step = advance(tr, control, t, dt, candidate);
                    solver_seconds += seconds() - solve_begin;
                    candidate = step.state;
                    if (splitting) {
                        Field after =
                            reaction(candidate, src, absorption, scattering, o.anisotropy, dt / 2);
                        trial_reaction += fe.mass.asDiagonal() * (after - candidate);
                        candidate = std::move(after);
                    }
                    tr.check_state(candidate);
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
            reaction_ledger += trial_reaction;
            boundary += step.integrated_boundary;
            selected = std::min(selected, dt);
            t += dt;
            ++steps;
            max_balance = std::max(max_balance, step.ledger_residual);
            min_margin = std::min(min_margin, step.minimum_margin);
            for (int i = 0; i < fe.nodes; ++i)
                min_margin = std::min(min_margin, margin(project(u.row(i).transpose())));
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
                lazy_certified_total += d.lazy_certified_pairs;
                lazy_fallback_total += d.lazy_fallback_pairs;
                closure_total += d.closure_evaluations;
                implicit_solve_total += d.implicit_solves;
                maximum_implicit_residual=std::max(maximum_implicit_residual,d.implicit_residual);
                maximum_implicit_relative_residual=std::max(maximum_implicit_relative_residual,d.implicit_relative_residual);
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
        Vec global = (fe.mass.asDiagonal() * (u - u0) + boundary - reaction_ledger)
                         .colwise()
                         .sum()
                         .transpose();
        if (world.rank == 0) {
            progress("completed",t,steps,selected);
            write_field(o.output + "/final.csv", fe, u);
            write_field(o.output + "/boundary_integral.csv", fe, boundary);
            write_field(o.output + "/reaction_integral.csv", fe, reaction_ledger);
            struct rusage usage{};
            getrusage(RUSAGE_SELF, &usage);
            std::ofstream result(o.output + "/summary.json");
            if (steps == 0) {
                min_margin = std::numeric_limits<double>::infinity();
                for (int i = 0; i < fe.nodes; ++i)
                    min_margin = std::min(min_margin, margin(project(u.row(i).transpose())));
            }
            result << std::setprecision(17) << "{\"status\":\"completed\",\"steps\":" << steps
                   << ",\"time\":" << t << ",\"total_seconds\":" << seconds() - begin
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
                   << ",\"closure_evaluations\":" << closure_total
                   << ",\"implicit_solve_total\":" << implicit_solve_total
                   << ",\"maximum_implicit_residual\":" << maximum_implicit_residual
                   << ",\"maximum_implicit_relative_residual\":" << maximum_implicit_relative_residual
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
