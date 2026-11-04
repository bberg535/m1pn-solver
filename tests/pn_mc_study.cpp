#include "time.hpp"
#include <Eigen/Eigenvalues>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <memory>

using namespace m1pn;

namespace {
struct Config {
    std::string suite = "patches", variant = "A", output;
};

Numerics numerics(const std::string &variant) {
    Numerics n;
    n.r2 = true;
    n.pn_boundary_quadrature = "consistent";
    if (variant == "A" || variant == "standard") {
        n.stabilization = "biharmonic-corrected";
        n.gamma = .004;
    } else if (variant == "PN" || variant == "gamma0") {
        n.stabilization = "biharmonic-corrected";
        n.gamma = 0.;
    } else if (variant == "B" || variant == "mass-correction") {
        n.stabilization = "m1pn_mass_correction";
        n.mass_mode = "lumped";
        n.gamma = .004;
    } else if (variant == "C") {
        n.stabilization = "m1pn_mass_correction_boundary_consistent";
        n.mass_mode = "lumped";
        n.gamma = .004;
    } else {
        throw std::runtime_error("variant must be A, PN, B, or C");
    }
    return n;
}

double patch_value(const std::string &kind, const V3 &x, int dim) {
    double s = 0.;
    for (int d = 0; d < dim; ++d) {
        if (kind == "affine") s += (d + 1.) * (x[d] - .5);
        if (kind == "quadratic") s += (d + 1.) * (x[d] - .5) * (x[d] - .5);
        if (kind == "smooth") s += std::sin(2 * pi * (d + 1) * x[d]);
    }
    return 1. + .05 * s;
}

double patch_derivative(const std::string &kind, const V3 &x, int d) {
    if (kind == "affine") return .05 * (d + 1.);
    if (kind == "quadratic") return .1 * (d + 1.) * (x[d] - .5);
    if (kind == "smooth") return .1 * pi * (d + 1.) * std::cos(2 * pi * (d + 1) * x[d]);
    return 0.;
}

Field patch_state(const Spatial &fe, const Angular &pn, const std::string &kind) {
    Field u = Field::Zero(fe.nodes, pn.q);
    for (int i = 0; i < fe.nodes; ++i) {
        V3 x = fe.coordinates.row(i).transpose();
        u(i, 0) = patch_value(kind, x, fe.dimension) / std::sqrt(4 * pi);
    }
    return u;
}

int codimension(const Spatial &fe, int i) {
    const double tol = 1e-10;
    int result = 0;
    for (int d = 0; d < fe.dimension; ++d)
        if (std::abs(fe.coordinates(i, d)) < tol || std::abs(fe.coordinates(i, d) - 1.) < tol)
            ++result;
    return result;
}

Field assembled_consistent_pn_boundary(const Transport &tr, const Field &u, double time) {
    Field boundary_rhs = Field::Zero(tr.fe.nodes, tr.pn.q);
    for (const Face &face : tr.fe.faces) {
        for (const SurfacePoint &p : face.points) {
            const Boundary &b = *tr.boundaries[p.normal_id];
            const Incoming data = b.data(tr.inflow, p.x, time);
            Vec trace = Vec::Zero(tr.pn.q);
            for (std::size_t j = 0; j < face.nodes.size(); ++j)
                trace.noalias() += p.shape[j] * u.row(face.nodes[j]).transpose();
            const Vec exterior = b.exterior(trace, data.d);
            const Vec flux = .5 * (tr.pn.apply(p.normal, trace) + tr.pn.apply(p.normal, exterior))
                           - .5 * (exterior - trace);
            for (std::size_t j = 0; j < face.nodes.size(); ++j)
                boundary_rhs.row(face.nodes[j]) -= (p.weight * p.shape[j]) * flux.transpose();
        }
    }
    return boundary_rhs;
}

void patches(const std::string &variant) {
    const Numerics opts = numerics(variant);
    std::cout << "suite,variant,dimension,element_family,cells,patch,codimension,nodes,max_Z_normalized,max_target_error,max_rate_error,max_mass_residual\n";
    for (int dim : {1, 2, 3}) for (int cells : {4, 8, 16}) {
        if (dim == 3 && cells > 8) continue;
        MeshSpec mesh; mesh.dimension = dim; mesh.elements = cells;
        Spatial fe(mesh, 1., {}, opts.mass_mode == "consistent");
        Angular pn(dim == 1 ? 1 : 3);
        std::array<Vec, 3> stream_mode;
        for (int d = 0; d < dim; ++d) stream_mode[d] = pn.A[d] * Vec::Unit(pn.q, 0);
        for (const std::string kind : {"constant", "affine", "quadratic", "smooth"}) {
            Inflow in;
            in.density = [kind, dim](const V3 &x, double, const V3 &) { return patch_value(kind, x, dim) / (4 * pi); };
            in.exterior_m1 = [kind, dim](const V3 &x, double) { return Four(patch_value(kind, x, dim), 0., 0., 0.); };
            Transport tr(pn, fe, opts, in);
            Field u = patch_state(fe, pn, kind);
            Stage stage = tr.evaluate(0., u, true);
            Field z;
            if (tr.stabilization_operator) z = tr.stabilization_operator->apply(u);
            else {
                const Field boundary_rhs = assembled_consistent_pn_boundary(tr, u, 0.);
                const Field low_rate = tr.pn_low_order_rate(u, boundary_rhs);
                z = tr.mass_correction(low_rate);
            }
            Field exact = Field::Zero(fe.nodes, pn.q);
            // The isotropic coefficient streams into all angular modes.
            for (int i = 0; i < fe.nodes; ++i) {
                exact.row(i).setZero();
                V3 x = fe.coordinates.row(i).transpose();
                for (int d = 0; d < dim; ++d)
                    exact.row(i).noalias() -= patch_derivative(kind, x, d) / std::sqrt(4 * pi) * stream_mode[d].transpose();
            }
            std::array<double, 4> zmax{}, terr{}, rerr{}, mres{};

            for (int i = 0; i < fe.nodes; ++i) {
                const int c = codimension(fe, i);
                zmax[c] = std::max(zmax[c], maxabs(z.row(i)) / fe.mass[i]);
                terr[c] = std::max(terr[c], maxabs(stage.target.row(i) - exact.row(i)));
                rerr[c] = std::max(rerr[c], maxabs(stage.rate.row(i) - exact.row(i)));
                mres[c] = std::max(mres[c], stage.diagnostics.mass_residual);
            }
            for (int c = 0; c <= dim; ++c) {
                int count = 0;
                for (int i = 0; i < fe.nodes; ++i) if (codimension(fe, i) == c) ++count;
                std::cout << "patches," << variant << ',' << dim << ',' << (dim == 1 ? "P1" : "Q1") << ',' << cells << ',' << kind << ',' << c << ','
                          << count << ','
                          << std::setprecision(16) << zmax[c] << ',' << terr[c] << ',' << rerr[c] << ',' << mres[c] << '\n';
            }
        }
    }
}

struct Wave {
    Vec vector;
    double speed = 0.;
};
Wave fastest_wave(const Angular &pn) {
    Eigen::SelfAdjointEigenSolver<Mat> eig(Mat(pn.A[0]));
    require(eig.info() == Eigen::Success, "PN streaming eigensystem failed");
    int k = 0;
    for (int j = 1; j < pn.q; ++j)
        if (std::abs(eig.eigenvalues()[j]) > std::abs(eig.eigenvalues()[k])) k = j;
    return {eig.eigenvectors().col(k), eig.eigenvalues()[k]};
}

Field wave_state(const Spatial &fe, const Angular &pn, const Wave &w, double t, double eps) {
    Field u = Field::Zero(fe.nodes, pn.q);
    for (int i = 0; i < fe.nodes; ++i) {
        const double ph = 2 * pi * (fe.coordinates(i, 0) - w.speed * t);
        u(i, 0) = 1. / std::sqrt(4 * pi);
        u.row(i).noalias() += eps * std::cos(ph) * w.vector.transpose();
    }
    return u;
}

Inflow wave_inflow(const Wave &w, double eps) {
    Inflow in;
    auto coeff = [w, eps](const V3 &x, double t) {
        Vec v = Vec::Zero(w.vector.size());
        v[0] = 1. / std::sqrt(4 * pi);
        v.noalias() += eps * std::cos(2 * pi * (x[0] - w.speed * t)) * w.vector;
        return v;
    };
    in.density = [coeff](const V3 &x, double t, const V3 &omega) {
        Vec a = coeff(x, t);
        Vec y = harmonics(static_cast<int>(std::sqrt(a.size())) - 1, omega);
        const double value = a.dot(y);
        require(value >= -1e-12 && std::isfinite(value), "Exact wave produced negative/nonfinite incoming kinetic density");
        return value;
    };
    in.exterior_m1 = [coeff](const V3 &x, double t) { return project(coeff(x, t)); };
    return in;
}

double pn_error(const Spatial &fe, const Field &a, const Field &b, int norm) {
    double sum = 0., maximum = 0., volume = 0.;
    for (int i = 0; i < fe.nodes; ++i) {
        const double e = (a.row(i) - b.row(i)).norm();
        maximum = std::max(maximum, e);
        sum += fe.mass[i] * (norm == 1 ? e : e * e);
        volume += fe.mass[i];
    }
    if (norm == 0) return maximum;
    return norm == 1 ? sum / volume : std::sqrt(sum / volume);
}

void convergence(const std::string &variant) {
    const Numerics opts = numerics(variant);
    std::cout << "suite,variant,boundary,cells,steps,dt,time,pn_l1,pn_l2,pn_linf,density_l1,density_l2,density_linf,active_pairs,mass_balance,elapsed_seconds\n";
    for (bool periodic : {true, false}) for (int cells : {16, 32, 64}) {
        MeshSpec mesh; mesh.dimension = 1; mesh.elements = cells; mesh.periodic = periodic;
        Spatial fe(mesh, 1., {}, opts.mass_mode == "consistent");
        Angular pn(3);
        Wave wave = fastest_wave(pn);
        constexpr double T = .05, eps = .001, target_dt = 1e-4;
        const int steps = static_cast<int>(std::ceil(T / target_dt));
        const double dt = T / steps;
        Transport tr(pn, fe, opts, wave_inflow(wave, eps));
        TimeControl control(tr, mesh, 3);
        control.check(dt);
        Field state = wave_state(fe, pn, wave, 0., eps);
        const double t0 = seconds();
        long active = 0; double balance = 0.;
        for (int k = 0; k < steps; ++k) {
            Step s = advance(tr, control, k * dt, dt, state);
            for (const Diagnostics &d : s.stages) active += d.active_pairs;
            balance = std::max(balance, s.ledger_residual);
            state = std::move(s.state);
        }
        const double elapsed = seconds() - t0;
        Field exact = wave_state(fe, pn, wave, T, eps);
        Field den(fe.nodes, 1), denex(fe.nodes, 1);
        for (int i = 0; i < fe.nodes; ++i) {
            den(i, 0) = project(state.row(i).transpose())[0];
            denex(i, 0) = project(exact.row(i).transpose())[0];
        }
        auto scalar_error = [&](int norm) { return pn_error(fe, den, denex, norm); };
        std::cout << "convergence," << variant << ',' << (periodic ? "periodic" : "open") << ',' << cells << ',' << steps << ','
                  << std::setprecision(16) << dt << ',' << T << ',' << pn_error(fe,state,exact,1) << ','
                  << pn_error(fe,state,exact,2) << ',' << pn_error(fe,state,exact,0) << ','
                  << scalar_error(1) << ',' << scalar_error(2) << ',' << scalar_error(0) << ',' << active << ','
                  << balance << ',' << elapsed << '\n';
    }
}

void spectrum(const std::string &variant) {
    const Numerics opts = numerics(variant);
    std::cout << "suite,variant,boundary,order,cells,row,column,value\n";
    for (bool periodic : {true, false}) for (int order : {1, 3}) {
        MeshSpec mesh; mesh.dimension = 1; mesh.elements = 4; mesh.periodic = periodic;
        Spatial fe(mesh, 1., {}, opts.mass_mode == "consistent");
        Angular pn(order);
        Transport tr(pn, fe, opts);
        const int n = fe.nodes * pn.q;
        Mat gen(n, n);
        for (int j = 0; j < n; ++j) {
            Field unit = Field::Zero(fe.nodes, pn.q); unit.data()[j] = 1.;
            Field rate = tr.linear_target(unit);
            gen.col(j) = Eigen::Map<const Vec>(rate.data(), n);
        }
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
            std::cout << "spectrum," << variant << ',' << (periodic ? "periodic" : "open") << ',' << order << ",4," << i << ',' << j << ',' << std::setprecision(17) << gen(i,j) << '\n';
    }
}

Config parse(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--suite" && i + 1 < argc) c.suite = argv[++i];
        else if (key == "--variant" && i + 1 < argc) c.variant = argv[++i];
        else if (key == "--output" && i + 1 < argc) c.output = argv[++i];
        else throw std::runtime_error("usage: pn_mc_study [--suite patches|convergence|spectrum] [--variant A|PN|B|C] [--output DIR]");
    }
    return c;
}
} // namespace

int main(int argc, char **argv) {
    try {
        struct Restore { std::streambuf* original = std::cout.rdbuf(); ~Restore(){ std::cout.rdbuf(original); } };
        Config c = parse(argc, argv);
        numerics(c.variant); // fail before emitting a partial CSV for an unknown variant
        std::unique_ptr<std::ofstream> file;
        Restore restore;
        if (!c.output.empty()) {
            std::filesystem::create_directories(c.output);
            const auto path = std::filesystem::path(c.output) / (c.suite + "_" + c.variant + ".csv");
            file = std::make_unique<std::ofstream>(path);
            if (!*file) throw std::runtime_error("cannot open output CSV: " + path.string());
            std::cout.rdbuf(file->rdbuf());
        }
        if (c.suite == "patches") patches(c.variant);
        else if (c.suite == "convergence") convergence(c.variant);
        else if (c.suite == "spectrum") spectrum(c.variant);
        else throw std::runtime_error("suite must be patches, convergence, or spectrum");
    } catch (const std::exception &e) {
        std::cerr << "pn_mc_study failure: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
