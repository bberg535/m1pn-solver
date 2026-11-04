#include "time.hpp"
#include "imex.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <complex>
namespace m1pn {
namespace {
double mass_correction_fourier_amplification(const Angular &pn,int cells,double dt,int rk,double scale) {
    Eigen::SelfAdjointEigenSolver<Mat> eig(pn.normal(V3::UnitX()));
    const double h=1./cells; double maximum=1.;
    for(int k=0;k<cells;++k) {
        const double theta=2*pi*k/cells, c=std::cos(theta), s=std::sin(theta);
        const double real=-scale*pn.speed*(1-c)*(1-c)/(3*h);
        const double adv=s*(1.+(1.-c)/3.)/h;
        for(double lambda:eig.eigenvalues()) {
            const std::complex<double> z(dt*real,-dt*lambda*adv);
            std::complex<double> r=1.+z+z*z/2.;
            if(rk==3)r+=z*z*z/6.;
            maximum=std::max(maximum,std::abs(r));
        }
    }
    return maximum;
}
}
double fourier_amplification(const Angular &pn, int cells, int dim, double gamma, double dt,
                             int rk, bool gradient_defect, int correction_degree) {
    Eigen::SelfAdjointEigenSolver<Mat> eig(pn.normal(V3::UnitZ()));
    const double h = 1. / cells, beta = gamma * std::pow(std::sqrt(double(dim)) * h, 3) * pn.speed;
    double maximum = 1;
    int count = 1;
    for (int d = 0; d < dim; ++d)
        count *= cells;
    for (int flat = 0; flat < count; ++flat) {
        int index_ = flat;
        double mu = 1, k2 = 0;
        double fourth = 0;
        double md[3], kd[3];
        for (int d = 0; d < dim; ++d) {
            double theta = 2 * pi * (index_ % cells) / cells;
            index_ /= cells;
            md[d] = h / 3 * (2 + std::cos(theta));
            kd[d] = 2 * (1 - std::cos(theta)) / h;
            fourth += std::pow(std::sin(theta / 2), 4);
            mu *= md[d];
            k2 += std::pow(std::sin(theta) / md[d], 2);
        }
        double stiffness = 0;
        for (int d = 0; d < dim; ++d)
            stiffness += kd[d] * mu / md[d];
        // R2 (4.21), with star length = element diameter sqrt(dim)*h.
        const double nu = gamma * pn.speed * std::sqrt(double(dim)) * h;
        const double damp = gradient_defect
            ? 4 * nu * std::pow(h, dim - 2) * fourth / mu
            : beta * stiffness * stiffness / (std::pow(h, dim) * mu);
        double mass_symbol = 1.;
        if (correction_degree >= 0) {
            const double ratio = mu / std::pow(h, dim);
            mass_symbol = 1. - std::pow(1. - ratio, correction_degree + 1);
        }
        for (double lambda : eig.eigenvalues()) {
            std::complex<double> z(-dt * mass_symbol * damp,
                                   -dt * mass_symbol * lambda * std::sqrt(k2));
            auto r = 1. + z + z * z / 2.;
            if (rk == 3)
                r += z * z * z / 6.;
            maximum = std::max(maximum, std::abs(r));
        }
    }
    return maximum;
}
TimeControl::TimeControl(Transport &tr, const MeshSpec &mesh, int rk, int max_dofs) : order(rk) {
    require(rk >= 1 && rk <= 3, "Only IMEX Euler, SSPRK2 and SSPRK3 supported");
    operator_bound = tr.generator_norm_bound;
    if (rk == 1) {
        require(tr.options.r2 && (tr.options.stabilization == "biharmonic-original" ||
                                 tr.options.stabilization == "biharmonic-corrected"),
                "IMEX Euler requires R2 with biharmonic-original or biharmonic-corrected");
        const double explicit_bound = std::max(0., tr.generator_norm_bound -
                                                    tr.stabilization_operator->norm_bound / tr.fe.mass_equivalence);
        // This is a CFL/resolution restriction on the explicit part; it is not a
        // stability theorem for the coupled nonlinear, limited IMEX method.
        method = "imex-euler-m1-cfl-explicit-resolution-policy";
        step_limit = .95 * std::min(tr.fe.cfl_step,
                                    explicit_bound > 0 ? 1. / explicit_bound : tr.fe.cfl_step);
        contraction = 1.;
        require(step_limit > 0 && std::isfinite(step_limit), "No finite positive IMEX explicit step restriction");
        return;
    }
    if(mesh.periodic&&mesh.file.empty()&&!mesh.tetrahedra&&
       tr.options.stabilization=="m1pn_mass_correction"&&mesh.dimension==1) {
        method="periodic-q1-pn-mass-correction-fourier";
        double hi=tr.fe.cfl_step,lo=0.;
        for(int i=0;i<50;++i){const double mid=.5*(lo+hi);
            if(mass_correction_fourier_amplification(tr.pn,mesh.elements,mid,rk,tr.options.pn_viscosity_scale)<=1.+2e-14)lo=mid;
            else hi=mid;
        }
        step_limit=.95*lo;
        contraction=mass_correction_fourier_amplification(tr.pn,mesh.elements,step_limit,rk,tr.options.pn_viscosity_scale);
    } else if (mesh.periodic && mesh.file.empty() && !mesh.tetrahedra &&
        tr.options.stabilization != "m1pn_mass_correction") {
        const int correction_degree = tr.options.mass_mode == "consistent" ? -1 :
                                      tr.options.mass_mode == "lumped" ? 0 : tr.options.mass_corrections;
        const bool kv = tr.options.r2 && tr.options.stabilization == "gradient-defect";
        const double gamma = kv ? tr.options.kv_gamma : tr.options.gamma;
        method = kv ? "periodic-q1-gradient-defect-fourier" : "periodic-q1-fourier";
        require(!(rk == 2 && gamma == 0),
                "SSPRK2 centered periodic PN has no positive energy-stable step (8.17)");
        double hi = tr.fe.cfl_step, lo = 0;
        for (int i = 0; i < 45; ++i) {
            double mid = (lo + hi) / 2;
            if (fourier_amplification(tr.pn, mesh.elements, mesh.dimension, gamma, mid, rk, kv,
                                      correction_degree) <= 1 + 2e-14)
                lo = mid;
            else
                hi = mid;
        }
        step_limit = .95 * lo;
        contraction = fourier_amplification(tr.pn, mesh.elements, mesh.dimension, gamma, step_limit, rk, kv,
                                            correction_degree);
    } else if (rk == 3) {
        // For R2 this is a conservative resolution policy, not a full coupled
        // energy/entropy theorem; each actual Euler and stored state is checked.
        method = tr.options.r2 ? "r2-rk3-operator-norm-resolution-policy"
                               : "dissipative-rk3-operator-norm-bound";
        step_limit = operator_bound > 0 ? .95 / operator_bound : tr.fe.cfl_step;
    } else {
        require(tr.options.mass_mode == "consistent",
                "Open SSPRK2 norm test is unavailable for approximate mass modes; use SSPRK3");
        method = "open-rk2-consistent-mass-singular-value";
        const int n = tr.fe.nodes, q = tr.pn.q, total = n * q;
        require(
            total <= max_dofs,
            "Open SSPRK2 norm test exceeds --pn-norm-max-dofs; use SSPRK3 or a smaller mesh/order");
        tr.options.budget.check(12.L * total * total * sizeof(double),
                                "Open SSPRK2 matrix norm test");
        Mat generator(total, total), metric = Mat::Zero(total, total);
        for (int j = 0; j < total; ++j) {
            Field unit = Field::Zero(n, q);
            unit.data()[j] = 1.;
            Field rate = tr.linear_target(unit);
            generator.col(j) = Eigen::Map<Vec>(rate.data(), total);
        }
        for (int i = 0; i < n; ++i)
            for (Sparse::InnerIterator it(tr.fe.M, i); it; ++it)
                for (int k = 0; k < q; ++k)
                    metric(i * q + k, it.col() * q + k) = it.value();
        Eigen::LLT<Mat> chol(metric);
        require(chol.info() == Eigen::Success, "Full consistent mass factor failed");
        Mat weight = chol.matrixU();
        Mat weighted = weight * generator *
                       weight.triangularView<Eigen::Upper>().solve(Mat::Identity(total, total));
        auto norm = [&](double dt) {
            Mat p =
                Mat::Identity(total, total) + dt * weighted + .5 * dt * dt * (weighted * weighted);
            Eigen::JacobiSVD<Mat> svd(p);
            return svd.singularValues()[0];
        };
        double dt = std::min(tr.fe.cfl_step, 1 / std::max(1., operator_bound));
        bool accepted = false;
        for (int i = 0; i < 30; ++i) {
            contraction = norm(dt);
            if (contraction <= 1 + 2e-13) {
                accepted = true;
                break;
            }
            dt *= .5;
        }
        require(accepted && dt > 1e-10 * tr.fe.cfl_step,
                "No resolved positive open SSPRK2 contraction step; use SSPRK3");
        step_limit = .95 * dt;
    }
    require(step_limit > 0 && std::isfinite(step_limit), "No finite positive PN time restriction");
}
void TimeControl::check(double dt) const {
    require(dt > 0 && std::isfinite(dt) && dt <= step_limit * (1 + 2e-14),
            "Requested step violates independent PN time restriction");
}
Step advance(Transport &tr, const TimeControl &control, double time, double dt,
             const Field &initial) {
    control.check(dt);
    tr.check_state(initial);
    Step out;
    if (control.order == 1) {
        require(dt <= tr.fe.cfl_step * (1 + 2e-14), "IMEX Euler violates open M1 CFL");
        Stage s = tr.evaluate_imex(time, dt, initial);
        Field candidate = initial + dt * s.rate;
        tr.check_state(candidate);
        s.diagnostics.cfl_ratio = dt / tr.fe.cfl_step;
        for (int i = 0; i < tr.fe.nodes; ++i)
            s.diagnostics.minimum_state = std::min(s.diagnostics.minimum_state,
                std::min(margin(project(initial.row(i).transpose())), margin(project(candidate.row(i).transpose()))));
        out.state = std::move(candidate);
        out.integrated_pairs = dt * s.accepted_pairs;
        out.integrated_boundary = dt * s.boundary_gather;
        out.integrated_source = dt * s.source;
        out.rate_residual = maxabs(out.state - initial - dt * s.rate);
        Field transfers = tr.gather_pairs(out.integrated_pairs) - out.integrated_boundary + out.integrated_source;
        tr.runtime.sum(transfers);
        out.ledger_residual = maxabs(tr.fe.mass.asDiagonal() * (out.state - initial) - transfers);
        out.stages.push_back(s.diagnostics);
        out.minimum_margin = std::numeric_limits<double>::infinity();
        for (int i = 0; i < tr.fe.nodes; ++i)
            out.minimum_margin = std::min(out.minimum_margin, margin(project(out.state.row(i).transpose())));
        require(out.ledger_residual < 2e-10 * std::max(1., maxabs(transfers)),
                "Integrated IMEX Q-component ledger failed");
        return out;
    }
    int stage_number = 0;
    auto stage = [&](double t, const Field &u) {
        require(dt <= tr.fe.cfl_step * (1 + 2e-14), "Actual SSP Euler branch violates open M1 CFL");
        try {
            control.check(dt);
            Stage s = tr.evaluate(t, u);
            Field euler = u + dt * s.rate;
            tr.check_state(euler);
            s.diagnostics.cfl_ratio = dt / tr.fe.cfl_step;
            for (int i = 0; i < tr.fe.nodes; ++i)
                s.diagnostics.minimum_state =
                    std::min(s.diagnostics.minimum_state,
                             std::min(margin(project(u.row(i).transpose())),
                                      margin(project(euler.row(i).transpose()))));
            ++stage_number;
            return s;
        } catch (const StepRejected &error) {
            throw StepRejected("SSP stage " + std::to_string(stage_number) + ": " + error.what());
        }
    };
    Stage s0 = stage(time, initial);
    Field u1 = initial + dt * s0.rate;
    Stage s1 = stage(time + dt, u1);
    std::vector<Stage> stages;
    stages.push_back(std::move(s0));
    stages.push_back(std::move(s1));
    std::vector<double> weights;
    if (control.order == 2) {
        out.state = .5 * initial + .5 * (u1 + dt * stages[1].rate);
        weights = {.5, .5};
    } else {
        Field u2 = .75 * initial + .25 * (u1 + dt * stages[1].rate);
        tr.check_state(u2);
        stages.push_back(stage(time + .5 * dt, u2));
        out.state = initial / 3. + (2. / 3) * (u2 + dt * stages[2].rate);
        weights = {1. / 6, 1. / 6, 2. / 3};
    }
    tr.check_state(out.state);
    out.integrated_pairs = Field::Zero(tr.fe.pairs.size(), tr.pn.q);
    out.integrated_boundary = Field::Zero(tr.fe.nodes, tr.pn.q);
    out.integrated_source = out.integrated_boundary;
    Field weighted_rates = out.integrated_boundary;
    for (std::size_t r = 0; r < stages.size(); ++r) {
        double w = dt * weights[r];
        out.integrated_pairs += w * stages[r].accepted_pairs;
        out.integrated_boundary += w * stages[r].boundary_gather;
        out.integrated_source += w * stages[r].source;
        weighted_rates += w * stages[r].rate;
        out.stages.push_back(stages[r].diagnostics);
    }
    Field transfers = tr.gather_pairs(out.integrated_pairs) - out.integrated_boundary;
    tr.runtime.sum(transfers);
    transfers += out.integrated_source;
    out.rate_residual = maxabs(out.state - initial - weighted_rates);
    out.ledger_residual = maxabs(tr.fe.mass.asDiagonal() * (out.state - initial) - transfers);
    out.minimum_margin = std::numeric_limits<double>::infinity();
    for (int i = 0; i < tr.fe.nodes; ++i)
        out.minimum_margin =
            std::min(out.minimum_margin, margin(project(out.state.row(i).transpose())));
    require(out.ledger_residual < 2e-10 * std::max(1., maxabs(transfers)),
            "Integrated RK Q-component ledger failed");
    return out;
}
Field reaction(const Field &u, const Field &source, const Vec &absorption, const Vec &scattering,
               double g, double tau) {
    require(std::abs(g) <= 1 && tau >= 0, "Invalid zonal Henyey-Greenstein reaction parameters");
    require(source.rows() == u.rows() && source.cols() == u.cols() &&
                absorption.size() == u.rows() && scattering.size() == u.rows(),
            "Reaction shape mismatch");
    Field v(u.rows(), u.cols());
    const int n = int(std::sqrt(u.cols())) - 1;
    for (int i = 0; i < u.rows(); ++i) {
        require(absorption[i] >= 0 && scattering[i] >= 0 &&
                    std::isfinite(absorption[i] + scattering[i]),
                "Invalid reaction coefficients");
        require(exact::cone(project(source.row(i).transpose())),
                "Reaction source outside physical cone");
        for (int l = 0; l <= n; ++l) {
            const double rate = absorption[i] + scattering[i] * (1 - std::pow(g, l));
            const double z = -rate * tau, decay = std::exp(z), phi = z == 0 ? 1 : std::expm1(z) / z;
            for (int m = -l; m <= l; ++m) {
                int k = index(l, m);
                v(i, k) = decay * u(i, k) + tau * phi * source(i, k);
            }
        }
        cone_require(exact::cone(project(v.row(i).transpose())),
                     "Stored split reaction violates physical cone");
    }
    return v;
}
} // namespace m1pn
