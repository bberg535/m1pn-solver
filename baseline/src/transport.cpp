#include "transport.hpp"
#include "base_bar.hpp"
#include "lazy_m1.hpp"
#include <mutex>
#include <atomic>
namespace m1pn {
Transport::Transport(const Angular &a, Spatial &f, const Numerics &n, Inflow g, Runtime r)
    : pn(a), fe(f), options(n), runtime(r), canonical(pn), inflow(std::move(g)) {
    require(n.gamma >= 0 && std::isfinite(n.gamma),
            "Stabilization gamma must be finite and nonnegative");
    require(n.mass_mode == "consistent" || n.mass_mode == "lumped" || n.mass_mode == "lumped_afc",
            "Unknown mass mode");
    require(n.mass_corrections >= 1, "Mass corrections must be at least one");
    require(n.bh_pair_mode == "element" || n.bh_pair_mode == "canonical" || n.bh_pair_mode == "factorized",
            "Unknown stabilization pair mode");
    require(n.implicit_solver == "direct" || n.implicit_solver == "defect" || n.implicit_solver == "pcg",
            "Unknown implicit solver mode");
    require(n.pn_boundary_quadrature == "lumped" || n.pn_boundary_quadrature == "consistent",
            "Unknown PN boundary quadrature mode");
    require(n.r2 || n.stabilization != "m1pn_ghost_local_projected_p4",
            "Local projected ghost stabilization requires the R2 transfer");
    if (n.r2) stabilization_operator = std::make_unique<SpatialStabilization>(f,n.stabilization,n.stabilization=="gradient-defect"?n.kv_gamma:n.gamma,pn.speed);
    if (n.geometry_cache) {
        pair_geometry.resize(f.pairs.size());
        for (std::size_t e=0;e<f.pairs.size();++e) for (int d=0;d<3;++d) {
            pair_geometry[e][0][d]=f.C[d].coeff(f.pairs[e].i,f.pairs[e].j);
            pair_geometry[e][1][d]=f.C[d].coeff(f.pairs[e].j,f.pairs[e].i);
        }
    }
    const long double spatial_bytes =
        8192.L * f.nodes + 192.L * std::pow(static_cast<long double>(f.nodes), 4.L / 3);
    const long double state_bytes =
        sizeof(double) * (66.L * f.nodes * pn.q + 8.L * f.pairs.size() * pn.q);
    n.budget.check(spatial_bytes + state_bytes + 32.L * pn.q * pn.q * sizeof(double),
                   "Transport and RK state/transfer buffers");
    beta = Vec::Zero(f.nodes);
    std::vector<bool> streaming(f.nodes, false);
    for (const auto &p : f.pairs)
        if (p.s.norm() > 0)
            streaming[p.i] = streaming[p.j] = true;
    for (int i = 0; i < f.nodes; ++i)
        if (streaming[i] && !(n.masked && f.boundary[i]))
            beta[i] = n.gamma * std::pow(f.diameter[i], 3) * pn.speed;
    for (auto &face : f.faces)
        for (auto &p : face.points) {
            int id = -1;
            for (std::size_t j = 0; j < boundaries.size(); ++j)
                if ((boundaries[j]->n - p.normal).norm() < 2e-13) {
                    id = int(j);
                    break;
                }
            if (id < 0) {
                n.budget.check(spatial_bytes + state_bytes +
                                   (32.L + 10.L * (boundaries.size() + 1)) * pn.q * pn.q *
                                       sizeof(double),
                               "Normal cache");
                id = int(boundaries.size());
                boundaries.push_back(std::make_unique<Boundary>(pn, canonical, p.normal, n.c0));
            }
            p.normal_id = id;
        }
    if (n.r2) {
        Vec rates=Vec::Zero(f.nodes);
        for (auto &p : f.pairs) {
            V3 cij=V3::Zero(), cji=V3::Zero();
            for(int k=0;k<3;++k){cij[k]=f.C[k].coeff(p.i,p.j); cji[k]=f.C[k].coeff(p.j,p.i);}
            const double edge=std::max(cij.norm(),cji.norm());
            // Explicit positive graph extension on every FE pair, including zero C edges.
            const double graph_extension=p.mass/std::max(f.diameter[p.i],f.diameter[p.j]);
            p.viscosity=std::max(n.anchor*std::max(edge,graph_extension),edge);
            while(!exact::cone(Four(p.viscosity,cij[0],cij[1],cij[2])) ||
                  !exact::cone(Four(p.viscosity,cji[0],cji[1],cji[2]))) {
                p.viscosity=std::nextafter(p.viscosity,std::numeric_limits<double>::infinity());
                require(std::isfinite(p.viscosity),"Unrepresentable directed graph speed");
            }
            rates[p.i]+=2*p.viscosity; rates[p.j]+=2*p.viscosity;
        }
        for(const auto &face:f.faces) for(size_t j=0;j<face.nodes.size();++j) rates[face.nodes[j]]+=face.mass[j];
        f.cfl_step=std::numeric_limits<double>::infinity();
        for(int i=0;i<f.nodes;++i) if(rates[i]>0) f.cfl_step=std::min(f.cfl_step,f.mass[i]/rates[i]);
    }
    // M >= c ML; tensor/submultiplicative bound in the consistent M norm.
    for (int d = 0; d < 3; ++d)
        generator_norm_bound += pn.speed * f.norm_bound(f.C[d]);
    generator_norm_bound += n.r2 ? stabilization_operator->norm_bound :
                            beta.maxCoeff() * std::pow(f.norm_bound(f.K), 2);
    Vec bound = Vec::Zero(f.nodes);
    for (const auto &face : f.faces)
        for (const auto &p : face.points) {
            double sum = 0;
            for (std::size_t j = 0; j < face.nodes.size(); ++j)
                sum += p.shape[j] / std::sqrt(f.mass[face.nodes[j]]);
            for (std::size_t i = 0; i < face.nodes.size(); ++i)
                bound[face.nodes[i]] += p.weight * p.shape[i] * sum *
                                        (n.r2 ? boundaries[p.normal_id]->llf_norm : boundaries[p.normal_id]->norm) /
                                        std::sqrt(f.mass[face.nodes[i]]);
        }
    generator_norm_bound =
        (generator_norm_bound + bound.maxCoeff()) / f.mass_equivalence * (1 + 1e-11);
}
Field Transport::biharmonic(const Field &u, Field *output) const {
    Field w = fe.K * u;
    for (int i = 0; i < fe.nodes; ++i)
        w.row(i) *= -beta[i] / fe.mass[i];
    Field z = fe.K * w;
    if (output)
        *output = std::move(w);
    return z;
}
Field Transport::gather_pairs(const Field &a) const {
    Field g = Field::Zero(fe.nodes, pn.q);
    for (std::size_t e = 0; e < fe.pairs.size(); ++e)
        if (runtime.owns(e)) {
            const auto &p = fe.pairs[e];
            g.row(p.i) += a.row(e);
            g.row(p.j) -= a.row(e);
        }
    return g;
}
Field Transport::linear_target(const Field &u) const {
    if(options.r2){
        Field rhs=stabilization_operator->apply(u);
        for(int k=0;k<3;++k) rhs.noalias()+=fe.C[k].transpose()*(u*pn.A[k].transpose());
        for(const auto &face:fe.faces) for(const auto &p:face.points) {
            if(options.pn_boundary_quadrature=="consistent") {
                Vec trace=Vec::Zero(pn.q);
                for(size_t j=0;j<face.nodes.size();++j)
                    trace.noalias()+=p.shape[j]*u.row(face.nodes[j]).transpose();
                const Vec flux=boundaries[p.normal_id]->llf_homogeneous*trace;
                for(size_t j=0;j<face.nodes.size();++j)
                    rhs.row(face.nodes[j])-=p.weight*p.shape[j]*flux.transpose();
            } else for(size_t j=0;j<face.nodes.size();++j) {
                Vec flux=boundaries[p.normal_id]->llf_homogeneous*u.row(face.nodes[j]).transpose();
                rhs.row(face.nodes[j])-=p.weight*p.shape[j]*flux.transpose();
            }
        }
        double residual=0; return mass_target(rhs,residual);
    }
    Field rhs = biharmonic(u);
    for (int d = 0; d < 3; ++d)
        rhs.noalias() += fe.C[d].transpose() * (u * pn.A[d].transpose());
    for (const auto &f : fe.faces)
        for (const auto &p : f.points) {
            Vec trace = Vec::Zero(pn.q);
            for (std::size_t j = 0; j < f.nodes.size(); ++j)
                trace.noalias() += p.shape[j] * u.row(f.nodes[j]).transpose();
            Vec flux = boundaries[p.normal_id]->homogeneous * trace;
            for (std::size_t j = 0; j < f.nodes.size(); ++j)
                rhs.row(f.nodes[j]) -= p.weight * p.shape[j] * flux.transpose();
        }
    double residual = 0;
    return mass_target(rhs, residual);
}
Field Transport::mass_target(const Field &rhs, double &residual, Field *predictor) const {
    if (options.mass_mode == "consistent") {
        Field target = fe.solve_mass(rhs, options.mass_tolerance, residual);
        if (predictor) *predictor = target;
        return target;
    }
    const int degree = options.mass_mode == "lumped" ? 0 : options.mass_corrections;
    auto inverse_lumped = [&](const Field &v) {
        Field out = v;
        for (int i = 0; i < fe.nodes; ++i) out.row(i) /= fe.mass[i];
        return out;
    };
    Field p = degree == 0 ? Field::Zero(rhs.rows(), rhs.cols()) : inverse_lumped(rhs);
    for (int k = 1; k < degree; ++k)
        p = inverse_lumped(rhs + fe.mass.asDiagonal() * p - fe.M * p);
    if (predictor) *predictor = p;
    Field target = degree == 0 ? inverse_lumped(rhs)
                               : inverse_lumped(rhs + fe.mass.asDiagonal() * p - fe.M * p);
    residual = maxabs(fe.M * target - rhs);
    return target;
}
void Transport::check_state(const Field &u) const {
    require(u.rows() == fe.nodes && u.cols() == pn.q && u.allFinite(),
            "Invalid/nonfinite PN field");
    for (int i = 0; i < fe.nodes; ++i)
        cone_require(exact::cone(project(u.row(i).transpose())),
                     "Stored PN projection outside M1 cone at node " + std::to_string(i));
}
Stage Transport::evaluate(double time, const Field &u, bool unlimited, const Field &source) {
    if (options.r2) return evaluate_r2(time,u,unlimited,source);
    check_state(u);
    Stage out;
    auto &diag = out.diagnostics;
    double start = seconds();
    Field w;
    Field z = biharmonic(u, &w);
    timings.biharmonic += seconds() - start;
    Field rhs = z;
    for (int d = 0; d < 3; ++d)
        rhs.noalias() += fe.C[d].transpose() * (u * pn.A[d].transpose());
    out.source = source.size() ? source : Field::Zero(fe.nodes, pn.q);
    require(out.source.rows() == fe.nodes && out.source.cols() == pn.q && out.source.allFinite(),
            "Invalid integrated source");
    if (!unlimited)
        for (int i = 0; i < fe.nodes; ++i)
            require(exact::cone(project(out.source.row(i).transpose())),
                    "Transport source must have nonnegative physical moments");
    rhs += out.source;
    start = seconds();
    std::vector<Four> low(fe.nodes);
    std::vector<Levermore> closure;
    closure.reserve(fe.nodes);
    for (int i = 0; i < fe.nodes; ++i)
        low[i] = project(u.row(i).transpose());
    for (int i = 0; i < fe.nodes; ++i)
        closure.emplace_back(low[i]);
    timings.mcl += seconds() - start;
    std::vector<Field> face_high, face_low, face_normal;
    start = seconds();
    for (const auto &f : fe.faces) {
        const int nd = int(f.nodes.size());
        Field h = Field::Zero(nd, pn.q), q0 = Field::Zero(nd, 4), normal = q0;
        // Only reuse identical normals, not the tolerance-merged C1 normal IDs.
        // Thus warped faces retain their original pointwise kinetic fluxes.
        V3 previous_normal = V3::Zero();
        std::vector<Four> outgoing(nd), full_flux(nd);
        for (const auto &p : f.points) {
            if ((p.normal.array() != previous_normal.array()).any()) {
                for (int j = 0; j < nd; ++j) {
                    outgoing[j] = closure[f.nodes[j]].outgoing(p.normal);
                    full_flux[j] = closure[f.nodes[j]].flux(p.normal);
                }
                previous_normal = p.normal;
            }
            const auto &b = *boundaries[p.normal_id];
            const auto data = b.data(inflow, p.x, time);
            Vec trace = Vec::Zero(pn.q);
            for (int j = 0; j < nd; ++j)
                trace.noalias() += p.shape[j] * u.row(f.nodes[j]).transpose();
            Vec gh = b.flux(trace, data.d);
            for (int j = 0; j < nd; ++j) {
                if (p.shape[j] == 0)
                    continue;
                const int i = f.nodes[j];
                const double weight = p.weight * p.shape[j];
                // Eq. (4.9), retaining the consistent surface mass exactly once.
                h.row(j) +=
                    weight *
                    (gh - .5 * pn.apply(p.normal, trace - u.row(i).transpose())).transpose();
                rhs.row(i) -= weight * gh.transpose();
                q0.row(j) += weight * (outgoing[j] + data.kinetic).transpose();
                normal.row(j) += weight * full_flux[j].transpose();
            }
        }
        face_high.push_back(std::move(h));
        face_low.push_back(std::move(q0));
        face_normal.push_back(std::move(normal));
    }
    timings.boundary += seconds() - start;
    start = seconds();
    Field mass_predictor;
    out.target = mass_target(rhs, diag.mass_residual, &mass_predictor);
    timings.mass += seconds() - start;
    if (options.mass_mode == "consistent") out.mass_error = fe.M * out.target - rhs;
    else out.mass_error = Field::Zero(fe.nodes, pn.q);
    Field high_pairs = Field::Zero(fe.pairs.size(), pn.q);
    out.accepted_pairs = high_pairs;
    Vec alphas = Vec::Ones(fe.pairs.size()),
        bar_margin = Vec::Constant(fe.pairs.size(), std::numeric_limits<double>::infinity());
    Vec density_clip = Vec::Zero(fe.pairs.size()), redirection = Vec::Zero(fe.pairs.size()),
        retention = Vec::Zero(fe.pairs.size());
    Eigen::VectorXi pair_pass = Eigen::VectorXi::Zero(fe.pairs.size()),
        pair_round = Eigen::VectorXi::Zero(fe.pairs.size()),
        pair_scalar = Eigen::VectorXi::Zero(fe.pairs.size()),
        base_bar_recomputed = Eigen::VectorXi::Zero(fe.pairs.size());
    start = seconds();
    out.boundary_gather = Field::Zero(fe.nodes, pn.q);
    Field target_boundary = out.boundary_gather;
    int local_failure = 0;
    std::string failure_message;
    try {
        parallel_for(int(fe.pairs.size()), [&](int e) {
            if (!runtime.owns(e))
                return;
            const auto &p = fe.pairs[e];
            const Vec ui = u.row(p.i).transpose(), uj = u.row(p.j).transpose();
            Vec h = -pn.apply(p.s, ui + uj) + p.stiffness * (w.row(p.j) - w.row(p.i)).transpose() +
                    p.mass * (mass_predictor.row(p.i) - mass_predictor.row(p.j)).transpose();
            const Four fi = closure[p.i].flux(p.s), fj = closure[p.j].flux(p.s);
            Four b = p.viscosity * (low[p.j] - low[p.i]) - fi - fj;
            Four bar = .5 * (low[p.i] + low[p.j]) - (fj - fi) / (2 * p.viscosity);
            if (!exact::cone(bar)) {
                bar = stable_base_bar(low[p.i], low[p.j], p.s, p.viscosity);
                base_bar_recomputed[e] = 1;
                cone_require(exact::cone(bar), "Stable low-order pair base outside stored cone");
            }
            Four correction = project(h) - b, delta = correction / (2 * p.viscosity);
            double alpha = 1;
            Four accepted_correction;
            Four left, right;
            try {
                if (!unlimited && options.density_vector) {
                    const auto limited = density_pair_limit(bar, bar, delta);
                    accepted_correction = 2 * p.viscosity * limited.correction;
                    const Four actual_delta = accepted_correction / (2 * p.viscosity);
                    left = bar + actual_delta;
                    right = bar - actual_delta;
                    density_clip[e] = std::abs(delta[0] - actual_delta[0]);
                    redirection[e] = (delta.tail<3>() - actual_delta.tail<3>()).norm();
                    retention[e] = delta[0] == 0 ? 1 : std::abs(actual_delta[0] / delta[0]);
                    pair_pass[e] = limited.target_pass;
                    pair_round[e] = limited.rounding_fallback;
                    pair_scalar[e] = limited.scalar_fallback;
                    alphas[e] = limited.target_pass ? 1 :
                                std::min(retention[e], limited.momentum_theta);
                } else {
                    if (!unlimited)
                        alpha = segment(bar, delta, true);
                    accepted_correction = alpha * correction;
                    left = bar + alpha * delta;
                    right = bar - alpha * delta;
                }
            } catch (const StepRejected &error) {
                throw StepRejected("pair " + std::to_string(e) + " nodes " + std::to_string(p.i) +
                                   "," + std::to_string(p.j) + ": " + error.what());
            }
            if (!unlimited && options.density_vector &&
                !(exact::cone(left) && exact::cone(right))) {
                // Multiplying the selected p by 2d and recovering its actual
                // stored endpoint increment can move a boundary value outward.
                // Retry with the established common scalar limiter.
                alpha = segment(bar, delta, true);
                accepted_correction = alpha * correction;
                Four actual_delta = accepted_correction / (2 * p.viscosity);
                left = bar + actual_delta;
                right = bar - actual_delta;
                if (!(exact::cone(left) && exact::cone(right))) {
                    double lo = 0, hi = alpha;
                    for (int k = 0; k < 60; ++k) {
                        const double mid = (lo + hi) / 2;
                        const Four trial = (mid * correction) / (2 * p.viscosity);
                        if (exact::cone(bar + trial) && exact::cone(bar - trial)) lo = mid;
                        else hi = mid;
                    }
                    alpha = lo;
                    accepted_correction = alpha * correction;
                    actual_delta = accepted_correction / (2 * p.viscosity);
                    left = bar + actual_delta;
                    right = bar - actual_delta;
                }
                pair_scalar[e] = 1;
                pair_round[e] = 1;
                pair_pass[e] = 0;
                density_clip[e] = std::abs(delta[0] - actual_delta[0]);
                redirection[e] = (delta.tail<3>() - actual_delta.tail<3>()).norm();
                retention[e] = delta[0] == 0 ? 1 : std::abs(actual_delta[0] / delta[0]);
                alphas[e] = alpha;
            }
            if (!options.density_vector || unlimited)
                alphas[e] = alpha;
            bar_margin[e] = std::min(margin(left), margin(right));
            if (!unlimited)
                cone_require(exact::cone(left) && exact::cone(right),
                             "Two-sided corrected bar failed");
            high_pairs.row(e) = h.transpose();
            set_low(h, b + accepted_correction);
            out.accepted_pairs.row(e) = h.transpose();
        });
        for (std::size_t fi = 0; fi < fe.faces.size(); ++fi)
            if (runtime.owns(fi)) {
                const auto &f = fe.faces[fi];
                for (std::size_t j = 0; j < f.nodes.size(); ++j) {
                    if (f.mass[j] <= 0)
                        continue;
                    const int i = f.nodes[j];
                    Vec h = face_high[fi].row(j).transpose();
                    const Four q0 = face_low[fi].row(j).transpose();
                    const double mass = f.mass[j] * options.anchor;
                    const Four bar = low[i] + (face_normal[fi].row(j).transpose() - q0) / mass;
                    const Four delta = project(h) - q0;
                    const Four change = -delta / mass;
                    double eta = 1;
                    try {
                        if (!unlimited)
                            eta = segment(bar, change);
                    } catch (const StepRejected &error) {
                        throw StepRejected("face " + std::to_string(fi) + " node " +
                                           std::to_string(i) + ": " + error.what());
                    }
                    Four accepted_bar = bar + eta * change;
                    if (!unlimited)
                        cone_require(exact::cone(accepted_bar), "One-sided corrected bar failed");
                    diag.minimum_bar = std::min(diag.minimum_bar, margin(accepted_bar));
                    diag.minimum_eta = std::min(diag.minimum_eta, eta);
                    if (eta < 1)
                        ++diag.active_faces;
                    target_boundary.row(i) += h.transpose();
                    set_low(h, q0 + eta * delta);
                    out.boundary_gather.row(i) += h.transpose();
                }
            }
    } catch (const StepRejected &e) {
        local_failure = 1;
        failure_message = e.what();
    } catch (const std::exception &e) {
        local_failure = 2;
        failure_message = e.what();
    }
    runtime.finish_local_phase(local_failure, failure_message);
    timings.mcl += seconds() - start;
    Field target_gather = gather_pairs(high_pairs) - target_boundary;
    out.gather = gather_pairs(out.accepted_pairs) - out.boundary_gather;
    start = seconds();
    runtime.sum(target_gather);
    runtime.sum(out.gather);
    // Keep local boundary/pair ledgers; gather is globally reduced, source replicated.
    timings.communication += seconds() - start;
    target_gather += out.source;
    out.gather += out.source;
    diag.target_recovery =
        maxabs(target_gather - fe.mass.asDiagonal() * out.target + out.mass_error);
    out.rate = out.target;
    for (int i = 0; i < fe.nodes; ++i) {
        Vec k = out.rate.row(i).transpose();
        set_low(k, project(out.gather.row(i).transpose()) / fe.mass[i]);
        out.rate.row(i) = k.transpose();
        diag.projection = std::max(
            diag.projection, (project(k) - project(out.gather.row(i).transpose()) / fe.mass[i])
                                 .lpNorm<Eigen::Infinity>());
    }
    Field defect = fe.mass.asDiagonal() * out.rate - out.gather;
    diag.local_balance = maxabs(defect);
    diag.mass_defect = maxabs(out.mass_error);
    for (int i = 0; i < fe.nodes; ++i) {
        Vec eps = out.mass_error.row(i).transpose();
        set_low(eps, Four::Zero());
        defect.row(i) -= eps.transpose();
    }
    require(maxabs(defect) < 5e-11 * std::max(1., maxabs(out.gather)),
            "Accepted Q-component local balance identity failed");
    if (pn.q > 4)
        diag.high_rate_error =
            maxabs(out.rate.rightCols(pn.q - 4) - out.target.rightCols(pn.q - 4));
    Field mu = fe.M * u;
    diag.target_energy = (mu.array() * out.target.array()).sum();
    diag.sync_work = (mu.array() * (out.rate - out.target).array()).sum();
    diag.minimum_alpha = runtime.reduce(alphas.minCoeff(), -1);
    diag.minimum_eta = runtime.reduce(diag.minimum_eta, -1);
    diag.minimum_bar = runtime.reduce(std::min(diag.minimum_bar, bar_margin.minCoeff()), -1);
    diag.active_pairs = long(runtime.reduce((alphas.array() < 1.).count(), 0));
    diag.active_faces = long(runtime.reduce(diag.active_faces, 0));
    diag.density_clip_sum = runtime.reduce(density_clip.sum(), 0);
    diag.momentum_redirection_sum = runtime.reduce(redirection.sum(), 0);
    diag.density_retention_sum = runtime.reduce(retention.sum(), 0);
    diag.pair_target_passes = long(runtime.reduce(pair_pass.cast<double>().sum(), 0));
    diag.pair_rounding_fallbacks = long(runtime.reduce(pair_round.cast<double>().sum(), 0));
    diag.pair_scalar_fallbacks = long(runtime.reduce(pair_scalar.cast<double>().sum(), 0));
    diag.base_bar_reevaluations = long(runtime.reduce(base_bar_recomputed.cast<double>().sum(), 0));
    return out;
}
} // namespace m1pn

namespace m1pn {
Stage Transport::evaluate_r2(double time, const Field &u, bool unlimited, const Field &source,
                             double implicit_dt) {
    check_state(u);
    Stage out;
    auto &diag=out.diagnostics;
    std::array<Field,3> cached_pn_flux;
    if(options.geometry_cache)
        for(int d=0;d<3;++d) cached_pn_flux[d]=u*pn.A[d].transpose();
    auto pair_streaming = [&](int i,int j,const V3 &cij,const V3 &cji) -> Vec {
        if(!options.geometry_cache) return Vec(pn.apply(cji,u.row(j).transpose())-
                                                pn.apply(cij,u.row(i).transpose()));
        Vec flux=Vec::Zero(pn.q);
        for(int d=0;d<3;++d)
            flux+=cji[d]*cached_pn_flux[d].row(j).transpose()-
                  cij[d]*cached_pn_flux[d].row(i).transpose();
        return flux;
    };
    double start=seconds();
    Field stabilization_pairs;
    Field rhs=stabilization_operator->apply(u,&stabilization_pairs,options.bh_pair_mode);
    timings.biharmonic+=seconds()-start;
    for(int d=0;d<3;++d) rhs.noalias()+=fe.C[d].transpose()*(u*pn.A[d].transpose());
    out.source=source.size()?source:Field::Zero(fe.nodes,pn.q);
    require(out.source.rows()==fe.nodes && out.source.cols()==pn.q && out.source.allFinite(),
            "Invalid integrated source");
    if(!unlimited) for(int i=0;i<fe.nodes;++i)
        require(exact::cone(project(out.source.row(i).transpose())),"Nonrealizable source load");
    rhs+=out.source;
    ImplicitResult implicit_result;
    std::vector<Four> low;
    std::vector<std::unique_ptr<Levermore>> closure;
    for(int i=0;i<fe.nodes;++i) low.push_back(project(u.row(i).transpose()));
    closure.resize(fe.nodes);
    auto get_closure = [&](int i) -> const Levermore & {
        if (!closure[i]) closure[i] = std::make_unique<Levermore>(low[i]);
        return *closure[i];
    };
    if (unlimited || !options.lazy)
        for(int i=0;i<fe.nodes;++i) get_closure(i);
    std::vector<Field> face_high,face_low,face_bar;
    std::vector<unsigned char> face_lazy_certified;
    start=seconds();
    for(const auto &face:fe.faces) {
        const int nd=face.nodes.size();
        Field hn=Field::Zero(nd,pn.q),hl=Field::Zero(nd,4),bar=hl;
        if (!unlimited && options.lazy) {
            for(const auto &p:face.points) {
                const auto &b=*boundaries[p.normal_id];
                const auto data=b.data(inflow,p.x,time);
                Vec common_flux;
                if(options.pn_boundary_quadrature=="consistent") {
                    Vec trace=Vec::Zero(pn.q);
                    for(int j=0;j<nd;++j) trace.noalias()+=p.shape[j]*u.row(face.nodes[j]).transpose();
                    const Vec ue=b.exterior(trace,data.d);
                    common_flux=.5*(pn.apply(p.normal,trace)+pn.apply(p.normal,ue))-.5*(ue-trace);
                }
                for(int j=0;j<nd;++j) {
                    const double weight=p.weight*p.shape[j]; if(weight==0) continue;
                    const int i=face.nodes[j];
                    const Four exterior=inflow.exterior_state ?
                        inflow.exterior_state(p.x,time,p.normal,low[i]) :
                        (inflow.exterior_m1?inflow.exterior_m1(p.x,time):data.moments);
                    require(exact::cone(exterior),"M1 exterior state outside cone");
                    Vec flux;
                    if(options.pn_boundary_quadrature=="consistent") flux=common_flux;
                    else {
                        const Vec ui=u.row(i).transpose(),ue=b.exterior(ui,data.d);
                        flux=.5*(pn.apply(p.normal,ui)+pn.apply(p.normal,ue))-.5*(ue-ui);
                    }
                    hn.row(j)+=weight*flux.transpose();
                }
            }
            bool certified=true;
            for(int j=0;j<nd && certified;++j) {
                const double a=face.mass[j]; if(a<=0) continue;
                const int i=face.nodes[j]; const Four ui=low[i];
                const Four base=ui-project(hn.row(j).transpose())/a;
                const auto chi=lazy_m1::chi_interval(ui);
                for(double c:{chi.lo,chi.hi}) {
                    Four f=Four::Zero();
                    for(const auto &p:face.points) {
                        const double weight=p.weight*p.shape[j]; if(weight!=0)
                            f+=weight*lazy_m1::pressure_times(ui,c,p.normal);
                    }
                    if(!lazy_m1::strictly_interior(base+f/a)) {certified=false;break;}
                }
            }
            if(certified) {
                face_lazy_certified.push_back(1);
                face_high.push_back(std::move(hn)); face_low.push_back(std::move(hl));
                face_bar.push_back(std::move(bar));
                for(int j=0;j<nd;++j) rhs.row(face.nodes[j])-=face_high.back().row(j);
                continue;
            }
            hn.setZero();
        }
        for(const auto &p:face.points) {
            const auto &b=*boundaries[p.normal_id];
            const auto data=b.data(inflow,p.x,time);
            Four shared_exterior=Four::Zero();
            std::optional<Levermore> shared_ext_closure;
            if(!inflow.exterior_state) {
                shared_exterior=inflow.exterior_m1?inflow.exterior_m1(p.x,time):data.moments;
                require(exact::cone(shared_exterior),"M1 exterior state outside cone");
                shared_ext_closure.emplace(shared_exterior);
            }
            Vec consistent_flux;
            if (options.pn_boundary_quadrature=="consistent") {
                Vec trace=Vec::Zero(pn.q);
                for(int j=0;j<nd;++j) trace.noalias()+=p.shape[j]*u.row(face.nodes[j]).transpose();
                const Vec exterior_trace=b.exterior(trace,data.d);
                consistent_flux=.5*(pn.apply(p.normal,trace)+pn.apply(p.normal,exterior_trace))-
                                 .5*(exterior_trace-trace);
            }
            for(int j=0;j<nd;++j) {
                const double weight=p.weight*p.shape[j];
                if(weight==0) continue;
                const int i=face.nodes[j];
                const Four exterior=inflow.exterior_state ? inflow.exterior_state(p.x,time,p.normal,low[i]) : shared_exterior;
                require(exact::cone(exterior),"M1 exterior state outside cone");
                std::optional<Levermore> node_ext_closure;
                const Levermore *ext_closure=nullptr;
                if(inflow.exterior_state) {
                    node_ext_closure.emplace(exterior);
                    ext_closure=&*node_ext_closure;
                } else ext_closure=&*shared_ext_closure;
                const Vec ui=u.row(i).transpose();
                const Vec ue=b.exterior(ui,data.d);
                const Vec flux=options.pn_boundary_quadrature=="consistent" ?
                               consistent_flux :
                               .5*(pn.apply(p.normal,ui)+pn.apply(p.normal,ue))-.5*(ue-ui);
                const Four fi=get_closure(i).flux(p.normal),fx=ext_closure->flux(p.normal);
                const Four lo_flux=.5*(fi+fx)-.5*(exterior-low[i]);
                Four ghost=.5*(low[i]+exterior)-.5*(fx-fi);
                if(!exact::cone(ghost)) ghost=stable_base_bar(low[i],exterior,.5*p.normal,.5);
                cone_require(exact::cone(ghost),"R2 LLF ghost base outside cone");
                hn.row(j)+=weight*flux.transpose();
                hl.row(j)+=weight*lo_flux.transpose();
                bar.row(j)+=weight*ghost.transpose();
            }
        }
        for(int j=0;j<nd;++j) {
            rhs.row(face.nodes[j])-=hn.row(j);
            if(face.mass[j]>0) bar.row(j)/=face.mass[j];
        }
        face_lazy_certified.push_back(0);
        face_high.push_back(std::move(hn)); face_low.push_back(std::move(hl));
        face_bar.push_back(std::move(bar));
    }
    timings.boundary+=seconds()-start;
    start=seconds();
    Field mass_predictor;
    if (implicit_dt > 0) {
        if (!implicit_solver) {
            implicit_positive=stabilization_operator->positive_matrix();
            if (implicit_positive.rows()!=fe.nodes) {
                std::vector<Triplet> entries;
                Field basis=Field::Zero(fe.nodes,1);
                for (int j=0;j<fe.nodes;++j) {
                    basis.setZero(); basis(j,0)=1.;
                    Field column=stabilization_operator->apply(basis);
                    for (int i=0;i<fe.nodes;++i)
                        if (column(i,0)!=0.) entries.emplace_back(i,j,-column(i,0));
                }
                implicit_positive.resize(fe.nodes,fe.nodes);
                implicit_positive.setFromTriplets(entries.begin(),entries.end());
            }
            implicit_solver=std::make_unique<ImplicitMassSolver>(fe,implicit_positive);
        }
        implicit_result=implicit_solver->solve(rhs,implicit_dt,options.implicit_solver,
                                                options.implicit_rtol,options.implicit_atol,
                                                options.implicit_max_iterations);
        Field new_state=u+implicit_dt*implicit_result.rate;
        stabilization_pairs=Field();
        stabilization_operator->apply(new_state,&stabilization_pairs,options.bh_pair_mode);
        diag.implicit_residual=implicit_result.residual;
        diag.implicit_relative_residual=implicit_result.relative_residual;
        diag.implicit_solves=implicit_result.solves;
        mass_predictor=implicit_result.predictor;
        Field A_rate=implicit_positive*implicit_result.rate;
        Field rhs_full=rhs-implicit_dt*A_rate;
        if (options.implicit_solver=="defect") {
            Field defect=fe.mass.asDiagonal()*mass_predictor-fe.M*mass_predictor;
            out.target=Field::Zero(fe.nodes,pn.q);
            for(int i=0;i<fe.nodes;++i)
                out.target.row(i)=(rhs_full.row(i)+defect.row(i))/fe.mass[i];
            out.mass_error=Field::Zero(fe.nodes,pn.q);
        } else {
            out.target=implicit_result.rate;
            out.mass_error=fe.M*implicit_result.rate-rhs_full;
        }
        diag.mass_residual=maxabs(out.mass_error);
    } else {
        out.target=mass_target(rhs,diag.mass_residual,&mass_predictor);
        if(options.mass_mode=="consistent") out.mass_error=fe.M*out.target-rhs;
        else out.mass_error=Field::Zero(fe.nodes,pn.q);
    }
    timings.mass+=seconds()-start;
    Field high_pairs=Field::Zero(fe.pairs.size(),pn.q);
    out.accepted_pairs=high_pairs;
    out.boundary_gather=Field::Zero(fe.nodes,pn.q);
    Field target_boundary=out.boundary_gather;
    Vec alphas=Vec::Ones(fe.pairs.size());
    Vec pair_margins=Vec::Constant(fe.pairs.size(),std::numeric_limits<double>::infinity());
    int local_failure=0; std::string failure;
    std::atomic<long> lazy_certified{0}, lazy_fallback{0};
    std::vector<unsigned char> certified_pair(fe.pairs.size(),0);
    start=seconds();
    if (!unlimited && options.lazy) {
        // Certify serially and instantiate each closure once only when a face
        // or an uncertified pair will use it. The hot pair loop is lock-free.
        for(std::size_t e=0;e<fe.pairs.size();++e) {
            if(!runtime.owns(e)) continue;
            const auto &p=fe.pairs[e]; V3 cij,cji;
            if(options.geometry_cache) { cij=pair_geometry[e][0]; cji=pair_geometry[e][1]; }
            else for(int d=0;d<3;++d) {cij[d]=fe.C[d].coeff(p.i,p.j);cji[d]=fe.C[d].coeff(p.j,p.i);}
            const Vec ui=u.row(p.i).transpose(),uj=u.row(p.j).transpose();
            const Vec h=pair_streaming(p.i,p.j,cij,cji)+stabilization_pairs.row(e).transpose()+
                        p.mass*(mass_predictor.row(p.i)-mass_predictor.row(p.j)).transpose();
            high_pairs.row(e)=h.transpose();
            if(lazy_m1::certify_full_pair(low[p.i],low[p.j],cij,cji,p.viscosity,project(h)))
                certified_pair[e]=1;
            else { (void)get_closure(p.i); (void)get_closure(p.j); }
        }
    }
    try {
        parallel_for(int(fe.pairs.size()),[&](int e) {
            if(!runtime.owns(e)) return;
            const auto &p=fe.pairs[e];
            V3 cij,cji;
            if (options.geometry_cache) {
                cij=pair_geometry[e][0]; cji=pair_geometry[e][1];
            } else for(int d=0;d<3;++d) {cij[d]=fe.C[d].coeff(p.i,p.j);cji[d]=fe.C[d].coeff(p.j,p.i);}
            const Vec ui=u.row(p.i).transpose(),uj=u.row(p.j).transpose();
            Vec h;
            if(!unlimited && options.lazy) h=high_pairs.row(e).transpose();
            else h=pair_streaming(p.i,p.j,cij,cji)+stabilization_pairs.row(e).transpose()+
                    p.mass*(mass_predictor.row(p.i)-mass_predictor.row(p.j)).transpose();
            if (certified_pair[e]) {
                high_pairs.row(e)=h.transpose();
                out.accepted_pairs.row(e)=h.transpose();
                alphas[e]=1.;
                pair_margins[e]=std::min(margin(low[p.i]),margin(low[p.j]));
                ++lazy_certified;
                return;
            }
            if (!unlimited && options.lazy) ++lazy_fallback;
            // This antisymmetric low pair includes the B_ij correction in (5.5).
            const Four b=p.viscosity*(low[p.j]-low[p.i])-
                         get_closure(p.j).flux(cij)+get_closure(p.i).flux(cji);
            Four left=.5*(low[p.i]+low[p.j])-
                        (get_closure(p.j).flux(cij)-get_closure(p.i).flux(cij))/(2*p.viscosity);
            Four right=.5*(low[p.i]+low[p.j])-
                         (get_closure(p.i).flux(cji)-get_closure(p.j).flux(cji))/(2*p.viscosity);
            if(!exact::cone(left)) left=stable_base_bar(low[p.i],low[p.j],cij,p.viscosity);
            if(!exact::cone(right)) right=stable_base_bar(low[p.j],low[p.i],cji,p.viscosity);
            cone_require(exact::cone(left)&&exact::cone(right),"R2 directed base bars outside cone");
            const Four raw=project(h)-b;
            const Four delta=raw/(2*p.viscosity);
            double alpha=unlimited?1.:std::min(segment(left,delta),segment(right,-delta));
            auto valid=[&](double a) {
                const Four step=(a*raw)/(2*p.viscosity);
                return exact::cone(left+step)&&exact::cone(right-step);
            };
            if(!unlimited && !valid(alpha)) {
                double lo=0,hi=alpha;
                for(int k=0;k<60;++k) {double mid=(lo+hi)/2;if(valid(mid))lo=mid;else hi=mid;}
                alpha=lo;
            }
            if(!unlimited) cone_require(valid(alpha),"R2 corrected directed bars outside cone");
            high_pairs.row(e)=h.transpose();
            set_low(h,b+alpha*raw);
            out.accepted_pairs.row(e)=h.transpose(); alphas[e]=alpha;
            const Four step=(alpha*raw)/(2*p.viscosity);
            pair_margins[e]=std::min(margin(left+step),margin(right-step));
        });
        for(int f=0;f<int(fe.faces.size());++f) if(runtime.owns(f)) {
            const auto &face=fe.faces[f];
            if(face_lazy_certified[f]) {
                for(int j=0;j<int(face.nodes.size());++j) {
                    const int i=face.nodes[j]; Vec h=face_high[f].row(j).transpose();
                    target_boundary.row(i)+=h.transpose();
                    set_low(h,project(h)); out.boundary_gather.row(i)+=h.transpose();
                }
                continue;
            }
            for(int j=0;j<int(face.nodes.size());++j) {
                const double a=face.mass[j]; if(a<=0) continue;
                const int i=face.nodes[j];
                Vec h=face_high[f].row(j).transpose();
                const Four hl=face_low[f].row(j).transpose();
                const Four base=face_bar[f].row(j).transpose();
                cone_require(exact::cone(base),"R2 integrated ghost base outside cone");
                const Four raw=hl-project(h),delta=raw/a;
                double theta=unlimited?1.:segment(base,delta);
                auto valid=[&](double t){return exact::cone(base+(t*raw)/a);};
                if(!unlimited && !valid(theta)) {
                    double lo=0,hi=theta;
                    for(int k=0;k<60;++k){double mid=(lo+hi)/2;if(valid(mid))lo=mid;else hi=mid;}
                    theta=lo;
                }
                if(!unlimited) cone_require(valid(theta),"R2 corrected ghost outside cone");
                target_boundary.row(i)+=h.transpose();
                set_low(h,hl-theta*raw);
                out.boundary_gather.row(i)+=h.transpose();
                diag.minimum_eta=std::min(diag.minimum_eta,theta);
                diag.minimum_bar=std::min(diag.minimum_bar,margin(base+theta*delta));
                diag.active_faces+=theta<1;
            }
        }
    } catch(const StepRejected &error) {local_failure=1;failure=error.what();}
      catch(const std::exception &error) {local_failure=2;failure=error.what();}
    runtime.finish_local_phase(local_failure,failure);
    diag.lazy_certified_pairs += lazy_certified.load();
    diag.lazy_fallback_pairs += lazy_fallback.load();
    timings.mcl+=seconds()-start;
    Field target_gather=gather_pairs(high_pairs)-target_boundary;
    out.gather=gather_pairs(out.accepted_pairs)-out.boundary_gather;
    runtime.sum(target_gather);runtime.sum(out.gather);
    target_gather+=out.source;out.gather+=out.source;
    diag.target_recovery=maxabs(target_gather-fe.mass.asDiagonal()*out.target+out.mass_error);
    require(diag.target_recovery<5e-11*std::max(1.,maxabs(rhs)),"R2 full residual identity failed");
    out.rate=out.target;
    for(int i=0;i<fe.nodes;++i) {
        Vec v=out.rate.row(i).transpose();
        const Four projected=project(out.gather.row(i).transpose())/fe.mass[i];
        set_low(v,projected);out.rate.row(i)=v.transpose();
        diag.projection=std::max(diag.projection,(project(v)-projected).lpNorm<Eigen::Infinity>());
    }
    Field defect=fe.mass.asDiagonal()*out.rate-out.gather;
    diag.local_balance=maxabs(defect);diag.mass_defect=maxabs(out.mass_error);
    for(int i=0;i<fe.nodes;++i) {
        Vec high_error=out.mass_error.row(i).transpose();set_low(high_error,Four::Zero());
        defect.row(i)-=high_error.transpose();
    }
    require(maxabs(defect)<5e-11*std::max(1.,maxabs(out.gather)),"R2 all-Q ledger failed");
    if(pn.q>4) diag.high_rate_error=maxabs(out.rate.rightCols(pn.q-4)-out.target.rightCols(pn.q-4));
    const Field mu=fe.M*u;
    diag.target_energy=(mu.array()*out.target.array()).sum();
    diag.sync_work=(mu.array()*(out.rate-out.target).array()).sum();
    diag.minimum_alpha=runtime.reduce(alphas.minCoeff(),-1);
    diag.minimum_eta=runtime.reduce(diag.minimum_eta,-1);
    diag.minimum_bar=runtime.reduce(std::min(diag.minimum_bar,pair_margins.minCoeff()),-1);
    diag.active_pairs=long(runtime.reduce((alphas.array()<1).count(),0));
    diag.active_faces=long(runtime.reduce(diag.active_faces,0));
    diag.closure_evaluations=long(runtime.reduce(
        std::count_if(closure.begin(),closure.end(),[](const auto &c){return bool(c);}),0));
    return out;
}

Stage Transport::evaluate_imex(double time, double dt, const Field &u) {
    require(options.r2,"IMEX evaluation requires the R2 transport path");
    require(dt>0 && std::isfinite(dt),"IMEX step size must be finite and positive");
    return evaluate_r2(time,u,false,Field(),dt);
}
}
