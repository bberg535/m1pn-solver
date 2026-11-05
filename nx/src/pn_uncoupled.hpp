#pragma once

#include "time.hpp"

namespace m1pn {

struct PNOnlyStateDiagnostics {
    double minimum_density = std::numeric_limits<double>::infinity();
    double minimum_cone_margin = std::numeric_limits<double>::infinity();
    long negative_density_nodes = 0;
    long outside_m1_cone_nodes = 0;
};

struct PNOnlyStage {
    Stage stage;
    PNOnlyStateDiagnostics state;
    double maximum_target_recovery = 0;
};

struct PNOnlyStep {
    Step step;
    std::vector<PNOnlyStateDiagnostics> observations;
    double maximum_target_recovery = 0;
};

inline void pn_only_require_finite(const Field &u, const Spatial &fe, const Angular &pn) {
    require(u.rows() == fe.nodes && u.cols() == pn.q && u.allFinite(),
            "PN-only evolution produced an invalid or nonfinite field");
}

inline PNOnlyStateDiagnostics pn_only_state_diagnostics(const Field &u,
                                                        const Spatial &fe,
                                                        const Angular &pn) {
    pn_only_require_finite(u, fe, pn);
    PNOnlyStateDiagnostics out;
    for (int i = 0; i < fe.nodes; ++i) {
        const Four low = project(u.row(i).transpose());
        const double rho = low[0];
        const double cone_margin = margin(low);
        out.minimum_density = std::min(out.minimum_density, rho);
        out.minimum_cone_margin = std::min(out.minimum_cone_margin, cone_margin);
        out.negative_density_nodes += rho < 0.;
        out.outside_m1_cone_nodes += !exact::cone(low);
    }
    return out;
}

// Assemble the full weak Galerkin PN residual with the existing time-dependent
// Onsager exterior and LLF boundary flux, then apply the standard-B low-order
// mass correction. This intentionally does not construct M1 closures or limiter
// bars and does not call Transport::evaluate/check_state.
inline PNOnlyStage evaluate_pn_only(Transport &tr, double time, const Field &u) {
    pn_only_require_finite(u, tr.fe, tr.pn);
    PNOnlyStage out;
    auto &s = out.stage;
    out.state = pn_only_state_diagnostics(u, tr.fe, tr.pn);

    Field rhs = Field::Zero(tr.fe.nodes, tr.pn.q);
    for (int d = 0; d < 3; ++d)
        rhs.noalias() += tr.fe.C[d].transpose() * (u * tr.pn.A[d].transpose());

    Field boundary_rhs = Field::Zero(tr.fe.nodes, tr.pn.q);
    s.boundary_gather = Field::Zero(tr.fe.nodes, tr.pn.q);
    for (std::size_t f = 0; f < tr.fe.faces.size(); ++f) {
        const auto &face = tr.fe.faces[f];
        for (const auto &p : face.points) {
            const auto &b = *tr.boundaries.at(p.normal_id);
            const Incoming data = b.data(tr.inflow, p.x, time);
            Vec consistent_flux;
            if (tr.options.pn_boundary_quadrature == "consistent") {
                Vec trace = Vec::Zero(tr.pn.q);
                for (std::size_t j = 0; j < face.nodes.size(); ++j)
                    trace.noalias() += p.shape[j] * u.row(face.nodes[j]).transpose();
                const Vec exterior = b.exterior(trace, data.d);
                consistent_flux = .5 * (tr.pn.apply(p.normal, trace) +
                                         tr.pn.apply(p.normal, exterior)) -
                                  .5 * (exterior - trace);
            }
            for (std::size_t j = 0; j < face.nodes.size(); ++j) {
                const double weight = p.weight * p.shape[j];
                if (weight == 0.) continue;
                const int node = face.nodes[j];
                Vec flux;
                if (tr.options.pn_boundary_quadrature == "consistent") {
                    flux = consistent_flux;
                } else {
                    const Vec interior = u.row(node).transpose();
                    const Vec exterior = b.exterior(interior, data.d);
                    flux = .5 * (tr.pn.apply(p.normal, interior) +
                                 tr.pn.apply(p.normal, exterior)) -
                           .5 * (exterior - interior);
                }
                boundary_rhs.row(node) -= weight * flux.transpose();
                if (tr.runtime.owns(f)) s.boundary_gather.row(node) += weight * flux.transpose();
            }
        }
    }
    rhs += boundary_rhs;

    const Field predictor = tr.pn_low_order_rate(u, boundary_rhs);
    s.target = tr.mass_correction_target(rhs, predictor);
    pn_only_require_finite(s.target, tr.fe, tr.pn);
    s.rate = s.target;
    s.source = Field::Zero(tr.fe.nodes, tr.pn.q);
    s.accepted_pairs = Field::Zero(tr.fe.pairs.size(), tr.pn.q);
    parallel_for(int(tr.fe.pairs.size()), [&](int ei) {
        const std::size_t e = std::size_t(ei);
        if (!tr.runtime.owns(e)) return;
        const auto &pair = tr.fe.pairs[e];
        V3 cij = V3::Zero(), cji = V3::Zero();
        for (int d = 0; d < 3; ++d) {
            cij[d] = tr.fe.C[d].coeff(pair.i, pair.j);
            cji[d] = tr.fe.C[d].coeff(pair.j, pair.i);
        }
        const Vec ui = u.row(pair.i).transpose();
        const Vec uj = u.row(pair.j).transpose();
        const Vec flux = tr.pn.apply(cji, uj) - tr.pn.apply(cij, ui) +
                         pair.mass * (predictor.row(pair.i) - predictor.row(pair.j)).transpose();
        s.accepted_pairs.row(e) = flux.transpose();
    });
    s.gather = tr.gather_pairs(s.accepted_pairs) - s.boundary_gather;
    tr.runtime.sum(s.gather);
    s.diagnostics.target_recovery = maxabs(tr.fe.mass.asDiagonal() * s.target - s.gather);
    s.diagnostics.local_balance = s.diagnostics.target_recovery;
    s.diagnostics.mass_residual = 0.;
    s.diagnostics.high_rate_error = 0.;
    s.diagnostics.minimum_state = out.state.minimum_cone_margin;
    out.maximum_target_recovery = s.diagnostics.target_recovery;
    return out;
}

inline PNOnlyStep advance_pn_only(Transport &tr, const TimeControl &control,
                                  double time, double dt, const Field &initial) {
    control.check(dt);
    require(control.order == 3, "PN-only stepping currently requires SSPRK3");
    require(dt <= tr.fe.cfl_step * (1 + 2e-14), "PN-only SSP Euler branch violates spatial CFL");
    PNOnlyStep result;
    Step &out = result.step;
    PNOnlyStage p0 = evaluate_pn_only(tr, time, initial);
    Field u1 = initial + dt * p0.stage.rate;
    PNOnlyStage p1 = evaluate_pn_only(tr, time + dt, u1);
    Field u2 = .75 * initial + .25 * (u1 + dt * p1.stage.rate);
    PNOnlyStage p2 = evaluate_pn_only(tr, time + .5 * dt, u2);
    out.state = initial / 3. + (2. / 3.) * (u2 + dt * p2.stage.rate);
    result.observations.push_back(p0.state);
    result.observations.push_back(p1.state);
    result.observations.push_back(p2.state);
    result.observations.push_back(pn_only_state_diagnostics(out.state, tr.fe, tr.pn));

    const std::array<PNOnlyStage *, 3> stages{{&p0, &p1, &p2}};
    const std::array<double, 3> weights{{1. / 6., 1. / 6., 2. / 3.}};
    out.integrated_pairs = Field::Zero(tr.fe.pairs.size(), tr.pn.q);
    out.integrated_boundary = Field::Zero(tr.fe.nodes, tr.pn.q);
    out.integrated_source = Field::Zero(tr.fe.nodes, tr.pn.q);
    Field weighted_rates = Field::Zero(tr.fe.nodes, tr.pn.q);
    for (int r = 0; r < 3; ++r) {
        const double factor = dt * weights[r];
        out.integrated_pairs += factor * stages[r]->stage.accepted_pairs;
        out.integrated_boundary += factor * stages[r]->stage.boundary_gather;
        weighted_rates += factor * stages[r]->stage.rate;
        out.stages.push_back(stages[r]->stage.diagnostics);
        result.maximum_target_recovery = std::max(result.maximum_target_recovery,
                                                  stages[r]->maximum_target_recovery);
    }
    Field transfers = tr.gather_pairs(out.integrated_pairs) - out.integrated_boundary;
    tr.runtime.sum(transfers);
    out.rate_residual = maxabs(out.state - initial - weighted_rates);
    out.ledger_residual = maxabs(tr.fe.mass.asDiagonal() * (out.state - initial) - transfers);
    out.minimum_margin = std::numeric_limits<double>::infinity();
    for (const auto &obs : result.observations)
        out.minimum_margin = std::min(out.minimum_margin, obs.minimum_cone_margin);
    require(out.state.allFinite(), "PN-only SSPRK3 produced a nonfinite field");
    require(out.ledger_residual < 2e-10 * std::max(1., maxabs(transfers)),
            "PN-only SSPRK3 all-Q ledger failed");
    return result;
}

}  // namespace m1pn
