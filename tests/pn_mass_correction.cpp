#include "time.hpp"
#include <iomanip>
#include <iostream>

using namespace m1pn;

namespace {
int checks = 0;

void check(bool ok, const std::string &name, double value = 0) {
    std::cout << std::setprecision(17) << "{\"check\":\"" << name
              << "\",\"value\":" << value << ",\"passed\":"
              << (ok ? "true" : "false") << "}\n";
    require(ok, name);
    ++checks;
}

void close(double error, double tolerance, const std::string &name) {
    check(std::isfinite(error) && std::abs(error) <= tolerance, name, error);
}

Field admissible_state(const Spatial &fe, const Angular &pn) {
    Field u = Field::Zero(fe.nodes, pn.q);
    for (int i = 0; i < fe.nodes; ++i) {
        const double x = fe.coordinates(i, 0);
        Vec row = Vec::Zero(pn.q);
        set_low(row, Four(2.0 + .05 * std::sin(2 * pi * x), .17, -.08, .04));
        for (int k = 4; k < pn.q; ++k)
            row[k] = .003 * std::sin((i + 1.) * (k + 2.));
        u.row(i) = row.transpose();
    }
    return u;
}

Numerics candidate_options(const std::string &kind = "m1pn_mass_correction") {
    Numerics n;
    n.r2 = true;
    n.stabilization = kind;
    n.mass_mode = "lumped";
    n.pn_viscosity_scale = 1.;
    return n;
}

void algebra_case(int dimension, bool periodic) {
    MeshSpec spec;
    spec.dimension = dimension;
    spec.elements = 3;
    spec.periodic = periodic;
    Spatial fe(spec, 1, {}, false);
    Angular pn(2);
    Numerics n = candidate_options();
    Transport tr(pn, fe, n);
    Field u = admissible_state(fe, pn);
    Field boundary = Field::Zero(fe.nodes, pn.q);
    Field source = Field::Zero(fe.nodes, pn.q);
    for (int i = 0; i < fe.nodes; ++i)
        for (int k = 0; k < pn.q; ++k) {
            boundary(i, k) = .013 * std::sin((i + 2.) * (k + 1.));
            source(i, k) = .021 * std::cos((i + 1.) * (k + 3.));
        }

    const Field low_with_load = tr.pn_low_order_rate(u, boundary, source);
    const Field low_without_load = tr.pn_low_order_rate(u);
    Field expected_load = boundary + source;
    for (int i = 0; i < fe.nodes; ++i)
        expected_load.row(i) /= fe.mass[i];
    close(maxabs(low_with_load - low_without_load - expected_load), 3e-13,
          "pn_lo_boundary_and_source_load_dim" + std::to_string(dimension) +
              (periodic ? "_periodic" : "_open"));

    // Independent endpoint-wise oracle for the strong graph predictor. In an
    // open mesh C_ij and C_ji differ; using a single antisymmetric edge flux
    // here would silently omit the boundary B_ij contribution.
    Field predictor_oracle = boundary + source;
    for (const auto &edge : fe.pairs) {
        V3 cij = V3::Zero(), cji = V3::Zero();
        for (int d = 0; d < 3; ++d) {
            cij[d] = fe.C[d].coeff(edge.i, edge.j);
            cji[d] = fe.C[d].coeff(edge.j, edge.i);
        }
        const Vec ai = u.row(edge.i).transpose(), aj = u.row(edge.j).transpose();
        const double dN = pn.speed * std::max(cij.norm(), cji.norm());
        const Vec at_i = aj - ai, at_j = ai - aj;
        predictor_oracle.row(edge.i) +=
            (dN * at_i - pn.apply(cij, at_i)).transpose();
        predictor_oracle.row(edge.j) +=
            (dN * at_j - pn.apply(cji, at_j)).transpose();
    }
    for (const auto &face : fe.faces)
        for (const auto &point : face.points) {
            Vec trace = Vec::Zero(pn.q);
            if (n.pn_boundary_quadrature == "consistent")
                for (std::size_t j = 0; j < face.nodes.size(); ++j)
                    trace.noalias() += point.shape[j] * u.row(face.nodes[j]).transpose();
            for (std::size_t j = 0; j < face.nodes.size(); ++j) {
                const int i = face.nodes[j];
                const Vec nodal = n.pn_boundary_quadrature == "consistent"
                                      ? trace
                                      : Vec(u.row(i).transpose());
                predictor_oracle.row(i) +=
                    (point.weight * point.shape[j] * pn.apply(point.normal, nodal)).transpose();
            }
        }
    for (int i = 0; i < fe.nodes; ++i)
        predictor_oracle.row(i) /= fe.mass[i];
    close(maxabs(low_with_load - predictor_oracle), 4e-13,
          "independent_directed_strong_predictor_oracle_dim" +
              std::to_string(dimension) + (periodic ? "_periodic" : "_open"));

    const Field correction = tr.mass_correction(low_with_load);
    const Field independent = fe.mass.asDiagonal() * low_with_load - fe.M * low_with_load;
    close(maxabs(correction - independent), 3e-13,
          "mass_correction_is_D_minus_M_dim" + std::to_string(dimension) +
              (periodic ? "_periodic" : "_open"));
    close(correction.colwise().sum().cwiseAbs().maxCoeff(), 5e-13,
          "mass_correction_conserves_each_PN_component_dim" +
              std::to_string(dimension) + (periodic ? "_periodic" : "_open"));

    Field consistent_rhs = Field::Zero(fe.nodes, pn.q);
    for (int d = 0; d < 3; ++d)
        consistent_rhs.noalias() += fe.C[d].transpose() * (u * pn.A[d].transpose());
    consistent_rhs += boundary + source;
    const Field target = tr.mass_correction_target(consistent_rhs, low_with_load);
    Field target_oracle = consistent_rhs + independent;
    for (int i = 0; i < fe.nodes; ++i)
        target_oracle.row(i) /= fe.mass[i];
    close(maxabs(target - target_oracle), 3e-13,
          "single_mass_correction_full_component_target_dim" +
              std::to_string(dimension) + (periodic ? "_periodic" : "_open"));
    close(maxabs(fe.mass.asDiagonal() * target - consistent_rhs - correction), 8e-13,
          "componentwise_target_reconstruction_dim" + std::to_string(dimension) +
              (periodic ? "_periodic" : "_open"));
    check(!fe.factorized && fe.mass_solve_calls == 0,
          "candidate_does_not_factor_or_solve_consistent_mass_dim" +
              std::to_string(dimension) + (periodic ? "_periodic" : "_open"));
    check(!tr.stabilization_operator,
          "candidate_has_no_biharmonic_or_ghost_operator_dim" +
              std::to_string(dimension) + (periodic ? "_periodic" : "_open"));
}

Field consistent_load(const Transport &tr, const Field &u) {
    Field rhs = Field::Zero(tr.fe.nodes, tr.pn.q);
    for (int d = 0; d < 3; ++d)
        rhs.noalias() += tr.fe.C[d].transpose() * (u * tr.pn.A[d].transpose());
    for (const auto &face : tr.fe.faces)
        for (const auto &point : face.points) {
            Vec trace = Vec::Zero(tr.pn.q);
            for (std::size_t j = 0; j < face.nodes.size(); ++j)
                trace.noalias() += point.shape[j] * u.row(face.nodes[j]).transpose();
            for (std::size_t j = 0; j < face.nodes.size(); ++j) {
                const Vec nodal = tr.options.pn_boundary_quadrature == "consistent"
                                      ? trace
                                      : Vec(u.row(face.nodes[j]).transpose());
                const Vec flux = tr.boundaries[point.normal_id]->llf_homogeneous * nodal;
                rhs.row(face.nodes[j]) -=
                    (point.weight * point.shape[j] * flux).transpose();
            }
        }
    return rhs;
}

void linear_target_case(int dimension, bool periodic) {
    MeshSpec spec;
    spec.dimension = dimension;
    spec.elements = 3;
    spec.periodic = periodic;
    Spatial fe(spec, 1, {}, false);
    Angular pn(3);
    Numerics n = candidate_options();
    Transport tr(pn, fe, n);
    Field u = admissible_state(fe, pn);
    const Field rhs = consistent_load(tr, u);
    Field boundary_rhs = rhs;
    for (int d = 0; d < 3; ++d)
        boundary_rhs.noalias() -= fe.C[d].transpose() * (u * pn.A[d].transpose());
    const Field low = tr.pn_low_order_rate(u, boundary_rhs);
    const Field expected = tr.mass_correction_target(rhs, low);
    const Field actual = tr.linear_target(u);
    close(maxabs(actual - expected), 5e-12,
          "production_linear_target_matches_independent_full_Q_reconstruction_dim" +
              std::to_string(dimension) + (periodic ? "_periodic" : "_open"));
    check(fe.mass_solve_calls == 0 && !fe.factorized,
          "linear_target_uses_only_lumped_mass_dim" + std::to_string(dimension) +
              (periodic ? "_periodic" : "_open"));
}

void boundary_exterior_independence() {
    MeshSpec spec;
    spec.dimension = 1;
    spec.elements = 3;
    Spatial fe(spec, 1, {}, false);
    Angular pn(2);
    Numerics n = candidate_options();
    int first_calls = 0, second_calls = 0;
    Inflow first, second;
    first.exterior_m1 = [&first_calls](const V3 &, double) {
        ++first_calls;
        return Four(2., .2, 0., 0.);
    };
    second.exterior_m1 = [&second_calls](const V3 &, double) {
        ++second_calls;
        return Four(3., -.3, .1, 0.);
    };
    Transport a(pn, fe, n, first), b(pn, fe, n, second);
    const Field u = admissible_state(fe, pn);
    const Stage sa = a.evaluate(0., u, true), sb = b.evaluate(0., u, true);
    close(maxabs(sa.target - sb.target), 2e-11,
          "PN_target_independent_of_independently_selected_M1_exterior");
    check(first_calls > 0 && second_calls > 0,
          "independent_M1_exteriors_are_consumed_by_boundary_path",
          first_calls + second_calls);
    check(fe.mass_solve_calls == 0 && !fe.factorized,
          "boundary_candidate_avoids_consistent_mass_solve");
}

Field pn_boundary_load(const Transport &tr, const Field &u, double time) {
    Field load = Field::Zero(tr.fe.nodes, tr.pn.q);
    for (const auto &face : tr.fe.faces)
        for (const auto &point : face.points) {
            const auto &boundary = *tr.boundaries[point.normal_id];
            const auto data = boundary.data(tr.inflow, point.x, time);
            for (std::size_t j = 0; j < face.nodes.size(); ++j) {
                const int i = face.nodes[j];
                Vec interior = u.row(i).transpose();
                if (tr.options.pn_boundary_quadrature == "consistent") {
                    interior.setZero();
                    for (std::size_t k = 0; k < face.nodes.size(); ++k)
                        interior.noalias() += point.shape[k] *
                            u.row(face.nodes[k]).transpose();
                }
                const Vec exterior = boundary.exterior(interior, data.d);
                const Vec flux = .5 * (tr.pn.apply(point.normal, interior) +
                                       tr.pn.apply(point.normal, exterior)) -
                                 .5 * (exterior - interior);
                load.row(i) -= (point.weight * point.shape[j] * flux).transpose();
            }
        }
    return load;
}

void full_mcl_identity(int dimension, bool periodic,
                       const std::string &kind = "m1pn_mass_correction") {
    MeshSpec spec;
    spec.dimension = dimension;
    spec.elements = 3;
    spec.periodic = periodic;
    Spatial fe(spec, 1, {}, false);
    Angular pn(2);
    Numerics n = candidate_options(kind);
    int exterior_calls = 0;
    Inflow inflow;
    inflow.density = [](const V3 &, double, const V3 &) { return .02 / (4 * pi); };
    inflow.exterior_m1 = [&exterior_calls](const V3 &, double) {
        ++exterior_calls;
        return Four(1.7, .11, -.04, .03);
    };
    Transport tr(pn, fe, n, inflow);
    const Field u = admissible_state(fe, pn);
    Field source = Field::Zero(fe.nodes, pn.q);
    for (int i = 0; i < fe.nodes; ++i) {
        source(i, 0) = .001 * fe.mass[i];
        for (int k = 4; k < pn.q; ++k)
            source(i, k) = .002 * fe.mass[i] * std::sin((i + 1.) * (k + 1.));
    }
    const Stage stage = tr.evaluate(0., u, true, source);
    const Field boundary_rhs = pn_boundary_load(tr, u, 0.);
    Field consistent_rhs = Field::Zero(fe.nodes, pn.q);
    for (int d = 0; d < 3; ++d)
        consistent_rhs.noalias() += fe.C[d].transpose() * (u * pn.A[d].transpose());
    consistent_rhs += boundary_rhs + source;
    const Field low_rate = tr.pn_low_order_rate(u, boundary_rhs, source);
    const Field correction = fe.mass.asDiagonal() * low_rate - fe.M * low_rate;
    Field target_oracle = consistent_rhs + correction;
    for (int i = 0; i < fe.nodes; ++i)
        target_oracle.row(i) /= fe.mass[i];
    close(maxabs(stage.target - target_oracle), 2e-10,
          "production_stage_full_Q_target_reconstruction_dim" +
              std::to_string(dimension) + (periodic ? "_periodic" : "_open"));
    if (pn.q > 4)
        close(maxabs(stage.rate.rightCols(pn.q - 4) - stage.target.rightCols(pn.q - 4)),
              2e-11, "MCL_leaves_all_high_PN_rates_unchanged_dim" +
                         std::to_string(dimension) + (periodic ? "_periodic" : "_open"));

    std::vector<Four> m1(fe.nodes);
    std::vector<Levermore> closure;
    closure.reserve(fe.nodes);
    for (int i = 0; i < fe.nodes; ++i) {
        m1[i] = project(u.row(i).transpose());
        closure.emplace_back(m1[i]);
    }
    Field pair_low = Field::Zero(fe.nodes, 4), pair_mc = pair_low;
    double high_pair_error = 0, bij_formula_error = 0, antisymmetry_error = 0;
    for (std::size_t e = 0; e < fe.pairs.size(); ++e) {
        const auto &edge = fe.pairs[e];
        V3 cij = V3::Zero(), cji = V3::Zero();
        for (int d = 0; d < 3; ++d) {
            cij[d] = fe.C[d].coeff(edge.i, edge.j);
            cji[d] = fe.C[d].coeff(edge.j, edge.i);
        }
        const Vec ai = u.row(edge.i).transpose(), aj = u.row(edge.j).transpose();
        const Vec high = pn.apply(cji, aj) - pn.apply(cij, ai) +
                         edge.mass * (low_rate.row(edge.i) - low_rate.row(edge.j)).transpose();
        const Four b = edge.viscosity * (m1[edge.j] - m1[edge.i]) -
                       closure[edge.j].flux(cij) + closure[edge.i].flux(cji);
        const Four direct_from_target_pair = project(stage.accepted_pairs.row(e).transpose()) - b;
        const Four Ti_cij = project(pn.apply(cij, ai)) - closure[edge.i].flux(cij);
        const Four Tj_cji = project(pn.apply(cji, aj)) - closure[edge.j].flux(cji);
        const V3 bij = cij + cji;
        const Four formula = edge.mass * project((low_rate.row(edge.i) -
                                                   low_rate.row(edge.j)).transpose()) +
                             edge.viscosity * (m1[edge.i] - m1[edge.j]) +
                             Tj_cji - Ti_cij + closure[edge.j].flux(bij) -
                             closure[edge.i].flux(bij);
        const Four reverse = edge.mass * project((low_rate.row(edge.j)-low_rate.row(edge.i)).transpose()) +
                             edge.viscosity*(m1[edge.j]-m1[edge.i]) +
                             Ti_cij-Tj_cji+closure[edge.i].flux(bij)-closure[edge.j].flux(bij);
        antisymmetry_error=std::max(antisymmetry_error,(formula+reverse).lpNorm<Eigen::Infinity>());
        high_pair_error = std::max(high_pair_error,
            (project(high) - project(stage.accepted_pairs.row(e).transpose())).norm());
        bij_formula_error = std::max(bij_formula_error,
            (direct_from_target_pair - formula).lpNorm<Eigen::Infinity>());
        pair_low.row(edge.i) += b.transpose();
        pair_low.row(edge.j) -= b.transpose();
        pair_mc.row(edge.i) += formula.transpose();
        pair_mc.row(edge.j) -= formula.transpose();
    }
    close(antisymmetry_error, 3e-13, "all_inner_raw_corrections_antisymmetric_dim"+std::to_string(dimension)+(periodic?"_periodic":"_open"));
    close(high_pair_error, 2e-10,
          "unlimited_stage_retains_full_PN_edge_target_dim" +
              std::to_string(dimension) + (periodic ? "_periodic" : "_open"));
    close(bij_formula_error, 3e-10,
          "independent_Bij_model_flux_defect_formula_dim" +
              std::to_string(dimension) + (periodic ? "_periodic" : "_open"));

    Field physical_m1_boundary = Field::Zero(fe.nodes, 4);
    Field physical_pn_boundary = Field::Zero(fe.nodes, pn.q);
    for (const auto &face : fe.faces)
        for (const auto &point : face.points) {
            const auto &boundary = *tr.boundaries[point.normal_id];
            const auto data = boundary.data(inflow, point.x, 0.);
            const Four exterior = inflow.exterior_m1(point.x, 0.);
            const Levermore ext_closure(exterior);
            for (std::size_t j = 0; j < face.nodes.size(); ++j) {
                const int i = face.nodes[j];
                const Vec interior = u.row(i).transpose();
                const Vec pn_exterior = boundary.exterior(interior, data.d);
                const Vec pn_flux = .5 * (pn.apply(point.normal, interior) +
                                          pn.apply(point.normal, pn_exterior)) -
                                    .5 * (pn_exterior - interior);
                const Four m1_flux = .5 * (closure[i].flux(point.normal) +
                                           ext_closure.flux(point.normal)) -
                                     .5 * (exterior - m1[i]);
                const double weight = point.weight * point.shape[j];
                physical_pn_boundary.row(i) -= (weight * pn_flux).transpose();
                physical_m1_boundary.row(i) += (weight * m1_flux).transpose();
            }
        }
    close(maxabs(physical_pn_boundary - boundary_rhs), 2e-12,
          "independent_physical_PN_boundary_load_matches_stage_input_dim" +
              std::to_string(dimension) + (periodic ? "_periodic" : "_open"));
    Field lhs = Field::Zero(fe.nodes, 4), rhs = pair_mc + physical_m1_boundary;
    for (int i = 0; i < fe.nodes; ++i) {
        lhs.row(i) = fe.mass[i] * project(stage.target.row(i).transpose()).transpose() -
                     pair_low.row(i) + physical_m1_boundary.row(i) -
                     project(source.row(i).transpose()).transpose();
        // L_M1 = pair_low - physical_m1_boundary; moving it to the left gives +H_M1.
    }
    // The boundary correction is H_M1 - Pi(H_N); `physical_pn_boundary` stores -H_N.
    for (int i = 0; i < fe.nodes; ++i)
        rhs.row(i) += project(physical_pn_boundary.row(i).transpose()).transpose();
    close(maxabs(lhs - rhs), 4e-10,
          "complete_componentwise_M1PN_MCL_target_identity_dim" +
              std::to_string(dimension) + (periodic ? "_periodic" : "_open"));
    check(periodic || exterior_calls > 0, "M1_exterior_used_in_full_boundary_identity",
          exterior_calls);
}

void exact_vacuum() {
    MeshSpec spec;
    spec.dimension = 2;
    spec.elements = 2;
    Spatial fe(spec, 1, {}, false);
    Angular pn(2);
    Numerics n = candidate_options();
    Transport tr(pn, fe, n);
    const Field zero = Field::Zero(fe.nodes, pn.q);
    const Stage stage = tr.evaluate(0., zero, true);
    check(maxabs(stage.target) == 0 && maxabs(stage.rate) == 0,
          "candidate_preserves_exact_vacuum");
    check(fe.mass_solve_calls == 0 && !fe.factorized,
          "vacuum_candidate_avoids_consistent_mass_solve");
}

void ssprk3_cone_and_high_modes() {
    MeshSpec spec;
    spec.dimension = 2;
    spec.elements = 2;
    Spatial fe(spec, 1, {}, false);
    Angular pn(2);
    Numerics n = candidate_options();
    Transport tr(pn, fe, n);
    Field u = admissible_state(fe, pn);
    const double h = .01 * fe.cfl_step;
    auto admissible = [&](const Field &v) {
        for (int i = 0; i < fe.nodes; ++i)
            if (!exact::cone(project(v.row(i).transpose()))) return false;
        return true;
    };
    auto stage_rate = [&](double t, const Field &v, const std::string &label) {
        const Stage s = tr.evaluate(t, v, false);
        if (pn.q > 4)
            close(maxabs(s.rate.rightCols(pn.q - 4) - s.target.rightCols(pn.q - 4)),
                  2e-10, label + "_high_PN_rates_preserved");
        return s.rate;
    };
    const Field k1 = stage_rate(0., u, "ssprk3_stage1");
    const Field u1 = u + h * k1;
    check(admissible(u1), "candidate_SSPRK3_forward_Euler_state_in_cone");
    const Field k2 = stage_rate(h, u1, "ssprk3_stage2");
    const Field u2 = .75 * u + .25 * (u1 + h * k2);
    check(admissible(u2), "candidate_SSPRK3_second_convex_state_in_cone");
    const Field k3 = stage_rate(.5 * h, u2, "ssprk3_stage3");
    const Field u3 = u / 3. + (2. / 3.) * (u2 + h * k3);
    check(admissible(u3), "candidate_SSPRK3_final_state_in_cone");
    check(fe.mass_solve_calls == 0 && !fe.factorized,
          "candidate_SSPRK3_stages_use_no_consistent_mass_solve");
}

void boundary_consistent_affine(int dimension) {
    MeshSpec spec;
    spec.dimension = dimension;
    spec.elements = 3;
    Spatial fe(spec, 1, {}, false);
    Angular pn(3);
    Numerics nb = candidate_options("m1pn_mass_correction");
    nb.pn_boundary_quadrature = "consistent";
    Numerics nc = candidate_options("m1pn_mass_correction_boundary_consistent");
    nc.pn_boundary_quadrature = "consistent";
    const V3 gradient(1., dimension > 1 ? .5 : 0., dimension == 3 ? -.3 : 0.);
    Inflow affine;
    affine.density=[gradient](const V3 &x,double t,const V3 &w){
        return (1.+.12*gradient.dot(x)-.12*t*gradient.dot(w))/(4*pi);
    };
    affine.exterior_m1=[gradient](const V3 &x,double t){
        return Four(1.+.12*gradient.dot(x),-.04*t*gradient[0],-.04*t*gradient[1],-.04*t*gradient[2]);
    };
    Transport b(pn, fe, nb, affine), c(pn, fe, nc, affine);
    const auto pairs_before = fe.pairs.size();
    Field u = Field::Zero(fe.nodes, pn.q), exact_rate = u;
    for (int i = 0; i < fe.nodes; ++i) {
        const V3 x(fe.coordinates(i,0), fe.coordinates(i,1), fe.coordinates(i,2));
        set_low(u.row(i), Four(1. + .12 * gradient.dot(x), 0., 0., 0.));
        set_low(exact_rate.row(i), Four(0., -.04*gradient[0], -.04*gradient[1], -.04*gradient[2]));
    }
    const Field boundary = pn_boundary_load(b, u, 0.);
    const Field rb = b.pn_low_order_rate(u, boundary);
    const Field rc = c.pn_low_order_rate(u, boundary);
    // Independent tensor-index coordinate clamp, matching the stated C map.
    std::array<std::vector<double>,3> axes;
    std::array<int,3> count{1,1,1};
    for(int d=0;d<dimension;++d) {
        for(int i=0;i<fe.nodes;++i) axes[d].push_back(fe.coordinates(i,d));
        std::sort(axes[d].begin(),axes[d].end());
        axes[d].erase(std::unique(axes[d].begin(),axes[d].end(),[](double x,double y){return std::abs(x-y)<2e-12;}),axes[d].end());
        count[d]=int(axes[d].size());
    }
    auto key=[&](const std::array<int,3>&q){return (q[0]*count[1]+q[1])*count[2]+q[2];};
    std::vector<int> at(fe.nodes,-1), map(fe.nodes);
    std::vector<std::array<int,3>> multi(fe.nodes);
    for(int i=0;i<fe.nodes;++i) {
        std::array<int,3> q{0,0,0};
        for(int d=0;d<dimension;++d) {
            auto it=std::lower_bound(axes[d].begin(),axes[d].end(),fe.coordinates(i,d));
            q[d]=int(it-axes[d].begin());
        }
        multi[i]=q; at[key(q)]=i;
    }
    for(int i=0;i<fe.nodes;++i) {
        auto q=multi[i];
        if(fe.boundary[i]) for(int d=0;d<dimension;++d) q[d]=std::clamp(q[d],1,count[d]-2);
        map[i]=at[key(q)];
    }
    check(map==c.pn_boundary_map,"C_coordinate_clamp_map_dim"+std::to_string(dimension));
    Field mapped=rc;
    for(int i=0;i<fe.nodes;++i) mapped.row(i)=rb.row(map[i]);
    close(maxabs(rc-mapped),3e-13,"C_predictor_is_B_rate_with_independent_clamp_dim"+std::to_string(dimension));
    check(maxabs(b.mass_correction(rb))>1e-6,"B_affine_predictor_has_nonzero_Z_dim"+std::to_string(dimension),maxabs(b.mass_correction(rb)));
    close(maxabs(c.mass_correction(rc)),2e-12,"C_affine_predictor_has_zero_Z_dim"+std::to_string(dimension));
    const Stage stage=c.evaluate(0.,u,true);
    close(maxabs(stage.target-exact_rate),3e-10,"C_affine_all_Q_target_exact_dim"+std::to_string(dimension));
    const Stage limited=c.evaluate(0.,u,false);
    close(maxabs(limited.rate-exact_rate),3e-10,"C_limited_affine_all_Q_target_exact_dim"+std::to_string(dimension));
    check(stage.diagnostics.active_pairs==0,"C_affine_target_requires_no_pair_clipping_dim"+std::to_string(dimension),stage.diagnostics.active_pairs);
    check(fe.pairs.size()==pairs_before,"C_preserves_spatial_graph_dim"+std::to_string(dimension));
    check(fe.mass_solve_calls==0&&!fe.factorized&&!c.stabilization_operator,
          "C_uses_no_consistent_mass_or_stabilization_dim"+std::to_string(dimension));
}

void boundary_consistent_guards_and_periodic() {
    for(int dim=1;dim<=3;++dim) {
        MeshSpec s; s.dimension=dim; s.elements=3; s.periodic=true;
        Spatial fe(s,1,{},false); Angular pn(2);
        Transport b(pn,fe,candidate_options()), c(pn,fe,candidate_options("m1pn_mass_correction_boundary_consistent"));
        const Field u=admissible_state(fe,pn);
        check(c.pn_boundary_map.size()==std::size_t(fe.nodes),"C_periodic_map_present");
        check(c.pn_boundary_map_norm==1.,"C_periodic_map_identity_norm");
        close(maxabs(c.pn_low_order_rate(u)-b.pn_low_order_rate(u)),2e-13,"C_periodic_equals_B_dim"+std::to_string(dim));
    }
    bool rejected=false;
    try { MeshSpec s; s.dimension=2; s.elements=2; Spatial fe(s,1,{},false); Angular pn(1); Transport c(pn,fe,candidate_options("m1pn_mass_correction_boundary_consistent")); }
    catch(const std::exception&) { rejected=true; }
    check(rejected,"C_rejects_fewer_than_three_open_cells");
    rejected=false;
    try { MeshSpec s; s.dimension=3; s.elements=3; s.tetrahedra=true; Spatial fe(s,1,{},false); Angular pn(1); Transport c(pn,fe,candidate_options("m1pn_mass_correction_boundary_consistent")); }
    catch(const std::exception&) { rejected=true; }
    check(rejected,"C_rejects_tetrahedral_grid");
}

void boundary_consistent_vacuum_ssprk() {
    MeshSpec spec; spec.dimension=2; spec.elements=3;
    Spatial fe(spec,1,{},false); Angular pn(2);
    Transport tr(pn,fe,candidate_options("m1pn_mass_correction_boundary_consistent"));
    const Field zero=Field::Zero(fe.nodes,pn.q);
    const Stage vacuum=tr.evaluate(0.,zero,true);
    check(maxabs(vacuum.target)==0&&maxabs(vacuum.rate)==0,"C_preserves_exact_vacuum");
    Field u=admissible_state(fe,pn);
    const double h=.01*fe.cfl_step;
    auto cone_ok=[&](const Field&v){for(int i=0;i<fe.nodes;++i)if(!exact::cone(project(v.row(i).transpose())))return false;return true;};
    auto rate=[&](double t,const Field&v){const Stage s=tr.evaluate(t,v,false);if(pn.q>4)close(maxabs(s.rate.rightCols(pn.q-4)-s.target.rightCols(pn.q-4)),2e-10,"C_SSPRK_high_modes_preserved");return s.rate;};
    const Field k1=rate(0.,u),u1=u+h*k1;
    check(cone_ok(u1),"C_SSPRK_forward_euler_cone");
    const Field k2=rate(h,u1),u2=.75*u+.25*(u1+h*k2);
    check(cone_ok(u2),"C_SSPRK_second_state_cone");
    const Field k3=rate(.5*h,u2),u3=u/3.+(2./3.)*(u2+h*k3);
    check(cone_ok(u3),"C_SSPRK_final_state_cone");
    check(fe.mass_solve_calls==0&&!fe.factorized&&!tr.stabilization_operator,"C_SSPRK_no_mass_solve_or_stabilizer");
}

} // namespace

int main() {
    try {
        for (int dimension = 1; dimension <= 3; ++dimension) {
            algebra_case(dimension, true);
            algebra_case(dimension, false);
        }
        linear_target_case(1, true);
        linear_target_case(1, false);
        linear_target_case(2, true);
        linear_target_case(2, false);
        linear_target_case(3, true);
        linear_target_case(3, false);
        boundary_exterior_independence();
        for (int dimension = 1; dimension <= 3; ++dimension) {
            full_mcl_identity(dimension, true);
            full_mcl_identity(dimension, false);
            full_mcl_identity(dimension, false, "m1pn_mass_correction_boundary_consistent");
        }
        exact_vacuum();
        ssprk3_cone_and_high_modes();
        for (int dimension=1; dimension<=3; ++dimension) boundary_consistent_affine(dimension);
        boundary_consistent_guards_and_periodic();
        boundary_consistent_vacuum_ssprk();
        std::cout << "pn_mass_correction checks=" << checks << " passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
