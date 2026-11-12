// Dynamical low-rank approximation of the element-WENO M1PN scheme; see element_weno_dlra.hpp and docs/DLRA.md.
#include "element_weno_dlra.hpp"
#include "parallel.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <random>
namespace m1pn {
namespace {
// Row blocks for the deterministic parallel reductions (independent of the thread count).
constexpr int kBlock = 2048;
// out = A B (nodes x t), parallel over row blocks.
Field mul(const Field &A, const Mat &B) {
    Field out(A.rows(), B.cols());
    if (A.rows() == 0 || B.cols() == 0)
        return out.setZero(), out;
    if (A.cols() == 0)
        return out.setZero(), out;
    const int blocks = (int(A.rows()) + kBlock - 1) / kBlock;
    parallel_for(blocks, [&](int b) {
        const int begin = b * kBlock, n = std::min<int>(kBlock, A.rows() - begin);
        out.middleRows(begin, n).noalias() = A.middleRows(begin, n) * B;
    });
    return out;
}
// A^T B (A.cols() x B.cols()), fixed row blocks summed in order.
Mat cross(const Field &A, const Field &B) {
    const int blocks = (int(A.rows()) + kBlock - 1) / kBlock;
    std::vector<Mat> part(blocks);
    parallel_for(blocks, [&](int b) {
        const int begin = b * kBlock, n = std::min<int>(kBlock, A.rows() - begin);
        part[b].noalias() = A.middleRows(begin, n).transpose() * B.middleRows(begin, n);
    });
    Mat out = Mat::Zero(A.cols(), B.cols());
    for (const Mat &p : part)
        out += p;
    return out;
}
Field hcat(const Field &A, const Field &B) {
    Field out(A.rows(), A.cols() + B.cols());
    const int blocks = (int(A.rows()) + kBlock - 1) / kBlock;
    parallel_for(blocks, [&](int b) {
        const int begin = b * kBlock, n = std::min<int>(kBlock, A.rows() - begin);
        out.block(begin, 0, n, A.cols()) = A.middleRows(begin, n);
        out.block(begin, A.cols(), n, B.cols()) = B.middleRows(begin, n);
    });
    return out;
}
// r <- old * a + fresh * (x + tau * r) (the SSP stage combination of element_weno.cpp).
void combine(Field &r, const Field &x, const Field &a, double old, double fresh, double tau) {
    const int q = r.cols();
    parallel_for(int(r.rows()), [&](int i) {
        double *ri = &r.coeffRef(i, 0);
        const double *xi = &x.coeffRef(i, 0), *ai = &a.coeffRef(i, 0);
        if (old == 0)
            for (int c = 0; c < q; ++c)
                ri[c] = fresh * (xi[c] + tau * ri[c]);
        else
            for (int c = 0; c < q; ++c)
                ri[c] = old * ai[c] + fresh * (xi[c] + tau * ri[c]);
    });
}
} // namespace

struct ElementWenoDlra::Context {
    Mat V;
    int c = 4;
    std::array<Mat, 3> P, PL; // c x moments: Qt A_d; and its first four columns
    std::array<Mat, 6> B;     // c x moments: Qt (A_n - llf_homogeneous)^T
    std::array<Mat, 6> H;     // outgoing points x c: harmonics Qt^T
};
struct ElementWenoDlra::Evaluation {
    double time = 0;
    Field low, low_flux, closure;
    Vec gamma;
    std::array<Field, 3> grad;
    Field negsh, stab_low;
    std::array<Field, 6> bnd, datum;
    Field boundary_low, flux_area, boundary_bar;
    Vec boundary_area, boundary_inflow;
};

ElementWenoDlra::ElementWenoDlra(const ElementWeno &s, DlraOptions o) : scheme(s), options(o) {
    const ElementWenoOptions &so = s.options;
    require(std::none_of(so.mirror.begin(), so.mirror.end(), [](bool m) { return m; }),
            "DLRA: mirror sides are not supported (use the full domain)");
    require(!so.m1_target, "DLRA needs the PN target");
    require(s.moments > 4, "DLRA needs PN order >= 2");
    for (int k = 0; k < 4; ++k)
        require(s.active[k] == k, "DLRA: the first four active moments must be l <= 1");
    require(o.initial_rank >= 1 && o.min_rank >= 1 && o.tolerance >= 0 && o.absolute_tolerance >= 0,
            "DLRA: invalid rank or tolerance");
    qh = s.moments - 4;
    for (int d = 0; d < s.dimension; ++d) {
        phi_dense_[d] = Mat(s.flux_t_[d]).transpose();
        phi_[d] = phi_dense_[d].sparseView();
        phi_[d].makeCompressed();
    }
    degree_.resize(s.moments);
    for (int k = 0; k < s.moments; ++k)
        degree_[k] = int(std::sqrt(double(s.active[k]) + .5));
}

ElementWenoDlra::Context ElementWenoDlra::context(const Mat &V) const {
    Context ctx;
    ctx.V = V;
    const int k = V.cols(), q = scheme.moments;
    ctx.c = 4 + k;
    const Mat Vt = V.transpose();
    for (int d = 0; d < scheme.dimension; ++d) {
        ctx.P[d].resize(ctx.c, q);
        ctx.P[d].topRows(4) = phi_dense_[d].topRows(4);
        ctx.P[d].bottomRows(k) = Vt * phi_[d].bottomRows(qh);
        ctx.PL[d] = ctx.P[d].leftCols(4);
    }
    for (int side = 0; side < 6; ++side) {
        if (!scheme.boundaries_[side])
            continue;
        const Mat &bf = scheme.boundary_flux_t_[side];
        ctx.B[side].resize(ctx.c, q);
        ctx.B[side].topRows(4) = bf.topRows(4);
        ctx.B[side].bottomRows(k).noalias() = Vt * bf.bottomRows(qh);
        const Mat &harm = scheme.out_rules_[side].harmonics;
        ctx.H[side].resize(harm.rows(), ctx.c);
        ctx.H[side].leftCols(4) = harm.leftCols(4);
        ctx.H[side].rightCols(k).noalias() = harm.rightCols(qh) * V;
    }
    return ctx;
}

ElementWenoDlra::Evaluation ElementWenoDlra::evaluate(double time, const Field &Z, const Context &ctx,
                                                     bool /*boundary_low*/, DlraStats *ds,
                                                     ElementWenoStats *stats) const {
    const ElementWeno &s = scheme;
    const double start = seconds();
    const int nodes = s.nodes, c = Z.cols();
    require(Z.rows() == nodes && c == ctx.c, "DLRA: factor size mismatch");
    {
        std::atomic<bool> finite{true};
        parallel_for(nodes, [&](int i) {
            if (!Z.row(i).allFinite())
                finite = false;
        });
        require(finite.load(), "Invalid element WENO state (DLRA)");
    }
    Evaluation ev;
    ev.time = time;
    ev.low.resize(nodes, 4);
    ev.low_flux = Field::Zero(nodes, 12);
    ev.closure = Field::Zero(nodes, 12);
    parallel_for(nodes, [&](int i) {
        const Four u = project(Z.row(i).transpose());
        ev.low.row(i) = u.transpose();
        const Levermore closure_state(u);
        for (int d = 0; d < s.dimension; ++d) {
            V3 n = V3::Zero();
            n[d] = 1;
            const Four fl = closure_state.flux(n);
            const Four fh = project((Z.row(i) * ctx.PL[d]).transpose());
            for (int k = 0; k < 4; ++k) {
                ev.low_flux(i, 4 * d + k) = fl[k];
                ev.closure(i, 4 * d + k) = fh[k] - fl[k];
            }
        }
    });
    ev.gamma = s.options.stabilize ? s.weno_gamma(Z) : Vec(Vec::Ones(s.cells));
    for (int d = 0; d < s.dimension; ++d)
        ev.grad[d] = s.projected_gradient(Z, d);
    ev.negsh = s.options.stabilize ? s.stabilization_residual(Z, ev.gamma, ev.grad, &ev.stab_low, stats)
                                   : Field(Field::Zero(nodes, c));
    ev.boundary_low = Field::Zero(nodes, 4);
    ev.flux_area = Field::Zero(nodes, 3);
    ev.boundary_bar = Field::Zero(nodes, 4);
    ev.boundary_area = Vec::Zero(nodes);
    ev.boundary_inflow = Vec::Zero(nodes);
    if (!s.faces_.empty()) {
        ev.bnd = face_loads(Z);
        ElementWeno::BoundaryExtras extras;
        extras.ghost_psi = [&](int, int side, int node, Vec &psi) {
            psi.noalias() = ctx.H[side] * Z.row(node).transpose();
        };
        extras.datum = &ev.datum;
        Field none(0, s.moments);
        s.add_boundary(time, Z, ev.low, ev.low_flux, none, ev.boundary_low, ev.flux_area, ev.boundary_area,
                       ev.boundary_bar, &ev.boundary_inflow, &extras);
    }
    if (ds)
        ds->t_spatial += seconds() - start;
    return ev;
}

std::array<Field, 6> ElementWenoDlra::face_loads(const Field &Z) const {
    const ElementWeno &s = scheme;
    std::array<Field, 6> out;
    const int nb = s.boundary_nodes_.size(), c = Z.cols();
    std::vector<Mat> parts(s.faces_.size());
    parallel_for(int(s.faces_.size()), [&](int index) {
        const auto &f = s.faces_[index];
        const auto &rule = s.rules_[2 * f.direction + f.side];
        const int nf = rule.local.size();
        Mat traces(nf, c);
        for (int n = 0; n < nf; ++n)
            traces.row(n) = Z.row(s.connectivity_[f.element * s.local_nodes + rule.local[n]]);
        parts[index].noalias() = rule.mass * traces;
    });
    for (int side = 0; side < 6; ++side)
        if (s.boundaries_[side])
            out[side] = Field::Zero(nb, c);
    for (std::size_t index = 0; index < parts.size(); ++index) {
        const auto &f = s.faces_[index];
        const int side = 2 * f.direction + f.side;
        const auto &rule = s.rules_[side];
        for (std::size_t n = 0; n < rule.local.size(); ++n) {
            const int node = s.connectivity_[f.element * s.local_nodes + rule.local[n]];
            out[side].row(s.boundary_slot_[node]) += parts[index].row(n);
        }
    }
    return out;
}

// ---- Fast S step: operators on the fixed bases -------------------------------------------------------------------
struct ElementWenoDlra::Fixed {
    const Field *Xh = nullptr, *MXh = nullptr;
    const Mat *Wh = nullptr;
    Field Xhb;                                  // boundary slot rows of X^
    std::array<Field, 3> GX;                    // M^-1 C_d X^
    std::array<Field, 6> bndX;                  // face loads of X^
    Mat Pneg;                                   // X^T (-s_h(X^)) with gamma^n
    std::array<Mat, 3> Pgrad;                   // (M X^)^T G_d X^
    std::array<Mat, 6> Pbnd;                    // X_b^T bndX
    std::array<Mat, 3> ALL, ALH, AHL, AHH;      // blocks of A_d (L = l <= 1, H = W^ coordinates)
    std::array<Mat, 6> BLL, BLH, BHL, BHH;      // blocks of (A_n - llf_homogeneous)^T
    std::array<Mat, 6> DL, DH;                  // datum_t blocks
    std::array<Mat, 6> HL, HH;                  // outgoing harmonics: low columns, high columns times W^
};

ElementWenoDlra::Fixed ElementWenoDlra::prepare(const Field &Xh, const Field &MXh, const Mat &Wh, const Vec &gamma,
                                                DlraStats *ds) const {
    const ElementWeno &s = scheme;
    const double start = seconds();
    Fixed f;
    f.Xh = &Xh, f.MXh = &MXh, f.Wh = &Wh;
    for (int d = 0; d < s.dimension; ++d) {
        f.GX[d] = s.projected_gradient(Xh, d);
        const Mat &P = phi_dense_[d];
        f.ALL[d] = P.topLeftCorner(4, 4);
        f.ALH[d] = P.topRightCorner(4, qh) * Wh;
        f.AHL[d] = Wh.transpose() * P.bottomLeftCorner(qh, 4);
        f.AHH[d] = Wh.transpose() * (phi_[d].bottomRightCorner(qh, qh) * Wh);
    }
    if (s.options.stabilize) {
        const Field neg = s.stabilization_residual(Xh, gamma, f.GX, nullptr, nullptr);
        f.Pneg = cross(Xh, neg);
    } else
        f.Pneg = Mat::Zero(Xh.cols(), Xh.cols());
    for (int d = 0; d < s.dimension; ++d)
        f.Pgrad[d] = cross(MXh, f.GX[d]);
    if (!s.faces_.empty()) {
        f.bndX = face_loads(Xh);
        f.Xhb.resize(s.boundary_nodes_.size(), Xh.cols());
        for (std::size_t j = 0; j < s.boundary_nodes_.size(); ++j)
            f.Xhb.row(j) = Xh.row(s.boundary_nodes_[j]);
        for (int side = 0; side < 6; ++side) {
            if (!s.boundaries_[side])
                continue;
            f.Pbnd[side] = f.Xhb.transpose() * f.bndX[side];
            const Mat &B = s.boundary_flux_t_[side];
            f.BLL[side] = B.topLeftCorner(4, 4);
            f.BLH[side] = B.topRightCorner(4, qh) * Wh;
            f.BHL[side] = Wh.transpose() * B.bottomLeftCorner(qh, 4);
            f.BHH[side] = Wh.transpose() * B.bottomRightCorner(qh, qh) * Wh;
            const Mat &D = s.datum_t_[side];
            f.DL[side] = D.leftCols(4);
            f.DH[side] = D.rightCols(qh) * Wh;
            const Mat &harm = s.out_rules_[side].harmonics;
            f.HL[side] = harm.leftCols(4);
            f.HH[side] = harm.rightCols(qh) * Wh;
        }
    }
    if (ds)
        ds->t_spatial += seconds() - start;
    return f;
}

void ElementWenoDlra::fast_stage(double time, const Fixed &f, const Field &low, const Mat &S, Field &low_out,
                                 Mat &Sdot, ElementWenoStats *stats, DlraStats *ds) const {
    const ElementWeno &s = scheme;
    const int nodes = s.nodes, dim = s.dimension;
    const Field &Xh = *f.Xh;
    double clock = seconds();
    auto lap = [&](double DlraStats::*slot) {
        const double now = seconds();
        if (ds)
            ds->*slot += now - clock;
        clock = now;
    };
    {
        std::atomic<bool> finite{true};
        parallel_for(nodes, [&](int i) {
            if (!low.row(i).allFinite())
                finite = false;
        });
        require(finite.load() && S.allFinite(), "Invalid element WENO state (DLRA)");
    }
    // l <= 1 rows of a_H A_d at the nodes: X^ (S AHL_d); and of G_d(a_H): GX_d (S AHL_d).
    Mat CL(Xh.cols(), 4 * dim);
    for (int d = 0; d < dim; ++d)
        CL.middleCols(4 * d, 4) = S * f.AHL[d];
    const Field YL = mul(Xh, CL);
    std::array<Field, 3> GY;
    for (int d = 0; d < dim; ++d)
        GY[d] = mul(f.GX[d], Mat(CL.middleCols(4 * d, 4)));
    lap(&DlraStats::t_products);
    Evaluation ev;
    ev.time = time;
    ev.low.resize(nodes, 4);
    ev.low_flux = Field::Zero(nodes, 12);
    ev.closure = Field::Zero(nodes, 12);
    parallel_for(nodes, [&](int i) {
        const Four u = project(low.row(i).transpose());
        ev.low.row(i) = u.transpose();
        const Levermore closure_state(u);
        for (int d = 0; d < dim; ++d) {
            V3 n = V3::Zero();
            n[d] = 1;
            const Four fl = closure_state.flux(n);
            const Four fh = project((low.row(i) * f.ALL[d] + YL.row(i).segment(4 * d, 4)).transpose());
            for (int k = 0; k < 4; ++k) {
                ev.low_flux(i, 4 * d + k) = fl[k];
                ev.closure(i, 4 * d + k) = fh[k] - fl[k];
            }
        }
    });
    ev.gamma = s.options.stabilize ? s.weno_gamma(low) : Vec(Vec::Ones(s.cells));
    std::array<Field, 3> gL;
    for (int d = 0; d < dim; ++d)
        gL[d] = s.projected_gradient(low, d);
    Field R = s.options.stabilize ? s.stabilization_residual(low, ev.gamma, gL, &ev.stab_low, stats)
                                  : Field(Field::Zero(nodes, 4));
    ev.boundary_low = Field::Zero(nodes, 4);
    ev.flux_area = Field::Zero(nodes, 3);
    ev.boundary_bar = Field::Zero(nodes, 4);
    ev.boundary_area = Vec::Zero(nodes);
    ev.boundary_inflow = Vec::Zero(nodes);
    Field pre = Field::Zero(0, 4);
    std::array<Field, 6> bndL;
    std::array<Mat, 6> HS;
    if (!s.faces_.empty()) {
        bndL = face_loads(low);
        for (int side = 0; side < 6; ++side)
            if (s.boundaries_[side])
                HS[side] = f.HH[side] * S.transpose();
        ElementWeno::BoundaryExtras extras;
        extras.ghost_psi = [&](int, int side, int node, Vec &psi) {
            psi.noalias() = f.HL[side] * low.row(node).transpose();
            psi.noalias() += HS[side] * Xh.row(node).transpose();
        };
        extras.datum = &ev.datum;
        Field none(0, s.moments);
        s.add_boundary(time, low, ev.low, ev.low_flux, none, ev.boundary_low, ev.flux_area, ev.boundary_area,
                       ev.boundary_bar, &ev.boundary_inflow, &extras);
        pre = Field::Zero(s.boundary_nodes_.size(), 4);
        for (int side = 0; side < 6; ++side) {
            if (!s.boundaries_[side])
                continue;
            pre.noalias() += bndL[side] * f.BLL[side];
            pre.noalias() += f.bndX[side] * Mat(S * f.BHL[side]);
            pre.noalias() -= ev.datum[side] * f.DL[side];
        }
        for (std::size_t j = 0; j < s.boundary_nodes_.size(); ++j)
            R.row(s.boundary_nodes_[j]) += pre.row(j);
    }
    Field F = s.solve_mass(std::move(R));
    parallel_for(nodes, [&](int i) {
        for (int d = 0; d < dim; ++d)
            F.row(i) -= gL[d].row(i) * f.ALL[d] + GY[d].row(i);
    });
    lap(&DlraStats::t_spatial);
    Field moment_rate(nodes, 4);
    parallel_for(nodes, [&](int i) { moment_rate.row(i) = project(F.row(i).transpose()).transpose(); });
    const ElementWeno::LowOrderInputs inputs{&ev.low,          &ev.low_flux,      &ev.closure,       &moment_rate,
                                             &ev.stab_low,     &pre,              &ev.boundary_low,  &ev.flux_area,
                                             &ev.boundary_bar, &ev.boundary_area, &ev.boundary_inflow, &ev.gamma};
    const Field load = s.limited_low_load(inputs, stats, nullptr);
    low_out.resize(nodes, 4);
    parallel_for(nodes, [&](int i) {
        set_low(Eigen::Map<Vec>(&low_out.coeffRef(i, 0), 4), Four(load.row(i).transpose()) / s.lumped[i]);
    });
    lap(&DlraStats::t_limiter);
    // Galerkin rate of the l >= 2 block: X^T R(a) W^ with a = [low | X^ S W^T].
    Sdot = f.Pneg * S;
    if (!s.faces_.empty())
        for (int side = 0; side < 6; ++side) {
            if (!s.boundaries_[side])
                continue;
            Sdot.noalias() += (f.Xhb.transpose() * bndL[side]) * f.BLH[side];
            Sdot.noalias() += f.Pbnd[side] * S * f.BHH[side];
            Sdot.noalias() -= (f.Xhb.transpose() * ev.datum[side]) * f.DH[side];
        }
    for (int d = 0; d < dim; ++d) {
        Sdot.noalias() -= cross(*f.MXh, gL[d]) * f.ALH[d];
        Sdot.noalias() -= f.Pgrad[d] * S * f.AHH[d];
    }
    lap(&DlraStats::t_products);
}

Field ElementWenoDlra::full_rate(const Evaluation &ev, const Context &ctx, const Mat *U, Field *pre,
                                 DlraStats *ds) const {
    const ElementWeno &s = scheme;
    const double start = seconds();
    const int k = ctx.V.cols(), t = U ? U->cols() : 4;
    Mat Cs = Mat::Zero(ctx.c, t);
    if (U)
        Cs.bottomRows(k).noalias() = ctx.V.transpose() * *U;
    else
        Cs.topRows(4).setIdentity();
    Field R = s.options.stabilize ? mul(ev.negsh, Cs) : Field(Field::Zero(s.nodes, t));
    if (!s.faces_.empty()) {
        Field rows = Field::Zero(s.boundary_nodes_.size(), t);
        for (int side = 0; side < 6; ++side) {
            if (!s.boundaries_[side])
                continue;
            const Mat Cb = U ? Mat(ctx.B[side].rightCols(qh) * *U) : Mat(ctx.B[side].leftCols(4));
            const Mat &dt = s.datum_t_[side];
            const Mat Cd = U ? Mat(dt.rightCols(qh) * *U) : Mat(dt.leftCols(4));
            rows.noalias() += ev.bnd[side] * Cb;
            rows.noalias() -= ev.datum[side] * Cd;
        }
        for (std::size_t j = 0; j < s.boundary_nodes_.size(); ++j)
            R.row(s.boundary_nodes_[j]) += rows.row(j);
        if (pre)
            *pre = std::move(rows);
    }
    Field F = s.solve_mass(std::move(R));
    for (int d = 0; d < s.dimension; ++d) {
        const Mat Cg = U ? Mat(ctx.P[d].rightCols(qh) * *U) : Mat(ctx.P[d].leftCols(4));
        F.noalias() -= mul(ev.grad[d], Cg);
    }
    if (ds)
        ds->t_products += seconds() - start;
    return F;
}

Mat ElementWenoDlra::projected_rate(const Evaluation &ev, const Context &ctx, const Field &X, const Field &MX,
                                   const Mat *U, DlraStats *ds) const {
    const ElementWeno &s = scheme;
    const double start = seconds();
    const int k = ctx.V.cols();
    auto right = [&](const Mat &m) { return U ? Mat(m.rightCols(qh) * *U) : Mat(m.rightCols(qh)); };
    Mat Cs = Mat::Zero(ctx.c, U ? U->cols() : qh);
    Cs.bottomRows(k) = U ? Mat(ctx.V.transpose() * *U) : Mat(ctx.V.transpose());
    Mat out = s.options.stabilize ? Mat(cross(X, ev.negsh) * Cs) : Mat(Mat::Zero(X.cols(), Cs.cols()));
    if (!s.faces_.empty()) {
        Field Xb(s.boundary_nodes_.size(), X.cols());
        for (std::size_t j = 0; j < s.boundary_nodes_.size(); ++j)
            Xb.row(j) = X.row(s.boundary_nodes_[j]);
        for (int side = 0; side < 6; ++side) {
            if (!s.boundaries_[side])
                continue;
            out.noalias() += (Xb.transpose() * ev.bnd[side]) * right(ctx.B[side]);
            out.noalias() -= (Xb.transpose() * ev.datum[side]) * right(s.datum_t_[side]);
        }
    }
    for (int d = 0; d < s.dimension; ++d)
        out.noalias() -= cross(MX, ev.grad[d]) * right(ctx.P[d]);
    if (ds)
        ds->t_products += seconds() - start;
    return out;
}

Field ElementWenoDlra::low_rate(const Evaluation &ev, const Context &ctx, ElementWenoStats *stats,
                                DlraStats *ds) const {
    const ElementWeno &s = scheme;
    Field pre;
    const Field F = full_rate(ev, ctx, nullptr, &pre, ds);
    if (pre.rows() == 0)
        pre = Field::Zero(0, 4);
    const double start = seconds();
    Field moment_rate(s.nodes, 4);
    parallel_for(s.nodes, [&](int i) { moment_rate.row(i) = project(F.row(i).transpose()).transpose(); });
    const ElementWeno::LowOrderInputs inputs{&ev.low,          &ev.low_flux,     &ev.closure,       &moment_rate,
                                             &ev.stab_low,     &pre,             &ev.boundary_low,  &ev.flux_area,
                                             &ev.boundary_bar, &ev.boundary_area, &ev.boundary_inflow, &ev.gamma};
    const Field load = s.limited_low_load(inputs, stats, nullptr);
    Field out(s.nodes, 4);
    parallel_for(s.nodes, [&](int i) {
        set_low(Eigen::Map<Vec>(&out.coeffRef(i, 0), 4), Four(load.row(i).transpose()) / s.lumped[i]);
    });
    if (ds)
        ds->t_limiter += seconds() - start;
    return out;
}

void ElementWenoDlra::factored_rate(double time, const Field &low, const Field &Y, const Mat &V, const Mat &U,
                                    Field &low_out, Field &high, ElementWenoStats *stats) const {
    const Context ctx = context(V);
    const Evaluation ev = evaluate(time, hcat(low, Y), ctx, true, nullptr, stats);
    low_out = low_rate(ev, ctx, stats, nullptr);
    high = full_rate(ev, ctx, &U, nullptr, nullptr);
}

Mat ElementWenoDlra::reorthonormalize(Field &X) const {
    const Field MX = scheme.apply_mass(X);
    Mat G = cross(X, MX);
    G = .5 * (G + G.transpose()).eval();
    Eigen::LLT<Mat> llt(G);
    require(llt.info() == Eigen::Success, "DLRA: M-Gram matrix is not positive definite");
    const Mat Lt = llt.matrixU(); // G = Lt^T Lt
    X = mul(X, Mat(Lt.triangularView<Eigen::Upper>().solve(Mat::Identity(G.rows(), G.cols()))));
    return Lt;
}

Field ElementWenoDlra::augment(const Field &X, const Field &MX, const Field &K) const {
    if (K.cols() == 0)
        return X;
    Field Kt = K;
    for (int pass = 0; pass < 2 && X.cols() > 0; ++pass)
        Kt.noalias() -= mul(X, cross(MX, Kt));
    Field MK = scheme.apply_mass(Kt);
    Mat G = cross(Kt, MK);
    G = .5 * (G + G.transpose()).eval();
    Eigen::SelfAdjointEigenSolver<Mat> eig(G);
    const Vec lambda = eig.eigenvalues();
    const double top = lambda.size() ? lambda.maxCoeff() : 0.;
    std::vector<int> keep;
    for (int j = int(lambda.size()) - 1; j >= 0; --j)
        if (lambda[j] > 1e-20 * top && lambda[j] > 0)
            keep.push_back(j);
    if (keep.empty())
        return X;
    Mat T(G.rows(), keep.size());
    for (std::size_t j = 0; j < keep.size(); ++j)
        T.col(j) = eig.eigenvectors().col(keep[j]) / std::sqrt(lambda[keep[j]]);
    Field Q = mul(Kt, T);
    if (X.cols() > 0)
        Q.noalias() -= mul(X, cross(MX, Q));
    reorthonormalize(Q);
    return X.cols() > 0 ? hcat(X, Q) : Q;
}

DlraState ElementWenoDlra::compress(const Field &full) const {
    const ElementWeno &s = scheme;
    require(full.rows() == s.nodes && full.cols() == s.moments, "DLRA: full state size mismatch");
    DlraState st;
    st.low = full.leftCols(4);
    const Field A = full.rightCols(qh);
    const Mat G0 = cross(A, s.apply_mass(A));
    const Mat G = .5 * (G0 + G0.transpose());
    Eigen::SelfAdjointEigenSolver<Mat> eig(G);
    const Vec lambda = eig.eigenvalues().reverse().cwiseMax(0.);
    const Mat U = eig.eigenvectors().rowwise().reverse();
    double low_norm2 = 0;
    {
        const Mat Gl = cross(st.low, s.apply_mass(st.low));
        low_norm2 = Gl.trace();
    }
    const double total = std::sqrt(std::max(0., low_norm2) + lambda.sum());
    const double theta = options.tolerance * (options.relative_to_high ? std::sqrt(lambda.sum()) : total) +
                         options.absolute_tolerance;
    const int cap = std::min(qh, s.nodes);
    int r = 0;
    {
        double tail = lambda.sum();
        while (r < cap && std::sqrt(std::max(0., tail)) > theta)
            tail -= lambda[r++];
    }
    int real = 0;
    while (real < r && lambda[real] > 1e-28 * std::max(lambda[0], 1e-300))
        ++real;
    int target = options.fixed_rank ? options.initial_rank : std::max(r, options.initial_rank);
    if (options.max_rank > 0)
        target = std::min(target, options.max_rank);
    target = std::max(1, std::min(target, cap));
    real = std::min(real, target);
    // Real directions: X = A U D^-1/2, S = D^1/2, then re-orthonormalized (rounding).
    Mat W = U.leftCols(real);
    Field X = mul(A, W);
    Mat S = Mat::Zero(real, real);
    if (real > 0) {
        Vec inv(real);
        for (int j = 0; j < real; ++j)
            inv[j] = 1 / std::sqrt(lambda[j]), S(j, j) = std::sqrt(lambda[j]);
        X = mul(X, Mat(inv.asDiagonal()));
        const Mat Lt = reorthonormalize(X);
        S = Lt * S;
    }
    // Padding with orthonormal completions (zero coefficients).
    if (target > real) {
        std::mt19937 rng(20261112);
        std::normal_distribution<double> normal;
        Field R(s.nodes, target - real);
        for (int i = 0; i < R.rows(); ++i)
            for (int j = 0; j < R.cols(); ++j)
                R(i, j) = normal(rng);
        const Field MX = s.apply_mass(X);
        X = augment(X, MX, R);
        require(X.cols() == target, "DLRA: padding of the spatial basis failed");
        Mat Wr(qh, target);
        Wr.leftCols(real) = W;
        for (int i = 0; i < qh; ++i)
            for (int j = real; j < target; ++j)
                Wr(i, j) = normal(rng);
        Eigen::HouseholderQR<Mat> qr(Wr);
        Mat Q = qr.householderQ() * Mat::Identity(qh, target);
        // Keep the real directions exactly: Q's first `real` columns span W; use W itself and the rest of Q.
        W.resize(qh, target);
        W.leftCols(real) = U.leftCols(real);
        W.rightCols(target - real) = Q.rightCols(target - real);
        Mat Sp = Mat::Zero(target, target);
        Sp.topLeftCorner(real, real) = S;
        S = Sp;
    }
    st.X = std::move(X);
    st.S = std::move(S);
    st.W = std::move(W);
    return st;
}

Field ElementWenoDlra::expand(const DlraState &st) const {
    Field out(scheme.nodes, scheme.moments);
    const Field high = mul(st.X, Mat(st.S * st.W.transpose()));
    parallel_for(scheme.nodes, [&](int i) {
        out.row(i).head(4) = st.low.row(i);
        out.row(i).tail(qh) = high.row(i);
    });
    return out;
}

double ElementWenoDlra::norm(const DlraState &st) const {
    const double low = cross(st.low, scheme.apply_mass(st.low)).trace();
    return std::sqrt(std::max(0., low) + st.S.squaredNorm());
}

Field ElementWenoDlra::react_low(const Field &a, double tau, double source_time, ElementWenoStats *stats) const {
    const ElementWeno &s = scheme;
    const ElementWenoOptions &o = s.options;
    const int nodes = s.nodes;
    const bool fcs = bool(o.uncollided_density);
    require(!fcs || std::isfinite(source_time), "FCS: react() needs the source time");
    const Vec fcs_source = fcs ? s.collided_source(source_time, nullptr, stats) : Vec();
    const double c0 = 1 / std::sqrt(4 * pi);
    auto filter = [&](int l) { return o.filter_strength * std::pow(double(l) / (s.pn.order + 1), 4); };
    if (!o.material) {
        const bool any = o.filter_strength != 0 || o.sigma_s != 0 || o.sigma_a != 0;
        Field out = a;
        if (any)
            for (int k = 0; k < 4; ++k) {
                const int l = degree_[k];
                const double decay = std::exp(-tau * (filter(l) + o.sigma_a + (l > 0 ? o.sigma_s : 0.)));
                out.col(k) *= decay;
            }
        if (fcs) {
            const double z = -o.sigma_a * tau, decay = std::exp(z), phi1 = z == 0 ? 1. : std::expm1(z) / z;
            for (int i = 0; i < nodes; ++i) {
                const double q0 = fcs_source[i], rho = a(i, 0) / c0;
                out(i, 0) += c0 * q0 * tau * phi1;
                if (stats) {
                    stats->absorbed += s.lumped[i] * ((1 - decay) * rho + q0 * tau * (1 - phi1));
                    stats->injected += s.lumped[i] * q0 * tau;
                }
            }
            return s.admissible_rows_public(std::move(out), stats);
        }
        if (stats && o.sigma_a > 0) {
            const double loss = -std::expm1(-o.sigma_a * tau);
            for (int i = 0; i < nodes; ++i)
                stats->absorbed += s.lumped[i] * loss * std::sqrt(4 * pi) * a(i, 0);
        }
        return any ? s.admissible_rows_public(std::move(out), stats) : out;
    }
    Field out(nodes, 4);
    std::vector<double> absorbed(nodes), injected(nodes);
    parallel_for(nodes, [&](int i) {
        const double sa = s.node_sigma_a[i], ss = s.node_sigma_s[i], q0 = s.node_source[i] + (fcs ? fcs_source[i] : 0.);
        const double z = -sa * tau, decay0 = std::exp(z), phi1 = z == 0 ? 1. : std::expm1(z) / z;
        const double rho = a(i, 0) / c0;
        out(i, 0) = decay0 * a(i, 0) + c0 * q0 * tau * phi1;
        for (int k = 1; k < 4; ++k)
            out(i, k) = std::exp(-tau * (sa + ss + filter(degree_[k]))) * a(i, k);
        absorbed[i] = s.lumped[i] * (-std::expm1(z) * rho + q0 * tau * (1 - phi1));
        injected[i] = s.lumped[i] * q0 * tau;
    });
    if (stats)
        for (int i = 0; i < nodes; ++i)
            stats->absorbed += absorbed[i], stats->injected += injected[i];
    return s.admissible_rows_public(std::move(out), stats);
}

DlraState ElementWenoDlra::react(const DlraState &in, double tau, double source_time, ElementWenoStats *stats) const {
    const ElementWeno &s = scheme;
    const ElementWenoOptions &o = s.options;
    DlraState st = in;
    st.low = react_low(in.low, tau, source_time, stats);
    // l >= 2: a_lm *= exp(-tau (sigma_a(x) + sigma_s(x))) exp(-tau f (l/(N+1))^4), separable in (x, angle).
    if (!o.material) {
        const double decay = std::exp(-tau * (o.sigma_a + o.sigma_s));
        st.S *= decay;
    } else {
        Vec e(s.nodes);
        for (int i = 0; i < s.nodes; ++i)
            e[i] = std::exp(-tau * (s.node_sigma_a[i] + s.node_sigma_s[i]));
        parallel_for(s.nodes, [&](int i) { st.X.row(i) *= e[i]; });
        const Mat Lt = reorthonormalize(st.X);
        st.S = Lt * st.S;
    }
    if (o.filter_strength != 0) {
        Mat W = st.W;
        for (int k = 0; k < qh; ++k)
            W.row(k) *= std::exp(-tau * o.filter_strength * std::pow(double(degree_[4 + k]) / (s.pn.order + 1), 4));
        Eigen::HouseholderQR<Mat> qr(W);
        const int r = W.cols();
        st.W = qr.householderQ() * Mat::Identity(qh, r);
        const Mat R = qr.matrixQR().topRows(r).triangularView<Eigen::Upper>();
        st.S = st.S * R.transpose();
    }
    return st;
}

DlraState ElementWenoDlra::advance(double time, double dt, const DlraState &input, ElementWenoStats *stats,
                                  DlraStats *ds) const {
    const ElementWeno &s = scheme;
    double clock = seconds();
    auto lap = [&](double DlraStats::*slot) {
        const double now = seconds();
        if (ds)
            ds->*slot += now - clock;
        clock = now;
    };
    const bool reacts = s.reacts();
    DlraState st = reacts ? react(input, .5 * dt, time + .25 * dt, stats) : input;
    lap(&DlraStats::t_reaction);
    const int r = st.rank();
    // ---- K and L step: one evaluation at a^n --------------------------------------------------------------------
    Field K1, MX;
    Mat L1;
    Vec gamma_n;
    {
        const Field Y0 = mul(st.X, st.S);
        const Context ctx = context(st.W);
        const Evaluation ev = evaluate(time, hcat(st.low, Y0), ctx, false, ds, nullptr);
        gamma_n = ev.gamma;
        K1 = full_rate(ev, ctx, &st.W, nullptr, ds);
        parallel_for(int(K1.rows()), [&](int i) { K1.row(i) = Y0.row(i) + dt * K1.row(i); });
        MX = s.apply_mass(st.X);
        const Mat Ldot = projected_rate(ev, ctx, st.X, MX, nullptr, ds); // r x qh
        L1 = st.W * st.S.transpose() + dt * Ldot.transpose();
    }
    lap(&DlraStats::t_kl);
    // ---- Augmented bases ------------------------------------------------------------------------------------------
    const Field Xh = augment(st.X, MX, K1);
    Mat Wh;
    {
        Mat A(qh, 2 * r);
        A << st.W, L1;
        Eigen::HouseholderQR<Mat> qr(A);
        const int rw = std::min(2 * r, qh);
        Wh = qr.householderQ() * Mat::Identity(qh, rw);
        Wh.leftCols(r) = st.W; // exact old basis (the Householder Q spans it up to signs and rounding)
        // Re-orthogonalize the new columns against the exact old ones.
        if (rw > r) {
            Mat N = Wh.rightCols(rw - r);
            N -= st.W * (st.W.transpose() * N);
            Eigen::HouseholderQR<Mat> q2(N);
            Wh.rightCols(rw - r) = q2.householderQ() * Mat::Identity(qh, rw - r);
            Wh.rightCols(rw - r) -= st.W * (st.W.transpose() * Wh.rightCols(rw - r));
        }
    }
    const Field MXh = s.apply_mass(Xh);
    const Mat Sh0 = cross(MXh, st.X) * st.S * (st.W.transpose() * Wh);
    lap(&DlraStats::t_basis);
    // ---- Galerkin S step coupled to the limited M1 update (the scheme's SSP stages) ------------------------------
    const bool fast = options.frozen_gamma;
    const Context ch = fast ? Context() : context(Wh);
    const Fixed fixed = fast ? prepare(Xh, MXh, Wh, gamma_n, ds) : Fixed();
    lap(&DlraStats::t_basis);
    struct Stage {
        Field low;
        Mat S;
    };
    const Stage a{st.low, Sh0};
    auto stage = [&](double t, const Stage &x, double old, double fresh, double tau, double weight) {
        Field lr;
        Mat Sdot;
        if (fast)
            fast_stage(t, fixed, x.low, x.S, lr, Sdot, stats, ds);
        else {
            const Evaluation ev = evaluate(t, hcat(x.low, mul(Xh, x.S)), ch, true, ds, stats);
            lr = low_rate(ev, ch, stats, ds);
            Sdot = projected_rate(ev, ch, Xh, MXh, &Wh, ds);
        }
        if (stats)
            stats->boundary_outflow -= dt * weight * stats->last_boundary_mass_rate;
        combine(lr, x.low, a.low, old, fresh, tau);
        Stage out;
        out.low = s.admissible_rows_public(std::move(lr), stats);
        out.S = old == 0 ? Mat(fresh * (x.S + tau * Sdot)) : Mat(old * a.S + fresh * (x.S + tau * Sdot));
        return out;
    };
    Stage last;
    if (!s.options.ssprk43) {
        const Stage first = stage(time, a, 0, 1, dt, 1. / 6);
        const Stage second = stage(time + dt, first, .75, .25, dt, 1. / 6);
        last = stage(time + .5 * dt, second, 1. / 3, 2. / 3, dt, 2. / 3);
    } else {
        const double half = .5 * dt;
        const Stage first = stage(time, a, 0, 1, half, 1. / 6);
        const Stage second = stage(time + half, first, 0, 1, half, 1. / 6);
        const Stage third = stage(time + dt, second, 2. / 3, 1. / 3, half, 1. / 6);
        last = stage(time + half, third, 0, 1, half, .5);
    }
    lap(&DlraStats::t_galerkin);
    // ---- Truncation -----------------------------------------------------------------------------------------------
    DlraState out;
    out.low = std::move(last.low);
    {
        Eigen::JacobiSVD<Mat> svd(last.S, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Vec sigma = svd.singularValues();
        const int cap = sigma.size();
        double norm = sigma.norm();
        if (!options.relative_to_high) {
            const double low = cross(out.low, s.apply_mass(out.low)).trace();
            norm = std::sqrt(std::max(0., low) + norm * norm);
        }
        const double theta = options.tolerance * norm * (options.per_unit_time ? dt : 1.) + options.absolute_tolerance;
        int r1 = 0;
        double tail2 = sigma.squaredNorm();
        while (r1 < cap && std::sqrt(std::max(0., tail2)) > theta)
            tail2 -= sigma[r1] * sigma[r1], ++r1;
        if (options.fixed_rank)
            r1 = options.initial_rank;
        r1 = std::max(r1, options.min_rank);
        if (options.max_rank > 0)
            r1 = std::min(r1, options.max_rank);
        r1 = std::max(1, std::min(r1, cap));
        double discarded = 0;
        for (int j = r1; j < cap; ++j)
            discarded += sigma[j] * sigma[j];
        discarded = std::sqrt(discarded);
        out.X = mul(Xh, Mat(svd.matrixU().leftCols(r1)));
        out.W = Wh * svd.matrixV().leftCols(r1);
        out.S = sigma.head(r1).asDiagonal();
        if (ds) {
            ds->steps += 1;
            ds->last_rank = r1;
            ds->min_rank = std::min(ds->min_rank, r1);
            ds->max_rank = std::max(ds->max_rank, r1);
            ds->rank_sum += r1;
            ds->discarded += discarded;
            if (norm > 0)
                ds->max_relative_discard = std::max(ds->max_relative_discard, discarded / norm);
        }
    }
    lap(&DlraStats::t_truncate);
    if (reacts)
        out = react(out, .5 * dt, time + .75 * dt, stats);
    lap(&DlraStats::t_reaction);
    return out;
}
} // namespace m1pn
