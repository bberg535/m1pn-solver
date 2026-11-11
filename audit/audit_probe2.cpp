// Probe 2: boundaries actually built inside ElementWeno for the T3 state (uniform psi = (1-wx)^3).
#include "audit_common.hpp"

int main() {
    Angular pn(3);
    auto psi = [](const V3 &om) { return std::pow(1 - om[0], 3); };
    Vec c = Vec::Zero(pn.q);
    for (const auto &p : sphere_rule(pn.order + 8)) c += p.weight * psi(p.omega) * harmonics(pn.order, p.omega);
    ElementWenoOptions o; o.degree = 2; o.dimension = 2; o.elements = 4; o.periodic = false;
    Inflow in; in.density = [&](const V3 &, double, const V3 &om) { return psi(om); };
    ElementWeno s(pn, o, in);
    for (int k = 0; k < 4; ++k) {
        Incoming d = s.boundaries_[k]->data(in, V3(.5, .5, 0), 0);
        Vec ext = s.boundaries_[k]->exterior(c, d.d);
        const Mat A = pn.normal(s.boundaries_[k]->n);
        J("PROBE2_T3_faces").add("face", k).add("n_x", s.boundaries_[k]->n[0]).add("n_y", s.boundaries_[k]->n[1])
            .add("ext_minus_u", (ext - c).norm()).add("u", c.norm())
            .add("integrand_norm", (.5 * (A - Mat::Identity(pn.q, pn.q)) * (c - ext)).norm()).done();
    }
    Field a = s.interpolate([&](const V3 &) { return c; });
    Field low(s.nodes, 4), lf(s.nodes, 12);
    lf.setZero();
    for (int i = 0; i < s.nodes; ++i) {
        Four u = project(a.row(i).transpose());
        low.row(i) = u.transpose();
        Levermore lv(u);
        for (int d = 0; d < 2; ++d) { V3 nn = V3::Zero(); nn[d] = 1; Four f = lv.flux(nn); for (int q = 0; q < 4; ++q) lf(i, 4 * d + q) = f[q]; }
    }
    Field bh = Field::Zero(s.nodes, pn.q), bl = Field::Zero(s.nodes, 4), fa = Field::Zero(s.nodes, 3), bb = Field::Zero(s.nodes, 4);
    Vec area = Vec::Zero(s.nodes);
    s.add_boundary(0, a, low, lf, bh, bl, fa, area, bb);
    J("PROBE2_T3_boundary_high").add("max_abs_bh", bh.cwiseAbs().maxCoeff()).add("max_abs_bl", bl.cwiseAbs().maxCoeff()).done();
    // volume residual of the uniform field
    ElementWenoStats st; ElementWenoDebug dbg;
    Field r = s.rate(0, a, &st, &dbg);
    Field zero_bnd_rate = Field::Zero(s.nodes, pn.q);
    J("PROBE2_T3_rate").add("max_abs_rate", r.cwiseAbs().maxCoeff()).add("max_abs_ho_moment_rate", dbg.ho_moment_rate.cwiseAbs().maxCoeff())
        .add("max_abs_ho_rate_rows_ge4", 0.0).done();
    return 0;
}
