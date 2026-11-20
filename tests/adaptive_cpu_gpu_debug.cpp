// Diagnostic: CPU vs GPU adaptive PN operator by operator on a state with mixed tile orders.
// Compares (1) the state after the initial adaptation, (2) the gradients M^-1 C_d a, (3) the stage mass solve,
// (4) one full step. Only active entries (c < tile_q of the node) are compared. For the worst entry the distance to
// the nearest node of lower order is printed (order interfaces).
#include "element_weno.hpp"
#include "element_weno_gpu.hpp"
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

using namespace m1pn;
namespace {
struct Report {
    double max_diff = 0, scale = 0;
    int node = -1, comp = -1;
};

Report compare_active(const ElementWeno &s, const AdaptivePnPlan &P, const Field &cpu, const Field &gpu) {
    Report r;
    for (int i = 0; i < s.nodes; ++i) {
        const int qi = P.tile_q[P.node_tile(i)];
        for (int c = 0; c < qi; ++c) {
            r.scale = std::max(r.scale, std::abs(cpu(i, c)));
            const double d = std::abs(cpu(i, c) - gpu(i, c));
            if (d > r.max_diff)
                r.max_diff = d, r.node = i, r.comp = c;
        }
    }
    return r;
}

// Manhattan distance (in nodes) from node i to the nearest node whose tile has a smaller tile_q (2D, tensor mesh).
int distance_to_lower(const ElementWeno &s, const AdaptivePnPlan &P, int i) {
    const int n0 = P.npa[0], n1 = P.npa[1];
    const int ix = i % n0, iy = i / n0, qi = P.tile_q[P.node_tile(i)];
    int best = 1 << 20;
    for (int y = 0; y < n1; ++y)
        for (int x = 0; x < n0; ++x)
            if (P.tile_q[P.node_tile(y * n0 + x)] < qi)
                best = std::min(best, std::abs(x - ix) + std::abs(y - iy));
    (void)s;
    return best;
}

void print(const char *what, const ElementWeno &s, const AdaptivePnPlan &P, const Report &r) {
    const double rel = r.max_diff / std::max(r.scale, 1e-300);
    std::printf("%-34s rel %.2e  abs %.2e", what, rel, r.max_diff);
    if (r.node >= 0) {
        const int qi = P.tile_q[P.node_tile(r.node)];
        std::printf("  worst node %d (ix %d, iy %d) moment %d, tile_q %d, dist to lower order %d", r.node,
                    r.node % P.npa[0], r.node / P.npa[0], r.comp, qi, distance_to_lower(s, P, r.node));
    }
    std::printf("\n");
}

// Largest |entry| at inactive positions (c >= tile_q).
double max_inactive(const ElementWeno &s, const AdaptivePnPlan &P, const Field &a) {
    double m = 0;
    for (int i = 0; i < s.nodes; ++i)
        for (int c = P.tile_q[P.node_tile(i)]; c < s.moments; ++c)
            m = std::max(m, std::abs(a(i, c)));
    return m;
}
} // namespace

int main(int argc, char **argv) {
    const int order = argc > 1 ? std::atoi(argv[1]) : 7;
    const int elements = argc > 2 ? std::atoi(argv[2]) : 32;
    const int halo = argc > 3 ? std::atoi(argv[3]) : 1;
    const bool weno = argc > 4 ? std::atoi(argv[4]) != 0 : true;
    const bool limit = argc > 5 ? std::atoi(argv[5]) != 0 : true;
    const bool material = argc > 6 ? std::atoi(argv[6]) != 0 : false;
    Angular pn(order);
    ElementWenoOptions o;
    o.dimension = 2;
    o.degree = 1;
    o.elements = elements;
    o.periodic = false;
    o.ssprk43 = true;
    o.reduce_angular = true;
    o.weno = weno;
    o.limit = limit;
    o.adaptive_pn = true;
    o.adaptive_tol = 3e-4;
    o.adaptive_tile = 4;
    o.adaptive_stride = 4;
    o.adaptive_min_order = 1;
    o.adaptive_halo = halo;
    if (material) { // lattice-like: absorber block, scattering elsewhere, source in a corner
        auto mat = std::make_shared<ElementWenoMaterial>();
        mat->sigma_a = [](const V3 &x) { return (x[0] > 0.6 && x[1] > 0.6) ? 5.0 : 0.0; };
        mat->sigma_s = [](const V3 &x) { return (x[0] > 0.6 && x[1] > 0.6) ? 0.0 : 1.0; };
        mat->source = [](const V3 &x) { return (x[0] < 0.3 && x[1] < 0.3) ? 2.0 : 0.0; };
        o.material = mat;
    }
    ElementWeno s(pn, o);
    std::printf("material %d, reacts %d\n", int(material), int(s.reacts_public()));

    // Localized anisotropic bump: high orders in the centre, order 1 outside -> mixed tile orders.
    auto ic = [&](const V3 &x) {
        Vec v = Vec::Zero(pn.q);
        const double dx = x[0] - 0.5, dy = x[1] - 0.5;
        const double bump = std::exp(-180.0 * (dx * dx + dy * dy));
        set_low(v, Four(1e-3 + bump, 0.3 * bump, 0.1 * bump, 0.0));
        v[index(2, 0)] = 0.2 * bump;
        v[index(3, 1)] = 0.05 * bump;
        v[index(4, 0)] = 0.04 * bump;
        return v;
    };
    const Field a0 = s.interpolate(ic);

    // (1) initial adaptation
    Field a = a0;
    s.adapt_pn_state(a, true, nullptr);
    const AdaptivePnPlan &P = *s.adaptive_plan();
    std::vector<int> count(pn.order + 2, 0);
    for (int T = 0; T < P.num_tiles; ++T)
        ++count[P.tile_order[T]];
    std::printf("P%d, %d elements, %d nodes, %d tiles; tile orders:", order, elements, s.nodes, P.num_tiles);
    for (int k = 0; k <= pn.order; ++k)
        if (count[k])
            std::printf(" N=%d:%d", k, count[k]);
    std::printf("\n");

    ElementWenoGpu g(s);
    g.begin(a0);
    const Field ga = g.state();
    const AdaptivePnPlan &PG = *g.adaptive_plan();
    int plan_diff = 0;
    for (int T = 0; T < P.num_tiles; ++T)
        plan_diff += P.tile_q[T] != PG.tile_q[T];
    std::printf("tiles with different tile_q CPU/GPU: %d\n", plan_diff);
    print("(1) state after initial adapt", s, P, compare_active(s, P, a, ga));
    std::printf("    inactive max |a|: CPU %.2e  GPU %.2e\n", max_inactive(s, P, a), max_inactive(s, P, ga));

    // (2) gradients of the same state
    const auto gg = g.debug_gradients();
    for (int d = 0; d < 2; ++d) {
        const Field gc = s.gradient(a, d);
        char what[64];
        std::snprintf(what, sizeof what, "(2) gradient d=%d", d);
        print(what, s, P, compare_active(s, P, gc, gg[d]));
    }

    // (3) mass solve of a random right-hand side (inactive entries zero)
    std::mt19937 rng(7);
    std::normal_distribution<double> nd;
    Field rhs = Field::Zero(s.nodes, s.moments);
    for (int i = 0; i < s.nodes; ++i)
        for (int c = 0; c < P.tile_q[P.node_tile(i)]; ++c)
            rhs(i, c) = nd(rng);
    const Field mc = s.solve_mass(rhs);
    const Field mg = g.debug_mass_solve(rhs);
    print("(3) mass solve", s, P, compare_active(s, P, mc, mg));
    {
        Field rhs_all = rhs;
        for (int i = 0; i < s.nodes; ++i)
            for (int c = P.tile_q[P.node_tile(i)]; c < s.moments; ++c)
                rhs_all(i, c) = 1e-3 * nd(rng);
        print("(3c) mass solve, inactive rhs != 0", s, P, compare_active(s, P, s.solve_mass(rhs_all), g.debug_mass_solve(rhs_all)));
        print("(3d) CPU: effect of inactive rhs", s, P, compare_active(s, P, s.solve_mass(rhs_all), mc));
        print("(3e) GPU: effect of inactive rhs", s, P, compare_active(s, P, g.debug_mass_solve(rhs_all), mg));
    }

    // (3b) one SSPRK(4,3) stage: u + dt/2 rate(0, u), split into low (c < 4) and high (c >= 4) moments
    {
        const double dt = 0.5 * s.step_bound();
        const Field xc = s.ssprk43_stage(0.0, dt, a, a, 0, 1, 1. / 6, nullptr);
        const Field xg = g.debug_stage(0.0, dt);
        Field lc = Field::Zero(s.nodes, s.moments), lg = lc, hc = lc, hg = lc;
        lc.leftCols(4) = xc.leftCols(4), lg.leftCols(4) = xg.leftCols(4);
        hc.rightCols(s.moments - 4) = xc.rightCols(s.moments - 4), hg.rightCols(s.moments - 4) = xg.rightCols(s.moments - 4);
        print("(3b) stage, all moments", s, P, compare_active(s, P, xc, xg));
        print("(3b) stage, c < 4", s, P, compare_active(s, P, lc, lg));
        print("(3b) stage, c >= 4", s, P, compare_active(s, P, hc, hg));
        std::printf("    inactive max |stage|: CPU %.2e  GPU %.2e\n", max_inactive(s, P, xc), max_inactive(s, P, xg));
    }

    // (3f) reaction half step on the adapted state (GPU: device state = adapted a)
    if (s.reacts_public()) {
        const double tau = 0.25 * s.step_bound();
        ElementWenoStats st;
        const Field rc = s.react(a, tau, nullptr, 0.0);
        g.lockstep_react(tau, 0.0, st);
        const Field rg = g.state();
        print("(3f) react", s, P, compare_active(s, P, rc, rg));
        Field lc = Field::Zero(s.nodes, s.moments), lg = lc;
        lc.leftCols(4) = rc.leftCols(4), lg.leftCols(4) = rg.leftCols(4);
        print("(3f) react, c < 4", s, P, compare_active(s, P, lc, lg));
        g.begin(a); // restore the device state
    }

    // (4) steps with fresh CPU and GPU objects from a0 (the debug calls above touched both)
    {
        ElementWeno s2(pn, o);
        Field c2 = a0;
        s2.adapt_pn_state(c2, true, nullptr);
        ElementWenoGpu g2(s2);
        g2.begin(a0);
        const double dt = 0.5 * s2.step_bound();
        ElementWenoStats sc, sg;
        for (int k = 0; k < 8; ++k) {
            const std::vector<int> q_before = s2.adaptive_plan()->tile_q;
            if (k == 1) { // first stage of step 2 (after the in-run update), before stepping
                const Field xc = s2.ssprk43_stage(k * dt, dt, c2, c2, 0, 1, 1. / 6, nullptr);
                const Field xg = g2.debug_stage(k * dt, dt);
                {
                    ElementWenoDebug dbg;
                    s2.rate(k * dt, c2, nullptr, &dbg);
                    const Field mg = g2.debug_moment_rate();
                    double md = 0, ms = 0;
                    int wi = -1;
                    for (int i = 0; i < s2.nodes; ++i)
                        for (int r = 0; r < 4; ++r) {
                            ms = std::max(ms, std::abs(dbg.ho_moment_rate(i, r)));
                            const double dd = std::abs(dbg.ho_moment_rate(i, r) - mg(i, r));
                            if (dd > md) md = dd, wi = i;
                        }
                    std::printf("(4b) step 2 PN moment_rate: rel %.2e (worst node ix %d iy %d)\n", md / ms, wi % P.npa[0], wi / P.npa[0]);
                    {
                        const Field sg = g2.debug_stab_low();
                        const Field &scpu = dbg.stabilization_low;
                        double sd = 0, ss = 0;
                        int se = -1, sk = -1;
                        for (int e = 0; e < scpu.rows(); ++e)
                            for (int kk = 0; kk < scpu.cols(); ++kk) {
                                ss = std::max(ss, std::abs(scpu(e, kk)));
                                const double dd = std::abs(scpu(e, kk) - sg(e, kk));
                                if (dd > sd) sd = dd, se = e, sk = kk;
                            }
                        std::printf("(4d) step 2 stab_low: rel %.2e (element %d, entry %d: cpu %.6e gpu %.6e)\n", sd / ss,
                                    se, sk, scpu(se, sk), sg(se, sk));
                    }
                    const Field lg = g2.debug_load();
                    double ld = 0, ls = 0;
                    int li = -1, lr = -1;
                    for (int i = 0; i < s2.nodes; ++i)
                        for (int r = 0; r < 4; ++r) {
                            ls = std::max(ls, std::abs(dbg.unlimited_rate(i, r)));
                            const double dd = std::abs(dbg.unlimited_rate(i, r) - lg(i, r));
                            if (dd > ld) ld = dd, li = i, lr = r;
                        }
                    std::printf("(4c) step 2 low-order load/lumped: rel %.2e (worst node ix %d iy %d, row %d: cpu %.6e gpu %.6e)\n",
                                ld / ls, li % P.npa[0], li / P.npa[0], lr, dbg.unlimited_rate(li, lr), lg(li, lr));
                }
                print("(4a) step 2, stage 1", s2, *s2.adaptive_plan(), compare_active(s2, *s2.adaptive_plan(), xc, xg));
                if (FILE *f = std::fopen("tmp/stage_diff.csv", "w")) {
                    const AdaptivePnPlan &P2 = *s2.adaptive_plan();
                    std::fprintf(f, "ix,iy,q_now,q_prev,diff_low,diff_high\n");
                    for (int i = 0; i < s2.nodes; ++i) {
                        const int T = P2.node_tile(i), qi = P2.tile_q[T];
                        double dl = 0, dh = 0;
                        for (int c = 0; c < qi; ++c)
                            (c < 4 ? dl : dh) = std::max(c < 4 ? dl : dh, std::abs(xc(i, c) - xg(i, c)));
                        std::fprintf(f, "%d,%d,%d,%d,%.3e,%.3e\n", i % P2.npa[0], i / P2.npa[0], qi, P2.tile_q_prev[T], dl, dh);
                    }
                    std::fclose(f);
                }
                const auto gg = g2.debug_gradients();
                for (int d = 0; d < 2; ++d)
                    print("(4a) step 2, gradient", s2, *s2.adaptive_plan(),
                          compare_active(s2, *s2.adaptive_plan(), s2.gradient(c2, d), gg[d]));
            }
            c2 = s2.advance(k * dt, dt, c2, &sc);
            g2.step(k * dt, dt, (k + 1) * dt, sg);
            const Field g2s = g2.state();
            char what[64];
            std::snprintf(what, sizeof what, "(4) after step %d", k + 1);
            const AdaptivePnPlan &P2 = *s2.adaptive_plan();
            const Report r = compare_active(s2, P2, c2, g2s);
            print(what, s2, P2, r);
            int changed = 0, grew = 0;
            for (int T = 0; T < P2.num_tiles; ++T)
                changed += P2.tile_q[T] != q_before[T], grew += P2.tile_q[T] > q_before[T];
            if (r.node >= 0) {
                const int T = P2.node_tile(r.node);
                std::printf("    tiles changed in this step's update: %d (grew %d); worst node tile %d: tile_q %d -> %d"
                            "; GPU plan tile_q %d\n", changed, grew, T, q_before[T], P2.tile_q[T], g2.adaptive_plan()->tile_q[T]);
            }
        }
    }
    return 0;
}
