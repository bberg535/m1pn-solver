// C1 probe: replay the SSPRK3 stages of the C1 counterexample (T2 corner_cone_boundary_J) and report the pair
// (u_i,u_j,edge,d) for which stable_base_bar throws, in hexfloat (exact binary64 inputs).
#include "audit_common.hpp"
#include <cstdio>

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Angular pn(3);
    ElementWenoOptions o; o.degree = 1; o.dimension = 2; o.elements = 8; o.periodic = false;
    ElementWeno s(pn, o);
    Field a = Field::Zero(s.nodes, pn.q);
    Vec row = Vec::Zero(pn.q);
    set_low(row, Four(1., -1., 0., 0.));
    while (!exact::cone(project(row))) row[0] = std::nextafter(row[0], 1e300);
    a.row(0) = row.transpose();
    int failures = 0, calls = 0;
    auto scan = [&](const Field &x, const std::string &where) {
        for (int e = 0; e < s.cells; ++e) {
            const int *map = &s.connectivity_[e * s.local_nodes];
            for (int i = 0; i < s.local_nodes; ++i)
                for (int j = 0; j < s.local_nodes; ++j) {
                    if (i == j) continue;
                    const double dij = s.viscosity_ij_(i, j);
                    if (!(dij > 0)) continue;
                    const Four ui = project(x.row(map[i]).transpose()), uj = project(x.row(map[j]).transpose());
                    V3 edge = V3::Zero();
                    Four fd = Four::Zero();
                    const Levermore li(ui), lj(uj);
                    for (int d = 0; d < s.dimension; ++d) {
                        const double cij = s.convection_[d](i, j);
                        edge[d] = cij;
                        V3 nn = V3::Zero();
                        nn[d] = 1;
                        fd += cij * (lj.flux(nn) - li.flux(nn));
                    }
                    const Four pair_bar = .5 * (ui + uj) - fd / (2 * dij);
                    if (exact::cone(pair_bar)) continue;
                    ++calls;
                    try {
                        stable_base_bar(ui, uj, edge, dij);
                    } catch (const std::exception &ex) {
                        ++failures;
                        if (failures <= 3) {
                            std::printf("{\"test\":\"C1_probe\",\"where\":\"%s\",\"element\":%d,\"i\":%d,\"j\":%d,\"what\":\"%s\",", where.c_str(), e, i, j, ex.what());
                            std::printf("\"ui\":[\"%a\",\"%a\",\"%a\",\"%a\"],\"uj\":[\"%a\",\"%a\",\"%a\",\"%a\"],\"edge\":[\"%a\",\"%a\",\"%a\"],\"d\":\"%a\"}\n",
                                        ui[0], ui[1], ui[2], ui[3], uj[0], uj[1], uj[2], uj[3], edge[0], edge[1], edge[2], dij);
                        }
                    }
                }
        }
    };
    auto adm = [&](Field x) {
        for (int i = 0; i < s.nodes; ++i) {
            Four u = project(x.row(i).transpose());
            if (nudge(u)) {
                Vec r = x.row(i).transpose();
                set_low(r, u);
                x.row(i) = r.transpose();
            }
        }
        return x;
    };
    const int steps = std::max(1, int(std::ceil(.1 / s.step_bound() - 1e-12)));
    const double dt = .1 / steps;
    double t = 0;
    for (int k = 0; k < steps && failures == 0; ++k) {
        scan(a, "step" + std::to_string(k) + " stage1");
        Field y1 = adm(a + dt * s.rate(t, a));
        scan(y1, "step" + std::to_string(k) + " stage2");
        Field c2 = adm(.75 * a + .25 * (y1 + dt * s.rate(t + dt, y1)));
        scan(c2, "step" + std::to_string(k) + " stage3");
        a = adm((1. / 3) * a + (2. / 3) * (c2 + dt * s.rate(t + .5 * dt, c2)));
        t += dt;
    }
    std::printf("{\"test\":\"C1_probe_summary\",\"pairs_needing_stable_bar\":%d,\"failures\":%d}\n", calls, failures);
}
