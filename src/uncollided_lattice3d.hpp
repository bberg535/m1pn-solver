#pragma once
// Uncollided density of the lattice-3d source (constant isotropic q0 on one cell, switched on at t = 0) in the 3D cell
// lattice; formulas and quadrature in src/uncollided_lattice3d_core.hpp, docs/FCS_LATTICE.md (section 3D).
#include "uncollided.hpp"
#include "uncollided_lattice3d_core.hpp"
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace m1pn {
namespace uncollided {
struct CellLattice3D {
    int n[3] = {7, 7, 7};
    std::vector<double> sigma_t; // n0 n1 n2, cell (i, j, k) at i + n0 (j + n1 k)
    int source[3] = {3, 3, 3};
    double q0 = 1;
    // Gauss points per piece and equal sub-panels (tests/uncollided_lattice3d.cpp): without the front in S the rule is
    // exact to 1e-10; the front kinks give ~1e-5 relative (1e-4 right after arrival) with 8 sub-panels, ~5 ms per point.
    int outer_order = 6, inner_order = 6, inside_panels = 8, outside_panels = 8;
    double tau_max = 40;
    // Polar rule (lattice3d_row_polar, docs/FCS_LATTICE.md): adaptive polar quadrature about the foot point.
    bool polar = false;
    double polar_tol = 1e-4, polar_floor = 1e-9;

    Lattice3dView view(const std::vector<double> &ox, const std::vector<double> &ow, const std::vector<double> &ix,
                       const std::vector<double> &iw) const {
        if (n[0] > 8 || n[1] > 8 || n[2] > 8 || int(sigma_t.size()) != n[0] * n[1] * n[2])
            throw std::runtime_error("CellLattice3D: at most 8^3 cells, sigma_t of size n0 n1 n2");
        Lattice3dView v;
        for (int a = 0; a < 3; ++a)
            v.n[a] = n[a], v.s[a] = source[a];
        v.sigma_t = sigma_t.data(), v.q0 = q0;
        v.outer_n = int(ox.size()), v.inner_n = int(ix.size());
        v.outer_x = ox.data(), v.outer_w = ow.data(), v.inner_x = ix.data(), v.inner_w = iw.data();
        v.inside_panels = inside_panels, v.outside_panels = outside_panels, v.tau_max = tau_max;
        v.polar = polar, v.polar_tol = polar_tol, v.polar_floor = polar_floor;
        if (!(lattice3d_sigma(v, source[0], source[1], source[2]) > 0))
            throw std::runtime_error("CellLattice3D: sigma_t of the source cell must be positive");
        return v;
    }
    template <class F> auto with_view(F &&f) const {
        static thread_local std::vector<double> ox, ow, ix, iw;
        static thread_local int go = -1, gi = -1;
        if (go != outer_order) {
            auto r = gauss_legendre(outer_order, 0, 1);
            ox = r.first, ow = r.second, go = outer_order;
        }
        if (gi != inner_order) {
            auto r = gauss_legendre(inner_order, 0, 1);
            ix = r.first, iw = r.second, gi = inner_order;
        }
        return f(view(ox, ow, ix, iw));
    }
    double density(const V3 &x, double t) const {
        const double p[3] = {x[0], x[1], x[2]};
        return with_view([&](const Lattice3dView &v) { return lattice3d_density(v, p, t); });
    }
    // Distance from x to the source cell (rho_u = 0 for t <= distance).
    double distance(const V3 &x) const {
        double s = 0;
        for (int a = 0; a < 3; ++a) {
            const double e = std::max({source[a] - x[a], 0., x[a] - source[a] - 1});
            s += e * e;
        }
        return std::sqrt(s);
    }
};
// The lattice-3d case (src/benchmark_cases.hpp): sigma_t = sigma_a + sigma_s per cell, source cell (3, 3, 3).
template <class SigmaT> CellLattice3D lattice3d_cells(SigmaT sigma_t) {
    CellLattice3D c;
    c.sigma_t.resize(343);
    for (int k = 0; k < 7; ++k)
        for (int j = 0; j < 7; ++j)
            for (int i = 0; i < 7; ++i)
                c.sigma_t[i + 7 * (j + 7 * k)] = sigma_t(i, j, k);
    return c;
}
// Compact list of active ray-traced points for nodal FCS table construction (shared by CPU and GPU).
struct Lattice3dEvalPlan {
    std::vector<double> px, py, pz; // coordinates of evaluated points
    std::vector<int> elem;          // element index
    std::vector<int> local_k;       // Gauss point index 0..P-1, or -1 for element-midpoint 1-pt rule
    std::vector<double> weight;     // quadrature weight (fw[e,k] or element volume for midpoint)
    std::vector<double> sigma_a;    // absorption cross section at the point
    std::vector<int> owned;         // 1 if element is owned by this MPI rank
    double summary_points = 0;      // owned FCS quadrature point count (cells_owned * P)
    std::size_t adaptive_elements = 0;
    std::size_t size() const { return px.size(); }
};
// rho_u at fixed points on t_m = m dt, m = 0..M, cubic interpolation in t; same interface as LatticeUncollidedTable.
// In nodal mode (nodal = true), rows are reduced on the fly into a nodal FCS source table of size nodes x (M + 1)
// (8x less memory in double, 16x in float) together with the exact scalar summary and 1D mass table.
struct Lattice3dUncollidedTable {
    CellLattice3D lattice;
    double dt = .01;
    int M = 0;
    std::vector<V3> points;
    std::vector<double> table; // legacy point-major table: points x (M + 1)
    // Nodal mode (Lever A) and adaptive 1-pt vs 8-pt spatial quadrature (Lever C).
    bool nodal = false;
    bool nodal_fp32 = true;
    bool adaptive_quad = false;
    double adaptive_quad_tau = 0.25;  // max sigma_t * h for 1-pt midpoint rule
    double adaptive_quad_dist = 1.0;  // min distance from element centre to S for 1-pt midpoint rule
    int nodes = 0;
    bool nodal_built = false;
    std::vector<float> node_table_f;  // nodes x (M + 1) when nodal_fp32
    std::vector<double> node_table_d; // nodes x (M + 1) when !nodal_fp32
    std::vector<double> mass_table;   // M + 1: owned-domain int rho_u(x, t_m) dx
    double summary_mass = 0, summary_abs_rate = 0, summary_absorbed = 0, summary_points = 0;
    std::size_t evaluated_points = 0, adaptive_elements = 0;
    double build_seconds = 0;
    double final_time = 0;
    int times_for(double T) const { return int(std::ceil(T / dt)) + 2; }
    void reset() {
        points.clear();
        table.clear();
        node_table_f.clear();
        node_table_d.clear();
        mass_table.clear();
        nodal_built = false;
        nodes = 0;
        summary_mass = summary_abs_rate = summary_absorbed = summary_points = 0;
        evaluated_points = adaptive_elements = 0;
    }
    void build(const std::vector<V3> &x, double T) {
        const auto start = std::chrono::steady_clock::now();
        points = x;
        M = times_for(T);
        table.assign(points.size() * std::size_t(M + 1), 0.);
        parallel_for(int(points.size()), [&](int k) {
            if (lattice.distance(points[k]) >= M * dt)
                return; // the front does not reach x within the table
            thread_local std::vector<double> work;
            work.resize(lattice3d_work_size(lattice.view({}, {}, {}, {}), M));
            const double p[3] = {points[k][0], points[k][1], points[k][2]};
            lattice.with_view([&](const Lattice3dView &v) {
                lattice3d_row(v, p, dt, M, work.data(), &table[std::size_t(k) * (M + 1)]);
                return 0;
            });
        });
        build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    // Build the compact list of active ray-traced points (horizon-culled, optionally 1-pt midpoint in smooth far cells).
    template <class OwnedFn, class SigmaAFn>
    Lattice3dEvalPlan plan_eval_points(int cells, int P, const std::vector<V3> &fcs_points,
                                       const std::vector<double> &fcs_weight, const std::vector<double> &fcs_sigma_s,
                                       OwnedFn &&is_owned, SigmaAFn &&sigma_a_of, int M_times) const {
        Lattice3dEvalPlan plan;
        const double horizon = M_times * dt;
        plan.px.reserve(std::size_t(cells));
        plan.py.reserve(std::size_t(cells));
        plan.pz.reserve(std::size_t(cells));
        plan.elem.reserve(std::size_t(cells));
        plan.local_k.reserve(std::size_t(cells));
        plan.weight.reserve(std::size_t(cells));
        plan.sigma_a.reserve(std::size_t(cells));
        plan.owned.reserve(std::size_t(cells));
        for (int e = 0; e < cells; ++e) {
            const bool own = is_owned(e);
            if (own)
                plan.summary_points += double(P);
            const std::size_t base = std::size_t(e) * P;
            double vol = 0;
            V3 center(0, 0, 0);
            double sa0 = sigma_a_of(fcs_points[base]), ss0 = fcs_sigma_s[base];
            bool uniform_mat = true;
            for (int k = 0; k < P; ++k) {
                const V3 &pt = fcs_points[base + k];
                vol += fcs_weight[base + k];
                center += pt;
                if (k > 0) {
                    const double sak = sigma_a_of(pt);
                    if (std::abs(sak - sa0) > 1e-14 || std::abs(fcs_sigma_s[base + k] - ss0) > 1e-14)
                        uniform_mat = false;
                }
            }
            center /= double(P);
            const double h = std::cbrt(std::max(vol, 0.));
            const double d_center = lattice.distance(center);
            if (adaptive_quad && P == 8 && uniform_mat && (sa0 + ss0) * h <= adaptive_quad_tau &&
                d_center >= adaptive_quad_dist) {
                ++plan.adaptive_elements;
                if (d_center < horizon) {
                    plan.px.push_back(center[0]);
                    plan.py.push_back(center[1]);
                    plan.pz.push_back(center[2]);
                    plan.elem.push_back(e);
                    plan.local_k.push_back(-1);
                    plan.weight.push_back(vol);
                    plan.sigma_a.push_back(sa0);
                    plan.owned.push_back(own ? 1 : 0);
                }
                continue;
            }
            for (int k = 0; k < P; ++k) {
                const V3 &pt = fcs_points[base + k];
                if (lattice.distance(pt) >= horizon)
                    continue;
                plan.px.push_back(pt[0]);
                plan.py.push_back(pt[1]);
                plan.pz.push_back(pt[2]);
                plan.elem.push_back(e);
                plan.local_k.push_back(k);
                plan.weight.push_back(fcs_weight[base + k]);
                plan.sigma_a.push_back(sigma_a_of(pt));
                plan.owned.push_back(own ? 1 : 0);
            }
        }
        return plan;
    }
    // CPU nodal build: streams ray-traced rows directly into the nodal source table (nodes x (M + 1)) and summary.
    template <class OwnedFn, class SigmaAFn>
    void build_nodal(int cells, int n_nodes, int local_nodes, int P, const std::vector<int> &connectivity,
                     const std::vector<V3> &fcs_points, const std::vector<double> &fcs_weight,
                     const std::vector<double> &fcs_sigma_s, const double *fcs_shape_row_major,
                     const double *fcs_node_mass, OwnedFn &&is_owned, SigmaAFn &&sigma_a_of, double T) {
        const auto start = std::chrono::steady_clock::now();
        final_time = T;
        M = times_for(T);
        nodes = n_nodes;
        const std::size_t W = std::size_t(M + 1);
        const Lattice3dEvalPlan plan =
            plan_eval_points(cells, P, fcs_points, fcs_weight, fcs_sigma_s, is_owned, sigma_a_of, M);
        summary_points = plan.summary_points;
        evaluated_points = plan.size();
        adaptive_elements = plan.adaptive_elements;
        std::vector<double> acc(std::size_t(nodes) * W, 0.);
        mass_table.assign(W, 0.);
        std::vector<double> sm(plan.size(), 0.), sr(plan.size(), 0.), sa_tot(plan.size(), 0.);
        parallel_for(int(plan.size()), [&](int q) {
            thread_local std::vector<double> work, row;
            const std::size_t ws = lattice.polar ? 10 * std::size_t(M + 2) : 3 * std::size_t(M + 2);
            if (work.size() < ws)
                work.resize(ws);
            if (row.size() < W)
                row.resize(W);
            const double p[3] = {plan.px[q], plan.py[q], plan.pz[q]};
            lattice.with_view([&](const Lattice3dView &v) {
                lattice3d_row(v, p, dt, M, work.data(), row.data());
                return 0;
            });
            const int e = plan.elem[q], lk = plan.local_k[q];
            const std::size_t base = std::size_t(e) * P;
            for (int l = 0; l < local_nodes; ++l) {
                double c = 0;
                if (lk >= 0) {
                    c = fcs_shape_row_major[std::size_t(l) * P + lk] * fcs_weight[base + lk] * fcs_sigma_s[base + lk];
                } else {
                    for (int k = 0; k < P; ++k)
                        c += fcs_shape_row_major[std::size_t(l) * P + k] * fcs_weight[base + k] * fcs_sigma_s[base + k];
                }
                if (c == 0)
                    continue;
                const int node = connectivity[std::size_t(e) * local_nodes + l];
                const double scale = c / fcs_node_mass[node];
                double *dst = &acc[std::size_t(node) * W];
                for (std::size_t m = 0; m < W; ++m) {
                    const double add = scale * row[m];
                    if (add == 0)
                        continue;
#ifdef _OPENMP
#pragma omp atomic
#endif
                    dst[m] += add;
                }
            }
            if (plan.owned[q]) {
                const double w = plan.weight[q], sa = plan.sigma_a[q];
                for (std::size_t m = 0; m < W; ++m) {
                    const double add = w * row[m];
                    if (add == 0)
                        continue;
#ifdef _OPENMP
#pragma omp atomic
#endif
                    mass_table[m] += add;
                }
                const double rT = lattice3d_eval_row(row.data(), M, dt, T);
                sm[q] = w * rT;
                sr[q] = w * sa * rT;
                if (sa > 0)
                    sa_tot[q] = w * sa * lattice3d_time_integral_row(row.data(), M, dt, T);
            }
        });
        summary_mass = summary_abs_rate = summary_absorbed = 0;
        for (std::size_t q = 0; q < plan.size(); ++q)
            summary_mass += sm[q], summary_abs_rate += sr[q], summary_absorbed += sa_tot[q];
        if (nodal_fp32) {
            node_table_d.clear();
            node_table_f.resize(acc.size());
            for (std::size_t i = 0; i < acc.size(); ++i)
                node_table_f[i] = float(acc[i]);
        } else {
            node_table_f.clear();
            node_table_d = std::move(acc);
        }
        nodal_built = true;
        build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    double nodal_value(int node, double t) const {
        const std::size_t off = std::size_t(node) * std::size_t(M + 1);
        if (nodal_fp32)
            return lattice3d_eval_row(&node_table_f[off], M, dt, t);
        return lattice3d_eval_row(&node_table_d[off], M, dt, t);
    }
    double nodal_mass(double t) const {
        if (mass_table.empty())
            return 0;
        return lattice3d_eval_row(mass_table.data(), M, dt, t);
    }
    bool matches(const std::vector<V3> &x) const {
        if (x.size() != points.size() || x.empty())
            return x.size() == points.size();
        for (std::size_t k : {std::size_t(0), x.size() / 2, x.size() - 1})
            if ((x[k] - points[k]).norm() > 0)
                return false;
        return true;
    }
    double value(std::size_t k, double t) const {
        if (t <= 0)
            return 0;
        const double s = t / dt;
        int i = int(s);
        if (i >= M - 1)
            throw std::runtime_error("Lattice3dUncollidedTable: time beyond the table");
        const double f = s - i, *row = &table[k * std::size_t(M + 1)];
        const double p1 = row[i], p2 = row[i + 1], p3 = row[i + 2], p0 = i > 0 ? row[i - 1] : 2 * p1 - p2;
        const double v = p1 + .5 * f * (p2 - p0 + f * (2 * p0 - 5 * p1 + 4 * p2 - p3 + f * (3 * (p1 - p2) + p3 - p0)));
        return std::max(v, 0.);
    }
    void density(double t, const std::vector<V3> &x, std::vector<double> &out) const {
        if (!matches(x))
            throw std::runtime_error("Lattice3dUncollidedTable: points differ from the tabulated ones");
        out.resize(x.size());
        parallel_for(int(x.size()), [&](int k) { out[k] = value(std::size_t(k), t); });
    }
    double time_integral(std::size_t k, double t) const {
        const double *row = &table[k * std::size_t(M + 1)];
        return lattice3d_time_integral_row(row, M, dt, t);
    }
};
} // namespace uncollided
} // namespace m1pn
