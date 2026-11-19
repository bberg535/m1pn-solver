#pragma once
// Steady uncollided density of a constant isotropic source in one cell of a 3D rectilinear cell grid (Kobayashi
// benchmarks, docs/KOBAYASHI.md). Same ray integral and quadrature as src/uncollided_lattice3d_core.hpp, with arbitrary
// plane positions per axis instead of unit cells and t -> infinity (no time table):
//
//   rho_u(x) = q0/(4 pi) int_{S^2} dOmega exp(-tau_in) (1 - exp(-sigma_S L)) / sigma_S,   L = s_out - s_in,
//
// sigma_t constant per cell, vacuum outside the grid. Directions are parametrised by the visible faces of the source
// cell S (inside S: all six faces as exit faces); the face integrals are split where tau_in or the exit face of S stop
// being smooth (rays through grid edge lines, projections of grid vertices), see the core header. Host only (CPU);
// the lattice path (uncollided_lattice3d*.hpp) is unchanged.
#include "uncollided.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace m1pn {
namespace uncollided {
struct Box3dGeometry {
    std::array<std::vector<double>, 3> planes; // increasing, n_a + 1 per axis
    std::vector<double> sigma_t;               // cell (i, j, k) at i + n0 (j + n1 k)
    std::array<int, 3> source{0, 0, 0};        // source cell
    double q0 = 1;
    int order = 6;           // Gauss points per piece (outer and inner)
    int outside_panels = 2;  // equal sub-panels per piece for x outside S (no time front: smooth pieces)
    int inside_panels = 4;   // panels per face direction for x inside S (1/r^2 near the face)
    double tau_max = 60;     // rays with tau_in above are dropped (< e^-60 q0)
    static constexpr int max_breaks = 4096;

    int n(int a) const { return int(planes[a].size()) - 1; }
    double sigma(int i, int j, int k) const { return sigma_t[i + n(0) * (j + n(1) * k)]; }
    void check() const {
        for (int a = 0; a < 3; ++a) {
            if (n(a) < 1 || !std::is_sorted(planes[a].begin(), planes[a].end()))
                throw std::runtime_error("Box3dGeometry: planes must be increasing, at least one cell per axis");
            if (source[a] < 0 || source[a] >= n(a))
                throw std::runtime_error("Box3dGeometry: source cell outside the grid");
        }
        if (int(sigma_t.size()) != n(0) * n(1) * n(2))
            throw std::runtime_error("Box3dGeometry: sigma_t of size n0 n1 n2");
        if (!(sigma(source[0], source[1], source[2]) > 0))
            throw std::runtime_error("Box3dGeometry: sigma_t of the source cell must be positive");
    }
    double lo(int a) const { return planes[a][source[a]]; }
    double hi(int a) const { return planes[a][source[a] + 1]; }

    // Ray p + s d (|d| = 1) against S: [s_in, s_out] (false: no hit or touch only).
    bool hit(const double p[3], const double d[3], double &s_in, double &s_out) const {
        double a0 = 0, a1 = 1e300;
        for (int a = 0; a < 3; ++a) {
            if (std::fabs(d[a]) < 1e-300) {
                if (p[a] < lo(a) || p[a] > hi(a))
                    return false;
                continue;
            }
            double t0 = (lo(a) - p[a]) / d[a], t1 = (hi(a) - p[a]) / d[a];
            if (t0 > t1)
                std::swap(t0, t1);
            a0 = std::max(a0, t0), a1 = std::min(a1, t1);
        }
        if (a1 <= a0)
            return false;
        s_in = a0, s_out = a1;
        return true;
    }
    // Optical depth along p + s d for s in [0, length] (cell traversal on the rectilinear grid).
    double depth(const double p[3], const double d[3], double length) const {
        if (length <= 0)
            return 0;
        int c[3], step[3];
        double next[3];
        const double inf = 1e300;
        for (int a = 0; a < 3; ++a) {
            const auto &P = planes[a];
            int ci = int(std::upper_bound(P.begin(), P.end(), p[a]) - P.begin()) - 1;
            if (ci >= 0 && ci <= n(a) && p[a] == P[ci] && d[a] < 0)
                ci -= 1;
            ci = std::clamp(ci, 0, n(a) - 1);
            c[a] = ci;
            step[a] = d[a] > 0 ? 1 : -1;
            next[a] = std::fabs(d[a]) < 1e-300 ? inf : ((d[a] > 0 ? P[ci + 1] : P[ci]) - p[a]) / d[a];
        }
        double s = 0, T = 0;
        for (int guard = 0; guard < 2 * (n(0) + n(1) + n(2)) + 8; ++guard) {
            const int a = next[0] <= next[1] ? (next[0] <= next[2] ? 0 : 2) : (next[1] <= next[2] ? 1 : 2);
            const double end = std::min(next[a], length);
            T += sigma(c[0], c[1], c[2]) * (end - s);
            s = end;
            if (s >= length)
                return T;
            c[a] += step[a];
            if (c[a] < 0 || c[a] >= n(a))
                return T; // left the grid (vacuum)
            next[a] = ((d[a] > 0 ? planes[a][c[a] + 1] : planes[a][c[a]]) - p[a]) / d[a];
        }
        return T;
    }
    // Parameters w in (w0, w1) at which the ray from x through x + D0 + w Dd (a line in a face plane, Dd a unit
    // coordinate vector) passes through a grid edge line inside the grid and before its exit from S.
    void edge_breaks(const double x[3], const double D0[3], const double Dd[3], double w0, double w1,
                     std::vector<double> &out) const {
        for (int g = 0; g < 3; ++g) {
            const int p1 = (g + 1) % 3, p2 = (g + 2) % 3;
            for (double L1 : planes[p1])
                for (double L2 : planes[p2]) {
                    const double t1 = L1 - x[p1], t2 = L2 - x[p2];
                    const double A = Dd[p1] * t2 - Dd[p2] * t1, B = D0[p2] * t1 - D0[p1] * t2;
                    if (std::fabs(A) < 1e-300)
                        continue;
                    const double w = B / A;
                    if (!(w > w0 && w < w1))
                        continue;
                    const double dp1 = D0[p1] + w * Dd[p1], dp2 = D0[p2] + w * Dd[p2];
                    const double dm = std::fabs(dp1) > std::fabs(dp2) ? dp1 : dp2,
                                 tm = std::fabs(dp1) > std::fabs(dp2) ? t1 : t2;
                    if (std::fabs(dm) < 1e-300)
                        continue;
                    const double lambda = tm / dm; // fraction of the way to the face point
                    if (lambda <= 0)
                        continue;
                    const double hg = x[g] + lambda * (D0[g] + w * Dd[g]);
                    if (hg < planes[g].front() || hg > planes[g].back())
                        continue;
                    if (lambda > 1) { // behind the entry face: only up to the exit from S
                        double dd[3] = {D0[0] + w * Dd[0], D0[1] + w * Dd[1], D0[2] + w * Dd[2]};
                        const double r = std::sqrt(dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]);
                        dd[0] /= r, dd[1] /= r, dd[2] /= r;
                        double si, so;
                        if (!hit(x, dd, si, so) || lambda * r > so * (1 + 1e-12))
                            continue;
                    }
                    if (int(out.size()) < max_breaks)
                        out.push_back(w);
                }
        }
    }
    // Steady rho_u(x).
    double density(const double x[3]) const {
        const double pi = 3.14159265358979323846;
        const double scale = q0 / (4 * pi), sS = sigma(source[0], source[1], source[2]);
        thread_local std::vector<double> gx, gw, bu, bv;
        thread_local int g_order = -1;
        if (g_order != order) {
            auto r = gauss_legendre(order, 0, 1);
            gx = r.first, gw = r.second, g_order = order;
        }
        bool inside = true;
        for (int a = 0; a < 3; ++a)
            inside = inside && x[a] >= lo(a) && x[a] <= hi(a);
        double sum = 0;
        for (int a = 0; a < 3; ++a)
            for (int side = 0; side < 2; ++side) {
                const double c = side ? hi(a) : lo(a); // face plane x_a = c
                const double h = std::fabs(x[a] - c);
                if (h < 1e-14)
                    continue;
                if (!inside && (side == 0 ? x[a] >= c : x[a] <= c))
                    continue; // not visible
                const int b = (a + 1) % 3, e = (a + 2) % 3; // face coordinates u = y_b, v = y_e
                const double u0 = lo(b), u1 = hi(b), v0 = lo(e), v1 = hi(e);
                auto ray = [&](double u, double w) {
                    double d[3];
                    d[a] = c - x[a], d[b] = u - x[b], d[e] = w - x[e];
                    const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                    d[0] /= r, d[1] /= r, d[2] /= r;
                    const double weight = h / (r * r * r);
                    if (inside)
                        return weight * -std::expm1(-sS * r) / sS;
                    double si, so;
                    if (!hit(x, d, si, so))
                        return 0.;
                    const double tau = depth(x, d, r); // the visible face is the entry face: s_in = r
                    if (tau > tau_max)
                        return 0.;
                    return weight * std::exp(-tau) * -std::expm1(-sS * (so - r)) / sS;
                };
                bu.assign({u0, u1});
                if (!inside) {
                    for (double P0 : planes[0])
                        for (double P1 : planes[1])
                            for (double P2 : planes[2]) {
                                const double P[3] = {P0, P1, P2};
                                const double da = P[a] - x[a], dc = c - x[a];
                                const bool of_S = (P0 == lo(0) || P0 == hi(0)) && (P1 == lo(1) || P1 == hi(1)) &&
                                                  (P2 == lo(2) || P2 == hi(2));
                                if (std::fabs(da) < 1e-300 || da / dc <= 0 || (da / dc >= 1 && !of_S))
                                    continue;
                                const double pu = x[b] + dc / da * (P[b] - x[b]);
                                if (pu > u0 && pu < u1 && int(bu.size()) < max_breaks)
                                    bu.push_back(pu);
                            }
                    for (int side_v = 0; side_v < 2; ++side_v) {
                        double D0[3], Du[3] = {0, 0, 0};
                        D0[a] = c - x[a], D0[b] = -x[b], D0[e] = (side_v ? v1 : v0) - x[e];
                        Du[b] = 1;
                        edge_breaks(x, D0, Du, u0, u1, bu);
                    }
                }
                std::sort(bu.begin(), bu.end());
                const int up = inside ? inside_panels : outside_panels;
                for (std::size_t iu = 0; iu + 1 < bu.size(); ++iu)
                    for (int pu_ = 0; pu_ < up; ++pu_) {
                        const double ua = bu[iu] + (bu[iu + 1] - bu[iu]) * pu_ / up;
                        const double ub = bu[iu] + (bu[iu + 1] - bu[iu]) * (pu_ + 1) / up;
                        if (ub - ua < 1e-14)
                            continue;
                        for (int qu = 0; qu < order; ++qu) {
                            const double u = ua + (ub - ua) * gx[qu], wu = (ub - ua) * gw[qu];
                            bv.assign({v0, v1});
                            if (!inside) {
                                double D0[3], De[3] = {0, 0, 0};
                                D0[a] = c - x[a], D0[b] = u - x[b], D0[e] = -x[e];
                                De[e] = 1;
                                edge_breaks(x, D0, De, v0, v1, bv);
                            }
                            std::sort(bv.begin(), bv.end());
                            for (std::size_t iv = 0; iv + 1 < bv.size(); ++iv)
                                for (int pv_ = 0; pv_ < up; ++pv_) {
                                    const double va = bv[iv] + (bv[iv + 1] - bv[iv]) * pv_ / up;
                                    const double vb = bv[iv] + (bv[iv + 1] - bv[iv]) * (pv_ + 1) / up;
                                    if (vb - va < 1e-14)
                                        continue;
                                    for (int qv = 0; qv < order; ++qv) {
                                        const double w = va + (vb - va) * gx[qv], ww = (vb - va) * gw[qv];
                                        sum += wu * ww * ray(u, w);
                                    }
                                }
                        }
                    }
            }
        return scale * sum;
    }
    double density(const V3 &x) const {
        const double p[3] = {x[0], x[1], x[2]};
        return density(p);
    }
};
// Kobayashi benchmarks (Kobayashi, Sugimura, Nagaya, Prog. Nucl. Energy 39 (2001); NEA/NSC/DOC(2000)4), on the full
// box mirrored at the three reflecting faces x = y = z = 0 (the solver's mirror reduction computes the octant):
// source |x|,|y|,|z| < 10, sigma_t = 0.1 (source, absorber) and 1e-4 (void), q0 = 1; scattering ratio 0 (case i) or
// 1/2 (case ii).
//   problem 1: box [-100,100]^3, void |x|,|y|,|z| < 50 outside the source;
//   problem 2: box [-60,60] x [-100,100] x [-60,60], void duct |x|,|z| < 10, 10 < |y| (straight duct).
namespace kobayashi {
enum class Region { source, void_, absorber };
inline Region region(int problem, const V3 &x) {
    const double ax = std::fabs(x[0]), ay = std::fabs(x[1]), az = std::fabs(x[2]);
    if (ax < 10 && ay < 10 && az < 10)
        return Region::source;
    if (problem == 1)
        return ax < 50 && ay < 50 && az < 50 ? Region::void_ : Region::absorber;
    if (problem == 2)
        return ax < 10 && az < 10 ? Region::void_ : Region::absorber;
    throw std::runtime_error("kobayashi: problem 1 or 2");
}
inline double sigma_t(Region r) { return r == Region::void_ ? 1e-4 : .1; }
inline std::array<double, 3> box_half(int problem) {
    return problem == 1 ? std::array<double, 3>{100, 100, 100} : std::array<double, 3>{60, 100, 60};
}
inline Box3dGeometry geometry(int problem) {
    Box3dGeometry g;
    const auto half = box_half(problem);
    for (int a = 0; a < 3; ++a) {
        if (problem == 1)
            g.planes[a] = {-100, -50, -10, 10, 50, 100};
        else
            g.planes[a] = {-half[a], -10, 10, half[a]};
        g.source[a] = problem == 1 ? 2 : 1;
    }
    g.sigma_t.resize(g.n(0) * g.n(1) * g.n(2));
    for (int k = 0; k < g.n(2); ++k)
        for (int j = 0; j < g.n(1); ++j)
            for (int i = 0; i < g.n(0); ++i) {
                const V3 c(.5 * (g.planes[0][i] + g.planes[0][i + 1]), .5 * (g.planes[1][j] + g.planes[1][j + 1]),
                           .5 * (g.planes[2][k] + g.planes[2][k + 1]));
                g.sigma_t[i + g.n(0) * (j + g.n(1) * k)] = sigma_t(region(problem, c));
            }
    g.check();
    return g;
}
// Detector lines of the benchmark (octant coordinates, cm): (line id, x, y, z).
struct Detector {
    char line;
    double x, y, z;
};
inline std::vector<Detector> detectors(int problem) {
    std::vector<Detector> d;
    for (int k = 0; k < 10; ++k) {
        const double s = 5 + 10 * k;
        d.push_back({'A', 5, s, 5}); // y direction at x = z = 5
        if (problem == 1) {
            d.push_back({'B', s, s, s});  // diagonal
            d.push_back({'C', s, 55, 5}); // x direction at y = 55, z = 5
        } else if (k < 6)
            d.push_back({'B', s, 95, 5}); // x direction at y = 95, z = 5 (x <= 55)
    }
    return d;
}
} // namespace kobayashi
} // namespace uncollided
} // namespace m1pn
