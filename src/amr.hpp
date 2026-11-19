#pragma once
// Dynamic adaptive mesh refinement on graded tensor meshes (experiment/dynamic-amr, docs/DYNAMIC_AMR.md).
//
// Every axis keeps a fixed base mesh (the coarsest planes, with the broken lines of the case on base planes). Each base
// cell b of axis d carries a level l(d, b) in [0, L] and is split into 2^l equal elements. The mesh is the tensor product
// of these 1D meshes, so the scheme stays an ElementWeno on a graded tensor mesh (line solves, size classes, GPU path
// unchanged); only the planes change between remeshes.
//
// State transfer between two such meshes A -> B, per axis, through the common refinement C (union of the planes):
//   1. exact prolongation A -> C by Bernstein subdivision (blossoming, de Casteljau): the function is unchanged,
//      every new coefficient is a convex combination of old ones;
//   2. lumped-mass restriction C -> B: u_I = sum_j P_BC(j, I) m_j u_j / m_I with P_BC the (nonnegative, row-stochastic)
//      prolongation B -> C and m the lumped masses int phi. Because phi_I^B = sum_j P_BC(j, I) phi_j^C exactly,
//      sum_j P_BC(j, I) m_j = m_I: the rows of the restriction are convex weights, and sum_I m_I u_I = sum_j m_j u_j.
// Where B refines A the restriction is the identity (exact prolongation); where B coarsens A it is the positive,
// conservative average. The d-dimensional transfer is the tensor product of the 1D ones, applied axis by axis; the
// tensor Bernstein lumped mass is the product of the 1D lumped masses, so the density integral sum_i m_i rho_i is kept
// exactly (rounding only) and every node of B gets a convex combination of nodes of A: the M1 cone (a convex cone in
// (rho, J)) holds node by node. All PN moments are transferred by the same linear map.
#include "common.hpp"
#include "parallel.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace m1pn::amr {

// One axis: planes (count + 1 increasing coordinates), broken element lines (0 < k < count), Bernstein degree p.
struct Line {
    std::vector<double> planes;
    std::vector<int> breaks;
    int p = 1;
    int elements() const { return int(planes.size()) - 1; }
    int nodes() const { return elements() * p + 1 + int(breaks.size()); }
    // Number of broken lines k <= e per element (node index of local j in element e: e p + j + offset[e]).
    std::vector<int> offsets() const {
        std::vector<int> off(std::max(0, elements()), 0);
        for (int e = 0, m = 0; e < elements(); ++e) {
            while (m < int(breaks.size()) && breaks[m] <= e)
                ++m;
            off[e] = m;
        }
        return off;
    }
    int node(int e, int j, const std::vector<int> &off) const { return e * p + j + off[e]; }
    // int phi_i (1D Bernstein: width / (p + 1) per element and local function).
    std::vector<double> lumped() const {
        const auto off = offsets();
        std::vector<double> m(nodes(), 0.);
        for (int e = 0; e < elements(); ++e)
            for (int j = 0; j <= p; ++j)
                m[node(e, j, off)] += (planes[e + 1] - planes[e]) / (p + 1);
        return m;
    }
    int element_of(double x) const {
        const auto it = std::upper_bound(planes.begin(), planes.end(), x);
        return std::clamp(int(it - planes.begin()) - 1, 0, elements() - 1);
    }
};

// Sparse rows (CSR) of a 1D transfer: new node r <- sum_k w[k] old node col[k], k in [start[r], start[r + 1]).
struct Rows {
    int cols = 0;
    std::vector<int> start{0}, col;
    std::vector<double> w;
    int rows() const { return int(start.size()) - 1; }
};

// Bernstein coefficients on [s, t] of the polynomial with coefficients e_j on [0, 1] (row k, column j): blossom with
// p - k arguments s and k arguments t. Nonnegative, rows sum to one (s, t in [0, 1]).
inline Mat subdivision(int p, double s, double t) {
    Mat S(p + 1, p + 1);
    std::vector<double> b(p + 1);
    for (int k = 0; k <= p; ++k)
        for (int j = 0; j <= p; ++j) {
            std::fill(b.begin(), b.end(), 0.);
            b[j] = 1;
            for (int r = 1; r <= p; ++r) {
                const double u = r <= p - k ? s : t;
                for (int i = 0; i <= p - r; ++i)
                    b[i] = (1 - u) * b[i] + u * b[i + 1];
            }
            S(k, j) = b[0];
        }
    return S;
}

inline bool same_plane(double a, double b) { return std::abs(a - b) <= 1e-12 * std::max(1., std::abs(a)); }

// Exact prolongation from A to a mesh C whose planes contain those of A (same broken-line coordinates).
inline Rows prolongation(const Line &A, const Line &C) {
    const auto offA = A.offsets(), offC = C.offsets();
    Rows R;
    R.cols = A.nodes();
    int next = 0;
    for (int c = 0; c < C.elements(); ++c) {
        const double x0 = C.planes[c], x1 = C.planes[c + 1];
        const int e = A.element_of(.5 * (x0 + x1));
        const double a0 = A.planes[e], w = A.planes[e + 1] - a0;
        const double s = std::clamp((x0 - a0) / w, 0., 1.), t = std::clamp((x1 - a0) / w, 0., 1.);
        const Mat S = subdivision(C.p, s, t);
        for (int j = 0; j <= C.p; ++j) {
            const int n = C.node(c, j, offC);
            if (n < next)
                continue; // shared with the previous element (continuous segment)
            if (n != next)
                throw std::runtime_error("amr::prolongation: node order");
            for (int k = 0; k <= A.p; ++k)
                if (S(j, k) != 0) {
                    R.col.push_back(A.node(e, k, offA));
                    R.w.push_back(S(j, k));
                }
            R.start.push_back(int(R.col.size()));
            ++next;
        }
    }
    if (next != C.nodes())
        throw std::runtime_error("amr::prolongation: node count");
    return R;
}

// Planes of the common refinement of A and B (A and B must share the broken-line coordinates).
inline Line common(const Line &A, const Line &B) {
    Line C;
    C.p = A.p;
    std::vector<double> all = A.planes;
    all.insert(all.end(), B.planes.begin(), B.planes.end());
    std::sort(all.begin(), all.end());
    for (double x : all)
        if (C.planes.empty() || !same_plane(C.planes.back(), x))
            C.planes.push_back(x);
    for (int k : A.breaks) {
        const double x = A.planes[k];
        const auto it = std::find_if(C.planes.begin(), C.planes.end(), [&](double y) { return same_plane(x, y); });
        C.breaks.push_back(int(it - C.planes.begin()));
    }
    std::vector<double> bx, cx;
    for (int k : A.breaks)
        bx.push_back(A.planes[k]);
    for (int k : B.breaks)
        cx.push_back(B.planes[k]);
    bool ok = bx.size() == cx.size();
    for (std::size_t i = 0; ok && i < bx.size(); ++i)
        ok = same_plane(bx[i], cx[i]);
    if (!ok)
        throw std::runtime_error("amr::common: broken lines differ");
    return C;
}

// Transfer A -> B: exact prolongation to the common refinement, then the lumped-mass restriction to B.
inline Rows transfer(const Line &A, const Line &B) {
    const Line C = common(A, B);
    const Rows PA = prolongation(A, C), PB = prolongation(B, C);
    const auto mC = C.lumped();
    // Transpose of PB scaled: R(I, j) = PB(j, I) m_j / m_I.
    std::vector<std::vector<std::pair<int, double>>> R(B.nodes());
    for (int j = 0; j < PB.rows(); ++j)
        for (int k = PB.start[j]; k < PB.start[j + 1]; ++k)
            R[PB.col[k]].push_back({j, PB.w[k] * mC[j]});
    Rows T;
    T.cols = A.nodes();
    std::vector<double> acc(A.nodes(), 0.);
    std::vector<int> touched;
    for (int I = 0; I < B.nodes(); ++I) {
        // Exact copy where the restriction row is a single node of C with weight one (refined or unchanged).
        double total = 0;
        for (const auto &[j, v] : R[I])
            total += v;
        for (const auto &[j, v] : R[I]) {
            const double f = v / total; // = PB(j, I) m_j / m_I with m_I computed as the same sum (exactly one row sum)
            for (int k = PA.start[j]; k < PA.start[j + 1]; ++k) {
                const int c = PA.col[k];
                if (acc[c] == 0)
                    touched.push_back(c);
                acc[c] += f * PA.w[k];
            }
        }
        std::sort(touched.begin(), touched.end());
        for (int c : touched)
            if (acc[c] != 0) {
                T.col.push_back(c);
                T.w.push_back(acc[c]);
                acc[c] = 0;
            }
        touched.clear();
        T.start.push_back(int(T.col.size()));
    }
    return T;
}

// Apply the 1D transfers axis by axis to a node field (node index = sum_d coordinate_d * stride_d, rows = nodes).
inline Field apply(const Field &in, std::array<int, 3> n, int dim, const std::array<Rows, 3> &T) {
    Field cur = in;
    const int q = int(in.cols());
    for (int d = 0; d < dim; ++d) {
        if (T[d].cols != n[d])
            throw std::runtime_error("amr::apply: transfer size");
        std::array<int, 3> m = n;
        m[d] = T[d].rows();
        long low = 1, high = 1;
        for (int k = 0; k < d; ++k)
            low *= n[k];
        for (int k = d + 1; k < 3; ++k)
            high *= n[k];
        Field out = Field::Zero(long(m[0]) * m[1] * m[2], q);
        const Rows &R = T[d];
        parallel_for(int(low * high), [&](int line) {
            const long a = line % low, b = line / low;
            const long in_base = a + b * low * n[d], out_base = a + b * low * m[d];
            for (int I = 0; I < R.rows(); ++I) {
                double *o = &out(out_base + I * low, 0);
                for (int k = R.start[I]; k < R.start[I + 1]; ++k) {
                    const double w = R.w[k];
                    const double *x = &cur(in_base + long(R.col[k]) * low, 0);
                    for (int c = 0; c < q; ++c)
                        o[c] += w * x[c];
                }
            }
        });
        cur.swap(out);
        n = m;
    }
    return cur;
}

// Base mesh and levels of one axis.
struct Axis {
    std::vector<double> base;   // base planes (count_base + 1)
    std::vector<double> breaks; // coordinates of the broken lines (base planes)
    std::vector<int> level;     // per base cell
    int p = 1;
    Line line() const {
        Line L;
        L.p = p;
        L.planes.push_back(base.front());
        for (std::size_t b = 0; b + 1 < base.size(); ++b) {
            const int n = 1 << level[b];
            const double x0 = base[b], w = base[b + 1] - base[b];
            for (int k = 1; k < n; ++k)
                L.planes.push_back(x0 + (w * k) / n); // dyadic: coincident planes of different levels are bitwise equal
            L.planes.push_back(base[b + 1]);
        }
        for (double x : breaks) {
            const auto it = std::find_if(L.planes.begin(), L.planes.end(), [&](double y) { return same_plane(x, y); });
            if (it == L.planes.end())
                throw std::runtime_error("amr::Axis: broken line not on a base plane");
            L.breaks.push_back(int(it - L.planes.begin()));
        }
        return L;
    }
    // Base cell of every element of line().
    std::vector<int> element_base() const {
        std::vector<int> eb;
        for (std::size_t b = 0; b + 1 < base.size(); ++b)
            eb.insert(eb.end(), std::size_t(1) << level[b], int(b));
        return eb;
    }
};

struct Settings {
    int max_level = 2, min_level = 0;
    double tol = 1e-3;        // refine where eta > tol (eta relative to the largest density coefficient)
    double hysteresis = .25;  // coarsen where 4 eta < hysteresis * tol
    int buffer = 1;           // base cells of dilation around every refined cell (front motion per interval)
};

// Indicator per axis and base cell: eta = max over the nodes of the base cell (along d) of the second divided
// difference of the four low coefficients along d (Greville abscissae of the Bernstein control points), times the
// squared element width, relative to `scale`. Estimates the interpolation error at the current width (h^2 u'' / 8 up
// to the constant): coarsening multiplies it by 4. Differences never cross a broken line (separate continuous spaces).
// low: nodes x 4 (density-like coefficient in column 0).
inline std::array<std::vector<double>, 3> indicator(const Field &u4, int dim, const std::array<Axis, 3> &axes,
                                                     double scale) {
    std::array<std::vector<double>, 3> eta;
    std::array<int, 3> n{1, 1, 1};
    std::array<Line, 3> L;
    for (int d = 0; d < dim; ++d)
        L[d] = axes[d].line(), n[d] = L[d].nodes();
    const double inv = 1 / std::max(scale, 1e-300);
    for (int d = 0; d < dim; ++d) {
        const Line &l = L[d];
        const auto off = l.offsets();
        const auto eb = axes[d].element_base();
        const int nd = n[d], p = l.p;
        // Per 1D node: Greville abscissa, element (lower one if shared), segment id, width; base cells it touches.
        std::vector<double> x(nd), width(nd);
        std::vector<int> seg(nd), base_lo(nd), base_hi(nd);
        for (int e = 0; e < l.elements(); ++e) {
            const double x0 = l.planes[e], w = l.planes[e + 1] - x0;
            for (int j = 0; j <= p; ++j) {
                const int i = l.node(e, j, off);
                const bool first = j == 0 && (e == 0 || off[e] != off[e - 1]);
                if (j == 0 && !first) { // shared with element e - 1
                    base_hi[i] = eb[e];
                    width[i] = std::max(width[i], w);
                    continue;
                }
                x[i] = x0 + w * j / p;
                width[i] = w;
                seg[i] = off[e];
                base_lo[i] = base_hi[i] = eb[e];
            }
        }
        const int cells = int(axes[d].base.size()) - 1;
        const long low = d == 0 ? 1 : (d == 1 ? n[0] : long(n[0]) * n[1]);
        const long high = (d == 0 ? long(n[1]) * n[2] : (d == 1 ? n[2] : 1));
        std::vector<std::vector<double>> part(std::max(1, omp_get_max_threads()), std::vector<double>(cells, 0.));
        parallel_for(int(low * high), [&](int line) {
            auto &mine = part[omp_get_thread_num()];
            const long a = line % low, b = line / low, base = a + b * low * nd;
            for (int k = 1; k + 1 < nd; ++k) {
                if (seg[k - 1] != seg[k] || seg[k + 1] != seg[k])
                    continue;
                const double hl = x[k] - x[k - 1], hr = x[k + 1] - x[k];
                if (hl <= 0 || hr <= 0)
                    continue;
                const double *um = &u4(base + (k - 1) * low, 0), *u0 = &u4(base + k * low, 0),
                             *up = &u4(base + (k + 1) * low, 0);
                double m = 0;
                for (int c = 0; c < 4; ++c) {
                    const double dd = 2 * ((up[c] - u0[c]) / hr - (u0[c] - um[c]) / hl) / (hl + hr);
                    m = std::max(m, std::abs(dd));
                }
                const double e = m * width[k] * width[k] * inv;
                mine[base_lo[k]] = std::max(mine[base_lo[k]], e);
                mine[base_hi[k]] = std::max(mine[base_hi[k]], e);
            }
        });
        eta[d].assign(cells, 0.);
        for (const auto &v : part)
            for (int c = 0; c < cells; ++c)
                eta[d][c] = std::max(eta[d][c], v[c]);
    }
    return eta;
}

// New levels from the indicator: refine (possibly several levels, eta / 4^k <= tol), coarsen by one level, dilate by
// `buffer` base cells, and grade (neighbouring base cells differ by at most one level). Returns true if any changed.
inline bool update_levels(std::array<Axis, 3> &axes, int dim, const std::array<std::vector<double>, 3> &eta,
                          const Settings &s) {
    bool changed = false;
    for (int d = 0; d < dim; ++d) {
        auto &lev = axes[d].level;
        const int cells = int(lev.size());
        std::vector<int> target(cells);
        for (int b = 0; b < cells; ++b) {
            int t = lev[b];
            if (eta[d][b] > s.tol) {
                double e = eta[d][b];
                while (e > s.tol && t < s.max_level)
                    ++t, e /= 4;
            } else if (4 * eta[d][b] < s.hysteresis * s.tol && t > s.min_level)
                --t;
            target[b] = std::clamp(t, s.min_level, s.max_level);
        }
        std::vector<int> dil = target;
        for (int b = 0; b < cells; ++b)
            for (int k = std::max(0, b - s.buffer); k <= std::min(cells - 1, b + s.buffer); ++k)
                dil[k] = std::max(dil[k], target[b]);
        for (bool again = true; again;) { // 2:1 grading
            again = false;
            for (int b = 0; b < cells; ++b)
                for (int k : {b - 1, b + 1})
                    if (k >= 0 && k < cells && dil[b] < dil[k] - 1)
                        dil[b] = dil[k] - 1, again = true;
        }
        if (dil != lev)
            changed = true;
        lev = dil;
    }
    return changed;
}

} // namespace m1pn::amr
