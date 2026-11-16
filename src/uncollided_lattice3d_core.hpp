#pragma once
// Uncollided density of a constant isotropic source switched on at t = 0 in a 3D cell lattice (lattice-3d,
// docs/FCS_LATTICE.md, section 3D). Written once for host and device: plain arrays, no allocation.
//
// sigma_t is constant on the unit cells [i,i+1] x [j,j+1] x [k,k+1] of [0,nx] x [0,ny] x [0,nz]; vacuum outside. The
// source q0 is isotropic on one cell S with sigma_t = sigma_S there. With s the distance along -Omega,
//
//   rho_u(x, t) = q0/(4 pi) int_{S^2} dOmega int_0^t chi_S(x - s Omega) exp(-tau(x, x - s Omega)) ds.
//
// S is convex: a ray meets it in [s_in, s_out], tau = tau_in + sigma_S (s - s_in) there, so the s integral is
//
//   exp(-tau_in) (1 - exp(-sigma_S L)) / sigma_S,   L = min(s_out, t) - s_in > 0.
//
// Directions: x outside S sees up to three faces F of S; on F the ray to y in F enters S at y (s_in = |y - x|) and
// dOmega = |n.(y - x)| / |y - x|^3 dA(y). Inside S every direction counts; the six faces are then exit faces
// (s_in = tau_in = 0, s_out = |y - x|).
//
// Quadrature (time independent): on every face an iterated Gauss rule, outer variable u, inner v. The inner line
// y(u, .) is split where the ray x -> y(u, v) passes through a lattice edge line (axis-parallel line through integer
// coordinates) before its exit from S: there the cell sequence before S or the exit face changes. The outer variable is
// split at the projections of the lattice vertices. On each piece tau_in, s_in and s_out are smooth. The time
// dependence is exact per ray (lattice3d_rays emits (weight, tau_in, s_in, s_out)); only the front s_in = t and
// s_out = t, kinks of the integrand at fixed t, are not resolved by the panels (second order there).
#include <math.h>
#if defined(__CUDACC__)
#define M1PN_HD3 __host__ __device__
#else
#define M1PN_HD3
#endif
namespace m1pn {
namespace uncollided {
constexpr int lattice3d_max_breaks = 1024; // per line: ends + edge lines (3 * 9 * 9) + vertices (9^3 projected)
struct Lattice3dView {
    int n[3] = {7, 7, 7};
    const double *sigma_t = nullptr; // n0 n1 n2, cell (i, j, k) at i + n0 (j + n1 k)
    int s[3] = {3, 3, 3};            // source cell
    double q0 = 1;
    int outer_n = 0, inner_n = 0; // Gauss points per piece (rules on [0, 1] from the caller)
    const double *outer_x = nullptr, *outer_w = nullptr, *inner_x = nullptr, *inner_w = nullptr;
    int inside_panels = 4;  // panels per face direction for x inside S (no breakpoints needed there)
    int outside_panels = 1; // equal sub-panels per piece and direction for x outside S (front kinks, see above)
    double tau_max = 40;   // rays with tau_in above are dropped (< e^-40 q0)
};
M1PN_HD3 inline double lattice3d_sigma(const Lattice3dView &v, int i, int j, int k) {
    return v.sigma_t[i + v.n[0] * (j + v.n[1] * k)];
}
// Ray p + s d (|d| = 1) against S: [s_in, s_out] (false: no hit or touch only).
M1PN_HD3 inline bool lattice3d_hit(const Lattice3dView &v, const double p[3], const double d[3], double &s_in,
                                   double &s_out) {
    double lo = 0, hi = 1e300;
    for (int a = 0; a < 3; ++a) {
        const double b0 = v.s[a], b1 = v.s[a] + 1.;
        if (fabs(d[a]) < 1e-300) {
            if (p[a] < b0 || p[a] > b1)
                return false;
            continue;
        }
        double t0 = (b0 - p[a]) / d[a], t1 = (b1 - p[a]) / d[a];
        if (t0 > t1) {
            const double w = t0;
            t0 = t1, t1 = w;
        }
        lo = fmax(lo, t0), hi = fmin(hi, t1);
    }
    if (hi <= lo)
        return false;
    s_in = lo, s_out = hi;
    return true;
}
// Optical depth along p + s d for s in [0, length] (voxel traversal, Amanatides-Woo).
M1PN_HD3 inline double lattice3d_depth(const Lattice3dView &v, const double p[3], const double d[3], double length) {
    if (length <= 0)
        return 0;
    int c[3], step[3];
    double next[3], delta[3];
    const double inf = 1e300;
    for (int a = 0; a < 3; ++a) {
        double f = floor(p[a]);
        if (p[a] == f && d[a] < 0)
            f -= 1;
        int ci = int(f);
        ci = ci < 0 ? 0 : (ci > v.n[a] - 1 ? v.n[a] - 1 : ci);
        c[a] = ci;
        step[a] = d[a] > 0 ? 1 : -1;
        next[a] = fabs(d[a]) < 1e-300 ? inf : ((d[a] > 0 ? ci + 1 : ci) - p[a]) / d[a];
        delta[a] = fabs(d[a]) < 1e-300 ? inf : 1 / fabs(d[a]);
    }
    double s = 0, T = 0;
    for (int guard = 0; guard < 4 * (v.n[0] + v.n[1] + v.n[2]) + 8; ++guard) {
        const int a = next[0] <= next[1] ? (next[0] <= next[2] ? 0 : 2) : (next[1] <= next[2] ? 1 : 2);
        const double end = fmin(next[a], length);
        T += lattice3d_sigma(v, c[0], c[1], c[2]) * (end - s);
        s = end;
        if (s >= length)
            return T;
        c[a] += step[a];
        next[a] += delta[a];
        if (c[a] < 0 || c[a] >= v.n[a])
            return T; // left the lattice (vacuum)
    }
    return T;
}
M1PN_HD3 inline void lattice3d_sort(double *b, int n) { // insertion sort (n <= lattice3d_max_breaks)
    for (int i = 1; i < n; ++i) {
        const double key = b[i];
        int j = i - 1;
        while (j >= 0 && b[j] > key)
            b[j + 1] = b[j], --j;
        b[j + 1] = key;
    }
}
// Parameters w in (lo, hi) at which the ray from x through x + D0 + w Dd (a line in a face plane) passes through a
// lattice edge line (axis-parallel line through integer coordinates) inside the lattice and before its exit from S.
// Appends to out[n]. Dd is a unit coordinate vector.
M1PN_HD3 inline void lattice3d_edge_breaks(const Lattice3dView &v, const double x[3], const double D0[3],
                                           const double Dd[3], double lo, double hi, double *out, int &n) {
    for (int g = 0; g < 3; ++g) {
        const int p1 = (g + 1) % 3, p2 = (g + 2) % 3;
        for (int I = 0; I <= v.n[p1]; ++I)
            for (int J = 0; J <= v.n[p2]; ++J) {
                // lambda (D0[p] + w Dd[p]) = target[p] - x[p] for p = p1, p2.
                const double t1 = I - x[p1], t2 = J - x[p2];
                const double A = Dd[p1] * t2 - Dd[p2] * t1, B = D0[p2] * t1 - D0[p1] * t2;
                if (fabs(A) < 1e-300)
                    continue;
                const double w = B / A;
                if (!(w > lo && w < hi))
                    continue;
                const double dp1 = D0[p1] + w * Dd[p1], dp2 = D0[p2] + w * Dd[p2];
                const double dm = fabs(dp1) > fabs(dp2) ? dp1 : dp2, tm = fabs(dp1) > fabs(dp2) ? t1 : t2;
                if (fabs(dm) < 1e-300)
                    continue;
                const double lambda = tm / dm; // fraction of the way to the face point
                if (lambda <= 0)
                    continue;
                const double hg = x[g] + lambda * (D0[g] + w * Dd[g]);
                if (hg < 0 || hg > v.n[g])
                    continue;
                if (lambda > 1) { // behind the entry face: only up to the exit from S (exit face changes)
                    double dd[3] = {D0[0] + w * Dd[0], D0[1] + w * Dd[1], D0[2] + w * Dd[2]};
                    const double r = sqrt(dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]);
                    dd[0] /= r, dd[1] /= r, dd[2] /= r;
                    double si, so;
                    if (!lattice3d_hit(v, x, dd, si, so) || lambda * r > so * (1 + 1e-12))
                        continue;
                }
                if (n < lattice3d_max_breaks)
                    out[n++] = w;
            }
    }
}
// Calls emit(weight, tau_in, s_in, s_out) for every quadrature ray of x; sum_rays weight = solid angle of S / (4 pi)
// times q0, i.e. rho_u(x, t) = sum weight exp(-tau_in) (1 - exp(-sigma_S L)) / sigma_S.
template <class Emit> M1PN_HD3 void lattice3d_rays(const Lattice3dView &v, const double x[3], Emit &&emit) {
    const double pi = 3.14159265358979323846;
    const double scale = v.q0 / (4 * pi);
    bool inside = true;
    for (int a = 0; a < 3; ++a)
        inside = inside && x[a] >= v.s[a] && x[a] <= v.s[a] + 1;
    double bu[lattice3d_max_breaks], bv[lattice3d_max_breaks];
    for (int a = 0; a < 3; ++a)
        for (int side = 0; side < 2; ++side) {
            const double c = v.s[a] + side; // face plane x_a = c
            const double h = fabs(x[a] - c);
            if (h < 1e-14)
                continue;
            if (!inside && (side == 0 ? x[a] >= c : x[a] <= c))
                continue; // not visible
            const int b = (a + 1) % 3, e = (a + 2) % 3; // face coordinates u = y_b, v = y_e
            const double u0 = v.s[b], u1 = v.s[b] + 1., v0 = v.s[e], v1 = v.s[e] + 1.;
            // Ray of the face point y(u, w): direction, length, entry/exit and depth.
            auto ray = [&](double u, double w, double &weight, double &tau_in, double &s_in, double &s_out) -> bool {
                double d[3];
                d[a] = c - x[a], d[b] = u - x[b], d[e] = w - x[e];
                const double r = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                d[0] /= r, d[1] /= r, d[2] /= r;
                weight = h / (r * r * r);
                if (inside) {
                    s_in = 0, s_out = r, tau_in = 0;
                    return true;
                }
                double si, so;
                if (!lattice3d_hit(v, x, d, si, so))
                    return false;
                s_in = r; // the visible face is the entry face
                s_out = so;
                tau_in = lattice3d_depth(v, x, d, r);
                return tau_in <= v.tau_max;
            };
            // Outer breaks: ends and the projections (from x onto the face plane) of the lattice vertices that lie
            // between x and the plane.
            int nu = 0;
            bu[nu++] = u0, bu[nu++] = u1;
            if (!inside)
                for (int i = 0; i <= v.n[0]; ++i)
                    for (int j = 0; j <= v.n[1]; ++j)
                        for (int k = 0; k <= v.n[2]; ++k) {
                            const double P[3] = {double(i), double(j), double(k)};
                            const double da = P[a] - x[a], dc = c - x[a];
                            // Vertices strictly between x and the plane (cell sequence before S) and the vertices
                            // of S itself (exit face).
                            const bool of_S = (i == v.s[0] || i == v.s[0] + 1) && (j == v.s[1] || j == v.s[1] + 1) &&
                                              (k == v.s[2] || k == v.s[2] + 1);
                            if (fabs(da) < 1e-300 || da / dc <= 0 || (da / dc >= 1 && !of_S))
                                continue;
                            const double f = dc / da;
                            // Edges along e project to lines u = const that can cross the face even when their
                            // end points project outside it in v: every u in range is a break.
                            const double pu = x[b] + f * (P[b] - x[b]);
                            if (pu > u0 && pu < u1 && nu < lattice3d_max_breaks)
                                bu[nu++] = pu;
                        }
            // Where a projected edge line crosses the face boundary v = v0 or v = v1, the inner integral has a jump
            // in its second derivative in u: break there too.
            if (!inside)
                for (int side_v = 0; side_v < 2; ++side_v) {
                    double D0[3], Du[3] = {0, 0, 0};
                    D0[a] = c - x[a], D0[b] = -x[b], D0[e] = (side_v ? v1 : v0) - x[e];
                    Du[b] = 1;
                    lattice3d_edge_breaks(v, x, D0, Du, u0, u1, bu, nu);
                }
            lattice3d_sort(bu, nu);
            const int up = inside ? v.inside_panels : v.outside_panels;
            for (int iu = 0; iu + 1 < nu; ++iu)
                for (int pu_ = 0; pu_ < up; ++pu_) {
                    const double ua = bu[iu] + (bu[iu + 1] - bu[iu]) * pu_ / up;
                    const double ub = bu[iu] + (bu[iu + 1] - bu[iu]) * (pu_ + 1) / up;
                    if (ub - ua < 1e-14)
                        continue;
                    for (int qu = 0; qu < v.outer_n; ++qu) {
                        const double u = ua + (ub - ua) * v.outer_x[qu], wu = (ub - ua) * v.outer_w[qu];
                        // Inner breaks on the line y(u, w), w in [v0, v1]: rays through lattice edge lines.
                        int nv = 0;
                        bv[nv++] = v0, bv[nv++] = v1;
                        if (!inside) {
                            // Points of the line: Y(w) = x + D0 + w De with D0, De in the plane x_a = c.
                            double D0[3], De[3] = {0, 0, 0};
                            D0[a] = c - x[a], D0[b] = u - x[b], D0[e] = -x[e];
                            De[e] = 1;
                            lattice3d_edge_breaks(v, x, D0, De, v0, v1, bv, nv);
                        }
                        lattice3d_sort(bv, nv);
                        const int vp = inside ? v.inside_panels : v.outside_panels;
                        for (int iv = 0; iv + 1 < nv; ++iv)
                            for (int pv_ = 0; pv_ < vp; ++pv_) {
                                const double va = bv[iv] + (bv[iv + 1] - bv[iv]) * pv_ / vp;
                                const double vb = bv[iv] + (bv[iv + 1] - bv[iv]) * (pv_ + 1) / vp;
                                if (vb - va < 1e-14)
                                    continue;
                                for (int qv = 0; qv < v.inner_n; ++qv) {
                                    const double w = va + (vb - va) * v.inner_x[qv], ww = (vb - va) * v.inner_w[qv];
                                    double weight, tau_in, s_in, s_out;
                                    if (ray(u, w, weight, tau_in, s_in, s_out))
                                        emit(scale * wu * ww * weight, tau_in, s_in, s_out);
                                }
                            }
                    }
                }
        }
}
// rho_u(x, t) by direct summation over the rays (sigma_S > 0).
M1PN_HD3 inline double lattice3d_density(const Lattice3dView &v, const double x[3], double t) {
    if (t <= 0)
        return 0;
    const double sS = lattice3d_sigma(v, v.s[0], v.s[1], v.s[2]);
    double sum = 0;
    lattice3d_rays(v, x, [&](double w, double tau, double s_in, double s_out) {
        const double L = fmin(s_out, t) - s_in;
        if (L > 0)
            sum += w * exp(-tau) * -expm1(-sS * L) / sS;
    });
    return sum;
}
// Row rho_u(x, m dt), m = 0..M, of the time table: every ray adds its exact time dependence through three
// difference arrays (P: constant part, E: part times exp(-sigma_S t), C: saturated part after s_out); work of size
// 3 (M + 2), row of size M + 1. One pass over the rays for all times.
M1PN_HD3 inline void lattice3d_row(const Lattice3dView &v, const double x[3], double dt, int M, double *work,
                                   double *row) {
    const double sS = lattice3d_sigma(v, v.s[0], v.s[1], v.s[2]);
    double *P = work, *E = work + (M + 2), *C = work + 2 * (M + 2);
    for (int m = 0; m < 3 * (M + 2); ++m)
        work[m] = 0;
    lattice3d_rays(v, x, [&](double w, double tau, double s_in, double s_out) {
        const int m_in = int(floor(s_in / dt)) + 1; // first m with m dt > s_in
        if (m_in > M)
            return;
        const double A = w * exp(-tau) / sS, Ae = A * exp(sS * s_in);
        int m_out = int(ceil(s_out / dt)); // first m with m dt >= s_out
        if (m_out < m_in)
            m_out = m_in;
        P[m_in] += A, E[m_in] += Ae;
        if (m_out <= M) {
            P[m_out] -= A, E[m_out] -= Ae;
            C[m_out] += A * -expm1(-sS * (s_out - s_in));
        }
    });
    double p = 0, e = 0, cc = 0;
    row[0] = 0;
    for (int m = 1; m <= M; ++m) {
        p += P[m], e += E[m], cc += C[m];
        const double val = p - exp(-sS * m * dt) * e + cc;
        row[m] = val > 0 ? val : 0;
    }
}
} // namespace uncollided
} // namespace m1pn
