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
#include <cstddef>
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
    // Polar rule (lattice3d_row_polar): polar coordinates about the foot point of x on each face, radial variable =
    // distance s, adaptive Chebyshev panels in s and phi. polar_tol: relative Chebyshev tail, polar_floor: absolute.
    bool polar = false;
    double polar_tol = 1e-4, polar_floor = 1e-9, polar_max_dphi = 0.4, polar_max_ds = 0.25, polar_near_ratio = 2;
    int polar_depth = 10;
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
        if (s >= length || T > v.tau_max)
            return T;
        c[a] += step[a];
        next[a] += delta[a];
        if (c[a] < 0 || c[a] >= v.n[a])
            return T; // left the lattice (vacuum)
    }
    return T;
}
// Fast exit distance from S for a ray p + s d (|d| = 1) known to enter S on a visible face.
M1PN_HD3 inline double lattice3d_exit_S(const Lattice3dView &v, const double p[3], const double d[3]) {
    double hi = 1e300;
    for (int a = 0; a < 3; ++a) {
        if (d[a] > 1e-300)
            hi = fmin(hi, (v.s[a] + 1. - p[a]) / d[a]);
        else if (d[a] < -1e-300)
            hi = fmin(hi, (v.s[a] - p[a]) / d[a]);
    }
    return hi;
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
// lattice edge line (axis-parallel line through integer coordinates) where sigma_t jumps before S, or an edge of S
// (exit face changes). Appends to out[n].
M1PN_HD3 inline void lattice3d_edge_breaks(const Lattice3dView &v, const double x[3], const double D0[3],
                                           const double Dd[3], double lo, double hi, double *out, int &n,
                                           int max_breaks = lattice3d_max_breaks) {
    int imin[3], imax[3];
    for (int a = 0; a < 3; ++a) {
        const double p_lo = x[a] + D0[a] + lo * Dd[a];
        const double p_hi = x[a] + D0[a] + hi * Dd[a];
        const double mn = fmin(fmin(x[a], double(v.s[a])), fmin(p_lo, p_hi));
        const double mx = fmax(fmax(x[a], double(v.s[a] + 1)), fmax(p_lo, p_hi));
        const int lo_i = int(floor(mn)), hi_i = int(ceil(mx));
        imin[a] = lo_i < 0 ? 0 : lo_i;
        imax[a] = hi_i > v.n[a] ? v.n[a] : hi_i;
    }
    for (int g = 0; g < 3; ++g) {
        const int p1 = (g + 1) % 3, p2 = (g + 2) % 3;
        for (int I = imin[p1]; I <= imax[p1]; ++I) {
            const double t1 = I - x[p1];
            const double Dd_p2_t1 = Dd[p2] * t1, D0_p2_t1 = D0[p2] * t1;
            for (int J = imin[p2]; J <= imax[p2]; ++J) {
                // lambda (D0[p] + w Dd[p]) = target[p] - x[p] for p = p1, p2.
                const double t2 = J - x[p2];
                const double A = Dd[p1] * t2 - Dd_p2_t1;
                if (fabs(A) < 1e-300)
                    continue;
                const double w = (D0_p2_t1 - D0[p1] * t2) / A;
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
                if (lambda > 1) { // behind the entry face: only edges of S can change the exit face
                    if ((I != v.s[p1] && I != v.s[p1] + 1) || (J != v.s[p2] && J != v.s[p2] + 1))
                        continue;
                    if (hg < v.s[g] - 1e-12 || hg > v.s[g] + 1 + 1e-12)
                        continue;
                } else {
                    // Skip edges where all adjacent cells at hg share the same sigma_t (no kink in tau_in).
                    int cg = int(floor(hg));
                    if (cg < 0)
                        cg = 0;
                    if (cg >= v.n[g])
                        cg = v.n[g] - 1;
                    bool diff = fabs(hg - floor(hg + 0.5)) < 1e-12;
                    double s0 = -1;
                    for (int d1 = -1; d1 <= 0 && !diff; ++d1) {
                        const int c1 = I + d1;
                        if (c1 < 0 || c1 >= v.n[p1])
                            continue;
                        for (int d2 = -1; d2 <= 0; ++d2) {
                            const int c2 = J + d2;
                            if (c2 < 0 || c2 >= v.n[p2])
                                continue;
                            int idx[3];
                            idx[g] = cg, idx[p1] = c1, idx[p2] = c2;
                            const double sg = lattice3d_sigma(v, idx[0], idx[1], idx[2]);
                            if (s0 < 0)
                                s0 = sg;
                            else if (sg != s0) {
                                diff = true;
                                break;
                            }
                        }
                    }
                    if (!diff)
                        continue;
                }
                if (n < max_breaks)
                    out[n++] = w;
            }
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
M1PN_HD3 inline void lattice3d_row_cartesian(const Lattice3dView &v, const double x[3], double dt, int M,
                                             double *work, double *row) {
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

// ---- Polar rule ------------------------------------------------------------------------------------------------------
// Face F in the plane x_a = c at distance h, foot point (x_b, x_e). y = foot + rho (cos phi, sin phi) and
// s = sqrt(h^2 + rho^2) give dOmega = h / s^3 dA = (h / s^2) ds dphi (bounded: the near-singular kernel h / r^3 of the
// Cartesian rule disappears). Outside S, s is the entry distance s_in, so the front s_in = t lies at constant s for every
// t: with A(s, phi) = q0/(4 pi) h/s^2 e^{-tau_in}/sigma_S and Chebyshev antiderivatives on each radial panel,
//   rho_u(t) = int dphi [ G0(t) - e^{-sigma_S t} G1(t) ] + (saturation after s_out, node-wise),
//   G0(t) = int_{s<t} A ds, G1(t) = int_{s<t} A e^{sigma_S s} ds,
// exact in s at every table time. Inside S (s = s_out, s_in = tau = 0):
//   rho_u(t) = int dphi [ G2(t) + (1 - e^{-sigma_S t}) (I3 - G3(t)) ], G2 = int_{s<t} A (1 - e^{-sigma_S s}), G3 = int A.
// Radial breaks: lattice edge crossings along the radial line (kinks of tau_in and of the exit face), geometric breaks
// rho = h r^k near the foot; phi breaks: face corners, projected lattice vertices, projected edge lines on the face
// boundary. Panels and phi pieces are split adaptively until the last two Chebyshev coefficients (of A in s, of the
// radial integral Q(phi) in phi) fall below polar_tol * max + polar_floor; a phi piece is accumulated in a temporary
// row and added only when accepted (depth-first stack, one temporary). work: lattice3d_work_size doubles.
constexpr int lattice3d_cheb_n = 8;
constexpr int lattice3d_stack = 48;
constexpr int lattice3d_polar_max_breaks = 192;
constexpr int lattice3d_polar_max_rbreaks = 96;
struct Lattice3dCheb {
    M1PN_HD3 static double x(int j) {
        constexpr double t[8] = {0.98078528040323043,  0.83146961230254524,  0.55557023301960229,  0.19509032201612833,
                                 -0.19509032201612819, -0.55557023301960196, -0.83146961230254535, -0.98078528040323043};
        return t[j];
    }
    M1PN_HD3 static double wq(int j) {
        constexpr double t[8] = {0.066982945698589813, 0.22298793301457881, 0.32415251906452436, 0.38587660222230702,
                                 0.38587660222230702,  0.32415251906452441, 0.22298793301457878, 0.06698294569858998};
        return t[j];
    }
    M1PN_HD3 static double cosjk(int k, int j) {
        constexpr double t[64] = {
            1, 1, 1, 1, 1, 1, 1, 1,
            0.98078528040323043, 0.83146961230254524, 0.55557023301960229, 0.19509032201612833,
            -0.19509032201612819, -0.55557023301960196, -0.83146961230254535, -0.98078528040323043,
            0.92387953251128674, 0.38268343236508984, -0.38268343236508973, -0.92387953251128674,
            -0.92387953251128685, -0.38268343236509034, 0.38268343236509, 0.92387953251128652,
            0.83146961230254524, -0.19509032201612819, -0.98078528040323043, -0.55557023301960218,
            0.55557023301960184, 0.98078528040323043, 0.19509032201612878, -0.83146961230254512,
            0.70710678118654757, -0.70710678118654746, -0.70710678118654768, 0.70710678118654735,
            0.70710678118654768, -0.70710678118654668, -0.70710678118654713, 0.70710678118654657,
            0.55557023301960229, -0.98078528040323043, 0.1950903220161283, 0.83146961230254546,
            -0.83146961230254512, -0.19509032201612803, 0.98078528040323065, -0.55557023301960151,
            0.38268343236508984, -0.92387953251128685, 0.92387953251128652, -0.38268343236508989,
            -0.38268343236509056, 0.92387953251128674, -0.92387953251128641, 0.38268343236508956,
            0.19509032201612833, -0.55557023301960218, 0.83146961230254546, -0.98078528040323065,
            0.98078528040323043, -0.83146961230254501, 0.55557023301960151, -0.19509032201612858};
        return t[k * 8 + j];
    }
    // Chebyshev coefficients c_0..c_{n-1} of the interpolant (f = c_0/2 + sum c_k T_k), even/odd folded.
    M1PN_HD3 static void coeffs(const double *f, double *c) {
        const double e0 = f[0] + f[7], e1 = f[1] + f[6], e2 = f[2] + f[5], e3 = f[3] + f[4];
        const double o0 = f[0] - f[7], o1 = f[1] - f[6], o2 = f[2] - f[5], o3 = f[3] - f[4];
        c[0] = 0.25 * (e0 + e1 + e2 + e3);
        c[2] = 0.25 * ((e0 - e3) * 0.92387953251128674 + (e1 - e2) * 0.38268343236508984);
        c[4] = 0.25 * 0.70710678118654757 * (e0 - e1 - e2 + e3);
        c[6] = 0.25 * ((e0 - e3) * 0.38268343236508984 + (e2 - e1) * 0.92387953251128674);
        c[1] = 0.25 * (o0 * 0.98078528040323043 + o1 * 0.83146961230254524 + o2 * 0.55557023301960229 +
                       o3 * 0.19509032201612833);
        c[3] = 0.25 * (o0 * 0.83146961230254524 - o1 * 0.19509032201612833 - o2 * 0.98078528040323043 -
                       o3 * 0.55557023301960229);
        c[5] = 0.25 * (o0 * 0.55557023301960229 - o1 * 0.98078528040323043 + o2 * 0.19509032201612833 +
                       o3 * 0.83146961230254524);
        c[7] = 0.25 * (o0 * 0.19509032201612833 - o1 * 0.55557023301960229 + o2 * 0.83146961230254524 -
                       o3 * 0.98078528040323043);
    }
    M1PN_HD3 static double tail(const double *f) {
        const double e0 = f[0] + f[7], e1 = f[1] + f[6], e2 = f[2] + f[5], e3 = f[3] + f[4];
        const double o0 = f[0] - f[7], o1 = f[1] - f[6], o2 = f[2] - f[5], o3 = f[3] - f[4];
        const double c6 = (e0 - e3) * 0.38268343236508984 + (e2 - e1) * 0.92387953251128674;
        const double c7 = o0 * 0.19509032201612833 - o1 * 0.55557023301960229 + o2 * 0.83146961230254524 -
                          o3 * 0.98078528040323043;
        return 0.25 * (fabs(c6) + fabs(c7));
    }
    // Antiderivative (C_0..C_n) in x of the interpolant, zero at x = -1.
    M1PN_HD3 static void antider(const double *f, double *C) {
        const int n = lattice3d_cheb_n;
        double c[lattice3d_cheb_n + 2];
        coeffs(f, c);
        c[n] = c[n + 1] = 0;
        double s = 0;
        for (int k = 1; k <= n; ++k) {
            C[k] = (c[k - 1] - c[k + 1]) / (2. * k);
            s += (k & 1) ? -C[k] : C[k];
        }
        C[0] = -s;
    }
    M1PN_HD3 static double eval(const double *C, double x) { // sum_{k=0}^{n} C_k T_k(x)
        double b1 = 0, b2 = 0;
        for (int k = lattice3d_cheb_n; k >= 1; --k) {
            const double b0 = C[k] + 2 * x * b1 - b2;
            b2 = b1, b1 = b0;
        }
        return C[0] + x * b1 - b2;
    }
};
M1PN_HD3 inline std::size_t lattice3d_work_size(const Lattice3dView &v, int M) {
    return (v.polar ? 10 : 3) * std::size_t(M + 2);
}
M1PN_HD3 inline double lattice3d_angle(double du, double dv, double ref) {
    const double pi = 3.14159265358979323846;
    double p = atan2(dv, du) - ref;
    while (p <= -pi)
        p += 2 * pi;
    while (p > pi)
        p -= 2 * pi;
    return p;
}
// Accumulators: 0 P (constant, difference), 1 E (times e^{-sS t}, difference), 2 C (saturated, difference),
// 3 D (constant, direct), 4 D1 (times e^{-sS t}, direct); each M + 2 long.
M1PN_HD3 inline void lattice3d_row_polar(const Lattice3dView &v, const double x[3], double dt, int M, double *work,
                                         double *row) {
    const double pi = 3.14159265358979323846;
    const int n = lattice3d_cheb_n;
    const std::size_t L = std::size_t(M + 2);
    const double scale = v.q0 / (4 * pi);
    const double sS = lattice3d_sigma(v, v.s[0], v.s[1], v.s[2]);
    const double Tmax = M * dt;
    double *Acc = work, *Tmp = work + 5 * L;
    for (std::size_t m = 0; m < 10 * L; ++m)
        work[m] = 0;
    bool inside = true;
    for (int a = 0; a < 3; ++a)
        inside = inside && x[a] >= v.s[a] && x[a] <= v.s[a] + 1;
    double brk[lattice3d_polar_max_breaks];
    for (int a = 0; a < 3; ++a)
        for (int side = 0; side < 2; ++side) {
            const double c = v.s[a] + side, h = fabs(x[a] - c);
            if (h < 1e-14)
                continue;
            if (!inside && (side == 0 ? x[a] >= c : x[a] <= c))
                continue;
            const int b = (a + 1) % 3, e = (a + 2) % 3;
            const double u0 = v.s[b], u1 = v.s[b] + 1., v0 = v.s[e], v1 = v.s[e] + 1.;
            const double Fu = x[b], Fv = x[e];
            // Horizon culling: skip the face if its nearest point is beyond M * dt.
            const double du_min = fmax(fmax(u0 - Fu, 0.), Fu - u1);
            const double dv_min = fmax(fmax(v0 - Fv, 0.), Fv - v1);
            if (!inside && h * h + du_min * du_min + dv_min * dv_min >= Tmax * Tmax)
                continue;
            const double r_horizon = !inside ? sqrt(fmax(0., Tmax * Tmax - h * h)) : 1e300;
            const bool foot_in = Fu > u0 && Fu < u1 && Fv > v0 && Fv < v1;
            const double ref = foot_in ? 0. : atan2(v0 + .5 - Fv, u0 + .5 - Fu);
            // phi breaks
            int np = 0;
            const double cu[4] = {u0, u1, u1, u0}, cv[4] = {v0, v0, v1, v1};
            for (int k = 0; k < 4; ++k)
                brk[np++] = lattice3d_angle(cu[k] - Fu, cv[k] - Fv, ref);
            double plo = brk[0], phi_hi = brk[0];
            for (int k = 1; k < 4; ++k)
                plo = fmin(plo, brk[k]), phi_hi = fmax(phi_hi, brk[k]);
            if (foot_in) {
                plo = -pi, phi_hi = pi;
                brk[np++] = -pi, brk[np++] = pi;
            }
            if (!inside) {
                int imin[3], imax[3];
                for (int q = 0; q < 3; ++q) {
                    const int lo_i = int(floor(fmin(x[q], double(v.s[q])))),
                              hi_i = int(ceil(fmax(x[q], double(v.s[q] + 1))));
                    imin[q] = lo_i < 0 ? 0 : lo_i;
                    imax[q] = hi_i > v.n[q] ? v.n[q] : hi_i;
                }
                const double dc = c - x[a];
                for (int i = imin[0]; i <= imax[0]; ++i)
                    for (int j = imin[1]; j <= imax[1]; ++j)
                        for (int k = imin[2]; k <= imax[2]; ++k) {
                            const double P[3] = {double(i), double(j), double(k)};
                            const double da = P[a] - x[a];
                            const bool of_S = (i == v.s[0] || i == v.s[0] + 1) && (j == v.s[1] || j == v.s[1] + 1) &&
                                              (k == v.s[2] || k == v.s[2] + 1);
                            if (fabs(da) < 1e-300 || da / dc <= 0 || (da / dc >= 1 && !of_S))
                                continue;
                            const double f = dc / da, pu = x[b] + f * (P[b] - x[b]), pv = x[e] + f * (P[e] - x[e]);
                            if (pu > u0 && pu < u1 && pv > v0 && pv < v1 && np < lattice3d_polar_max_breaks)
                                brk[np++] = lattice3d_angle(pu - Fu, pv - Fv, ref);
                        }
                double tmp[lattice3d_polar_max_breaks];
                for (int sv = 0; sv < 2; ++sv) { // projected edge lines crossing v = v0 / v1 (parameter u)
                    double D0[3], Dd[3] = {0, 0, 0};
                    D0[a] = c - x[a], D0[b] = -x[b], D0[e] = (sv ? v1 : v0) - x[e];
                    Dd[b] = 1;
                    int nt = 0;
                    lattice3d_edge_breaks(v, x, D0, Dd, u0, u1, tmp, nt, lattice3d_polar_max_breaks);
                    for (int q = 0; q < nt && np < lattice3d_polar_max_breaks; ++q)
                        brk[np++] = lattice3d_angle(tmp[q] - Fu, (sv ? v1 : v0) - Fv, ref);
                }
                for (int su = 0; su < 2; ++su) { // ... crossing u = u0 / u1 (parameter v)
                    double D0[3], Dd[3] = {0, 0, 0};
                    D0[a] = c - x[a], D0[b] = (su ? u1 : u0) - x[b], D0[e] = -x[e];
                    Dd[e] = 1;
                    int nt = 0;
                    lattice3d_edge_breaks(v, x, D0, Dd, v0, v1, tmp, nt, lattice3d_polar_max_breaks);
                    for (int q = 0; q < nt && np < lattice3d_polar_max_breaks; ++q)
                        brk[np++] = lattice3d_angle((su ? u1 : u0) - Fu, tmp[q] - Fv, ref);
                }
            }
            lattice3d_sort(brk, np);
            int npu = 0;
            for (int k = 0; k < np; ++k)
                if (npu == 0 || brk[k] - brk[npu - 1] > 1e-13)
                    brk[npu++] = brk[k];
            np = npu;
            // radial integration along one phi; writes weight * contributions into Tmp, returns Q = int A ds (or I3)
            auto radial = [&](double phi, double weight, int &mlo, int &mhi) -> double {
                const double du = cos(phi), dv = sin(phi);
                double rlo = 0, rhi = r_horizon;
                bool empty = false;
                auto slab = [&](double F, double d, double lo, double hi) {
                    if (fabs(d) < 1e-300) {
                        if (F < lo || F > hi)
                            empty = true;
                        return;
                    }
                    double t0 = (lo - F) / d, t1 = (hi - F) / d;
                    if (t0 > t1) {
                        const double w = t0;
                        t0 = t1, t1 = w;
                    }
                    rlo = fmax(rlo, t0), rhi = fmin(rhi, t1);
                };
                slab(Fu, du, u0, u1), slab(Fv, dv, v0, v1);
                if (empty || !(rhi > rlo + 1e-14))
                    return 0.;
                double rb[lattice3d_polar_max_rbreaks];
                int nr = 0;
                rb[nr++] = rlo, rb[nr++] = rhi;
                if (!inside) {
                    double D0[3], Dd[3];
                    D0[a] = c - x[a], D0[b] = 0, D0[e] = 0;
                    Dd[a] = 0, Dd[b] = du, Dd[e] = dv;
                    lattice3d_edge_breaks(v, x, D0, Dd, rlo, rhi, rb, nr, lattice3d_polar_max_rbreaks);
                }
                for (double r = h; r < rhi && nr < lattice3d_polar_max_rbreaks; r *= v.polar_near_ratio)
                    if (r > rlo)
                        rb[nr++] = r;
                lattice3d_sort(rb, nr);
                double Q = 0;
                double st_a[lattice3d_stack], st_z[lattice3d_stack];
                int st_d[lattice3d_stack];
                double sprev = -1;
                for (int ir = 0; ir < nr; ++ir) {
                    const double s_next = sqrt(h * h + rb[ir] * rb[ir]);
                    if (sprev >= 0 && s_next - sprev > 1e-14) {
                        // initial panels of length <= polar_max_ds, then adaptive bisection (stack, depth-first)
                        const int nsub = int(ceil((s_next - sprev) / v.polar_max_ds)) > 1 ?
                                             int(ceil((s_next - sprev) / v.polar_max_ds)) : 1;
                        for (int ks = nsub - 1; ks >= 0; --ks) {
                            int top = 0;
                            st_a[top] = sprev + (s_next - sprev) * ks / nsub;
                            st_z[top] = sprev + (s_next - sprev) * (ks + 1) / nsub;
                            st_d[top++] = 0;
                            while (top > 0) {
                                --top;
                                const double sa = st_a[top], sz = st_z[top];
                                const int depth = st_d[top];
                                const double mid = .5 * (sa + sz), half = .5 * (sz - sa);
                                double A[lattice3d_cheb_n], so[lattice3d_cheb_n], sj[lattice3d_cheb_n];
                                bool any = false;
                                double amax = 0;
                                for (int j = 0; j < n; ++j) {
                                    const double s = mid + half * Lattice3dCheb::x(j);
                                    sj[j] = s;
                                    const double inv_s = 1.0 / s;
                                    const double r = sqrt(fmax(0., s * s - h * h));
                                    double d[3];
                                    d[a] = (c - x[a]) * inv_s, d[b] = r * du * inv_s, d[e] = r * dv * inv_s;
                                    if (inside) {
                                        A[j] = scale * h * inv_s * inv_s / sS, so[j] = s, any = true;
                                    } else {
                                        const double tau = lattice3d_depth(v, x, d, s);
                                        if (tau <= v.tau_max) {
                                            A[j] = scale * h * inv_s * inv_s * exp(-tau) / sS;
                                            so[j] = lattice3d_exit_S(v, x, d);
                                            any = true;
                                        } else {
                                            A[j] = 0, so[j] = s;
                                        }
                                    }
                                    amax = fmax(amax, A[j]);
                                }
                                if (!any)
                                    continue;
                                if (depth < v.polar_depth && top + 2 <= lattice3d_stack &&
                                    Lattice3dCheb::tail(A) > v.polar_tol * amax + v.polar_floor) {
                                    st_a[top] = mid, st_z[top] = sz, st_d[top++] = depth + 1;
                                    st_a[top] = sa, st_z[top] = mid, st_d[top++] = depth + 1;
                                    continue;
                                }
                                const int m_a = int(floor(sa / dt)) + 1, m_b = int(ceil(sz / dt));
                                const int m_lo = m_a < 1 ? 1 : m_a;
                                const int m_hi_in = m_b < M + 1 ? m_b : M + 1;
                                const double wh = weight * half;
                                if (!inside) {
                                    double f1[lattice3d_cheb_n], esj[lattice3d_cheb_n];
                                    double sum0 = 0, sum1 = 0;
                                    for (int j = 0; j < n; ++j) {
                                        const double e_j = exp(sS * sj[j]);
                                        esj[j] = e_j;
                                        f1[j] = A[j] * e_j;
                                        const double wq = Lattice3dCheb::wq(j);
                                        sum0 += wq * A[j];
                                        sum1 += wq * f1[j];
                                    }
                                    const double I0 = half * sum0, I1 = half * sum1;
                                    Q += I0;
                                    if (m_lo < m_hi_in) {
                                        double ca[lattice3d_cheb_n + 1], cb[lattice3d_cheb_n + 1];
                                        Lattice3dCheb::antider(A, ca), Lattice3dCheb::antider(f1, cb);
                                        const double inv_half = 1.0 / half;
                                        for (int m = m_lo; m < m_hi_in; ++m) {
                                            const double xx = (m * dt - mid) * inv_half;
                                            Tmp[3 * L + m] += wh * Lattice3dCheb::eval(ca, xx);
                                            Tmp[4 * L + m] += wh * Lattice3dCheb::eval(cb, xx);
                                        }
                                        mlo = m_lo < mlo ? m_lo : mlo;
                                        mhi = (m_hi_in - 1) > mhi ? (m_hi_in - 1) : mhi;
                                    }
                                    if (m_b <= M) {
                                        Tmp[m_b] += weight * I0;
                                        Tmp[L + m_b] += weight * I1;
                                        mlo = m_b < mlo ? m_b : mlo, mhi = m_b > mhi ? m_b : mhi;
                                    }
                                    for (int j = 0; j < n; ++j) { // saturation after s_out (node-wise)
                                        if (A[j] == 0)
                                            continue;
                                        int m_out = int(ceil(so[j] / dt));
                                        m_out = m_out < m_lo ? m_lo : m_out;
                                        if (m_out > M)
                                            continue;
                                        const double wj = wh * Lattice3dCheb::wq(j) * A[j];
                                        Tmp[L + m_out] -= wj * esj[j];
                                        Tmp[2 * L + m_out] -= wj * exp(-sS * (so[j] - sj[j]));
                                        mlo = m_out < mlo ? m_out : mlo, mhi = m_out > mhi ? m_out : mhi;
                                    }
                                } else {
                                    double f0[lattice3d_cheb_n];
                                    double sum0 = 0, sum1 = 0;
                                    for (int j = 0; j < n; ++j) {
                                        f0[j] = A[j] * -expm1(-sS * sj[j]);
                                        const double wq = Lattice3dCheb::wq(j);
                                        sum0 += wq * f0[j];
                                        sum1 += wq * A[j];
                                    }
                                    const double I2 = half * sum0, I3 = half * sum1;
                                    Q += I3;
                                    const int m_end = m_lo < M + 1 ? m_lo : M + 1;
                                    if (1 < m_end) { // t < sa: (1 - e^{-sS t}) I3
                                        Tmp[1] += weight * I3, Tmp[L + 1] += weight * I3;
                                        Tmp[m_end] -= weight * I3, Tmp[L + m_end] -= weight * I3;
                                        mlo = 1 < mlo ? 1 : mlo, mhi = m_end > mhi ? m_end : mhi;
                                    }
                                    if (m_lo < m_hi_in) {
                                        double ca[lattice3d_cheb_n + 1], cb[lattice3d_cheb_n + 1];
                                        Lattice3dCheb::antider(f0, ca), Lattice3dCheb::antider(A, cb);
                                        const double inv_half = 1.0 / half;
                                        for (int m = m_lo; m < m_hi_in; ++m) {
                                            const double t = m * dt, xx = (t - mid) * inv_half;
                                            Tmp[3 * L + m] += weight * (half * Lattice3dCheb::eval(ca, xx) +
                                                                        -expm1(-sS * t) *
                                                                            (I3 - half * Lattice3dCheb::eval(cb, xx)));
                                        }
                                        mlo = m_lo < mlo ? m_lo : mlo;
                                        mhi = (m_hi_in - 1) > mhi ? (m_hi_in - 1) : mhi;
                                    }
                                    if (m_b <= M) {
                                        Tmp[2 * L + m_b] += weight * I2;
                                        mlo = m_b < mlo ? m_b : mlo, mhi = m_b > mhi ? m_b : mhi;
                                    }
                                }
                            }
                        }
                    }
                    sprev = s_next;
                }
                return Q;
            };
            // phi pieces: initial pieces between breaks (width <= polar_max_dphi), adaptive bisection (stack)
            double pst_a[lattice3d_stack], pst_z[lattice3d_stack];
            int pst_d[lattice3d_stack];
            for (int ip = np - 2; ip >= 0; --ip) {
                const double ba = fmax(brk[ip], plo), bz = fmin(brk[ip + 1], phi_hi);
                if (!(bz - ba > 1e-13))
                    continue;
                const int sub = int(ceil((bz - ba) / v.polar_max_dphi)) > 1 ? int(ceil((bz - ba) / v.polar_max_dphi)) : 1;
                for (int sp = sub - 1; sp >= 0; --sp) {
                    int top = 0;
                    pst_a[top] = ba + (bz - ba) * sp / sub, pst_z[top] = ba + (bz - ba) * (sp + 1) / sub;
                    pst_d[top++] = 0;
                    while (top > 0) {
                        --top;
                        const double pa = pst_a[top], pz = pst_z[top];
                        const int depth = pst_d[top];
                        const double mid = .5 * (pa + pz), half = .5 * (pz - pa);
                        int mlo = M + 2, mhi = -1;
                        double Qn[lattice3d_cheb_n], qmax = 0;
                        for (int j = 0; j < n; ++j) {
                            Qn[j] = radial(mid + half * Lattice3dCheb::x(j) + ref, half * Lattice3dCheb::wq(j), mlo, mhi);
                            qmax = fmax(qmax, fabs(Qn[j]));
                        }
                        const bool split = depth < v.polar_depth && top + 2 <= lattice3d_stack &&
                                           Lattice3dCheb::tail(Qn) > v.polar_tol * qmax + v.polar_floor;
                        for (int q = 0; q < 5; ++q)
                            for (int m = mlo; m <= mhi; ++m) {
                                if (!split)
                                    Acc[q * L + m] += Tmp[q * L + m];
                                Tmp[q * L + m] = 0;
                            }
                        if (split) {
                            pst_a[top] = mid, pst_z[top] = pz, pst_d[top++] = depth + 1;
                            pst_a[top] = pa, pst_z[top] = mid, pst_d[top++] = depth + 1;
                        }
                    }
                }
            }
        }
    double p = 0, ee = 0, cc = 0;
    row[0] = 0;
    for (int m = 1; m <= M; ++m) {
        p += Acc[m], ee += Acc[L + m], cc += Acc[2 * L + m];
        const double val = p + Acc[3 * L + m] - exp(-sS * m * dt) * (ee + Acc[4 * L + m]) + cc;
        row[m] = val > 0 ? val : 0;
    }
}
// Row of the time table with the rule selected in the view (work: lattice3d_work_size(v, M) doubles).
M1PN_HD3 inline void lattice3d_row(const Lattice3dView &v, const double x[3], double dt, int M, double *work,
                                   double *row) {
    if (v.polar)
        lattice3d_row_polar(v, x, dt, M, work, row);
    else
        lattice3d_row_cartesian(v, x, dt, M, work, row);
}
// Cubic interpolation of a tabulated row row[0..M] at time t (same formula on host and device).
template <class T> M1PN_HD3 inline double lattice3d_eval_row(const T *row, int M, double dt, double t) {
    if (t <= 0)
        return 0;
    const double s = t / dt;
    int i = int(s);
    if (i > M - 2)
        i = M - 2;
    if (i < 0)
        i = 0;
    const double f = s - i;
    const double p1 = double(row[i]), p2 = double(row[i + 1]), p3 = double(row[i + 2]),
                 p0 = i > 0 ? double(row[i - 1]) : 2 * p1 - p2;
    const double v = p1 + .5 * f * (p2 - p0 + f * (2 * p0 - 5 * p1 + 4 * p2 - p3 + f * (3 * (p1 - p2) + p3 - p0)));
    return v > 0 ? v : 0;
}
// Time integral int_0^t row(s) ds by Simpson's rule on the cubic interpolant of row[0..M].
M1PN_HD3 inline double lattice3d_time_integral_row(const double *row, int M, double dt, double t) {
    if (t <= 0)
        return 0;
    const int n = int(floor(t / dt));
    double s = 0;
    for (int m = 0; m < n; ++m) {
        const double mid = lattice3d_eval_row(row, M, dt, (m + .5) * dt);
        s += dt / 6 * (row[m] + 4 * mid + row[m + 1]);
    }
    const double rest = t - n * dt;
    if (rest > 0) {
        const double a = row[n], b = lattice3d_eval_row(row, M, dt, t),
                     mid = lattice3d_eval_row(row, M, dt, n * dt + .5 * rest);
        s += rest / 6 * (a + 4 * mid + b);
    }
    return s;
}
} // namespace uncollided
} // namespace m1pn
