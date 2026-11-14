#include "positivity_projection.hpp"
#include <Eigen/Cholesky>

namespace m1pn {
namespace {
// One pass over the points at multiplier xi: F = sum w m z - U, H = sum_{z>0} w m m^T, phi = -1/2 sum w z^2 - xi.U,
// and min_k xi.m_k (Farkas test). Returns the number of active points.
struct Eval {
    Four F;
    Eigen::Matrix4d H;
    double phi, min_xm;
    int active;
};
// Weighted form (c, cinv given): objective 1/2 sum w c (z - y)^2, z = max(0, y - xi.m / c), H = sum_{z>0} (w / c) m m^T,
// phi = -1/2 sum w c z^2 - xi.U.
template <bool Weighted>
Eval evaluate_t(int q, const double *ox, const double *oy, const double *oz, const double *w, const double *y,
                const double *c, const double *cinv, const Four &U, const Four &xi) {
    double f0 = 0, f1 = 0, f2 = 0, f3 = 0, zz = 0, mn = std::numeric_limits<double>::infinity();
    double h00 = 0, h01 = 0, h02 = 0, h03 = 0, h11 = 0, h12 = 0, h13 = 0, h22 = 0, h23 = 0, h33 = 0;
    int active = 0;
    const double x0 = xi[0], x1 = xi[1], x2 = xi[2], x3 = xi[3];
#pragma omp simd reduction(+ : f0, f1, f2, f3, zz, h00, h01, h02, h03, h11, h12, h13, h22, h23, h33, active) \
    reduction(min : mn)
    for (int k = 0; k < q; ++k) {
        const double xm = x0 + x1 * ox[k] + x2 * oy[k] + x3 * oz[k];
        const double r = Weighted ? y[k] - xm * cinv[k] : y[k] - xm;
        const double pos = r > 0 ? 1. : 0.;
        const double z = pos * r, wz = w[k] * z, wa = Weighted ? w[k] * pos * cinv[k] : w[k] * pos;
        f0 += wz, f1 += wz * ox[k], f2 += wz * oy[k], f3 += wz * oz[k], zz += Weighted ? wz * z * c[k] : wz * z;
        h00 += wa, h01 += wa * ox[k], h02 += wa * oy[k], h03 += wa * oz[k];
        h11 += wa * ox[k] * ox[k], h12 += wa * ox[k] * oy[k], h13 += wa * ox[k] * oz[k];
        h22 += wa * oy[k] * oy[k], h23 += wa * oy[k] * oz[k], h33 += wa * oz[k] * oz[k];
        active += r > 0;
        mn = std::min(mn, xm);
    }
    Eval e;
    e.F = Four(f0, f1, f2, f3) - U;
    e.phi = -.5 * zz - xi.dot(U);
    e.min_xm = mn;
    e.active = active;
    e.H << h00, h01, h02, h03, h01, h11, h12, h13, h02, h12, h22, h23, h03, h13, h23, h33;
    return e;
}
Eval evaluate(int q, const double *ox, const double *oy, const double *oz, const double *w, const double *y,
              const Four &U, const Four &xi, const PositivityProjectionOptions &o) {
    return o.c ? evaluate_t<true>(q, ox, oy, oz, w, y, o.c, o.cinv, U, xi)
               : evaluate_t<false>(q, ox, oy, oz, w, y, nullptr, nullptr, U, xi);
}
} // namespace

PositivityProjectionResult project_positive(int q, const double *ox, const double *oy, const double *oz,
                                            const double *w, const double *y, const Four &U, double *z,
                                            const PositivityProjectionOptions &options, const Four &xi0) {
    PositivityProjectionResult out;
    double sy = 0, ymin = std::numeric_limits<double>::infinity();
    Four F0 = -U;
    for (int k = 0; k < q; ++k) {
        const double wy = w[k] * y[k];
        sy += std::abs(wy);
        ymin = std::min(ymin, y[k]);
        F0[0] += wy, F0[1] += wy * ox[k], F0[2] += wy * oy[k], F0[3] += wy * oz[k];
    }
    out.scale = sy + std::abs(U[0]);
    const double tol = options.tolerance * out.scale;
    auto finish = [&](const Four &xi, const Eval &e) {
        for (int k = 0; k < q; ++k) {
            const double xm = xi[0] + xi[1] * ox[k] + xi[2] * oy[k] + xi[3] * oz[k];
            const double r = options.c ? y[k] - xm * options.cinv[k] : y[k] - xm;
            z[k] = r > 0 ? r : 0.;
        }
        out.xi = xi;
        out.active = e.active;
        out.residual = out.scale > 0 ? e.F.cwiseAbs().maxCoeff() / out.scale : e.F.cwiseAbs().maxCoeff();
        return out;
    };
    // Vacuum: the only nonnegative measure with zero density is zero.
    if (!(U[0] > 0)) {
        for (int k = 0; k < q; ++k)
            z[k] = 0;
        out.converged = U[0] == 0 && U.tail<3>().isZero(0);
        out.residual = U.cwiseAbs().maxCoeff();
        return out;
    }
    // Identity: y > 0 is feasible up to the quadrature consistency of its moments, then z = y (xi = 0).
    if (ymin > 0 && F0.cwiseAbs().maxCoeff() <= tol) {
        for (int k = 0; k < q; ++k)
            z[k] = y[k];
        out.identity = out.converged = true;
        out.active = q;
        out.residual = F0.cwiseAbs().maxCoeff() / out.scale;
        return out;
    }
    Four xi = xi0.allFinite() ? xi0 : Four::Zero();
    Eval e = evaluate(q, ox, oy, oz, w, y, U, xi, options);
    for (int it = 0; it < options.max_iterations; ++it) {
        if (e.F.cwiseAbs().maxCoeff() <= tol) {
            out.converged = true;
            break;
        }
        // Farkas certificate of infeasibility: xi.m_k >= 0 for all k and xi.U < 0 (then sum w (xi.m) z = xi.U is
        // impossible for z >= 0); the dual is unbounded along xi.
        if (e.min_xm >= 0 && xi.dot(U) < -1e-14 * xi.norm() * U[0]) {
            out.infeasible = true;
            return finish(xi, e);
        }
        // Newton direction for the concave dual: (H + mu I) d = F. H is singular for degenerate active sets
        // (fewer than four affinely independent active directions); the shift keeps d an ascent direction.
        const double trace = e.H.trace();
        double mu = 1e-14 * std::max(trace, 1e-300);
        Four d;
        for (int attempt = 0;; ++attempt) {
            Eigen::LDLT<Eigen::Matrix4d> ldlt(e.H + mu * Eigen::Matrix4d::Identity());
            if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
                d = ldlt.solve(e.F);
                if (d.allFinite() && d.dot(e.F) > 0)
                    break;
            }
            if (attempt > 30)
                return finish(xi, e);
            mu = std::max(mu * 100, 1e-12 * std::max(trace, 1e-300));
        }
        // Armijo backtracking on phi (concave, C^1 with piecewise linear gradient).
        const double slope = d.dot(e.F);
        double t = 1;
        Eval trial;
        bool accepted = false;
        for (int b = 0; b <= options.max_backtracks; ++b) {
            const Four candidate = xi + t * d;
            trial = evaluate(q, ox, oy, oz, w, y, U, candidate, options);
            if (trial.phi >= e.phi + 1e-4 * t * slope || trial.F.cwiseAbs().maxCoeff() <= tol) {
                xi = candidate;
                accepted = true;
                break;
            }
            t *= .5;
            ++out.backtracks;
        }
        ++out.iterations;
        if (!accepted || !xi.allFinite())
            return finish(xi, e);
        e = trial;
    }
    if (!out.converged && e.F.cwiseAbs().maxCoeff() <= tol)
        out.converged = true;
    return finish(xi, e);
}

PositivityProjectionOutcome project_positive_moments(int q, const double *ox, const double *oy, const double *oz,
                                                     const double *w, const double *y, const Four &U, double *z,
                                                     const PositivityProjectionOptions &options,
                                                     const PositivityProjectionWarm *warm,
                                                     const PositivityFacets *facets) {
    PositivityProjectionOutcome out;
    const double rho = U[0], jn = U.tail<3>().norm();
    if (!U.allFinite() || rho < jn) { // outside R1: no nonnegative measure has these moments
        for (int k = 0; k < q; ++k)
            z[k] = 0;
        out.result.residual = std::numeric_limits<double>::infinity();
        return out;
    }
    if (!(rho > 0)) { // rho = |J| = 0
        out.result = project_positive(q, ox, oy, oz, w, y, U, z, options);
        return out;
    }
    const V3 jhat = jn > 0 ? V3(U.tail<3>() / jn) : V3::Zero();
    // Solvability of the plain rule (only relevant for strongly anisotropic states): exact with the hull facets,
    // else the necessary certificate. A relative interior margin keeps the dual well posed for Newton.
    if (facets && facets->count() > 0) {
        if (jn >= .999 * facets->inradius * rho) {
            out.hull_margin = facets->margin(V3(U.tail<3>() / rho));
            out.certificate = out.hull_margin > -1e-9;
        }
    } else if (jn > .5 * rho) {
        double best = -1;
        for (int k = 0; k < q; ++k)
            best = std::max(best, ox[k] * jhat[0] + oy[k] * jhat[1] + oz[k] * jhat[2]);
        out.certificate = jn > rho * best;
    }
    if (!out.certificate) {
        out.result = project_positive(q, ox, oy, oz, w, y, U, z, options, warm ? warm->xi : Four::Zero());
        out.newton_iterations = out.result.iterations;
        // A warm start that stalls (line search, max_iterations) must not decide the solution: retry cold before
        // augmenting, so that the result does not depend on the history of the node. A Farkas certificate is
        // rigorous and needs no retry.
        if (!out.result.converged && !out.result.infeasible && warm && !warm->xi.isZero(0)) {
            const int first = out.result.iterations;
            out.result = project_positive(q, ox, oy, oz, w, y, U, z, options, Four::Zero());
            out.result.iterations += first;
            out.newton_iterations = out.result.iterations;
        }
        out.farkas = out.result.infeasible;
        if (out.result.converged)
            return out;
    }
    if (!options.augment)
        return out; // unconverged: the caller decides (no atom)
    // Augmented rule: one atom at J^ with the mean weight.
    out.augmented = true;
    out.atom_direction = jhat;
    double wsum = 0;
    for (int k = 0; k < q; ++k)
        wsum += w[k];
    const double wstar = wsum / q;
    if (jn >= rho * (1 - 1e-14)) { // cone boundary: the atom carries everything
        out.dirac = true;
        for (int k = 0; k < q; ++k)
            z[k] = 0;
        out.atom = rho * Four(1, jhat[0], jhat[1], jhat[2]);
        out.result = PositivityProjectionResult{};
        out.result.converged = true;
        out.result.scale = rho;
        out.result.residual = (out.atom - U).cwiseAbs().maxCoeff() / rho;
        return out;
    }
    thread_local std::vector<double> ax, ay, az, aw, ayv, azv;
    ax.assign(ox, ox + q), ay.assign(oy, oy + q), az.assign(oz, oz + q), aw.assign(w, w + q), ayv.assign(y, y + q);
    ax.push_back(jhat[0]), ay.push_back(jhat[1]), az.push_back(jhat[2]), aw.push_back(wstar), ayv.push_back(0.);
    azv.resize(q + 1);
    const Four xi0 = warm ? warm->xi_augmented : Four::Zero();
    PositivityProjectionOptions aopt = options;
    thread_local std::vector<double> ac, acinv;
    if (options.c) {
        // The atom gets the weight of its hemisphere (that of the nearest rule point).
        int nearest = 0;
        double best = -2;
        for (int k = 0; k < q; ++k) {
            const double d = ox[k] * jhat[0] + oy[k] * jhat[1] + oz[k] * jhat[2];
            if (d > best)
                best = d, nearest = k;
        }
        ac.assign(options.c, options.c + q), acinv.assign(options.cinv, options.cinv + q);
        ac.push_back(options.c[nearest]), acinv.push_back(options.cinv[nearest]);
        aopt.c = ac.data(), aopt.cinv = acinv.data();
    }
    out.result = project_positive(q + 1, ax.data(), ay.data(), az.data(), aw.data(), ayv.data(), U, azv.data(),
                                  aopt, xi0);
    if (!out.result.converged && !out.result.infeasible && !xi0.isZero(0)) { // history-independent, as above
        const int first = out.result.iterations;
        out.result = project_positive(q + 1, ax.data(), ay.data(), az.data(), aw.data(), ayv.data(), U, azv.data(),
                                      aopt, Four::Zero());
        out.result.iterations += first;
    }
    out.newton_iterations += out.result.iterations;
    std::copy(azv.begin(), azv.begin() + q, z);
    out.atom = wstar * azv[q] * Four(1, jhat[0], jhat[1], jhat[2]);
    return out;
}

double PositivityFacets::margin(const V3 &u) const {
    double m = -std::numeric_limits<double>::infinity();
    const int n = count();
#pragma omp simd reduction(max : m)
    for (int f = 0; f < n; ++f)
        m = std::max(m, nx[f] * u[0] + ny[f] * u[1] + nz[f] * u[2] - h[f]);
    return m;
}

PositivityFacets lat_long_facets(const std::vector<double> &ring_z, int azimuth, const Eigen::Matrix3d &frame) {
    std::vector<double> z = ring_z;
    std::sort(z.begin(), z.end());
    const int R = int(z.size()), A = azimuth;
    auto point = [&](int i, int j) {
        const double r = std::sqrt(1 - z[i] * z[i]), a = 2 * pi * j / A;
        return V3(r * std::cos(a), r * std::sin(a), z[i]);
    };
    std::vector<V3> normals;
    std::vector<double> offsets;
    for (int i = 0; i + 1 < R; ++i)
        for (int j = 0; j < A; ++j) {
            // Isosceles trapezoid between rings i, i+1 and meridians j, j+1 (coplanar by mirror symmetry).
            const V3 p1 = point(i, j), p2 = point(i, (j + 1) % A), p3 = point(i + 1, j);
            V3 n = (p2 - p1).cross(p3 - p1);
            n.normalize();
            if (n.dot(p1) < 0)
                n = -n;
            normals.push_back(n);
            offsets.push_back(n.dot(p1));
        }
    normals.push_back(V3(0, 0, 1)), offsets.push_back(z[R - 1]);
    normals.push_back(V3(0, 0, -1)), offsets.push_back(-z[0]);
    PositivityFacets f;
    // Verify: every rule point lies in every half-space, and every facet touches the point set.
    double worst = 0;
    for (std::size_t k = 0; k < normals.size(); ++k) {
        double top = -1;
        for (int i = 0; i < R; ++i)
            for (int j = 0; j < A; ++j)
                top = std::max(top, normals[k].dot(point(i, j)) - offsets[k]);
        worst = std::max(worst, std::abs(top));
    }
    if (!(worst < 1e-13))
        return f;
    const int F = int(normals.size());
    f.nx.resize(F), f.ny.resize(F), f.nz.resize(F), f.h.resize(F);
    for (int k = 0; k < F; ++k) {
        const V3 n = frame * normals[k];
        f.nx[k] = n[0], f.ny[k] = n[1], f.nz[k] = n[2], f.h[k] = offsets[k];
    }
    f.inradius = f.h.minCoeff();
    return f;
}

} // namespace m1pn
