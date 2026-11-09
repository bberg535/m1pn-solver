#include "boundary.hpp"
#include "cone.hpp"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
namespace m1pn {
Four copied_velocity_exterior(double density, const Four &interior,
                              const std::optional<V3> &vacuum_velocity) {
    require(std::isfinite(density) && density >= 0 && exact::cone(interior),
            "Invalid copied-velocity exterior data");
    if (density == 0) return Four::Zero();
    V3 velocity;
    if (interior[0] > 0) velocity = interior.tail<3>() / interior[0];
    else {
        require(vacuum_velocity.has_value(),
                "Positive exterior density at interior vacuum needs an explicit exterior velocity");
        velocity = *vacuum_velocity;
    }
    Four out; out[0] = density; out.tail<3>() = density * velocity;
    require(exact::cone(out), "Prescribed exterior velocity is outside the M1 cone");
    return out;
}
CanonicalC1::CanonicalC1(const Angular &pn) : q(pn.q), r(pn.order * (pn.order + 1) / 2), s(q - r) {
    for (int l = 0; l <= pn.order; ++l)
        for (int m = -l; m <= l; ++m)
            ((l + std::abs(m)) % 2 ? odd : even).push_back(index(l, m));
    require(int(odd.size()) == r && int(even.size()) == s, "PN parity dimensions failed");
    B.resize(r, s);
    L = Mat::Zero(r, r);
    Mat az = pn.normal(V3::UnitZ());
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < s; ++j)
            B(i, j) = az(odd[i], even[j]);
    for (const auto &p : sphere_rule(pn.order, 1)) {
        Vec y = harmonics(pn.order, p.omega), o(r);
        for (int i = 0; i < r; ++i)
            o[i] = y[odd[i]];
        L.noalias() += 2 * p.weight / p.omega[2] * (o * o.transpose());
    }
    Eigen::LLT<Mat> positive(L);
    require(positive.info() == Eigen::Success, "Onsager L is not SPD");
    Eigen::JacobiSVD<Mat> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    U = svd.matrixU();
    V = svd.matrixV();
    sigma = svd.singularValues();
    root = sigma.cwiseSqrt();
    require(sigma.minCoeff() > 1e-12, "Unresolved PN characteristic gap");
    Rplus = Mat::Zero(q, r);
    Rminus = Mat::Zero(q, r);
    for (int i = 0; i < r; ++i) {
        Rplus.row(odd[i]) = U.row(i) / std::sqrt(2.);
        Rminus.row(odd[i]) = U.row(i) / std::sqrt(2.);
    }
    for (int i = 0; i < s; ++i) {
        Rplus.row(even[i]) = V.row(i) / std::sqrt(2.);
        Rminus.row(even[i]) = -V.row(i) / std::sqrt(2.);
    }
    S = root.asDiagonal() * U.transpose() * L * U * root.asDiagonal();
    solve.compute(Mat::Identity(r, r) + S);
    require(solve.info() == Eigen::Success, "C1 incoming system not SPD");
    Mat weighted = solve.solve(S - Mat::Identity(r, r));
    reflection = root.cwiseInverse().asDiagonal() * weighted * root.asDiagonal();
    Eigen::SelfAdjointEigenSolver<Mat> eig(S);
    require(eig.info() == Eigen::Success && eig.eigenvalues().minCoeff() > 0, "C1 spectrum failed");
    contraction =
        ((eig.eigenvalues().array() - 1) / (eig.eigenvalues().array() + 1)).abs().maxCoeff();
    condition = (1 + eig.eigenvalues().maxCoeff()) / (1 + eig.eigenvalues().minCoeff());
    require(contraction < 1, "C1 weighted contraction failed");
}
Vec CanonicalC1::incoming(const Vec &wp, const Vec &d) const {
    Vec rhs = (S - Mat::Identity(r, r)) * (root.array() * wp.array()).matrix() +
              std::sqrt(2.) * root.asDiagonal() * U.transpose() * d;
    Vec z = solve.solve(rhs);
    require(((Mat::Identity(r, r) + S) * z - rhs).norm() < 2e-12 * std::max(1., rhs.norm()),
            "C1 SPD true residual failed");
    return (z.array() / root.array()).matrix();
}
Boundary::Boundary(const Angular &a, const CanonicalC1 &c, const V3 &normal, bool zero)
    : pn(a), canonical(c), n(normal), c0(zero) {
    require(std::abs(n.norm() - 1) < 2e-12, "Boundary normal must be unit");
    V3 ref = std::abs(n[0]) < .8 ? V3::UnitX() : V3::UnitY();
    frame.col(0) = (ref - ref.dot(n) * n).normalized();
    frame.col(1) = n.cross(frame.col(0));
    frame.col(2) = n;
    D = pn.rotation(frame);
    Rplus = D * c.Rplus;
    Rminus = D * c.Rminus;
    require(maxabs(pn.normal(n) - D * pn.normal(V3::UnitZ()) * D.transpose()) < 2e-11,
            "PN rotation covariance failed");
    homogeneous = Rplus * c.sigma.asDiagonal() * Rplus.transpose();
    if (!c0)
        homogeneous.noalias() -= Rminus * c.sigma.asDiagonal() * c.reflection * Rplus.transpose();
    Eigen::JacobiSVD<Mat> normsolve(homogeneous);
    norm = normsolve.singularValues()[0];
    const Mat exterior_map = Mat::Identity(pn.q,pn.q) +
        Rminus * (c.reflection * Rplus.transpose() - Rminus.transpose());
    const Mat normal_matrix = pn.normal(n);
    llf_homogeneous = .5 * (normal_matrix + Mat::Identity(pn.q,pn.q)) +
        .5 * (normal_matrix - Mat::Identity(pn.q,pn.q)) * exterior_map;
    Eigen::JacobiSVD<Mat> llf_svd(llf_homogeneous);
    llf_norm = llf_svd.singularValues()[0];
    // Exterior datum map: incoming(0,d) of CanonicalC1 as a matrix, with the
    // same true-residual guard that incoming() applies on every call.
    {
        const Mat load = std::sqrt(2.) * c.root.asDiagonal() * c.U.transpose();
        const Mat z = c.solve.solve(load);
        require(((Mat::Identity(c.r, c.r) + c.S) * z - load).norm() < 2e-12 * std::max(1., load.norm()),
                "C1 SPD true residual failed for datum map");
        llf_datum = .5 * (normal_matrix - Mat::Identity(pn.q, pn.q)) *
                    (Rminus * (c.root.cwiseInverse().asDiagonal() * z));
    }
    rule = sphere_rule(pn.order + 2, -1);
    odd_basis.resize(rule.size(), c.r);
    for (std::size_t p = 0; p < rule.size(); ++p) {
        Vec y = harmonics(pn.order, rule[p].omega);
        for (int j = 0; j < c.r; ++j)
            odd_basis(p, j) = y[c.odd[j]];
        rule[p].omega = frame * rule[p].omega;
    }
}
Incoming Boundary::data(const Inflow &in, const V3 &x, double time) const {
    Incoming out;
    out.d = Vec::Zero(canonical.r);
    if (in.density)
        for (std::size_t k = 0; k < rule.size(); ++k) {
            const auto &p = rule[k];
            double g = in.density(x, time, p.omega);
            require(std::isfinite(g) && g >= 0, "Physical inflow density must be nonnegative");
            out.d.noalias() += 2 * p.weight * g * odd_basis.row(k).transpose();
            out.moments[0] += p.weight*g;
            out.moments.tail<3>() += p.weight*g*p.omega;
            const double mass = p.weight * g * n.dot(p.omega);
            out.kinetic[0] += mass;
            out.kinetic.tail<3>() += mass * p.omega;
        }
    if (in.atoms)
        for (const auto &a : in.atoms(x, time)) {
            require(a.mass >= 0 && std::isfinite(a.mass) &&
                        std::abs(a.direction.norm() - 1) < 2e-12,
                    "Invalid incoming atom");
            const double mu = n.dot(a.direction);
            if (mu >= 0)
                continue;
            out.moments[0] += a.mass;
            out.moments.tail<3>() += a.mass*a.direction;
            Vec y = harmonics(pn.order, frame.transpose() * a.direction);
            for (int j = 0; j < canonical.r; ++j)
                out.d[j] += 2 * a.mass * y[canonical.odd[j]];
            out.kinetic[0] += mu * a.mass;
            out.kinetic.tail<3>() += mu * a.mass * a.direction;
        }
    if (c0)
        require(out.d.norm() == 0 && out.kinetic.norm() == 0,
                "C0 h=0 requires vacuum physical inflow; it is not C1");
    // Convert the nonnegative kinetic quadrature to an admissible stored M1
    // exterior. Products of a collimated atom's mass and rounded direction can
    // put its stored current outside its stored density by an ulp. Round the
    // exterior density outward, without a density floor or an admission tolerance.
    // This conversion affects only the independent M1 exterior, never the PN datum.
    require(out.moments.allFinite() && out.moments[0] >= 0, "Invalid incoming moment quadrature");
    for(int k=0;k<64 && !exact::cone(out.moments);++k)
        out.moments[0]=std::nextafter(out.moments[0],std::numeric_limits<double>::infinity());
    require(exact::cone(out.moments),"Unable to represent incoming moments in M1 cone");
    return out;
}
Vec Boundary::flux(const Vec &u, const Vec &d) const {
    // The homogeneous trace action is cached. Only physical datum solves remain.
    Vec g = homogeneous * u;
    if (!c0 && d.squaredNorm() != 0) {
        Vec h = canonical.incoming(Vec::Zero(canonical.r), d);
        g.noalias() -= Rminus * (canonical.sigma.array() * h.array()).matrix();
    }
    return g;
}
Vec Boundary::exterior(const Vec &interior, const Vec &d) const {
    require(interior.size() == pn.q && interior.allFinite(), "Invalid PN boundary trace");
    require(d.size() == canonical.r && d.allFinite(), "Invalid Onsager inflow vector");
    const Vec outgoing = Rplus.transpose() * interior;
    const Vec incoming = canonical.incoming(outgoing, d);
    // R+ and R- are orthonormal characteristic frames. Their orthogonal
    // complement is precisely ker A(n), whose amplitudes are copied through.
    const Vec zero = interior - Rplus * outgoing - Rminus * (Rminus.transpose() * interior);
    return Rminus * incoming + Rplus * outgoing + zero;
}
} // namespace m1pn
