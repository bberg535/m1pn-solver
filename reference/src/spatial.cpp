#include "spatial.hpp"
#include "cone.hpp"
#include <Eigen/Eigenvalues>
#include <set>
namespace m1pn {
std::unique_ptr<mfem::Mesh> make_mesh(const MeshSpec &spec) {
    require(spec.dimension >= 1 && spec.dimension <= 3 && spec.elements >= 1,
            "Invalid mesh dimensions");
    if (!spec.file.empty()) {
        require(
            !spec.periodic,
            "A file mesh needs an explicit periodic topology; --periodic is for generated meshes");
        return std::make_unique<mfem::Mesh>(spec.file.c_str(), 1, 1);
    }
    if (spec.single_tet) {
        require(!spec.periodic && spec.dimension == 3, "Single tetrahedron must be open 3D");
        auto m = std::make_unique<mfem::Mesh>(3, 4, 1, 4, 3);
        double a[3] = {0, 0, 0}, b[3] = {1, 0, 0}, c[3] = {0, 1, 0}, d[3] = {0, 0, 1};
        m->AddVertex(a);
        m->AddVertex(b);
        m->AddVertex(c);
        m->AddVertex(d);
        m->AddTet(0, 1, 2, 3);
        m->AddBdrTriangle(1, 2, 3);
        m->AddBdrTriangle(0, 3, 2);
        m->AddBdrTriangle(0, 1, 3);
        m->AddBdrTriangle(0, 2, 1);
        m->FinalizeTetMesh(1, 0, true);
        return m;
    }
    mfem::Mesh m;
    if (spec.dimension == 1)
        m = mfem::Mesh::MakeCartesian1D(spec.elements);
    if (spec.dimension == 2)
        m = mfem::Mesh::MakeCartesian2D(
            spec.elements, spec.elements,
            spec.tetrahedra ? mfem::Element::TRIANGLE : mfem::Element::QUADRILATERAL, true);
    if (spec.dimension == 3)
        m = mfem::Mesh::MakeCartesian3D(spec.elements, spec.elements, spec.elements,
                                        spec.tetrahedra ? mfem::Element::TETRAHEDRON
                                                        : mfem::Element::HEXAHEDRON);
    if (spec.periodic) {
        require(spec.elements >= 3, "Periodic H1 mesh requires at least three cells per direction");
        std::vector<mfem::Vector> translations;
        for (int d = 0; d < spec.dimension; ++d) {
            mfem::Vector t(spec.dimension);
            t = 0.;
            t[d] = 1.;
            translations.push_back(t);
        }
        auto map = m.CreatePeriodicVertexMapping(translations);
        return std::make_unique<mfem::Mesh>(mfem::Mesh::MakePeriodic(m, map));
    }
    return std::make_unique<mfem::Mesh>(std::move(m));
}
static std::unique_ptr<mfem::Mesh> checked_mesh(const MeshSpec &spec, const Budget &budget) {
    if (spec.file.empty()) {
        const long double cells = std::pow(static_cast<long double>(spec.elements), spec.dimension);
        const long double nodes = std::pow(spec.elements + 1.L, spec.dimension);
        require(cells > 0 && cells * 6 < std::numeric_limits<int>::max() &&
                    nodes < std::numeric_limits<int>::max(),
                "Mesh index overflow");
        budget.check(cells * (spec.tetrahedra ? 6144.L : 4096.L) + nodes * 8192.L,
                     "Mesh/FE preallocation estimate");
    }
    return make_mesh(spec);
}
Spatial::Spatial(const MeshSpec &spec, double anchor, const Budget &budget, bool factorize)
    : mesh(checked_mesh(spec, budget)) {
    dimension = mesh->Dimension();
    require(mesh->SpaceDimension() == dimension, "Embedded manifolds unsupported");
    require(anchor >= 1 && std::isfinite(anchor), "Graph anchor must be at least one");
    collection = std::make_unique<mfem::H1_FECollection>(1, dimension);
    space = std::make_unique<mfem::FiniteElementSpace>(mesh.get(), collection.get());
    nodes = space->GetNDofs();
    require(space->GetNConformingDofs() == nodes, "Nonconforming meshes unsupported");
    budget.check(8192.L * nodes + 192.L * std::pow(static_cast<long double>(nodes), 4.L / 3),
                 "Spatial assembly and mass factorization estimate");
    mass = Vec::Zero(nodes);
    diameter = Vec::Zero(nodes);
    coordinates = Field::Zero(nodes, 3);
    boundary.assign(nodes, false);
    std::vector<Triplet> mt, kt;
    std::array<std::vector<Triplet>, 3> ct, bt;
    std::set<std::pair<int, int>> graph;
    for (int e = 0; e < mesh->GetNE(); ++e) {
        const mfem::FiniteElement *fe = space->GetFE(e);
        const int nd = fe->GetDof();
        auto *tr = mesh->GetElementTransformation(e);
        require(tr->Order() <= 1,
                "Curved/high-order geometry needs a separately validated quadrature");
        mfem::Array<int> dofs;
        space->GetElementDofs(e, dofs);
        Mat me = Mat::Zero(nd, nd), ke = me;
        std::array<Mat, 3> ce = {me, me, me};
        Field points(nd, 3);
        points.setZero();
        mfem::Vector x(dimension), shape(nd);
        mfem::DenseMatrix grad(nd, dimension);
        for (int i = 0; i < nd; ++i) {
            tr->Transform(fe->GetNodes().IntPoint(i), x);
            for (int d = 0; d < dimension; ++d)
                coordinates(dofs[i], d) = points(i, d) = x[d];
        }
        double h = 0;
        for (int i = 0; i < nd; ++i)
            for (int j = 0; j < i; ++j)
                h = std::max(h, (points.row(i) - points.row(j)).norm());
        require(h > 0, "Degenerate element diameter");
        const auto &rule = mfem::IntRules.Get(fe->GetGeomType(), 6);
        for (int p = 0; p < rule.GetNPoints(); ++p) {
            const auto &ip = rule.IntPoint(p);
            tr->SetIntPoint(&ip);
            require(tr->Jacobian().Det() > 0, "Inverted element");
            fe->CalcShape(ip, shape);
            fe->CalcPhysDShape(*tr, grad);
            const double w = ip.weight * tr->Weight();
            require(w > 0, "FE quadrature must have positive weights");
            for (int i = 0; i < nd; ++i) {
                require(shape[i] >= -1e-15, "Nonnegative FE basis required");
                for (int j = 0; j < nd; ++j) {
                    me(i, j) += w * shape[i] * shape[j];
                    for (int d = 0; d < dimension; ++d) {
                        ke(i, j) += w * grad(i, d) * grad(j, d);
                        ce[d](i, j) += w * shape[i] * grad(j, d);
                    }
                }
            }
        }
        Vec row = me.rowwise().sum();
        require(row.minCoeff() > 0, "Nonpositive element mass");
        Vec inv = row.cwiseSqrt().cwiseInverse();
        Eigen::SelfAdjointEigenSolver<Mat> eig(inv.asDiagonal() * me * inv.asDiagonal());
        require(eig.info() == Eigen::Success && eig.eigenvalues()[0] > 0,
                "Element mass is not SPD");
        mass_equivalence = std::min(mass_equivalence, eig.eigenvalues()[0] * (1 - 1e-12));
        for (int i = 0; i < nd; ++i) {
            const int ii = dofs[i];
            diameter[ii] = std::max(diameter[ii], h);
            for (int j = 0; j < nd; ++j) {
                const int jj = dofs[j];
                mt.emplace_back(ii, jj, me(i, j));
                kt.emplace_back(ii, jj, ke(i, j));
                for (int d = 0; d < 3; ++d)
                    ct[d].emplace_back(ii, jj, ce[d](i, j));
                if (ii < jj)
                    graph.insert({ii, jj});
            }
        }
    }
    M.resize(nodes, nodes);
    M.setFromTriplets(mt.begin(), mt.end());
    mass = M * Vec::Ones(nodes);
    K.resize(nodes, nodes);
    K.setFromTriplets(kt.begin(), kt.end());
    for (int d = 0; d < 3; ++d) {
        C[d].resize(nodes, nodes);
        C[d].setFromTriplets(ct[d].begin(), ct[d].end());
    }
    if (!spec.periodic)
        for (int b = 0; b < mesh->GetNBE(); ++b) {
            auto *tr = mesh->GetBdrFaceTransformations(b);
            require(tr, "Missing boundary face transform");
            const mfem::FiniteElement *fe = space->GetFE(tr->Elem1No);
            mfem::Array<int> dofs;
            space->GetElementDofs(tr->Elem1No, dofs);
            Face face;
            face.nodes.assign(dofs.begin(), dofs.end());
            face.mass = Vec::Zero(dofs.Size());
            const auto &rule = mfem::IntRules.Get(tr->GetGeometryType(), 6);
            for (int p = 0; p < rule.GetNPoints(); ++p) {
                const auto &ip = rule.IntPoint(p);
                tr->SetAllIntPoints(&ip);
                mfem::Vector shape(dofs.Size()), normal(dimension), x(dimension);
                fe->CalcShape(tr->GetElement1IntPoint(), shape);
                tr->Transform(ip, x);
                if (dimension == 1)
                    normal[0] = tr->GetElement1IntPoint().x < .5 ? -1 : 1;
                else
                    mfem::CalcOrtho(tr->Jacobian(), normal);
                const double jac = normal.Norml2();
                SurfacePoint point;
                point.x = V3::Zero();
                point.normal = V3::Zero();
                for (int d = 0; d < dimension; ++d) {
                    point.x[d] = x[d];
                    point.normal[d] = normal[d] / jac;
                }
                point.weight = ip.weight * jac;
                point.shape = Vec::Zero(dofs.Size());
                for (int i = 0; i < dofs.Size(); ++i) {
                    point.shape[i] = shape[i];
                    face.mass[i] += point.weight * shape[i];
                    if (shape[i] > 1e-14)
                        boundary[dofs[i]] = true;
                    for (int j = 0; j < dofs.Size(); ++j)
                        for (int d = 0; d < 3; ++d)
                            bt[d].emplace_back(dofs[i], dofs[j],
                                               point.weight * shape[i] * shape[j] *
                                                   point.normal[d]);
                }
                face.points.push_back(std::move(point));
            }
            faces.push_back(std::move(face));
        }
    for (int d = 0; d < 3; ++d) {
        B[d].resize(nodes, nodes);
        B[d].setFromTriplets(bt[d].begin(), bt[d].end());
        Sparse defect = C[d] + Sparse(C[d].transpose()) - B[d];
        ibp_error = std::max(ibp_error, defect.norm());
        constant_error =
            std::max(constant_error, (C[d] * Vec::Ones(nodes)).lpNorm<Eigen::Infinity>());
    }
    constant_error = std::max(constant_error, (K * Vec::Ones(nodes)).lpNorm<Eigen::Infinity>());
    require(ibp_error < 2e-11 * std::max(1., K.norm()) &&
                constant_error < 2e-11 * std::max(1., K.norm()),
            "FE integration-by-parts identity failed");
    Vec rates = Vec::Zero(nodes);
    for (auto [i, j] : graph) {
        Pair p;
        p.i = i;
        p.j = j;
        for (int d = 0; d < 3; ++d)
            p.s[d] = .5 * (C[d].coeff(i, j) - C[d].coeff(j, i));
        p.mass = M.coeff(i, j);
        p.stiffness = K.coeff(i, j);
        p.viscosity = anchor * std::max(p.s.norm(), p.mass / std::max(diameter[i], diameter[j]));
        require(p.viscosity > 0, "All correction pairs require positive viscosity");
        // A nearest-rounded norm can lie just below the actual stored edge
        // length. Round only this scalar graph-speed bound outward, before
        // constructing CFL rates and pair fluxes from that same value.
        while (!exact::cone(Four(p.viscosity, p.s[0], p.s[1], p.s[2]))) {
            p.viscosity = std::nextafter(p.viscosity, std::numeric_limits<double>::infinity());
            require(std::isfinite(p.viscosity), "Unrepresentable graph speed");
        }
        pairs.push_back(p);
        rates[i] += 2 * p.viscosity;
        rates[j] += 2 * p.viscosity;
    }
    for (const auto &f : faces)
        for (std::size_t i = 0; i < f.nodes.size(); ++i)
            rates[f.nodes[i]] += anchor * f.mass[i];
    cfl_step = std::numeric_limits<double>::infinity();
    for (int i = 0; i < nodes; ++i)
        if (rates[i] > 0)
            cfl_step = std::min(cfl_step, mass[i] / rates[i]);
    if (factorize) {
        factor.compute(Eigen::SparseMatrix<double>(M));
        require(factor.info() == Eigen::Success, "Consistent mass factorization failed");
        factorized = true;
    }
}
Field Spatial::solve_mass(const Field &rhs, double tol, double &residual) const {
    require(factorized, "Consistent mass solve requested without a factorization");
    ++mass_solve_calls;
    // Contiguous independent components avoid strided reads/writes through
    // the row-major transport field. Factor and substitution order are unchanged.
    Mat columns = rhs, solution(rhs.rows(), rhs.cols());
#pragma omp parallel for schedule(static)
    for (int k = 0; k < rhs.cols(); ++k)
        solution.col(k) = factor.solve(Vec(columns.col(k)));
    Field x = solution;
    residual = maxabs(M * x - rhs);
    require(x.allFinite() && residual <= tol * std::max(1., maxabs(rhs)),
            "Consistent mass true residual exceeds tolerance");
    return x;
}
double Spatial::norm_bound(const Sparse &a) const {
    Vec row = Vec::Zero(nodes), col = row;
    for (int i = 0; i < a.outerSize(); ++i)
        for (Sparse::InnerIterator it(a, i); it; ++it) {
            double v = std::abs(it.value()) / std::sqrt(mass[it.row()] * mass[it.col()]);
            row[it.row()] += v;
            col[it.col()] += v;
        }
    return std::sqrt(row.maxCoeff() * col.maxCoeff());
}
} // namespace m1pn
