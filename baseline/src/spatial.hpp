#pragma once
#include "common.hpp"
#include <mfem.hpp>
#include <Eigen/SparseCholesky>
#include <memory>
namespace m1pn {
struct Pair {
    int i, j;
    V3 s;
    double mass, stiffness, viscosity;
};
struct SurfacePoint {
    V3 x, normal;
    double weight;
    Vec shape;
    int normal_id = -1;
};
struct Face {
    std::vector<int> nodes;
    std::vector<SurfacePoint> points;
    Vec mass;
};
struct MeshSpec {
    int dimension = 3, elements = 2;
    bool periodic = false, tetrahedra = false, single_tet = false;
    std::string file;
};
std::unique_ptr<mfem::Mesh> make_mesh(const MeshSpec &);
struct Spatial {
    std::unique_ptr<mfem::Mesh> mesh;
    std::unique_ptr<mfem::H1_FECollection> collection;
    std::unique_ptr<mfem::FiniteElementSpace> space;
    int nodes, dimension;
    Sparse M, K;
    std::array<Sparse, 3> C, B;
    Vec mass, diameter;
    Field coordinates;
    std::vector<bool> boundary;
    std::vector<Pair> pairs;
    std::vector<Face> faces;
    double mass_equivalence = 1, ibp_error = 0, constant_error = 0, cfl_step;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factor;
    bool factorized = false;
    mutable long mass_solve_calls = 0;
    Spatial(const MeshSpec &, double anchor_scale = 1, const Budget &budget = {}, bool factorize = true);
    Field solve_mass(const Field &, double tolerance, double &residual) const;
    double norm_bound(const Sparse &a) const;
};
} // namespace m1pn
