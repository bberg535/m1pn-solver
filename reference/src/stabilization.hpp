#pragma once
#include "spatial.hpp"
#include "ghost_local.hpp"
namespace m1pn {
// R2 (4.9)-(4.18). Geometry is frozen; application uses incidence passes.
class SpatialStabilization {
    struct Element {
        std::vector<int> nodes;
        std::vector<Mat> gradients; // vertex -> physical direction x local basis
        Vec omega;
        Mat L;
        std::vector<std::array<int,3>> edges; // local i,j, global pair index
    };
    const Spatial &fe_;
    std::string kind_;
    std::vector<Element> elements_;
    Vec star_mass_, nu_, beta_;
    Sparse L_;
    std::unique_ptr<GhostLocalProjection> ghost_local_;
public:
    double norm_bound = 0;
    SpatialStabilization(const Spatial &, const std::string &, double gamma, double speed);
    Field apply(const Field &, Field *pairs = nullptr) const;
};
}
