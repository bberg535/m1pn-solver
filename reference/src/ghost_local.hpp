#pragma once
#include "spatial.hpp"
namespace m1pn {
// Frozen sparse geometry for open uniform Cartesian Q1, >=5 nodes per axis.
// Wider than one FE ring. R2 uses replicated mesh/state, partitioned ledgers.
class GhostLocalProjection {
    struct Star { std::vector<int> nodes; Vec fractions, weights; int offset; };
    const Spatial &fe_;
    Sparse laplace_, routing_;
    std::vector<Star> stars_;
    int incidences_ = 0;
    bool disabled_ = false;
public:
    double norm_bound = 0; // ML metric; caller applies mass equivalence once
    GhostLocalProjection(const Spatial &, double gamma, double speed);
    Field apply(const Field &, Field *pairs = nullptr) const;
    int incidences() const { return incidences_; }
    int routing_nonzeros() const { return routing_.nonZeros(); }
};
}
