#pragma once
// Distributed (MPI + OpenMP) element WENO: slab decomposition of the tensor mesh along its last axis.
//
// Every rank builds the unchanged ElementWeno on its slab of elements plus `halo` element layers on each internal
// side (ElementWenoOptions::cut: no boundary faces there). All element, node and boundary kernels then run locally;
// the only global couplings of the scheme are
//   - the consistent-mass line solves along the decomposition axis (projected gradient, mass solve): a distributed
//     forward/backward substitution in the order of the serial LDL^T factor of the global 1D mass, pipelined over
//     column blocks. Every operation y_i -= l y_j happens in the serial order, so the result is bitwise equal to the
//     serial solve for every rank count;
//   - halo rows (after each line solve along the axis, the high-order moment rates before the limiter, and every
//     stage rate) from their owners;
//   - reductions (time step bound, boundary mass ledger, diagnostics).
// Requirements: dimension >= 2, open mesh (no periodic axis), no mirror sides, standard residual.
#include "element_weno.hpp"
#ifdef M1PN_MPI
#include <mpi.h>

namespace m1pn {

struct SlabPartition {
    int ranks = 1, rank = 0, axis = 1, elements = 0, halo = 2; // global elements along the axis
    std::vector<int> first;                                   // first[r]: first owned element of rank r (size ranks+1)
    int owned_first(int r) const { return first[r]; }
    int owned_end(int r) const { return first[r + 1]; }
    int local_first(int r) const { return std::max(0, first[r] - halo); }
    int local_end(int r) const { return std::min(elements, first[r + 1] + halo); }
};
SlabPartition slab_partition(int elements, int ranks, int rank, int axis, int halo);
// Options of the local scheme of `part.rank` (elements along the axis, origin, cut sides).
ElementWenoOptions slab_options(const ElementWenoOptions &global, const SlabPartition &part);

class ElementWenoMpi : public DistributedHooks {
  public:
    // Registers itself with `scheme` (set_distributed). chunks: column blocks of the pipelined line solve (0: auto).
    ElementWenoMpi(ElementWeno &scheme, MPI_Comm comm, const SlabPartition &part, int chunks = 0);
    ~ElementWenoMpi() override;
    void solve_axis(Field &b) const override;
    void exchange(Field &f) const override;
    double sum(double local) const override;
    double min(double local) const override;
    double max(double local) const;
    int rank() const { return part_.rank; }
    int ranks() const { return part_.ranks; }
    long global_id(int local) const { return node_offset + local; }
    // Owned rows (global ids, row-major data) gathered to rank 0 in global order; empty on other ranks.
    Field gather(const Field &f, std::vector<long> *ids = nullptr) const;
    mutable double t_solve = 0, t_exchange = 0, t_wait = 0;
    bool threaded() const { return thread_multiple_; }

  private:
    ElementWeno &s_;
    MPI_Comm comm_;
    SlabPartition part_;
    int p_ = 1, layer_size_ = 0, L0_ = 0, layers_ = 0, O0_ = 0, O1_ = 0, chunks_ = 1;
    bool thread_multiple_ = false; // MPI_THREAD_MULTIPLE: every thread pipelines its own column blocks
    void solve_axis_threaded(Field &b) const;
    std::vector<int> owner_;                     // global layer -> rank
    // Global 1D mass factor (as in the ElementWeno constructor): CSC of L, D, perm (layer -> position), inverse.
    std::vector<int> outer_, inner_, perm_, inv_;
    std::vector<double> values_, diag_;
    // Runs of consecutive factor positions with one owner (global, in position order). The substitution processes
    // them in this order (backward: reversed); after a run its owner sends the values other ranks need.
    struct Run {
        int owner, first, end;
    };
    std::vector<Run> runs_;
    struct Message {
        int run, peer;                           // global run index; peer = destination (send) or source (recv)
        std::vector<int> positions;              // ascending
    };
    std::vector<Message> fwd_send_, fwd_recv_, bwd_send_, bwd_recv_; // ordered by run (backward: descending)
    struct Range {
        int rank, layer0, layer1; // local layers [layer0, layer1)
    };
    std::vector<Range> halo_recv_, halo_send_;
    double *layer(Field &b, int global_layer) const {
        return b.data() + std::size_t(global_layer - L0_) * layer_size_ * b.cols();
    }
};

} // namespace m1pn
#endif
