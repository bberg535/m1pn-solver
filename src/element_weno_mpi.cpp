#include "element_weno_mpi.hpp"
#ifdef M1PN_MPI
#include "parallel.hpp"
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <map>
#include <omp.h>
#include <cstdlib>

namespace m1pn {

SlabPartition slab_partition(int elements, int ranks, int rank, int axis, int halo) {
    require(ranks >= 1 && rank >= 0 && rank < ranks, "slab_partition: invalid rank");
    require(elements >= ranks * std::max(1, halo), "MPI: every rank needs at least `halo` elements along the axis");
    SlabPartition p;
    p.ranks = ranks, p.rank = rank, p.axis = axis, p.elements = elements, p.halo = halo;
    p.first.resize(ranks + 1);
    for (int r = 0; r <= ranks; ++r)
        p.first[r] = int(long(r) * elements / ranks);
    return p;
}

ElementWenoOptions slab_options(const ElementWenoOptions &g, const SlabPartition &part) {
    const int D = part.axis;
    require(D == g.dimension - 1 && D >= 1, "MPI: slabs along the last axis of a 2D/3D mesh");
    require(!g.periodic, "MPI: periodic meshes are not supported");
    require(std::none_of(g.mirror.begin(), g.mirror.end(), [](bool m) { return m; }), "MPI: no mirror sides");
    ElementWenoOptions o = g;
    const int r = part.rank, global = g.elements_per_axis[D] > 0 ? g.elements_per_axis[D] : g.elements;
    require(global == part.elements, "slab_options: partition does not match the options");
    o.elements_per_axis[D] = part.local_end(r) - part.local_first(r);
    o.element_offset[D] = part.local_first(r); // positions from the global index (bitwise serial coordinates)
    o.cut[2 * D] = part.local_first(r) > 0;
    o.cut[2 * D + 1] = part.local_end(r) < part.elements;
    return o;
}

ElementWenoMpi::ElementWenoMpi(ElementWeno &scheme, MPI_Comm comm, const SlabPartition &part, int chunks)
    : s_(scheme), comm_(comm), part_(part), chunks_(chunks) {
    const int D = part.axis, r = part.rank, P = part.ranks, E = part.elements;
    axis = D;
    p_ = s_.degree;
    layer_size_ = s_.nodes / s_.nodes_per_axis[D];
    L0_ = part.local_first(r) * p_;
    layers_ = s_.nodes_per_axis[D];
    require(layers_ == (part.local_end(r) - part.local_first(r)) * p_ + 1, "MPI: local layer count mismatch");
    const int n1 = E * p_ + 1;
    auto owned_range = [&](int t) {
        return std::make_pair(part.owned_first(t) * p_, t == P - 1 ? n1 : part.owned_end(t) * p_);
    };
    std::tie(O0_, O1_) = owned_range(r);
    owner_.assign(n1, -1);
    for (int t = 0; t < P; ++t) {
        const auto [a, b] = owned_range(t);
        for (int k = a; k < b; ++k)
            owner_[k] = t;
    }
    node_offset = long(L0_) * layer_size_;
    global_nodes = long(n1) * layer_size_;
    // Global 1D mass along the axis, assembled and factored exactly as in the ElementWeno constructor.
    {
        const Mat &m1 = s_.m1_reference_;
        std::vector<Eigen::Triplet<double>> triplets;
        for (int e = 0; e < E; ++e)
            for (int i = 0; i <= p_; ++i)
                for (int j = 0; j <= p_; ++j)
                    triplets.emplace_back(e * p_ + i, e * p_ + j, s_.h * m1(i, j));
        Eigen::SparseMatrix<double> m(n1, n1);
        m.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> f(m);
        require(f.info() == Eigen::Success, "MPI: global line mass factorization failed");
        Eigen::SparseMatrix<double> L = f.matrixL();
        L.makeCompressed();
        outer_.assign(L.outerIndexPtr(), L.outerIndexPtr() + n1 + 1);
        inner_.assign(L.innerIndexPtr(), L.innerIndexPtr() + L.nonZeros());
        values_.assign(L.valuePtr(), L.valuePtr() + L.nonZeros());
        const Vec d = f.vectorD();
        diag_.assign(d.data(), d.data() + n1);
        perm_.resize(n1), inv_.resize(n1);
        for (int k = 0; k < n1; ++k)
            perm_[k] = f.permutationP().indices()[k], inv_[perm_[k]] = k;
    }
    // Runs of positions with one owner.
    for (int j = 0; j < n1; ++j) {
        const int o = owner_[inv_[j]];
        if (runs_.empty() || runs_.back().owner != o)
            runs_.push_back({o, j, j + 1});
        else
            runs_.back().end = j + 1;
    }
    std::vector<int> run_of(n1);
    for (std::size_t g = 0; g < runs_.size(); ++g)
        for (int j = runs_[g].first; j < runs_[g].end; ++j)
            run_of[j] = int(g);
    // Messages. Forward: column j (owner oj) updates row i (owner oi != oj): oj sends y_j after the run of j.
    // Backward: column j reads the final y_i (i > j): oi sends y_i after the (backward) run of i.
    std::map<std::pair<int, int>, std::vector<int>> fs, fr, bs, br; // (run, peer) -> positions
    for (int j = 0; j < n1; ++j)
        for (int k = outer_[j]; k < outer_[j + 1]; ++k) {
            const int i = inner_[k];
            if (i == j)
                continue;
            const int oj = owner_[inv_[j]], oi = owner_[inv_[i]];
            if (oj == oi)
                continue;
            if (oj == r)
                fs[{run_of[j], oi}].push_back(j), br[{run_of[i], oi}].push_back(i);
            if (oi == r)
                fr[{run_of[j], oj}].push_back(j), bs[{run_of[i], oj}].push_back(i);
        }
    auto messages = [](std::map<std::pair<int, int>, std::vector<int>> &m, bool descending) {
        std::vector<Message> out;
        for (auto &[key, v] : m) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            out.push_back({key.first, key.second, v});
        }
        if (descending)
            std::stable_sort(out.begin(), out.end(), [](const Message &x, const Message &y) { return x.run > y.run; });
        return out;
    };
    fwd_send_ = messages(fs, false), fwd_recv_ = messages(fr, false);
    bwd_send_ = messages(bs, true), bwd_recv_ = messages(br, true);
    // Halo layers: mine from their owners; the halo layers of the other ranks that I own.
    auto runs = [&](int first_layer, int count, int self, bool mine, std::vector<Range> &out, int base) {
        int k = first_layer;
        while (k < first_layer + count) {
            const int o = owner_[k];
            int e = k;
            while (e < first_layer + count && owner_[e] == o)
                ++e;
            if (mine ? (o != self) : (o == r))
                out.push_back({mine ? o : self, k - base, e - base});
            k = e;
        }
    };
    runs(L0_, layers_, r, true, halo_recv_, L0_);
    for (int t = 0; t < P; ++t) {
        if (t == r)
            continue;
        const int lt0 = part.local_first(t) * p_, nt = (part.local_end(t) - part.local_first(t)) * p_ + 1;
        runs(lt0, nt, t, false, halo_send_, L0_);
    }
    {
        int provided = 0;
        MPI_Query_thread(&provided);
        // Opt-in only: with UCX, 32 threads blocking in MPI_Recv contend badly (NX: 4 ranks 1665 s instead of 47 s).
        thread_multiple_ = provided >= MPI_THREAD_MULTIPLE && std::getenv("M1PN_MPI_THREADED_PIPELINE");
    }
    std::vector<char> owned(s_.nodes, 0);
    for (int i = 0; i < s_.nodes; ++i) {
        const int k = L0_ + i / layer_size_;
        owned[i] = k >= O0_ && k < O1_;
    }
    s_.set_distributed(this, std::move(owned));
}

ElementWenoMpi::~ElementWenoMpi() { s_.set_distributed(nullptr, {}); }

double ElementWenoMpi::sum(double v) const {
    double out = 0;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return out;
}
double ElementWenoMpi::min(double v) const {
    double out = 0;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MIN, comm_);
    return out;
}
double ElementWenoMpi::max(double v) const {
    double out = 0;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, comm_);
    return out;
}

void ElementWenoMpi::exchange(Field &f) const {
    const double start = MPI_Wtime();
    const std::size_t row = std::size_t(layer_size_) * f.cols();
    std::vector<MPI_Request> req;
    req.reserve(halo_recv_.size() + halo_send_.size());
    for (const auto &h : halo_recv_) {
        req.emplace_back();
        MPI_Irecv(f.data() + h.layer0 * row, int((h.layer1 - h.layer0) * row), MPI_DOUBLE, h.rank, 7001, comm_,
                  &req.back());
    }
    for (const auto &h : halo_send_) {
        req.emplace_back();
        MPI_Isend(f.data() + h.layer0 * row, int((h.layer1 - h.layer0) * row), MPI_DOUBLE, h.rank, 7001, comm_,
                  &req.back());
    }
    MPI_Waitall(int(req.size()), req.data(), MPI_STATUSES_IGNORE);
    t_exchange += MPI_Wtime() - start;
}

void ElementWenoMpi::solve_axis(Field &b) const {
    if (thread_multiple_) {
        solve_axis_threaded(b);
        return;
    }
    const double start = MPI_Wtime();
    const int me = part_.rank;
    const std::size_t len = std::size_t(layer_size_) * b.cols();
    const int C = chunks_ > 0 ? chunks_ : int(std::max<std::size_t>(1, std::min<std::size_t>(4 * part_.ranks, len / 2048)));
    std::vector<std::size_t> cs(C + 1);
    for (int c = 0; c <= C; ++c)
        cs[c] = len * c / C;
    // Message buffers in chunk-major layout: chunk c of position m at npos * cs[c] + m * (cs[c+1] - cs[c]).
    auto alloc = [&](const std::vector<Message> &msgs) {
        std::vector<std::vector<double>> out(msgs.size());
        for (std::size_t l = 0; l < msgs.size(); ++l)
            out[l].resize(msgs[l].positions.size() * len);
        return out;
    };
    auto slot = [&](const std::vector<double> &buf, std::size_t npos, int c, std::size_t m) {
        return const_cast<double *>(buf.data()) + npos * cs[c] + m * (cs[c + 1] - cs[c]);
    };
    const int threads = std::max(1, omp_get_max_threads());
    auto split = [&](std::size_t n, const auto &body) { // element-parallel (all operations are element-wise)
        const int T = int(std::min<std::size_t>(threads, std::max<std::size_t>(1, n / 256)));
        parallel_for(T, [&](int t) { body(n * t / T, n * (t + 1) / T); });
    };
    auto tag = [&](int run, int c, bool backward) { return int((backward ? 1 : 0) + 2 * (c + C * run)) % 30000 + 100; };
    double wait = 0;
    // ---- forward: positions j ascending, y_i -= L_ij y_j ----
    {
        auto rbuf = alloc(fwd_recv_), sbuf = alloc(fwd_send_);
        std::vector<MPI_Request> req;
        for (int c = 0; c < C; ++c) {
            const std::size_t cl = cs[c + 1] - cs[c];
            std::size_t next_recv = 0, next_send = 0;
            for (std::size_t g = 0; g < runs_.size(); ++g) {
                if (runs_[g].owner != me)
                    continue;
                // Foreign columns of earlier runs: receive and apply in run (= position) order.
                while (next_recv < fwd_recv_.size() && fwd_recv_[next_recv].run < int(g)) {
                    const Message &msg = fwd_recv_[next_recv];
                    const std::size_t np = msg.positions.size();
                    const double w0 = MPI_Wtime();
                    MPI_Recv(slot(rbuf[next_recv], np, c, 0), int(np * cl), MPI_DOUBLE, msg.peer, tag(msg.run, c, false),
                             comm_, MPI_STATUS_IGNORE);
                    wait += MPI_Wtime() - w0;
                    const double *base = slot(rbuf[next_recv], np, c, 0);
                    split(cl, [&](std::size_t from, std::size_t to) {
                        for (std::size_t m = 0; m < np; ++m) {
                            const int j = msg.positions[m];
                            const double *yj = base + m * cl;
                            for (int k = outer_[j]; k < outer_[j + 1]; ++k) {
                                const int i = inner_[k];
                                if (i == j || owner_[inv_[i]] != me)
                                    continue;
                                const double v = values_[k];
                                double *yi = layer(b, inv_[i]) + cs[c];
                                for (std::size_t t = from; t < to; ++t) yi[t] -= v * yj[t];
                            }
                        }
                    });
                    ++next_recv;
                }
                split(cl, [&](std::size_t from, std::size_t to) {
                    for (int j = runs_[g].first; j < runs_[g].end; ++j) {
                        const double *yj = layer(b, inv_[j]) + cs[c];
                        for (int k = outer_[j]; k < outer_[j + 1]; ++k) {
                            const int i = inner_[k];
                            if (i == j || owner_[inv_[i]] != me)
                                continue;
                            const double v = values_[k];
                            double *yi = layer(b, inv_[i]) + cs[c];
                            for (std::size_t t = from; t < to; ++t) yi[t] -= v * yj[t];
                        }
                    }
                });
                while (next_send < fwd_send_.size() && fwd_send_[next_send].run == int(g)) {
                    const Message &msg = fwd_send_[next_send];
                    const std::size_t np = msg.positions.size();
                    for (std::size_t m = 0; m < np; ++m)
                        std::copy_n(layer(b, inv_[msg.positions[m]]) + cs[c], cl, slot(sbuf[next_send], np, c, m));
                    req.emplace_back();
                    MPI_Isend(slot(sbuf[next_send], np, c, 0), int(np * cl), MPI_DOUBLE, msg.peer, tag(msg.run, c, false),
                              comm_, &req.back());
                    ++next_send;
                }
            }
            // Columns of later foreign runs never update my rows (L is lower triangular in position order); every
            // forward message to me has been received above.
        }
        MPI_Waitall(int(req.size()), req.data(), MPI_STATUSES_IGNORE);
    }
    // ---- diagonal: fused into the backward sweep (y_j *= 1/D_j right before its backward row; y_j is not read
    // between the forward and the backward sweep, so the arithmetic is the serial one) ----
    // ---- backward: positions j descending, y_j -= L_ij y_i ----
    {
        auto rbuf = alloc(bwd_recv_), sbuf = alloc(bwd_send_);
        std::map<int, std::pair<int, int>> where; // foreign row position -> (message, index)
        for (std::size_t l = 0; l < bwd_recv_.size(); ++l)
            for (std::size_t m = 0; m < bwd_recv_[l].positions.size(); ++m)
                where[bwd_recv_[l].positions[m]] = {int(l), int(m)};
        std::vector<MPI_Request> req;
        for (int c = 0; c < C; ++c) {
            const std::size_t cl = cs[c + 1] - cs[c];
            std::size_t next_recv = 0, next_send = 0;
            for (int g = int(runs_.size()) - 1; g >= 0; --g) {
                if (runs_[g].owner != me)
                    continue;
                while (next_recv < bwd_recv_.size() && bwd_recv_[next_recv].run > g) {
                    const Message &msg = bwd_recv_[next_recv];
                    const std::size_t np = msg.positions.size();
                    const double w0 = MPI_Wtime();
                    MPI_Recv(slot(rbuf[next_recv], np, c, 0), int(np * cl), MPI_DOUBLE, msg.peer, tag(msg.run, c, true),
                             comm_, MPI_STATUS_IGNORE);
                    wait += MPI_Wtime() - w0;
                    ++next_recv;
                }
                split(cl, [&](std::size_t from, std::size_t to) {
                    for (int j = runs_[g].end - 1; j >= runs_[g].first; --j) {
                        double *yj = layer(b, inv_[j]) + cs[c];
                        const double inv = 1. / diag_[j];
                        for (std::size_t t = from; t < to; ++t) yj[t] *= inv;
                        for (int k = outer_[j]; k < outer_[j + 1]; ++k) {
                            const int i = inner_[k];
                            if (i == j)
                                continue;
                            const double v = values_[k];
                            const double *yi;
                            if (owner_[inv_[i]] == me)
                                yi = layer(b, inv_[i]) + cs[c];
                            else {
                                const auto [l, m] = where.at(i);
                                yi = slot(rbuf[l], bwd_recv_[l].positions.size(), c, m);
                            }
                            for (std::size_t t = from; t < to; ++t) yj[t] -= v * yi[t];
                        }
                    }
                });
                while (next_send < bwd_send_.size() && bwd_send_[next_send].run == g) {
                    const Message &msg = bwd_send_[next_send];
                    const std::size_t np = msg.positions.size();
                    for (std::size_t m = 0; m < np; ++m)
                        std::copy_n(layer(b, inv_[msg.positions[m]]) + cs[c], cl, slot(sbuf[next_send], np, c, m));
                    req.emplace_back();
                    MPI_Isend(slot(sbuf[next_send], np, c, 0), int(np * cl), MPI_DOUBLE, msg.peer, tag(msg.run, c, true),
                              comm_, &req.back());
                    ++next_send;
                }
            }
        }
        MPI_Waitall(int(req.size()), req.data(), MPI_STATUSES_IGNORE);
    }
    t_wait += wait;
    t_solve += MPI_Wtime() - start;
}

// Thread-parallel pipeline (MPI_THREAD_MULTIPLE): the columns are cut into blocks of whole grid lines; every thread
// takes its own blocks through the forward and the backward chain over the ranks, with its own messages (tags per
// block). Within a block the operations are those of solve_axis (and of the serial solve_line), so the result is
// bitwise the same; the blocks are independent, so no thread waits for another.
void ElementWenoMpi::solve_axis_threaded(Field &b) const {
    const double start = MPI_Wtime();
    const int me = part_.rank, q = int(b.cols());
    const int T = std::max(1, omp_get_max_threads());
    const int lines = layer_size_;
    const int C = chunks_ > 0 ? chunks_ : std::max(1, std::min(lines, 2 * T)); // blocks of whole lines
    std::vector<std::size_t> cs(C + 1);
    for (int c = 0; c <= C; ++c)
        cs[c] = std::size_t(long(lines) * c / C) * q;
    const std::size_t len = std::size_t(lines) * q;
    auto alloc = [&](const std::vector<Message> &msgs) {
        std::vector<std::vector<double>> out(msgs.size());
        for (std::size_t l = 0; l < msgs.size(); ++l)
            out[l].resize(msgs[l].positions.size() * len);
        return out;
    };
    auto fr = alloc(fwd_recv_), fs = alloc(fwd_send_), br = alloc(bwd_recv_), bs = alloc(bwd_send_);
    auto slot = [&](const std::vector<double> &buf, std::size_t npos, int c, std::size_t m) {
        return const_cast<double *>(buf.data()) + npos * cs[c] + m * (cs[c + 1] - cs[c]);
    };
    std::map<int, std::pair<int, int>> where;
    for (std::size_t l = 0; l < bwd_recv_.size(); ++l)
        for (std::size_t m = 0; m < bwd_recv_[l].positions.size(); ++m)
            where[bwd_recv_[l].positions[m]] = {int(l), int(m)};
    auto tag = [&](int run, int c, bool backward) { return 100 + int((long(run) * C + c) * 2 + backward) % 30000; };
    double wait_total = 0;
#pragma omp parallel num_threads(T) reduction(+ : wait_total)
    {
        std::vector<MPI_Request> req;
#pragma omp for schedule(static, 1) nowait
        for (int c = 0; c < C; ++c) {
            const std::size_t cl = cs[c + 1] - cs[c];
            if (cl == 0)
                continue;
            // forward
            std::size_t next_recv = 0, next_send = 0;
            for (std::size_t g = 0; g < runs_.size(); ++g) {
                if (runs_[g].owner != me)
                    continue;
                while (next_recv < fwd_recv_.size() && fwd_recv_[next_recv].run < int(g)) {
                    const Message &msg = fwd_recv_[next_recv];
                    const std::size_t np = msg.positions.size();
                    const double w0 = MPI_Wtime();
                    MPI_Recv(slot(fr[next_recv], np, c, 0), int(np * cl), MPI_DOUBLE, msg.peer, tag(msg.run, c, false),
                             comm_, MPI_STATUS_IGNORE);
                    wait_total += MPI_Wtime() - w0;
                    const double *base = slot(fr[next_recv], np, c, 0);
                    for (std::size_t m = 0; m < np; ++m) {
                        const int j = msg.positions[m];
                        const double *yj = base + m * cl;
                        for (int k = outer_[j]; k < outer_[j + 1]; ++k) {
                            const int i = inner_[k];
                            if (i == j || owner_[inv_[i]] != me)
                                continue;
                            const double v = values_[k];
                            double *yi = layer(b, inv_[i]) + cs[c];
                            for (std::size_t t = 0; t < cl; ++t) yi[t] -= v * yj[t];
                        }
                    }
                    ++next_recv;
                }
                for (int j = runs_[g].first; j < runs_[g].end; ++j) {
                    const double *yj = layer(b, inv_[j]) + cs[c];
                    for (int k = outer_[j]; k < outer_[j + 1]; ++k) {
                        const int i = inner_[k];
                        if (i == j || owner_[inv_[i]] != me)
                            continue;
                        const double v = values_[k];
                        double *yi = layer(b, inv_[i]) + cs[c];
                        for (std::size_t t = 0; t < cl; ++t) yi[t] -= v * yj[t];
                    }
                }
                while (next_send < fwd_send_.size() && fwd_send_[next_send].run == int(g)) {
                    const Message &msg = fwd_send_[next_send];
                    const std::size_t np = msg.positions.size();
                    for (std::size_t m = 0; m < np; ++m)
                        std::copy_n(layer(b, inv_[msg.positions[m]]) + cs[c], cl, slot(fs[next_send], np, c, m));
                    req.emplace_back();
                    MPI_Isend(slot(fs[next_send], np, c, 0), int(np * cl), MPI_DOUBLE, msg.peer, tag(msg.run, c, false),
                              comm_, &req.back());
                    ++next_send;
                }
            }
            // backward (diagonal fused)
            next_recv = 0, next_send = 0;
            for (int g = int(runs_.size()) - 1; g >= 0; --g) {
                if (runs_[g].owner != me)
                    continue;
                while (next_recv < bwd_recv_.size() && bwd_recv_[next_recv].run > g) {
                    const Message &msg = bwd_recv_[next_recv];
                    const std::size_t np = msg.positions.size();
                    const double w0 = MPI_Wtime();
                    MPI_Recv(slot(br[next_recv], np, c, 0), int(np * cl), MPI_DOUBLE, msg.peer, tag(msg.run, c, true),
                             comm_, MPI_STATUS_IGNORE);
                    wait_total += MPI_Wtime() - w0;
                    ++next_recv;
                }
                for (int j = runs_[g].end - 1; j >= runs_[g].first; --j) {
                    double *yj = layer(b, inv_[j]) + cs[c];
                    const double inv = 1. / diag_[j];
                    for (std::size_t t = 0; t < cl; ++t) yj[t] *= inv;
                    for (int k = outer_[j]; k < outer_[j + 1]; ++k) {
                        const int i = inner_[k];
                        if (i == j)
                            continue;
                        const double v = values_[k];
                        const double *yi;
                        if (owner_[inv_[i]] == me)
                            yi = layer(b, inv_[i]) + cs[c];
                        else {
                            const auto [l, m] = where.at(i);
                            yi = slot(br[l], bwd_recv_[l].positions.size(), c, m);
                        }
                        for (std::size_t t = 0; t < cl; ++t) yj[t] -= v * yi[t];
                    }
                }
                while (next_send < bwd_send_.size() && bwd_send_[next_send].run == g) {
                    const Message &msg = bwd_send_[next_send];
                    const std::size_t np = msg.positions.size();
                    for (std::size_t m = 0; m < np; ++m)
                        std::copy_n(layer(b, inv_[msg.positions[m]]) + cs[c], cl, slot(bs[next_send], np, c, m));
                    req.emplace_back();
                    MPI_Isend(slot(bs[next_send], np, c, 0), int(np * cl), MPI_DOUBLE, msg.peer, tag(msg.run, c, true),
                              comm_, &req.back());
                    ++next_send;
                }
            }
        }
        MPI_Waitall(int(req.size()), req.data(), MPI_STATUSES_IGNORE);
    }
    t_wait += wait_total / T;
    t_solve += MPI_Wtime() - start;
}

Field ElementWenoMpi::gather(const Field &f, std::vector<long> *ids) const {
    const int P = part_.ranks, q = int(f.cols());
    const long row0 = long(O0_ - L0_) * layer_size_, rows = long(O1_ - O0_) * layer_size_;
    Field out;
    if (part_.rank == 0) {
        out.resize(global_nodes, q);
        long at = 0;
        for (int t = 0; t < P; ++t) {
            long n = rows;
            if (t > 0)
                MPI_Recv(&n, 1, MPI_LONG, t, 9000, comm_, MPI_STATUS_IGNORE);
            if (t == 0)
                std::copy_n(f.data() + row0 * q, rows * q, out.data());
            else
                MPI_Recv(out.data() + at * q, int(n * q), MPI_DOUBLE, t, 9001, comm_, MPI_STATUS_IGNORE);
            at += n;
        }
        require(at == global_nodes, "MPI gather: row count mismatch");
        if (ids) {
            ids->resize(global_nodes);
            for (long g = 0; g < global_nodes; ++g)
                (*ids)[g] = g;
        }
    } else {
        long n = rows;
        MPI_Send(&n, 1, MPI_LONG, 0, 9000, comm_);
        MPI_Send(f.data() + row0 * q, int(n * q), MPI_DOUBLE, 0, 9001, comm_);
    }
    return out;
}

} // namespace m1pn
#endif
