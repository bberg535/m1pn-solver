#include "element_weno_mpi.hpp"
#ifdef M1PN_MPI
#include "parallel.hpp"
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <map>
#include <omp.h>
#include <atomic>
#include <mutex>
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
    if (const char *env = std::getenv("M1PN_MPI_CHUNKS"))
        chunks_ = std::atoi(env);
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
        tasks_ = !std::getenv("M1PN_MPI_CHUNK_SPLIT");
        static_ = tasks_ && !std::getenv("M1PN_MPI_TASKS"); // M1PN_MPI_TASKS: the OpenMP task scheduler instead // M1PN_MPI_CHUNK_SPLIT: the earlier block-split pipeline
    }
    // ---- partitioned (Schur) solver setup: identical on every rank (global 1D data) ----
    // Opt-in (M1PN_MPI_SCHUR=1). NX: the Allreduce of whole separator layers costs 18-54 s, and the rounding
    // differences grow chaotically in the limited scheme to 1e-8..1e-5 after 455-910 steps; the bitwise pipeline is
    // both faster and exact (docs/ELEMENT_WENO_MPI.md).
    schur_ = std::getenv("M1PN_MPI_SCHUR") && P > 1;
    if (schur_) {
        // dense band of the global 1D mass (same assembly as above)
        const Mat &m1 = s_.m1_reference_;
        bw_ = p_;
        band_.assign(std::size_t(n1) * (2 * bw_ + 1), 0.);
        for (int e = 0; e < E; ++e)
            for (int i = 0; i <= p_; ++i)
                for (int j = 0; j <= p_; ++j)
                    band_[std::size_t(e * p_ + i) * (2 * bw_ + 1) + (j - i + bw_)] += s_.h * m1(i, j);
        ns_ = (P - 1) * bw_;
        auto interior = [&](int t) {
            const auto [a, b] = owned_range(t);
            return std::make_pair(t > 0 ? a + bw_ : a, b);
        };
        for (int t = 0; t < P; ++t) {
            const auto [a, b] = interior(t);
            require(b - a >= bw_ + 1, "MPI: every rank needs more owned layers than the bandwidth");
        }
        // banded LDL^T of an interior block [a, b)
        auto factor = [&](int a, int b, std::vector<double> &L, std::vector<double> &D) {
            const int n = b - a, w = bw_;
            L.assign(std::size_t(n) * w, 0.), D.assign(n, 0.);
            for (int i = 0; i < n; ++i) {
                for (int j = std::max(0, i - w); j < i; ++j) {
                    double v = band(a + i, a + j);
                    for (int k = std::max(0, i - w); k < j; ++k)
                        v -= L[std::size_t(i) * w + (k - i + w)] * L[std::size_t(j) * w + (k - j + w)] * D[k];
                    L[std::size_t(i) * w + (j - i + w)] = v / D[j];
                }
                double v = band(a + i, a + i);
                for (int k = std::max(0, i - w); k < i; ++k)
                    v -= L[std::size_t(i) * w + (k - i + w)] * L[std::size_t(i) * w + (k - i + w)] * D[k];
                D[i] = v;
            }
        };
        auto solve = [&](const std::vector<double> &L, const std::vector<double> &D, std::vector<double> &x) {
            const int n = int(D.size()), w = bw_;
            for (int i = 0; i < n; ++i)
                for (int k = std::max(0, i - w); k < i; ++k)
                    x[i] -= L[std::size_t(i) * w + (k - i + w)] * x[k];
            for (int i = 0; i < n; ++i)
                x[i] /= D[i];
            for (int i = n - 1; i >= 0; --i)
                for (int k = std::max(0, i - w); k < i; ++k)
                    x[k] -= L[std::size_t(i) * w + (k - i + w)] * x[i];
        };
        Mat S = Mat::Zero(ns_, ns_);
        for (int u = 0; u < ns_; ++u)
            for (int v = 0; v < ns_; ++v)
                S(u, v) = band(sep_layer(u), sep_layer(v));
        for (int t = 0; t < P; ++t) {
            const auto [a, b] = interior(t);
            std::vector<double> L, D;
            factor(a, b, L, D);
            std::vector<int> adj; // separator unknowns coupled to [a, b)
            for (int u = 0; u < ns_; ++u) {
                const int k = sep_layer(u);
                if (k >= a - bw_ && k < b + bw_)
                    adj.push_back(u);
            }
            std::vector<double> W(std::size_t(b - a) * adj.size());
            for (std::size_t m = 0; m < adj.size(); ++m) {
                std::vector<double> x(b - a, 0.);
                for (int i = a; i < b; ++i)
                    x[i - a] = band(i, sep_layer(adj[m]));
                solve(L, D, x);
                for (int i = a; i < b; ++i)
                    W[std::size_t(i - a) * adj.size() + m] = x[i - a];
                for (std::size_t mu = 0; mu < adj.size(); ++mu) { // S -= M_{S,I} W
                    double v = 0;
                    for (int i = a; i < b; ++i)
                        v += band(sep_layer(adj[mu]), i) * x[i - a];
                    S(adj[mu], adj[m]) -= v;
                }
            }
            if (t == r) {
                I0_ = a, I1_ = b, iL_ = L, iD_ = D, adj_ = adj, spike_ = W;
            }
        }
        schur_inverse_ = S.inverse();
        require(schur_inverse_.allFinite(), "MPI: Schur complement of the line mass not invertible");
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
double ElementWenoMpi::ordered_sum(const std::vector<double> &owned) const {
    const int P = part_.ranks;
    int count = int(owned.size());
    std::vector<int> counts(P), offsets(P + 1, 0);
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm_);
    for (int r = 0; r < P; ++r)
        offsets[r + 1] = offsets[r] + counts[r];
    std::vector<double> all(offsets[P]);
    MPI_Allgatherv(owned.data(), count, MPI_DOUBLE, all.data(), counts.data(), offsets.data(), MPI_DOUBLE, comm_);
    double total = 0;
    for (double v : all) // rank order = global layer order
        total += v;
    return total;
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
    if (schur_) {
        solve_axis_schur(b);
        return;
    }
    if (thread_multiple_) {
        solve_axis_threaded(b);
        return;
    }
    if (static_) {
        solve_axis_static(b);
        return;
    }
    if (tasks_) {
        solve_axis_tasks(b);
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

// Task pipeline. The columns are cut into C blocks of whole grid lines (C ~ 3 threads). For every block the forward
// sweep runs through this rank's runs in order, then the backward sweep in reverse order (diagonal fused); a phase
// may start once the messages it reads have arrived. One thread (the master) posts all receives up front, tests them,
// starts ready phases as OpenMP tasks and sends what finished phases produced; the other threads execute the tasks.
// Within a block the operations are those of the serial solve_line, so the result is bitwise the serial one.
void ElementWenoMpi::solve_axis_overlap(Field &b, int count, const std::function<void(int)> &filler) const {
    if (static_ && !thread_multiple_ && !schur_) {
        solve_axis_static(b, count, &filler);
        return;
    }
    if (tasks_ && !thread_multiple_ && !schur_) {
        solve_axis_tasks(b, count, &filler);
        return;
    }
    parallel_for(count, [&](int i) { filler(i); });
    solve_axis(b);
}

void ElementWenoMpi::solve_axis_tasks(Field &b, int fillers, const std::function<void(int)> *filler) const {
    const double start = MPI_Wtime();
    const int me = part_.rank, q = int(b.cols());
    const int T = std::max(1, omp_get_max_threads());
    const int lines = layer_size_;
    // Blocks of whole lines. Trapezoidal sizes (default): the first and the last blocks are small (1, 2, 4, ...
    // lines) so that the chain over the ranks fills and drains quickly; the middle blocks have the base size.
    const int base = std::max(1, chunks_ > 0 ? lines / chunks_ : lines / std::max(1, 3 * T));
    std::vector<int> sizes;
    if (std::getenv("M1PN_MPI_NO_RAMP") || chunks_ > 0) {
        const int C0 = std::max(1, std::min(lines, lines / base));
        for (int c = 0; c < C0; ++c)
            sizes.push_back(int(long(lines) * (c + 1) / C0 - long(lines) * c / C0));
    } else {
        std::vector<int> ramp;
        for (int r = 1; r < base; r *= 2)
            ramp.push_back(r);
        int rest = lines;
        std::vector<int> head, tail;
        for (int r : ramp)
            if (rest >= 2 * r + base)
                head.push_back(r), tail.push_back(r), rest -= 2 * r;
        const int middle = std::max(1, rest / base);
        sizes = head;
        for (int c = 0; c < middle; ++c)
            sizes.push_back(int(long(rest) * (c + 1) / middle - long(rest) * c / middle));
        sizes.insert(sizes.end(), tail.rbegin(), tail.rend());
    }
    const int C = int(sizes.size());
    std::vector<std::size_t> cs(C + 1, 0);
    for (int c = 0; c < C; ++c)
        cs[c + 1] = cs[c] + std::size_t(sizes[c]) * q;
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
    // Phases of a block: forward over my runs (ascending), then backward over my runs (descending).
    std::vector<int> mine;
    for (std::size_t g = 0; g < runs_.size(); ++g)
        if (runs_[g].owner == me)
            mine.push_back(int(g));
    const int R = int(mine.size()), phases = 2 * R;
    // Receives: all posted up front; arrived[dir][msg][c].
    const int nf = int(fwd_recv_.size()), nb = int(bwd_recv_.size());
    std::vector<MPI_Request> rreq;
    std::vector<std::array<int, 3>> rinfo; // dir, msg, chunk
    for (int c = 0; c < C; ++c) {
        const std::size_t cl = cs[c + 1] - cs[c];
        for (int l = 0; l < nf; ++l) {
            const std::size_t np = fwd_recv_[l].positions.size();
            rreq.emplace_back();
            MPI_Irecv(slot(fr[l], np, c, 0), int(np * cl), MPI_DOUBLE, fwd_recv_[l].peer, tag(fwd_recv_[l].run, c, false),
                      comm_, &rreq.back());
            rinfo.push_back({0, l, c});
        }
        for (int l = 0; l < nb; ++l) {
            const std::size_t np = bwd_recv_[l].positions.size();
            rreq.emplace_back();
            MPI_Irecv(slot(br[l], np, c, 0), int(np * cl), MPI_DOUBLE, bwd_recv_[l].peer, tag(bwd_recv_[l].run, c, true),
                      comm_, &rreq.back());
            rinfo.push_back({1, l, c});
        }
    }
    // Missing messages per (block, phase): phase k of block c may start when missing[c * phases + k] == 0.
    std::vector<int> missing(std::size_t(C) * phases, 0);
    auto phase_of_recv = [&](int dir, int l, int k) { // does phase k wait for message l of direction dir?
        if (dir == 0)
            return k < R && fwd_recv_[l].run < mine[k];
        return k >= R && bwd_recv_[l].run > mine[phases - 1 - k];
    };
    for (int c = 0; c < C; ++c)
        for (int k = 0; k < phases; ++k) {
            int m = 0;
            for (int l = 0; l < nf; ++l)
                m += phase_of_recv(0, l, k);
            for (int l = 0; l < nb; ++l)
                m += phase_of_recv(1, l, k);
            missing[std::size_t(c) * phases + k] = m;
        }
    auto ready = [&](int c, int k) { return missing[std::size_t(c) * phases + k] == 0; };
    // Execute phase k of block c (any thread).
    std::vector<std::vector<char>> applied(C, std::vector<char>(nf, 0));
    auto execute = [&](int c, int k) {
        const std::size_t cl = cs[c + 1] - cs[c];
        if (k < R) {
            const int g = mine[k];
            for (int l = 0; l < nf; ++l) { // foreign columns of earlier runs, in run (= position) order
                const Message &msg = fwd_recv_[l];
                if (msg.run >= g || applied[c][l])
                    continue;
                applied[c][l] = 1;
                const std::size_t np = msg.positions.size();
                const double *base = slot(fr[l], np, c, 0);
                for (std::size_t m = 0; m < np; ++m) {
                    const int j = msg.positions[m];
                    const double *yj = base + m * cl;
                    for (int kk = outer_[j]; kk < outer_[j + 1]; ++kk) {
                        const int i = inner_[kk];
                        if (i == j || owner_[inv_[i]] != me)
                            continue;
                        const double v = values_[kk];
                        double *yi = layer(b, inv_[i]) + cs[c];
                        for (std::size_t t = 0; t < cl; ++t) yi[t] -= v * yj[t];
                    }
                }
            }
            for (int j = runs_[g].first; j < runs_[g].end; ++j) {
                const double *yj = layer(b, inv_[j]) + cs[c];
                for (int kk = outer_[j]; kk < outer_[j + 1]; ++kk) {
                    const int i = inner_[kk];
                    if (i == j || owner_[inv_[i]] != me)
                        continue;
                    const double v = values_[kk];
                    double *yi = layer(b, inv_[i]) + cs[c];
                    for (std::size_t t = 0; t < cl; ++t) yi[t] -= v * yj[t];
                }
            }
            for (std::size_t l = 0; l < fwd_send_.size(); ++l)
                if (fwd_send_[l].run == g) {
                    const auto &pos = fwd_send_[l].positions;
                    for (std::size_t m = 0; m < pos.size(); ++m)
                        std::copy_n(layer(b, inv_[pos[m]]) + cs[c], cl, slot(fs[l], pos.size(), c, m));
                }
        } else {
            const int g = mine[phases - 1 - k];
            for (int j = runs_[g].end - 1; j >= runs_[g].first; --j) {
                double *yj = layer(b, inv_[j]) + cs[c];
                const double inv = 1. / diag_[j];
                for (std::size_t t = 0; t < cl; ++t) yj[t] *= inv;
                for (int kk = outer_[j]; kk < outer_[j + 1]; ++kk) {
                    const int i = inner_[kk];
                    if (i == j)
                        continue;
                    const double v = values_[kk];
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
            for (std::size_t l = 0; l < bwd_send_.size(); ++l)
                if (bwd_send_[l].run == g) {
                    const auto &pos = bwd_send_[l].positions;
                    for (std::size_t m = 0; m < pos.size(); ++m)
                        std::copy_n(layer(b, inv_[pos[m]]) + cs[c], cl, slot(bs[l], pos.size(), c, m));
                }
        }
    };
    std::vector<int> next(C, 0); // next phase of each block
    std::vector<char> running(C, 0);
    std::mutex finished_lock;
    std::vector<int> finished_blocks, touched(C);
    for (int c = 0; c < C; ++c)
        touched[c] = c;
    std::vector<MPI_Request> sreq;
    double wait = 0;
    if (R == 0) { // owns no positions (cannot happen with owned layers)
        parallel_for(fillers, [&](int i) { (*filler)(i); });
        MPI_Waitall(int(rreq.size()), rreq.data(), MPI_STATUSES_IGNORE);
        return;
    }
#pragma omp parallel num_threads(T)
#pragma omp single
    {
        // Fillers first, at low priority: idle workers take them, pipeline phases (priority 1) go ahead.
        for (int i = 0; i < fillers; ++i) {
#pragma omp task firstprivate(i) priority(0)
            (*filler)(i);
        }
        int finished = 0;
        std::vector<int> idx(rreq.size());
        while (finished < C) {
            bool progress = false;
            // arrivals: decrement the missing counts of the phases that wait for them
            if (!rreq.empty()) {
                int count = 0;
                MPI_Testsome(int(rreq.size()), rreq.data(), &count, idx.data(), MPI_STATUSES_IGNORE);
                if (count > 0 && count != MPI_UNDEFINED) {
                    progress = true;
                    for (int n = 0; n < count; ++n) {
                        const auto &info = rinfo[idx[n]];
                        for (int k = 0; k < phases; ++k)
                            if (phase_of_recv(info[0], info[1], k))
                                --missing[std::size_t(info[2]) * phases + k];
                        touched.push_back(info[2]);
                    }
                }
            }
            // finished phases (tasks append their block to `finished_blocks` under a lock)
            {
                std::vector<int> fin;
                {
                    const std::lock_guard<std::mutex> lock(finished_lock);
                    fin.swap(finished_blocks);
                }
                for (int c : fin) {
                    running[c] = 0;
                    const int k = next[c]++;
                    const bool backward = k >= R;
                    const int g = backward ? mine[phases - 1 - k] : mine[k];
                    const auto &msgs = backward ? bwd_send_ : fwd_send_;
                    auto &bufs = backward ? bs : fs;
                    const std::size_t cl = cs[c + 1] - cs[c];
                    for (std::size_t l = 0; l < msgs.size(); ++l)
                        if (msgs[l].run == g) {
                            const std::size_t np = msgs[l].positions.size();
                            sreq.emplace_back();
                            MPI_Isend(slot(bufs[l], np, c, 0), int(np * cl), MPI_DOUBLE, msgs[l].peer,
                                      tag(g, c, backward), comm_, &sreq.back());
                        }
                    if (next[c] == phases)
                        ++finished;
                    touched.push_back(c);
                    progress = true;
                }
            }
            // start ready phases of the touched blocks
            for (int c : touched)
                if (!running[c] && next[c] < phases && ready(c, next[c])) {
                    running[c] = 1;
                    const int k = next[c];
#pragma omp task firstprivate(c, k) priority(1)
                    {
                        execute(c, k);
                        const std::lock_guard<std::mutex> lock(finished_lock);
                        finished_blocks.push_back(c);
                    }
                }
            touched.clear();
            if (!progress) {
                const double w0 = MPI_Wtime();
#pragma omp taskyield
                wait += MPI_Wtime() - w0;
            }
        }
#pragma omp taskwait
    }
    MPI_Waitall(int(rreq.size()), rreq.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(int(sreq.size()), sreq.data(), MPI_STATUSES_IGNORE);
    t_wait += wait;
    t_solve += MPI_Wtime() - start;
}

// Partitioned solve along the axis (see schur_ in the header). Per call: interior solves (line-parallel, in
// cache), the separator right-hand side g = b_S - M_SI z_I (one Allreduce over the ranks), x_S = S^-1 g on every
// rank, and x_I = z_I - W x_S.
void ElementWenoMpi::solve_axis_schur(Field &b) const {
    const double start = MPI_Wtime();
    const int q = int(b.cols()), lines = layer_size_, n = I1_ - I0_, w = bw_;
    const std::size_t K = std::size_t(lines) * q;
    // 1. interior: z = M_II^-1 b_I, line by line
    parallel_for(lines, [&](int t) {
        thread_local std::vector<double> work;
        work.resize(std::size_t(n) * q);
        for (int i = 0; i < n; ++i)
            std::copy_n(layer(b, I0_ + i) + std::size_t(t) * q, q, &work[std::size_t(i) * q]);
        for (int i = 0; i < n; ++i) {
            double *xi = &work[std::size_t(i) * q];
            for (int k = std::max(0, i - w); k < i; ++k) {
                const double l = iL_[std::size_t(i) * w + (k - i + w)];
                const double *xk = &work[std::size_t(k) * q];
                for (int c = 0; c < q; ++c) xi[c] -= l * xk[c];
            }
        }
        for (int i = 0; i < n; ++i) {
            const double inv = 1. / iD_[i];
            double *xi = &work[std::size_t(i) * q];
            for (int c = 0; c < q; ++c) xi[c] *= inv;
        }
        for (int i = n - 1; i >= 0; --i) {
            const double *xi = &work[std::size_t(i) * q];
            for (int k = std::max(0, i - w); k < i; ++k) {
                const double l = iL_[std::size_t(i) * w + (k - i + w)];
                double *xk = &work[std::size_t(k) * q];
                for (int c = 0; c < q; ++c) xk[c] -= l * xi[c];
            }
        }
        for (int i = 0; i < n; ++i)
            std::copy_n(&work[std::size_t(i) * q], q, layer(b, I0_ + i) + std::size_t(t) * q);
    });
    // 2. separator right-hand side: own separator rows (b_S) minus M_{S,I} z_I near my interior
    std::vector<double> g(std::size_t(ns_) * K, 0.);
    const int me = part_.rank;
    for (int u : adj_) {
        const int ks = sep_layer(u);
        double *gu = g.data() + std::size_t(u) * K;
        const bool mine = owner_[ks] == me;
        parallel_for(int((K + 4095) / 4096), [&](int blk) {
            const std::size_t from = std::size_t(blk) * 4096, to = std::min(K, from + 4096);
            if (mine) {
                const double *bs = layer(b, ks);
                for (std::size_t x = from; x < to; ++x) gu[x] = bs[x];
            }
            for (int i = std::max(I0_, ks - w); i < std::min(I1_, ks + w + 1); ++i) {
                const double m = band(ks, i);
                if (m == 0)
                    continue;
                const double *zi = layer(b, i);
                for (std::size_t x = from; x < to; ++x) gu[x] -= m * zi[x];
            }
        });
    }
    const double c0 = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, g.data(), int(g.size()), MPI_DOUBLE, MPI_SUM, comm_);
    t_wait += MPI_Wtime() - c0;
    // 3. x_S = S^-1 g (small dense, every rank), only the unknowns I need: adjacent ones and my own separator
    std::vector<int> need = adj_;
    std::sort(need.begin(), need.end());
    std::vector<double> xs(need.size() * K);
    parallel_for(int((K + 4095) / 4096), [&](int blk) {
        const std::size_t from = std::size_t(blk) * 4096, to = std::min(K, from + 4096);
        for (std::size_t m = 0; m < need.size(); ++m) {
            double *out = &xs[m * K];
            for (std::size_t x = from; x < to; ++x) out[x] = 0;
            for (int v = 0; v < ns_; ++v) {
                const double a = schur_inverse_(need[m], v);
                const double *gv = g.data() + std::size_t(v) * K;
                for (std::size_t x = from; x < to; ++x) out[x] += a * gv[x];
            }
        }
    });
    // 4. interior correction x_I = z_I - W x_S, and my separator layers
    std::vector<int> slot(ns_, -1);
    for (std::size_t m = 0; m < need.size(); ++m)
        slot[need[m]] = int(m);
    const std::size_t na = adj_.size();
    parallel_for(n, [&](int i) {
        double *xi = layer(b, I0_ + i);
        for (std::size_t m = 0; m < na; ++m) {
            const double wv = spike_[std::size_t(i) * na + m];
            if (wv == 0)
                continue;
            const double *x = &xs[std::size_t(slot[adj_[m]]) * K];
            for (std::size_t c = 0; c < K; ++c) xi[c] -= wv * x[c];
        }
    });
    for (std::size_t m = 0; m < need.size(); ++m) {
        const int ks = sep_layer(need[m]);
        if (owner_[ks] == me)
            std::copy_n(&xs[m * K], K, layer(b, ks));
    }
    t_solve += MPI_Wtime() - start;
}

// Static variant of the task pipeline (default): worker thread w owns blocks c = w, w + W, ...; it runs forward and
// backward of its blocks in a contiguous block buffer (owned layers x block width, as compact as the serial line
// work array), so a block stays in the cache of one core / CCD. The master thread (0) does all MPI and releases
// phases through atomic flags; waiting workers take filler work. Same arithmetic as the serial solve_line.
void ElementWenoMpi::solve_axis_static(Field &b, int fillers, const std::function<void(int)> *filler) const {
    const double start = MPI_Wtime();
    const int me = part_.rank, q = int(b.cols());
    const int T = std::max(1, omp_get_max_threads());
    const int lines = layer_size_;
    // Blocks of whole lines. Trapezoidal sizes (default): the first and the last blocks are small (1, 2, 4, ...
    // lines) so that the chain over the ranks fills and drains quickly; the middle blocks have the base size.
    const int base = std::max(1, chunks_ > 0 ? lines / chunks_ : lines / std::max(1, 3 * T));
    std::vector<int> sizes;
    if (std::getenv("M1PN_MPI_NO_RAMP") || chunks_ > 0) {
        const int C0 = std::max(1, std::min(lines, lines / base));
        for (int c = 0; c < C0; ++c)
            sizes.push_back(int(long(lines) * (c + 1) / C0 - long(lines) * c / C0));
    } else {
        std::vector<int> ramp;
        for (int r = 1; r < base; r *= 2)
            ramp.push_back(r);
        int rest = lines;
        std::vector<int> head, tail;
        for (int r : ramp)
            if (rest >= 2 * r + base)
                head.push_back(r), tail.push_back(r), rest -= 2 * r;
        const int middle = std::max(1, rest / base);
        sizes = head;
        for (int c = 0; c < middle; ++c)
            sizes.push_back(int(long(rest) * (c + 1) / middle - long(rest) * c / middle));
        sizes.insert(sizes.end(), tail.rbegin(), tail.rend());
    }
    const int C = int(sizes.size());
    std::vector<std::size_t> cs(C + 1, 0);
    for (int c = 0; c < C; ++c)
        cs[c + 1] = cs[c] + std::size_t(sizes[c]) * q;
    const std::size_t len = std::size_t(lines) * q;
    auto alloc = [&](const std::vector<Message> &msgs) {
        std::vector<std::vector<double>> out(msgs.size());
        for (std::size_t l = 0; l < msgs.size(); ++l)
            out[l].resize(msgs[l].positions.size() * len);
        return out;
    };
    const double setup_start = MPI_Wtime();
    auto pooled = [&](const std::vector<Message> &msgs, std::vector<std::vector<double>> &pool) -> std::vector<std::vector<double>> & {
        if (pool.size() < msgs.size())
            pool.resize(msgs.size());
        for (std::size_t l = 0; l < msgs.size(); ++l)
            if (pool[l].size() < msgs[l].positions.size() * len)
                pool[l].resize(msgs[l].positions.size() * len);
        return pool;
    };
    auto &fr = pooled(fwd_recv_, pool_fr_), &fs = pooled(fwd_send_, pool_fs_), &br = pooled(bwd_recv_, pool_br_),
         &bs = pooled(bwd_send_, pool_bs_);
    auto slot = [&](const std::vector<double> &buf, std::size_t npos, int c, std::size_t m) {
        return const_cast<double *>(buf.data()) + npos * cs[c] + m * (cs[c + 1] - cs[c]);
    };
    std::map<int, std::pair<int, int>> where;
    for (std::size_t l = 0; l < bwd_recv_.size(); ++l)
        for (std::size_t m = 0; m < bwd_recv_[l].positions.size(); ++m)
            where[bwd_recv_[l].positions[m]] = {int(l), int(m)};
    auto tag = [&](int run, int c, bool backward) { return 100 + int((long(run) * C + c) * 2 + backward) % 30000; };
    // Phases of a block: forward over my runs (ascending), then backward over my runs (descending).
    std::vector<int> mine;
    for (std::size_t g = 0; g < runs_.size(); ++g)
        if (runs_[g].owner == me)
            mine.push_back(int(g));
    const int R = int(mine.size()), phases = 2 * R;
    // Receives: all posted up front; arrived[dir][msg][c].
    const int nf = int(fwd_recv_.size()), nb = int(bwd_recv_.size());
    std::vector<MPI_Request> rreq;
    std::vector<std::array<int, 3>> rinfo; // dir, msg, chunk
    for (int c = 0; c < C; ++c) {
        const std::size_t cl = cs[c + 1] - cs[c];
        for (int l = 0; l < nf; ++l) {
            const std::size_t np = fwd_recv_[l].positions.size();
            rreq.emplace_back();
            MPI_Irecv(slot(fr[l], np, c, 0), int(np * cl), MPI_DOUBLE, fwd_recv_[l].peer, tag(fwd_recv_[l].run, c, false),
                      comm_, &rreq.back());
            rinfo.push_back({0, l, c});
        }
        for (int l = 0; l < nb; ++l) {
            const std::size_t np = bwd_recv_[l].positions.size();
            rreq.emplace_back();
            MPI_Irecv(slot(br[l], np, c, 0), int(np * cl), MPI_DOUBLE, bwd_recv_[l].peer, tag(bwd_recv_[l].run, c, true),
                      comm_, &rreq.back());
            rinfo.push_back({1, l, c});
        }
    }
    // Missing messages per (block, phase): phase k of block c may start when missing[c * phases + k] == 0.
    std::vector<int> missing(std::size_t(C) * phases, 0);
    auto phase_of_recv = [&](int dir, int l, int k) { // does phase k wait for message l of direction dir?
        if (dir == 0)
            return k < R && fwd_recv_[l].run < mine[k];
        return k >= R && bwd_recv_[l].run > mine[phases - 1 - k];
    };
    for (int c = 0; c < C; ++c)
        for (int k = 0; k < phases; ++k) {
            int m = 0;
            for (int l = 0; l < nf; ++l)
                m += phase_of_recv(0, l, k);
            for (int l = 0; l < nb; ++l)
                m += phase_of_recv(1, l, k);
            missing[std::size_t(c) * phases + k] = m;
        }
    auto ready_phase = [&](int c, int k) { return missing[std::size_t(c) * phases + k] == 0; };
    // Block buffers: owned layers [O0, O1) x block width, copied in before the first and out after the last phase.
    if (pool_work_.size() < std::size_t(C))
        pool_work_.resize(C);
    auto &work = pool_work_;
    for (int c = 0; c < C; ++c)
        if (work[c].size() < std::size_t(O1_ - O0_) * (cs[c + 1] - cs[c]))
            work[c].resize(std::size_t(O1_ - O0_) * (cs[c + 1] - cs[c]));
    auto wl = [&](int c, int global_layer) { return work[c].data() + std::size_t(global_layer - O0_) * (cs[c + 1] - cs[c]); };
    // Execute phase k of block c (the owning worker).
    std::vector<std::vector<char>> applied(C, std::vector<char>(nf, 0));
    auto execute = [&](int c, int k) {
        const std::size_t cl = cs[c + 1] - cs[c];
        if (k == 0) {
            for (int g = O0_; g < O1_; ++g)
                std::copy_n(layer(b, g) + cs[c], cl, wl(c, g));
        }
        if (k < R) {
            const int g = mine[k];
            for (int l = 0; l < nf; ++l) { // foreign columns of earlier runs, in run (= position) order
                const Message &msg = fwd_recv_[l];
                if (msg.run >= g || applied[c][l])
                    continue;
                applied[c][l] = 1;
                const std::size_t np = msg.positions.size();
                const double *base = slot(fr[l], np, c, 0);
                for (std::size_t m = 0; m < np; ++m) {
                    const int j = msg.positions[m];
                    const double *yj = base + m * cl;
                    for (int kk = outer_[j]; kk < outer_[j + 1]; ++kk) {
                        const int i = inner_[kk];
                        if (i == j || owner_[inv_[i]] != me)
                            continue;
                        const double v = values_[kk];
                        double *yi = wl(c, inv_[i]);
                        for (std::size_t t = 0; t < cl; ++t) yi[t] -= v * yj[t];
                    }
                }
            }
            for (int j = runs_[g].first; j < runs_[g].end; ++j) {
                const double *yj = wl(c, inv_[j]);
                for (int kk = outer_[j]; kk < outer_[j + 1]; ++kk) {
                    const int i = inner_[kk];
                    if (i == j || owner_[inv_[i]] != me)
                        continue;
                    const double v = values_[kk];
                    double *yi = wl(c, inv_[i]);
                    for (std::size_t t = 0; t < cl; ++t) yi[t] -= v * yj[t];
                }
            }
            for (std::size_t l = 0; l < fwd_send_.size(); ++l)
                if (fwd_send_[l].run == g) {
                    const auto &pos = fwd_send_[l].positions;
                    for (std::size_t m = 0; m < pos.size(); ++m)
                        std::copy_n(wl(c, inv_[pos[m]]), cl, slot(fs[l], pos.size(), c, m));
                }
        } else {
            const int g = mine[phases - 1 - k];
            for (int j = runs_[g].end - 1; j >= runs_[g].first; --j) {
                double *yj = wl(c, inv_[j]);
                const double inv = 1. / diag_[j];
                for (std::size_t t = 0; t < cl; ++t) yj[t] *= inv;
                for (int kk = outer_[j]; kk < outer_[j + 1]; ++kk) {
                    const int i = inner_[kk];
                    if (i == j)
                        continue;
                    const double v = values_[kk];
                    const double *yi;
                    if (owner_[inv_[i]] == me)
                        yi = wl(c, inv_[i]);
                    else {
                        const auto [l, m] = where.at(i);
                        yi = slot(br[l], bwd_recv_[l].positions.size(), c, m);
                    }
                    for (std::size_t t = 0; t < cl; ++t) yj[t] -= v * yi[t];
                }
            }
            for (std::size_t l = 0; l < bwd_send_.size(); ++l)
                if (bwd_send_[l].run == g) {
                    const auto &pos = bwd_send_[l].positions;
                    for (std::size_t m = 0; m < pos.size(); ++m)
                        std::copy_n(wl(c, inv_[pos[m]]), cl, slot(bs[l], pos.size(), c, m));
                }
        }
        if (k == phases - 1)
            for (int g = O0_; g < O1_; ++g)
                std::copy_n(wl(c, g), cl, layer(b, g) + cs[c]);
    };
    std::vector<int> next(C, 0); // next phase of each block

    std::vector<MPI_Request> sreq;
    double wait = 0;
    if (R == 0) { // owns no positions (cannot happen with owned layers)
        parallel_for(fillers, [&](int i) { (*filler)(i); });
        MPI_Waitall(int(rreq.size()), rreq.data(), MPI_STATUSES_IGNORE);
        return;
    }
    const int Wk = std::max(1, T - 1);
    std::vector<std::atomic<int>> ready(C), done(C);
    for (int c = 0; c < C; ++c)
        ready[c].store(0), done[c].store(0);
    std::atomic<int> filler_next{0};
    // Workers call this in their wait loops (possibly for minutes while a neighbour rank is busy): increment only while
    // fillers remain, otherwise the counter overflows int after 2^31 calls, turns negative and runs filler(i < 0).
    auto run_filler = [&]() {
        if (filler_next.load(std::memory_order_relaxed) >= fillers)
            return false;
        const int i = filler_next.fetch_add(1);
        if (i >= fillers)
            return false;
        (*filler)(i);
        return true;
    };
    std::vector<int> released(C, 0), seen(C, 0);
    auto release = [&](int c) {
        if (next[c] < phases && released[c] == next[c] && ready_phase(c, next[c])) {
            released[c] = next[c] + 1;
            ready[c].store(released[c], std::memory_order_release);
        }
    };
    double exec_total = 0;
    const double loop_start = MPI_Wtime();
    t_setup += loop_start - setup_start;
#pragma omp parallel num_threads(T) reduction(+ : exec_total)
    {
        const int tid = omp_get_thread_num();
        if (tid == 0) {
            int finished = 0;
            std::vector<int> idx(rreq.size());
            for (int c = 0; c < C; ++c)
                release(c);
            while (finished < C) {
                bool progress = false;
                if (!rreq.empty()) {
                    int count = 0;
                    MPI_Testsome(int(rreq.size()), rreq.data(), &count, idx.data(), MPI_STATUSES_IGNORE);
                    if (count > 0 && count != MPI_UNDEFINED) {
                        progress = true;
                        for (int n = 0; n < count; ++n) {
                            const auto &info = rinfo[idx[n]];
                            for (int k = 0; k < phases; ++k)
                                if (phase_of_recv(info[0], info[1], k))
                                    --missing[std::size_t(info[2]) * phases + k];
                            release(info[2]);
                        }
                    }
                }
                for (int c = 0; c < C; ++c) {
                    const int d = done[c].load(std::memory_order_acquire);
                    while (seen[c] < d) { // phase seen[c] of block c finished: send its products
                        const int k = seen[c]++;
                        next[c] = k + 1;
                        const bool backward = k >= R;
                        const int g = backward ? mine[phases - 1 - k] : mine[k];
                        const auto &msgs = backward ? bwd_send_ : fwd_send_;
                        auto &bufs = backward ? bs : fs;
                        const std::size_t cl = cs[c + 1] - cs[c];
                        for (std::size_t l = 0; l < msgs.size(); ++l)
                            if (msgs[l].run == g) {
                                const std::size_t np = msgs[l].positions.size();
                                sreq.emplace_back();
                                MPI_Isend(slot(bufs[l], np, c, 0), int(np * cl), MPI_DOUBLE, msgs[l].peer,
                                          tag(g, c, backward), comm_, &sreq.back());
                            }
                        if (next[c] == phases)
                            ++finished;
                        release(c);
                        progress = true;
                    }
                }
                if (!progress && T == 1) { // no workers: the master runs the released phases itself
                    for (int c = 0; c < C; ++c)
                        if (ready[c].load() > done[c].load()) {
                            execute(c, done[c].load());
                            done[c].store(done[c].load() + 1);
                        }
                    while (run_filler()) {
                    }
                }
            }
        } else {
            // worker: its blocks, phase-major (all forwards first, then the backwards)
            for (int k = 0; k < phases; ++k)
                for (int c = tid - 1; c < C; c += Wk) {
                    while (ready[c].load(std::memory_order_acquire) < k + 1)
                        if (!run_filler()) {
                        }
                    const double e0 = MPI_Wtime();
                    execute(c, k);
                    exec_total += MPI_Wtime() - e0;
                    done[c].store(k + 1, std::memory_order_release);
                }
            while (run_filler()) {
            }
        }
    }
    t_loop += MPI_Wtime() - loop_start;
    t_exec += exec_total;
    ++n_solves;
    MPI_Waitall(int(rreq.size()), rreq.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(int(sreq.size()), sreq.data(), MPI_STATUSES_IGNORE);
    t_wait += wait;
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
