#include "element_weno_mpi.hpp"
#ifdef M1PN_MPI
#include "parallel.hpp"
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <map>
#include <omp.h>

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
    const double h = g.length / g.elements;
    o.elements_per_axis[D] = part.local_end(r) - part.local_first(r);
    o.origin[D] = g.origin[D] + part.local_first(r) * h;
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
    // The owners along the factor positions must form one contiguous run per rank (sequential chain).
    {
        std::vector<int> seen(P, 0);
        int last = -1;
        j0_ = j1_ = -1;
        for (int j = 0; j < n1; ++j) {
            const int o = owner_[inv_[j]];
            if (o != last) {
                require(!seen[o], "MPI: the line factor order visits a rank twice (use more elements per rank)");
                seen[o] = 1, last = o;
            }
            if (o == r) {
                if (j0_ < 0)
                    j0_ = j;
                j1_ = j + 1;
            }
        }
    }
    // Communication links of the substitution: forward, column j (owner of j) updates rows i; backward, column j
    // reads the final rows i.
    std::map<int, std::vector<int>> fs, fr, bs, br;
    for (int j = 0; j < n1; ++j)
        for (int k = outer_[j]; k < outer_[j + 1]; ++k) {
            const int i = inner_[k];
            if (i == j)
                continue;
            const int oj = owner_[inv_[j]], oi = owner_[inv_[i]];
            if (oj == oi)
                continue;
            if (oj == r)
                fs[oi].push_back(j), br[oi].push_back(i);
            if (oi == r)
                fr[oj].push_back(j), bs[oj].push_back(i);
        }
    auto links = [](std::map<int, std::vector<int>> &m) {
        std::vector<Link> out;
        for (auto &[rank, v] : m) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            out.push_back({rank, v});
        }
        return out;
    };
    fwd_send_ = links(fs), fwd_recv_ = links(fr), bwd_send_ = links(bs), bwd_recv_ = links(br);
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
    const double start = MPI_Wtime();
    const int me = part_.rank;
    const std::size_t len = std::size_t(layer_size_) * b.cols();
    const int C = chunks_ > 0 ? chunks_ : int(std::max<std::size_t>(1, std::min<std::size_t>(4 * part_.ranks, len / 2048)));
    std::vector<std::size_t> cs(C + 1);
    for (int c = 0; c <= C; ++c)
        cs[c] = len * c / C;
    // Link buffers in chunk-major layout: chunk c of position m at npos * cs[c] + m * (cs[c+1] - cs[c]).
    auto alloc = [&](const std::vector<Link> &links) {
        std::vector<std::vector<double>> out(links.size());
        for (std::size_t l = 0; l < links.size(); ++l)
            out[l].resize(links[l].positions.size() * len);
        return out;
    };
    auto slot = [&](std::vector<double> &buf, std::size_t npos, int c, std::size_t m) {
        return buf.data() + npos * cs[c] + m * (cs[c + 1] - cs[c]);
    };
    const int threads = std::max(1, max_threads());
    // Element-parallel execution of `body(from, to)` on the chunk range (all operations are element-wise).
    auto split = [&](std::size_t n, const auto &body) {
        const int T = int(std::min<std::size_t>(threads, std::max<std::size_t>(1, n / 256)));
        parallel_for(T, [&](int t) { body(n * t / T, n * (t + 1) / T); });
    };
    // Foreign positions of a link set, merged in ascending position order: (position, link, index).
    auto merged = [](const std::vector<Link> &links) {
        std::vector<std::array<int, 3>> out;
        for (std::size_t l = 0; l < links.size(); ++l)
            for (std::size_t m = 0; m < links[l].positions.size(); ++m)
                out.push_back({links[l].positions[m], int(l), int(m)});
        std::sort(out.begin(), out.end());
        return out;
    };
    double wait = 0;
    // ---- forward: for positions j ascending, y_i -= L_ij y_j ----
    {
        auto rbuf = alloc(fwd_recv_), sbuf = alloc(fwd_send_);
        const auto foreign = merged(fwd_recv_);
        std::vector<MPI_Request> req;
        for (int c = 0; c < C; ++c) {
            const std::size_t cl = cs[c + 1] - cs[c];
            const double w0 = MPI_Wtime();
            for (std::size_t l = 0; l < fwd_recv_.size(); ++l) {
                const std::size_t np = fwd_recv_[l].positions.size();
                MPI_Recv(slot(rbuf[l], np, c, 0), int(np * cl), MPI_DOUBLE, fwd_recv_[l].rank, 100 + c, comm_,
                         MPI_STATUS_IGNORE);
            }
            wait += MPI_Wtime() - w0;
            split(cl, [&](std::size_t from, std::size_t to) {
                for (const auto &[j, l, m] : foreign) {
                    const double *yj = slot(rbuf[l], fwd_recv_[l].positions.size(), c, m);
                    for (int k = outer_[j]; k < outer_[j + 1]; ++k) {
                        const int i = inner_[k];
                        if (i == j || owner_[inv_[i]] != me)
                            continue;
                        const double v = values_[k];
                        double *yi = layer(b, inv_[i]) + cs[c];
                        for (std::size_t t = from; t < to; ++t) yi[t] -= v * yj[t];
                    }
                }
                for (int j = j0_; j < j1_; ++j) {
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
            for (std::size_t l = 0; l < fwd_send_.size(); ++l) {
                const auto &pos = fwd_send_[l].positions;
                for (std::size_t m = 0; m < pos.size(); ++m)
                    std::copy_n(layer(b, inv_[pos[m]]) + cs[c], cl, slot(sbuf[l], pos.size(), c, m));
                req.emplace_back();
                MPI_Isend(slot(sbuf[l], pos.size(), c, 0), int(pos.size() * cl), MPI_DOUBLE, fwd_send_[l].rank, 100 + c,
                          comm_, &req.back());
            }
        }
        MPI_Waitall(int(req.size()), req.data(), MPI_STATUSES_IGNORE);
    }
    // ---- diagonal ----
    for (int j = j0_; j < j1_; ++j) {
        const double inv = 1. / diag_[j];
        double *yj = layer(b, inv_[j]);
        split(len, [&](std::size_t from, std::size_t to) {
            for (std::size_t t = from; t < to; ++t) yj[t] *= inv;
        });
    }
    // ---- backward: for positions j descending, y_j -= L_ij y_i ----
    {
        auto rbuf = alloc(bwd_recv_), sbuf = alloc(bwd_send_);
        std::map<int, std::pair<int, int>> where; // foreign row position -> (link, index)
        for (std::size_t l = 0; l < bwd_recv_.size(); ++l)
            for (std::size_t m = 0; m < bwd_recv_[l].positions.size(); ++m)
                where[bwd_recv_[l].positions[m]] = {int(l), int(m)};
        std::vector<MPI_Request> req;
        for (int c = 0; c < C; ++c) {
            const std::size_t cl = cs[c + 1] - cs[c];
            const double w0 = MPI_Wtime();
            for (std::size_t l = 0; l < bwd_recv_.size(); ++l) {
                const std::size_t np = bwd_recv_[l].positions.size();
                MPI_Recv(slot(rbuf[l], np, c, 0), int(np * cl), MPI_DOUBLE, bwd_recv_[l].rank, 100000 + c, comm_,
                         MPI_STATUS_IGNORE);
            }
            wait += MPI_Wtime() - w0;
            split(cl, [&](std::size_t from, std::size_t to) {
                for (int j = j1_ - 1; j >= j0_; --j) {
                    double *yj = layer(b, inv_[j]) + cs[c];
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
            for (std::size_t l = 0; l < bwd_send_.size(); ++l) {
                const auto &pos = bwd_send_[l].positions;
                for (std::size_t m = 0; m < pos.size(); ++m)
                    std::copy_n(layer(b, inv_[pos[m]]) + cs[c], cl, slot(sbuf[l], pos.size(), c, m));
                req.emplace_back();
                MPI_Isend(slot(sbuf[l], pos.size(), c, 0), int(pos.size() * cl), MPI_DOUBLE, bwd_send_[l].rank,
                          100000 + c, comm_, &req.back());
            }
        }
        MPI_Waitall(int(req.size()), req.data(), MPI_STATUSES_IGNORE);
    }
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
