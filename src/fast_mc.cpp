// Low-moment evaluation of the standard PN mass-correction method.
//
// Same semidiscrete system and SSP scheme as Transport::evaluate_r2/advance, with
// floating-point reassociation only. It relies on three identities:
//  * The limiter changes only the four M1 moments, so unlimited pair fluxes and
//    their gather are recovered exactly by the target; pair data is 4-component.
//  * The LLF boundary flux is affine in the trace and the datum, so each face
//    node needs one cached operator action instead of one per quadrature point.
//  * C+C^T=B and C*1=0 give the weak load and the strong predictor from one C*a.
//
// Every options.ledger_stride-th step still runs the all-Q path in time.cpp, so
// these identities stay checked at runtime against that independent path.
#include "base_bar.hpp"
#include "cuda_path.hpp"
#include "time.hpp"
#include <algorithm>
#include <map>
#include <mutex>
#include <optional>
#include <atomic>
#include <deque>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace m1pn {
namespace {
struct Csr {
    std::vector<int> ptr, col;
    std::vector<double> val;
    int rows() const { return int(ptr.size()) - 1; }
};
Csr csr_nonzero(const Sparse &a, double drop = 0.) {
    Csr out;
    out.ptr.reserve(a.rows() + 1);
    for (int i = 0; i < a.rows(); ++i) {
        out.ptr.push_back(int(out.col.size()));
        for (Sparse::InnerIterator it(a, i); it; ++it)
            if (std::abs(it.value()) > drop) {
                out.col.push_back(int(it.col()));
                out.val.push_back(it.value());
            }
    }
    out.ptr.push_back(int(out.col.size()));
    return out;
}
inline Four unproject(const Four &v) {
    const double r0 = std::sqrt(4 * pi), r1 = std::sqrt(4 * pi / 3);
    Four u;
    u[0] = v[0] / r0;
    u[3] = v[1] / r1;
    u[1] = v[2] / r1;
    u[2] = v[3] / r1;
    return u;
}
inline int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}
inline int thread_limit() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
// Reused workspace. Left unfilled: every row is first written in the parallel
// compute loops, which also places the pages on the NUMA node that uses them.
void ensure_field(Field &field, Eigen::Index rows, Eigen::Index cols) {
    if (field.rows() != rows || field.cols() != cols)
        field.resize(rows, cols);
}
void parallel_set_zero(Field &field) {
    double *data = field.data();
    const Eigen::Index stride = field.cols(), total = field.rows();
#pragma omp parallel for schedule(static)
    for (Eigen::Index i = 0; i < total; ++i)
        std::fill_n(data + i * stride, stride, 0.);
}
} // namespace

struct FastMassCorrection {
    Transport &tr;
    const Angular &pn;
    Spatial &fe;
    int n, q, r, np;
    int nact = 0;
    std::array<int, 3> act{};
    std::array<Csr, 3> ang, grad, bnd_form;
    Csr visc, mass_offdiag;
    std::vector<double> visc_sum, mass_defect;
    struct BoundaryEntry {
        int node, normal;
        double weight;
    };
    std::vector<BoundaryEntry> entries;
    std::vector<int> touched;
    std::vector<unsigned char> on_boundary;
    struct NormalOps {
        Csr llf;
        Mat datum;
        Eigen::Matrix<double, 4, Eigen::Dynamic> llf_low, datum_low;
        V3 n;
        Vec constant_d;
        Four constant_moments = Four::Zero();
        bool has_constant = false;
        // Odd harmonics of each atom direction. Stored once, read lock-free.
        mutable std::deque<std::pair<V3, Vec>> atom_y;
        mutable std::atomic<int> atom_count{0};
        mutable std::mutex atom_guard;
    };
    std::deque<NormalOps> ops;
    struct FaceNode {
        int j, node, entry;
        double area;
    };
    std::vector<int> face_ptr, face_normal;
    std::vector<V3> face_direction;
    std::vector<FaceNode> face_nodes;

    // Stage workspace.
    std::vector<std::optional<Levermore>> closure;
    std::vector<Four> low, slot_ghost, slot_low, slot_high, slot_flux, slot_accepted;
    std::vector<Four> pair_high, pair_accepted, node_high, node_accepted;
    std::array<std::vector<Four>, 3> stage_gather;
    std::vector<double> slot_eta, slot_bar, pair_alpha, pair_margin;
    Field slot_datum, entry_datum; // row-major: read below as contiguous rows
    Field entry_flux, predictor, weak_load, euler_workspace;
    std::array<Field, 3> angular;
    std::array<Stage, 3> stage_pool;
    std::vector<std::vector<double>> scratch;

    explicit FastMassCorrection(Transport &t);
    const Vec &atom_harmonics(const NormalOps &op, const Boundary &b, const V3 &direction) const;
    void boundary_datum(const NormalOps &op, const Boundary &b, const V3 &x, double time, double *d,
                        Four &moments) const;
    void boundary_stage(double time, const Field &u, Stage &out);
    void stage(double time, const Field &u, Stage &out, std::vector<Four> &gather);
    double checked_margin(const Field &state) const;
};
void FastMassCorrectionDeleter::operator()(FastMassCorrection *p) const { delete p; }

FastMassCorrection *make_fast_mass_correction(Transport &tr) {
    const auto &o = tr.options;
    if (!o.r2 || o.stabilization != "m1pn_mass_correction" || o.pn_boundary_quadrature != "lumped" ||
        o.lazy || cuda_requested() || tr.inflow.exterior_state || tr.runtime.size != 1 || tr.pn.q < 4)
        return nullptr;
    for (const auto &face : tr.fe.faces)
        for (const auto &p : face.points)
            if (p.normal_id != face.points.front().normal_id)
                return nullptr;
    return new FastMassCorrection(tr);
}

FastMassCorrection::FastMassCorrection(Transport &t)
    : tr(t), pn(t.pn), fe(t.fe), n(t.fe.nodes), q(t.pn.q), r(t.canonical.r), np(int(t.fe.pairs.size())) {
    for (int d = 0; d < 3; ++d) {
        bool used = false;
        for (int k = 0; k < fe.C[d].nonZeros() && !used; ++k)
            used = fe.C[d].valuePtr()[k] != 0.;
        for (const auto &face : fe.faces)
            for (const auto &p : face.points)
                used = used || p.normal[d] != 0.;
        if (used)
            act[nact++] = d;
    }
    for (int k = 0; k < nact; ++k) {
        const int d = act[k];
        ang[k] = csr_nonzero(pn.A[d]);
        grad[k] = csr_nonzero(fe.C[d]);
        bnd_form[k] = csr_nonzero(fe.B[d]);
    }
    visc.ptr.assign(n + 1, 0);
    visc_sum.assign(n, 0.);
    for (int i = 0; i < n; ++i) {
        visc.ptr[i] = int(visc.col.size());
        for (std::size_t k = tr.pair_node_offsets[i]; k < tr.pair_node_offsets[i + 1]; ++k) {
            const int e = tr.pair_node_edges[k];
            const auto &p = fe.pairs[e];
            const double d = tr.pn_low_order_edge_viscosity[e];
            visc.col.push_back(p.i == i ? p.j : p.i);
            visc.val.push_back(d);
            visc_sum[i] += d;
        }
    }
    visc.ptr[n] = int(visc.col.size());
    mass_defect.resize(n);
    mass_offdiag.ptr.reserve(n + 1);
    for (int i = 0; i < n; ++i) {
        mass_offdiag.ptr.push_back(int(mass_offdiag.col.size()));
        double diagonal = 0;
        for (Sparse::InnerIterator it(fe.M, i); it; ++it) {
            if (it.col() == i) diagonal = it.value();
            else if (it.value() != 0.) {
                mass_offdiag.col.push_back(int(it.col()));
                mass_offdiag.val.push_back(it.value());
            }
        }
        mass_defect[i] = fe.mass[i] - diagonal;
    }
    mass_offdiag.ptr.push_back(int(mass_offdiag.col.size()));
    // One operator set per distinct stored normal.
    for (const auto &boundary : tr.boundaries) {
        ops.emplace_back();
        NormalOps &op = ops.back();
        op.n = boundary->n;
        Sparse llf = boundary->llf_homogeneous.sparseView(1., 5e-16 / std::max(1., boundary->llf_homogeneous.cwiseAbs().maxCoeff()));
        llf.makeCompressed();
        op.llf = csr_nonzero(llf);
        op.datum = boundary->llf_datum;
        op.llf_low = boundary->llf_homogeneous.topRows<4>();
        op.datum_low = boundary->llf_datum.topRows<4>();
        if (tr.inflow.constant_density) {
            const double g = *tr.inflow.constant_density;
            require(std::isfinite(g) && g >= 0, "Physical inflow density must be nonnegative");
            op.constant_d = Vec::Zero(r);
            for (std::size_t k = 0; k < boundary->rule.size(); ++k) {
                const auto &p = boundary->rule[k];
                op.constant_d.noalias() += 2 * p.weight * g * boundary->odd_basis.row(k).transpose();
                op.constant_moments[0] += p.weight * g;
                op.constant_moments.tail<3>() += p.weight * g * p.omega;
            }
            op.has_constant = true;
        }
    }
    // Aggregate equal (node, normal) boundary loads.
    on_boundary.assign(n, 0);
    std::map<std::pair<int, int>, int> entry_of;
    face_ptr.reserve(fe.faces.size() + 1);
    for (std::size_t f = 0; f < fe.faces.size(); ++f) {
        const auto &face = fe.faces[f];
        face_ptr.push_back(int(face_nodes.size()));
        const int normal = face.points.front().normal_id;
        face_normal.push_back(normal);
        face_direction.push_back(face.points.front().normal);
        for (int j = 0; j < int(face.nodes.size()); ++j) {
            bool active = false;
            for (const auto &p : face.points)
                active = active || p.weight * p.shape[j] != 0.;
            if (!active) continue;
            const int node = face.nodes[j];
            auto [it, created] = entry_of.emplace(std::make_pair(node, normal), int(entries.size()));
            if (created) entries.push_back({node, normal, 0.});
            entries[it->second].weight += face.mass[j];
            face_nodes.push_back({j, node, it->second, face.mass[j]});
            if (!on_boundary[node]) {
                on_boundary[node] = 1;
                touched.push_back(node);
            }
        }
    }
    face_ptr.push_back(int(face_nodes.size()));
    const std::size_t slots = face_nodes.size();
    tr.options.budget.check(sizeof(double) * (.5L * entries.size() * q * q + 10.L * n * q + 64.L * np + 16.L * slots * r),
                            "Fast mass-correction workspace");
    closure.resize(n);
    low.resize(n);
    slot_ghost.resize(slots); slot_low.resize(slots); slot_high.resize(slots); slot_flux.resize(slots);
    slot_accepted.resize(slots); slot_eta.resize(slots); slot_bar.resize(slots);
    pair_high.resize(np); pair_accepted.resize(np);
    pair_alpha.resize(np); pair_margin.resize(np);
    node_high.resize(n); node_accepted.resize(n);
    for (auto &g : stage_gather) g.resize(n);
    slot_datum.resize(slots, r);
    entry_datum.resize(entries.size(), r);
    ensure_field(entry_flux, Eigen::Index(entries.size()), q);
    ensure_field(predictor, n, q);
    ensure_field(weak_load, n, q);
    for (int k = 0; k < nact; ++k) ensure_field(angular[k], n, q);
    for (auto &stage : stage_pool) {
        ensure_field(stage.target, n, q);
        ensure_field(stage.boundary_gather, n, q);
    }
    scratch.assign(thread_limit(), std::vector<double>(q));
}

const Vec &FastMassCorrection::atom_harmonics(const NormalOps &op, const Boundary &b, const V3 &direction) const {
    const int count = op.atom_count.load(std::memory_order_acquire);
    for (int k = 0; k < count; ++k)
        if ((op.atom_y[k].first.array() == direction.array()).all())
            return op.atom_y[k].second;
    std::lock_guard<std::mutex> lock(op.atom_guard);
    for (int k = 0; k < op.atom_count.load(std::memory_order_relaxed); ++k)
        if ((op.atom_y[k].first.array() == direction.array()).all())
            return op.atom_y[k].second;
    const Vec y = harmonics(pn.order, b.frame.transpose() * direction);
    Vec odd(r);
    for (int j = 0; j < r; ++j) odd[j] = y[tr.canonical.odd[j]];
    op.atom_y.emplace_back(direction, std::move(odd));
    op.atom_count.store(int(op.atom_y.size()), std::memory_order_release);
    return op.atom_y.back().second;
}

void FastMassCorrection::boundary_datum(const NormalOps &op, const Boundary &b, const V3 &x, double time,
                                        double *d, Four &moments) const {
    const Inflow &in = tr.inflow;
    std::fill_n(d, r, 0.);
    moments.setZero();
    if (op.has_constant) {
        for (int j = 0; j < r; ++j) d[j] = op.constant_d[j];
        moments = op.constant_moments;
    } else if (in.density) {
        for (std::size_t k = 0; k < b.rule.size(); ++k) {
            const auto &p = b.rule[k];
            const double g = in.density(x, time, p.omega);
            require(std::isfinite(g) && g >= 0, "Physical inflow density must be nonnegative");
            const double w = 2 * p.weight * g;
            const double *row = &b.odd_basis.coeff(Eigen::Index(k), 0);
            for (int j = 0; j < r; ++j) d[j] += w * row[j];
            moments[0] += p.weight * g;
            moments.tail<3>() += p.weight * g * p.omega;
        }
    }
    if (in.atoms)
        for (const auto &a : in.atoms(x, time)) {
            require(a.mass >= 0 && std::isfinite(a.mass) && std::abs(a.direction.norm() - 1) < 2e-12,
                    "Invalid incoming atom");
            if (b.n.dot(a.direction) >= 0) continue;
            moments[0] += a.mass;
            moments.tail<3>() += a.mass * a.direction;
            const Vec &odd = atom_harmonics(op, b, a.direction);
            const double w = 2 * a.mass;
            for (int j = 0; j < r; ++j) d[j] += w * odd[j];
        }
    require(moments.allFinite() && moments[0] >= 0, "Invalid incoming moment quadrature");
    for (int k = 0; k < 64 && !exact::cone(moments); ++k)
        moments[0] = std::nextafter(moments[0], std::numeric_limits<double>::infinity());
    require(exact::cone(moments), "Unable to represent incoming moments in M1 cone");
}

double FastMassCorrection::checked_margin(const Field &state) const {
    double minimum = std::numeric_limits<double>::infinity();
    int failing = -1;
#pragma omp parallel for schedule(static) reduction(min : minimum)
    for (int i = 0; i < n; ++i) {
        const Four p = project(state.row(i).transpose());
        minimum = std::min(minimum, margin(p));
        if (!exact::cone(p)) {
#pragma omp critical(m1pn_fast_failing)
            if (failing < 0 || i < failing) failing = i;
        }
    }
    if (failing >= 0)
        cone_require(false, "Stored PN projection outside M1 cone at node " + std::to_string(failing));
    return minimum;
}

// Face work for the M1 limiter plus the cached full-Q boundary flux per entry.
void FastMassCorrection::boundary_stage(double time, const Field &u, Stage &out) {
    const int nfaces = int(fe.faces.size());
    parallel_for(nfaces, [&](int f) {
        const auto &face = fe.faces[f];
        const int begin = face_ptr[f], end = face_ptr[f + 1];
        if (begin == end) return;
        const NormalOps &op = ops[face_normal[f]];
        const Boundary &b = *tr.boundaries[face_normal[f]];
        const V3 &normal = face_direction[f];
        for (int s = begin; s < end; ++s) {
            slot_ghost[s].setZero();
            slot_low[s].setZero();
            slot_datum.row(s).setZero();
            slot_flux[s] = closure[face_nodes[s].node]->flux(normal);
        }
        std::vector<double> d(r);
        for (const auto &p : face.points) {
            Four moments;
            boundary_datum(op, b, p.x, time, d.data(), moments);
            const Four exterior = tr.inflow.exterior_m1 ? tr.inflow.exterior_m1(p.x, time) : moments;
            require(exact::cone(exterior), "M1 exterior state outside cone");
            const Levermore external(exterior);
            const Four fx = external.flux(p.normal);
            for (int s = begin; s < end; ++s) {
                const FaceNode &node = face_nodes[s];
                const double weight = p.weight * p.shape[node.j];
                if (weight == 0) continue;
                const Four &inner = low[node.node];
                const Four &fi = slot_flux[s];
                const Four lo_flux = .5 * (fi + fx) - .5 * (exterior - inner);
                Four ghost = .5 * (inner + exterior) - .5 * (fx - fi);
                if (!exact::cone(ghost)) ghost = stable_base_bar(inner, exterior, .5 * p.normal, .5);
                cone_require(exact::cone(ghost), "R2 LLF ghost base outside cone");
                slot_low[s] += weight * lo_flux;
                slot_ghost[s] += weight * ghost;
                for (int j = 0; j < r; ++j) slot_datum(s, j) += weight * d[j];
            }
        }
        for (int s = begin; s < end; ++s) {
            const FaceNode &node = face_nodes[s];
            const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> state(&u.coeff(node.node, 0), q);
            const Four h = (node.area * (op.llf_low * state) + op.datum_low * slot_datum.row(s).transpose()).eval();
            slot_high[s] = h;
            if (!(node.area > 0)) {
                slot_accepted[s] = h;
                slot_eta[s] = 1;
                slot_bar[s] = std::numeric_limits<double>::infinity();
                continue;
            }
            const Four base = slot_ghost[s] / node.area;
            cone_require(exact::cone(base), "R2 integrated ghost base outside cone");
            const Four raw = slot_low[s] - project(h);
            const Four delta = raw / node.area;
            double theta = segment(base, delta);
            auto valid = [&](double t) { return exact::cone(base + (t * raw) / node.area); };
            if (!valid(theta)) {
                double shrink = 4 * std::numeric_limits<double>::epsilon();
                while (shrink < 1e-6 && !valid(theta * (1 - shrink))) shrink *= 2;
                if (shrink < 1e-6) theta *= 1 - shrink;
                else {
                    double lo = 0, hi = theta;
                    for (int k = 0; k < 60; ++k) { const double mid = (lo + hi) / 2; if (valid(mid)) lo = mid; else hi = mid; }
                    theta = lo;
                }
            }
            cone_require(valid(theta), "R2 corrected ghost outside cone");
            slot_accepted[s] = unproject(slot_low[s] - theta * raw);
            slot_eta[s] = theta;
            slot_bar[s] = margin(base + theta * delta);
        }
    });
    // Datum sums are shared by the faces meeting at one (node, normal).
    entry_datum.setZero();
    for (std::size_t s = 0; s < face_nodes.size(); ++s)
        entry_datum.row(face_nodes[s].entry) += slot_datum.row(Eigen::Index(s));
    const int count = int(entries.size());
#pragma omp parallel for schedule(static)
    for (int k = 0; k < count; ++k) {
        const BoundaryEntry &entry = entries[k];
        const NormalOps &op = ops[entry.normal];
        const double *state = &u.coeff(entry.node, 0);
        double *flux = &entry_flux.coeffRef(k, 0);
        std::fill_n(flux, q, 0.);
        for (int c = 0; c < q; ++c) {
            double value = 0;
            for (int m = op.llf.ptr[c]; m < op.llf.ptr[c + 1]; ++m)
                value += op.llf.val[m] * state[op.llf.col[m]];
            flux[c] = entry.weight * value;
        }
        const double *dk = &entry_datum.coeff(k, 0);
        for (int c = 0; c < q; ++c) {
            double value = 0;
            const double *row = &op.datum.coeff(c, 0);
            const Eigen::Index stride = op.datum.outerStride();
            for (int j = 0; j < r; ++j) value += row[j * stride] * dk[j];
            flux[c] += value;
        }
    }
    Field &gather = out.boundary_gather;
    for (const int node : touched) gather.row(node).setZero();
    for (int k = 0; k < count; ++k) gather.row(entries[k].node) += entry_flux.row(k);
}

void FastMassCorrection::stage(double time, const Field &u, Stage &out, std::vector<Four> &gather) {
    Diagnostics &diag = out.diagnostics;
    diag = Diagnostics{};
    parallel_for(n, [&](int i) {
        low[i] = project(u.row(i).transpose());
        closure[i].emplace(low[i]);
    });
    // Angular fluxes in the active directions only.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const double *in = &u.coeff(i, 0);
        for (int k = 0; k < nact; ++k) {
            const Csr &a = ang[k];
            double *flux = &angular[k].coeffRef(i, 0);
            for (int c = 0; c < q; ++c) {
                double value = 0;
                for (int m = a.ptr[c]; m < a.ptr[c + 1]; ++m)
                    value += a.val[m] * in[a.col[m]];
                flux[c] = value;
            }
        }
    }
    double started = seconds();
    boundary_stage(time, u, out);
    tr.timings.boundary += seconds() - started;
    started = seconds();
    // Common node load, then both assembled forms.
#pragma omp parallel
    {
        std::vector<double> &buffer = scratch[thread_id()];
        double *common = buffer.data();
#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            std::fill_n(common, q, 0.);
            for (int k = 0; k < nact; ++k) {
                const Csr &g = grad[k];
                for (int m = g.ptr[i]; m < g.ptr[i + 1]; ++m) {
                    const double value = g.val[m];
                    const double *flux = &angular[k].coeff(g.col[m], 0);
                    for (int c = 0; c < q; ++c)
                        common[c] -= value * flux[c];
                }
            }
            if (on_boundary[i]) {
                const double *b = &out.boundary_gather.coeff(i, 0);
                for (int c = 0; c < q; ++c)
                    common[c] -= b[c];
            }
            double *weak = &weak_load.coeffRef(i, 0);
            double *pred = &predictor.coeffRef(i, 0);
            std::copy_n(common, q, weak);
            std::copy_n(common, q, pred);
            const double *ui = &u.coeff(i, 0);
            const double sum = visc_sum[i];
            for (int c = 0; c < q; ++c)
                pred[c] -= sum * ui[c];
            for (int m = visc.ptr[i]; m < visc.ptr[i + 1]; ++m) {
                const double dn = visc.val[m];
                const double *uj = &u.coeff(visc.col[m], 0);
                for (int c = 0; c < q; ++c)
                    pred[c] += dn * uj[c];
            }
        }
    }
    // Boundary rows: B*a for the weak form, lumped physical trace for the predictor.
    for (const int i : touched) {
        double *weak = &weak_load.coeffRef(i, 0);
        for (int k = 0; k < nact; ++k) {
            const Csr &b = bnd_form[k];
            for (int m = b.ptr[i]; m < b.ptr[i + 1]; ++m) {
                const double value = b.val[m];
                const double *flux = &angular[k].coeff(b.col[m], 0);
                for (int c = 0; c < q; ++c)
                    weak[c] += value * flux[c];
            }
        }
    }
    for (const BoundaryEntry &entry : entries) {
        double *pred = &predictor.coeffRef(entry.node, 0);
        for (int k = 0; k < nact; ++k) {
            const double scale = entry.weight * ops[entry.normal].n[act[k]];
            if (scale == 0.) continue;
            const double *flux = &angular[k].coeff(entry.node, 0);
            for (int c = 0; c < q; ++c)
                pred[c] += scale * flux[c];
        }
    }
    tr.timings.pn_predictor += seconds() - started;
    started = seconds();
    // target_i = (load_i + (m_i-M_ii) r_i - sum_j M_ij r_j) / m_i with r the predictor.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const double inv = 1. / fe.mass[i];
        double *pred = &predictor.coeffRef(i, 0);
        for (int c = 0; c < q; ++c)
            pred[c] *= inv;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double *target = &out.target.coeffRef(i, 0);
        const double *weak = &weak_load.coeff(i, 0);
        const double *pred = &predictor.coeff(i, 0);
        const double defect = mass_defect[i];
        for (int c = 0; c < q; ++c)
            target[c] = weak[c] + defect * pred[c];
        for (int m = mass_offdiag.ptr[i]; m < mass_offdiag.ptr[i + 1]; ++m) {
            const double value = mass_offdiag.val[m];
            const double *other = &predictor.coeff(mass_offdiag.col[m], 0);
            for (int c = 0; c < q; ++c)
                target[c] -= value * other[c];
        }
        const double inv = 1. / fe.mass[i];
        for (int c = 0; c < q; ++c)
            target[c] *= inv;
    }
    tr.timings.pn_mass_correction += seconds() - started;
    started = seconds();
    // Four-moment pair fluxes and the cone-only scalar limiter.
    int local_failure = 0;
    std::string failure;
    try {
        parallel_for(np, [&](int e) {
            const auto &p = fe.pairs[e];
            const V3 &cij = tr.pair_geometry[e][0], &cji = tr.pair_geometry[e][1];
            Four h = Four::Zero();
            for (int k = 0; k < nact; ++k) {
                const int d = act[k];
                const double cj = cji[d], ci = cij[d];
                const double *rj = &angular[k].coeff(p.j, 0), *ri = &angular[k].coeff(p.i, 0);
                for (int c = 0; c < 4; ++c)
                    h[c] += cj * rj[c] - ci * ri[c];
            }
            const double *pi = &predictor.coeff(p.i, 0), *pj = &predictor.coeff(p.j, 0);
            for (int c = 0; c < 4; ++c)
                h[c] += p.mass * (pi[c] - pj[c]);
            pair_high[e] = h;
            const Four &li = low[p.i], &lj = low[p.j];
            const Levermore &ci = *closure[p.i], &cj = *closure[p.j];
            const Four fj_cij = cj.flux(cij), fi_cij = ci.flux(cij);
            const Four fi_cji = ci.flux(cji), fj_cji = cj.flux(cji);
            const double twice = 2 * p.viscosity;
            const Four base = p.viscosity * (lj - li) - fj_cij + fi_cji;
            const Four mean = .5 * (li + lj);
            Four left = mean - (fj_cij - fi_cij) / twice;
            Four right = mean - (fi_cji - fj_cji) / twice;
            if (!exact::cone(left)) left = stable_base_bar(li, lj, cij, p.viscosity);
            if (!exact::cone(right)) right = stable_base_bar(lj, li, cji, p.viscosity);
            cone_require(exact::cone(left) && exact::cone(right), "R2 directed base bars outside cone");
            const Four raw = project(h) - base;
            const Four delta = raw / twice;
            double alpha = segment(left, delta);
            if (alpha > 0) alpha = std::min(alpha, segment(right, Four(-delta)));
            auto valid = [&](double a) {
                const Four step = (a * raw) / twice;
                return exact::cone(left + step) && exact::cone(right - step);
            };
            if (!valid(alpha)) {
                double shrink = 4 * std::numeric_limits<double>::epsilon();
                while (shrink < 1e-6 && !valid(alpha * (1 - shrink))) shrink *= 2;
                if (shrink < 1e-6) alpha *= 1 - shrink;
                else {
                    double lo = 0, hi = alpha;
                    for (int k = 0; k < 60; ++k) { const double mid = (lo + hi) / 2; if (valid(mid)) lo = mid; else hi = mid; }
                    alpha = lo;
                }
            }
            cone_require(valid(alpha), "R2 corrected directed bars outside cone");
            pair_accepted[e] = unproject(base + alpha * raw);
            pair_alpha[e] = alpha;
            const Four step = (alpha * raw) / twice;
            pair_margin[e] = std::min(margin(left + step), margin(right - step));
        });
    } catch (const StepRejected &error) { local_failure = 1; failure = error.what(); }
      catch (const std::exception &error) { local_failure = 2; failure = error.what(); }
    tr.runtime.finish_local_phase(local_failure, failure);
    tr.timings.mcl += seconds() - started;
    // Node gather, then reduce the face slots into the node sums.
    parallel_for(n, [&](int i) {
        Four sum_high = Four::Zero(), sum_accepted = Four::Zero();
        for (std::size_t k = tr.pair_node_offsets[i]; k < tr.pair_node_offsets[i + 1]; ++k) {
            const int e = tr.pair_node_edges[k];
            if (tr.pair_node_sign[k] > 0) { sum_high += pair_high[e]; sum_accepted += pair_accepted[e]; }
            else { sum_high -= pair_high[e]; sum_accepted -= pair_accepted[e]; }
        }
        node_high[i] = sum_high;
        node_accepted[i] = sum_accepted;
    });
    for (const int i : touched) {
        double *row = &out.boundary_gather.coeffRef(i, 0);
        for (int c = 0; c < 4; ++c) row[c] = 0.;
    }
    for (std::size_t s = 0; s < face_nodes.size(); ++s) {
        const FaceNode &node = face_nodes[s];
        node_high[node.node] -= slot_high[s];
        node_accepted[node.node] -= slot_accepted[s];
        double *row = &out.boundary_gather.coeffRef(node.node, 0);
        for (int c = 0; c < 4; ++c) row[c] += slot_accepted[s][c];
        diag.minimum_eta = std::min(diag.minimum_eta, slot_eta[s]);
        diag.minimum_bar = std::min(diag.minimum_bar, slot_bar[s]);
        diag.active_faces += slot_eta[s] < 1;
    }
    // The full four-moment recovery and the low-moment rate.
    double recovery = 0, load_scale = 1, balance = 0, sync = 0;
    double minimum_alpha = 1, minimum_bar = diag.minimum_bar;
    long active = 0;
#pragma omp parallel for schedule(static) reduction(max : recovery, load_scale, balance) reduction(+ : sync)
    for (int i = 0; i < n; ++i) {
        double *row = &out.target.coeffRef(i, 0);
        const double mi = fe.mass[i];
        const Four target_low(row[0], row[1], row[2], row[3]);
        for (int c = 0; c < 4; ++c) {
            recovery = std::max(recovery, std::abs(node_high[i][c] - mi * target_low[c]));
            load_scale = std::max(load_scale, std::abs(weak_load(i, c)));
        }
        const Four rate = unproject(project(node_accepted[i]) / mi);
        // Sync work needs M*u only in the four coupled moments.
        Four mu = mass_defect[i] * Four(u(i, 0), u(i, 1), u(i, 2), u(i, 3));
        mu = mi * Four(u(i, 0), u(i, 1), u(i, 2), u(i, 3)) - mu;
        for (int m = mass_offdiag.ptr[i]; m < mass_offdiag.ptr[i + 1]; ++m) {
            const int j = mass_offdiag.col[m];
            mu += mass_offdiag.val[m] * Four(u(j, 0), u(j, 1), u(j, 2), u(j, 3));
        }
        for (int c = 0; c < 4; ++c) {
            sync += mu[c] * (rate[c] - target_low[c]);
            row[c] = rate[c];
            balance = std::max(balance, std::abs(mi * rate[c] - node_accepted[i][c]));
        }
        gather[i] = node_accepted[i];
    }
    diag.target_recovery = recovery;
    diag.local_balance = balance;
    require(recovery < 5e-11 * std::max(1., load_scale), "R2 four-moment residual identity failed");
#pragma omp parallel for schedule(static) reduction(min : minimum_alpha, minimum_bar) reduction(+ : active)
    for (int e = 0; e < np; ++e) {
        minimum_alpha = std::min(minimum_alpha, pair_alpha[e]);
        minimum_bar = std::min(minimum_bar, pair_margin[e]);
        active += pair_alpha[e] < 1;
    }
    diag.minimum_alpha = minimum_alpha;
    diag.minimum_bar = minimum_bar;
    diag.active_pairs = active;
    diag.closure_evaluations = n;
    diag.sync_work = sync;
    diag.target_energy = std::numeric_limits<double>::quiet_NaN();
}

Step advance_fast_mc(Transport &tr, const TimeControl &control, double time, double dt, const Field &initial) {
    FastMassCorrection &plan = *tr.fast_mc;
    const int n = plan.n, q = plan.q;
    control.check(dt);
    double initial_margin = plan.checked_margin(initial);
    Step out;
    Field &u1 = tr.stage_u1, &u2 = tr.stage_u2;
    ensure_field(u1, n, q);
    ensure_field(u2, n, q);
    int stage_number = 0;
    auto stage = [&](double t, const Field &u, Field &euler, int r, double state_margin) {
        require(dt <= tr.fe.cfl_step * (1 + 2e-14), "Actual SSP Euler branch violates open M1 CFL");
        try {
            control.check(dt);
            Stage &s = plan.stage_pool[r];
            plan.stage(t, u, s, plan.stage_gather[r]);
            const Field &rate = s.target;
            ensure_field(euler, n, q);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                const double *a = &u.coeff(i, 0), *b = &rate.coeff(i, 0);
                double *e = &euler.coeffRef(i, 0);
                for (int c = 0; c < q; ++c) e[c] = a[c] + dt * b[c];
            }
            const double euler_margin = plan.checked_margin(euler);
            s.diagnostics.cfl_ratio = dt / tr.fe.cfl_step;
            s.diagnostics.minimum_state = std::min(state_margin, euler_margin);
            ++stage_number;
            return euler_margin;
        } catch (const StepRejected &error) {
            throw StepRejected("SSP stage " + std::to_string(stage_number) + ": " + error.what());
        }
    };
    // Convex combination of nodal rows, overwriting `target`.
    auto combine = [&](Field &target, double a, const Field &left, double b, const Field &right) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            const double *x = &left.coeff(i, 0), *y = &right.coeff(i, 0);
            double *z = &target.coeffRef(i, 0);
            for (int c = 0; c < q; ++c) z[c] = a * x[c] + b * y[c];
        }
    };
    ensure_field(out.state, n, q);
    double margin_u1 = stage(time, initial, u1, 0, initial_margin);
    std::vector<double> weights;
    double final_margin;
    if (control.order == 2) {
        Field &euler = plan.euler_workspace;
        stage(time + dt, u1, euler, 1, margin_u1);
        combine(out.state, .5, initial, .5, euler);
        weights = {.5, .5};
    } else {
        Field &euler = plan.euler_workspace;
        stage(time + dt, u1, euler, 1, margin_u1);
        combine(u2, .75, initial, .25, euler);
        const double margin_u2 = plan.checked_margin(u2);
        stage(time + .5 * dt, u2, euler, 2, margin_u2);
        combine(out.state, 1. / 3, initial, 2. / 3, euler);
        weights = {1. / 6, 1. / 6, 2. / 3};
    }
    final_margin = plan.checked_margin(out.state);
    ensure_field(out.integrated_boundary, n, q);
    parallel_set_zero(out.integrated_boundary);
    std::vector<Four> ledger(n, Four::Zero());
    double rate_residual = 0;
#pragma omp parallel for schedule(static) reduction(max : rate_residual)
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < q; ++c) {
            double accumulated = 0;
            for (std::size_t r = 0; r < weights.size(); ++r)
                accumulated += dt * weights[r] * plan.stage_pool[r].target(i, c);
            rate_residual = std::max(rate_residual, std::abs(out.state(i, c) - initial(i, c) - accumulated));
        }
    }
    for (std::size_t r = 0; r < weights.size(); ++r) {
        const double w = dt * weights[r];
        for (const int node : plan.touched)
            out.integrated_boundary.row(node) += w * plan.stage_pool[r].boundary_gather.row(node);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            ledger[i] += w * plan.stage_gather[r][i];
        out.stages.push_back(plan.stage_pool[r].diagnostics);
    }
    double ledger_residual = 0, transfer_scale = 1;
#pragma omp parallel for schedule(static) reduction(max : ledger_residual, transfer_scale)
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < 4; ++c) {
            ledger_residual = std::max(ledger_residual,
                                       std::abs(tr.fe.mass[i] * (out.state(i, c) - initial(i, c)) - ledger[i][c]));
            transfer_scale = std::max(transfer_scale, std::abs(ledger[i][c]));
        }
    out.rate_residual = rate_residual;
    out.ledger_residual = ledger_residual;
    out.minimum_margin = final_margin;
    require(ledger_residual < 2e-10 * transfer_scale, "Integrated RK four-moment ledger failed");
    return out;
}
} // namespace m1pn
