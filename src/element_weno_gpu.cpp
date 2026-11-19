#include "element_weno_gpu.hpp"
#include <iostream>
#include "base_bar.hpp"
#include "element_weno_gpu_device.hpp"
#include "parallel.hpp"
#include "uncollided.hpp"
#include "uncollided_lattice.hpp"
#include "uncollided_lattice3d.hpp"
#include <algorithm>
#include <random>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace m1pn {
namespace {
int omp_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
int omp_thread() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}
// Copies of the file-local helpers of src/element_weno.cpp (same code).
double norm_up(const Four &v) {
    const double scale = std::ldexp(1., 600);
    const bool tiny = std::max({std::abs(v[1]), std::abs(v[2]), std::abs(v[3])}) < 1e-290;
    const double s = tiny ? scale : 1.;
    const double n = std::sqrt((v[1] * s) * (v[1] * s) + (v[2] * s) * (v[2] * s) + (v[3] * s) * (v[3] * s));
    double up = n * (1 + 8 * std::numeric_limits<double>::epsilon());
    up = tiny ? up / scale : up;
    return std::nextafter(up, std::numeric_limits<double>::infinity());
}
bool nudge_into_cone(Four &v) {
    if (exact::cone(v))
        return false;
    for (int k = 0; k < 256 && !exact::cone(v); ++k)
        v[0] = std::nextafter(v[0], std::numeric_limits<double>::infinity());
    if (!exact::cone(v) && v[0] < 1e-290) {
        v[0] = std::max(v[0], norm_up(v));
        for (int k = 0; k < 256 && !exact::cone(v); ++k)
            v[0] = std::nextafter(v[0], std::numeric_limits<double>::infinity());
    }
    cone_require(exact::cone(v), "State left the M1 cone beyond rounding (bar state, 256 ulp)");
    return true;
}
void add_flux(const Sparse &at, const double *x, double scale, double *out, int count) {
    const int *outer = at.outerIndexPtr(), *inner = at.innerIndexPtr();
    const double *value = at.valuePtr();
    for (int j = 0; j < count; ++j) {
        double sum = 0;
        for (int k = outer[j]; k < outer[j + 1]; ++k)
            sum += value[k] * x[inner[k]];
        out[j] += scale * sum;
    }
}
std::vector<double> row_major(const Mat &m) {
    std::vector<double> out(std::size_t(m.rows()) * m.cols());
    for (int r = 0; r < m.rows(); ++r)
        for (int c = 0; c < m.cols(); ++c)
            out[std::size_t(r) * m.cols() + c] = m(r, c);
    return out;
}
} // namespace

ElementWenoGpu::ElementWenoGpu(const ElementWeno &s) : s_(s) {
    const ElementWenoOptions &o = s.options;
    require(!s.dist_, "GPU path: serial scheme only (no MPI)");
    // Interface sides: node-by-node coupling (nodal_interface, conforming blocks) keeps the regular pipeline (partner
    // term as face datum, node-wise bar states on the host); otherwise the whole face work runs on the host.
    for (int side = 0; side < 6; ++side) {
        interfaces_ |= s.options.interface_side[side] && !s.nodal_interface(side);
        nodal_interfaces_ |= s.nodal_interface(side);
    }
    require(!o.symmetric_residual && !o.m1_target, "GPU path: standard residual, PN target");
    require(!o.uncollided_density || ((o.uncollided_pulse || o.uncollided_lattice) && s.dimension == 2) ||
                (o.uncollided_lattice3d && s.dimension == 3),
            "GPU path: FCS needs the line-source pulse or the lattice table (2D), or the lattice-3d table");
    require(!o.projection_high_order && !o.projection_hybrid && !o.projection_diagnostics,
            "GPU path: projection_high_order/hybrid/diagnostics are not supported");
    require(s.degree <= 4 && s.local_nodes <= gpu::max_local, "GPU path: degree <= 4 and at most 27 local nodes");
    const int n = s.local_nodes, p = s.degree, q = s.moments;
    gpu::Setup g;
    g.dimension = s.dimension, g.degree = p, g.nodes = s.nodes, g.cells = s.cells, g.local = n, g.q = q;
    // Fallback lists hold every node / element, so a stage can never overflow them (2048^2 grids exceed 2^16).
    g.fallback_capacity = std::max({1 << 16, s.nodes, s.cells});
    fallback_capacity_ = g.fallback_capacity;
    for (int d = 0; d < 3; ++d)
        g.nodes_per_axis[d] = s.nodes_per_axis[d], g.count[d] = s.count[d], g.stride[d] = s.stride(d);
    g.viscosity = s.viscosity;
    g.connectivity = s.connectivity_.data();
    g.neighbors = s.neighbors_.data();
    g.node_offsets = s.node_offsets_.data();
    g.node_entries = s.node_entries_.data();
    // Element tables: one class (the members) on a uniform mesh, the stacked size classes on a graded one.
    const int classes = s.graded_ ? int(s.classes_.size()) : 1;
    std::vector<double> class_stiffness, class_viscosity, ext_list;
    std::array<std::vector<double>, 3> class_convection;
    g.classes = classes;
    g.structured = !s.broken_;
    std::array<std::vector<int>, 2> strip_elem;
    if (s.broken_ && s.dimension == 2 && s.degree == 1 && !s.options.periodic) {
        const int npx = s.nodes_per_axis[0];
        for (int d = 0; d < 2; ++d)
            strip_elem[d].assign(2 * std::size_t(s.nodes_per_axis[d]), -1);
        for (int e = 0; e < s.cells; ++e) {
            const int corner = s.connectivity_[std::size_t(e) * s.local_nodes]; // local node 0 = (0, 0)
            const int line[2] = {corner % npx, corner / npx}, id[2] = {e % s.count[0], e / s.count[0]};
            for (int d = 0; d < 2; ++d) {
                strip_elem[d][2 * line[d] + 1] = id[d]; // right of its first node line
                strip_elem[d][2 * (line[d] + 1)] = id[d]; // left of its second
            }
        }
        g.strip_elem[0] = strip_elem[0].data(), g.strip_elem[1] = strip_elem[1].data();
    }
    if (s.graded_) {
        for (const auto &c : s.classes_) {
            class_stiffness.insert(class_stiffness.end(), c.stiffness_rows.begin(), c.stiffness_rows.end());
            class_viscosity.push_back(c.viscosity);
            for (int d = 0; d < s.dimension; ++d)
                class_convection[d].insert(class_convection[d].end(), c.convection_t_rows[d].begin(),
                                           c.convection_t_rows[d].end());
        }
        for (const Mat &X : s.extension_list_) {
            const std::vector<double> r = row_major(X);
            ext_list.insert(ext_list.end(), r.begin(), r.end());
        }
        g.element_class = s.element_class_.data();
        g.class_viscosity = class_viscosity.data();
        g.ext_count = int(s.extension_list_.size());
        g.ext_list = ext_list.data();
        g.ext_index = s.extension_index_.data();
        g.stiffness_rows = class_stiffness.data();
    } else
        g.stiffness_rows = s.stiffness_rows_.data();
    const std::vector<double> semi = row_major(s.seminorm_);
    g.seminorm = semi.data();
    std::array<std::array<std::vector<double>, 2>, 3> ext;
    std::array<std::vector<double>, 3> band, lower, diagonal, dlower, ddiagonal, tail;
    g.periodic = o.periodic;
    for (int d = 0; d < s.dimension; ++d) {
        g.convection_t_rows[d] = s.graded_ ? class_convection[d].data() : s.convection_t_rows_[d].data();
        for (int side = 0; side < 2; ++side) {
            // graded: ext_index selects from ext_list; the uniform slot only needs n x n readable values
            ext[d][side] = s.graded_ ? std::vector<double>(ext_list.begin(), ext_list.begin() + n * n)
                                     : row_major(s.extension_[d][side]);
            g.extension[d][side] = ext[d][side].data();
        }
        // Banded 1D convection rows.
        const int n1 = s.nodes_per_axis[d], w = 2 * p + 1;
        band[d].assign(std::size_t(n1) * w, 0.);
        const Sparse &C = s.convection_1d_[d];
        for (int k = 0; k < n1; ++k)
            for (int e = C.outerIndexPtr()[k]; e < C.outerIndexPtr()[k + 1]; ++e) {
                const int col = C.innerIndexPtr()[e];
                int off = col - k; // periodic: cyclic offset
                if (o.periodic && off > n1 / 2)
                    off -= n1;
                if (o.periodic && off < -n1 / 2)
                    off += n1;
                require(std::abs(off) <= p, "GPU path: 1D convection is not banded");
                band[d][std::size_t(k) * w + (off + p)] += C.valuePtr()[e];
            }
        // Natural-order banded LDL^T of the 1D mass.
        const Sparse &M = s.mass_1d_[d];
        auto entry = [&](int r, int c) {
            for (int e = M.outerIndexPtr()[r]; e < M.outerIndexPtr()[r + 1]; ++e)
                if (M.innerIndexPtr()[e] == c)
                    return M.valuePtr()[e];
            return 0.;
        };
        // rows first .. first + count - 1 of the mass (count x count principal block, contiguous since a mirror
        // node is the first or the last on its line).
        auto factor = [&](int first, int count, std::vector<double> &lo, std::vector<double> &dg) {
            lo.assign(std::size_t(count) * p, 0.);
            dg.assign(count, 0.);
            auto L = [&](int r, int c) -> double & { return lo[std::size_t(r) * p + (r - c) - 1]; };
            for (int k = 0; k < count; ++k) {
                for (int j = std::max(0, k - p); j < k; ++j) {
                    double v = entry(first + k, first + j);
                    for (int i = std::max(0, k - p); i < j; ++i)
                        v -= L(k, i) * L(j, i) * dg[i];
                    L(k, j) = v / dg[j];
                }
                double v = entry(first + k, first + k);
                for (int j = std::max(0, k - p); j < k; ++j)
                    v -= L(k, j) * L(k, j) * dg[j];
                require(v > 0, "GPU path: 1D mass factorization failed");
                dg[k] = v;
            }
        };
        if (!o.periodic)
            factor(0, n1, lower[d], diagonal[d]);
        else {
            // Cyclic mass: band rows k < nb = n1 - p, dense tail rows (arrow LDL^T, no fill in the band rows).
            require(n1 >= 2 * p + 2, "GPU path: periodic direction too short for the cyclic line solve");
            const int nb = n1 - p;
            lower[d].assign(std::size_t(n1) * p, 0.);
            diagonal[d].assign(n1, 0.);
            tail[d].assign(std::size_t(p) * n1, 0.);
            auto L = [&](int r, int c) -> double { // L(r, c), c < r
                if (r >= nb)
                    return tail[d][std::size_t(r - nb) * n1 + c];
                return r - c <= p ? lower[d][std::size_t(r) * p + (r - c) - 1] : 0.;
            };
            for (int k = 0; k < n1; ++k) {
                const int j0 = k >= nb ? 0 : std::max(0, k - p);
                for (int j = j0; j < k; ++j) {
                    double v = entry(k, j);
                    const int i0 = j >= nb ? 0 : std::max(0, j - p);
                    for (int i = i0; i < j; ++i)
                        v -= L(k, i) * L(j, i) * diagonal[d][i];
                    const double l = v / diagonal[d][j];
                    if (k >= nb)
                        tail[d][std::size_t(k - nb) * n1 + j] = l;
                    else
                        lower[d][std::size_t(k) * p + (k - j) - 1] = l;
                }
                double v = entry(k, k);
                for (int j = j0; j < k; ++j)
                    v -= L(k, j) * L(k, j) * diagonal[d][j];
                require(v > 0, "GPU path: cyclic 1D mass factorization failed");
                diagonal[d][k] = v;
            }
            g.tail[d] = tail[d].data();
        }
        g.convection_band[d] = band[d].data();
        g.lower[d] = lower[d].data();
        g.diagonal[d] = diagonal[d].data();
        g.mirror_index[d] = s.mirror_index_[d];
        if (s.mirror_index_[d] >= 0) {
            require(s.mirror_index_[d] == 0 || s.mirror_index_[d] == n1 - 1, "GPU path: mirror node not at a line end");
            factor(s.mirror_index_[d] == 0 ? 1 : 0, n1 - 1, dlower[d], ddiagonal[d]);
            g.dlower[d] = dlower[d].data();
            g.ddiagonal[d] = ddiagonal[d].data();
            g.reflected[d] = s.reflected_local_[d].data();
        }
        const Sparse &F = s.flux_t_[d];
        require(F.isCompressed(), "GPU path: flux matrix not compressed");
        g.flux_outer[d] = F.outerIndexPtr();
        g.flux_inner[d] = F.innerIndexPtr();
        g.flux_value[d] = F.valuePtr();
        g.flux_nnz[d] = int(F.nonZeros());
    }
    // Parity bits of the active moments and mirror-plane bits of the nodes (mirror directions only).
    std::vector<unsigned char> comp_odd, node_mirror;
    if (std::any_of(s.mirror_index_.begin(), s.mirror_index_.end(), [](int m) { return m >= 0; })) {
        comp_odd.assign(q, 0);
        node_mirror.assign(s.nodes, 0);
        for (int d = 0; d < s.dimension; ++d) {
            if (s.mirror_index_[d] < 0)
                continue;
            for (int c = 0; c < q; ++c)
                if (s.reflect_odd_[d][c])
                    comp_odd[c] |= 1u << d;
            for (int i : s.mirror_nodes_[d])
                node_mirror[i] |= 1u << d;
        }
        g.comp_odd = comp_odd.data();
        g.node_mirror = node_mirror.data();
    }
    std::vector<ElementWeno::PairEntry> stacked_pairs;
    std::vector<double> stacked_total;
    if (s.graded_)
        for (const auto &c : s.classes_) {
            stacked_pairs.insert(stacked_pairs.end(), c.pairs.begin(), c.pairs.end());
            stacked_total.insert(stacked_total.end(), c.total_viscosity.data(), c.total_viscosity.data() + n);
        }
    const std::vector<ElementWeno::PairEntry> &pair_source = s.graded_ ? stacked_pairs : s.pair_table_;
    const std::size_t pairs = pair_source.size();
    std::vector<int> pj(pairs);
    std::vector<double> pd(pairs), pm(pairs), pc(3 * pairs), pct(3 * pairs);
    for (std::size_t k = 0; k < pairs; ++k) {
        const auto &pe = pair_source[k];
        pj[k] = pe.j, pd[k] = pe.d, pm[k] = pe.m;
        for (int d = 0; d < 3; ++d)
            pc[3 * k + d] = pe.c[d], pct[3 * k + d] = pe.ct[d];
    }
    g.pair_j = pj.data(), g.pair_d = pd.data(), g.pair_m = pm.data(), g.pair_c = pc.data(), g.pair_ct = pct.data();
    g.total_viscosity = s.graded_ ? stacked_total.data() : s.total_viscosity_.data();
    g.lumped = s.lumped.data();
    g.weno = o.weno, g.stabilize = o.stabilize, g.limit = o.limit, g.outward_bar_rounding = o.outward_bar_rounding;
    g.scale_invariant_weno = o.scale_invariant_weno;
    g.weno_q = o.weno_q, g.weno_r = o.weno_r, g.weno_epsilon = o.weno_epsilon, g.weno_linear = o.weno_linear;
    g.element_theta = o.element_theta;
    ctx_ = gpu::create(g);
    nb_ = int(s.boundary_nodes_.size());
    d_bnodes_ = gpu::device_ints(ctx_, 3, std::max(1, nb_));
    if (nb_)
        gpu::upload(d_bnodes_, s.boundary_nodes_.data(), sizeof(int) * nb_);
    a_h_ = Field::Zero(s.nodes, q);
    low_h_ = Field::Zero(s.nodes, 4);
    low_flux_h_ = Field::Zero(s.nodes, 12);
    closure_h_ = Field::Zero(s.nodes, 12);
    boundary_low_ = Field::Zero(s.nodes, 4);
    flux_area_ = Field::Zero(s.nodes, 3);
    boundary_bar_ = Field::Zero(s.nodes, 4);
    boundary_area_ = Vec::Zero(s.nodes);
    boundary_inflow_ = Vec::Zero(s.nodes);
    boundary_high_ = Field::Zero(nb_, q);
    load_b_ = Field::Zero(nb_, 4);
    const std::size_t nb = std::size_t(std::max(1, nb_));
    pin_fetch_ = static_cast<double *>(gpu::host_alloc(sizeof(double) * nb * (q + 28)));
    pin_high_ = static_cast<double *>(gpu::host_alloc(sizeof(double) * nb * q));
    pin_load_ = static_cast<double *>(gpu::host_alloc(sizeof(double) * nb * 4));
    pin_load_up_ = static_cast<double *>(gpu::host_alloc(sizeof(double) * nb * 4));
    // Split boundary buffers.
    const int face_rows = s.side_row_start_[6];
    for (int side = 0; side < 6; ++side)
        face_points_ = std::max(face_points_, int(s.rules_[side].weights.size()));
    if (!s.inflow.steady)
        for (auto &buffer : face_data_)
            buffer.resize(s.faces_.size() * std::size_t(face_points_));
    G_ = Field::Zero(face_rows, q);
    D_ = Field::Zero(face_rows, s.canonical_ ? s.canonical_->r : 1);
    H_ = Field::Zero(face_rows, q);
    L_ = Field::Zero(face_rows, 4);
    BR_ = Field::Zero(face_rows, 4);
    FA_ = Field::Zero(face_rows, 3);
    AR_ = Vec::Zero(face_rows);
    IN_ = Vec::Zero(face_rows);
    face_datum_.assign(s.faces_.size(), 0);
    require(!s.inflow.exterior_state && !s.inflow.exterior_m1, "GPU path: kinetic inflow data only (no exterior state)");
    if (s.reacts()) {
        gpu::ReactionSetup r;
        const bool material = bool(o.material);
        if (material)
            r.sigma_a = s.node_sigma_a.data(), r.sigma_s = s.node_sigma_s.data(), r.source = s.node_source.data();
        r.sa = o.sigma_a, r.ss = o.sigma_s;
        std::vector<double> filter(q);
        for (int k = 0; k < q; ++k) {
            const int l = int(std::sqrt(double(s.active[k]) + .5));
            filter[k] = o.filter_strength * std::pow(double(l) / (s.pn.order + 1), 4);
        }
        r.filter = filter.data();
        std::vector<double> fx, fy, shape;
        if (o.uncollided_density) {
            pulse_ = o.uncollided_pulse.get();
            lattice_ = o.uncollided_lattice.get();
            lattice3_ = o.uncollided_lattice3d.get();
            r.fcs_points = s.fcs_per_element_;
            fx.resize(s.fcs_points_.size()), fy.resize(s.fcs_points_.size());
            for (std::size_t k = 0; k < fx.size(); ++k)
                fx[k] = s.fcs_points_[k][0], fy[k] = s.fcs_points_[k][1];
            shape = row_major(s.fcs_shape_);
            r.fcs_x = fx.data(), r.fcs_y = fy.data(), r.fcs_weight = s.fcs_weight_.data();
            r.fcs_sigma_s = s.fcs_sigma_s_.data(), r.fcs_shape = shape.data(), r.fcs_node_mass = s.fcs_node_mass_.data();
        }
        gpu::reaction_setup(ctx_, r);
        if (lattice3_) {
            // lattice-3d: time table on the device (one thread per point), host copy for the summary.
            const double start = seconds();
            uncollided::Lattice3dUncollidedTable &T = *lattice3_;
            T.points = s.fcs_points_;
            T.M = T.times_for(T.final_time);
            T.table.assign(T.points.size() * std::size_t(T.M + 1), 0.);
            std::vector<double> px(T.points.size()), py(px.size()), pz(px.size());
            for (std::size_t k = 0; k < px.size(); ++k)
                px[k] = T.points[k][0], py[k] = T.points[k][1], pz[k] = T.points[k][2];
            T.lattice.with_view([&](const uncollided::Lattice3dView &view) {
                gpu::lattice3d_table(ctx_, view, px.data(), py.data(), pz.data(), T.M, T.dt, T.table.data());
                return 0;
            });
            T.build_seconds = seconds() - start;
        }
        if (lattice_) {
            // Build the time table on the device and hand the host copy to the shared table object (the host FCS
            // path then finds it built for the same points).
            const double start = seconds();
            uncollided::LatticeUncollidedTable &T = *lattice_;
            const auto px = uncollided::gauss_legendre(T.lattice.phi_order, 0, 1);
            const auto tx = uncollided::gauss_legendre(T.lattice.theta_order, 0, 1);
            const uncollided::LatticeView view = T.lattice.view(px.first, px.second, tx.first, tx.second);
            T.points = s.fcs_points_;
            T.M = T.times_for(T.final_time);
            T.table.assign(T.points.size() * std::size_t(T.M + 1), 0.);
            const double assigned = seconds();
            gpu::lattice_table(ctx_, view, T.M, T.dt, T.table.data());
            T.build_seconds = seconds() - start;
            if (const char *e = std::getenv("M1PN_CTOR_TIMING"); e && std::atoi(e))
                std::cerr << "gpu: lattice table host alloc " << assigned - start << " s, device build + copy "
                          << seconds() - assigned << " s (" << T.table.size() * 8e-9 << " GB)\n";
        }
    }
    // Ghost projection on the device (plain full-sphere rule, unweighted); M1PN_GPU_GHOST=0 keeps it on the host.
    // Default: on the device when the boundary work (ghost pairs x rule points) is large against the volume work
    // (nodes x moments); measured break-even ~0.1 (2D P7 512^2: 0.08, host faster; 2D P21 256^2: 0.14 and 3D: device
    // faster). M1PN_GPU_GHOST=0/1 forces host/device.
    const char *env = std::getenv("M1PN_GPU_GHOST");
    int max_points = 0;
    for (int side = 0; side < 6; ++side)
        if (s.boundaries_[side])
            max_points = std::max(max_points, int(s.out_rules_[side].weight.size()));
    const double ratio = double(s.ghost_node_.size()) * max_points / (double(s.nodes) * s.moments);
    device_ghost_ = (env ? std::atoi(env) != 0 : ratio > .1) && o.fullsphere_ghost && o.positivity_projection &&
                    !o.projection_half_range && !s.ghost_node_.empty() && !interfaces_;
    for (int side = 0; side < 6 && device_ghost_; ++side)
        if (s.boundaries_[side] && s.out_rules_[side].c.size())
            device_ghost_ = false;
    if (device_ghost_) {
        gpu::GhostSetup g;
        g.pairs = int(s.ghost_node_.size());
        g.node = s.ghost_node_.data();
        g.side = s.ghost_side_.data();
        std::array<std::vector<double>, 6> normals;
        for (int side = 0; side < 6; ++side) {
            if (!s.boundaries_[side])
                continue;
            const auto &out = s.out_rules_[side];
            g.points[side] = int(out.weight.size());
            g.outgoing[side] = out.outgoing;
            g.max_points = std::max(g.max_points, g.points[side]);
            g.harmonics[side] = s.out_harmonics_[side].data();
            g.ox[side] = out.ox.data(), g.oy[side] = out.oy.data(), g.oz[side] = out.oz.data();
            g.w[side] = out.weight.data();
            g.facets[side] = out.facets.count();
            g.fnx[side] = out.facets.nx.data(), g.fny[side] = out.facets.ny.data(), g.fnz[side] = out.facets.nz.data();
            g.fh[side] = out.facets.h.data();
            g.inradius[side] = out.facets.inradius;
            const V3 n = s.boundaries_[side]->n;
            normals[side] = {n[0], n[1], n[2]};
            g.normal[side] = normals[side].data();
        }
        g.sigma = s.ghost_sigma_.size() ? s.ghost_sigma_.data() : nullptr;
        g.tolerance = o.projection_tolerance;
        g.warm_start = o.projection_warm_start;
        gpu::ghost_setup(ctx_, g);
        pin_ghost_m_ = static_cast<double *>(gpu::host_alloc(sizeof(double) * 4 * g.pairs));
        pin_ghost_flags_ = static_cast<int *>(gpu::host_alloc(sizeof(int) * (g.pairs + 1)));
    }
    // Ghost traces on the device (host projection path only): psi = harmonics (ghost filter applied) x stage input rows,
    // computed on the aux stream while the device runs the volume kernels; the host projections read them in phase B.
    {
        const char *tenv = std::getenv("M1PN_GPU_GHOST_TRACES");
        // Default off (docs/STATIC_HP_BLOCKS.md): no speed-up once the device is the bottleneck, and the validation
        // case graded_lattice_q1_P7 deviates by 4e-7 from the CPU with it (cause not yet understood).
        trace_ghost_ = (tenv ? std::atoi(tenv) != 0 : false) && o.fullsphere_ghost && !device_ghost_ &&
                       !s.ghost_node_.empty() && !interfaces_;
    }
    if (trace_ghost_) {
        gpu::GhostTraceSetup g;
        g.q = s.moments;
        std::array<std::vector<int>, 6> nodes;
        std::array<Mat, 6> harmonics;
        std::size_t total = 0;
        for (int side = 0; side < 6; ++side) {
            trace_offset_[side] = total;
            const std::vector<int> &list = s.ghost_by_side_[side];
            if (list.empty())
                continue;
            for (int k : list)
                nodes[side].push_back(s.ghost_node_[k]);
            harmonics[side] = s.out_rules_[side].harmonics; // points x q, column-major
            if (s.ghost_sigma_.size())
                harmonics[side] = harmonics[side] * s.ghost_sigma_.asDiagonal();
            g.count[side] = int(list.size()), g.points[side] = int(harmonics[side].rows());
            g.nodes[side] = nodes[side].data(), g.harmonics[side] = harmonics[side].data();
            total += std::size_t(g.points[side]) * g.count[side];
        }
        gpu::ghost_traces_setup(ctx_, g);
        pin_traces_ = static_cast<double *>(gpu::host_alloc(sizeof(double) * std::max<std::size_t>(1, total)));
    }
    // Phase A on the device: the per-row trace weights M_side(n, m) and nodes, the side matrices and the node lists.
    const char *high_env = std::getenv("M1PN_GPU_BOUNDARY_HIGH");
    device_high_ = (high_env ? std::atoi(high_env) != 0 : true) && nb_ > 0 && face_rows > 0 && !interfaces_;
    require(!s.broken_ || interfaces_ || device_high_, "GPU path: broken lines need the device face load (phase A)");
    require(!nodal_interfaces_ || device_high_, "GPU path: conforming interfaces need the device face load (phase A)");
    {
        std::vector<char> full(s.nodes, 0);
        const bool all = !device_high_ || interfaces_ || !s.broken_;
        for (std::size_t index = 0; index < s.faces_.size() && !all; ++index)
            if (!s.face_break_[index]) {
                const auto &f = s.faces_[index];
                for (int l : s.rules_[2 * f.direction + f.side].local)
                    full[s.connectivity_[f.element * s.local_nodes + l]] = 1;
            }
        full_nodes_.clear();
        for (int k = 0; k < nb_; ++k)
            if (all || full[s.boundary_nodes_[k]])
                full_nodes_.push_back(s.boundary_nodes_[k]);
        nf_ = int(full_nodes_.size());
        if (nf_ == nb_)
            d_fnodes_ = d_bnodes_;
        else {
            d_fnodes_ = gpu::device_ints(ctx_, 5, std::max(1, nf_));
            if (nf_)
                gpu::upload(d_fnodes_, full_nodes_.data(), sizeof(int) * nf_);
        }
    }
    if (device_high_) {
        const int nf = int(s.rules_[2 * s.faces_[0].direction + s.faces_[0].side].local.size());
        // Broken lines: each row also carries the partner nodes with negative weights (M_face (u - u_partner)).
        const int width = s.broken_ ? 2 * nf : nf;
        std::vector<int> row_nodes(std::size_t(face_rows) * width, 0);
        std::vector<double> row_mass(std::size_t(face_rows) * width, 0.);
        for (std::size_t index = 0; index < s.faces_.size(); ++index) {
            const auto &f = s.faces_[index];
            const auto &rule = s.rules_[2 * f.direction + f.side];
            require(int(rule.local.size()) == nf, "GPU boundary: faces with different node counts");
            const int r0 = s.face_row_[index];
            for (int n = 0; n < nf; ++n)
                for (int m = 0; m < nf; ++m) {
                    const std::size_t at = std::size_t(r0 + n) * width + m;
                    row_nodes[at] = s.connectivity_[std::size_t(f.element) * s.local_nodes + rule.local[m]];
                    row_mass[at] = s.face_scale_[index] * rule.mass(n, m); // graded: area of this face
                    if (s.broken_ && s.face_break_[index]) {
                        row_nodes[at + nf] = s.broken_partner_node(int(index), m);
                        row_mass[at + nf] = -row_mass[at];
                    }
                }
        }
        gpu::BoundaryHighSetup h;
        h.q = q, h.rows = face_rows, h.nf = width, h.nb = nb_;
        h.row_nodes = row_nodes.data(), h.row_mass = row_mass.data();
        for (int side = 0; side <= 6; ++side)
            h.side_start[side] = s.side_row_start_[side];
        for (int side = 0; side < 6; ++side)
            if (s.boundaries_[side]) {
                require(s.boundary_flux_t_[side].rows() == q && s.boundary_flux_t_[side].cols() == q,
                        "GPU boundary: flux matrix size");
                h.flux_t[side] = s.boundary_flux_t_[side].data();
                if (s.broken_) {
                    h.break_start[side] = s.break_row_start_[side];
                    h.iface_t[side] = s.interface_t_[side].data();
                }
            }
        h.offsets = s.boundary_row_offsets_.data(), h.rows_list = s.boundary_rows_.data(), h.bnodes = d_bnodes_;
        split_ = s.options.split_breaks && s.broken_;
        std::vector<int> row_break;
        if (split_) {
            row_break.assign(std::max(1, face_rows), 0);
            for (int r = 0; r < face_rows; ++r)
                row_break[r] = r >= s.break_row_start_[s.face_row_side_[r]];
            h.row_break = row_break.data();
            pin_low4b_ = static_cast<double *>(gpu::host_alloc(sizeof(double) * std::size_t(nb_) * 4));
            brk_low_ = Field::Zero(s.nodes, 4), brk_bar_ = Field::Zero(s.nodes, 4), brk_fa_ = Field::Zero(s.nodes, 3);
            brk_high4_ = Field::Zero(s.nodes, 4), split_term_h_ = Field::Zero(s.nodes, 4);
            brk_area_ = Vec::Zero(s.nodes);
        }
        gpu::boundary_high_setup(ctx_, h);
        pin_db_ = static_cast<double *>(gpu::host_alloc(sizeof(double) * std::size_t(face_rows) * q));
        pin_low4_ = static_cast<double *>(gpu::host_alloc(sizeof(double) * std::size_t(nb_) * 4));
    }
}
ElementWenoGpu::~ElementWenoGpu() {
    gpu::host_free(pin_fetch_);
    gpu::host_free(pin_high_);
    gpu::host_free(pin_load_);
    gpu::host_free(pin_load_up_);
    if (pin_ghost_m_)
        gpu::host_free(pin_ghost_m_), gpu::host_free(pin_ghost_flags_);
    if (pin_traces_)
        gpu::host_free(pin_traces_);
    if (pin_db_)
        gpu::host_free(pin_db_), gpu::host_free(pin_low4_);
    if (pin_low4b_)
        gpu::host_free(pin_low4b_);
    gpu::destroy(ctx_);
}
// Phase A of add_boundary: the PN face load sum_points w phi_n [(A_n - llf_homogeneous) trace - llf_datum datum] per
// boundary node (compact rows of boundary_high_). Same operations as add_boundary; the per-point inflow data are kept
// in face_data_ for phase B.
int ElementWenoGpu::select_face(double time) {
    if (s_.inflow.steady) {
        if (!steady_ready_) {
            fill_datum(0);
            steady_ready_ = true;
        }
        return 0;
    }
    for (int k = 0; k < 2; ++k)
        if (face_time_[k] == time)
            return last_face_ = k;
    prepare_face_data(time);
    return select_face(time);
}
void ElementWenoGpu::prepare_face_data(double time) {
    const ElementWeno &s = s_;
    if (s.inflow.steady)
        return;
    for (int k = 0; k < 2; ++k)
        if (face_time_[k] == time)
            return;
    const int keep = last_face_ == 0 ? 1 : 0; // never the buffer of the current stage
    std::vector<Incoming> &buffer = face_data_[keep];
    parallel_for(int(s.faces_.size()), [&](int index) {
        const auto &f = s.faces_[index];
        const int side_id = 2 * f.direction + f.side;
        const auto &rule = s.rules_[side_id];
        V3 origin;
        s.element_origin(f.element, origin);
        for (int point = 0; point < rule.weights.size(); ++point) {
            V3 x = origin;
            for (int d = 0; d < s.dimension; ++d)
                x[d] += s.width(f.element, d) * rule.points(point, d);
            s.face_data_into(side_id, x, time, buffer[std::size_t(index) * face_points_ + point]);
        }
    });
    fill_datum(keep);
    face_time_[keep] = time;
}
// Face datum loads D (rows n of a face: sum_points w phi_n d) and DB = D llf_datum^T, as in add_boundary.
void ElementWenoGpu::fill_datum(int k) {
    const ElementWeno &s = s_;
    const int face_rows = s.side_row_start_[6];
    if (DB_[k].rows() != face_rows)
        DB_[k] = Field::Zero(face_rows, s.moments);
    parallel_for(int(s.faces_.size()), [&](int index) {
        const auto &f = s.faces_[index];
        const int side_id = 2 * f.direction + f.side;
        const auto &rule = s.rules_[side_id];
        const int nf = rule.local.size(), r0 = s.face_row_[index];
        auto datum = D_.middleRows(r0, nf);
        datum.setZero();
        if ((s.broken_ && s.face_break_[index]) || s.nodal_interface(side_id)) {
            face_datum_[index] = 0; // broken line / conforming interface: the partner term is not an inflow datum
            return;
        }
        for (int point = 0; point < rule.weights.size(); ++point) {
            const Incoming &data = s.inflow.steady ? s.steady_data_[std::size_t(index) * rule.weights.size() + point]
                                                   : face_data_[k][std::size_t(index) * face_points_ + point];
            for (int n = 0; n < nf; ++n) {
                const double weight = s.face_scale_[index] * rule.weights[point] * rule.shape(n, point);
                if (weight == 0)
                    continue;
                datum.row(n).noalias() += weight * data.d.transpose();
            }
        }
        face_datum_[index] = !datum.isZero(0);
    });
    datum_zero_[k].fill(true);
    for (std::size_t index = 0; index < s.faces_.size(); ++index)
        if (face_datum_[index])
            datum_zero_[k][2 * s.faces_[index].direction + s.faces_[index].side] = false;
    constexpr int rows_per_task = 16;
    std::vector<std::array<int, 3>> tasks;
    for (int side = 0; side < 6; ++side)
        if (s.boundaries_[side] && !datum_zero_[k][side])
            for (int r = s.side_row_start_[side]; r < s.side_row_start_[side + 1]; r += rows_per_task)
                tasks.push_back({side, r, std::min(rows_per_task, s.side_row_start_[side + 1] - r)});
    parallel_for(int(tasks.size()), [&](int t) {
        const auto [side, r, n] = tasks[t];
        DB_[k].middleRows(r, n).noalias() = D_.middleRows(r, n).lazyProduct(s.datum_t_[side]);
    });
}
// Phase A of add_boundary: the PN face load sum_points w phi_n [(A_n - llf_homogeneous) trace - llf_datum datum] per
// boundary node (compact rows of boundary_high_), with the datum part DB prepared ahead.
void ElementWenoGpu::boundary_high(int buffer) {
    const ElementWeno &s = s_;
    const int q = s.moments;
    parallel_for(int(s.faces_.size()), [&](int index) {
        const auto &f = s.faces_[index];
        const int side_id = 2 * f.direction + f.side;
        const auto &rule = s.rules_[side_id];
        const int nf = rule.local.size(), r0 = s.face_row_[index];
        thread_local Mat traces;
        traces.resize(nf, q);
        for (int n = 0; n < nf; ++n)
            traces.row(n) = a_h_.row(s.connectivity_[f.element * s.local_nodes + rule.local[n]]);
        G_.middleRows(r0, nf).noalias() = rule.mass * traces;
        if (s.graded_)
            G_.middleRows(r0, nf) *= s.face_scale_[index];
    });
    constexpr int rows_per_task = 16;
    std::vector<std::array<int, 3>> tasks;
    for (int side = 0; side < 6; ++side)
        if (s.boundaries_[side])
            for (int r = s.side_row_start_[side]; r < s.side_row_start_[side + 1]; r += rows_per_task)
                tasks.push_back({side, r, std::min(rows_per_task, s.side_row_start_[side + 1] - r)});
    const Field &DB = DB_[buffer];
    const bool *zero = datum_zero_[buffer].data();
    parallel_for(int(tasks.size()), [&](int t) {
        const auto [side, r, n] = tasks[t];
        H_.middleRows(r, n).noalias() = G_.middleRows(r, n).lazyProduct(s.boundary_flux_t_[side]);
        if (!zero[side])
            H_.middleRows(r, n) -= DB.middleRows(r, n);
    });
    parallel_for(nb_, [&](int k) {
        boundary_high_.row(k).setZero();
        for (int e = s.boundary_row_offsets_[k]; e < s.boundary_row_offsets_[k + 1]; ++e)
            boundary_high_.row(k) += H_.row(s.boundary_rows_[e]);
    });
}
// Phase B of add_boundary: ghost moments (C6 / positivity projection) and the low-order face terms, reduced into the
// node rows of boundary_low_, flux_area_, boundary_bar_, boundary_area_, boundary_inflow_.
void ElementWenoGpu::boundary_low(int buffer, PositivityProjectionStats *projection) {
    const ElementWeno &s = s_;
    const std::vector<Incoming> *fd = s.inflow.steady ? nullptr : &face_data_[buffer];
    const ElementWenoOptions &options = s.options;
    const int dimension = s.dimension;
    const Field &low = low_h_, &low_flux = low_flux_h_;
    const double ghost_start = seconds();
    const bool ghost = options.fullsphere_ghost;
    outgoing_.resize(ghost ? s.ghost_node_.size() : 0);
    const bool project = ghost && options.positivity_projection;
    PositivityProjectionOptions popt;
    popt.tolerance = options.projection_tolerance;
    std::vector<PositivityProjectionStats> pstats(project ? std::max(1, omp_threads()) : 0);
    auto ghost_pair = [&](int k, const Vec &psi) {
        const int side_id = s.ghost_side_[k], node = s.ghost_node_[k];
        const auto &out = s.out_rules_[side_id];
        const Boundary &b = *s.boundaries_[side_id];
        thread_local Vec z;
        auto positive_part = [&]() {
            Four m = Four::Zero();
            for (int j = 0; j < out.outgoing; ++j) {
                const double v = std::max(0., psi[j]) * out.weight[j];
                m[0] += v;
                m[1] += v * out.omega(j, 0);
                m[2] += v * out.omega(j, 1);
                m[3] += v * out.omega(j, 2);
            }
            return m;
        };
        Four m;
        bool projected = false;
        if (project) {
            PositivityProjectionStats &ps = pstats[omp_thread()];
            const bool half = options.projection_half_range;
            Four U(low(node, 0), low(node, 1), low(node, 2), low(node, 3));
            if (half) {
                U.setZero();
                for (int j = 0; j < out.outgoing; ++j) {
                    const double v = psi[j] * out.weight[j];
                    U[0] += v, U[1] += v * out.omega(j, 0), U[2] += v * out.omega(j, 1), U[3] += v * out.omega(j, 2);
                }
            }
            const int q = half ? out.outgoing : int(psi.size());
            z.resize(q);
            PositivityProjectionWarm &warm = s.projection_warm_[k];
            PositivityProjectionOptions nopt = popt;
            nopt.augment = !half;
            if (out.c.size())
                nopt.c = out.c.data(), nopt.cinv = out.cinv.data();
            const auto o = project_positive_moments(q, out.ox.data(), out.oy.data(), out.oz.data(), out.weight.data(),
                                                    psi.data(), U, z.data(), nopt,
                                                    options.projection_warm_start ? &warm : nullptr, &out.facets);
            const auto &r = o.result;
            ++ps.calls;
            ps.certificates += o.certificate;
            ps.farkas += o.farkas;
            if (r.converged) {
                if (!o.augmented)
                    warm.xi = r.xi;
                else if (!o.dirac)
                    warm.xi_augmented = r.xi;
                projected = true;
                ps.identities += r.identity;
                ps.vacuum += !(U[0] > 0);
                ps.augmented += o.augmented;
                ps.diracs += o.dirac;
                ps.iterations += o.newton_iterations;
                ps.backtracks += r.backtracks;
                ps.max_iterations = std::max(ps.max_iterations, o.newton_iterations);
                ps.max_residual = std::max(ps.max_residual, r.residual);
                if (U[0] > 0 && !o.augmented)
                    ps.min_active_fraction = std::min(ps.min_active_fraction, double(r.active) / q);
                m.setZero();
                for (int j = 0; j < out.outgoing; ++j) {
                    const double v = z[j] * out.weight[j];
                    m[0] += v;
                    m[1] += v * out.omega(j, 0);
                    m[2] += v * out.omega(j, 1);
                    m[3] += v * out.omega(j, 2);
                }
                if (o.augmented && o.atom_direction.dot(b.n) > 0)
                    m += o.atom;
                if (m[0] > U[0] * (1 + 64 * std::numeric_limits<double>::epsilon())) {
                    ++ps.capped;
                    m *= U[0] > 0 ? U[0] / m[0] : 0.;
                }
            } else {
                ++ps.failures;
                if (options.projection_strict)
                    throw StepRejected("Positivity projection failed at boundary node " + std::to_string(node));
            }
        }
        if (!projected)
            m = positive_part();
        const double b_node = low(node, 0) + b.n[0] * low(node, 1) + b.n[1] * low(node, 2) + b.n[2] * low(node, 3);
        const double a_ext = m[0] - b.n.dot(m.tail<3>());
        if (a_ext > b_node) {
            if (projected)
                ++pstats[omp_thread()].scaled;
            m *= b_node > 0 ? b_node / a_ext : 0.;
        }
        outgoing_[k] = m;
    };
    if (ghost && device_ghost_) {
        // Device projection (gpu::ghost): outgoing moments of all pairs; the flagged ones (hull certificate, augmented
        // rule) on the host with the original code.
        {
            const double t0 = seconds();
            gpu::wait(ctx_, 2);
            t_wait_ghost += seconds() - t0;
        }
        const int pairs = int(s.ghost_node_.size());
        for (int k = 0; k < pairs; ++k)
            outgoing_[k] = Four(pin_ghost_m_[4 * k], pin_ghost_m_[4 * k + 1], pin_ghost_m_[4 * k + 2], pin_ghost_m_[4 * k + 3]);
        const int flagged = pin_ghost_flags_[0];
        ghost_flagged += flagged;
        parallel_for(flagged, [&](int f) {
            const int k = pin_ghost_flags_[1 + f], side = s.ghost_side_[k];
            Vec coeff = a_h_.row(s.ghost_node_[k]).transpose();
            if (s.ghost_sigma_.size())
                coeff = s.ghost_sigma_.cwiseProduct(coeff);
            const Vec psi = s.out_rules_[side].harmonics * coeff;
            ghost_pair(k, psi);
        });
    } else if (ghost) {
        const int ghost_block = std::clamp(int(s.ghost_node_.size()) / std::max(1, 4 * omp_threads()), 4, 32);
        std::vector<std::array<int, 3>> tasks;
        for (int side = 0; side < 6; ++side)
            for (int begin = 0; begin < int(s.ghost_by_side_[side].size()); begin += ghost_block)
                tasks.push_back({side, begin, std::min<int>(begin + ghost_block, s.ghost_by_side_[side].size())});
        if (trace_ghost_) {
            const double t0 = seconds();
            gpu::wait(ctx_, 2);
            t_wait_ghost += seconds() - t0;
        }
        parallel_for(int(tasks.size()), [&](int t) {
            const auto [side, begin, end] = tasks[t];
            const std::vector<int> &list = s.ghost_by_side_[side];
            if (trace_ghost_) { // traces from the device: column j of side `side` (points_s doubles each)
                const int points = int(s.out_rules_[side].harmonics.rows());
                thread_local Vec psi_d;
                const double t1 = seconds();
                for (int j = begin; j < end; ++j) {
                    psi_d = Eigen::Map<const Vec>(pin_traces_ + trace_offset_[side] + std::size_t(j) * points, points);
                    ghost_pair(list[j], psi_d);
                }
                if (project)
                    pstats[omp_thread()].reduce_seconds += seconds() - t1;
                return;
            }
            thread_local Mat at, traces;
            thread_local Vec psi;
            at.resize(s.moments, end - begin);
            for (int j = begin; j < end; ++j)
                at.col(j - begin) = a_h_.row(s.ghost_node_[list[j]]).transpose();
            if (s.ghost_sigma_.size())
                at = s.ghost_sigma_.asDiagonal() * at;
            const double t0 = seconds();
            if (s.separable_[side])
                s.separable_[side]->evaluate(at, traces);
            else
                traces.noalias() = s.out_rules_[side].harmonics * at;
            const double t1 = seconds();
            for (int j = begin; j < end; ++j) {
                psi = traces.col(j - begin);
                ghost_pair(list[j], psi);
            }
            const double t2 = seconds();
            if (project) { // diagnostics (thread seconds): trace products vs projections
                PositivityProjectionStats &ps = pstats[omp_thread()];
                ps.product_seconds += t1 - t0, ps.reduce_seconds += t2 - t1;
            }
        });
    }
    if (project && projection)
        for (const auto &ps : pstats)
            projection->merge(ps);
    const double ghost_end = seconds();
    t_ghost += ghost_end - ghost_start;
    if (projection)
        projection->seconds += ghost_end - ghost_start;
    parallel_for(int(s.faces_.size()), [&](int index) {
        const auto &f = s.faces_[index];
        const int side_id = 2 * f.direction + f.side;
        const auto &rule = s.rules_[side_id];
        const Boundary &b = *s.boundaries_[side_id];
        const int nf = rule.local.size(), r0 = s.face_row_[index];
        thread_local std::vector<int> global;
        global.resize(nf);
        auto load_row = L_.middleRows(r0, nf), bar_row = BR_.middleRows(r0, nf), fa_row = FA_.middleRows(r0, nf);
        load_row.setZero(), bar_row.setZero(), fa_row.setZero();
        AR_.segment(r0, nf).setZero(), IN_.segment(r0, nf).setZero();
        for (int n = 0; n < nf; ++n)
            global[n] = s.connectivity_[f.element * s.local_nodes + rule.local[n]];
        if (s.broken_ && s.face_break_[index]) { // broken line: node-wise bar states with the partner (no ghost)
            s.broken_face_low(int(index), low_h_, low_flux_h_, L_, BR_, FA_, AR_, IN_);
            return;
        }
        if (s.nodal_interface(side_id)) { // conforming interface: the same with the neighbour block's node
            s.interface_face_low(int(index), low_h_, low_flux_h_, L_, BR_, FA_, AR_, IN_);
            return;
        }
        const int *ghost_id = &s.ghost_of_face_[std::size_t(index) * nf];
        thread_local std::vector<char> memo_valid;
        thread_local std::vector<Four> memo_moments, memo_state;
        memo_valid.assign(nf, 0);
        memo_moments.resize(nf), memo_state.resize(nf);
        // Bar state of face node n (exterior from the data plus the ghost); as ElementWeno::add_boundary.
        auto bar_state = [&](int n, int i, const Four &ui, const Incoming &data) {
            Four state;
            Four ext = data.moments;
            if (ghost)
                ext += outgoing_[ghost_id[n]];
            bool in_cone = exact::cone(ext);
            for (int k = 0; ghost && k < 64 && !in_cone; ++k) {
                ext[0] = std::nextafter(ext[0], std::numeric_limits<double>::infinity());
                in_cone = exact::cone(ext);
            }
            cone_require(in_cone, "M1 exterior state outside cone");
            Four own = Four::Zero();
            for (int d = 0; d < dimension; ++d)
                own += b.n[d] * Four(low_flux(i, 4 * d), low_flux(i, 4 * d + 1), low_flux(i, 4 * d + 2),
                                     low_flux(i, 4 * d + 3));
            const Four outer = Levermore(ext).flux(b.n);
            state = .5 * (ui + own) + .5 * (ext - outer);
            if (options.outward_bar_rounding && !exact::cone(state)) {
                const double size = std::abs(ui[0]) + std::abs(own[0]) + std::abs(ext[0]) + std::abs(outer[0]);
                if (state.tail<3>().norm() - state[0] <= 64 * std::numeric_limits<double>::epsilon() * size)
                    state[0] = norm_up(state);
            }
            return state;
        };
        if (!s.face_uniform_.empty() && s.face_uniform_[index]) { // uniform face (steady): point-independent terms
            const Incoming &data = s.steady_data_[std::size_t(index) * rule.weights.size()];
            for (int n = 0; n < nf; ++n) {
                const double weight = s.face_weight_[r0 + n];
                if (weight == 0)
                    continue;
                const int i = global[n];
                const Four ui(low(i, 0), low(i, 1), low(i, 2), low(i, 3));
                IN_[r0 + n] = weight * std::max(0., -data.kinetic[0]);
                const Four state = bar_state(n, i, ui, data);
                AR_[r0 + n] = weight;
                bar_row.row(n) = weight * state.transpose();
                for (int d = 0; d < dimension; ++d)
                    fa_row(n, d) = weight * b.n[d];
                load_row.row(n) = weight * (state - ui).transpose();
            }
        } else
        for (int point = 0; point < rule.weights.size(); ++point) {
            const Incoming &data = s.inflow.steady ? s.steady_data_[std::size_t(index) * rule.weights.size() + point]
                                                   : (*fd)[std::size_t(index) * face_points_ + point];
            for (int n = 0; n < nf; ++n) {
                const double weight = s.face_scale_[index] * rule.weights[point] * rule.shape(n, point);
                if (weight == 0)
                    continue;
                const int i = global[n];
                const Four ui(low(i, 0), low(i, 1), low(i, 2), low(i, 3));
                IN_[r0 + n] += weight * std::max(0., -data.kinetic[0]);
                Four state;
                if (memo_valid[n] && (memo_moments[n].array() == data.moments.array()).all())
                    state = memo_state[n];
                else {
                    state = bar_state(n, i, ui, data);
                    memo_valid[n] = 1, memo_moments[n] = data.moments, memo_state[n] = state;
                }
                AR_[r0 + n] += weight;
                bar_row.row(n) += weight * state.transpose();
                for (int d = 0; d < dimension; ++d)
                    fa_row(n, d) += weight * b.n[d];
                load_row.row(n) += weight * (state - ui).transpose();
            }
        }
    });
    if (projection)
        projection->face_seconds += seconds() - ghost_end;
    parallel_for(nb_, [&](int k) {
        const int i = s.boundary_nodes_[k];
        boundary_low_.row(i).setZero();
        flux_area_.row(i).setZero();
        boundary_bar_.row(i).setZero();
        boundary_area_[i] = 0;
        boundary_inflow_[i] = 0;
        if (split_)
            brk_low_.row(i).setZero(), brk_bar_.row(i).setZero(), brk_fa_.row(i).setZero(), brk_area_[i] = 0;
        for (int e = s.boundary_row_offsets_[k]; e < s.boundary_row_offsets_[k + 1]; ++e) {
            const int r = s.boundary_rows_[e];
            if (split_ && r >= s.break_row_start_[s.face_row_side_[r]]) { // split_breaks: broken-line rows apart
                brk_low_.row(i) += L_.row(r);
                brk_bar_.row(i) += BR_.row(r);
                brk_fa_.row(i) += FA_.row(r);
                brk_area_[i] += AR_[r];
                continue;
            }
            boundary_low_.row(i) += L_.row(r);
            boundary_bar_.row(i) += BR_.row(r);
            flux_area_.row(i) += FA_.row(r);
            boundary_area_[i] += AR_[r];
            boundary_inflow_[i] += IN_[r];
        }
    });
}
void ElementWenoGpu::check_error() {
    const gpu::Counters c = gpu::counters(ctx_);
    if (!c.error)
        return;
    static const char *what[] = {"",
                                 "Levermore state outside the stored cone",
                                 "",
                                 "bar state left the M1 cone beyond rounding (256 ulp)",
                                 "low-order base bar outside the stored cone",
                                 "limiter endpoint rounding check failed",
                                 "stored row left the M1 cone beyond rounding (4096 ulp)",
                                 "stored row outside the exact device range of the cone predicate"};
    throw StepRejected(std::string("GPU path: ") + (c.error < 8 ? what[c.error] : "unknown device error"));
}
// Closure rows of near-beam nodes with the host Levermore closure (long double gap).
void ElementWenoGpu::closure_fallback(const double *x) {
    const gpu::Counters c = gpu::counters(ctx_);
    const int count = c.closure_fallbacks;
    if (!count)
        return;
    require(count <= fallback_capacity_, "GPU path: too many closure fallbacks");
    const double start = seconds();
    const int q = s_.moments;
    const int *list = gpu::closure_fallback_list(ctx_);
    std::vector<double> rows(std::size_t(count) * q), low(std::size_t(count) * 4, 0.), lf(std::size_t(count) * 12, 0.),
        cl(std::size_t(count) * 12, 0.);
    gpu::gather_device(ctx_, x, q, list, count, rows.data());
    for (int k = 0; k < count; ++k) {
        const double *row = &rows[std::size_t(k) * q];
        const Four u = project(Eigen::Map<const Vec>(row, q));
        for (int r = 0; r < 4; ++r)
            low[std::size_t(k) * 4 + r] = u[r];
        const Levermore closure_state(u);
        for (int d = 0; d < s_.dimension; ++d) {
            V3 n = V3::Zero();
            n[d] = 1;
            const Four fl = closure_state.flux(n);
            Four head = Four::Zero();
            add_flux(s_.flux_t_[d], row, 1., head.data(), 4);
            const Four fh = project(head);
            for (int r = 0; r < 4; ++r) {
                lf[std::size_t(k) * 12 + 4 * d + r] = fl[r];
                cl[std::size_t(k) * 12 + 4 * d + r] = fh[r] - fl[r];
            }
        }
    }
    const gpu::Buffers b = gpu::buffers(ctx_);
    gpu::scatter_device(ctx_, b.low, 4, list, count, low.data(), false, s_.nodes);
    gpu::scatter_device(ctx_, b.low_flux, 12, list, count, lf.data(), false, s_.nodes);
    gpu::scatter_device(ctx_, b.closure, 12, list, count, cl.data(), false, s_.nodes);
    closure_fallbacks += count;
    t_fallback += seconds() - start;
}
// Elements whose limiter needs the stable base bar (long double): the element body of limited_low_load on the host.
void ElementWenoGpu::limiter_fallback(ElementWenoStats &stats) {
    const gpu::Counters c = gpu::counters(ctx_);
    const int count = c.limiter_fallbacks;
    if (!count)
        return;
    require(count <= fallback_capacity_, "GPU path: too many limiter fallbacks");
    const double start = seconds();
    const int n = s_.local_nodes, dimension = s_.dimension;
    const ElementWenoOptions &options = s_.options;
    std::vector<int> elements(count);
    gpu::download(elements.data(), gpu::limiter_fallback_list(ctx_), sizeof(int) * count);
    std::vector<int> nodes(std::size_t(count) * n);
    for (int k = 0; k < count; ++k)
        for (int l = 0; l < n; ++l)
            nodes[std::size_t(k) * n + l] = s_.connectivity_[std::size_t(elements[k]) * n + l];
    int *d_nodes = gpu::device_ints(ctx_, 1, int(nodes.size()));
    gpu::upload(d_nodes, nodes.data(), sizeof(int) * nodes.size());
    const gpu::Buffers b = gpu::buffers(ctx_);
    const int m = int(nodes.size());
    std::vector<double> low(std::size_t(m) * 4), lf(std::size_t(m) * 12), cl(std::size_t(m) * 12), mr(std::size_t(m) * 4),
        stab(std::size_t(count) * 4 * n), out(std::size_t(count) * 4 * n);
    gpu::gather_device(ctx_, b.low, 4, d_nodes, m, low.data(), s_.nodes);
    gpu::gather_device(ctx_, b.low_flux, 12, d_nodes, m, lf.data(), s_.nodes);
    gpu::gather_device(ctx_, b.closure, 12, d_nodes, m, cl.data(), s_.nodes);
    gpu::gather_device(ctx_, b.moment_rate, 4, d_nodes, m, mr.data(), s_.nodes);
    const int *d_elements = gpu::limiter_fallback_list(ctx_);
    gpu::gather_device(ctx_, b.stab_low, 4 * n, d_elements, count, stab.data(), s_.cells);
    using Quad = Eigen::Map<const Four>;
    for (int k = 0; k < count; ++k) {
        const Vec &total_viscosity = s_.graded_ ? s_.klass(elements[k]).total_viscosity : s_.total_viscosity_;
        const std::vector<ElementWeno::PairEntry> &pair_table = s_.graded_ ? s_.klass(elements[k]).pairs : s_.pair_table_;
        std::vector<double> copies(std::size_t(32) * n);
        for (int l = 0; l < n; ++l) {
            const std::size_t r = std::size_t(k) * n + l;
            double *cp = &copies[std::size_t(32) * l];
            std::copy_n(&low[r * 4], 4, cp);
            std::copy_n(&lf[r * 12], 12, cp + 4);
            std::copy_n(&cl[r * 12], 12, cp + 16);
            std::copy_n(&mr[r * 4], 4, cp + 28);
        }
        std::vector<Four> ubar(n), contribution(n), limited;
        long nudged = 0, outward = 0;
        for (int i = 0; i < n; ++i) {
            const double *ci = &copies[std::size_t(32) * i];
            const Four ui = Quad(ci), qi = Quad(ci + 28);
            Four weighted = Four::Zero(), fi = Four::Zero();
            const auto *pairs = &pair_table[std::size_t(i) * (n - 1)];
            for (int kk = 0; kk < n - 1; ++kk) {
                const auto &pe = pairs[kk];
                const int j = pe.j;
                const double dij = pe.d;
                const double *cj = &copies[std::size_t(32) * j];
                const Four uj = Quad(cj);
                V3 edge = V3::Zero();
                Four flux_difference = Four::Zero(), closure_term = Four::Zero();
                for (int d = 0; d < dimension; ++d) {
                    const double cij = pe.c[d], cji = pe.ct[d];
                    edge[d] = cij;
                    flux_difference += cij * (Quad(cj + 4 + 4 * d) - Quad(ci + 4 + 4 * d));
                    closure_term += cji * Quad(ci + 16 + 4 * d) - cij * Quad(cj + 16 + 4 * d);
                }
                if (dij > 0) {
                    Four pair_bar = .5 * (ui + uj) - flux_difference / (2 * dij);
                    if (!exact::cone(pair_bar)) {
                        if (options.outward_bar_rounding) {
                            int raised = 0;
                            pair_bar = stable_base_bar_outward(ui, uj, edge, dij, &raised);
                            outward += raised > 0;
                        } else
                            pair_bar = stable_base_bar(ui, uj, edge, dij);
                    }
                    weighted += dij * pair_bar;
                    fi += dij * (ui - uj);
                }
                fi += pe.m * (qi - Quad(cj + 28)) + closure_term;
            }
            if (options.stabilize)
                fi -= Quad(&stab[std::size_t(k) * 4 * n + 4 * i]);
            Four bar = weighted / total_viscosity[i];
            if (nudge_into_cone(bar))
                ++nudged;
            ubar[i] = bar;
            contribution[i] = fi;
        }
        Four sum = Four::Zero();
        for (const auto &f : contribution)
            sum += f;
        const double raw_sum = sum.cwiseAbs().maxCoeff();
        limited = contribution;
        double factor = 1;
        if (options.limit) {
            for (int i = 0; i < n; ++i) {
                const Four g = limited[i] / (2 * total_viscosity[i]);
                if (exact::cone(ubar[i] + g))
                    continue;
                factor = std::min(factor, options.element_theta * segment(ubar[i], g));
            }
            for (auto &f : limited)
                f *= factor;
        }
        Four after = Four::Zero();
        for (const auto &f : limited)
            after += f;
        for (int i = 0; i < n; ++i) {
            const Four ui = Quad(&copies[std::size_t(32) * i]);
            const Four lo = 2 * total_viscosity[i] * (ubar[i] - ui);
            const Four o4 = lo + limited[i];
            for (int r = 0; r < 4; ++r)
                out[std::size_t(k) * 4 * n + 4 * i + r] = o4[r];
        }
        stats.min_limiter = std::min(stats.min_limiter, factor);
        stats.limited_elements += factor < 1;
        stats.max_raw_zero_sum = std::max(stats.max_raw_zero_sum, raw_sum);
        stats.max_limited_zero_sum = std::max(stats.max_limited_zero_sum, after.cwiseAbs().maxCoeff());
        stats.nudged_states += nudged;
        stats.outward_bars += outward;
    }
    gpu::scatter_device(ctx_, b.element_limited, 4 * n, d_elements, count, out.data(), false, s_.cells);
    limiter_fallbacks += count;
    t_fallback += seconds() - start;
}
// Boundary corrections of limited_low_load (load_b_ in boundary-slot order, closure_h_ etc. on the boundary rows).
void ElementWenoGpu::boundary_correction(ElementWenoStats &stats) {
    const ElementWenoOptions &options = s_.options;
    const int dimension = s_.dimension;
    const auto &sorted = s_.boundary_nodes_sorted_;
    const int nb = int(sorted.size());
    std::vector<double> node_theta(nb, 1.), node_sign(nb, 1.), node_mass(nb, 0.);
    std::vector<char> node_signed(nb, 0);
    const Field &low = low_h_, &low_flux = low_flux_h_, &closure = closure_h_;
    auto block = [](const Field &f, int row, int d) {
        return Four(f(row, 4 * d), f(row, 4 * d + 1), f(row, 4 * d + 2), f(row, 4 * d + 3));
    };
    parallel_for(nb, [&](int k) {
        const int i = sorted[k];
        if (boundary_area_[i] <= 0)
            return;
        const int slot = s_.boundary_slot_[i];
        Four correction = project(boundary_high_.row(slot).transpose()) -
                          Four(boundary_low_(i, 0), boundary_low_(i, 1), boundary_low_(i, 2), boundary_low_(i, 3));
        if (split_) // split_breaks: the domain part only (the broken-line rows go to the second step)
            correction -= Four(brk_high4_.row(i).transpose());
        for (int d = 0; d < dimension; ++d)
            correction -= flux_area_(i, d) * block(closure, i, d);
        Four bar(boundary_bar_(i, 0), boundary_bar_(i, 1), boundary_bar_(i, 2), boundary_bar_(i, 3));
        bar /= boundary_area_[i];
        nudge_into_cone(bar);
        double theta = 1.;
        if (options.limit) {
            const Four g = correction / boundary_area_[i];
            if (!exact::cone(bar + g))
                theta = options.element_theta * segment(bar, g);
            if (options.vacuum_sign_limit && s_.vacuum_limit_node_[i]) { // C2: domain boundary nodes only
                double lo_rate = boundary_area_[i] * (bar[0] - low(i, 0));
                for (int d = 0; d < dimension; ++d)
                    lo_rate -= flux_area_(i, d) * low_flux(i, 4 * d);
                const double allowance = std::max(0., boundary_inflow_[i] - lo_rate);
                if (correction[0] > allowance) {
                    double factor = allowance / correction[0];
                    while (factor > 0 && factor * correction[0] > allowance)
                        factor = std::nextafter(factor, 0.);
                    theta = std::min(theta, factor);
                    node_sign[k] = factor;
                    node_signed[k] = 1;
                }
            }
        }
        node_theta[k] = theta;
        const Four ui(low(i, 0), low(i, 1), low(i, 2), low(i, 3));
        const Four base = boundary_area_[i] * (bar - ui);
        load_b_.row(slot) += (base + theta * correction).transpose();
        double m = base[0] + theta * correction[0];
        // (split_breaks: a pure broken-line node has boundary_area_ = 0 and is skipped above.)
        for (int d = 0; d < dimension; ++d)
            m -= flux_area_(i, d) * low_flux(i, 4 * d);
        node_mass[k] = m;
    });
    double min_boundary = 1, min_sign = 1, boundary_mass_rate = 0;
    long limited_boundary = 0, sign_limited = 0;
    for (int k = 0; k < nb; ++k) {
        if (boundary_area_[sorted[k]] <= 0)
            continue;
        boundary_mass_rate += node_mass[k];
        if (options.limit) {
            min_boundary = std::min(min_boundary, node_theta[k]);
            limited_boundary += node_theta[k] < 1;
            if (node_signed[k]) {
                min_sign = std::min(min_sign, node_sign[k]);
                ++sign_limited;
            }
        }
    }
    stats.min_boundary_limiter = std::min(stats.min_boundary_limiter, min_boundary);
    stats.last_boundary_mass_rate = boundary_mass_rate;
    stats.sign_limited_nodes += sign_limited;
    stats.min_sign_factor = std::min(stats.min_sign_factor, min_sign);
    stats.limited_boundary_nodes += limited_boundary;
}
void ElementWenoGpu::stage(double time, const double *x, double *out, double *u, double old, double fresh, double tau,
                           double weight, double dt, ElementWenoStats &stats, double next_time) {
    const int buffer = stage_begin(time, x, out, next_time);
    stage_end(buffer, time, x, out, u, old, fresh, tau, weight, dt, stats);
}
// First half of a stage: device volume work up to the boundary rows on the host (a_h_, low_h_, low_flux_h_,
// closure_h_ valid on the boundary nodes). Multi-block drivers call it for all blocks before any stage_end, so the
// interface traces of every block are on the host when the face terms are formed.
int ElementWenoGpu::stage_begin(double time, const double *x, double *out, double next_time) {
    const int q = s_.moments;
    const gpu::Buffers b = gpu::buffers(ctx_);
    const auto &bn = s_.boundary_nodes_;
    double *fx = pin_fetch_, *fl = fx + std::size_t(nb_) * q, *ff = fl + std::size_t(nb_) * 4,
           *fc = ff + std::size_t(nb_) * 12;
    auto unpack = [&]() {
        parallel_for(nb_, [&](int k) {
            if (k < nf_)
                std::copy_n(fx + std::size_t(k) * q, q, &a_h_.coeffRef(full_nodes_[k], 0));
            std::copy_n(fl + std::size_t(k) * 4, 4, &low_h_.coeffRef(bn[k], 0));
            std::copy_n(ff + std::size_t(k) * 12, 12, &low_flux_h_.coeffRef(bn[k], 0));
            std::copy_n(fc + std::size_t(k) * 12, 12, &closure_h_.coeffRef(bn[k], 0));
        });
    };
    auto fetch = [&]() {
        gpu::gather_async(ctx_, x, q, d_fnodes_, nf_, fx, 0);
        gpu::gather_async(ctx_, b.low, 4, d_bnodes_, nb_, fl, 1, s_.nodes);
        gpu::gather_async(ctx_, b.low_flux, 12, d_bnodes_, nb_, ff, 2, s_.nodes);
        gpu::gather_async(ctx_, b.closure, 12, d_bnodes_, nb_, fc, 3, s_.nodes);
    };
    // Device: closure, boundary rows to the host, counter snapshot (event 0); then the volume work that needs x only.
    if (!closure_ready_)
        gpu::closure(ctx_, x);
    fetch();
    gpu::snapshot_counters(ctx_);
    gpu::record(ctx_, 0);
    if (device_ghost_) // concurrent with the volume kernels below (aux stream, after the closure)
        gpu::ghost(ctx_, x, pin_ghost_m_, pin_ghost_flags_);
    else if (trace_ghost_) // ghost traces of the stage input, same overlap (aux stream)
        gpu::ghost_traces(ctx_, x, pin_traces_);
    gpu::weno_gamma(ctx_, x);
    gpu::gradients(ctx_, x);
    gpu::stabilization(ctx_, x, out, true, -1);
    // While the device works: the inflow data of this stage (normally prepared already) and of the next one.
    const int buffer = select_face(time);
    if (!std::isnan(next_time)) {
        const double t1 = seconds();
        prepare_face_data(next_time);
        t_face_data += seconds() - t1;
    }
    {
        const double t0 = seconds();
        gpu::wait(ctx_, 0);
        t_sync += seconds() - t0;
    }
    if (gpu::snapshot(ctx_).closure_fallbacks) { // rare: near-beam closures on the host, then fetch again
        gpu::synchronize();
        closure_fallback(x);
        fetch();
        gpu::synchronize();
    }
    unpack();
    return buffer;
}
void ElementWenoGpu::stage_end(int buffer, double time, const double *x, double *out, double *u, double old,
                               double fresh, double tau, double weight, double dt, ElementWenoStats &stats) {
    const int q = s_.moments;
    const gpu::Buffers b = gpu::buffers(ctx_);
    // Phase A: PN boundary load rows, added before the mass solve (device, or host with M1PN_GPU_BOUNDARY_HIGH=0).
    if (interfaces_) {
        // Multi-block (interface sides): the whole face work on the host by ElementWeno::add_boundary (phase A and B
        // in one pass; the neighbour traces come from the interface callback).
        const double t0 = seconds();
        const auto &bn = s_.boundary_nodes_;
        boundary_high_.setZero();
        parallel_for(nb_, [&](int k) {
            const int i = bn[k];
            boundary_low_.row(i).setZero(), flux_area_.row(i).setZero(), boundary_bar_.row(i).setZero();
            boundary_area_[i] = 0, boundary_inflow_[i] = 0;
        });
        s_.add_boundary(time, a_h_, low_h_, low_flux_h_, boundary_high_, boundary_low_, flux_area_, boundary_area_,
                        boundary_bar_, &boundary_inflow_, &stats.projection);
        std::copy_n(boundary_high_.data(), std::size_t(nb_) * q, pin_high_);
        double *rows = gpu::device_doubles(ctx_, 4, std::size_t(std::max(1, nb_)) * q);
        gpu::upload_async(rows, pin_high_, sizeof(double) * std::size_t(nb_) * q);
        gpu::add_rows(ctx_, out, d_bnodes_, rows, nb_);
        t_boundary += seconds() - t0;
    } else if (device_high_) {
        const double t0 = seconds();
        bool datum = false;
        for (int side = 0; side < 6; ++side)
            datum |= s_.boundaries_[side] && !datum_zero_[buffer][side];
        if (datum) { // row-major copy of the datum rows (time-dependent inflow only)
            const Field &DB = DB_[buffer];
            const int rows = int(DB.rows());
            parallel_for(rows, [&](int r) {
                for (int c = 0; c < q; ++c)
                    pin_db_[std::size_t(r) * q + c] = DB(r, c);
            });
        } else if (nodal_interfaces_ && !nodal_db_zeroed_) {
            std::fill_n(pin_db_, std::size_t(s_.side_row_start_[6]) * q, 0.);
            nodal_db_zeroed_ = true;
        }
        if (nodal_interfaces_) {
            // Conforming interfaces: datum rows (M_face u_partner) .5 (A_n - I)^T, so that the device face load is
            // .5 (A_n - I) M_face (u - u_partner) as on a broken line (partner rows from the neighbour's host state).
            parallel_for(int(s_.faces_.size()), [&](int index) {
                const auto &f = s_.faces_[index];
                const int side_id = 2 * f.direction + f.side;
                if (!s_.nodal_interface(side_id) || (s_.broken_ && s_.face_break_[index]))
                    return; // broken lines: partner term on the device (phase A partner rows), datum stays zero
                const int nf = int(s_.rules_[side_id].local.size()), r0 = s_.face_row_[index];
                thread_local Field rows, db;
                rows.resize(nf, q);
                s_.nodal_partner_rows(index, a_h_, rows);
                db = rows * s_.interface_sparse_t_[side_id];
                for (int n = 0; n < nf; ++n)
                    for (int c = 0; c < q; ++c)
                        pin_db_[std::size_t(r0 + n) * q + c] = db(n, c);
            });
            datum = true;
        }
        gpu::boundary_high(ctx_, x, out, datum ? pin_db_ : nullptr, pin_low4_, split_ ? pin_low4b_ : nullptr);
        t_boundary_high += seconds() - t0;
    } else {
        const double t0 = seconds();
        boundary_high(buffer);
        std::copy_n(boundary_high_.data(), std::size_t(nb_) * q, pin_high_);
        double *rows = gpu::device_doubles(ctx_, 4, std::size_t(std::max(1, nb_)) * q);
        gpu::upload_async(rows, pin_high_, sizeof(double) * std::size_t(nb_) * q);
        gpu::add_rows(ctx_, out, d_bnodes_, rows, nb_);
        t_boundary_high += seconds() - t0;
    }
    const bool pn_combined = gpu::mass_solve_combine(ctx_, out, x, u, old, fresh, tau);
    gpu::flux_moments(ctx_, out); // the advective term of the PN rows is added in finish_stage
    gpu::limiter(ctx_);
    gpu::gather_load(ctx_);
    gpu::snapshot_counters(ctx_);
    gpu::gather_async(ctx_, b.load, 4, d_bnodes_, nb_, pin_load_, 5);
    gpu::record(ctx_, 1);
    if (!pn_combined)
        gpu::combine_pn(ctx_, out, x, u, old, fresh, tau); // PN rows: independent of the boundary corrections
    // Diagnostics (M1PN_GPU_DIAG_SERIAL=1, timing only, same results): wait for the device before phase B, so that
    // t_diag_device is the device work that phase B normally overlaps.
    static const bool diag_serial = [] {
        const char *e = std::getenv("M1PN_GPU_DIAG_SERIAL");
        return e && std::atoi(e) != 0;
    }();
    if (diag_serial) {
        const double t0 = seconds();
        gpu::synchronize();
        t_diag_device += seconds() - t0;
    }
    // Host phase B (overlaps the device work above): ghost moments and low-order face terms.
    if (!interfaces_) {
        const double t0 = seconds();
        boundary_low(buffer, &stats.projection);
        t_boundary += seconds() - t0;
    }
    {
        const double t0 = seconds();
        gpu::wait(ctx_, 1);
        t_wait += seconds() - t0;
    }
    if (device_high_) // l <= 1 columns of the node sums (all that the boundary corrections read)
        parallel_for(nb_, [&](int k) {
            for (int c = 0; c < 4; ++c)
                boundary_high_(k, c) = pin_low4_[std::size_t(k) * 4 + c];
            if (split_) { // projected (rho, J) of the broken-line rows' l <= 1 coefficient sums
                const double *b = pin_low4b_ + std::size_t(k) * 4;
                brk_high4_.row(s_.boundary_nodes_[k]) = project(Vec(Eigen::Map<const Vec>(b, 4))).transpose();
            }
        });
    const gpu::Counters c = gpu::snapshot(ctx_);
    if (c.error)
        check_error();
    if (c.limiter_fallbacks) { // rare: stable base bars on the host, then gather again
        gpu::synchronize();
        limiter_fallback(stats);
        gpu::gather_load(ctx_);
        gpu::gather_async(ctx_, b.load, 4, d_bnodes_, nb_, pin_load_, 5);
        gpu::synchronize();
    }
    {
        const double t0 = seconds();
        std::copy_n(pin_load_, std::size_t(nb_) * 4, load_b_.data());
        boundary_correction(stats);
        if (split_) { // split_breaks: the broken-line face terms as the second convex step (ElementWeno::split_second_step)
            const auto &nodes = s_.split_nodes_;
            const int K = int(nodes.size());
            std::vector<Four> u(K), rate(K);
            const int dimension = s_.dimension;
            parallel_for(K, [&](int k) {
                const int i = nodes[k], slot = s_.boundary_slot_[i];
                Four t = Four(brk_high4_.row(i).transpose());
                for (int d = 0; d < dimension; ++d)
                    t -= brk_fa_(i, d) * Four(closure_h_(i, 4 * d), closure_h_(i, 4 * d + 1), closure_h_(i, 4 * d + 2),
                                              closure_h_(i, 4 * d + 3));
                if (s_.options.m1_target)
                    t = Four(brk_low_.row(i).transpose());
                split_term_h_.row(i) = t.transpose();
                u[k] = Four(low_h_(i, 0), low_h_(i, 1), low_h_(i, 2), low_h_(i, 3));
                rate[k] = Four(load_b_.row(slot).transpose()) / s_.lumped[i];
            });
            s_.split_second_step(u, rate, split_term_h_, brk_area_, tau, &stats);
            parallel_for(K, [&](int k) {
                const int i = nodes[k];
                load_b_.row(s_.boundary_slot_[i]) = (s_.lumped[i] * rate[k]).transpose();
            });
        }
        std::copy_n(load_b_.data(), std::size_t(nb_) * 4, pin_load_up_);
        gpu::scatter_async(ctx_, b.load, 4, d_bnodes_, nb_, pin_load_up_, false, 6);
        t_correction += seconds() - t0;
    }
    stats.boundary_outflow -= dt * weight * stats.last_boundary_mass_rate;
    gpu::reset_fallbacks(ctx_); // the fused closure below records the fallbacks of the next stage
    gpu::finish_stage(ctx_, out, x, u, old, fresh, tau, true, true);
    closure_ready_ = true;
}
void ElementWenoGpu::begin(const Field &a) {
    require(a.rows() == s_.nodes && a.cols() == s_.moments, "GPU path: invalid state");
    const gpu::Buffers b = gpu::buffers(ctx_);
    u_ = b.a, b1_ = b.x, b2_ = b.r;
    gpu::upload(u_, a.data(), sizeof(double) * std::size_t(s_.nodes) * s_.moments);
    gpu::reset_counters(ctx_);
    closure_ready_ = false;
}
// Exact local reaction over tau on the device state (FCS: radial table of rho_u at source_time from the host).
void ElementWenoGpu::react(double tau, double source_time, ElementWenoStats &stats) {
    const double start = seconds();
    if (pulse_) {
        const uncollided::GaussianPulse &p = *pulse_;
        const double dr = p.width / 8, reach = std::max(source_time, 0.) + 10 * p.width;
        const int n = int(std::ceil(reach / dr)) + 3;
        table_.assign(n + 1, 0.);
        parallel_for(n, [&](int k) { table_[k + 1] = p.line(k * dr, source_time); });
        table_[0] = table_[2];
        gpu::react(ctx_, u_, tau, table_.data(), n, dr, p.centre[0], p.centre[1]);
    } else if (lattice_ || lattice3_)
        gpu::react_lattice(ctx_, u_, tau, source_time);
    else
        gpu::react(ctx_, u_, tau, nullptr, 0, 0, 0, 0);
    closure_ready_ = false; // the fused closure of the last stage belongs to the state before the reaction
    stats.t_reaction += seconds() - start;
}
void ElementWenoGpu::step(double t, double dt, double next, ElementWenoStats &stats) {
    const bool four = s_.options.ssprk43, reacts = s_.reacts();
    if (reacts)
        react(.5 * dt, t + .25 * dt, stats);
    double *u = u_, *b1 = b1_, *b2 = b2_;
    if (s_.options.ssp_n >= 3) { // SSPRK(n^2,3) (experiment): three buffers, ym kept in one of them
        static thread_local ElementWeno::SspPlan plan;
        if (plan.n != s_.options.ssp_n)
            plan = ElementWeno::ssp_plan(s_.options.ssp_n);
        double *buf[3] = {u_, b1_, b2_};
        int c = 0, y = -1;
        const double tau = dt * plan.tau_fraction;
        for (int k = 0; k < plan.stages; ++k) {
            if (k == plan.saved_after)
                y = c;
            int o = 0;
            while (o == c || o == y)
                ++o;
            const auto &st = plan.stage[k];
            const double t_next = k + 1 < plan.stages ? t + plan.stage[k + 1].time * dt : next;
            if (st.combination) {
                stage(t + st.time * dt, buf[c], buf[o], buf[y], plan.wa, plan.wb, tau, st.weight, dt, stats, t_next);
                y = -1;
            } else
                stage(t + st.time * dt, buf[c], buf[o], buf[c], 0, 1, tau, st.weight, dt, stats, t_next);
            c = o;
        }
        u_ = buf[c], b1_ = buf[(c + 1) % 3], b2_ = buf[(c + 2) % 3];
        if (reacts)
            react(.5 * dt, t + .75 * dt, stats);
        return;
    }
    if (!four) {
        stage(t, u, b1, u, 0, 1, dt, 1. / 6, dt, stats, t + dt);
        stage(t + dt, b1, b2, u, .75, .25, dt, 1. / 6, dt, stats, t + .5 * dt);
        stage(t + .5 * dt, b2, b1, u, 1. / 3, 2. / 3, dt, 2. / 3, dt, stats, next);
        std::swap(u, b1);
    } else {
        const double half = .5 * dt;
        stage(t, u, b1, u, 0, 1, half, 1. / 6, dt, stats, t + half);
        stage(t + half, b1, b2, u, 0, 1, half, 1. / 6, dt, stats, t + dt);
        stage(t + dt, b2, b1, u, 2. / 3, 1. / 3, half, 1. / 6, dt, stats, t + half);
        stage(t + half, b1, b2, u, 0, 1, half, .5, dt, stats, next);
        std::swap(u, b2);
    }
    u_ = u, b1_ = b1, b2_ = b2;
    if (reacts)
        react(.5 * dt, t + .75 * dt, stats);
}
void ElementWenoGpu::lockstep_begin(int s, double time, double dt) {
    require(s_.options.ssprk43 && s_.options.ssp_n < 3 && s >= 0 && s < 4, "lockstep: SSPRK(4,3) stages 0..3");
    const double half = .5 * dt, t[4] = {time, time + half, time + dt, time + half};
    const double *x[4] = {u_, b1_, b2_, b1_};
    double *out[4] = {b1_, b2_, b1_, b2_};
    lockstep_buffer_ = stage_begin(t[s], x[s], out[s], std::numeric_limits<double>::quiet_NaN());
}
void ElementWenoGpu::lockstep_end(int s, double time, double dt, ElementWenoStats &stats) {
    const double half = .5 * dt, t[4] = {time, time + half, time + dt, time + half};
    const double old[4] = {0, 0, 2. / 3, 0}, fresh[4] = {1, 1, 1. / 3, 1}, weight[4] = {1. / 6, 1. / 6, 1. / 6, .5};
    const double *x[4] = {u_, b1_, b2_, b1_};
    double *out[4] = {b1_, b2_, b1_, b2_};
    stage_end(lockstep_buffer_, t[s], x[s], out[s], u_, old[s], fresh[s], half, weight[s], dt, stats);
    if (s == 3)
        std::swap(u_, b2_);
}
Field ElementWenoGpu::state() {
    gpu::synchronize();
    Field out(s_.nodes, s_.moments);
    gpu::download(out.data(), u_, sizeof(double) * std::size_t(s_.nodes) * s_.moments);
    return out;
}
Field ElementWenoGpu::end(ElementWenoStats &stats) {
    const gpu::Counters c = gpu::counters(ctx_);
    check_error();
    stats.limited_elements += c.limited_elements - merged_.limited_elements;
    stats.min_limiter = std::min(stats.min_limiter, c.min_limiter);
    stats.max_raw_zero_sum = std::max(stats.max_raw_zero_sum, c.max_raw_zero_sum);
    stats.max_limited_zero_sum = std::max(stats.max_limited_zero_sum, c.max_limited_zero_sum);
    stats.nudged_states += (c.nudged_bars + c.nudged_rows) - (merged_.nudged_bars + merged_.nudged_rows);
    stats.nudge_mass += c.nudge_mass - merged_.nudge_mass;
    stats.min_gamma = std::min(stats.min_gamma, c.min_gamma);
    merged_ = c;
    if (device_ghost_) {
        // Device projections (flagged pairs are counted by the host when it solves them).
        const gpu::GhostStats g = gpu::ghost_stats(ctx_);
        PositivityProjectionStats p;
        p.calls = long(g.calls) - (ghost_flagged - ghost_flagged_merged);
        p.identities = g.identities, p.vacuum = g.vacuum, p.failures = g.failures, p.scaled = g.scaled;
        p.capped = g.capped, p.iterations = g.iterations, p.backtracks = g.backtracks;
        p.max_iterations = g.max_iterations, p.max_residual = g.max_residual, p.min_active_fraction = g.min_active_fraction;
        PositivityProjectionStats delta = p; // stats are cumulative on the device: merge only the increment
        delta.calls -= ghost_merged_.calls, delta.identities -= ghost_merged_.identities;
        delta.vacuum -= ghost_merged_.vacuum, delta.failures -= ghost_merged_.failures;
        delta.scaled -= ghost_merged_.scaled, delta.capped -= ghost_merged_.capped;
        delta.iterations -= ghost_merged_.iterations, delta.backtracks -= ghost_merged_.backtracks;
        stats.projection.merge(delta);
        ghost_merged_ = p;
        ghost_flagged_merged = ghost_flagged;
    }
    if (s_.reacts()) {
        const gpu::Ledger l = gpu::ledger(ctx_);
        stats.absorbed += l.absorbed - ledger_absorbed_, stats.injected += l.injected - ledger_injected_;
        ledger_absorbed_ = l.absorbed, ledger_injected_ = l.injected;
    }
    return state();
}
Field ElementWenoGpu::run(double time, double dt, int steps, const Field &a, ElementWenoStats *stats_in) {
    ElementWenoStats scratch;
    ElementWenoStats &stats = stats_in ? *stats_in : scratch;
    begin(a);
    double t = time;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int n = 0; n < steps; ++n) {
        double tn = t; // the next step starts at t + dt (accumulated as the driver does)
        tn += dt;
        step(t, dt, n + 1 < steps ? tn : nan, stats);
        t = tn;
    }
    return end(stats);
}
double ElementWenoGpu::spectral_radius() {
    const int q = s_.moments;
    const gpu::Buffers b = gpu::buffers(ctx_);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> uniform(-1, 1);
    double radius = 0;
    for (double gamma_value : {0., 1.}) {
        Field x(s_.nodes, q);
        for (int i = 0; i < s_.nodes; ++i)
            for (int c = 0; c < q; ++c)
                x(i, c) = uniform(rng);
        x /= x.norm();
        gpu::upload(b.x, x.data(), sizeof(double) * std::size_t(s_.nodes) * q);
        const int iterations = 60;
        for (int k = 0; k < iterations; ++k) {
            gpu::gradients(ctx_, b.x);
            gpu::stabilization(ctx_, b.x, b.r, false, gamma_value);
            gpu::mass_solve(ctx_, b.r);
            gpu::flux_rows(ctx_, b.r, false);
            const double growth = gpu::norm(ctx_, b.r);
            require(std::isfinite(growth) && growth > 0, "High-order spectral estimate failed (GPU)");
            if (k >= iterations - 15)
                radius = std::max(radius, growth);
            gpu::divide(ctx_, b.x, b.r, growth);
        }
    }
    gpu::synchronize();
    return radius;
}
} // namespace m1pn
