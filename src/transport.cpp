#include "transport.hpp"
#include "cuda_path.hpp"
#include "base_bar.hpp"
#include "lazy_m1.hpp"
#include <algorithm>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <atomic>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace m1pn {
namespace {
// Cached CSR of a compressed row-major angular matrix, in InnerIterator order.
struct MatrixFingerprint {
    const double *values = nullptr;
    int nnz = 0;
    int rows = 0;
    double first = 0;
    double last = 0;
};
MatrixFingerprint fingerprint(const Sparse &matrix) {
    MatrixFingerprint id;
    id.values = matrix.valuePtr();
    id.nnz = int(matrix.nonZeros());
    id.rows = int(matrix.rows());
    if (id.nnz > 0 && id.values) {
        id.first = id.values[0];
        id.last = id.values[id.nnz - 1];
    }
    return id;
}
bool same_matrix(const MatrixFingerprint &id, const Sparse &matrix) {
    const MatrixFingerprint now = fingerprint(matrix);
    return id.values == now.values && id.nnz == now.nnz && id.rows == now.rows &&
           id.first == now.first && id.last == now.last;
}
struct AngularPattern {
    MatrixFingerprint id;
    int q = 0;
    std::vector<int> row_ptr;
    std::vector<int> col;
    std::vector<double> value;
};
void fill_angular_pattern(AngularPattern &pattern, const Sparse &A) {
    pattern.id = fingerprint(A);
    pattern.q = int(A.rows());
    pattern.row_ptr.clear();
    pattern.col.clear();
    pattern.value.clear();
    pattern.row_ptr.reserve(pattern.q + 1);
    for (int i = 0; i < pattern.q; ++i) {
        pattern.row_ptr.push_back(int(pattern.col.size()));
        for (Sparse::InnerIterator it(A, i); it; ++it) {
            pattern.col.push_back(int(it.index()));
            pattern.value.push_back(it.value());
        }
    }
    pattern.row_ptr.push_back(int(pattern.col.size()));
}
const AngularPattern &angular_pattern(const Sparse &A) {
    static std::mutex guard;
    static std::deque<std::pair<const Sparse *, AngularPattern>> cache;
    for (auto &item : cache)
        if (item.first == &A && same_matrix(item.second.id, A))
            return item.second;
    std::lock_guard<std::mutex> lock(guard);
    for (auto &item : cache)
        if (item.first == &A) {
            if (!same_matrix(item.second.id, A))
                fill_angular_pattern(item.second, A);
            return item.second;
        }
    AngularPattern pattern;
    fill_angular_pattern(pattern, A);
    cache.emplace_back(&A, std::move(pattern));
    return cache.back().second;
}
std::array<const AngularPattern *, 3> angular_patterns(const Angular &pn) {
    return {&angular_pattern(pn.A[0]), &angular_pattern(pn.A[1]), &angular_pattern(pn.A[2])};
}
// Eigen rewrites n*(A*u) as (n*A)*u, so each stored entry is scaled before the product.
void pn_apply(const std::array<const AngularPattern *, 3> &pattern, const V3 &direction,
              const double *state, double *out) {
    const int q = pattern[0]->q;
    std::fill_n(out, q, 0.);
    for (int d = 0; d < 3; ++d) {
        if (direction[d] == 0.)
            continue;
        const AngularPattern &rows = *pattern[d];
        const double scale = direction[d];
        for (int i = 0; i < q; ++i) {
            double tmp = 0.;
            for (int k = rows.row_ptr[i]; k < rows.row_ptr[i + 1]; ++k)
                tmp += (scale * rows.value[k]) * state[rows.col[k]];
            out[i] += tmp;
        }
    }
}
Vec fast_apply(const Angular &pn, const V3 &direction, const Vec &state) {
    Vec out(state.size());
    pn_apply(angular_patterns(pn), direction, state.data(), out.data());
    return out;
}
// Row dots of A^T against each nodal moment row. Matches `u * A.transpose()` bitwise.
Field parallel_angular_product(const Sparse &A, const Field &u) {
    const AngularPattern &pattern = angular_pattern(A);
    Field flux(u.rows(), pattern.q);
    parallel_for(int(u.rows()), [&](int node) {
        const double *in = &u.coeff(node, 0);
        double *out = &flux.coeffRef(node, 0);
        for (int i = 0; i < pattern.q; ++i) {
            double tmp = 0.;
            for (int k = pattern.row_ptr[i]; k < pattern.row_ptr[i + 1]; ++k)
                tmp += pattern.value[k] * in[pattern.col[k]];
            out[i] = tmp;
        }
    });
    return flux;
}
// One read of each nodal row, three directional images. Each image matches
// `parallel_angular_product` for that direction.
void fill_angular_products(const Angular &pn, const Field &u, std::array<Field, 3> &flux) {
    if (cuda_ready()) {
        cuda_fill_angular_products(pn, u, flux);
        return;
    }
    const auto patterns = angular_patterns(pn);
    const int n = int(u.rows());
    const int q = patterns[0]->q;
    for (int d = 0; d < 3; ++d)
        if (flux[d].rows() != n || flux[d].cols() != q)
            flux[d].resize(n, q);
    parallel_for(n, [&](int node) {
        const double *in = &u.coeff(node, 0);
        for (int d = 0; d < 3; ++d) {
            const AngularPattern &pattern = *patterns[d];
            double *out = &flux[d].coeffRef(node, 0);
            for (int i = 0; i < q; ++i) {
                double tmp = 0.;
                for (int k = pattern.row_ptr[i]; k < pattern.row_ptr[i + 1]; ++k)
                    tmp += pattern.value[k] * in[pattern.col[k]];
                out[i] = tmp;
            }
        }
    });
}
// Nonzero order of Eigen's Transpose iterator. Adding through this order into an
// existing row-major field matches `destination += matrix.transpose() * flux`.
struct ScatterPattern {
    const Sparse *source = nullptr;
    MatrixFingerprint id;
    std::vector<int> outer_ptr;
    std::vector<int> index;
    std::vector<double> value;
};
const ScatterPattern &scatter_pattern(const Sparse &matrix) {
    static std::mutex guard;
    static std::deque<ScatterPattern> cache;
    for (auto &item : cache)
        if (item.source == &matrix && same_matrix(item.id, matrix))
            return item;
    std::lock_guard<std::mutex> lock(guard);
    for (auto &item : cache)
        if (item.source == &matrix) {
            if (!same_matrix(item.id, matrix)) {
                item.outer_ptr.clear();
                item.index.clear();
                item.value.clear();
                item.id = fingerprint(matrix);
                item.source = &matrix;
                using Transposed = Eigen::Transpose<const Sparse>;
                const Transposed transposed = matrix.transpose();
                Eigen::internal::evaluator<Transposed> eval(transposed);
                item.outer_ptr.reserve(transposed.outerSize() + 1);
                for (int j = 0; j < transposed.outerSize(); ++j) {
                    item.outer_ptr.push_back(int(item.index.size()));
                    for (typename Eigen::internal::evaluator<Transposed>::InnerIterator it(eval, j); it;
                         ++it) {
                        item.index.push_back(int(it.index()));
                        item.value.push_back(it.value());
                    }
                }
                item.outer_ptr.push_back(int(item.index.size()));
            }
            return item;
        }
    ScatterPattern pattern;
    pattern.source = &matrix;
    pattern.id = fingerprint(matrix);
    using Transposed = Eigen::Transpose<const Sparse>;
    const Transposed transposed = matrix.transpose();
    Eigen::internal::evaluator<Transposed> eval(transposed);
    pattern.outer_ptr.reserve(transposed.outerSize() + 1);
    for (int j = 0; j < transposed.outerSize(); ++j) {
        pattern.outer_ptr.push_back(int(pattern.index.size()));
        for (typename Eigen::internal::evaluator<Transposed>::InnerIterator it(eval, j); it; ++it) {
            pattern.index.push_back(int(it.index()));
            pattern.value.push_back(it.value());
        }
    }
    pattern.outer_ptr.push_back(int(pattern.index.size()));
    cache.push_back(std::move(pattern));
    return cache.back();
}
// M1PN_RELAXED=1 keeps the same products but drops Eigen's summation order.
bool relaxed_association() {
    static const bool enabled = [] {
        const char *value = std::getenv("M1PN_RELAXED");
        return value && std::string(value) == "1";
    }();
    return enabled;
}
double identity_tolerance() {
    return relaxed_association() ? 1e-8 : 5e-11;
}
void directional_from_node(const std::array<Field, 3> &flux, int node, const V3 &direction, double *out,
                           int q) {
    std::fill_n(out, q, 0.);
    for (int d = 0; d < 3; ++d) {
        if (direction[d] == 0.)
            continue;
        const double *src = &flux[d].coeff(node, 0);
        const double scale = direction[d];
        for (int c = 0; c < q; ++c)
            out[c] += scale * src[c];
    }
}
// Byte-exact stand-in for `destination.noalias() += matrix.transpose() * flux`.
// A node-parallel gather was slower on 256²/P11: each moment row is 144 doubles and
// the grid stencil jumps through that field, while this column axpy streams it.
void add_scatter_product(Field &destination, const Sparse &matrix, const Field &flux) {
    const ScatterPattern &pattern = scatter_pattern(matrix);
    const int q = int(flux.cols());
    const int outers = int(pattern.outer_ptr.size()) - 1;
#pragma omp parallel
    {
        const int threads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int c0 = (q * tid) / threads;
        const int c1 = (q * (tid + 1)) / threads;
        const int width = c1 - c0;
        for (int j = 0; j < outers; ++j) {
            const double *src = &flux.coeff(j, c0);
            for (int k = pattern.outer_ptr[j]; k < pattern.outer_ptr[j + 1]; ++k) {
                double *dst = &destination.coeffRef(pattern.index[k], c0);
                const double value = pattern.value[k];
                for (int c = 0; c < width; ++c)
                    dst[c] += value * src[c];
            }
        }
    }
}
// Column axpy of all three transport matrices while each source row is hot.
// Interleaving the directions changes the addition order by about 1 ulp.
// A node-parallel gather of full moment rows was slower at P11 width.
void add_directional_scatter(Field &destination, const std::array<Sparse, 3> &matrices,
                             const std::array<Field, 3> &flux) {
    if (cuda_ready()) {
        cuda_add_directional_scatter(destination, matrices, flux);
        return;
    }
    const ScatterPattern *pattern[3] = {&scatter_pattern(matrices[0]), &scatter_pattern(matrices[1]),
                                        &scatter_pattern(matrices[2])};
    const int q = int(flux[0].cols());
    const int outers = int(pattern[0]->outer_ptr.size()) - 1;
    for (int d = 1; d < 3; ++d) {
        if (int(pattern[d]->outer_ptr.size()) - 1 != outers) {
            for (int k = 0; k < 3; ++k)
                add_scatter_product(destination, matrices[k], flux[k]);
            return;
        }
    }
#pragma omp parallel
    {
        const int threads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int c0 = (q * tid) / threads;
        const int c1 = (q * (tid + 1)) / threads;
        const int width = c1 - c0;
        for (int j = 0; j < outers && width > 0; ++j) {
            for (int d = 0; d < 3; ++d) {
                const ScatterPattern &pat = *pattern[d];
                const double *src = &flux[d].coeff(j, c0);
                for (int k = pat.outer_ptr[j]; k < pat.outer_ptr[j + 1]; ++k) {
                    double *dst = &destination.coeffRef(pat.index[k], c0);
                    const double value = pat.value[k];
                    for (int c = 0; c < width; ++c)
                        dst[c] += value * src[c];
                }
            }
        }
    }
}
} // namespace
Transport::Transport(const Angular &a, Spatial &f, const Numerics &n, Inflow g, Runtime r)
    : pn(a), fe(f), options(n), runtime(r), canonical(pn), inflow(std::move(g)) {
    require(n.gamma >= 0 && std::isfinite(n.gamma),
            "Stabilization gamma must be finite and nonnegative");
    require(n.mass_mode == "consistent" || n.mass_mode == "lumped" || n.mass_mode == "lumped_afc",
            "Unknown mass mode");
    require(n.mass_corrections >= 1, "Mass corrections must be at least one");
    require(n.bh_pair_mode == "element" || n.bh_pair_mode == "canonical" || n.bh_pair_mode == "factorized",
            "Unknown stabilization pair mode");
    require(n.implicit_solver == "direct" || n.implicit_solver == "defect" || n.implicit_solver == "pcg",
            "Unknown implicit solver mode");
    require(n.pn_boundary_quadrature == "lumped" || n.pn_boundary_quadrature == "consistent",
            "Unknown PN boundary quadrature mode");
    require(n.r2 || n.stabilization != "m1pn_ghost_local_projected_p4",
            "Local projected ghost stabilization requires the R2 transfer");
    const bool pn_mass_correction = n.stabilization == "m1pn_mass_correction" ||
                                    n.stabilization == "m1pn_mass_correction_boundary_consistent";
    require(!pn_mass_correction || (n.r2 && n.mass_mode == "lumped"),
            "PN mass correction requires R2 and lumped mass mode");
    require(n.pn_viscosity_scale >= 1 && std::isfinite(n.pn_viscosity_scale),
            "PN viscosity scale must be finite and at least one");
    if(n.stabilization=="m1pn_mass_correction_boundary_consistent") {
        pn_boundary_map.resize(f.nodes); std::iota(pn_boundary_map.begin(),pn_boundary_map.end(),0);
        if(!f.faces.empty()) {
            std::array<std::vector<double>,3> axes;
            std::array<int,3> count{1,1,1};
            for(int d=0;d<f.dimension;++d) {
                for(int i=0;i<f.nodes;++i)axes[d].push_back(f.coordinates(i,d));
                std::sort(axes[d].begin(),axes[d].end());
                axes[d].erase(std::unique(axes[d].begin(),axes[d].end(),[](double x,double y){return std::abs(x-y)<2e-12;}),axes[d].end());
                count[d]=int(axes[d].size());
                require(count[d]>=4,"Boundary-consistent PN mass correction requires at least three Cartesian cells per axis");
                const double h=axes[d][1]-axes[d][0];
                for(int k=2;k<count[d];++k)require(std::abs((axes[d][k]-axes[d][k-1])-h)<2e-10*std::max(1.,std::abs(h)),
                    "Boundary-consistent PN mass correction requires a uniform Cartesian grid");
            }
            const long long grid_nodes=1LL*count[0]*count[1]*count[2];
            require(grid_nodes==f.nodes,"Boundary-consistent PN mass correction requires a full Cartesian Q1 grid");
            long long tensor_elements=1;
            for(int d=0;d<f.dimension;++d)tensor_elements*=count[d]-1;
            require(f.mesh->GetNE()==tensor_elements,"Boundary-consistent PN mass correction requires a full tensor-product element grid");
            for(int e=0;e<f.mesh->GetNE();++e)
                require(f.mesh->GetElement(e)->GetNVertices()==(1<<f.dimension),
                        "Boundary-consistent PN mass correction requires line, quadrilateral, or hexahedral Q1 elements");
            auto key=[&](const std::array<int,3>&q){return (q[0]*count[1]+q[1])*count[2]+q[2];};
            std::vector<int> node_at(f.nodes,-1); std::vector<std::array<int,3>> multi(f.nodes);
            for(int i=0;i<f.nodes;++i) {
                std::array<int,3> q{0,0,0};
                for(int d=0;d<f.dimension;++d) {
                    auto it=std::lower_bound(axes[d].begin(),axes[d].end(),f.coordinates(i,d));
                    require(it!=axes[d].end()&&std::abs(*it-f.coordinates(i,d))<2e-12,"Non-Cartesian node in boundary-consistent PN grid");
                    q[d]=int(it-axes[d].begin());
                }
                require(node_at[key(q)]<0,"Duplicate Cartesian node in boundary-consistent PN grid");
                node_at[key(q)]=i; multi[i]=q;
            }
            for(int x:node_at)require(x>=0,"Missing Cartesian node in boundary-consistent PN grid");
            std::set<std::array<int,3>> cells_seen;
            for(int e=0;e<f.mesh->GetNE();++e) {
                mfem::Array<int> vertices; f.mesh->GetElementVertices(e,vertices);
                require(vertices.Size()==(1<<f.dimension),"Boundary-consistent PN element is not tensor-product Q1");
                std::array<int,3> lo{count[0],count[1],count[2]},hi{-1,-1,-1};
                for(int k=0;k<vertices.Size();++k)for(int d=0;d<f.dimension;++d) {
                    lo[d]=std::min(lo[d],multi[vertices[k]][d]); hi[d]=std::max(hi[d],multi[vertices[k]][d]);
                }
                require(cells_seen.insert(lo).second,"Duplicate Cartesian cell in boundary-consistent PN grid");
                int seen=0;
                for(int k=0;k<vertices.Size();++k) {
                    int mask=0;
                    for(int d=0;d<f.dimension;++d) {
                        require(hi[d]-lo[d]==1,"Boundary-consistent PN element spans more than one Cartesian cell");
                        mask|=(multi[vertices[k]][d]-lo[d])<<d;
                    }
                    require(!(seen&(1<<mask)),"Boundary-consistent PN element repeats a tensor vertex");
                    seen|=1<<mask;
                }
                require(seen==((1<<(1<<f.dimension))-1),"Boundary-consistent PN element does not cover one Cartesian cell");
            }
            for(int i=0;i<f.nodes;++i)if(f.boundary[i]) {
                auto q=multi[i];
                for(int d=0;d<f.dimension;++d)q[d]=std::clamp(q[d],1,count[d]-2);
                pn_boundary_map[i]=node_at[key(q)];
            }
            Vec accumulated=Vec::Zero(f.nodes);
            for(int i=0;i<f.nodes;++i)accumulated[pn_boundary_map[i]]+=f.mass[i];
            double ratio=1.;
            for(int j=0;j<f.nodes;++j)ratio=std::max(ratio,accumulated[j]/f.mass[j]);
            pn_boundary_map_norm=std::sqrt(ratio);
        }
    }
    if (n.r2 && !pn_mass_correction) stabilization_operator = std::make_unique<SpatialStabilization>(f,n.stabilization,n.stabilization=="gradient-defect"?n.kv_gamma:n.gamma,pn.speed);
    const long double spatial_bytes =
        8192.L * f.nodes + 192.L * std::pow(static_cast<long double>(f.nodes), 4.L / 3);
    const long double state_bytes =
        sizeof(double) * (66.L * f.nodes * pn.q + 8.L * f.pairs.size() * pn.q);
    const long double pair_adjacency_bytes =
        2.L * sizeof(std::size_t) * (static_cast<long double>(f.nodes) + 1.L) +
        2.L * (sizeof(int) + sizeof(signed char)) * f.pairs.size();
    const long double predictor_edge_cache_bytes = pn_mass_correction ?
        static_cast<long double>(f.pairs.size()) *
            (sizeof(std::array<V3,2>) + sizeof(double)) :
        (n.geometry_cache ? static_cast<long double>(f.pairs.size()) * sizeof(std::array<V3,2>) : 0.L);
    n.budget.check(spatial_bytes + state_bytes + pair_adjacency_bytes + predictor_edge_cache_bytes +
                       32.L * pn.q * pn.q * sizeof(double),
                   "Transport and RK state/transfer buffers");
    if (n.geometry_cache || pn_mass_correction) {
        pair_geometry.resize(f.pairs.size());
        if (pn_mass_correction) pn_low_order_edge_viscosity.resize(f.pairs.size());
        for (std::size_t e=0;e<f.pairs.size();++e) for (int d=0;d<3;++d) {
            pair_geometry[e][0][d]=f.C[d].coeff(f.pairs[e].i,f.pairs[e].j);
            pair_geometry[e][1][d]=f.C[d].coeff(f.pairs[e].j,f.pairs[e].i);
        }
        if (pn_mass_correction) for (std::size_t e=0;e<f.pairs.size();++e)
            pn_low_order_edge_viscosity[e]=n.pn_viscosity_scale*pn.speed*
                std::max(pair_geometry[e][0].norm(),pair_geometry[e][1].norm());
    }
    pair_node_offsets.assign(static_cast<std::size_t>(f.nodes) + 1, 0);
    for (const auto &p : f.pairs) {
        ++pair_node_offsets[static_cast<std::size_t>(p.i) + 1];
        ++pair_node_offsets[static_cast<std::size_t>(p.j) + 1];
    }
    std::partial_sum(pair_node_offsets.begin(), pair_node_offsets.end(), pair_node_offsets.begin());
    pair_node_edges.resize(2 * f.pairs.size());
    pair_node_sign.resize(2 * f.pairs.size());
    std::vector<std::size_t> pair_node_cursor = pair_node_offsets;
    for (std::size_t e = 0; e < f.pairs.size(); ++e) {
        const auto &p = f.pairs[e];
        const std::size_t i_slot = pair_node_cursor[p.i]++;
        pair_node_edges[i_slot] = static_cast<int>(e);
        pair_node_sign[i_slot] = 1;
        const std::size_t j_slot = pair_node_cursor[p.j]++;
        pair_node_edges[j_slot] = static_cast<int>(e);
        pair_node_sign[j_slot] = -1;
    }
    beta = Vec::Zero(f.nodes);
    std::vector<bool> streaming(f.nodes, false);
    for (const auto &p : f.pairs)
        if (p.s.norm() > 0)
            streaming[p.i] = streaming[p.j] = true;
    for (int i = 0; i < f.nodes; ++i)
        if (streaming[i] && !(n.masked && f.boundary[i]))
            beta[i] = n.gamma * std::pow(f.diameter[i], 3) * pn.speed;
    for (auto &face : f.faces)
        for (auto &p : face.points) {
            int id = -1;
            for (std::size_t j = 0; j < boundaries.size(); ++j)
                if ((boundaries[j]->n - p.normal).norm() < 2e-13) {
                    id = int(j);
                    break;
                }
            if (id < 0) {
                n.budget.check(spatial_bytes + state_bytes + pair_adjacency_bytes + predictor_edge_cache_bytes +
                                   (32.L + 10.L * (boundaries.size() + 1)) * pn.q * pn.q *
                                       sizeof(double),
                               "Normal cache");
                id = int(boundaries.size());
                boundaries.push_back(std::make_unique<Boundary>(pn, canonical, p.normal, n.c0));
            }
            p.normal_id = id;
        }
    if (n.r2) {
        Vec rates=Vec::Zero(f.nodes);
        for (auto &p : f.pairs) {
            V3 cij=V3::Zero(), cji=V3::Zero();
            for(int k=0;k<3;++k){cij[k]=f.C[k].coeff(p.i,p.j); cji[k]=f.C[k].coeff(p.j,p.i);}
            const double edge=std::max(cij.norm(),cji.norm());
            // Explicit positive graph extension on every FE pair, including zero C edges.
            const double graph_extension=p.mass/std::max(f.diameter[p.i],f.diameter[p.j]);
            p.viscosity=std::max(n.anchor*std::max(edge,graph_extension),edge);
            while(!exact::cone(Four(p.viscosity,cij[0],cij[1],cij[2])) ||
                  !exact::cone(Four(p.viscosity,cji[0],cji[1],cji[2]))) {
                p.viscosity=std::nextafter(p.viscosity,std::numeric_limits<double>::infinity());
                require(std::isfinite(p.viscosity),"Unrepresentable directed graph speed");
            }
            rates[p.i]+=2*p.viscosity; rates[p.j]+=2*p.viscosity;
        }
        for(const auto &face:f.faces) for(size_t j=0;j<face.nodes.size();++j) rates[face.nodes[j]]+=face.mass[j];
        f.cfl_step=std::numeric_limits<double>::infinity();
        for(int i=0;i<f.nodes;++i) if(rates[i]>0) f.cfl_step=std::min(f.cfl_step,f.mass[i]/rates[i]);
    }
    // M >= c ML; tensor/submultiplicative bound in the consistent M norm.
    for (int d = 0; d < 3; ++d)
        generator_norm_bound += pn.speed * f.norm_bound(f.C[d]);
    generator_norm_bound += n.r2 ? (pn_mass_correction ? 0. : stabilization_operator->norm_bound) :
                            beta.maxCoeff() * std::pow(f.norm_bound(f.K), 2);
    Vec bound = Vec::Zero(f.nodes);
    for (const auto &face : f.faces)
        for (const auto &p : face.points) {
            double sum = 0;
            for (std::size_t j = 0; j < face.nodes.size(); ++j)
                sum += p.shape[j] / std::sqrt(f.mass[face.nodes[j]]);
            for (std::size_t i = 0; i < face.nodes.size(); ++i)
                bound[face.nodes[i]] += p.weight * p.shape[i] * sum *
                                        (n.r2 ? boundaries[p.normal_id]->llf_norm : boundaries[p.normal_id]->norm) /
                                        std::sqrt(f.mass[face.nodes[i]]);
        }
    if(pn_mass_correction) {
        const double weak_advection=generator_norm_bound;
        Vec graph_degree=Vec::Zero(f.nodes), boundary_diagonal=Vec::Zero(f.nodes);
        double numerical_boundary_bound=bound.maxCoeff();
        if(n.pn_boundary_quadrature=="lumped") {
            Vec lumped_boundary=Vec::Zero(f.nodes);
            for(const auto &face:f.faces)for(const auto &p:face.points)
                for(size_t j=0;j<face.nodes.size();++j)
                    lumped_boundary[face.nodes[j]]+=p.weight*p.shape[j]*boundaries[p.normal_id]->llf_norm/f.mass[face.nodes[j]];
            numerical_boundary_bound=lumped_boundary.maxCoeff();
        }
        for(const auto &p:f.pairs) {
            V3 cij=V3::Zero(),cji=V3::Zero();
            for(int d=0;d<3;++d){cij[d]=f.C[d].coeff(p.i,p.j);cji[d]=f.C[d].coeff(p.j,p.i);}
            const double dN=n.pn_viscosity_scale*pn.speed*std::max(cij.norm(),cji.norm());
            graph_degree[p.i]+=dN; graph_degree[p.j]+=dN;
        }
        for(const auto &face:f.faces)for(const auto &p:face.points)
            for(size_t j=0;j<face.nodes.size();++j)
                boundary_diagonal[face.nodes[j]]+=p.weight*p.shape[j]*pn.speed/f.mass[face.nodes[j]];
        const double low_order_bound=weak_advection+2.*(graph_degree.array()/f.mass.array()).maxCoeff()+boundary_diagonal.maxCoeff()+numerical_boundary_bound;
        // In the lumped-mass norm, ||I-M_L^{-1/2} M M_L^{-1/2}|| <= 1.
        // This bounds R_C plus the mass correction composed with the PN LO rate.
        const double standard_bound=weak_advection+numerical_boundary_bound+low_order_bound;
        generator_norm_bound=(n.stabilization=="m1pn_mass_correction_boundary_consistent" ?
            weak_advection+numerical_boundary_bound+pn_boundary_map_norm*low_order_bound : standard_bound)*(1+1e-11);
    } else {
        generator_norm_bound=(generator_norm_bound+bound.maxCoeff())/f.mass_equivalence*(1+1e-11);
    }
    if (cuda_requested())
        cuda_prepare(pn, fe);
    fast_mc.reset(make_fast_mass_correction(*this));
}
Field Transport::biharmonic(const Field &u, Field *output) const {
    Field w = fe.K * u;
    for (int i = 0; i < fe.nodes; ++i)
        w.row(i) *= -beta[i] / fe.mass[i];
    Field z = fe.K * w;
    if (output)
        *output = std::move(w);
    return z;
}
Field Transport::gather_pairs(const Field &a) const {
    Field g = Field::Zero(fe.nodes, pn.q);
    parallel_for(fe.nodes, [&](int i) {
        for (std::size_t k = pair_node_offsets[i]; k < pair_node_offsets[i + 1]; ++k) {
            const int e = pair_node_edges[k];
            if (!runtime.owns(e)) continue;
            if (pair_node_sign[k] > 0) g.row(i) += a.row(e);
            else g.row(i) -= a.row(e);
        }
    });
    return g;
}
Field Transport::linear_target(const Field &u) const {
    if(options.r2 && (options.stabilization=="m1pn_mass_correction" || options.stabilization=="m1pn_mass_correction_boundary_consistent")) {
        Field rhs=Field::Zero(fe.nodes,pn.q),boundary_rhs=Field::Zero(fe.nodes,pn.q);
        for(int d=0;d<3;++d) add_scatter_product(rhs, fe.C[d], parallel_angular_product(pn.A[d], u));
        for(const auto &face:fe.faces)for(const auto &p:face.points){
            if(options.pn_boundary_quadrature=="consistent") {
                Vec trace=Vec::Zero(pn.q);
                for(size_t j=0;j<face.nodes.size();++j)trace.noalias()+=p.shape[j]*u.row(face.nodes[j]).transpose();
                const Vec flux=boundaries[p.normal_id]->llf_homogeneous*trace;
                for(size_t j=0;j<face.nodes.size();++j)boundary_rhs.row(face.nodes[j])-=p.weight*p.shape[j]*flux.transpose();
            } else for(size_t j=0;j<face.nodes.size();++j) {
                const Vec flux=boundaries[p.normal_id]->llf_homogeneous*u.row(face.nodes[j]).transpose();
                boundary_rhs.row(face.nodes[j])-=p.weight*p.shape[j]*flux.transpose();
            }
        }
        rhs+=boundary_rhs;
        return mass_correction_target(rhs,pn_low_order_rate(u,boundary_rhs));
    }
    if(options.r2){
        Field rhs=stabilization_operator->apply(u);
        for(int k=0;k<3;++k) add_scatter_product(rhs, fe.C[k], parallel_angular_product(pn.A[k], u));
        for(const auto &face:fe.faces) for(const auto &p:face.points) {
            if(options.pn_boundary_quadrature=="consistent") {
                Vec trace=Vec::Zero(pn.q);
                for(size_t j=0;j<face.nodes.size();++j)
                    trace.noalias()+=p.shape[j]*u.row(face.nodes[j]).transpose();
                const Vec flux=boundaries[p.normal_id]->llf_homogeneous*trace;
                for(size_t j=0;j<face.nodes.size();++j)
                    rhs.row(face.nodes[j])-=p.weight*p.shape[j]*flux.transpose();
            } else for(size_t j=0;j<face.nodes.size();++j) {
                Vec flux=boundaries[p.normal_id]->llf_homogeneous*u.row(face.nodes[j]).transpose();
                rhs.row(face.nodes[j])-=p.weight*p.shape[j]*flux.transpose();
            }
        }
        double residual=0; return mass_target(rhs,residual);
    }
    Field rhs = biharmonic(u);
    for (int d = 0; d < 3; ++d)
        add_scatter_product(rhs, fe.C[d], parallel_angular_product(pn.A[d], u));
    for (const auto &f : fe.faces)
        for (const auto &p : f.points) {
            Vec trace = Vec::Zero(pn.q);
            for (std::size_t j = 0; j < f.nodes.size(); ++j)
                trace.noalias() += p.shape[j] * u.row(f.nodes[j]).transpose();
            Vec flux = boundaries[p.normal_id]->homogeneous * trace;
            for (std::size_t j = 0; j < f.nodes.size(); ++j)
                rhs.row(f.nodes[j]) -= p.weight * p.shape[j] * flux.transpose();
        }
    double residual = 0;
    return mass_target(rhs, residual);
}
Field Transport::mass_target(const Field &rhs, double &residual, Field *predictor) const {
    if (options.mass_mode == "consistent") {
        Field target = fe.solve_mass(rhs, options.mass_tolerance, residual);
        if (predictor) *predictor = target;
        return target;
    }
    const int degree = options.mass_mode == "lumped" ? 0 : options.mass_corrections;
    auto inverse_lumped = [&](const Field &v) {
        Field out = v;
        for (int i = 0; i < fe.nodes; ++i) out.row(i) /= fe.mass[i];
        return out;
    };
    Field p = degree == 0 ? Field::Zero(rhs.rows(), rhs.cols()) : inverse_lumped(rhs);
    for (int k = 1; k < degree; ++k)
        p = inverse_lumped(rhs + fe.mass.asDiagonal() * p - fe.M * p);
    if (predictor) *predictor = p;
    Field target = degree == 0 ? inverse_lumped(rhs)
                               : inverse_lumped(rhs + fe.mass.asDiagonal() * p - fe.M * p);
    residual = maxabs(fe.M * target - rhs);
    return target;
}
Field Transport::pn_low_order_rate(const Field &u, const Field &boundary_rhs, const Field &source,
                                     const std::array<Field, 3> *angular_flux) const {
    const bool mass_correction_mode=options.stabilization=="m1pn_mass_correction" ||
                                    options.stabilization=="m1pn_mass_correction_boundary_consistent";
    require(mass_correction_mode, "PN low-order rate is only available for a PN mass correction");
    require(u.rows()==fe.nodes&&u.cols()==pn.q&&u.allFinite(),"Invalid PN field for low-order predictor");
    const bool relaxed = relaxed_association();
    std::array<Field, 3> owned_flux;
    if (relaxed && !angular_flux) {
        for (int d = 0; d < 3; ++d)
            owned_flux[d] = parallel_angular_product(pn.A[d], u);
        angular_flux = &owned_flux;
    }
    Field rhs=Field::Zero(fe.nodes,pn.q);
    if (angular_flux) {
        const int q = pn.q;
        int team = 1;
#ifdef _OPENMP
        team = omp_get_max_threads();
#endif
        std::vector<std::vector<double>> buffers(team, std::vector<double>(q));
        parallel_for(fe.nodes, [&](int i) {
            double *acc = buffers[omp_get_thread_num()].data();
            double *row = &rhs.coeffRef(i, 0);
            std::fill_n(row, q, 0.);
            for (std::size_t k = pair_node_offsets[i]; k < pair_node_offsets[i + 1]; ++k) {
                const int e = pair_node_edges[k];
                const auto &p = fe.pairs[e];
                const V3 &cij = pair_geometry[e][0], &cji = pair_geometry[e][1];
                const double dN = pn_low_order_edge_viscosity[e];
                const double *ui = &u.coeff(p.i, 0), *uj = &u.coeff(p.j, 0);
                const bool from_i = pair_node_sign[k] > 0;
                const V3 &direction = from_i ? cij : cji;
                const double sign = from_i ? 1. : -1.;
                for (int c = 0; c < q; ++c)
                    acc[c] = dN * sign * (uj[c] - ui[c]);
                for (int d = 0; d < 3; ++d) {
                    if (direction[d] == 0.)
                        continue;
                    const double *ai = &(*angular_flux)[d].coeff(p.i, 0);
                    const double *aj = &(*angular_flux)[d].coeff(p.j, 0);
                    const double scale = direction[d] * sign;
                    for (int c = 0; c < q; ++c)
                        acc[c] -= scale * (aj[c] - ai[c]);
                }
                for (int c = 0; c < q; ++c)
                    row[c] += acc[c];
            }
        });
    } else {
        // Same products as `dN*(aj-ai) - fast_apply(...)`, without a Vec per edge.
        // pn_apply scales each stored entry before the multiply, then accumulates.
        const auto patterns=angular_patterns(pn);
        const int q=pn.q;
        int team=1;
#ifdef _OPENMP
        team=omp_get_max_threads();
#endif
        struct EdgeBuf { std::vector<double> diff, applied, flux; };
        std::vector<EdgeBuf> bufs(team);
        for(auto &b:bufs) { b.diff.resize(q); b.applied.resize(q); b.flux.resize(q); }
        parallel_for(fe.nodes,[&](int i) {
            auto &b=bufs[omp_get_thread_num()];
            Eigen::Map<Vec> diff(b.diff.data(),q);
            Eigen::Map<Vec> flux(b.flux.data(),q);
            const Eigen::Map<const Vec> applied(b.applied.data(),q);
            for(std::size_t k=pair_node_offsets[i];k<pair_node_offsets[i+1];++k) {
                const int e=pair_node_edges[k];
                const auto &p=fe.pairs[e];
                const V3 &cij=pair_geometry[e][0], &cji=pair_geometry[e][1];
                const double dN=pn_low_order_edge_viscosity[e];
                if(pair_node_sign[k]>0) {
                    diff=u.row(p.j).transpose()-u.row(p.i).transpose();
                    pn_apply(patterns, cij, diff.data(), b.applied.data());
                    flux=dN*diff-applied;
                } else {
                    diff=u.row(p.i).transpose()-u.row(p.j).transpose();
                    pn_apply(patterns, cji, diff.data(), b.applied.data());
                    flux=dN*diff-applied;
                }
                rhs.row(i)+=flux.transpose();
            }
        });
    }
    if(boundary_rhs.size()){require(boundary_rhs.rows()==fe.nodes&&boundary_rhs.cols()==pn.q&&boundary_rhs.allFinite(),"Invalid assembled PN boundary load");rhs+=boundary_rhs;}
    // Convert the weak interior transport to its strong graph form. The
    // corresponding physical trace restores the integration-by-parts term;
    // boundary_rhs contains minus the actual numerical PN boundary flux.
    if (angular_flux) {
        const int q = pn.q;
        std::vector<double> physical(q);
        for (const auto &face : fe.faces)
            for (const auto &p : face.points)
                for (size_t j = 0; j < face.nodes.size(); ++j) {
                    const int node = face.nodes[j];
                    directional_from_node(*angular_flux, node, p.normal, physical.data(), q);
                    double *row = &rhs.coeffRef(node, 0);
                    const double weight = p.weight * p.shape[j];
                    for (int c = 0; c < q; ++c)
                        row[c] += weight * physical[c];
                }
    } else {
        const auto patterns=angular_patterns(pn);
        std::vector<double> state(pn.q), physical(pn.q);
        Eigen::Map<Vec> state_map(state.data(),pn.q);
        for(const auto &face:fe.faces)for(const auto &p:face.points) {
            for(size_t j=0;j<face.nodes.size();++j) {
                const int node=face.nodes[j];
                state_map=u.row(node).transpose();
                pn_apply(patterns, p.normal, state.data(), physical.data());
                const Eigen::Map<const Vec> applied(physical.data(),pn.q);
                rhs.row(node)+=p.weight*p.shape[j]*applied.transpose();
            }
        }
    }
    if(source.size()){require(source.rows()==fe.nodes&&source.cols()==pn.q&&source.allFinite(),"Invalid PN low-order source load");rhs+=source;}
    if (relaxed) {
        const int q = pn.q;
        parallel_for(fe.nodes, [&](int i) {
            double *row = &rhs.coeffRef(i, 0);
            const double inv = 1. / fe.mass[i];
            for (int c = 0; c < q; ++c)
                row[c] *= inv;
        });
    } else for(int i=0;i<fe.nodes;++i)rhs.row(i)/=fe.mass[i];
    if(options.stabilization=="m1pn_mass_correction_boundary_consistent") {
        require(int(pn_boundary_map.size())==fe.nodes,"Missing boundary-consistent PN predictor map");
        const Field standard=rhs;
        for(int i=0;i<fe.nodes;++i)rhs.row(i)=standard.row(pn_boundary_map[i]);
    }
    return rhs;
}
Field Transport::mass_correction(const Field &rate) const {
    require(rate.rows()==fe.nodes&&rate.cols()==pn.q&&rate.allFinite(),"Invalid PN rate for mass correction");
    if (cuda_ready()) {
        Field correction(rate.rows(), rate.cols());
        cuda_mass_correction(fe, rate, correction);
        return correction;
    }
    // Eigen's product is faster here than a hand-written row gather at P11 width.
    return fe.mass.asDiagonal()*rate-fe.M*rate;
}
Field Transport::mass_correction_target(const Field &consistent_rhs,const Field &low_rate) const {
    require(consistent_rhs.rows()==fe.nodes&&consistent_rhs.cols()==pn.q&&consistent_rhs.allFinite(),"Invalid consistent PN residual");
    Field target=consistent_rhs+mass_correction(low_rate);
    if (relaxed_association()) {
        const int q = pn.q;
        parallel_for(fe.nodes, [&](int i) {
            double *row = &target.coeffRef(i, 0);
            const double inv = 1. / fe.mass[i];
            for (int c = 0; c < q; ++c)
                row[c] *= inv;
        });
    } else for(int i=0;i<fe.nodes;++i)target.row(i)/=fe.mass[i];
    return target;
}
void Transport::check_state(const Field &u) const {
    require(u.rows() == fe.nodes && u.cols() == pn.q && u.allFinite(),
            "Invalid/nonfinite PN field");
    std::atomic<int> failing{-1};
    parallel_for(fe.nodes, [&](int i) {
        if (exact::cone(project(u.row(i).transpose())))
            return;
        int seen = failing.load(std::memory_order_relaxed);
        while (seen < 0 || i < seen) {
            if (failing.compare_exchange_weak(seen, i, std::memory_order_relaxed))
                break;
        }
    });
    const int node = failing.load(std::memory_order_relaxed);
    if (node >= 0)
        cone_require(false, "Stored PN projection outside M1 cone at node " + std::to_string(node));
}
Stage Transport::evaluate(double time, const Field &u, bool unlimited, const Field &source) {
    if (options.r2) return evaluate_r2(time,u,unlimited,source);
    check_state(u);
    Stage out;
    auto &diag = out.diagnostics;
    double start = seconds();
    Field w;
    Field z = biharmonic(u, &w);
    timings.biharmonic += seconds() - start;
    Field rhs = z;
    for (int d = 0; d < 3; ++d)
        add_scatter_product(rhs, fe.C[d], parallel_angular_product(pn.A[d], u));
    out.source = source.size() ? source : Field::Zero(fe.nodes, pn.q);
    require(out.source.rows() == fe.nodes && out.source.cols() == pn.q && out.source.allFinite(),
            "Invalid integrated source");
    if (!unlimited)
        for (int i = 0; i < fe.nodes; ++i)
            require(exact::cone(project(out.source.row(i).transpose())),
                    "Transport source must have nonnegative physical moments");
    rhs += out.source;
    start = seconds();
    std::vector<Four> low(fe.nodes);
    std::vector<Levermore> closure;
    closure.reserve(fe.nodes);
    for (int i = 0; i < fe.nodes; ++i)
        low[i] = project(u.row(i).transpose());
    for (int i = 0; i < fe.nodes; ++i)
        closure.emplace_back(low[i]);
    timings.mcl += seconds() - start;
    std::vector<Field> face_high, face_low, face_normal;
    start = seconds();
    for (const auto &f : fe.faces) {
        const int nd = int(f.nodes.size());
        Field h = Field::Zero(nd, pn.q), q0 = Field::Zero(nd, 4), normal = q0;
        // Only reuse identical normals, not the tolerance-merged C1 normal IDs.
        // Thus warped faces retain their original pointwise kinetic fluxes.
        V3 previous_normal = V3::Zero();
        std::vector<Four> outgoing(nd), full_flux(nd);
        for (const auto &p : f.points) {
            if ((p.normal.array() != previous_normal.array()).any()) {
                for (int j = 0; j < nd; ++j) {
                    outgoing[j] = closure[f.nodes[j]].outgoing(p.normal);
                    full_flux[j] = closure[f.nodes[j]].flux(p.normal);
                }
                previous_normal = p.normal;
            }
            const auto &b = *boundaries[p.normal_id];
            const auto data = b.data(inflow, p.x, time);
            Vec trace = Vec::Zero(pn.q);
            for (int j = 0; j < nd; ++j)
                trace.noalias() += p.shape[j] * u.row(f.nodes[j]).transpose();
            Vec gh = b.flux(trace, data.d);
            for (int j = 0; j < nd; ++j) {
                if (p.shape[j] == 0)
                    continue;
                const int i = f.nodes[j];
                const double weight = p.weight * p.shape[j];
                // Eq. (4.9), retaining the consistent surface mass exactly once.
                h.row(j) +=
                    weight *
                    (gh - .5 * fast_apply(pn, p.normal, trace - u.row(i).transpose())).transpose();
                rhs.row(i) -= weight * gh.transpose();
                q0.row(j) += weight * (outgoing[j] + data.kinetic).transpose();
                normal.row(j) += weight * full_flux[j].transpose();
            }
        }
        face_high.push_back(std::move(h));
        face_low.push_back(std::move(q0));
        face_normal.push_back(std::move(normal));
    }
    timings.boundary += seconds() - start;
    start = seconds();
    Field mass_predictor;
    out.target = mass_target(rhs, diag.mass_residual, &mass_predictor);
    timings.mass += seconds() - start;
    if (options.mass_mode == "consistent") out.mass_error = fe.M * out.target - rhs;
    else out.mass_error = Field::Zero(fe.nodes, pn.q);
    Field high_pairs = Field::Zero(fe.pairs.size(), pn.q);
    out.accepted_pairs = high_pairs;
    Vec alphas = Vec::Ones(fe.pairs.size()),
        bar_margin = Vec::Constant(fe.pairs.size(), std::numeric_limits<double>::infinity());
    Vec density_clip = Vec::Zero(fe.pairs.size()), redirection = Vec::Zero(fe.pairs.size()),
        retention = Vec::Zero(fe.pairs.size());
    Eigen::VectorXi pair_pass = Eigen::VectorXi::Zero(fe.pairs.size()),
        pair_round = Eigen::VectorXi::Zero(fe.pairs.size()),
        pair_scalar = Eigen::VectorXi::Zero(fe.pairs.size()),
        base_bar_recomputed = Eigen::VectorXi::Zero(fe.pairs.size());
    start = seconds();
    out.boundary_gather = Field::Zero(fe.nodes, pn.q);
    Field target_boundary = out.boundary_gather;
    int local_failure = 0;
    std::string failure_message;
    try {
        parallel_for(int(fe.pairs.size()), [&](int e) {
            if (!runtime.owns(e))
                return;
            const auto &p = fe.pairs[e];
            const Vec ui = u.row(p.i).transpose(), uj = u.row(p.j).transpose();
            Vec h = -fast_apply(pn, p.s, ui + uj) + p.stiffness * (w.row(p.j) - w.row(p.i)).transpose() +
                    p.mass * (mass_predictor.row(p.i) - mass_predictor.row(p.j)).transpose();
            const Four fi = closure[p.i].flux(p.s), fj = closure[p.j].flux(p.s);
            Four b = p.viscosity * (low[p.j] - low[p.i]) - fi - fj;
            Four bar = .5 * (low[p.i] + low[p.j]) - (fj - fi) / (2 * p.viscosity);
            if (!exact::cone(bar)) {
                bar = stable_base_bar(low[p.i], low[p.j], p.s, p.viscosity);
                base_bar_recomputed[e] = 1;
                cone_require(exact::cone(bar), "Stable low-order pair base outside stored cone");
            }
            Four correction = project(h) - b, delta = correction / (2 * p.viscosity);
            double alpha = 1;
            Four accepted_correction;
            Four left, right;
            try {
                if (!unlimited && options.density_vector) {
                    const auto limited = density_pair_limit(bar, bar, delta);
                    accepted_correction = 2 * p.viscosity * limited.correction;
                    const Four actual_delta = accepted_correction / (2 * p.viscosity);
                    left = bar + actual_delta;
                    right = bar - actual_delta;
                    density_clip[e] = std::abs(delta[0] - actual_delta[0]);
                    redirection[e] = (delta.tail<3>() - actual_delta.tail<3>()).norm();
                    retention[e] = delta[0] == 0 ? 1 : std::abs(actual_delta[0] / delta[0]);
                    pair_pass[e] = limited.target_pass;
                    pair_round[e] = limited.rounding_fallback;
                    pair_scalar[e] = limited.scalar_fallback;
                    alphas[e] = limited.target_pass ? 1 :
                                std::min(retention[e], limited.momentum_theta);
                } else {
                    if (!unlimited)
                        alpha = segment(bar, delta, true);
                    accepted_correction = alpha * correction;
                    left = bar + alpha * delta;
                    right = bar - alpha * delta;
                }
            } catch (const StepRejected &error) {
                throw StepRejected("pair " + std::to_string(e) + " nodes " + std::to_string(p.i) +
                                   "," + std::to_string(p.j) + ": " + error.what());
            }
            if (!unlimited && options.density_vector &&
                !(exact::cone(left) && exact::cone(right))) {
                // Multiplying the selected p by 2d and recovering its actual
                // stored endpoint increment can move a boundary value outward.
                // Retry with the established common scalar limiter.
                alpha = segment(bar, delta, true);
                accepted_correction = alpha * correction;
                Four actual_delta = accepted_correction / (2 * p.viscosity);
                left = bar + actual_delta;
                right = bar - actual_delta;
                if (!(exact::cone(left) && exact::cone(right))) {
                    double lo = 0, hi = alpha;
                    for (int k = 0; k < 60; ++k) {
                        const double mid = (lo + hi) / 2;
                        const Four trial = (mid * correction) / (2 * p.viscosity);
                        if (exact::cone(bar + trial) && exact::cone(bar - trial)) lo = mid;
                        else hi = mid;
                    }
                    alpha = lo;
                    accepted_correction = alpha * correction;
                    actual_delta = accepted_correction / (2 * p.viscosity);
                    left = bar + actual_delta;
                    right = bar - actual_delta;
                }
                pair_scalar[e] = 1;
                pair_round[e] = 1;
                pair_pass[e] = 0;
                density_clip[e] = std::abs(delta[0] - actual_delta[0]);
                redirection[e] = (delta.tail<3>() - actual_delta.tail<3>()).norm();
                retention[e] = delta[0] == 0 ? 1 : std::abs(actual_delta[0] / delta[0]);
                alphas[e] = alpha;
            }
            if (!options.density_vector || unlimited)
                alphas[e] = alpha;
            bar_margin[e] = std::min(margin(left), margin(right));
            if (!unlimited)
                cone_require(exact::cone(left) && exact::cone(right),
                             "Two-sided corrected bar failed");
            high_pairs.row(e) = h.transpose();
            set_low(h, b + accepted_correction);
            out.accepted_pairs.row(e) = h.transpose();
        });
        for (std::size_t fi = 0; fi < fe.faces.size(); ++fi)
            if (runtime.owns(fi)) {
                const auto &f = fe.faces[fi];
                for (std::size_t j = 0; j < f.nodes.size(); ++j) {
                    if (f.mass[j] <= 0)
                        continue;
                    const int i = f.nodes[j];
                    Vec h = face_high[fi].row(j).transpose();
                    const Four q0 = face_low[fi].row(j).transpose();
                    const double mass = f.mass[j] * options.anchor;
                    const Four bar = low[i] + (face_normal[fi].row(j).transpose() - q0) / mass;
                    const Four delta = project(h) - q0;
                    const Four change = -delta / mass;
                    double eta = 1;
                    try {
                        if (!unlimited)
                            eta = segment(bar, change);
                    } catch (const StepRejected &error) {
                        throw StepRejected("face " + std::to_string(fi) + " node " +
                                           std::to_string(i) + ": " + error.what());
                    }
                    Four accepted_bar = bar + eta * change;
                    if (!unlimited)
                        cone_require(exact::cone(accepted_bar), "One-sided corrected bar failed");
                    diag.minimum_bar = std::min(diag.minimum_bar, margin(accepted_bar));
                    diag.minimum_eta = std::min(diag.minimum_eta, eta);
                    if (eta < 1)
                        ++diag.active_faces;
                    target_boundary.row(i) += h.transpose();
                    set_low(h, q0 + eta * delta);
                    out.boundary_gather.row(i) += h.transpose();
                }
            }
    } catch (const StepRejected &e) {
        local_failure = 1;
        failure_message = e.what();
    } catch (const std::exception &e) {
        local_failure = 2;
        failure_message = e.what();
    }
    runtime.finish_local_phase(local_failure, failure_message);
    timings.mcl += seconds() - start;
    Field target_gather = gather_pairs(high_pairs) - target_boundary;
    out.gather = gather_pairs(out.accepted_pairs) - out.boundary_gather;
    start = seconds();
    runtime.sum(target_gather);
    runtime.sum(out.gather);
    // Keep local boundary/pair ledgers; gather is globally reduced, source replicated.
    timings.communication += seconds() - start;
    target_gather += out.source;
    out.gather += out.source;
    diag.target_recovery =
        maxabs(target_gather - fe.mass.asDiagonal() * out.target + out.mass_error);
    out.rate = out.target;
    for (int i = 0; i < fe.nodes; ++i) {
        Vec k = out.rate.row(i).transpose();
        set_low(k, project(out.gather.row(i).transpose()) / fe.mass[i]);
        out.rate.row(i) = k.transpose();
        diag.projection = std::max(
            diag.projection, (project(k) - project(out.gather.row(i).transpose()) / fe.mass[i])
                                 .lpNorm<Eigen::Infinity>());
    }
    Field defect = fe.mass.asDiagonal() * out.rate - out.gather;
    diag.local_balance = maxabs(defect);
    diag.mass_defect = maxabs(out.mass_error);
    for (int i = 0; i < fe.nodes; ++i) {
        Vec eps = out.mass_error.row(i).transpose();
        set_low(eps, Four::Zero());
        defect.row(i) -= eps.transpose();
    }
    require(maxabs(defect) < identity_tolerance() * std::max(1., maxabs(out.gather)),
            "Accepted Q-component local balance identity failed");
    if (pn.q > 4)
        diag.high_rate_error =
            maxabs(out.rate.rightCols(pn.q - 4) - out.target.rightCols(pn.q - 4));
    Field mu = fe.M * u;
    diag.target_energy = (mu.array() * out.target.array()).sum();
    diag.sync_work = (mu.array() * (out.rate - out.target).array()).sum();
    diag.minimum_alpha = runtime.reduce(alphas.minCoeff(), -1);
    diag.minimum_eta = runtime.reduce(diag.minimum_eta, -1);
    diag.minimum_bar = runtime.reduce(std::min(diag.minimum_bar, bar_margin.minCoeff()), -1);
    diag.active_pairs = long(runtime.reduce((alphas.array() < 1.).count(), 0));
    diag.active_faces = long(runtime.reduce(diag.active_faces, 0));
    diag.density_clip_sum = runtime.reduce(density_clip.sum(), 0);
    diag.momentum_redirection_sum = runtime.reduce(redirection.sum(), 0);
    diag.density_retention_sum = runtime.reduce(retention.sum(), 0);
    diag.pair_target_passes = long(runtime.reduce(pair_pass.cast<double>().sum(), 0));
    diag.pair_rounding_fallbacks = long(runtime.reduce(pair_round.cast<double>().sum(), 0));
    diag.pair_scalar_fallbacks = long(runtime.reduce(pair_scalar.cast<double>().sum(), 0));
    diag.base_bar_reevaluations = long(runtime.reduce(base_bar_recomputed.cast<double>().sum(), 0));
    return out;
}
} // namespace m1pn

namespace m1pn {
Stage Transport::evaluate_r2(double time, const Field &u, bool unlimited, const Field &source,
                             double implicit_dt) {
    check_state(u);
    const bool pn_mass_correction=options.stabilization=="m1pn_mass_correction" ||
                                  options.stabilization=="m1pn_mass_correction_boundary_consistent";
    Stage out;
    auto &diag=out.diagnostics;
    const bool relaxed = relaxed_association();
    fill_angular_products(pn, u, stage_angular);
    const std::array<Field, 3> &cached_pn_flux = stage_angular;
    auto pair_streaming = [&](int i,int j,const V3 &cij,const V3 &cji) -> Vec {
        Vec flux(pn.q);
        double *out = flux.data();
        std::fill_n(out, pn.q, 0.);
        for(int d=0;d<3;++d) {
            const double cj=cji[d], ci=cij[d];
            if(cj==0. && ci==0.) continue;
            const double *rj=&cached_pn_flux[d].coeff(j,0);
            const double *ri=&cached_pn_flux[d].coeff(i,0);
            for(int c=0;c<pn.q;++c) out[c]+=cj*rj[c]-ci*ri[c];
        }
        return flux;
    };
    double start=seconds();
    if(stage_rhs.rows()!=fe.nodes || stage_rhs.cols()!=pn.q) stage_rhs.resize(fe.nodes,pn.q);
    if(stage_stabilization_pairs.rows()!=int(fe.pairs.size()) || stage_stabilization_pairs.cols()!=pn.q)
        stage_stabilization_pairs.resize(fe.pairs.size(),pn.q);
    Field &stabilization_pairs=stage_stabilization_pairs;
    Field &rhs=stage_rhs;
    if(pn_mass_correction){rhs.setZero();stabilization_pairs.setZero();}
    else {
        rhs=stabilization_operator->apply(u,&stabilization_pairs,options.bh_pair_mode);
        timings.biharmonic+=seconds()-start;
    }
    add_directional_scatter(rhs, fe.C, cached_pn_flux);
    out.source=source.size()?source:Field::Zero(fe.nodes,pn.q);
    require(out.source.rows()==fe.nodes && out.source.cols()==pn.q && out.source.allFinite(),
            "Invalid integrated source");
    if(!unlimited) for(int i=0;i<fe.nodes;++i)
        require(exact::cone(project(out.source.row(i).transpose())),"Nonrealizable source load");
    rhs+=out.source;
    ImplicitResult implicit_result;
    std::vector<Four> low;
    std::vector<std::unique_ptr<Levermore>> closure;
    low.assign(fe.nodes, Four::Zero());
    closure.resize(fe.nodes);
    parallel_for(fe.nodes, [&](int i) { low[i] = project(u.row(i).transpose()); });
    auto get_closure = [&](int i) -> const Levermore & {
        if (!closure[i]) closure[i] = std::make_unique<Levermore>(low[i]);
        return *closure[i];
    };
    if (unlimited || !options.lazy)
        parallel_for(fe.nodes, [&](int i) { (void)get_closure(i); });
    std::vector<Field> face_high,face_low,face_bar;
    if(stage_boundary.rows()!=fe.nodes || stage_boundary.cols()!=pn.q)
        stage_boundary.resize(fe.nodes,pn.q);
    stage_boundary.setZero();
    Field &pn_boundary_rhs=stage_boundary;
    std::vector<unsigned char> face_lazy_certified;
    start=seconds();
    if(!relaxed) {
        const int nfaces=int(fe.faces.size());
        struct FaceSlot { Field high, low, bar; unsigned char certified=0; };
        std::vector<FaceSlot> slots(nfaces);
        std::mutex closure_gate;
        auto closure_of=[&](int i)->const Levermore & {
            if(unlimited || !options.lazy) return get_closure(i);
            std::lock_guard<std::mutex> lock(closure_gate);
            return get_closure(i);
        };
        parallel_for(nfaces,[&](int f) {
        const auto &face=fe.faces[f];
        const int nd=face.nodes.size();
        Field hn=Field::Zero(nd,pn.q),hl=Field::Zero(nd,4),bar=hl;
        if (!unlimited && options.lazy) {
            for(const auto &p:face.points) {
                const auto &b=*boundaries[p.normal_id];
                const auto data=b.data(inflow,p.x,time);
                Vec common_flux;
                if(options.pn_boundary_quadrature=="consistent") {
                    Vec trace=Vec::Zero(pn.q);
                    for(int j=0;j<nd;++j) trace.noalias()+=p.shape[j]*u.row(face.nodes[j]).transpose();
                    const Vec ue=b.exterior(trace,data.d);
                    common_flux=.5*(fast_apply(pn, p.normal,trace)+fast_apply(pn, p.normal,ue))-.5*(ue-trace);
                }
                for(int j=0;j<nd;++j) {
                    const double weight=p.weight*p.shape[j]; if(weight==0) continue;
                    const int i=face.nodes[j];
                    const Four exterior=inflow.exterior_state ?
                        inflow.exterior_state(p.x,time,p.normal,low[i]) :
                        (inflow.exterior_m1?inflow.exterior_m1(p.x,time):data.moments);
                    require(exact::cone(exterior),"M1 exterior state outside cone");
                    Vec flux;
                    if(options.pn_boundary_quadrature=="consistent") flux=common_flux;
                    else {
                        const Vec ui=u.row(i).transpose(),ue=b.exterior(ui,data.d);
                        flux=.5*(fast_apply(pn, p.normal,ui)+fast_apply(pn, p.normal,ue))-.5*(ue-ui);
                    }
                    hn.row(j)+=weight*flux.transpose();
                }
            }
            bool certified=true;
            for(int j=0;j<nd && certified;++j) {
                const double a=face.mass[j]; if(a<=0) continue;
                const int i=face.nodes[j]; const Four ui=low[i];
                const Four base=ui-project(hn.row(j).transpose())/a;
                const auto chi=lazy_m1::chi_interval(ui);
                for(double c:{chi.lo,chi.hi}) {
                    Four f=Four::Zero();
                    for(const auto &p:face.points) {
                        const double weight=p.weight*p.shape[j]; if(weight!=0)
                            f+=weight*lazy_m1::pressure_times(ui,c,p.normal);
                    }
                    if(!lazy_m1::strictly_interior(base+f/a)) {certified=false;break;}
                }
            }
            if(certified) {
                slots[f].certified=1;
                slots[f].high=std::move(hn); slots[f].low=std::move(hl);
                slots[f].bar=std::move(bar);
                return;
            }
            hn.setZero();
        }
        for(const auto &p:face.points) {
            const auto &b=*boundaries[p.normal_id];
            const auto data=b.data(inflow,p.x,time);
            Four shared_exterior=Four::Zero();
            std::optional<Levermore> shared_ext_closure;
            if(!inflow.exterior_state) {
                shared_exterior=inflow.exterior_m1?inflow.exterior_m1(p.x,time):data.moments;
                require(exact::cone(shared_exterior),"M1 exterior state outside cone");
                shared_ext_closure.emplace(shared_exterior);
            }
            Vec consistent_flux;
            if (options.pn_boundary_quadrature=="consistent") {
                Vec trace=Vec::Zero(pn.q);
                for(int j=0;j<nd;++j) trace.noalias()+=p.shape[j]*u.row(face.nodes[j]).transpose();
                const Vec exterior_trace=b.exterior(trace,data.d);
                consistent_flux=.5*(fast_apply(pn, p.normal,trace)+fast_apply(pn, p.normal,exterior_trace))-
                                 .5*(exterior_trace-trace);
            }
            for(int j=0;j<nd;++j) {
                const double weight=p.weight*p.shape[j];
                if(weight==0) continue;
                const int i=face.nodes[j];
                const Four exterior=inflow.exterior_state ? inflow.exterior_state(p.x,time,p.normal,low[i]) : shared_exterior;
                require(exact::cone(exterior),"M1 exterior state outside cone");
                std::optional<Levermore> node_ext_closure;
                const Levermore *ext_closure=nullptr;
                if(inflow.exterior_state) {
                    node_ext_closure.emplace(exterior);
                    ext_closure=&*node_ext_closure;
                } else ext_closure=&*shared_ext_closure;
                const Vec ui=u.row(i).transpose();
                const Vec ue=b.exterior(ui,data.d);
                const Vec flux=options.pn_boundary_quadrature=="consistent" ?
                               consistent_flux :
                               .5*(fast_apply(pn, p.normal,ui)+fast_apply(pn, p.normal,ue))-.5*(ue-ui);
                const Four fi=closure_of(i).flux(p.normal),fx=ext_closure->flux(p.normal);
                const Four lo_flux=.5*(fi+fx)-.5*(exterior-low[i]);
                Four ghost=.5*(low[i]+exterior)-.5*(fx-fi);
                if(!exact::cone(ghost)) ghost=stable_base_bar(low[i],exterior,.5*p.normal,.5);
                cone_require(exact::cone(ghost),"R2 LLF ghost base outside cone");
                hn.row(j)+=weight*flux.transpose();
                hl.row(j)+=weight*lo_flux.transpose();
                bar.row(j)+=weight*ghost.transpose();
            }
        }
        for(int j=0;j<nd;++j) if(face.mass[j]>0) bar.row(j)/=face.mass[j];
        slots[f].high=std::move(hn); slots[f].low=std::move(hl);
        slots[f].bar=std::move(bar);
        });
        // Original face order, so shared-node sums match the serial loop bitwise.
        for(int f=0;f<nfaces;++f) {
            const auto &face=fe.faces[f];
            face_lazy_certified.push_back(slots[f].certified);
            for(int j=0;j<int(face.nodes.size());++j) {
                rhs.row(face.nodes[j])-=slots[f].high.row(j);
                pn_boundary_rhs.row(face.nodes[j])-=slots[f].high.row(j);
            }
            face_high.push_back(std::move(slots[f].high));
            face_low.push_back(std::move(slots[f].low));
            face_bar.push_back(std::move(slots[f].bar));
        }
    } else {
        const int nfaces=int(fe.faces.size());
        struct FaceSlot { Field high, low, bar; unsigned char certified=0; };
        std::vector<FaceSlot> slots(nfaces);
        std::mutex closure_gate;
        auto closure_of=[&](int i)->const Levermore & {
            if(unlimited || !options.lazy) return get_closure(i);
            std::lock_guard<std::mutex> lock(closure_gate);
            return get_closure(i);
        };
        const auto patterns=angular_patterns(pn);
        parallel_for(nfaces,[&](int f) {
            const auto &face=fe.faces[f];
            const int nd=int(face.nodes.size());
            Field hn=Field::Zero(nd,pn.q),hl=Field::Zero(nd,4),bar=hl;
            auto commit_high=[&](int j,double weight,const double *flux) {
                double *row=&hn.coeffRef(j,0);
                for(int c=0;c<pn.q;++c) row[c]+=weight*flux[c];
            };
            auto nodal_llf=[&](int i,const Boundary &b,const Vec &data_d,const V3 &normal,double *flux) {
                Vec ui=Eigen::Map<const Vec>(&u.coeff(i,0),pn.q);
                const Vec ue=b.exterior(ui,data_d);
                std::vector<double> a_ui(pn.q),a_ue(pn.q);
                directional_from_node(cached_pn_flux,i,normal,a_ui.data(),pn.q);
                pn_apply(patterns,normal,ue.data(),a_ue.data());
                for(int c=0;c<pn.q;++c)
                    flux[c]=.5*(a_ui[c]+a_ue[c])-.5*(ue[c]-ui[c]);
            };
            if(!unlimited && options.lazy) {
                for(const auto &p:face.points) {
                    const auto &b=*boundaries[p.normal_id];
                    const auto data=b.data(inflow,p.x,time);
                    std::vector<double> common(pn.q);
                    bool have_common=false;
                    if(options.pn_boundary_quadrature=="consistent") {
                        Vec trace=Vec::Zero(pn.q),a_trace=Vec::Zero(pn.q),part(pn.q);
                        for(int j=0;j<nd;++j) {
                            const double *src=&u.coeff(face.nodes[j],0);
                            for(int c=0;c<pn.q;++c) trace[c]+=p.shape[j]*src[c];
                            if(p.shape[j]==0.) continue;
                            directional_from_node(cached_pn_flux,face.nodes[j],p.normal,part.data(),pn.q);
                            a_trace+=p.shape[j]*part;
                        }
                        const Vec ue=b.exterior(trace,data.d);
                        std::vector<double> a_ue(pn.q);
                        pn_apply(patterns,p.normal,ue.data(),a_ue.data());
                        for(int c=0;c<pn.q;++c)
                            common[c]=.5*(a_trace[c]+a_ue[c])-.5*(ue[c]-trace[c]);
                        have_common=true;
                    }
                    std::vector<double> flux(pn.q);
                    for(int j=0;j<nd;++j) {
                        const double weight=p.weight*p.shape[j]; if(weight==0) continue;
                        const int i=face.nodes[j];
                        const Four exterior=inflow.exterior_state ?
                            inflow.exterior_state(p.x,time,p.normal,low[i]) :
                            (inflow.exterior_m1?inflow.exterior_m1(p.x,time):data.moments);
                        require(exact::cone(exterior),"M1 exterior state outside cone");
                        if(!have_common) nodal_llf(i,b,data.d,p.normal,flux.data());
                        commit_high(j,weight,have_common?common.data():flux.data());
                    }
                }
                bool certified=true;
                for(int j=0;j<nd && certified;++j) {
                    const double a=face.mass[j]; if(a<=0) continue;
                    const int i=face.nodes[j]; const Four ui=low[i];
                    const Four base=ui-project(hn.row(j).transpose())/a;
                    const auto chi=lazy_m1::chi_interval(ui);
                    for(double c:{chi.lo,chi.hi}) {
                        Four pressure=Four::Zero();
                        for(const auto &p:face.points) {
                            const double weight=p.weight*p.shape[j]; if(weight!=0)
                                pressure+=weight*lazy_m1::pressure_times(ui,c,p.normal);
                        }
                        if(!lazy_m1::strictly_interior(base+pressure/a)) {certified=false;break;}
                    }
                }
                if(certified) {
                    slots[f].certified=1;
                    slots[f].high=std::move(hn);
                    slots[f].low=std::move(hl);
                    slots[f].bar=std::move(bar);
                    return;
                }
                hn.setZero();
            }
            for(const auto &p:face.points) {
                const auto &b=*boundaries[p.normal_id];
                const auto data=b.data(inflow,p.x,time);
                Four shared_exterior=Four::Zero();
                std::optional<Levermore> shared_ext_closure;
                if(!inflow.exterior_state) {
                    shared_exterior=inflow.exterior_m1?inflow.exterior_m1(p.x,time):data.moments;
                    require(exact::cone(shared_exterior),"M1 exterior state outside cone");
                    shared_ext_closure.emplace(shared_exterior);
                }
                std::vector<double> common(pn.q);
                bool have_common=false;
                if(options.pn_boundary_quadrature=="consistent") {
                    Vec trace=Vec::Zero(pn.q),a_trace=Vec::Zero(pn.q),part(pn.q);
                    for(int j=0;j<nd;++j) {
                        const double *src=&u.coeff(face.nodes[j],0);
                        for(int c=0;c<pn.q;++c) trace[c]+=p.shape[j]*src[c];
                        if(p.shape[j]==0.) continue;
                        directional_from_node(cached_pn_flux,face.nodes[j],p.normal,part.data(),pn.q);
                        a_trace+=p.shape[j]*part;
                    }
                    const Vec exterior_trace=b.exterior(trace,data.d);
                    std::vector<double> a_ue(pn.q);
                    pn_apply(patterns,p.normal,exterior_trace.data(),a_ue.data());
                    for(int c=0;c<pn.q;++c)
                        common[c]=.5*(a_trace[c]+a_ue[c])-.5*(exterior_trace[c]-trace[c]);
                    have_common=true;
                }
                std::vector<double> flux(pn.q);
                for(int j=0;j<nd;++j) {
                    const double weight=p.weight*p.shape[j];
                    if(weight==0) continue;
                    const int i=face.nodes[j];
                    const Four exterior=inflow.exterior_state ? inflow.exterior_state(p.x,time,p.normal,low[i]) : shared_exterior;
                    require(exact::cone(exterior),"M1 exterior state outside cone");
                    std::optional<Levermore> node_ext_closure;
                    const Levermore *ext_closure=nullptr;
                    if(inflow.exterior_state) {
                        node_ext_closure.emplace(exterior);
                        ext_closure=&*node_ext_closure;
                    } else ext_closure=&*shared_ext_closure;
                    if(!have_common) nodal_llf(i,b,data.d,p.normal,flux.data());
                    const Four fi=closure_of(i).flux(p.normal),fx=ext_closure->flux(p.normal);
                    const Four lo_flux=.5*(fi+fx)-.5*(exterior-low[i]);
                    Four ghost=.5*(low[i]+exterior)-.5*(fx-fi);
                    if(!exact::cone(ghost)) ghost=stable_base_bar(low[i],exterior,.5*p.normal,.5);
                    cone_require(exact::cone(ghost),"R2 LLF ghost base outside cone");
                    commit_high(j,weight,have_common?common.data():flux.data());
                    hl.row(j)+=weight*lo_flux.transpose();
                    bar.row(j)+=weight*ghost.transpose();
                }
            }
            for(int j=0;j<nd;++j) if(face.mass[j]>0) bar.row(j)/=face.mass[j];
            slots[f].high=std::move(hn);
            slots[f].low=std::move(hl);
            slots[f].bar=std::move(bar);
        });
        for(int f=0;f<nfaces;++f) {
            const auto &face=fe.faces[f];
            face_lazy_certified.push_back(slots[f].certified);
            for(int j=0;j<int(face.nodes.size());++j) {
                rhs.row(face.nodes[j])-=slots[f].high.row(j);
                pn_boundary_rhs.row(face.nodes[j])-=slots[f].high.row(j);
            }
            face_high.push_back(std::move(slots[f].high));
            face_low.push_back(std::move(slots[f].low));
            face_bar.push_back(std::move(slots[f].bar));
        }
    }
    timings.boundary+=seconds()-start;
    start=seconds();
    Field mass_predictor;
    if(pn_mass_correction) {
        start=seconds();
        mass_predictor=pn_low_order_rate(u,pn_boundary_rhs,out.source,&cached_pn_flux);
        timings.pn_predictor+=seconds()-start;
        start=seconds();
        out.target=mass_correction_target(rhs,mass_predictor);
        timings.pn_mass_correction+=seconds()-start;
        diag.mass_residual=0.;
        out.mass_error=Field::Zero(fe.nodes,pn.q);
    } else if (implicit_dt > 0) {
        if (!implicit_solver) {
            implicit_positive=stabilization_operator->positive_matrix();
            if (implicit_positive.rows()!=fe.nodes) {
                std::vector<Triplet> entries;
                Field basis=Field::Zero(fe.nodes,1);
                for (int j=0;j<fe.nodes;++j) {
                    basis.setZero(); basis(j,0)=1.;
                    Field column=stabilization_operator->apply(basis);
                    for (int i=0;i<fe.nodes;++i)
                        if (column(i,0)!=0.) entries.emplace_back(i,j,-column(i,0));
                }
                implicit_positive.resize(fe.nodes,fe.nodes);
                implicit_positive.setFromTriplets(entries.begin(),entries.end());
            }
            implicit_solver=std::make_unique<ImplicitMassSolver>(fe,implicit_positive);
        }
        implicit_result=implicit_solver->solve(rhs,implicit_dt,options.implicit_solver,
                                                options.implicit_rtol,options.implicit_atol,
                                                options.implicit_max_iterations);
        Field new_state=u+implicit_dt*implicit_result.rate;
        stabilization_pairs=Field();
        stabilization_operator->apply(new_state,&stabilization_pairs,options.bh_pair_mode);
        diag.implicit_residual=implicit_result.residual;
        diag.implicit_relative_residual=implicit_result.relative_residual;
        diag.implicit_solves=implicit_result.solves;
        mass_predictor=implicit_result.predictor;
        Field A_rate=implicit_positive*implicit_result.rate;
        Field rhs_full=rhs-implicit_dt*A_rate;
        if (options.implicit_solver=="defect") {
            Field defect=fe.mass.asDiagonal()*mass_predictor-fe.M*mass_predictor;
            out.target=Field::Zero(fe.nodes,pn.q);
            for(int i=0;i<fe.nodes;++i)
                out.target.row(i)=(rhs_full.row(i)+defect.row(i))/fe.mass[i];
            out.mass_error=Field::Zero(fe.nodes,pn.q);
        } else {
            out.target=implicit_result.rate;
            out.mass_error=fe.M*implicit_result.rate-rhs_full;
        }
        diag.mass_residual=maxabs(out.mass_error);
    } else {
        out.target=mass_target(rhs,diag.mass_residual,&mass_predictor);
        if(options.mass_mode=="consistent") out.mass_error=fe.M*out.target-rhs;
        else out.mass_error=Field::Zero(fe.nodes,pn.q);
    }
    timings.mass+=seconds()-start;
    Field high_pairs=Field::Zero(fe.pairs.size(),pn.q);
    out.accepted_pairs=high_pairs;
    out.boundary_gather=Field::Zero(fe.nodes,pn.q);
    Field target_boundary=out.boundary_gather;
    Vec alphas=Vec::Ones(fe.pairs.size());
    Vec pair_margins=Vec::Constant(fe.pairs.size(),std::numeric_limits<double>::infinity());
    int local_failure=0; std::string failure;
    std::atomic<long> lazy_certified{0}, lazy_fallback{0};
    std::vector<unsigned char> certified_pair(fe.pairs.size(),0);
    start=seconds();
    if (!unlimited && options.lazy) {
        // Certify serially and instantiate each closure once only when a face
        // or an uncertified pair will use it. The hot pair loop is lock-free.
        for(std::size_t e=0;e<fe.pairs.size();++e) {
            if(!runtime.owns(e)) continue;
            const auto &p=fe.pairs[e]; V3 cij,cji;
            if(relaxed && !pair_geometry.empty()) { cij=pair_geometry[e][0]; cji=pair_geometry[e][1]; }
            else if(options.geometry_cache) { cij=pair_geometry[e][0]; cji=pair_geometry[e][1]; }
            else for(int d=0;d<3;++d) {cij[d]=fe.C[d].coeff(p.i,p.j);cji[d]=fe.C[d].coeff(p.j,p.i);}
            Vec h;
            if(relaxed) {
                h=pair_streaming(p.i,p.j,cij,cji);
                const double *stab=&stabilization_pairs.coeff(e,0);
                const double *mi=&mass_predictor.coeff(p.i,0);
                const double *mj=&mass_predictor.coeff(p.j,0);
                for(int c=0;c<pn.q;++c) h[c]+=stab[c]+p.mass*(mi[c]-mj[c]);
            } else {
                const Vec ui=u.row(p.i).transpose(),uj=u.row(p.j).transpose();
                (void)ui;(void)uj;
                h=pair_streaming(p.i,p.j,cij,cji)+stabilization_pairs.row(e).transpose()+
                    p.mass*(mass_predictor.row(p.i)-mass_predictor.row(p.j)).transpose();
            }
            high_pairs.row(e)=h.transpose();
            if(lazy_m1::certify_full_pair(low[p.i],low[p.j],cij,cji,p.viscosity,project(h)))
                certified_pair[e]=1;
            else { (void)get_closure(p.i); (void)get_closure(p.j); }
        }
    }
    try {
        parallel_for(int(fe.pairs.size()),[&](int e) {
            if(!runtime.owns(e)) return;
            const auto &p=fe.pairs[e];
            V3 cij,cji;
            if (relaxed && !pair_geometry.empty()) {
                cij=pair_geometry[e][0]; cji=pair_geometry[e][1];
            } else if (options.geometry_cache) {
                cij=pair_geometry[e][0]; cji=pair_geometry[e][1];
            } else for(int d=0;d<3;++d) {cij[d]=fe.C[d].coeff(p.i,p.j);cji[d]=fe.C[d].coeff(p.j,p.i);}
            Vec h;
            if(!unlimited && options.lazy) h=high_pairs.row(e).transpose();
            else if(relaxed) {
                h=pair_streaming(p.i,p.j,cij,cji);
                const double *stab=&stabilization_pairs.coeff(e,0);
                const double *mi=&mass_predictor.coeff(p.i,0);
                const double *mj=&mass_predictor.coeff(p.j,0);
                for(int c=0;c<pn.q;++c) h[c]+=stab[c]+p.mass*(mi[c]-mj[c]);
            } else {
                const Vec ui=u.row(p.i).transpose(),uj=u.row(p.j).transpose();
                (void)ui;(void)uj;
                h=pair_streaming(p.i,p.j,cij,cji)+stabilization_pairs.row(e).transpose()+
                    p.mass*(mass_predictor.row(p.i)-mass_predictor.row(p.j)).transpose();
            }
            if (certified_pair[e]) {
                high_pairs.row(e)=h.transpose();
                out.accepted_pairs.row(e)=h.transpose();
                alphas[e]=1.;
                pair_margins[e]=std::min(margin(low[p.i]),margin(low[p.j]));
                ++lazy_certified;
                return;
            }
            if (!unlimited && options.lazy) ++lazy_fallback;
            // This antisymmetric low pair includes the B_ij correction in (5.5).
            const Four b=p.viscosity*(low[p.j]-low[p.i])-
                         get_closure(p.j).flux(cij)+get_closure(p.i).flux(cji);
            Four left=.5*(low[p.i]+low[p.j])-
                        (get_closure(p.j).flux(cij)-get_closure(p.i).flux(cij))/(2*p.viscosity);
            Four right=.5*(low[p.i]+low[p.j])-
                         (get_closure(p.i).flux(cji)-get_closure(p.j).flux(cji))/(2*p.viscosity);
            if(!exact::cone(left)) left=stable_base_bar(low[p.i],low[p.j],cij,p.viscosity);
            if(!exact::cone(right)) right=stable_base_bar(low[p.j],low[p.i],cji,p.viscosity);
            cone_require(exact::cone(left)&&exact::cone(right),"R2 directed base bars outside cone");
            const Four raw=project(h)-b;
            const Four delta=raw/(2*p.viscosity);
            double alpha=unlimited?1.:std::min(segment(left,delta),segment(right,-delta));
            auto valid=[&](double a) {
                const Four step=(a*raw)/(2*p.viscosity);
                return exact::cone(left+step)&&exact::cone(right-step);
            };
            if(!unlimited && !valid(alpha)) {
                double lo=0,hi=alpha;
                for(int k=0;k<60;++k) {double mid=(lo+hi)/2;if(valid(mid))lo=mid;else hi=mid;}
                alpha=lo;
            }
            if(!unlimited) cone_require(valid(alpha),"R2 corrected directed bars outside cone");
            high_pairs.row(e)=h.transpose();
            set_low(h,b+alpha*raw);
            out.accepted_pairs.row(e)=h.transpose(); alphas[e]=alpha;
            const Four step=(alpha*raw)/(2*p.viscosity);
            pair_margins[e]=std::min(margin(left+step),margin(right-step));
        });
        for(int f=0;f<int(fe.faces.size());++f) if(runtime.owns(f)) {
            const auto &face=fe.faces[f];
            if(face_lazy_certified[f]) {
                for(int j=0;j<int(face.nodes.size());++j) {
                    const int i=face.nodes[j]; Vec h=face_high[f].row(j).transpose();
                    target_boundary.row(i)+=h.transpose();
                    set_low(h,project(h)); out.boundary_gather.row(i)+=h.transpose();
                }
                continue;
            }
            for(int j=0;j<int(face.nodes.size());++j) {
                const double a=face.mass[j]; if(a<=0) continue;
                const int i=face.nodes[j];
                Vec h=face_high[f].row(j).transpose();
                const Four hl=face_low[f].row(j).transpose();
                const Four base=face_bar[f].row(j).transpose();
                cone_require(exact::cone(base),"R2 integrated ghost base outside cone");
                const Four raw=hl-project(h),delta=raw/a;
                double theta=unlimited?1.:segment(base,delta);
                auto valid=[&](double t){return exact::cone(base+(t*raw)/a);};
                if(!unlimited && !valid(theta)) {
                    double lo=0,hi=theta;
                    for(int k=0;k<60;++k){double mid=(lo+hi)/2;if(valid(mid))lo=mid;else hi=mid;}
                    theta=lo;
                }
                if(!unlimited) cone_require(valid(theta),"R2 corrected ghost outside cone");
                target_boundary.row(i)+=h.transpose();
                set_low(h,hl-theta*raw);
                out.boundary_gather.row(i)+=h.transpose();
                diag.minimum_eta=std::min(diag.minimum_eta,theta);
                diag.minimum_bar=std::min(diag.minimum_bar,margin(base+theta*delta));
                diag.active_faces+=theta<1;
            }
        }
    } catch(const StepRejected &error) {local_failure=1;failure=error.what();}
      catch(const std::exception &error) {local_failure=2;failure=error.what();}
    runtime.finish_local_phase(local_failure,failure);
    diag.lazy_certified_pairs += lazy_certified.load();
    diag.lazy_fallback_pairs += lazy_fallback.load();
    timings.mcl+=seconds()-start;
    Field target_gather=gather_pairs(high_pairs)-target_boundary;
    out.gather=gather_pairs(out.accepted_pairs)-out.boundary_gather;
    runtime.sum(target_gather);runtime.sum(out.gather);
    target_gather+=out.source;out.gather+=out.source;
    diag.target_recovery=maxabs(target_gather-fe.mass.asDiagonal()*out.target+out.mass_error);
    require(diag.target_recovery<identity_tolerance()*std::max(1.,maxabs(rhs)),"R2 full residual identity failed");
    out.rate=out.target;
    for(int i=0;i<fe.nodes;++i) {
        Vec v=out.rate.row(i).transpose();
        const Four projected=project(out.gather.row(i).transpose())/fe.mass[i];
        set_low(v,projected);out.rate.row(i)=v.transpose();
        diag.projection=std::max(diag.projection,(project(v)-projected).lpNorm<Eigen::Infinity>());
    }
    Field defect=fe.mass.asDiagonal()*out.rate-out.gather;
    diag.local_balance=maxabs(defect);diag.mass_defect=maxabs(out.mass_error);
    for(int i=0;i<fe.nodes;++i) {
        Vec high_error=out.mass_error.row(i).transpose();set_low(high_error,Four::Zero());
        defect.row(i)-=high_error.transpose();
    }
    require(maxabs(defect)<identity_tolerance()*std::max(1.,maxabs(out.gather)),"R2 all-Q ledger failed");
    if(pn.q>4) diag.high_rate_error=maxabs(out.rate.rightCols(pn.q-4)-out.target.rightCols(pn.q-4));
    const Field mu=fe.M*u;
    diag.target_energy=(mu.array()*out.target.array()).sum();
    diag.sync_work=(mu.array()*(out.rate-out.target).array()).sum();
    diag.minimum_alpha=runtime.reduce(alphas.minCoeff(),-1);
    diag.minimum_eta=runtime.reduce(diag.minimum_eta,-1);
    diag.minimum_bar=runtime.reduce(std::min(diag.minimum_bar,pair_margins.minCoeff()),-1);
    diag.active_pairs=long(runtime.reduce((alphas.array()<1).count(),0));
    diag.active_faces=long(runtime.reduce(diag.active_faces,0));
    diag.closure_evaluations=long(runtime.reduce(
        std::count_if(closure.begin(),closure.end(),[](const auto &c){return bool(c);}),0));
    return out;
}

Stage Transport::evaluate_imex(double time, double dt, const Field &u) {
    require(options.r2,"IMEX evaluation requires the R2 transport path");
    require(dt>0 && std::isfinite(dt),"IMEX step size must be finite and positive");
    return evaluate_r2(time,u,false,Field(),dt);
}
}
