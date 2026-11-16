#include "element_weno.hpp"
#include "base_bar.hpp"
#include "parallel.hpp"
#include <algorithm>
#include <atomic>
#include <string>
#include <iomanip>
#include <sstream>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <Eigen/Eigenvalues>
namespace m1pn {
namespace {
int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}
double binomial(int n, int k) {
    double out = 1;
    for (int i = 1; i <= k; ++i)
        out *= double(n - k + i) / i;
    return out;
}
// r-th derivative of the Bernstein polynomial B_{k,p} at x (x may lie outside [0,1]).
double bernstein(int k, int p, int r, double x) {
    if (k < 0 || k > p)
        return 0;
    if (r > 0) {
        if (p == 0)
            return 0;
        return p * (bernstein(k - 1, p - 1, r - 1, x) - bernstein(k, p - 1, r - 1, x));
    }
    return binomial(p, k) * std::pow(x, k) * std::pow(1 - x, p - k);
}
std::pair<Vec, Vec> unit_gauss(int n) {
    auto [x, w] = gauss(n);
    return {(x.array() + 1).matrix() / 2, w / 2};
}
Mat tensor(const std::vector<Mat> &factors, int p) {
    const int d = factors.size(), one = p + 1;
    int nd = 1;
    for (int k = 0; k < d; ++k)
        nd *= one;
    Mat out(nd, nd);
    for (int a = 0; a < nd; ++a)
        for (int b = 0; b < nd; ++b) {
            double value = 1;
            int x = a, y = b;
            for (int k = 0; k < d; ++k, x /= one, y /= one)
                value *= factors[k](x % one, y % one);
            out(a, b) = value;
        }
    return out;
}
// A rounding fix for stored states: raises only the density, by a few ulps.
std::string describe(const Four &v) {
    std::ostringstream out;
    out << std::setprecision(17) << "(rho=" << v[0] << ", J=" << v[1] << ", " << v[2] << ", " << v[3]
        << ", rho-|J|=" << v[0] - v.tail<3>().norm() << ")";
    return out.str();
}
// |J| rounded upwards, accurate also for subnormal components: below 1e-290 binary64 loses significant bits
// (subnormals), so |J| is evaluated after the exact power-of-two scaling by 2^600 and scaled back upwards.
double norm_up(const Four &v) {
    const double scale = std::ldexp(1., 600);
    const bool tiny = std::max({std::abs(v[1]), std::abs(v[2]), std::abs(v[3])}) < 1e-290;
    const double s = tiny ? scale : 1.;
    const double n = std::sqrt((v[1] * s) * (v[1] * s) + (v[2] * s) * (v[2] * s) + (v[3] * s) * (v[3] * s));
    double up = n * (1 + 8 * std::numeric_limits<double>::epsilon());
    up = tiny ? up / scale : up;
    return std::nextafter(up, std::numeric_limits<double>::infinity());
}
bool nudge_into_cone(Four &v, const char *where = "bar state") {
    if (exact::cone(v))
        return false;
    const Four original = v;
    for (int k = 0; k < 256 && !exact::cone(v); ++k)
        v[0] = std::nextafter(v[0], std::numeric_limits<double>::infinity());
    if (!exact::cone(v) && v[0] < 1e-290) {
        // Subnormal range (e.g. densities ~1e-316 behind a sigma = 1e7 absorber): the rounding error of a few
        // operations exceeds 256 ulp there. Raise rho to |J| rounded upwards (rounding fix, counted by the caller).
        v[0] = std::max(v[0], norm_up(v));
        for (int k = 0; k < 256 && !exact::cone(v); ++k)
            v[0] = std::nextafter(v[0], std::numeric_limits<double>::infinity());
    }
    cone_require(exact::cone(v), std::string("State left the M1 cone beyond rounding (") + where + ", 256 ulp): " +
                                     describe(original));
    return true;
}
inline Four block(const Field &f, int row, int direction) {
    return Four(f(row, 4 * direction), f(row, 4 * direction + 1), f(row, 4 * direction + 2),
                f(row, 4 * direction + 3));
}
// out[l*q + c] += scale * sum_m coef[l*n + m] * rows[m][c], l < count: a small dense matrix acting on whole
// PN rows that stay where they are (no element copies).
inline void apply_rows(int count, int n, int q, const double *coef, double scale, const double *const *rows,
                       double *__restrict out) {
    for (int l = 0; l < count; ++l) {
        double *__restrict o = out + std::size_t(l) * q;
        const double *cl = coef + std::size_t(l) * n;
        int m = 0;
        for (; m + 4 <= n; m += 4) {
            const double c0 = scale * cl[m], c1 = scale * cl[m + 1], c2 = scale * cl[m + 2], c3 = scale * cl[m + 3];
            const double *r0 = rows[m], *r1 = rows[m + 1], *r2 = rows[m + 2], *r3 = rows[m + 3];
#pragma omp simd
            for (int c = 0; c < q; ++c)
                o[c] += c0 * r0[c] + c1 * r1[c] + c2 * r2[c] + c3 * r3[c];
        }
        for (; m < n; ++m) {
            const double c0 = scale * cl[m];
            const double *r0 = rows[m];
#pragma omp simd
            for (int c = 0; c < q; ++c)
                o[c] += c0 * r0[c];
        }
    }
}
// out[j] += scale * (x A)_j for j in [0, count), with `at` = A^T in compressed row storage.
inline void add_flux(const Sparse &at, const double *x, double scale, double *out, int count) {
    const int *outer = at.outerIndexPtr(), *inner = at.innerIndexPtr();
    const double *value = at.valuePtr();
    for (int j = 0; j < count; ++j) {
        double sum = 0;
        for (int k = outer[j]; k < outer[j + 1]; ++k)
            sum += value[k] * x[inner[k]];
        out[j] += scale * sum;
    }
}
// Zero-filled buffers whose pages are first touched by the worker threads. Serial zeroing of node-sized buffers in
// every rate() call did not scale: measured on nx (E = 256, Q1, P7) it grew from 0.11 s (1 thread) to 1.15 s
// (32 threads) and was most of the closure phase.
template <class M> void parallel_fill_zero(M &m) {
    double *p = m.data();
    const std::size_t total = std::size_t(m.size()), chunk = 1 << 15;
    parallel_for(int((total + chunk - 1) / chunk), [&](int k) {
        const std::size_t begin = std::size_t(k) * chunk;
        std::fill(p + begin, p + std::min(total, begin + chunk), 0.);
    });
}
Field parallel_zero(Eigen::Index rows, Eigen::Index cols) {
    Field f(rows, cols);
    parallel_fill_zero(f);
    return f;
}
Vec parallel_zero(Eigen::Index n) {
    Vec v(n);
    parallel_fill_zero(v);
    return v;
}
// out.row(i) += scale * (A^T applied to x.row(i)) for all rows i, A^T in CSR form (`at`, as add_flux). Rows are
// processed in blocks of 8: the block is transposed into a column buffer, so every stored entry becomes one
// contiguous 8-wide multiply-add instead of eight indirect scalar loads. Per output entry the products are summed in
// the same order as add_flux and then scaled and added once, i.e. the same arithmetic.
void add_flux_rows(const Sparse &at, const Field &x, double scale, Field &out) {
    constexpr int B = 8;
    const int rows = int(x.rows()), q = int(x.cols()), count = int(out.cols());
    const int *outer = at.outerIndexPtr(), *inner = at.innerIndexPtr();
    const double *value = at.valuePtr();
    parallel_for((rows + B - 1) / B, [&](int block) {
        const int i0 = block * B, nb = std::min(B, rows - i0);
        thread_local std::vector<double> xin, sum;
        xin.assign(std::size_t(q) * B, 0.);
        sum.resize(B);
        for (int b = 0; b < nb; ++b) {
            const double *src = x.data() + std::size_t(i0 + b) * q;
            for (int k = 0; k < q; ++k)
                xin[std::size_t(k) * B + b] = src[k];
        }
        for (int j = 0; j < count; ++j) {
            double s[B] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int e = outer[j]; e < outer[j + 1]; ++e) {
                const double v = value[e], *col = &xin[std::size_t(inner[e]) * B];
#pragma omp simd
                for (int b = 0; b < B; ++b)
                    s[b] += v * col[b];
            }
            for (int b = 0; b < nb; ++b)
                out(i0 + b, j) += scale * s[b];
        }
    });
}
// A B with the rows of A split over the threads (rows are independent, so the result does not depend on the
// thread count). Used for the batched boundary products (hemisphere traces, PN face fluxes).
Field row_product(const Field &A, const Mat &B) {
    Field out(A.rows(), B.cols());
    if (A.rows() == 0 || B.cols() == 0)
        return out.setZero(), out;
#ifdef _OPENMP
    const int threads = omp_get_max_threads();
#else
    const int threads = 1;
#endif
    const int block = std::max(8, int((A.rows() + 4 * threads - 1) / (4 * threads)));
    const int blocks = (int(A.rows()) + block - 1) / block;
    parallel_for(blocks, [&](int b) {
        const int begin = b * block, n = std::min<int>(block, A.rows() - begin);
        out.middleRows(begin, n).noalias() = A.middleRows(begin, n) * B;
    });
    return out;
}
} // namespace

ElementWeno::ElementWeno(const Angular &angular, const ElementWenoOptions &o, Inflow in)
    : pn(angular), options(o), inflow(std::move(in)), degree(o.degree), dimension(o.dimension),
      elements(o.elements) {
    require(degree >= 1 && degree <= 12, "Element WENO supports Bernstein degrees 1 to 12");
    require(!o.m1_target || (angular.order == 1 && !o.periodic &&
                             std::none_of(o.mirror.begin(), o.mirror.end(), [](bool m) { return m; })),
            "The M1 target needs P1, an open mesh and no mirror sides");
    require(dimension >= 1 && dimension <= 3, "Element WENO supports dimensions 1 to 3");
    require(!o.positivity_projection || o.fullsphere_ghost,
            "The positivity projection replaces the C6 ghost and needs fullsphere_ghost");
    require(!o.positivity_projection || o.projection_rule == 0 || o.projection_rule >= angular.order + 1,
            "projection_rule must be at least N + 1 (moment consistency of the PN trace)");
    for (int d = 0; d < 3; ++d) {
        count[d] = d < o.dimension ? (o.elements_per_axis[d] > 0 ? o.elements_per_axis[d] : o.elements) : 1;
        require(d >= o.dimension || count[d] >= 2, "Element WENO needs at least two elements per direction");
    }
    require(o.elements_per_axis[0] == 0 || o.elements_per_axis[0] == o.elements,
            "elements_per_axis[0] must equal elements");
    require(o.positivity_fraction > 0 && o.positivity_fraction < 1, "epsilon must lie in (0,1)");
    require(o.element_theta > 0 && o.element_theta < 1, "theta must lie in (0,1)");
    require(o.filter_strength >= 0 && std::isfinite(o.filter_strength), "Invalid filter strength");
    require(o.sigma_s >= 0 && o.sigma_a >= 0 && std::isfinite(o.sigma_s) && std::isfinite(o.sigma_a),
            "Invalid scattering or absorption coefficient");
    require(!o.material || (o.sigma_s == 0 && o.sigma_a == 0),
            "Space-dependent material and constant sigma_s/sigma_a are exclusive");
    const int p = degree, one = p + 1;
    require(o.length > 0 && std::isfinite(o.length), "Invalid element WENO domain length");
    h = o.length / elements;
    nodes = cells = local_nodes = 1;
    for (int d = 0; d < dimension; ++d) {
        nodes_per_axis[d] = count[d] * p + (o.periodic ? 0 : 1);
        nodes *= nodes_per_axis[d];
        cells *= count[d];
        local_nodes *= one;
    }
    per_direction = nodes_per_axis[0];
    viscosity = o.viscosity_scale * h / (2. * p);
    // Active moments: parity of every real harmonic from its values at mirrored directions (convention free).
    {
        std::vector<char> keep(pn.q, 1);
        parity_odd_.assign(pn.q, 0);
        if (o.reduce_angular) {
            require(dimension < 3, "Angular parity reduction needs dimension 1 or 2");
            std::vector<double> even_z(pn.q, 0.), odd_z(pn.q, 0.), even_y(pn.q, 0.), odd_y(pn.q, 0.);
            for (const V3 &w : {V3(.37, -.52, .77), V3(-.61, .29, .44), V3(.13, .81, -.35)}) {
                const V3 u = w.normalized();
                const Vec y = harmonics(pn.order, u), mz = harmonics(pn.order, V3(u[0], u[1], -u[2])),
                          my = harmonics(pn.order, V3(u[0], -u[1], u[2]));
                for (int k = 0; k < pn.q; ++k) {
                    even_z[k] += std::abs(y[k] + mz[k]);
                    odd_z[k] += std::abs(y[k] - mz[k]);
                    even_y[k] += std::abs(y[k] + my[k]);
                    odd_y[k] += std::abs(y[k] - my[k]);
                }
            }
            for (int k = 0; k < pn.q; ++k) {
                require(std::min(even_z[k], odd_z[k]) < 1e-12 * std::max(even_z[k], odd_z[k]) &&
                            std::min(even_y[k], odd_y[k]) < 1e-12 * std::max(even_y[k], odd_y[k]),
                        "Spherical harmonic without definite parity");
                const bool z_odd = even_z[k] < odd_z[k], y_odd = even_y[k] < odd_y[k];
                parity_odd_[k] = z_odd || (dimension == 1 && y_odd);
                keep[k] = k < 4 || !parity_odd_[k]; // the M1 moments stay (J_z, and J_y in 1D, are then zero)
            }
        }
        for (int k = 0; k < pn.q; ++k)
            if (keep[k])
                active.push_back(k);
        moments = active.size();
    }
    std::vector<int> slot(pn.q, -1);
    for (int k = 0; k < moments; ++k)
        slot[active[k]] = k;
    // Parity of every active moment under x_d -> -x_d, for the mirror sides.
    for (int d = 0; d < dimension; ++d) {
        require(!(o.mirror[2 * d] && o.mirror[2 * d + 1]), "At most one mirror side per direction");
        if (o.mirror[2 * d] || o.mirror[2 * d + 1]) {
            require(!o.periodic && !o.symmetric_residual, "Mirror sides need an open mesh and the standard residual");
            mirror_index_[d] = o.mirror[2 * d] ? 0 : -2; // the high end is set once per_direction is known
        }
        reflect_odd_[d].assign(moments, 0);
        double even = 0, odd = 0;
        for (const V3 &w : {V3(.37, -.52, .77), V3(-.61, .29, .44), V3(.13, .81, -.35)}) {
            const V3 u = w.normalized();
            V3 m = u;
            m[d] = -m[d];
            const Vec y = harmonics(pn.order, u), r = harmonics(pn.order, m);
            for (int k = 0; k < moments; ++k) {
                const double e = std::abs(y[active[k]] + r[active[k]]), od = std::abs(y[active[k]] - r[active[k]]);
                reflect_odd_[d][k] = reflect_odd_[d][k] || (e < od);
                even = std::max(even, std::min(e, od) / std::max(e, od));
                (void)odd;
            }
        }
        require(even < 1e-12, "Spherical harmonic without definite reflection parity");
    }
    for (int side = 2 * dimension; side < 6; ++side)
        require(!o.mirror[side], "Mirror side beyond the dimension");
    // Restriction of a full q x q matrix to the active moments. Exactness needs that the matrix never couples
    // even and odd moments (odd ones vanish for even data, including the kept odd M1 moments).
    auto restrict_square = [&](const Mat &full, const char *what) {
        Mat out(moments, moments);
        double coupling = 0;
        int ci = -1, cj = -1;
        const double scale = full.cwiseAbs().maxCoeff();
        for (int i = 0; i < pn.q; ++i)
            for (int j = 0; j < pn.q; ++j) {
                if (slot[i] >= 0 && slot[j] >= 0)
                    out(slot[i], slot[j]) = full(i, j);
                if (parity_odd_[i] != parity_odd_[j] && std::abs(full(i, j)) > coupling)
                    coupling = std::abs(full(i, j)), ci = i, cj = j;
            }
        require(coupling <= 1e-12 * std::max(1., scale),
                std::string("Angular reduction: ") + what + " couples even and odd moments (entry " +
                    std::to_string(ci) + "," + std::to_string(cj) + " = " + std::to_string(coupling) + ")");
        return out;
    };

    // One-dimensional reference matrices on [0,1].
    const auto [xq, wq] = unit_gauss(2 * p + 4);
    Mat m1 = Mat::Zero(one, one), c1 = m1, k1 = m1;
    std::vector<Mat> derivative(p + 1, Mat::Zero(one, one));
    Vec value(one), slope(one);
    for (int q = 0; q < xq.size(); ++q) {
        for (int i = 0; i < one; ++i) {
            value[i] = bernstein(i, p, 0, xq[q]);
            slope[i] = bernstein(i, p, 1, xq[q]);
        }
        m1.noalias() += wq[q] * value * value.transpose();
        c1.noalias() += wq[q] * value * slope.transpose();
        k1.noalias() += wq[q] * slope * slope.transpose();
        for (int r = 0; r <= p; ++r) {
            Vec dr(one);
            for (int i = 0; i < one; ++i)
                dr[i] = bernstein(i, p, r, xq[q]);
            derivative[r].noalias() += wq[q] * dr * dr.transpose();
        }
    }
    const double volume = std::pow(h, dimension), area = volume / h;
    std::vector<Mat> factors(dimension, m1);
    mass_ = volume * tensor(factors, p);
    stiffness_ = Mat::Zero(local_nodes, local_nodes);
    for (int d = 0; d < 3; ++d)
        convection_[d] = Mat::Zero(local_nodes, local_nodes);
    for (int d = 0; d < dimension; ++d) {
        auto fc = factors, fk = factors;
        fc[d] = c1;
        fk[d] = k1;
        convection_[d] = area * tensor(fc, p);
        stiffness_ += (area / h) * tensor(fk, p);
    }
    // Scaled Sobolev semi-norm: the h powers of formula (8) cancel on a reference cell.
    seminorm_ = Mat::Zero(local_nodes, local_nodes);
    int multi = 1;
    for (int d = 0; d < dimension; ++d)
        multi *= one;
    for (int a = 0; a < multi; ++a) {
        int total = 0, x = a;
        auto f = factors;
        for (int d = 0; d < dimension; ++d, x /= one) {
            total += x % one;
            f[d] = derivative[x % one];
        }
        if (total >= 1 && total <= p)
            seminorm_ += tensor(f, p);
    }
    // Polynomial extension from a neighbor onto this cell, in Bernstein coefficients.
    Eigen::LDLT<Mat> m1_solve(m1);
    for (int side = 0; side < 2; ++side) {
        const double shift = side ? 1 : -1;
        Mat load = Mat::Zero(one, one);
        for (int q = 0; q < xq.size(); ++q)
            for (int i = 0; i < one; ++i)
                for (int j = 0; j < one; ++j)
                    load(i, j) += wq[q] * bernstein(i, p, 0, xq[q]) * bernstein(j, p, 0, xq[q] - shift);
        const Mat e1 = m1_solve.solve(load);
        for (int d = 0; d < dimension; ++d) {
            std::vector<Mat> f(dimension, Mat::Identity(one, one));
            f[d] = e1;
            extension_[d][side] = tensor(f, p);
        }
    }
    // Connectivity and von Neumann neighbors.
    connectivity_.resize(cells * local_nodes);
    neighbors_.assign(cells * 2 * dimension, -1);
    std::array<int, 3> powers{1, count[0], count[0] * count[1]};
    for (int e = 0; e < cells; ++e) {
        std::array<int, 3> id{e % count[0], (e / count[0]) % count[1], e / (count[0] * count[1])};
        for (int l = 0; l < local_nodes; ++l) {
            int x = l, global = 0, scale = 1;
            for (int d = 0; d < dimension; ++d) {
                int coordinate = id[d] * p + x % one;
                if (o.periodic)
                    coordinate %= nodes_per_axis[d];
                global += coordinate * scale;
                x /= one;
                scale *= nodes_per_axis[d];
            }
            connectivity_[e * local_nodes + l] = global;
        }
        for (int d = 0; d < dimension; ++d)
            for (int side = 0; side < 2; ++side) {
                const int next = id[d] + (side ? 1 : -1);
                if (next >= 0 && next < count[d])
                    neighbors_[e * 2 * dimension + 2 * d + side] = e + (next - id[d]) * powers[d];
                else if (o.periodic)
                    neighbors_[e * 2 * dimension + 2 * d + side] =
                        e + ((next + count[d]) % count[d] - id[d]) * powers[d];
                else if (o.mirror[2 * d + side])
                    neighbors_[e * 2 * dimension + 2 * d + side] = -2; // the mirrored element itself
                else if (!o.cut[2 * d + side])
                    faces_.push_back({e, d, side});
            }
    }
    // Element graph viscosity (report (10)), outward-rounded so |c|/d <= 1 is exact.
    viscosity_ij_ = Mat::Zero(local_nodes, local_nodes);
    for (int i = 0; i < local_nodes; ++i)
        for (int j = 0; j < local_nodes; ++j) {
            if (i == j)
                continue;
            V3 cij = V3::Zero(), cji = V3::Zero();
            for (int d = 0; d < dimension; ++d)
                cij[d] = convection_[d](i, j), cji[d] = convection_[d](j, i);
            double value_ij = std::max(cij.norm(), cji.norm());
            while (value_ij > 0 && (!exact::cone(Four(value_ij, cij[0], cij[1], cij[2])) ||
                                    !exact::cone(Four(value_ij, cji[0], cji[1], cji[2]))))
                value_ij = std::nextafter(value_ij, std::numeric_limits<double>::infinity());
            viscosity_ij_(i, j) = value_ij;
        }
    viscosity_ij_ = viscosity_ij_.cwiseMax(Mat(viscosity_ij_.transpose()));
    // Pair table for the limiter loop: (j, d_ij, m_ij, c_ij, c_ji) for all j != i, row by row, same values as
    // viscosity_ij_, mass_ and convection_ (contiguous instead of dynamic-matrix lookups).
    pair_table_.resize(std::size_t(local_nodes) * (local_nodes - 1));
    for (int i = 0, k = 0; i < local_nodes; ++i)
        for (int j = 0; j < local_nodes; ++j) {
            if (j == i)
                continue;
            PairEntry &pe = pair_table_[k++];
            pe.j = j;
            pe.d = viscosity_ij_(i, j);
            pe.m = mass_(i, j);
            for (int d = 0; d < 3; ++d) {
                pe.c[d] = d < dimension ? convection_[d](i, j) : 0.;
                pe.ct[d] = d < dimension ? convection_[d](j, i) : 0.;
            }
        }
    // Lumped masses, one-dimensional consistent mass factor, boundary rules.
    lumped = Vec::Zero(nodes);
    const Vec row = mass_.rowwise().sum();
    for (int e = 0; e < cells; ++e)
        for (int l = 0; l < local_nodes; ++l)
            lumped[connectivity_[e * local_nodes + l]] += row[l];
    // Assembled one-dimensional mass and convection matrices per axis (they differ only in the element count).
    std::array<Eigen::SparseMatrix<double>, 3> m1d;
    for (int d = 0; d < dimension; ++d) {
        const int n1 = nodes_per_axis[d];
        std::vector<Eigen::Triplet<double>> triplets, entries;
        for (int e = 0; e < count[d]; ++e)
            for (int i = 0; i < one; ++i)
                for (int j = 0; j < one; ++j) {
                    int a = e * p + i, b = e * p + j;
                    if (o.periodic)
                        a %= n1, b %= n1;
                    triplets.emplace_back(a, b, h * m1(i, j));
                    m1_reference_ = m1;
                    entries.emplace_back(a, b, c1(i, j));
                }
        m1d[d].resize(n1, n1);
        m1d[d].setFromTriplets(triplets.begin(), triplets.end());
        mass_1d_[d] = m1d[d];
        mass_1d_[d].makeCompressed();
        // On the tensor mesh M^-1 C_d = I x (M_1d^-1 C_1d).
        convection_1d_[d].resize(n1, n1);
        convection_1d_[d].setFromTriplets(entries.begin(), entries.end());
        convection_1d_[d].makeCompressed();
    }
    stiffness_rows_.resize(std::size_t(local_nodes) * local_nodes);
    for (int d = 0; d < dimension; ++d)
        convection_t_rows_[d].resize(std::size_t(local_nodes) * local_nodes);
    for (int l = 0; l < local_nodes; ++l)
        for (int m = 0; m < local_nodes; ++m) {
            stiffness_rows_[std::size_t(l) * local_nodes + m] = stiffness_(l, m);
            for (int d = 0; d < dimension; ++d)
                convection_t_rows_[d][std::size_t(l) * local_nodes + m] = convection_[d](m, l);
        }
    for (int d = 0; d < dimension; ++d) {
        flux_t_[d] = Mat(restrict_square(Mat(pn.A[d]), "A_d").transpose()).sparseView();
        flux_t_[d].makeCompressed();
    }
    for (int d = 0; d < dimension; ++d) {
        const int n1 = nodes_per_axis[d];
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> mass_1d(m1d[d]);
        require(mass_1d.info() == Eigen::Success, "Bernstein mass factorization failed");
        Eigen::SparseMatrix<double> L = mass_1d.matrixL();
        L.makeCompressed();
        LineFactor &f = natural_[d];
        f.n = n1;
        f.outer.assign(L.outerIndexPtr(), L.outerIndexPtr() + n1 + 1);
        f.inner.assign(L.innerIndexPtr(), L.innerIndexPtr() + L.nonZeros());
        f.values.assign(L.valuePtr(), L.valuePtr() + L.nonZeros());
        const Vec diagonal = mass_1d.vectorD(); // returned by value
        f.diagonal.assign(diagonal.data(), diagonal.data() + n1);
        f.perm.resize(n1);
        f.line.resize(n1);
        for (int k = 0; k < n1; ++k) {
            f.perm[k] = mass_1d.permutationP().indices()[k];
            f.line[k] = k;
        }
    }
    // Mirror directions: the same 1D mass without the row and column of the mirror node (Dirichlet rows for the
    // components that are odd under the reflection), the mirror nodes and the mirrored local numbering.
    for (int d = 0; d < dimension; ++d) {
        reflected_local_[d].resize(local_nodes);
        for (int l = 0; l < local_nodes; ++l) {
            int x = l, mirrored = 0, scale = 1;
            for (int k = 0; k < dimension; ++k, x /= one, scale *= one)
                mirrored += (k == d ? p - x % one : x % one) * scale;
            reflected_local_[d][l] = mirrored;
        }
        if (mirror_index_[d] == -1)
            continue;
        if (mirror_index_[d] == -2)
            mirror_index_[d] = nodes_per_axis[d] - 1;
        const int removed = mirror_index_[d];
        std::vector<int> keep;
        for (int k = 0; k < nodes_per_axis[d]; ++k)
            if (k != removed)
                keep.push_back(k);
        const Mat dense = Mat(m1d[d]);
        Eigen::SparseMatrix<double> reduced_mass(keep.size(), keep.size());
        {
            std::vector<Eigen::Triplet<double>> entries;
            for (std::size_t a = 0; a < keep.size(); ++a)
                for (std::size_t b = 0; b < keep.size(); ++b)
                    if (dense(keep[a], keep[b]) != 0)
                        entries.emplace_back(a, b, dense(keep[a], keep[b]));
            reduced_mass.setFromTriplets(entries.begin(), entries.end());
        }
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factor(reduced_mass);
        require(factor.info() == Eigen::Success, "Dirichlet Bernstein mass factorization failed");
        Eigen::SparseMatrix<double> L = factor.matrixL();
        L.makeCompressed();
        LineFactor &f = dirichlet_[d];
        f.n = keep.size();
        f.outer.assign(L.outerIndexPtr(), L.outerIndexPtr() + f.n + 1);
        f.inner.assign(L.innerIndexPtr(), L.innerIndexPtr() + L.nonZeros());
        f.values.assign(L.valuePtr(), L.valuePtr() + L.nonZeros());
        const Vec diagonal = factor.vectorD();
        f.diagonal.assign(diagonal.data(), diagonal.data() + f.n);
        f.perm.resize(f.n);
        f.line = keep;
        for (int k = 0; k < f.n; ++k)
            f.perm[k] = factor.permutationP().indices()[k];
        for (int i = 0; i < nodes; ++i)
            if ((i / stride(d)) % nodes_per_axis[d] == removed)
                mirror_nodes_[d].push_back(i);
    }
    // Nodal reaction data: positive averages int c phi_i / int phi_i by composite Gauss quadrature.
    node_sigma_a = node_sigma_s = node_source = Vec::Zero(nodes);
    if (o.material) {
        const ElementWenoMaterial &mat = *o.material;
        require(mat.subcells >= 1 && mat.points >= 0, "Invalid material quadrature");
        const auto [gx, gw] = unit_gauss(mat.points > 0 ? mat.points : p + 4);
        const int sub = mat.subcells, per = gx.size() * sub;
        Vec x1(per), w1(per);
        for (int c = 0; c < sub; ++c)
            for (int k = 0; k < gx.size(); ++k) {
                x1[c * gx.size() + k] = (c + gx[k]) / sub;
                w1[c * gx.size() + k] = gw[k] / sub;
            }
        Mat shape1(one, per); // 1D Bernstein values at the composite points
        for (int k = 0; k < per; ++k)
            for (int i = 0; i < one; ++i)
                shape1(i, k) = bernstein(i, p, 0, x1[k]);
        int points = 1;
        for (int d = 0; d < dimension; ++d)
            points *= per;
        const double volume = std::pow(h, dimension);
        // Element by element in parallel into per-element buffers, then gathered in element order (deterministic,
        // independent of the thread count).
        std::vector<double> local(std::size_t(cells) * local_nodes * 4, 0.), element_integrals(std::size_t(cells) * 4, 0.);
        parallel_for(cells, [&](int e) {
            V3 origin;
            element_origin(e, origin);
            double *loc = &local[std::size_t(e) * local_nodes * 4], *ints = &element_integrals[std::size_t(e) * 4];
            for (int point = 0; point < points; ++point) {
                V3 x = origin;
                double w = volume;
                std::array<int, 3> idx{0, 0, 0};
                for (int d = 0, k = point; d < dimension; ++d, k /= per) {
                    idx[d] = k % per;
                    x[d] += h * x1[idx[d]];
                    w *= w1[idx[d]];
                }
                const double ca = mat.sigma_a ? mat.sigma_a(x) : 0., cs = mat.sigma_s ? mat.sigma_s(x) : 0.,
                             cq = mat.source ? mat.source(x) : 0.;
                require(ca >= 0 && cs >= 0 && cq >= 0 && std::isfinite(ca + cs + cq),
                        "Material coefficients must be finite and nonnegative");
                ints[0] += w * ca, ints[1] += w * cs, ints[2] += w * cq, ints[3] += w;
                for (int l = 0; l < local_nodes; ++l) {
                    double phi = 1;
                    for (int d = 0, v = l; d < dimension; ++d, v /= one)
                        phi *= shape1(v % one, idx[d]);
                    double *c = loc + 4 * l;
                    c[0] += w * phi * ca, c[1] += w * phi * cs, c[2] += w * phi * cq, c[3] += w * phi;
                }
            }
        });
        std::vector<double> sa(nodes, 0.), ss(nodes, 0.), sq(nodes, 0.), sm(nodes, 0.);
        double ia = 0, is = 0, iq = 0, iv = 0;
        for (int e = 0; e < cells; ++e) {
            const double *ints = &element_integrals[std::size_t(e) * 4];
            ia += ints[0], is += ints[1], iq += ints[2], iv += ints[3];
            for (int l = 0; l < local_nodes; ++l) {
                const int i = connectivity_[e * local_nodes + l];
                const double *c = &local[(std::size_t(e) * local_nodes + l) * 4];
                sa[i] += c[0], ss[i] += c[1], sq[i] += c[2], sm[i] += c[3];
            }
        }
        for (int i = 0; i < nodes; ++i) {
            node_sigma_a[i] = sa[i] / sm[i];
            node_sigma_s[i] = ss[i] / sm[i];
            node_source[i] = sq[i] / sm[i];
        }
        integral_sigma_a = ia, integral_sigma_s = is, integral_source = iq, integral_volume = iv;
    }
    if (o.uncollided_density) {
        // FCS quadrature (docs/FCS.md): tensor Gauss rule per element, sigma_s fixed per point.
        require(o.uncollided_points >= 0, "Invalid FCS quadrature");
        const auto [gx, gw] = unit_gauss(o.uncollided_points > 0 ? o.uncollided_points : p + 3);
        const int per = gx.size();
        int points = 1;
        for (int d = 0; d < dimension; ++d)
            points *= per;
        fcs_per_element_ = points;
        fcs_shape_.resize(local_nodes, points);
        std::vector<double> wref(points);
        for (int k = 0; k < points; ++k) {
            double w = std::pow(h, dimension);
            std::array<int, 3> idx{0, 0, 0};
            for (int d = 0, r = k; d < dimension; ++d, r /= per)
                idx[d] = r % per, w *= gw[idx[d]];
            wref[k] = w;
            for (int l = 0; l < local_nodes; ++l) {
                double phi = 1;
                for (int d = 0, v = l; d < dimension; ++d, v /= one)
                    phi *= bernstein(v % one, p, 0, gx[idx[d]]);
                fcs_shape_(l, k) = phi;
            }
        }
        fcs_points_.resize(std::size_t(cells) * points);
        fcs_sigma_s_.resize(fcs_points_.size());
        fcs_weight_.resize(fcs_points_.size());
        parallel_for(cells, [&](int e) {
            V3 origin;
            element_origin(e, origin);
            for (int k = 0; k < points; ++k) {
                V3 x = origin;
                for (int d = 0, r = k; d < dimension; ++d, r /= per)
                    x[d] += h * gx[r % per];
                const std::size_t at = std::size_t(e) * points + k;
                fcs_points_[at] = x;
                fcs_weight_[at] = wref[k];
                const double ss = (o.material && o.material->sigma_s ? o.material->sigma_s(x) : 0.) + o.sigma_s;
                require(ss >= 0 && std::isfinite(ss), "FCS: sigma_s must be finite and nonnegative");
                fcs_sigma_s_[at] = ss;
            }
        });
        fcs_node_mass_ = Vec::Zero(nodes);
        for (int e = 0; e < cells; ++e)
            for (int l = 0; l < local_nodes; ++l)
                for (int k = 0; k < points; ++k)
                    fcs_node_mass_[connectivity_[e * local_nodes + l]] += wref[k] * fcs_shape_(l, k);
    }
    if (!o.periodic) {
        const auto [fx, fw] = unit_gauss(p + pn.order + 3);
        const int nfp = dimension == 1 ? 1 : (dimension == 2 ? fx.size() : fx.size() * fx.size());
        const int nfn = dimension == 1 ? 1 : (dimension == 2 ? one : one * one);
        for (int d = 0; d < dimension; ++d)
            for (int side = 0; side < 2; ++side) {
                FaceRule &rule = rules_[2 * d + side];
                for (int l = 0; l < local_nodes; ++l) {
                    int x = l;
                    for (int k = 0; k < d; ++k)
                        x /= one;
                    if (x % one == (side ? p : 0))
                        rule.local.push_back(l);
                }
                require(int(rule.local.size()) == nfn, "Inconsistent Bernstein face");
                rule.shape.resize(nfn, nfp);
                rule.points.resize(nfp, dimension);
                rule.weights.resize(nfp);
                for (int q = 0; q < nfp; ++q) {
                    int a = q;
                    double weight = 1;
                    Vec ref(dimension);
                    for (int k = 0; k < dimension; ++k) {
                        if (k == d) {
                            ref[k] = side;
                            continue;
                        }
                        ref[k] = fx[a % fx.size()];
                        weight *= fw[a % fx.size()];
                        a /= fx.size();
                    }
                    rule.points.row(q) = ref.transpose();
                    rule.weights[q] = weight * area;
                    for (int n = 0; n < nfn; ++n) {
                        int x = rule.local[n];
                        double value_n = 1;
                        for (int k = 0; k < dimension; ++k, x /= one)
                            value_n *= bernstein(x % one, p, 0, ref[k]);
                        rule.shape(n, q) = value_n;
                    }
                }
                rule.mass = rule.shape * rule.weights.asDiagonal() * rule.shape.transpose();
            }
        boundary_slot_.assign(nodes, -1);
        ghost_of_face_.reserve(faces_.size() * nfn);
        std::array<std::vector<int>, 6> ghost_slot;
        for (const auto &f : faces_) {
            const int side_id = 2 * f.direction + f.side;
            if (ghost_slot[side_id].empty())
                ghost_slot[side_id].assign(nodes, -1);
            for (int l : rules_[side_id].local) {
                const int node = connectivity_[f.element * local_nodes + l];
                if (boundary_slot_[node] < 0) {
                    boundary_slot_[node] = boundary_nodes_.size();
                    boundary_nodes_.push_back(node);
                }
                if (ghost_slot[side_id][node] < 0) {
                    ghost_slot[side_id][node] = ghost_node_.size();
                    ghost_node_.push_back(node);
                    ghost_side_.push_back(side_id);
                }
                ghost_of_face_.push_back(ghost_slot[side_id][node]);
            }
        }
        for (std::size_t k = 0; k < ghost_node_.size(); ++k)
            ghost_by_side_[ghost_side_[k]].push_back(int(k));
        // Face rows (add_boundary): the rows of face f are face_row_[f] + n, n < nf; the faces of one side are
        // contiguous (rows side_row_start_[s] ..), in ascending face index. boundary_rows_ lists, per boundary slot,
        // the rows of its node in ascending face index (the order of the former serial reduction).
        face_row_.assign(faces_.size(), 0);
        side_row_start_.fill(0);
        {
            int row = 0;
            for (int side = 0; side < 6; ++side) {
                side_row_start_[side] = row;
                for (std::size_t index = 0; index < faces_.size(); ++index)
                    if (2 * faces_[index].direction + faces_[index].side == side) {
                        face_row_[index] = row;
                        row += int(rules_[side].local.size());
                    }
            }
            side_row_start_[6] = row;
            face_row_side_.assign(row, 0);
            std::vector<std::vector<int>> per_slot(boundary_nodes_.size());
            for (std::size_t index = 0; index < faces_.size(); ++index) {
                const auto &f = faces_[index];
                const int side_id = 2 * f.direction + f.side;
                const auto &local = rules_[side_id].local;
                for (std::size_t n = 0; n < local.size(); ++n) {
                    const int node = connectivity_[f.element * local_nodes + local[n]];
                    per_slot[boundary_slot_[node]].push_back(face_row_[index] + int(n));
                    face_row_side_[face_row_[index] + n] = side_id;
                }
            }
            boundary_row_offsets_.assign(boundary_nodes_.size() + 1, 0);
            for (std::size_t k = 0; k < per_slot.size(); ++k)
                boundary_row_offsets_[k + 1] = boundary_row_offsets_[k] + int(per_slot[k].size());
            boundary_rows_.reserve(boundary_row_offsets_.back());
            for (const auto &list : per_slot)
                boundary_rows_.insert(boundary_rows_.end(), list.begin(), list.end());
        }
        if (options.positivity_projection)
            projection_warm_.assign(ghost_node_.size(), PositivityProjectionWarm{});
        if (options.ghost_filter > 0) {
            ghost_sigma_.resize(moments);
            for (int c = 0; c < moments; ++c) {
                const int l = int(std::sqrt(double(active[c]) + .5));
                ghost_sigma_[c] = l <= 1 ? 1. : std::exp(-options.ghost_filter * std::pow(l / (angular.order + 1.), 4));
            }
        }
        canonical_ = std::make_unique<CanonicalC1>(pn);
        for (int d = 0; d < dimension; ++d)
            for (int side = 0; side < 2; ++side) {
                V3 normal = V3::Zero();
                normal[d] = side ? 1 : -1;
                boundaries_[2 * d + side] = std::make_unique<Boundary>(pn, *canonical_, normal, false, false);
                // Face integrand .5 (A_n - I)(u - exterior(u, datum)) = (A_n - llf_homogeneous) u - llf_datum datum.
                boundary_flux_t_[2 * d + side] =
                    restrict_square(pn.normal(normal) - boundaries_[2 * d + side]->llf_homogeneous, "boundary flux").transpose();
                if (options.projection_high_order)
                    exterior_t_[2 * d + side] =
                        restrict_square(boundaries_[2 * d + side]->llf_homogeneous -
                                            .5 * (pn.normal(normal) + Mat::Identity(pn.q, pn.q)),
                                        "exterior flux")
                            .transpose();
                {
                    const Mat &datum = boundaries_[2 * d + side]->llf_datum;
                    datum_t_[2 * d + side].resize(datum.cols(), moments);
                    for (int k = 0; k < moments; ++k)
                        datum_t_[2 * d + side].col(k) = datum.row(active[k]).transpose();
                }
                // Outgoing hemisphere rule (local z in (0,1)), rotated to the global frame: C6 ghost state.
                OutRule &out = out_rules_[2 * d + side];
                // Positivity projection: full sphere = outgoing rows first, then the incoming hemisphere.
                // C6: 2N+1. Projection: N+1 by default (measured: ghost moments and solution unchanged to 3-4 digits
                // against 2N+1 and 4N+3, docs/POSITIVITY_PROJECTION.md), 2.5-3x cheaper.
                const int rule_n = !options.positivity_projection ? 2 * pn.order + 1
                                   : options.projection_rule > 0  ? options.projection_rule
                                                                  : pn.order + 1;
                auto points = sphere_rule(rule_n, +1);
                out.outgoing = int(points.size());
                if (options.positivity_projection && !options.projection_half_range) {
                    const auto incoming = sphere_rule(rule_n, -1);
                    points.insert(points.end(), incoming.begin(), incoming.end());
                }
                out.omega.resize(points.size(), 3);
                out.weight.resize(points.size());
                out.harmonics.resize(points.size(), moments);
                for (std::size_t k = 0; k < points.size(); ++k) {
                    const V3 w = boundaries_[2 * d + side]->frame * points[k].omega;
                    out.omega.row(k) = w.transpose();
                    out.weight[k] = points[k].weight;
                    const Vec y = harmonics(pn.order, w);
                    for (int c = 0; c < moments; ++c)
                        out.harmonics(k, c) = y[active[c]];
                }
                out.ox = out.omega.col(0);
                out.oy = out.omega.col(1);
                out.oz = out.omega.col(2);
                std::vector<int> hemispheres{+1};
                if (options.positivity_projection && !options.projection_half_range)
                    hemispheres.push_back(-1);
                separable_[2 * d + side] = std::make_shared<SeparableTrace>(
                    pn.order, active, boundaries_[2 * d + side]->frame, rule_n, hemispheres);
                if (options.positivity_projection) {
                    std::vector<double> rings;
                    for (int h : hemispheres) {
                        const auto local = sphere_rule(rule_n, h);
                        for (std::size_t k = 0; k < local.size(); k += 2 * rule_n + 4)
                            rings.push_back(local[k].omega[2]);
                    }
                    out.facets = lat_long_facets(rings, 2 * rule_n + 4, boundaries_[2 * d + side]->frame);
                    if (options.projection_incoming_weight != 1 && !options.projection_half_range) {
                        require(options.projection_incoming_weight > 0, "projection_incoming_weight must be positive");
                        const int all = int(out.weight.size());
                        out.c = Vec::Ones(all);
                        out.c.tail(all - out.outgoing).setConstant(options.projection_incoming_weight);
                        out.cinv = out.c.cwiseInverse();
                    }
                    require(out.facets.count() > 0, "Positivity projection: hull facets of the rule failed verification");
                }
                out_harmonics_[2 * d + side] = out.harmonics;
            }
    }
    // Forward Euler bound of report (19), including the boundary bar states.
    Vec rates = Vec::Zero(nodes);
    const Vec sum = viscosity_ij_.rowwise().sum();
    for (int e = 0; e < cells; ++e)
        for (int l = 0; l < local_nodes; ++l)
            rates[connectivity_[e * local_nodes + l]] += 2 * sum[l];
    for (const auto &f : faces_) {
        const auto &rule = rules_[2 * f.direction + f.side];
        for (std::size_t n = 0; n < rule.local.size(); ++n)
            rates[connectivity_[f.element * local_nodes + rule.local[n]]] +=
                (rule.shape.row(n).transpose().array() * rule.weights.array()).sum();
    }
    node_dt_bound = lumped.array() / rates.array();
    dt_bound = node_dt_bound.minCoeff();
    total_viscosity_ = viscosity_ij_.rowwise().sum();
    node_offsets_.assign(nodes + 1, 0);
    for (int entry = 0; entry < cells * local_nodes; ++entry)
        ++node_offsets_[connectivity_[entry] + 1];
    for (int i = 0; i < nodes; ++i)
        node_offsets_[i + 1] += node_offsets_[i];
    node_entries_.resize(cells * local_nodes);
    {
        std::vector<int> fill(node_offsets_.begin(), node_offsets_.end() - 1);
        for (int entry = 0; entry < cells * local_nodes; ++entry)
            node_entries_[fill[connectivity_[entry]]++] = entry;
    }
    // A constant inflow density gives the same angular data at every boundary
    // point of one normal; evaluate that part once instead of per point and stage.
    if (!o.periodic && inflow.density && inflow.constant_density) {
        Inflow density_only;
        density_only.density = inflow.density;
        for (int k = 0; k < 2 * dimension; ++k)
            density_inflow_[k] = boundaries_[k]->data(density_only, V3::Zero(), 0.);
        cached_density_ = true;
    }
    if (!o.periodic && inflow.steady && !faces_.empty()) {
        const std::size_t points = rules_[2 * faces_[0].direction + faces_[0].side].weights.size();
        steady_data_.resize(faces_.size() * points);
        parallel_for(int(faces_.size()), [&](int index) {
            const auto &f = faces_[index];
            const int side_id = 2 * f.direction + f.side;
            const FaceRule &rule = rules_[side_id];
            require(std::size_t(rule.weights.size()) == points, "Face rules of different size");
            V3 origin;
            element_origin(f.element, origin);
            for (std::size_t point = 0; point < points; ++point) {
                V3 x = origin;
                for (int d = 0; d < dimension; ++d)
                    x[d] += h * rule.points(point, d);
                steady_data_[index * points + point] = face_data(side_id, x, 0.);
            }
        });
    }
    if (!o.periodic) {
        boundary_nodes_sorted_ = boundary_nodes_;
        std::sort(boundary_nodes_sorted_.begin(), boundary_nodes_sorted_.end());
    }
}

int ElementWeno::stride(int direction) const {
    int s = 1;
    for (int d = 0; d < direction; ++d)
        s *= nodes_per_axis[d];
    return s;
}
void ElementWeno::element_origin(int element, V3 &origin) const {
    origin = V3::Zero();
    int x = element;
    for (int d = 0; d < dimension; ++d) {
        origin[d] = options.origin[d] + (options.element_offset[d] + x % count[d]) * h;
        x /= count[d];
    }
}
V3 ElementWeno::node_position(int node) const {
    V3 x = V3::Zero();
    int value = node;
    for (int d = 0; d < dimension; ++d) {
        x[d] = options.origin[d] + (options.element_offset[d] * degree + value % nodes_per_axis[d]) * h / degree;
        value /= nodes_per_axis[d];
    }
    return x;
}
// P^T L D L^T P x = b for one grid line, applied to whole rows (all PN components) at once.
// On entry work[perm[k]] holds b_k, on exit x_k.
void ElementWeno::solve_line(const LineFactor &f, double *work, int q) const {
    for (int j = 0; j < f.n; ++j) {
        const double *yj = &work[std::size_t(j) * q];
        for (int k = f.outer[j]; k < f.outer[j + 1]; ++k) {
            const int i = f.inner[k];
            if (i == j) continue;
            const double l = f.values[k];
            double *yi = &work[std::size_t(i) * q];
            for (int c = 0; c < q; ++c) yi[c] -= l * yj[c];
        }
    }
    for (int j = 0; j < f.n; ++j) {
        const double inv = 1. / f.diagonal[j];
        double *yj = &work[std::size_t(j) * q];
        for (int c = 0; c < q; ++c) yj[c] *= inv;
    }
    for (int j = f.n - 1; j >= 0; --j) {
        double *yj = &work[std::size_t(j) * q];
        for (int k = f.outer[j]; k < f.outer[j + 1]; ++k) {
            const int i = f.inner[k];
            if (i == j) continue;
            const double l = f.values[k];
            const double *yi = &work[std::size_t(i) * q];
            for (int c = 0; c < q; ++c) yj[c] -= l * yi[c];
        }
    }
}
void ElementWeno::set_distributed(const DistributedHooks *hooks, std::vector<char> owned) {
    require(!hooks || (int(owned.size()) == nodes && !options.periodic && !options.symmetric_residual &&
                       std::none_of(mirror_index_.begin(), mirror_index_.end(), [](int m) { return m >= 0; })),
            "Distributed element WENO: open mesh without mirror sides and with the standard residual");
    dist_ = hooks;
    owned_ = std::move(owned);
    spectral_radius_ = -1;
    if (dist_) {
        double m = std::numeric_limits<double>::infinity();
        for (int i = 0; i < nodes; ++i)
            if (owned_[i])
                m = std::min(m, node_dt_bound[i]);
        dt_bound = dist_->min(m);
    }
}
void ElementWeno::distributed_random(Field &x, std::mt19937 &rng, std::uniform_real_distribution<double> &u) const {
    // The serial start vector draws node by node in global order; skip the draws of the nodes before this slab.
    const long skip = dist_->node_offset * long(moments);
    for (long k = 0; k < skip; ++k)
        u(rng);
    for (int i = 0; i < nodes; ++i)
        for (int c = 0; c < moments; ++c)
            x(i, c) = u(rng);
    const long rest = (dist_->global_nodes - dist_->node_offset - nodes) * long(moments);
    for (long k = 0; k < rest; ++k)
        u(rng);
}
void ElementWeno::solve_lines(Field &b, int d, bool gradient) const {
    if (dist_ && d == dist_->axis) {
        dist_->solve_axis(b);
        return;
    }
    const int q = b.cols(), s = stride(d), per_direction = nodes_per_axis[d], lines = nodes / per_direction;
    const LineFactor &natural = natural_[d];
    if (mirror_index_[d] < 0 || q != moments) {
        parallel_for(lines, [&](int line) {
            const int start = (line / s) * (s * per_direction) + line % s;
            thread_local std::vector<double> work;
            work.resize(std::size_t(per_direction) * q);
            for (int k = 0; k < per_direction; ++k)
                std::copy_n(&b.coeffRef(start + k * s, 0), q, &work[std::size_t(natural.perm[k]) * q]);
            solve_line(natural, work.data(), q);
            for (int k = 0; k < per_direction; ++k)
                std::copy_n(&work[std::size_t(natural.perm[k]) * q], q, &b.coeffRef(start + k * s, 0));
        });
        return;
    }
    std::array<std::vector<int>, 2> groups; // [0] natural (even), [1] Dirichlet at the mirror node (odd)
    for (int c = 0; c < q; ++c)
        groups[bool(reflect_odd_[d][c]) != gradient].push_back(c);
    parallel_for(lines, [&](int line) {
        const int start = (line / s) * (s * per_direction) + line % s;
        thread_local std::vector<double> work;
        for (int g = 0; g < 2; ++g) {
            const std::vector<int> &comps = groups[g];
            const int width = comps.size();
            if (width == 0)
                continue;
            const LineFactor &f = g ? dirichlet_[d] : natural;
            work.resize(std::size_t(f.n) * width);
            for (int r = 0; r < f.n; ++r) {
                const double *row = &b.coeffRef(start + f.line[r] * s, 0);
                double *w = &work[std::size_t(f.perm[r]) * width];
                for (int k = 0; k < width; ++k)
                    w[k] = row[comps[k]];
            }
            solve_line(f, work.data(), width);
            for (int r = 0; r < f.n; ++r) {
                double *row = &b.coeffRef(start + f.line[r] * s, 0);
                const double *w = &work[std::size_t(f.perm[r]) * width];
                for (int k = 0; k < width; ++k)
                    row[comps[k]] = w[k];
            }
            if (g) {
                double *row = &b.coeffRef(start + mirror_index_[d] * s, 0);
                for (int c : comps)
                    row[c] = 0;
            }
        }
    });
}
Field ElementWeno::apply_mass(const Field &x) const {
    require(x.rows() == nodes, "Mass product size mismatch");
    require(std::none_of(mirror_index_.begin(), mirror_index_.end(), [](int m) { return m >= 0; }),
            "apply_mass: mirror sides are not supported");
    Field in = x, out(nodes, x.cols());
    const int q = x.cols();
    for (int d = 0; d < dimension; ++d) {
        const int s = stride(d), per_direction = nodes_per_axis[d], lines = nodes / per_direction;
        const int *outer = mass_1d_[d].outerIndexPtr(), *inner = mass_1d_[d].innerIndexPtr();
        const double *value = mass_1d_[d].valuePtr();
        const double *data = in.data();
        parallel_for(lines, [&](int line) {
            const int start = (line / s) * (s * per_direction) + line % s;
            for (int k = 0; k < per_direction; ++k) {
                double *__restrict y = &out.coeffRef(start + k * s, 0);
                std::fill_n(y, q, 0.);
                for (int e = outer[k]; e < outer[k + 1]; ++e) {
                    const double c = value[e];
                    const double *z = data + std::size_t(start + inner[e] * s) * q;
#pragma omp simd
                    for (int j = 0; j < q; ++j) y[j] += c * z[j];
                }
            }
        });
        std::swap(in, out);
    }
    return in;
}
Field ElementWeno::solve_mass(Field b) const {
    require(b.rows() == nodes, "Mass solve size mismatch");
    for (int d = 0; d < dimension; ++d)
        solve_lines(b, d, false);
    return b;
}
// Consistent L2 projection of d u_h / d x_d. Mass and convection matrices are tensor products of the
// assembled one-dimensional matrices, so M^-1 C_d acts along the grid lines of direction d only.
Field ElementWeno::projected_gradient(const Field &a, int d) const {
    require(a.rows() == nodes, "Gradient size mismatch");
    const int q = a.cols(), s = stride(d), per_direction = nodes_per_axis[d], lines = nodes / per_direction;
    const LineFactor &natural = natural_[d];
    Field out(nodes, q);
    const int *outer = convection_1d_[d].outerIndexPtr(), *inner = convection_1d_[d].innerIndexPtr();
    const double *value = convection_1d_[d].valuePtr();
    const double *data = a.data();
    if (dist_ && d == dist_->axis) {
        // Distributed axis: the banded product C_1d along the line (same summation order as below), the global line
        // solve on the owned rows, then the halo rows from their owners.
        parallel_for(lines, [&](int line) {
            const int start = (line / s) * (s * per_direction) + line % s;
            for (int k = 0; k < per_direction; ++k) {
                double *__restrict y = &out.coeffRef(start + k * s, 0);
                std::fill_n(y, q, 0.);
                for (int e = outer[k]; e < outer[k + 1]; ++e) {
                    const double c = value[e];
                    const double *x = data + std::size_t(start + inner[e] * s) * q;
#pragma omp simd
                    for (int j = 0; j < q; ++j) y[j] += c * x[j];
                }
            }
        });
        dist_->solve_axis(out);
        dist_->exchange(out);
        return out;
    }
    if (mirror_index_[d] >= 0 && q == moments) {
        // Mirror in d: the banded product C_1d along d, then the split natural/Dirichlet solve.
        parallel_for(lines, [&](int line) {
            const int start = (line / s) * (s * per_direction) + line % s;
            for (int k = 0; k < per_direction; ++k) {
                double *__restrict y = &out.coeffRef(start + k * s, 0);
                std::fill_n(y, q, 0.);
                for (int e = outer[k]; e < outer[k + 1]; ++e) {
                    const double c = value[e];
                    const double *x = data + std::size_t(start + inner[e] * s) * q;
                    for (int j = 0; j < q; ++j) y[j] += c * x[j];
                }
            }
        });
        solve_lines(out, d, true);
        return out;
    }
    parallel_for(lines, [&](int line) { gradient_line(a, d, line, out); });
    return out;
}
void ElementWeno::gradient_line(const Field &a, int d, int line, Field &out) const {
    const int q = a.cols(), s = stride(d), per_direction = nodes_per_axis[d];
    const LineFactor &natural = natural_[d];
    const int *outer = convection_1d_[d].outerIndexPtr(), *inner = convection_1d_[d].innerIndexPtr();
    const double *value = convection_1d_[d].valuePtr();
    const double *data = a.data();
    const int start = (line / s) * (s * per_direction) + line % s;
    thread_local std::vector<double> work;
    work.resize(std::size_t(per_direction) * q);
    for (int k = 0; k < per_direction; ++k) {
        double *__restrict y = &work[std::size_t(natural.perm[k]) * q];
        std::fill_n(y, q, 0.);
        for (int e = outer[k]; e < outer[k + 1]; ++e) {
            const double c = value[e];
            const double *x = data + std::size_t(start + inner[e] * s) * q;
#pragma omp simd
            for (int j = 0; j < q; ++j) y[j] += c * x[j];
        }
    }
    solve_line(natural, work.data(), q);
    for (int k = 0; k < per_direction; ++k)
        std::copy_n(&work[std::size_t(natural.perm[k]) * q], q, &out.coeffRef(start + k * s, 0));
}
Field ElementWeno::gather_load(const Field &local, double scale) const {
    const int q = local.cols() / local_nodes;
    Field out(nodes, q);
    const double *data = local.data();
    parallel_for(nodes, [&](int i) {
        double *row = &out.coeffRef(i, 0);
        std::fill_n(row, q, 0.);
        for (int k = node_offsets_[i]; k < node_offsets_[i + 1]; ++k) {
            const double *source = data + std::size_t(node_entries_[k]) * q;
            for (int c = 0; c < q; ++c)
                row[c] += scale * source[c];
        }
    });
    return out;
}
Vec ElementWeno::weno_gamma(const Field &a) const {
    Vec gamma = Vec::Ones(cells);
    if (!options.weno)
        return gamma;
    const Vec density = a.col(0); // contiguous copy of psi^(0)
    const int n = local_nodes;
    const bool square = options.weno_r == 2;
    auto power = [&](double v) { return square ? v * v : std::pow(v, options.weno_r); };
    parallel_for(cells, [&](int e) {
        thread_local Mat candidates;
        thread_local Vec other, product, star;
        candidates.resize(n, 2 * dimension + 1);
        other.resize(n);
        product.resize(n);
        star.resize(n);
        auto norm = [&](const auto &v) {
            product.noalias() = seminorm_ * v;
            return std::sqrt(std::max(0., v.dot(product)));
        };
        for (int l = 0; l < n; ++l)
            candidates(l, 0) = density[connectivity_[e * n + l]];
        const double scale = norm(candidates.col(0));
        if (!(scale > 0))
            return;
        int m = 0;
        for (int d = 0; d < dimension; ++d)
            for (int side = 0; side < 2; ++side) {
                const int next = neighbors_[e * 2 * dimension + 2 * d + side];
                if (next == -1)
                    continue;
                for (int l = 0; l < n; ++l) // -2: mirror side, the neighbour is this element reflected in d
                    other[l] = density[next >= 0 ? connectivity_[next * n + l] : connectivity_[e * n + reflected_local_[d][l]]];
                candidates.col(++m).noalias() = extension_[d][side] * other;
            }
        Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 7, 1> smoothness(m + 1), weights(m + 1);
        smoothness[0] = power(scale);
        for (int l = 1; l <= m; ++l)
            smoothness[l] = power(norm(candidates.col(l)));
        // C5: epsilon relative to the stencil scale makes gamma invariant under u -> s u (s > 0). The scale is
        // the largest candidate seminorm^r; it is positive here because ||own|| > 0. Exact vacuum or constant
        // density in the element (scale == 0) returns gamma = 1 above, by definition.
        double epsilon = options.weno_epsilon;
        if (options.scale_invariant_weno)
            epsilon *= smoothness.maxCoeff();
        for (int l = 0; l <= m; ++l)
            weights[l] = (l ? options.weno_linear : 1 - m * options.weno_linear) / (smoothness[l] + epsilon);
        weights /= weights.sum();
        star.setZero();
        for (int l = 0; l <= m; ++l)
            star += weights[l] * candidates.col(l);
        other = candidates.col(0) - star;
        const double ratio = std::min(1., norm(other) / scale);
        gamma[e] = 1 - (options.weno_q == 1 ? ratio : std::pow(ratio, options.weno_q));
    });
    return gamma;
}
Field ElementWeno::stabilization_load(const Field &a, const Vec &gamma, ElementWenoStats *stats) const {
    const double start = seconds();
    std::array<Field, 3> gradient;
    for (int d = 0; d < dimension; ++d)
        gradient[d] = projected_gradient(a, d);
    if (stats)
        stats->t_gradient_solve += seconds() - start;
    return stabilization_load(a, gamma, gradient, stats);
}
Field ElementWeno::stabilization_load(const Field &a, const Vec &gamma, const std::array<Field, 3> &gradient,
                                      ElementWenoStats *stats) const {
    const int q = a.cols();
    double clock = seconds();
    ElementWenoStats scratch;
    ElementWenoStats &timing = stats ? *stats : scratch;
    auto lap = [&](double &slot) { const double now = seconds(); slot += now - clock; clock = now; };
    if (options.symmetric_residual) {
        // Symmetric residual form (docs/BOUNDARY_FIXES.md, L3):
        //   s_sym(v,u) = sum_e nu_e [ (1-gamma_e) (grad v, grad u)_e + gamma_e ((I-P) grad v, (I-P) grad u)_e ]
        // with P the global L2 projection onto the continuous space. Symmetric and PSD for every gamma_e in [0,1]
        // and every positive nu_e; equal to the original form for gamma = 1 and uniform nu (P is an orthogonal projection).
        std::array<Field, 3> y;
        for (int d = 0; d < dimension; ++d) {
            Field weighted(cells, local_nodes * q);
            parallel_for(cells, [&](int e) {
                Mat ae(local_nodes, q), ge(local_nodes, q);
                for (int l = 0; l < local_nodes; ++l) {
                    ae.row(l) = a.row(connectivity_[e * local_nodes + l]);
                    ge.row(l) = gradient[d].row(connectivity_[e * local_nodes + l]);
                }
                const Mat res = (convection_[d] * ae - mass_ * ge) * (viscosity * gamma[e]);
                for (int l = 0; l < local_nodes; ++l)
                    for (int c = 0; c < q; ++c) weighted(e, l * q + c) = res(l, c);
            });
            y[d] = solve_mass(gather_load(weighted));
        }
        Field sym(cells, local_nodes * q);
        parallel_for(cells, [&](int e) {
            Mat ae(local_nodes, q);
            for (int l = 0; l < local_nodes; ++l)
                ae.row(l) = a.row(connectivity_[e * local_nodes + l]);
            Mat r = stiffness_ * ae;
            Mat second = Mat::Zero(local_nodes, q);
            for (int d = 0; d < dimension; ++d) {
                Mat g(local_nodes, q), ye(local_nodes, q);
                for (int l = 0; l < local_nodes; ++l) {
                    g.row(l) = gradient[d].row(connectivity_[e * local_nodes + l]);
                    ye.row(l) = y[d].row(connectivity_[e * local_nodes + l]);
                }
                r.noalias() -= gamma[e] * convection_[d].transpose() * g;
                second.noalias() += convection_[d].transpose() * ye;
            }
            r = viscosity * r - second;
            for (int l = 0; l < local_nodes; ++l)
                for (int c = 0; c < q; ++c) sym(e, l * q + c) = r(l, c);
        });
        lap(timing.t_stab_local);
        return sym;
    }
    // nu_e (K_e a_e - gamma_e sum_d C_{d,e}^T g_{d,e}), written straight into the element buffer.
    Field local(cells, local_nodes * q);
    parallel_for(cells, [&](int e) {
        thread_local std::vector<const double *> rows;
        rows.resize(local_nodes);
        const int *map = &connectivity_[e * local_nodes];
        double *out = &local.coeffRef(e, 0);
        std::fill_n(out, std::size_t(local_nodes) * q, 0.);
        for (int l = 0; l < local_nodes; ++l)
            rows[l] = a.data() + std::size_t(map[l]) * q;
        apply_rows(local_nodes, local_nodes, q, stiffness_rows_.data(), viscosity, rows.data(), out);
        const double weight = -viscosity * gamma[e];
        if (weight != 0)
            for (int d = 0; d < dimension; ++d) {
                for (int l = 0; l < local_nodes; ++l)
                    rows[l] = gradient[d].data() + std::size_t(map[l]) * q;
                apply_rows(local_nodes, local_nodes, q, convection_t_rows_[d].data(), weight, rows.data(), out);
            }
    });
    lap(timing.t_stab_local);
    return local;
}
// residual_i = -sum_{(e,l) at node i} s_h^e(phi_l, a), gathered node by node without an element buffer. The four
// projected low moments of every element load are kept in `low` (cells x 4 local_nodes) for the limiter.
Field ElementWeno::stabilization_residual(const Field &a, const Vec &gamma, const std::array<Field, 3> &gradient,
                                          Field *low, ElementWenoStats *stats) const {
    const int q = a.cols(), n = local_nodes;
    const double start = seconds();
    if (low)
        low->resize(cells, 4 * n);
    Field residual(nodes, q);
    if (options.symmetric_residual) {
        const Field local = stabilization_load(a, gamma, gradient);
        if (low)
            parallel_for(cells, [&](int e) {
                for (int l = 0; l < n; ++l)
                    Eigen::Map<Four>(&low->coeffRef(e, 4 * l)) = project(local.row(e).segment(l * q, q).transpose());
            });
        residual = gather_load(local, -1.);
    } else
        parallel_for(nodes, [&](int i) {
            thread_local std::vector<const double *> rows;
            thread_local std::vector<double> term;
            rows.resize(n);
            term.resize(q);
            double *__restrict out = &residual.coeffRef(i, 0);
            std::fill_n(out, q, 0.);
            for (int k = node_offsets_[i]; k < node_offsets_[i + 1]; ++k) {
                const int e = node_entries_[k] / n, l = node_entries_[k] % n;
                const int *map = &connectivity_[e * n];
                std::fill_n(term.data(), q, 0.);
                for (int m = 0; m < n; ++m)
                    rows[m] = a.data() + std::size_t(map[m]) * q;
                apply_rows(1, n, q, &stiffness_rows_[std::size_t(l) * n], viscosity, rows.data(), term.data());
                const double weight = -viscosity * gamma[e];
                if (weight != 0)
                    for (int d = 0; d < dimension; ++d) {
                        for (int m = 0; m < n; ++m)
                            rows[m] = gradient[d].data() + std::size_t(map[m]) * q;
                        apply_rows(1, n, q, &convection_t_rows_[d][std::size_t(l) * n], weight, rows.data(), term.data());
                    }
                if (low)
                    Eigen::Map<Four>(&low->coeffRef(e, 4 * l)) = project(Eigen::Map<const Vec>(term.data(), q));
                const double *__restrict t = term.data();
#pragma omp simd
                for (int c = 0; c < q; ++c)
                    out[c] -= t[c];
            }
        });
    if (stats)
        stats->t_stab_local += seconds() - start;
    return residual;
}
void ElementWeno::face_data_into(int side_id, const V3 &x, double time, Incoming &data) const {
    // Same operations as face_data(), reusing the buffers of `data` (no allocation for the cached density).
    if (!cached_density_) {
        data = face_data(side_id, x, time);
        return;
    }
    const Boundary &b = *boundaries_[side_id];
    const Incoming &base = density_inflow_[side_id];
    data.d = base.d;
    data.kinetic = base.kinetic;
    data.moments = base.moments;
    if (inflow.atoms) {
        thread_local Incoming atoms;
        atoms.d.setZero(canonical_->r);
        atoms.kinetic.setZero();
        atoms.moments.setZero();
        b.add_atoms(inflow.atoms(x, time), atoms);
        b.finish_data(atoms);
        data.d += atoms.d;
        data.moments += atoms.moments;
        data.kinetic += atoms.kinetic;
        bool in_cone = exact::cone(data.moments);
        for (int k = 0; k < 64 && !in_cone; ++k) {
            data.moments[0] = std::nextafter(data.moments[0], std::numeric_limits<double>::infinity());
            in_cone = exact::cone(data.moments);
        }
        cone_require(in_cone, "Unable to represent incoming moments in M1 cone");
    }
}
Incoming ElementWeno::face_data(int side_id, const V3 &x, double time) const {
    const Boundary &b = *boundaries_[side_id];
    Incoming data;
    if (cached_density_) {
        data = density_inflow_[side_id];
        if (inflow.atoms) {
            Inflow atoms_only;
            atoms_only.atoms = inflow.atoms;
            const Incoming atoms = b.data(atoms_only, x, time);
            data.d += atoms.d;
            data.moments += atoms.moments;
            data.kinetic += atoms.kinetic;
            for (int k = 0; k < 64 && !exact::cone(data.moments); ++k)
                data.moments[0] = std::nextafter(data.moments[0], std::numeric_limits<double>::infinity());
            cone_require(exact::cone(data.moments), "Unable to represent incoming moments in M1 cone");
        }
    } else
        data = b.data(inflow, x, time);
    return data;
}
void ElementWeno::add_boundary(double time, const Field &a, const Field &low, const Field &low_flux,
                               Field &high, Field &low_load, Field &node_flux_area, Vec &area,
                               Field &bar, Vec *inflow_rate, PositivityProjectionStats *projection) const {
    const double ghost_start = seconds();
    Inflow atoms_only;
    atoms_only.atoms = inflow.atoms;
    // C6: outgoing half-range moments of the positive part of the nodal PN trace (point independent),
    // evaluated once per boundary node and face side.
    const bool ghost = options.fullsphere_ghost && !inflow.exterior_state && !inflow.exterior_m1;
    std::vector<Four> outgoing_moments(ghost ? ghost_node_.size() : 0);
    const bool project = ghost && options.positivity_projection;
    PositivityProjectionOptions popt;
    popt.tolerance = options.projection_tolerance;
    // Per-thread statistics, merged after the loop (parallel_for gives no reduction).
    std::vector<PositivityProjectionStats> pstats(project ? std::max(1, max_threads()) : 0);
    // projection_high_order: delta of the exterior PN state per ghost pair (zero rows: identity / C6).
    const bool high_exterior = project && options.projection_high_order && !options.projection_half_range &&
                               !options.projection_hybrid;
    Field delta;
    std::vector<char> has_delta;
    if (high_exterior) {
        delta.resize(ghost_node_.size(), moments);
        has_delta.assign(ghost_node_.size(), 0);
    }
    // Ghost moments of pair k from the PN trace psi at the points of out_rules_[side].
    auto ghost_pair = [&](int k, const Vec &psi) {
        const int side_id = ghost_side_[k], node = ghost_node_[k];
        const OutRule &out = out_rules_[side_id];
        const Boundary &b = *boundaries_[side_id];
        thread_local Vec z;
        // C6: positive part on the outgoing hemisphere (first `outgoing` points).
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
        // Hybrid: keep the C6 ghost where it is admissible (a_ext <= b, no cone scaling needed); project otherwise.
        bool need = project;
        if (project && options.projection_hybrid) {
            m = positive_part();
            const double bn = low(node, 0) + b.n.dot(Four(low.row(node).transpose()).tail<3>());
            need = m[0] - b.n.dot(m.tail<3>()) > bn;
            if (!need)
                ++pstats[thread_id()].hybrid_kept;
        }
        if (need) {
            PositivityProjectionStats &ps = pstats[thread_id()];
            const bool half = options.projection_half_range;
            Four U(low(node, 0), low(node, 1), low(node, 2), low(node, 3));
            if (half) { // target: outgoing half moments of the PN trace
                U.setZero();
                for (int j = 0; j < out.outgoing; ++j) {
                    const double v = psi[j] * out.weight[j];
                    U[0] += v, U[1] += v * out.omega(j, 0), U[2] += v * out.omega(j, 1), U[3] += v * out.omega(j, 2);
                }
            }
            const int q = half ? out.outgoing : int(psi.size());
            z.resize(q);
            PositivityProjectionWarm &warm = projection_warm_[k];
            PositivityProjectionOptions nopt = popt;
            nopt.augment = !half;
            if (out.c.size())
                nopt.c = out.c.data(), nopt.cinv = out.cinv.data();
            const auto o = project_positive_moments(q, out.ox.data(), out.oy.data(), out.oz.data(),
                                                    out.weight.data(), psi.data(), U, z.data(), nopt,
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
                // The atom of an augmented rule belongs to the outgoing half if Omega*.n > 0.
                if (o.augmented && o.atom_direction.dot(b.n) > 0)
                    m += o.atom;
                // The exact projection has outgoing mass <= U[0] (z >= 0, total mass U[0]). The Newton tolerance is
                // relative to the trace scale sum w|y| + rho; near vacuum the trace can exceed rho by orders of
                // magnitude (e.g. rho ~ 1e-33 next to a beam), and the converged moments may then overshoot rho.
                // A ghost with rho_o >> rho turns ext - F_n(ext) into a cancellation that leaves the cone, so cap it.
                if (m[0] > U[0] * (1 + 64 * std::numeric_limits<double>::epsilon())) {
                    ++ps.capped;
                    m *= U[0] > 0 ? U[0] / m[0] : 0.;
                }
                if (high_exterior && !r.identity) {
                    thread_local Vec diff;
                    diff.resize(q);
                    for (int j = 0; j < q; ++j)
                        diff[j] = out.weight[j] * (z[j] - psi[j]);
                    delta.row(k).noalias() = (out.harmonics.transpose() * diff).transpose();
                    if (o.augmented && o.atom[0] != 0) {
                        const Vec y = harmonics(pn.order, V3(o.atom_direction));
                        for (int c = 0; c < moments; ++c)
                            delta(k, c) += o.atom[0] * y[active[c]];
                    }
                    has_delta[k] = 1;
                }
            } else {
                ++ps.failures;
                if (options.projection_strict)
                    throw StepRejected("Positivity projection failed at boundary node " + std::to_string(node) +
                                       " (relative moment residual " + std::to_string(r.residual) + ")");
            }
        }
        if (!projected && !(project && options.projection_hybrid))
            m = positive_part();
        else if (!projected && options.projection_hybrid && need)
            m = positive_part(); // projection failed: C6 (scaled below)
        else if (projected && options.projection_diagnostics) {
            // Diagnostic: the C6 ghost of the same node (with its cone scaling) and its distance to the projection.
            Four c6 = positive_part();
            const double bn = low(node, 0) + b.n.dot(Four(low.row(node).transpose()).tail<3>());
            const double ae = c6[0] - b.n.dot(c6.tail<3>());
            PositivityProjectionStats &ps = pstats[thread_id()];
            if (ae > bn) {
                ++ps.c6_scaled;
                c6 *= bn > 0 ? bn / ae : 0.;
            }
            const double scale = std::max(low(node, 0), 1e-300);
            const double d = (m - c6).cwiseAbs().maxCoeff() / scale;
            ps.c6_diff_sum += d;
            ps.c6_diff_max = std::max(ps.c6_diff_max, d);
            ps.c6_diff_side[side_id] += d;
            ps.c6_scaled_diff += ae > bn ? d : 0.;
        }
        // Low-order vacuum outflow property: B^L = -(b - a_ext)/2 <= 0 needs a_ext <= b, where
        // b = rho + n.J of the node (>= 0 in the cone) and a_ext = rho_o - n.J_o >= 0 of the ghost part.
        // For a nonnegative PN trace a <= b holds automatically (mu > 0 on the outgoing hemisphere);
        // a PN trace with negative lobes can violate it, then the outgoing part is scaled by b/a (a cone scaling).
        // A converged projection has full moments (rho, J) of the node and z >= 0, so a_ext <= b up to rounding.
        const double b_node = low(node, 0) + b.n[0] * low(node, 1) + b.n[1] * low(node, 2) + b.n[2] * low(node, 3);
        const double a_ext = m[0] - b.n.dot(m.tail<3>());
        if (a_ext > b_node) {
            if (projected)
                ++pstats[thread_id()].scaled;
            m *= b_node > 0 ? b_node / a_ext : 0.;
        }
        outgoing_moments[k] = m;
    };
    if (ghost) {
        // PN traces in blocks of ghost_block pairs of one side (bounded memory, GEMM-sized blocks): separable
        // evaluation (SeparableTrace) or, if unavailable, the dense harmonics product.
        // At least ~4 blocks per thread, at most 32 pairs per block (GEMM width) and at least 4.
        const int ghost_block = std::clamp(int(ghost_node_.size()) / std::max(1, 4 * max_threads()), 4, 32);
        std::vector<std::array<int, 3>> tasks; // side, begin, end in ghost_by_side_[side]
        for (int side = 0; side < 6; ++side)
            for (int begin = 0; begin < int(ghost_by_side_[side].size()); begin += ghost_block)
                tasks.push_back({side, begin, std::min<int>(begin + ghost_block, ghost_by_side_[side].size())});
        parallel_for(int(tasks.size()), [&](int t) {
            const auto [side, begin, end] = tasks[t];
            const std::vector<int> &list = ghost_by_side_[side];
            thread_local Mat at, traces;
            thread_local Vec psi;
            at.resize(moments, end - begin);
            for (int j = begin; j < end; ++j)
                at.col(j - begin) = a.row(ghost_node_[list[j]]).transpose();
            if (ghost_sigma_.size())
                at = ghost_sigma_.asDiagonal() * at;
            if (separable_[side])
                separable_[side]->evaluate(at, traces);
            else
                traces.noalias() = out_rules_[side].harmonics * at;
            for (int j = begin; j < end; ++j) {
                psi = traces.col(j - begin);
                ghost_pair(list[j], psi);
            }
        });
    }
    if (project && projection)
        for (const auto &ps : pstats)
            projection->merge(ps);
    const double ghost_end = seconds();
    if (projection)
        projection->seconds += ghost_end - ghost_start;
    // Face rows (constructor): one row per (face, face node). The face loop writes its rows directly; no per-face
    // matrices (thousands of small allocations per call did not scale with the thread count).
    const int face_rows = side_row_start_[6];
    Field G, D(face_rows, canonical_->r), L(face_rows, 4), BR(face_rows, 4), FA(face_rows, 3);
    Vec AR(face_rows), IN(face_rows);
    G.resize(face_rows, moments);
    Field K; // face rows of delta (projection_high_order)
    std::vector<char> face_delta;
    if (high_exterior) {
        K.resize(face_rows, moments);
        face_delta.assign(6, 0);
        for (std::size_t k = 0; k < has_delta.size(); ++k)
            if (has_delta[k])
                face_delta[ghost_side_[k]] = 1;
    }
    std::vector<char> face_datum(faces_.size(), 0);
    parallel_for(int(faces_.size()), [&](int index) {
        const auto &f = faces_[index];
        const int side_id = 2 * f.direction + f.side;
        const FaceRule &rule = rules_[side_id];
        const Boundary &b = *boundaries_[side_id];
        const int nf = rule.local.size(), r0 = face_row_[index];
        thread_local std::vector<int> global;
        global.resize(nf);
        auto load_row = L.middleRows(r0, nf), bar_row = BR.middleRows(r0, nf), fa_row = FA.middleRows(r0, nf);
        auto datum = D.middleRows(r0, nf);
        load_row.setZero(), bar_row.setZero(), fa_row.setZero(), datum.setZero();
        AR.segment(r0, nf).setZero(), IN.segment(r0, nf).setZero();
        V3 origin;
        element_origin(f.element, origin);
        for (int n = 0; n < nf; ++n)
            global[n] = connectivity_[f.element * local_nodes + rule.local[n]];
        const int *ghost_id = &ghost_of_face_[std::size_t(index) * nf];
        // Memo per face node: the exterior state, its Levermore flux and the bar state depend on the point only
        // through data.moments. For spatially uniform inflow (vacuum, plane waves, isotropic) they are evaluated
        // once per node instead of once per (point, node); identical inputs give identical values.
        const bool memo = !inflow.exterior_state && !inflow.exterior_m1;
        thread_local std::vector<char> memo_valid;
        thread_local std::vector<Four> memo_moments, memo_state;
        memo_valid.assign(nf, 0);
        memo_moments.resize(nf), memo_state.resize(nf);
        for (int point = 0; point < rule.weights.size(); ++point) {
            V3 x = origin;
            for (int d = 0; d < dimension; ++d)
                x[d] += h * rule.points(point, d);
            thread_local Incoming fresh;
            if (!inflow.steady)
                face_data_into(side_id, x, time, fresh);
            const Incoming &data = inflow.steady ? steady_data_[std::size_t(index) * rule.weights.size() + point]
                                                 : fresh;
            for (int n = 0; n < nf; ++n) {
                const double weight = rule.weights[point] * rule.shape(n, point);
                if (weight == 0)
                    continue;
                const int i = global[n];
                datum.row(n).noalias() += weight * data.d.transpose();
                const Four ui(low(i, 0), low(i, 1), low(i, 2), low(i, 3));
                IN[r0 + n] += weight * std::max(0., -data.kinetic[0]);
                Four state;
                if (memo && memo_valid[n] && (memo_moments[n].array() == data.moments.array()).all())
                    state = memo_state[n];
                else {
                    Four ext = inflow.exterior_state ? inflow.exterior_state(x, time, b.n, ui)
                               : inflow.exterior_m1  ? inflow.exterior_m1(x, time)
                                                     : data.moments;
                    // Sum of two cone elements is in the cone; outward ulp rounding only for binary64 sums.
                    if (ghost)
                        ext += outgoing_moments[ghost_id[n]];
                    bool in_cone = exact::cone(ext);
                    for (int k = 0; ghost && k < 64 && !in_cone; ++k) {
                        ext[0] = std::nextafter(ext[0], std::numeric_limits<double>::infinity());
                        in_cone = exact::cone(ext);
                    }
                    cone_require(in_cone, "M1 exterior state outside cone");
                    Four own = Four::Zero();
                    for (int d = 0; d < dimension; ++d)
                        own += b.n[d] * block(low_flux, i, d);
                    const Four outer = Levermore(ext).flux(b.n);
                    state = .5 * (ui + own) + .5 * (ext - outer);
                    // Both halves are cone elements in exact arithmetic, but each is a cancellation: for a beam
                    // u = rho (1, -n) (incoming at this side) 1/2 (u + F_n(u)) = 0 exactly and evaluates to
                    // O(eps rho), possibly outside the cone by much more than an ulp of the (tiny) result. Before,
                    // the spurious outgoing mass of the C6 ghost usually masked this; the projection gives the exact
                    // zero ghost. Outward rounding relative to the size of the summands (C1 philosophy).
                    if (options.outward_bar_rounding && !exact::cone(state)) {
                        const double size = std::abs(ui[0]) + std::abs(own[0]) + std::abs(ext[0]) + std::abs(outer[0]);
                        if (state.tail<3>().norm() - state[0] <= 64 * std::numeric_limits<double>::epsilon() * size)
                            state[0] = norm_up(state);
                    }
                    if (memo)
                        memo_valid[n] = 1, memo_moments[n] = data.moments, memo_state[n] = state;
                }
                AR[r0 + n] += weight;
                bar_row.row(n) += weight * state.transpose();
                for (int d = 0; d < dimension; ++d)
                    fa_row(n, d) += weight * b.n[d];
                load_row.row(n) += weight * (state - ui).transpose();
            }
        }
        face_datum[index] = !datum.isZero(0);
        // sum_points w phi_n [(A_n - llf_homogeneous) trace - llf_datum datum], with the exact face mass matrix;
        // the products with the dense q x q (and r x q) matrices are done per side below.
        thread_local Mat traces;
        traces.resize(nf, moments);
        for (int n = 0; n < nf; ++n)
            traces.row(n) = a.row(global[n]);
        G.middleRows(r0, nf).noalias() = rule.mass * traces;
        if (high_exterior) {
            bool any = false;
            for (int n = 0; n < nf; ++n)
                if (has_delta[ghost_id[n]]) {
                    traces.row(n) = delta.row(ghost_id[n]);
                    any = true;
                } else
                    traces.row(n).setZero();
            if (any)
                K.middleRows(r0, nf).noalias() = rule.mass * traces;
            else
                K.middleRows(r0, nf).setZero();
        }
    });
    const double faces_done = seconds();
    if (projection)
        projection->face_seconds += faces_done - ghost_end;
    // Per side: H = G B - D B_datum on the contiguous rows of the side. The matrices are small (rows x q x q), so all
    // sides run in one parallel loop over blocks of rows with allocation-free coefficient products (a GEMM call per
    // small block allocated workspace and did not scale with the thread count).
    Field H;
    {
        H.resize(face_rows, moments);
        std::array<bool, 6> datum_zero;
        datum_zero.fill(true);
        for (std::size_t index = 0; index < faces_.size(); ++index)
            if (face_datum[index])
                datum_zero[2 * faces_[index].direction + faces_[index].side] = false;
        constexpr int rows_per_task = 16;
        std::vector<std::array<int, 3>> tasks; // side, first row, row count
        for (int side = 0; side < 6; ++side)
            if (boundaries_[side])
                for (int r = side_row_start_[side]; r < side_row_start_[side + 1]; r += rows_per_task)
                    tasks.push_back({side, r, std::min(rows_per_task, side_row_start_[side + 1] - r)});
        parallel_for(int(tasks.size()), [&](int t) {
            const auto [side, r, n] = tasks[t];
            H.middleRows(r, n).noalias() = G.middleRows(r, n).lazyProduct(boundary_flux_t_[side]);
            if (!datum_zero[side])
                H.middleRows(r, n).noalias() -= D.middleRows(r, n).lazyProduct(datum_t_[side]);
            if (high_exterior && face_delta[side])
                H.middleRows(r, n).noalias() -= K.middleRows(r, n).lazyProduct(exterior_t_[side]);
        });
    }
    const double products_done = seconds();
    if (projection)
        projection->product_seconds += products_done - faces_done;
    // `high` has one row per node or, in rate(), one row per boundary node (boundary_slot_). Node by node in
    // parallel; each node sums its face rows in ascending face index (the order of the former serial loop).
    const bool compact = high.rows() != nodes;
    parallel_for(int(boundary_nodes_.size()), [&](int k) {
        const int i = boundary_nodes_[k];
        for (int e = boundary_row_offsets_[k]; e < boundary_row_offsets_[k + 1]; ++e) {
            const int r = boundary_rows_[e];
            high.row(compact ? k : i) += H.row(r);
            low_load.row(i) += L.row(r);
            bar.row(i) += BR.row(r);
            node_flux_area.row(i) += FA.row(r);
            area[i] += AR[r];
            if (inflow_rate)
                (*inflow_rate)[i] += IN[r];
        }
    });
    if (projection)
        projection->reduce_seconds += seconds() - products_done;
}
Field ElementWeno::rate(double time, const Field &a, ElementWenoStats *stats,
                        ElementWenoDebug *debug) const {
    require(a.rows() == nodes && a.cols() == moments, "Invalid element WENO state");
    const double finite_start = seconds();
    {
        // Parallel finiteness check; the serial a.allFinite() cost a quarter of the run time on 8 threads.
        std::atomic<bool> finite{true};
        const double *data = a.data();
        const std::size_t total = std::size_t(nodes) * std::size_t(a.cols()), chunk = 4096;
        parallel_for(int((total + chunk - 1) / chunk), [&](int k) {
            const std::size_t begin = std::size_t(k) * chunk, end = std::min(total, begin + chunk);
            bool ok = true;
            for (std::size_t i = begin; i < end; ++i)
                ok &= std::isfinite(data[i]);
            if (!ok)
                finite = false;
        });
        require(finite.load(), "Invalid element WENO state");
    }
    const int q = moments;
    double clock = seconds();
    ElementWenoStats scratch;
    ElementWenoStats &timing = stats ? *stats : scratch;
    timing.t_finite += clock - finite_start;
    auto lap = [&](double &slot) {
        const double now = seconds();
        slot += now - clock;
        clock = now;
    };
    const double closure_setup_before = timing.t_closure_setup;
    Field low(nodes, 4), low_flux = parallel_zero(nodes, 12), closure = parallel_zero(nodes, 12);
    lap(timing.t_closure_setup);
    // Only the four low moments of the PN flux a A_d enter the closure difference G = F^H - F^L.
    parallel_for(nodes, [&](int i) {
        const double *row = a.data() + std::size_t(i) * q;
        const Four u = project(a.row(i).transpose());
        low.row(i) = u.transpose();
        const Levermore closure_state(u);
        for (int d = 0; d < dimension; ++d) {
            V3 n = V3::Zero();
            n[d] = 1;
            const Four fl = closure_state.flux(n);
            Four head = Four::Zero();
            add_flux(flux_t_[d], row, 1., head.data(), 4);
            const Four fh = project(head);
            for (int c = 0; c < 4; ++c) {
                low_flux(i, 4 * d + c) = fl[c];
                closure(i, 4 * d + c) = fh[c] - fl[c];
            }
        }
    });
    if (options.m1_target)
        closure.setZero(); // target flux = F^L
    lap(timing.t_closure);
    timing.t_closure += timing.t_closure_setup - closure_setup_before; // keep t_closure = setup + loop

    // High-order PN residual and its stabilization.
    const Vec gamma = options.stabilize ? weno_gamma(a) : Vec(Vec::Ones(cells));
    lap(timing.t_weno);
    // g_d = M^-1 C_d a serves twice: in s_h and as the advective rate -sum_d g_d A_d (A_d acts on the PN index only).
    const double stabilization_start = clock;
    std::array<Field, 3> gradient;
    if (dist_) {
        // Distributed: the line solve along the axis is a pipeline over the ranks; the gradients of the other
        // directions (independent, rank-local lines) fill its idle threads. Same arithmetic as projected_gradient.
        const int D = dist_->axis, q = a.cols();
        Field &g = gradient[D];
        g.resize(nodes, q);
        {
            const int s = stride(D), per_direction = nodes_per_axis[D], lines = nodes / per_direction;
            const int *outer = convection_1d_[D].outerIndexPtr(), *inner = convection_1d_[D].innerIndexPtr();
            const double *value = convection_1d_[D].valuePtr();
            const double *data = a.data();
            parallel_for(lines, [&](int line) {
                const int start = (line / s) * (s * per_direction) + line % s;
                for (int k = 0; k < per_direction; ++k) {
                    double *__restrict y = &g.coeffRef(start + k * s, 0);
                    std::fill_n(y, q, 0.);
                    for (int e = outer[k]; e < outer[k + 1]; ++e) {
                        const double c = value[e];
                        const double *x = data + std::size_t(start + inner[e] * s) * q;
#pragma omp simd
                        for (int j = 0; j < q; ++j) y[j] += c * x[j];
                    }
                }
            });
        }
        std::vector<std::array<int, 3>> blocks; // direction, first line, end line
        for (int d = 0; d < dimension; ++d) {
            if (d == D)
                continue;
            gradient[d].resize(nodes, q);
            const int lines = nodes / nodes_per_axis[d], per = std::max(1, lines / (4 * max_threads()));
            for (int l = 0; l < lines; l += per)
                blocks.push_back({d, l, std::min(lines, l + per)});
        }
        dist_->solve_axis_overlap(g, int(blocks.size()), [&](int i) {
            for (int line = blocks[i][1]; line < blocks[i][2]; ++line)
                gradient_line(a, blocks[i][0], line, gradient[blocks[i][0]]);
        });
        dist_->exchange(g);
    } else
        for (int d = 0; d < dimension; ++d)
            gradient[d] = projected_gradient(a, d);
    lap(timing.t_gradient_solve);
    // Gathered -s_h and the projected low moments of the element loads P s_h^e(phi_i, a).
    Field stabilization_low;
    Field residual = options.stabilize ? stabilization_residual(a, gamma, gradient, &stabilization_low, &timing)
                                       : Field(Field::Zero(nodes, q));
    clock = seconds();
    timing.t_stabilization += clock - stabilization_start;
    // boundary_high holds one row per boundary node (boundary_slot_).
    Field boundary_high = parallel_zero(boundary_nodes_.size(), q), boundary_low = parallel_zero(nodes, 4);
    Field flux_area = parallel_zero(nodes, 3), boundary_bar = parallel_zero(nodes, 4);
    Vec boundary_area = parallel_zero(nodes), boundary_inflow = parallel_zero(nodes);
    if (!faces_.empty()) {
        add_boundary(time, a, low, low_flux, boundary_high, boundary_low, flux_area, boundary_area,
                     boundary_bar, &boundary_inflow, &timing.projection);
        if (!options.m1_target)
            for (std::size_t k = 0; k < boundary_nodes_.size(); ++k)
                residual.row(boundary_nodes_[k]) += boundary_high.row(k);
    }
    lap(timing.t_boundary);
    Field high_rate = options.stabilize || !faces_.empty() ? solve_mass(std::move(residual)) : std::move(residual);
    Field moment_rate(nodes, 4);
    if (!options.m1_target)
        for (int d = 0; d < dimension; ++d)
            add_flux_rows(flux_t_[d], gradient[d], -1., high_rate);
    parallel_for(nodes, [&](int i) { moment_rate.row(i) = project(high_rate.row(i).transpose()).transpose(); });
    if (dist_ && !options.m1_target)
        dist_->exchange(moment_rate); // the limiter of owned nodes reads the halo rows
    if (options.m1_target)
        for (int d = 0; d < dimension; ++d)
            moment_rate -= projected_gradient(Field(low_flux.middleCols(4 * d, 4)), d);
    lap(timing.t_mass);

    Field load;
    Field raw_load;
    {
        const LowOrderInputs inputs{&low, &low_flux, &closure, &moment_rate, &stabilization_low, &boundary_high,
                                    &boundary_low, &flux_area, &boundary_bar, &boundary_area, &boundary_inflow, &gamma};
        const long before = stats ? stats->limited_elements + stats->limited_boundary_nodes : 0;
        load = limited_low_load(inputs, stats, debug ? &raw_load : nullptr);
        if (stats) {
            ++stats->rate_calls;
            stats->unlimited_rate_calls += stats->limited_elements + stats->limited_boundary_nodes == before;
        }
    }
    clock = seconds();
    parallel_for(nodes, [&](int i) {
        set_low(Eigen::Map<Vec>(&high_rate.coeffRef(i, 0), q), Four(load.row(i).transpose()) / lumped[i]);
    });
    // Mirror planes: moments odd under the reflection vanish (the full scheme's two mirrored halves cancel).
    // Dropping J_d from a cone state keeps it in the cone, so the realizability argument is unchanged.
    for (int d = 0; d < dimension; ++d)
        if (mirror_index_[d] >= 0)
            parallel_for(int(mirror_nodes_[d].size()), [&](int k) {
                double *row = &high_rate.coeffRef(mirror_nodes_[d][k], 0);
                for (int c = 0; c < q; ++c)
                    if (reflect_odd_[d][c])
                        row[c] = 0;
            });
    lap(timing.t_finish);
    timing.t_rate += seconds() - finite_start;
    if (debug) {
        debug->ho_moment_rate = moment_rate;
        debug->unlimited_rate = raw_load;
        for (int i = 0; i < nodes; ++i)
            debug->unlimited_rate.row(i) /= lumped[i];
    }
    return high_rate;
}
// Limited low-order load of the four M1 moments (element bar states, antidiffusive element contributions, boundary
// corrections). Used by rate(). Returns the gathered nodal
// load in (rho, J) units, not yet divided by the lumped mass; *raw_out receives the unlimited load when requested.
Field ElementWeno::limited_low_load(const LowOrderInputs &in, ElementWenoStats *stats, Field *raw_out) const {
    const Field &low = *in.low, &low_flux = *in.low_flux, &closure = *in.closure, &moment_rate = *in.moment_rate;
    const Field &stabilization_low = *in.stabilization_low, &boundary_high = *in.boundary_high;
    const Field &boundary_low = *in.boundary_low, &flux_area = *in.flux_area, &boundary_bar = *in.boundary_bar;
    const Vec &boundary_area = *in.boundary_area, &boundary_inflow = *in.boundary_inflow;
    double clock = seconds();
    ElementWenoStats scratch;
    ElementWenoStats &timing = stats ? *stats : scratch;
    auto lap = [&](double &slot) {
        const double now = seconds();
        slot += now - clock;
        clock = now;
    };
    // Element bar states and antidiffusive contributions.
    const int width = local_nodes * 4;
    Field element_limited(cells, width), element_raw;
    if (raw_out)
        element_raw.resize(cells, width);
    std::vector<double> element_factor(cells, 1.), raw_sum(cells, 0.), limited_sum(cells, 0.);
    std::vector<long> nudged(cells, 0), outward(cells, 0);
    parallel_for(cells, [&](int e) {
        const int *map = &connectivity_[e * local_nodes];
        thread_local std::vector<Four> ubar, contribution, limited;
        thread_local std::vector<double> copies;
        thread_local Vec clipped;
        ubar.resize(local_nodes);
        contribution.resize(local_nodes);
        clipped.resize(local_nodes);
        // Element copies per local node: u (4), F^L (12), G (12), projected high-order rate (4).
        copies.resize(std::size_t(32) * local_nodes);
        for (int l = 0; l < local_nodes; ++l) {
            double *c = &copies[std::size_t(32) * l];
            std::copy_n(&low.coeffRef(map[l], 0), 4, c);
            std::copy_n(&low_flux.coeffRef(map[l], 0), 12, c + 4);
            std::copy_n(&closure.coeffRef(map[l], 0), 12, c + 16);
            std::copy_n(&moment_rate.coeffRef(map[l], 0), 4, c + 28);
        }
        using Quad = Eigen::Map<const Four>;
        const Vec &total_viscosity = total_viscosity_;
        for (int i = 0; i < local_nodes; ++i) {
            const double *ci = &copies[std::size_t(32) * i];
            const Four ui = Quad(ci), qi = Quad(ci + 28);
            Four weighted = Four::Zero(), fi = Four::Zero();
            const PairEntry *pairs = &pair_table_[std::size_t(i) * (local_nodes - 1)];
            for (int k = 0; k < local_nodes - 1; ++k) {
                const PairEntry &pe = pairs[k];
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
                            outward[e] += raised > 0;
                        } else
                            pair_bar = stable_base_bar(ui, uj, edge, dij);
                    }
                    weighted += dij * pair_bar;
                    fi += dij * (ui - uj);
                }
                fi += pe.m * (qi - Quad(cj + 28)) + closure_term;
            }
            if (options.stabilize)
                fi -= Quad(&stabilization_low.coeffRef(e, 4 * i));
            Four bar = weighted / total_viscosity[i];
            if (nudge_into_cone(bar))
                ++nudged[e];
            ubar[i] = bar;
            contribution[i] = fi;
        }
        Four sum = Four::Zero();
        for (const auto &f : contribution)
            sum += f;
        raw_sum[e] = sum.cwiseAbs().maxCoeff();
        limited = contribution;
        if (options.limit) {
            // Step 1: clip and scale the zeroth moment.
            double pos = 0, neg = 0;
            for (int i = 0; i < local_nodes; ++i) {
                const double fmin = -2 * total_viscosity[i] * (1 - options.positivity_fraction) * ubar[i][0];
                clipped[i] = std::max(contribution[i][0], fmin);
                (clipped[i] > 0 ? pos : neg) += clipped[i];
            }
            for (int i = 0; i < local_nodes; ++i) {
                double value = clipped[i];
                if (value > 0 && pos + neg > 0)
                    value *= -neg / pos;
                else if (value < 0 && pos + neg < 0)
                    value *= -pos / neg;
                limited[i][0] = value;
            }
            // Step 2: one element factor against the M1 cone.
            double factor = 1;
            for (int i = 0; i < local_nodes; ++i) {
                const Four g = limited[i] / (2 * total_viscosity[i]);
                if (exact::cone(ubar[i] + g))
                    continue;
                factor = std::min(factor, options.element_theta * segment(ubar[i], g));
            }
            for (auto &f : limited)
                f *= factor;
            element_factor[e] = factor;
        }
        Four after = Four::Zero();
        for (const auto &f : limited)
            after += f;
        limited_sum[e] = after.cwiseAbs().maxCoeff();
        for (int i = 0; i < local_nodes; ++i) {
            const int ni = map[i];
            const Four ui(low(ni, 0), low(ni, 1), low(ni, 2), low(ni, 3));
            const Four lo = 2 * total_viscosity[i] * (ubar[i] - ui);
            const Four out = lo + limited[i];
            for (int c = 0; c < 4; ++c)
                element_limited(e, i * 4 + c) = out[c];
            if (raw_out) {
                const Four raw = lo + contribution[i];
                for (int c = 0; c < 4; ++c)
                    element_raw(e, i * 4 + c) = raw[c];
            }
        }
    });

    Field load = gather_load(element_limited);
    lap(timing.t_limiter);
    Field raw_load;
    if (raw_out)
        raw_load = gather_load(element_raw);
    // Boundary corrections h_i = P b^H - b^L - (int phi_i n) . G_i, limited node by node.
    double min_boundary = 1, min_sign = 1, boundary_mass_rate = 0;
    long limited_boundary = 0, sign_limited = 0;
    if (!faces_.empty()) {
        // Node-independent, so in parallel; the reductions run afterwards in ascending node order (as before).
        const int nb = boundary_nodes_sorted_.size();
        std::vector<double> node_theta(nb, 1.), node_sign(nb, 1.), node_mass(nb, 0.);
        std::vector<char> node_signed(nb, 0);
        parallel_for(nb, [&](int k) {
            const int i = boundary_nodes_sorted_[k];
            if (boundary_area[i] <= 0)
                return;
            Four correction = project(boundary_high.row(boundary_slot_[i]).transpose()) -
                              Four(boundary_low(i, 0), boundary_low(i, 1), boundary_low(i, 2), boundary_low(i, 3));
            for (int d = 0; d < dimension; ++d)
                correction -= flux_area(i, d) * block(closure, i, d);
            Four bar(boundary_bar(i, 0), boundary_bar(i, 1), boundary_bar(i, 2), boundary_bar(i, 3));
            bar /= boundary_area[i];
            nudge_into_cone(bar);
            double theta = options.m1_target ? 0. : 1.;
            if (options.limit && !options.m1_target) {
                const Four g = correction / boundary_area[i];
                if (!exact::cone(bar + g))
                    theta = options.element_theta * segment(bar, g);
                if (options.vacuum_sign_limit) {
                    // C2: accepted mass rate B_i = L_i + theta*h_i[0] with the low-order rate
                    //   L_i = A_i (bar_i - u_i)[0] - (int phi_i n).F^L_i[0]   (geometric corner/edge terms included)
                    // must not exceed max(L_i, inflow_i): theta*h_i[0] <= max(0, inflow_i - L_i) =: allowance.
                    // In vacuum (inflow 0) L_i = -q^L_i <= 0, so B_i <= 0 (outflow only). theta is only reduced,
                    // and bar + theta*g stays in the cone by convexity of the cone.
                    double lo_rate = boundary_area[i] * (bar[0] - low(i, 0));
                    for (int d = 0; d < dimension; ++d)
                        lo_rate -= flux_area(i, d) * low_flux(i, 4 * d);
                    const double allowance = std::max(0., boundary_inflow[i] - lo_rate);
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
            const Four base = boundary_area[i] * (bar - ui);
            load.row(i) += (base + theta * correction).transpose();
            // Total mass rate of node i: the gathered low-order element terms contain the lumped boundary flux
            // -(int phi_i n).F^L_i (only the antidiffusive element contributions sum to zero), plus base + h_i.
            double m = base[0] + theta * correction[0];
            for (int d = 0; d < dimension; ++d)
                m -= flux_area(i, d) * low_flux(i, 4 * d);
            node_mass[k] = m;
            if (raw_out)
                raw_load.row(i) += (base + correction).transpose();
        });
        for (int k = 0; k < nb; ++k) {
            if (boundary_area[boundary_nodes_sorted_[k]] <= 0)
                continue;
            if (is_owned(boundary_nodes_sorted_[k]))
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
    }
    lap(timing.t_finish);
    if (raw_out)
        *raw_out = std::move(raw_load);
    if (stats) {
        stats->min_gamma = std::min(stats->min_gamma, in.gamma->minCoeff());
        for (int e = 0; e < cells; ++e) {
            stats->min_limiter = std::min(stats->min_limiter, element_factor[e]);
            stats->limited_elements += element_factor[e] < 1;
            stats->max_raw_zero_sum = std::max(stats->max_raw_zero_sum, raw_sum[e]);
            stats->max_limited_zero_sum = std::max(stats->max_limited_zero_sum, limited_sum[e]);
            stats->nudged_states += nudged[e];
            stats->outward_bars += outward[e];
        }
        stats->min_boundary_limiter = std::min(stats->min_boundary_limiter, min_boundary);
        stats->last_boundary_mass_rate = dist_ ? dist_->sum(boundary_mass_rate) : boundary_mass_rate;
        stats->sign_limited_nodes += sign_limited;
        stats->min_sign_factor = std::min(stats->min_sign_factor, min_sign);
        stats->limited_boundary_nodes += limited_boundary;
    }
    return load;
}
Field ElementWeno::linear_operator(const Field &x, double gamma_value) const {
    const int q = x.cols();
    std::array<Field, 3> gradient;
    for (int d = 0; d < dimension; ++d)
        gradient[d] = projected_gradient(x, d);
    Field out = options.stabilize
                    ? solve_mass(stabilization_residual(x, Vec::Constant(cells, gamma_value), gradient, nullptr))
                    : Field(Field::Zero(nodes, q));
    parallel_for(nodes, [&](int i) {
        for (int d = 0; d < dimension; ++d)
            add_flux(flux_t_[d], gradient[d].data() + std::size_t(i) * q, -1., &out.coeffRef(i, 0), q);
    });
    return out;
}
double ElementWeno::operator_probe() const {
    std::mt19937 rng(4242);
    std::uniform_real_distribution<double> uniform(-1, 1);
    Field x(nodes, moments);
    if (!dist_) {
        for (int i = 0; i < nodes; ++i)
            for (int c = 0; c < moments; ++c)
                x(i, c) = uniform(rng);
    } else
        distributed_random(x, rng, uniform);
    const Field y = linear_operator(x, .5);
    if (!dist_)
        return y.norm();
    double s2 = 0;
    for (int i = 0; i < nodes; ++i)
        if (owned_[i])
            s2 += y.row(i).squaredNorm();
    return std::sqrt(dist_->sum(s2));
}
double ElementWeno::high_order_step() const {
    if (spectral_radius_ <= 0) {
        std::mt19937 rng(12345);
        std::uniform_real_distribution<double> uniform(-1, 1);
        double radius = 0;
        // Distributed: the norms over the owned rows (global sums), the start vector of the same random sequence over
        // the global node order (rank-independent; equal to serial up to the summation order of the norms).
        auto norm = [&](const Field &f) {
            if (!dist_)
                return f.norm();
            double s2 = 0;
            for (int i = 0; i < nodes; ++i)
                if (owned_[i])
                    s2 += f.row(i).squaredNorm();
            return std::sqrt(dist_->sum(s2));
        };
        for (double gamma_value : {0., 1.}) {
            Field x(nodes, moments);
            if (!dist_) {
                for (int i = 0; i < nodes; ++i)
                    for (int c = 0; c < moments; ++c)
                        x(i, c) = uniform(rng);
            } else
                distributed_random(x, rng, uniform);
            x /= norm(x);
            const int iterations = 60;
            for (int k = 0; k < iterations; ++k) {
                Field y = linear_operator(x, gamma_value);
                const double growth = norm(y);
                require(std::isfinite(growth) && growth > 0, "High-order spectral estimate failed");
                if (k >= iterations - 15)
                    radius = std::max(radius, growth);
                x = y / growth;
                if (dist_)
                    dist_->exchange(x);
            }
        }
        spectral_radius_ = radius;
    }
    // Negative real axis limits: roots of 1 + z + z^2/2 + z^3/6 = -1 and of 1 + z/2 + z^2/6 + z^3/48 = 0.
    const double limit = options.ssprk43 ? 5.149486147 : 2.512745327;
    return .9 * limit / spectral_radius_;
}
Field ElementWeno::angular_filter(const Field &a, double tau) const {
    if (options.filter_strength == 0 && options.sigma_s == 0 && options.sigma_a == 0)
        return a;
    Vec decay(moments);
    for (int k = 0; k < moments; ++k) {
        const int l = int(std::sqrt(double(active[k]) + .5)); // full index l^2 + l + m
        decay[k] = std::exp(-tau * (options.filter_strength * std::pow(double(l) / (pn.order + 1), 4) +
                                    options.sigma_a + (l > 0 ? options.sigma_s : 0.)));
    }
    return a * decay.asDiagonal();
}
// Exact solution of the local reaction ODE over tau, node by node (no spatial coupling):
//   d rho / dt = -sigma_a rho + q0,  d a_lm / dt = -(sigma_a + sigma_s [l >= 1] + f (l/(N+1))^4) a_lm  (l >= 1),
// with f the constant angular filter. rho(tau) = e^{-sigma_a tau} rho + q0 tau phi1(-sigma_a tau),
// phi1(z) = expm1(z) / z (phi1(0) = 1), so sigma_a -> 0 is exact. Without material this reduces to
// angular_filter() bit for bit. Absorbed mass m_i sigma_a int rho dt = m_i [rho (1 - e^{-sigma_a tau})
// + q0 tau (1 - phi1)] is computed analytically, not as a difference. Realizability: rho(tau) >= e^{-sigma_a tau}
// rho >= e^{-sigma_a tau} |J| >= e^{-(sigma_a + sigma_s + f_1) tau} |J| = |J(tau)| (q0 >= 0, phi1 > 0);
// binary64 rounding is corrected by the same ulp nudge as after the transport stages (counted in the ledger).
Vec ElementWeno::collided_source(double t, double *uncollided_mass, ElementWenoStats *stats) const {
    if (!options.uncollided_density) {
        if (uncollided_mass)
            *uncollided_mass = 0;
        return Vec::Zero(nodes);
    }
    const double start = seconds();
    std::vector<double> rho;
    options.uncollided_density(t, fcs_points_, rho);
    const double evaluated = seconds();
    require(rho.size() == fcs_points_.size(), "FCS: uncollided density has the wrong size");
    const int P = fcs_per_element_;
    Field local(cells, local_nodes);
    std::vector<double> mass(cells, 0.);
    std::atomic<bool> valid{true};
    parallel_for(cells, [&](int e) {
        thread_local Vec q;
        q.resize(P);
        bool any = false, ok = true;
        for (int k = 0; k < P; ++k) {
            const std::size_t at = std::size_t(e) * P + k;
            ok &= rho[at] >= 0 && std::isfinite(rho[at]);
            q[k] = fcs_weight_[at] * fcs_sigma_s_[at] * rho[at];
            mass[e] += fcs_weight_[at] * rho[at];
            any |= rho[at] != 0;
        }
        if (!ok)
            valid = false;
        // Elements outside the support of rho_u (most of the box at early times) skip the shape product.
        if (any)
            local.row(e) = (fcs_shape_ * q).transpose();
        else
            local.row(e).setZero();
    });
    require(valid.load(), "FCS: uncollided density must be finite and nonnegative");
    Field gathered = gather_load(local);
    Vec out(nodes);
    for (int i = 0; i < nodes; ++i)
        out[i] = gathered(i, 0) / fcs_node_mass_[i];
    if (stats)
        stats->t_fcs_density += evaluated - start, stats->t_fcs_quadrature += seconds() - evaluated;
    if (uncollided_mass) {
        double m = 0;
        for (double v : mass)
            m += v;
        *uncollided_mass = m;
    }
    return out;
}
Field ElementWeno::react(const Field &a, double tau, ElementWenoStats *stats, double source_time) const {
    const bool fcs = bool(options.uncollided_density);
    require(!fcs || std::isfinite(source_time), "FCS: react() needs the source time");
    const double react_start = seconds();
    const Vec fcs_source = fcs ? collided_source(source_time, nullptr, stats) : Vec();
    if (!options.material) {
        if (fcs) {
            // Constant sigma_a, sigma_s: angular_filter() gives the homogeneous part; the isotropic source enters
            // the density only, integrated exactly: rho += q tau phi1(-sigma_a tau).
            const double c0 = 1 / std::sqrt(4 * pi), z = -options.sigma_a * tau, decay = std::exp(z),
                         phi1 = z == 0 ? 1. : std::expm1(z) / z;
            Field out = angular_filter(a, tau);
            for (int i = 0; i < nodes; ++i) {
                const double q0 = fcs_source[i], rho = a(i, 0) / c0;
                out(i, 0) += c0 * q0 * tau * phi1;
                if (stats) {
                    stats->absorbed += lumped[i] * ((1 - decay) * rho + q0 * tau * (1 - phi1));
                    stats->injected += lumped[i] * q0 * tau;
                }
            }
            if (stats)
                stats->t_reaction += seconds() - react_start;
            return admissible_rows_public(std::move(out), stats);
        }
        if (stats && options.sigma_a > 0) { // ledger only; the state update is angular_filter() as before
            const double loss = -std::expm1(-options.sigma_a * tau);
            for (int i = 0; i < nodes; ++i)
                stats->absorbed += lumped[i] * loss * std::sqrt(4 * pi) * a(i, 0);
        }
        Field out = angular_filter(a, tau);
        return options.filter_strength == 0 && options.sigma_s == 0 && options.sigma_a == 0
                   ? out
                   : admissible_rows_public(std::move(out), stats);
    }
    const double start = seconds();
    const int q = a.cols();
    require(q == moments && a.rows() == nodes, "react: state size mismatch");
    std::vector<int> degree_of(q);
    std::vector<double> filter(q);
    for (int k = 0; k < q; ++k) {
        degree_of[k] = int(std::sqrt(double(active[k]) + .5));
        filter[k] = options.filter_strength * std::pow(double(degree_of[k]) / (pn.order + 1), 4);
    }
    Field out(nodes, q);
    std::vector<double> absorbed(nodes), injected(nodes);
    const double c0 = 1 / std::sqrt(4 * pi);
    parallel_for(nodes, [&](int i) {
        const double sa = node_sigma_a[i], ss = node_sigma_s[i], q0 = node_source[i] + (fcs ? fcs_source[i] : 0.);
        const double z = -sa * tau, decay0 = std::exp(z), phi1 = z == 0 ? 1. : std::expm1(z) / z;
        const double *in = &a.coeffRef(i, 0);
        double *o = &out.coeffRef(i, 0);
        const double rho = in[0] / c0;
        o[0] = decay0 * in[0] + c0 * q0 * tau * phi1;
        for (int k = 1; k < q; ++k)
            o[k] = std::exp(-tau * (sa + ss + filter[k])) * in[k];
        absorbed[i] = lumped[i] * (-std::expm1(z) * rho + q0 * tau * (1 - phi1));
        injected[i] = lumped[i] * q0 * tau;
    });
    if (stats) {
        for (int i = 0; i < nodes; ++i)
            stats->absorbed += absorbed[i], stats->injected += injected[i];
        stats->t_reaction += seconds() - start;
    }
    return admissible_rows_public(std::move(out), stats);
}
namespace {
// Nudges the stored low moments of every node into the M1 cone (rounding only).
Field admissible_rows(Field x, int nodes, ElementWenoStats *stats, const Vec &lumped) {
    // Rounding fix only: raise the stored density coefficient by ulps until the stored row itself projects into
    // the M1 cone (coefficient level, so no set_low() round trip can move it out again).
    std::atomic<long> nudged{0};
    Vec added = parallel_zero(nodes);
    const int q = x.cols();
    parallel_for(nodes, [&](int i) {
        double *row = &x.coeffRef(i, 0);
        const Eigen::Map<const Vec> view(row, q);
        if (exact::cone(project(view)))
            return;
        const double before = row[0];
        for (int k = 0; k < 4096 && !exact::cone(project(view)); ++k)
            row[0] = std::nextafter(row[0], std::numeric_limits<double>::infinity());
        if (!exact::cone(project(view)) && std::abs(row[0]) < 1e-290) {
            // Subnormal densities: start from |J| rounded upwards (see norm_up), then ulp steps as above.
            row[0] = std::max(row[0], norm_up(project(view)) / std::sqrt(4 * pi));
            for (int k = 0; k < 4096 && !exact::cone(project(view)); ++k)
                row[0] = std::nextafter(row[0], std::numeric_limits<double>::infinity());
        }
        cone_require(exact::cone(project(view)),
                     "State left the M1 cone beyond rounding (stored row " + std::to_string(i) + ", 4096 ulp): " +
                         describe(project(Eigen::Map<const Vec>(row, q))) + ", before nudge rho coefficient " +
                         std::to_string(before));
        added[i] = lumped[i] * std::sqrt(4 * pi) * (row[0] - before);
        ++nudged;
    });
    if (stats && nudged.load()) {
        stats->nudged_states += nudged.load();
        for (int i = 0; i < nodes; ++i)
            stats->nudge_mass += added[i];
    }
    return x;
}
// r <- old * a + fresh * (x + tau * r), row-parallel (the Eigen expression ran on one thread).
void combine(Field &r, const Field &x, const Field &a, double old, double fresh, double tau, int nodes) {
    const int q = r.cols();
    parallel_for(nodes, [&](int i) {
        double *ri = &r.coeffRef(i, 0);
        const double *xi = &x.coeffRef(i, 0), *ai = &a.coeffRef(i, 0);
        if (old == 0)
            for (int c = 0; c < q; ++c)
                ri[c] = fresh * (xi[c] + tau * ri[c]);
        else
            for (int c = 0; c < q; ++c)
                ri[c] = old * ai[c] + fresh * (xi[c] + tau * ri[c]);
    });
}
} // namespace
Field ElementWeno::advance_ssprk3(double time, double dt, const Field &input, ElementWenoStats *stats) const {
    // Without filter and reactions the input is used in place (no copy).
    const Field filtered = reacts() ? react(input, .5 * dt, stats, time + .25 * dt) : Field();
    const Field &a = reacts() ? filtered : input;
    // Effective weights of the stage rates: u^{n+1} = u^n + dt (L1 + L2 + 4 L3) / 6 (boundary mass ledger).
    auto stage = [&](double t, const Field &x, double old, double fresh, double weight) {
        Field r = rate(t, x, stats);
        if (dist_)
            dist_->exchange(r); // halo rows of the stage = the owners' values (deterministic, no garbage)
        if (stats)
            stats->boundary_outflow -= dt * weight * stats->last_boundary_mass_rate;
        const double combine_start = seconds();
        combine(r, x, a, old, fresh, dt, nodes);
        Field out = admissible_rows(std::move(r), nodes, stats, lumped);
        if (stats)
            stats->t_ssp_stage += seconds() - combine_start;
        return out;
    };
    const Field first = stage(time, a, 0, 1, 1. / 6);
    const Field second = stage(time + dt, first, .75, .25, 1. / 6);
    Field last = stage(time + .5 * dt, second, 1. / 3, 2. / 3, 2. / 3);
    if (reacts())
        return react(last, .5 * dt, stats, time + .75 * dt);
    return last; // no copy (a conditional operator would copy the lvalue)
}
// SSPRK(4,3) (Kraaijevanger; Spiteri and Ruuth): four forward Euler stages of dt/2 at times t, t+dt/2, t+dt,
// t+dt/2; u3 = 2/3 u + 1/3 (u2 + dt/2 L(u2)). Realizable for dt <= 2 dt_bound.
Field ElementWeno::advance_ssprk43(double time, double dt, const Field &input, ElementWenoStats *stats) const {
    const Field filtered = reacts() ? react(input, .5 * dt, stats, time + .25 * dt) : Field();
    const Field &a = reacts() ? filtered : input;
    const double half = .5 * dt;
    // Effective weights: u^{n+1} = u^n + dt (L1 + L2 + L3 + 3 L4) / 6.
    auto stage = [&](double t, const Field &x, double old, double fresh, double weight) {
        Field r = rate(t, x, stats);
        if (dist_)
            dist_->exchange(r); // halo rows of the stage = the owners' values (deterministic, no garbage)
        if (stats)
            stats->boundary_outflow -= dt * weight * stats->last_boundary_mass_rate;
        const double combine_start = seconds();
        combine(r, x, a, old, fresh, half, nodes);
        Field out = admissible_rows(std::move(r), nodes, stats, lumped);
        if (stats)
            stats->t_ssp_stage += seconds() - combine_start;
        return out;
    };
    const Field first = stage(time, a, 0, 1, 1. / 6);
    const Field second = stage(time + half, first, 0, 1, 1. / 6);
    const Field third = stage(time + dt, second, 2. / 3, 1. / 3, 1. / 6);
    Field last = stage(time + half, third, 0, 1, .5);
    if (reacts())
        return react(last, .5 * dt, stats, time + .75 * dt);
    return last; // no copy (a conditional operator would copy the lvalue)
}
Field ElementWeno::admissible_rows_public(Field x, ElementWenoStats *stats) const {
    return admissible_rows(std::move(x), nodes, stats, lumped);
}
ElementWeno::GhostEstimates ElementWeno::ghost_estimates(const Field &a, double alpha) const {
    require(options.positivity_projection && !options.projection_half_range,
            "ghost_estimates needs the full-sphere projection rule");
    GhostEstimates g;
    const int K = int(ghost_node_.size());
    g.node.resize(K), g.side.resize(K), g.m.resize(K);
    Vec sigma(moments);
    for (int c = 0; c < moments; ++c) {
        const int l = int(std::sqrt(double(active[c]) + .5));
        sigma[c] = l <= 1 ? 1. : std::exp(-alpha * std::pow(l / (pn.order + 1.), 4));
    }
    parallel_for(K, [&](int k) {
        const int side = ghost_side_[k], node = ghost_node_[k];
        const OutRule &out = out_rules_[side];
        const V3 n = boundaries_[side]->n;
        g.node[k] = node, g.side[k] = side;
        const Vec coeff = a.row(node).transpose();
        const Vec psi = out.harmonics * coeff, psif = out.harmonics * Vec(sigma.cwiseProduct(coeff));
        const Four U = project(coeff);
        const double bn = U[0] + n.dot(U.tail<3>());
        auto mom = [&](const double *z, int count) {
            Four m = Four::Zero();
            for (int j = 0; j < count; ++j) {
                const double v = z[j] * out.weight[j];
                m += v * Four(1, out.omega(j, 0), out.omega(j, 1), out.omega(j, 2));
            }
            return m;
        };
        auto c6 = [&](const Vec &y) {
            Vec p = y.head(out.outgoing).cwiseMax(0.);
            Four m = mom(p.data(), out.outgoing);
            const double ae = m[0] - n.dot(m.tail<3>());
            if (ae > bn)
                m *= bn > 0 ? bn / ae : 0.;
            return m;
        };
        const Four m_c6 = c6(psi);
        Vec z(psi.size());
        PositivityProjectionOptions opt;
        opt.tolerance = options.projection_tolerance;
        auto o = project_positive_moments(int(psi.size()), out.ox.data(), out.oy.data(), out.oz.data(),
                                          out.weight.data(), psi.data(), U, z.data(), opt, nullptr, &out.facets);
        Four m_full = m_c6;
        if (o.result.converged) {
            m_full = mom(z.data(), out.outgoing);
            if (o.augmented && o.atom_direction.dot(n) > 0)
                m_full += o.atom;
        }
        // Half range: target = outgoing half moments of the trace, outgoing points only, no atom.
        const Four Uh = mom(psi.data(), out.outgoing);
        PositivityProjectionOptions hopt = opt;
        hopt.augment = false;
        Vec zh(out.outgoing);
        auto oh = project_positive_moments(out.outgoing, out.ox.data(), out.oy.data(), out.oz.data(),
                                           out.weight.data(), psi.data(), Uh, zh.data(), hopt);
        const Four m_half = oh.result.converged ? mom(zh.data(), out.outgoing) : m_c6;
        Vec p = psi.head(out.outgoing).cwiseMax(0.);
        const Four raw = mom(p.data(), out.outgoing);
        const Four m_hybrid = raw[0] - n.dot(raw.tail<3>()) > bn ? m_full : m_c6;
        g.m[k] = {m_c6, m_full, m_half, m_hybrid, c6(psif)};
    });
    return g;
}
Field ElementWeno::interpolate(const std::function<Vec(const V3 &)> &f) const {
    Field a(nodes, moments);
    for (int i = 0; i < nodes; ++i)
        a.row(i) = reduce(f(node_position(i))).transpose();
    return a;
}
Vec ElementWeno::reduce(const Vec &full) const {
    require(full.size() == pn.q, "reduce expects a full PN vector");
    Vec out(moments);
    for (int k = 0; k < moments; ++k)
        out[k] = full[active[k]];
    return out;
}
Field ElementWeno::expand(const Field &a) const {
    require(a.cols() == moments, "expand expects active moments");
    Field out = Field::Zero(a.rows(), pn.q);
    parallel_for(int(a.rows()), [&](int i) {
        for (int k = 0; k < moments; ++k)
            out(i, active[k]) = a(i, k);
    });
    return out;
}
double ElementWeno::dropped_inflow(double time) const {
    if (moments == pn.q || faces_.empty())
        return 0;
    std::vector<char> is_active(pn.q, 0);
    for (int k : active)
        is_active[k] = 1;
    // Per-thread maxima (a maximum is exact in any order, so the result does not depend on the thread count).
    std::vector<std::array<double, 2>> partial(std::max(1, max_threads()), {0., 0.});
    parallel_for(int(faces_.size()), [&](int index) {
        const auto &f = faces_[index];
        const int side_id = 2 * f.direction + f.side;
        const FaceRule &rule = rules_[side_id];
        const Boundary &b = *boundaries_[side_id];
        double &kept = partial[thread_id()][0], &dropped = partial[thread_id()][1];
        V3 origin;
        element_origin(f.element, origin);
        for (int point = 0; point < rule.weights.size(); ++point) {
            V3 x = origin;
            for (int d = 0; d < dimension; ++d)
                x[d] += h * rule.points(point, d);
            const Vec d = b.data(inflow, x, time).d;
            if (d.isZero(0)) // vacuum: nothing dropped or kept (skips a dense q x r product per point)
                continue;
            const Vec full = b.llf_datum * d;
            for (int k = 0; k < pn.q; ++k) {
                double &slot_max = is_active[k] ? kept : dropped;
                slot_max = std::max(slot_max, std::abs(full[k]));
            }
        }
    });
    double kept = 0, dropped = 0;
    for (const auto &p : partial)
        kept = std::max(kept, p[0]), dropped = std::max(dropped, p[1]);
    return kept > 0 ? dropped / kept : dropped;
}
Field ElementWeno::project_l2(const std::function<Vec(const V3 &)> &f) const {
    const auto [xq, wq] = unit_gauss(2 * degree + 6);
    Field load = Field::Zero(nodes, moments);
    int points = 1;
    for (int d = 0; d < dimension; ++d)
        points *= xq.size();
    const double volume = std::pow(h, dimension);
    for (int e = 0; e < cells; ++e) {
        V3 origin;
        element_origin(e, origin);
        for (int p = 0; p < points; ++p) {
            V3 x = origin;
            double w = volume;
            Vec ref(dimension);
            int value = p;
            for (int d = 0; d < dimension; ++d, value /= xq.size()) {
                ref[d] = xq[value % xq.size()];
                w *= wq[value % xq.size()];
                x[d] += h * ref[d];
            }
            const Vec state = reduce(f(x));
            for (int l = 0; l < local_nodes; ++l) {
                int v = l;
                double phi = 1;
                for (int d = 0; d < dimension; ++d, v /= degree + 1)
                    phi *= bernstein(v % (degree + 1), degree, 0, ref[d]);
                load.row(connectivity_[e * local_nodes + l]) += w * phi * state.transpose();
            }
        }
    }
    return solve_mass(load);
}
Vec ElementWeno::evaluate(const Field &a, const V3 &x) const {
    int element = 0, scale = 1;
    Vec ref(dimension);
    for (int d = 0; d < dimension; ++d) {
        double s = (x[d] - options.origin[d]) / h - options.element_offset[d];
        int id = std::min(count[d] - 1, std::max(0, int(std::floor(s))));
        ref[d] = s - id;
        element += id * scale;
        scale *= count[d];
    }
    Vec out = Vec::Zero(a.cols());
    double values[3][13];
    for (int d = 0; d < dimension; ++d)
        for (int i = 0; i <= degree; ++i)
            values[d][i] = bernstein(i, degree, 0, ref[d]);
    for (int l = 0; l < local_nodes; ++l) {
        int v = l;
        double phi = 1;
        for (int d = 0; d < dimension; ++d, v /= degree + 1)
            phi *= values[d][v % (degree + 1)];
        out += phi * a.row(connectivity_[element * local_nodes + l]).transpose();
    }
    return out;
}
double ElementWeno::l2_density_error(const Field &a, const std::function<double(const V3 &)> &exact_density) const {
    const auto [xq, wq] = unit_gauss(degree + 5);
    int points = 1;
    for (int d = 0; d < dimension; ++d)
        points *= xq.size();
    const double volume = std::pow(h, dimension);
    // The point terms in parallel, summed afterwards in the serial (element, point) order: same result for every
    // thread count.
    std::vector<double> terms(std::size_t(cells) * points);
    parallel_for(cells, [&](int e) {
        V3 origin;
        element_origin(e, origin);
        for (int p = 0; p < points; ++p) {
            V3 x = origin;
            double w = volume;
            int value = p;
            for (int d = 0; d < dimension; ++d, value /= xq.size()) {
                x[d] += h * xq[value % xq.size()];
                w *= wq[value % xq.size()];
            }
            const double difference = project(evaluate(a, x))[0] - exact_density(x);
            terms[std::size_t(e) * points + p] = w * difference * difference;
        }
    });
    double sum = 0;
    for (const double t : terms)
        sum += t;
    return std::sqrt(sum);
}
Vec reflect_pn(const Angular &pn, const Vec &full, int direction) {
    require(full.size() == pn.q && direction >= 0 && direction < 3, "reflect_pn: invalid input");
    static thread_local std::vector<std::vector<double>> cache(3);
    static thread_local int cached_order = -1;
    if (cached_order != pn.order) {
        for (int d = 0; d < 3; ++d) {
            cache[d].assign(pn.q, 1.);
            const V3 u = V3(.37, -.52, .77).normalized();
            V3 m = u;
            m[d] = -m[d];
            const Vec y = harmonics(pn.order, u), r = harmonics(pn.order, m);
            for (int k = 0; k < pn.q; ++k)
                cache[d][k] = std::abs(y[k] - r[k]) > std::abs(y[k] + r[k]) ? -1. : 1.;
        }
        cached_order = pn.order;
    }
    Vec out = full;
    for (int k = 0; k < pn.q; ++k)
        out[k] *= cache[direction][k];
    return out;
}
Vec element_weno_smooth_exact(const Angular &pn, const V3 &x, double t) {
    Eigen::SelfAdjointEigenSolver<Mat> eig(pn.normal(V3::UnitX()));
    const double c = 1 / std::sqrt(4 * pi), s = .2 * c;
    Vec e0 = Vec::Zero(pn.q);
    e0[0] = 1;
    const Vec coefficients = eig.eigenvectors().transpose() * e0;
    Vec modes(pn.q);
    for (int k = 0; k < pn.q; ++k)
        modes[k] = coefficients[k] * std::sin(2 * pi * (x[0] - eig.eigenvalues()[k] * t));
    Vec out = s * (eig.eigenvectors() * modes);
    out[0] += c;
    return out;
}
} // namespace m1pn
