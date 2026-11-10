#include "element_weno.hpp"
#include "base_bar.hpp"
#include "parallel.hpp"
#include <atomic>
#include <random>
#include <Eigen/Eigenvalues>
namespace m1pn {
namespace {
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
bool nudge_into_cone(Four &v) {
    if (exact::cone(v))
        return false;
    for (int k = 0; k < 256 && !exact::cone(v); ++k)
        v[0] = std::nextafter(v[0], std::numeric_limits<double>::infinity());
    cone_require(exact::cone(v), "State left the M1 cone beyond rounding");
    return true;
}
inline Four block(const Field &f, int row, int direction) {
    return Four(f(row, 4 * direction), f(row, 4 * direction + 1), f(row, 4 * direction + 2),
                f(row, 4 * direction + 3));
}
} // namespace

ElementWeno::ElementWeno(const Angular &angular, const ElementWenoOptions &o, Inflow in)
    : pn(angular), options(o), inflow(std::move(in)), degree(o.degree), dimension(o.dimension),
      elements(o.elements) {
    require(degree >= 1 && degree <= 12, "Element WENO supports Bernstein degrees 1 to 12");
    require(dimension >= 1 && dimension <= 3, "Element WENO supports dimensions 1 to 3");
    require(elements >= 2, "Element WENO needs at least two elements per direction");
    require(o.positivity_fraction > 0 && o.positivity_fraction < 1, "epsilon must lie in (0,1)");
    require(o.element_theta > 0 && o.element_theta < 1, "theta must lie in (0,1)");
    require(o.filter_strength >= 0 && std::isfinite(o.filter_strength), "Invalid filter strength");
    const int p = degree, one = p + 1;
    h = 1. / elements;
    per_direction = elements * p + (o.periodic ? 0 : 1);
    nodes = cells = local_nodes = 1;
    for (int d = 0; d < dimension; ++d) {
        nodes *= per_direction;
        cells *= elements;
        local_nodes *= one;
    }
    viscosity = o.viscosity_scale * h / (2. * p);

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
    std::array<int, 3> powers{1, elements, elements * elements};
    for (int e = 0; e < cells; ++e) {
        std::array<int, 3> id{e % elements, (e / elements) % elements, e / (elements * elements)};
        for (int l = 0; l < local_nodes; ++l) {
            int x = l, global = 0, scale = 1;
            for (int d = 0; d < dimension; ++d, x /= one, scale *= per_direction) {
                int coordinate = id[d] * p + x % one;
                if (o.periodic)
                    coordinate %= per_direction;
                global += coordinate * scale;
            }
            connectivity_[e * local_nodes + l] = global;
        }
        for (int d = 0; d < dimension; ++d)
            for (int side = 0; side < 2; ++side) {
                const int next = id[d] + (side ? 1 : -1);
                if (next >= 0 && next < elements)
                    neighbors_[e * 2 * dimension + 2 * d + side] = e + (next - id[d]) * powers[d];
                else if (o.periodic)
                    neighbors_[e * 2 * dimension + 2 * d + side] =
                        e + ((next + elements) % elements - id[d]) * powers[d];
                else
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
    // Lumped masses, one-dimensional consistent mass factor, boundary rules.
    lumped = Vec::Zero(nodes);
    const Vec row = mass_.rowwise().sum();
    for (int e = 0; e < cells; ++e)
        for (int l = 0; l < local_nodes; ++l)
            lumped[connectivity_[e * local_nodes + l]] += row[l];
    std::vector<Eigen::Triplet<double>> triplets;
    for (int e = 0; e < elements; ++e)
        for (int i = 0; i < one; ++i)
            for (int j = 0; j < one; ++j) {
                int a = e * p + i, b = e * p + j;
                if (o.periodic)
                    a %= per_direction, b %= per_direction;
                triplets.emplace_back(a, b, h * m1(i, j));
            }
    Eigen::SparseMatrix<double> m1d(per_direction, per_direction);
    m1d.setFromTriplets(triplets.begin(), triplets.end());
    mass_1d_.compute(m1d);
    require(mass_1d_.info() == Eigen::Success, "Bernstein mass factorization failed");
    {
        Eigen::SparseMatrix<double> L = mass_1d_.matrixL();
        L.makeCompressed();
        factor_outer_.assign(L.outerIndexPtr(), L.outerIndexPtr() + per_direction + 1);
        factor_inner_.assign(L.innerIndexPtr(), L.innerIndexPtr() + L.nonZeros());
        factor_values_.assign(L.valuePtr(), L.valuePtr() + L.nonZeros());
        const Vec diagonal = mass_1d_.vectorD(); // returned by value
        factor_diagonal_.assign(diagonal.data(), diagonal.data() + per_direction);
        factor_perm_.resize(per_direction);
        for (int k = 0; k < per_direction; ++k)
            factor_perm_[k] = mass_1d_.permutationP().indices()[k];
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
            }
        canonical_ = std::make_unique<CanonicalC1>(pn);
        for (int d = 0; d < dimension; ++d)
            for (int side = 0; side < 2; ++side) {
                V3 normal = V3::Zero();
                normal[d] = side ? 1 : -1;
                boundaries_[2 * d + side] = std::make_unique<Boundary>(pn, *canonical_, normal, false);
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
    dt_bound = (lumped.array() / rates.array()).minCoeff();
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
}

int ElementWeno::stride(int direction) const {
    int s = 1;
    for (int d = 0; d < direction; ++d)
        s *= per_direction;
    return s;
}
void ElementWeno::element_origin(int element, V3 &origin) const {
    origin = V3::Zero();
    int x = element;
    for (int d = 0; d < dimension; ++d, x /= elements)
        origin[d] = (x % elements) * h;
}
V3 ElementWeno::node_position(int node) const {
    V3 x = V3::Zero();
    int value = node;
    for (int d = 0; d < dimension; ++d, value /= per_direction)
        x[d] = (value % per_direction) * h / degree;
    return x;
}
Field ElementWeno::solve_mass(Field b) const {
    require(b.rows() == nodes, "Mass solve size mismatch");
    const int q = b.cols();
    for (int d = 0; d < dimension; ++d) {
        const int s = stride(d), lines = nodes / per_direction;
        parallel_for(lines, [&](int line) {
            int low = line % s, high = (line / s) * (s * per_direction);
            const int start = high + low;
            // P^T L D L^T P x = b, applied to whole rows (all PN components) at once.
            thread_local std::vector<double> work;
            work.resize(std::size_t(per_direction) * q);
            for (int k = 0; k < per_direction; ++k)
                std::copy_n(&b.coeffRef(start + k * s, 0), q, &work[std::size_t(factor_perm_[k]) * q]);
            for (int j = 0; j < per_direction; ++j) {
                const double *yj = &work[std::size_t(j) * q];
                for (int k = factor_outer_[j]; k < factor_outer_[j + 1]; ++k) {
                    const int i = factor_inner_[k];
                    if (i == j) continue;
                    const double l = factor_values_[k];
                    double *yi = &work[std::size_t(i) * q];
                    for (int c = 0; c < q; ++c) yi[c] -= l * yj[c];
                }
            }
            for (int j = 0; j < per_direction; ++j) {
                const double inv = 1. / factor_diagonal_[j];
                double *yj = &work[std::size_t(j) * q];
                for (int c = 0; c < q; ++c) yj[c] *= inv;
            }
            for (int j = per_direction - 1; j >= 0; --j) {
                double *yj = &work[std::size_t(j) * q];
                for (int k = factor_outer_[j]; k < factor_outer_[j + 1]; ++k) {
                    const int i = factor_inner_[k];
                    if (i == j) continue;
                    const double l = factor_values_[k];
                    const double *yi = &work[std::size_t(i) * q];
                    for (int c = 0; c < q; ++c) yj[c] -= l * yi[c];
                }
            }
            for (int k = 0; k < per_direction; ++k)
                std::copy_n(&work[std::size_t(factor_perm_[k]) * q], q, &b.coeffRef(start + k * s, 0));
        });
    }
    return b;
}
Field ElementWeno::gather_load(const Field &local) const {
    const int q = local.cols() / local_nodes;
    Field out(nodes, q);
    const double *data = local.data();
    parallel_for(nodes, [&](int i) {
        double *row = &out.coeffRef(i, 0);
        std::fill_n(row, q, 0.);
        for (int k = node_offsets_[i]; k < node_offsets_[i + 1]; ++k) {
            const double *source = data + std::size_t(node_entries_[k]) * q;
            for (int c = 0; c < q; ++c)
                row[c] += source[c];
        }
    });
    return out;
}
Vec ElementWeno::weno_gamma(const Field &a) const {
    Vec gamma = Vec::Ones(cells);
    if (!options.weno)
        return gamma;
    auto element_density = [&](int e) {
        Vec v(local_nodes);
        for (int l = 0; l < local_nodes; ++l)
            v[l] = a(connectivity_[e * local_nodes + l], 0);
        return v;
    };
    auto norm = [&](const Vec &v) { return std::sqrt(std::max(0., v.dot(seminorm_ * v))); };
    parallel_for(cells, [&](int e) {
        const Vec own = element_density(e);
        const double scale = norm(own);
        if (!(scale > 0))
            return;
        std::vector<Vec> candidates{own};
        std::vector<double> linear;
        for (int d = 0; d < dimension; ++d)
            for (int side = 0; side < 2; ++side) {
                const int next = neighbors_[e * 2 * dimension + 2 * d + side];
                if (next < 0)
                    continue;
                candidates.push_back(extension_[d][side] * element_density(next));
            }
        const int m = candidates.size() - 1;
        linear.assign(m + 1, options.weno_linear);
        linear[0] = 1 - m * options.weno_linear;
        Vec weights(m + 1);
        for (int l = 0; l <= m; ++l)
            weights[l] = linear[l] / (std::pow(norm(candidates[l]), options.weno_r) + options.weno_epsilon);
        weights /= weights.sum();
        Vec star = Vec::Zero(local_nodes);
        for (int l = 0; l <= m; ++l)
            star += weights[l] * candidates[l];
        gamma[e] = 1 - std::pow(std::min(1., norm(own - star) / scale), options.weno_q);
    });
    return gamma;
}
Field ElementWeno::stabilization_load(const Field &a, const Vec &gamma, ElementWenoStats *stats) const {
    const int q = a.cols();
    double clock = seconds();
    ElementWenoStats scratch;
    ElementWenoStats &timing = stats ? *stats : scratch;
    auto lap = [&](double &slot) { const double now = seconds(); slot += now - clock; clock = now; };
    std::array<Field, 3> gradient;
    for (int d = 0; d < dimension; ++d) {
        Field local(cells, local_nodes * q);
        parallel_for(cells, [&](int e) {
            Mat ae(local_nodes, q);
            for (int l = 0; l < local_nodes; ++l)
                ae.row(l) = a.row(connectivity_[e * local_nodes + l]);
            const Mat r = convection_[d] * ae;
            for (int l = 0; l < local_nodes; ++l)
                for (int c = 0; c < q; ++c)
                    local(e, l * q + c) = r(l, c);
        });
        lap(timing.t_gradient_local);
        Field gathered = gather_load(local);
        lap(timing.t_gradient_gather);
        gradient[d] = solve_mass(std::move(gathered));
        lap(timing.t_gradient_solve);
    }
    Field local(cells, local_nodes * q);
    parallel_for(cells, [&](int e) {
        Mat ae(local_nodes, q);
        for (int l = 0; l < local_nodes; ++l)
            ae.row(l) = a.row(connectivity_[e * local_nodes + l]);
        Mat r = stiffness_ * ae;
        for (int d = 0; d < dimension; ++d) {
            Mat g(local_nodes, q);
            for (int l = 0; l < local_nodes; ++l)
                g.row(l) = gradient[d].row(connectivity_[e * local_nodes + l]);
            r.noalias() -= gamma[e] * convection_[d].transpose() * g;
        }
        r *= viscosity;
        for (int l = 0; l < local_nodes; ++l)
            for (int c = 0; c < q; ++c)
                local(e, l * q + c) = r(l, c);
    });
    lap(timing.t_stab_local);
    return local;
}
void ElementWeno::add_boundary(double time, const Field &a, const Field &low, const Field &low_flux,
                               Field &high, Field &low_load, Field &node_flux_area, Vec &area,
                               Field &bar) const {
    struct Local {
        std::vector<int> global;
        Mat high, load, bar, flux_area;
        Vec area;
    };
    std::vector<Local> parts(faces_.size());
    Inflow atoms_only;
    atoms_only.atoms = inflow.atoms;
    parallel_for(int(faces_.size()), [&](int index) {
        const auto &f = faces_[index];
        const int side_id = 2 * f.direction + f.side;
        const FaceRule &rule = rules_[side_id];
        const Boundary &b = *boundaries_[side_id];
        const int nf = rule.local.size();
        Local &part = parts[index];
        part.global.resize(nf);
        part.high = Mat::Zero(nf, pn.q);
        part.load = Mat::Zero(nf, 4);
        part.bar = Mat::Zero(nf, 4);
        part.flux_area = Mat::Zero(nf, 3);
        part.area = Vec::Zero(nf);
        V3 origin;
        element_origin(f.element, origin);
        for (int n = 0; n < nf; ++n)
            part.global[n] = connectivity_[f.element * local_nodes + rule.local[n]];
        for (int point = 0; point < rule.weights.size(); ++point) {
            V3 x = origin;
            for (int d = 0; d < dimension; ++d)
                x[d] += h * rule.points(point, d);
            Incoming data;
            if (cached_density_) {
                data = density_inflow_[side_id];
                if (inflow.atoms) {
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
            Vec trace = Vec::Zero(pn.q);
            for (int n = 0; n < nf; ++n)
                trace.noalias() += rule.shape(n, point) * a.row(part.global[n]).transpose();
            const Vec outside = b.exterior(trace, data.d);
            const Vec integrand = .5 * pn.apply(b.n, trace - outside) + .5 * (outside - trace);
            for (int n = 0; n < nf; ++n) {
                const double weight = rule.weights[point] * rule.shape(n, point);
                if (weight == 0)
                    continue;
                const int i = part.global[n];
                part.high.row(n) += weight * integrand.transpose();
                const Four ui(low(i, 0), low(i, 1), low(i, 2), low(i, 3));
                const Four ext = inflow.exterior_state ? inflow.exterior_state(x, time, b.n, ui)
                                 : inflow.exterior_m1  ? inflow.exterior_m1(x, time)
                                                       : data.moments;
                cone_require(exact::cone(ext), "M1 exterior state outside cone");
                Four own = Four::Zero();
                for (int d = 0; d < dimension; ++d)
                    own += b.n[d] * block(low_flux, i, d);
                const Four outer = Levermore(ext).flux(b.n);
                const Four state = .5 * (ui + own) + .5 * (ext - outer);
                part.area[n] += weight;
                part.bar.row(n) += weight * state.transpose();
                for (int d = 0; d < dimension; ++d)
                    part.flux_area(n, d) += weight * b.n[d];
                part.load.row(n) += weight * (state - ui).transpose();
            }
        }
    });
    for (const auto &part : parts)
        for (std::size_t n = 0; n < part.global.size(); ++n) {
            const int i = part.global[n];
            high.row(i) += part.high.row(n);
            low_load.row(i) += part.load.row(n);
            bar.row(i) += part.bar.row(n);
            node_flux_area.row(i) += part.flux_area.row(n);
            area[i] += part.area[n];
        }
}
Field ElementWeno::rate(double time, const Field &a, ElementWenoStats *stats,
                        ElementWenoDebug *debug) const {
    require(a.rows() == nodes && a.cols() == pn.q && a.allFinite(), "Invalid element WENO state");
    const int q = pn.q;
    double clock = seconds();
    ElementWenoStats scratch;
    ElementWenoStats &timing = stats ? *stats : scratch;
    auto lap = [&](double &slot) {
        const double now = seconds();
        slot += now - clock;
        clock = now;
    };
    Field low(nodes, 4), low_flux(nodes, 12), high_flux(nodes, 12), closure(nodes, 12);
    low_flux.setZero();
    high_flux.setZero();
    std::array<Field, 3> angular_flux;
    const int block_rows = 256, blocks = (nodes + block_rows - 1) / block_rows;
    for (int d = 0; d < dimension; ++d) {
        angular_flux[d].resize(nodes, q);
        parallel_for(blocks, [&](int b) {
            const int first = b * block_rows, count = std::min(block_rows, nodes - first);
            angular_flux[d].middleRows(first, count).noalias() = a.middleRows(first, count) * pn.A[d];
        });
    }
    parallel_for(nodes, [&](int i) {
        const Four u = project(a.row(i).transpose());
        low.row(i) = u.transpose();
        const Levermore closure_state(u);
        for (int d = 0; d < dimension; ++d) {
            V3 n = V3::Zero();
            n[d] = 1;
            const Four fl = closure_state.flux(n);
            const Four fh = project(angular_flux[d].row(i).transpose());
            for (int c = 0; c < 4; ++c) {
                low_flux(i, 4 * d + c) = fl[c];
                high_flux(i, 4 * d + c) = fh[c];
            }
        }
    });
    closure = high_flux - low_flux;
    lap(timing.t_closure);

    // High-order PN residual and its stabilization.
    const Vec gamma = options.stabilize ? weno_gamma(a) : Vec(Vec::Ones(cells));
    lap(timing.t_weno);
    Field stabilization_local;
    if (options.stabilize)
        stabilization_local = stabilization_load(a, gamma, &timing);
    lap(timing.t_stabilization);
    Field residual_local(cells, local_nodes * q);
    parallel_for(cells, [&](int e) {
        Mat r = Mat::Zero(local_nodes, q);
        for (int d = 0; d < dimension; ++d) {
            Mat flux(local_nodes, q);
            for (int l = 0; l < local_nodes; ++l)
                flux.row(l) = angular_flux[d].row(connectivity_[e * local_nodes + l]);
            r.noalias() -= convection_[d] * flux;
        }
        for (int l = 0; l < local_nodes; ++l)
            for (int c = 0; c < q; ++c)
                residual_local(e, l * q + c) =
                    r(l, c) - (options.stabilize ? stabilization_local(e, l * q + c) : 0.);
    });
    Field residual = gather_load(residual_local);
    lap(timing.t_residual);
    Field boundary_high = Field::Zero(nodes, q), boundary_low = Field::Zero(nodes, 4);
    Field flux_area = Field::Zero(nodes, 3), boundary_bar = Field::Zero(nodes, 4);
    Vec boundary_area = Vec::Zero(nodes);
    if (!faces_.empty()) {
        add_boundary(time, a, low, low_flux, boundary_high, boundary_low, flux_area, boundary_area,
                     boundary_bar);
        residual += boundary_high;
    }
    lap(timing.t_boundary);
    Field high_rate = solve_mass(residual);
    Field moment_rate(nodes, 4);
    parallel_for(nodes, [&](int i) { moment_rate.row(i) = project(high_rate.row(i).transpose()).transpose(); });
    lap(timing.t_mass);

    // Element bar states and antidiffusive contributions.
    const int width = local_nodes * 4;
    Field element_limited(cells, width), element_raw;
    if (debug)
        element_raw.resize(cells, width);
    std::vector<double> element_factor(cells, 1.), raw_sum(cells, 0.), limited_sum(cells, 0.);
    std::vector<long> nudged(cells, 0);
    parallel_for(cells, [&](int e) {
        const int *map = &connectivity_[e * local_nodes];
        std::vector<Four> ubar(local_nodes), contribution(local_nodes);
        const Vec &total_viscosity = total_viscosity_;
        for (int i = 0; i < local_nodes; ++i) {
            const int ni = map[i];
            const Four ui(low(ni, 0), low(ni, 1), low(ni, 2), low(ni, 3));
            Four weighted = Four::Zero(), fi = Four::Zero();
            for (int j = 0; j < local_nodes; ++j) {
                if (j == i)
                    continue;
                const double dij = viscosity_ij_(i, j);
                const int nj = map[j];
                const Four uj(low(nj, 0), low(nj, 1), low(nj, 2), low(nj, 3));
                V3 edge = V3::Zero();
                Four flux_difference = Four::Zero(), closure_term = Four::Zero();
                for (int d = 0; d < dimension; ++d) {
                    const double cij = convection_[d](i, j), cji = convection_[d](j, i);
                    edge[d] = cij;
                    flux_difference += cij * (block(low_flux, nj, d) - block(low_flux, ni, d));
                    closure_term += cji * block(closure, ni, d) - cij * block(closure, nj, d);
                }
                if (dij > 0) {
                    Four pair_bar = .5 * (ui + uj) - flux_difference / (2 * dij);
                    if (!exact::cone(pair_bar))
                        pair_bar = stable_base_bar(ui, uj, edge, dij);
                    weighted += dij * pair_bar;
                    fi += dij * (ui - uj);
                }
                const Four qi(moment_rate(ni, 0), moment_rate(ni, 1), moment_rate(ni, 2), moment_rate(ni, 3));
                const Four qj(moment_rate(nj, 0), moment_rate(nj, 1), moment_rate(nj, 2), moment_rate(nj, 3));
                fi += mass_(i, j) * (qi - qj) + closure_term;
            }
            if (options.stabilize)
                fi -= project(stabilization_local.row(e).segment(i * q, q).transpose());
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
        std::vector<Four> limited = contribution;
        if (options.limit) {
            // Step 1: clip and scale the zeroth moment.
            double pos = 0, neg = 0;
            Vec clipped(local_nodes);
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
            if (debug) {
                const Four raw = lo + contribution[i];
                for (int c = 0; c < 4; ++c)
                    element_raw(e, i * 4 + c) = raw[c];
            }
        }
    });

    Field load = gather_load(element_limited);
    lap(timing.t_limiter);
    Field raw_load;
    if (debug)
        raw_load = gather_load(element_raw);
    // Boundary corrections h_i = P b^H - b^L - (int phi_i n) . G_i, limited node by node.
    double min_boundary = 1;
    long limited_boundary = 0;
    if (!faces_.empty())
        for (int i = 0; i < nodes; ++i) {
            if (boundary_area[i] <= 0)
                continue;
            Four correction = project(boundary_high.row(i).transpose()) -
                              Four(boundary_low(i, 0), boundary_low(i, 1), boundary_low(i, 2), boundary_low(i, 3));
            for (int d = 0; d < dimension; ++d)
                correction -= flux_area(i, d) * block(closure, i, d);
            Four bar(boundary_bar(i, 0), boundary_bar(i, 1), boundary_bar(i, 2), boundary_bar(i, 3));
            bar /= boundary_area[i];
            nudge_into_cone(bar);
            double theta = 1;
            if (options.limit) {
                const Four g = correction / boundary_area[i];
                if (!exact::cone(bar + g))
                    theta = options.element_theta * segment(bar, g);
                min_boundary = std::min(min_boundary, theta);
                limited_boundary += theta < 1;
            }
            const Four ui(low(i, 0), low(i, 1), low(i, 2), low(i, 3));
            const Four base = boundary_area[i] * (bar - ui);
            load.row(i) += (base + theta * correction).transpose();
            if (debug)
                raw_load.row(i) += (base + correction).transpose();
        }
    Field out = high_rate;
    parallel_for(nodes, [&](int i) {
        Vec row = out.row(i).transpose();
        set_low(row, Four(load.row(i).transpose()) / lumped[i]);
        out.row(i) = row.transpose();
    });
    lap(timing.t_finish);
    if (debug) {
        debug->ho_moment_rate = moment_rate;
        debug->unlimited_rate = raw_load;
        for (int i = 0; i < nodes; ++i)
            debug->unlimited_rate.row(i) /= lumped[i];
    }
    if (stats) {
        stats->min_gamma = std::min(stats->min_gamma, gamma.minCoeff());
        for (int e = 0; e < cells; ++e) {
            stats->min_limiter = std::min(stats->min_limiter, element_factor[e]);
            stats->limited_elements += element_factor[e] < 1;
            stats->max_raw_zero_sum = std::max(stats->max_raw_zero_sum, raw_sum[e]);
            stats->max_limited_zero_sum = std::max(stats->max_limited_zero_sum, limited_sum[e]);
            stats->nudged_states += nudged[e];
        }
        stats->min_boundary_limiter = std::min(stats->min_boundary_limiter, min_boundary);
        stats->limited_boundary_nodes += limited_boundary;
    }
    return out;
}
Field ElementWeno::linear_operator(const Field &x, double gamma_value) const {
    const int q = x.cols();
    std::array<Field, 3> flux;
    const int block_rows = 256, blocks = (nodes + block_rows - 1) / block_rows;
    for (int d = 0; d < dimension; ++d) {
        flux[d].resize(nodes, q);
        parallel_for(blocks, [&](int b) {
            const int first = b * block_rows, count = std::min(block_rows, nodes - first);
            flux[d].middleRows(first, count).noalias() = x.middleRows(first, count) * pn.A[d];
        });
    }
    const Field stabilization = options.stabilize ? stabilization_load(x, Vec::Constant(cells, gamma_value)) : Field();
    Field local(cells, local_nodes * q);
    parallel_for(cells, [&](int e) {
        Mat r = Mat::Zero(local_nodes, q);
        for (int d = 0; d < dimension; ++d) {
            Mat f(local_nodes, q);
            for (int l = 0; l < local_nodes; ++l)
                f.row(l) = flux[d].row(connectivity_[e * local_nodes + l]);
            r.noalias() -= convection_[d] * f;
        }
        for (int l = 0; l < local_nodes; ++l)
            for (int c = 0; c < q; ++c)
                local(e, l * q + c) = r(l, c) - (options.stabilize ? stabilization(e, l * q + c) : 0.);
    });
    return solve_mass(gather_load(local));
}
double ElementWeno::high_order_step() const {
    if (high_order_step_ > 0)
        return high_order_step_;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> uniform(-1, 1);
    double radius = 0;
    for (double gamma_value : {0., 1.}) {
        Field x(nodes, pn.q);
        for (int i = 0; i < nodes; ++i)
            for (int c = 0; c < pn.q; ++c)
                x(i, c) = uniform(rng);
        x /= x.norm();
        const int iterations = 60;
        for (int k = 0; k < iterations; ++k) {
            Field y = linear_operator(x, gamma_value);
            const double growth = y.norm();
            require(std::isfinite(growth) && growth > 0, "High-order spectral estimate failed");
            if (k >= iterations - 15)
                radius = std::max(radius, growth);
            x = y / growth;
        }
    }
    high_order_step_ = .9 * std::sqrt(3.) / radius;
    return high_order_step_;
}
Field ElementWeno::angular_filter(const Field &a, double tau) const {
    if (options.filter_strength == 0)
        return a;
    Vec decay(pn.q);
    for (int l = 0; l <= pn.order; ++l)
        decay.segment(l * l, 2 * l + 1).setConstant(
            std::exp(-tau * options.filter_strength * std::pow(double(l) / (pn.order + 1), 4)));
    return a * decay.asDiagonal();
}
Field ElementWeno::advance_ssprk3(double time, double dt, const Field &input, ElementWenoStats *stats) const {
    const Field a = angular_filter(input, .5 * dt);
    auto euler = [&](double t, const Field &x) { return Field(x + dt * rate(t, x, stats)); };
    auto admissible = [&](Field x) {
        std::atomic<long> nudged{0};
        parallel_for(nodes, [&](int i) {
            Four u = project(x.row(i).transpose());
            if (nudge_into_cone(u)) {
                Vec row = x.row(i).transpose();
                set_low(row, u);
                x.row(i) = row.transpose();
                ++nudged;
            }
        });
        if (stats)
            stats->nudged_states += nudged.load();
        return x;
    };
    const Field first = admissible(euler(time, a));
    const Field second = admissible(.75 * a + .25 * euler(time + dt, first));
    return angular_filter(admissible((1. / 3) * a + (2. / 3) * euler(time + .5 * dt, second)), .5 * dt);
}
Field ElementWeno::interpolate(const std::function<Vec(const V3 &)> &f) const {
    Field a(nodes, pn.q);
    for (int i = 0; i < nodes; ++i)
        a.row(i) = f(node_position(i)).transpose();
    return a;
}
Field ElementWeno::project_l2(const std::function<Vec(const V3 &)> &f) const {
    const auto [xq, wq] = unit_gauss(2 * degree + 6);
    Field load = Field::Zero(nodes, pn.q);
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
            const Vec state = f(x);
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
    for (int d = 0; d < dimension; ++d, scale *= elements) {
        double s = x[d] / h;
        int id = std::min(elements - 1, std::max(0, int(std::floor(s))));
        ref[d] = s - id;
        element += id * scale;
    }
    Vec out = Vec::Zero(a.cols());
    for (int l = 0; l < local_nodes; ++l) {
        int v = l;
        double phi = 1;
        for (int d = 0; d < dimension; ++d, v /= degree + 1)
            phi *= bernstein(v % (degree + 1), degree, 0, ref[d]);
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
    double sum = 0;
    for (int e = 0; e < cells; ++e) {
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
            sum += w * difference * difference;
        }
    }
    return std::sqrt(sum);
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
