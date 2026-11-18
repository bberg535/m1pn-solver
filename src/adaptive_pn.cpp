#include "adaptive_pn.hpp"
#include "base_bar.hpp"
#include "m1.hpp"
#include <Eigen/SparseCholesky>
#include <numeric>

namespace m1pn {
namespace {

int moment_count(int order) { return (order + 1) * (order + 1); }

double mass_entry(int i, int j, int bit, double h) {
    return (i & bit) == (j & bit) ? h / 3 : h / 6;
}

} // namespace

struct AdaptivePn::DegreeSpace {
    int nodes = 0;
    std::vector<int> local_to_global;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factor;
};

AdaptivePn::AdaptivePn(const Angular &angular, int dimension, int count, std::vector<int> orders)
    : angular_(angular), dimension_(dimension), count_(count), corners_(0), h_(0), lump_(0),
      euler_cone_step_(0), orders_(std::move(orders)) {
    require(dimension >= 1 && dimension <= 3 && count >= 2, "Invalid periodic Q1 mesh");
    corners_ = 1 << dimension;
    h_ = 1. / count;
    std::array<int, 3> stride{1, 1, 1};
    int num_cells = 1;
    for (int d = 0; d < dimension; ++d) {
        stride[d] = num_cells;
        require(num_cells <= std::numeric_limits<int>::max() / corners_ / count, "Mesh index overflow");
        num_cells *= count;
    }
    require(orders_.size() == static_cast<std::size_t>(num_cells), "One PN order per element is required");
    for (int order : orders_)
        require(order >= 1 && order <= angular.order, "Element PN order out of range");

    neighbor_.resize(num_cells);
    for (int k = 0; k < num_cells; ++k)
        for (int d = 0; d < dimension; ++d)
            neighbor_[k][d] = ((k / stride[d]) % count == count - 1)
                                  ? k - (count - 1) * stride[d] : k + stride[d];

    mass_ = Mat::Ones(corners_, corners_);
    for (int d = 0; d < dimension; ++d)
        for (int i = 0; i < corners_; ++i)
            for (int j = 0; j < corners_; ++j)
                mass_(i, j) *= mass_entry(i, j, 1 << d, h_);
    lump_ = mass_.row(0).sum();

    for (int d = 0; d < dimension; ++d) {
        const int bit = 1 << d;
        convection_[d].resize(corners_, corners_);
        for (int i = 0; i < corners_; ++i) {
            if (i & bit)
                face_nodes_[d].push_back(i);
            for (int j = 0; j < corners_; ++j) {
                double entry = (j & bit) ? .5 : -.5;
                for (int trans = 0; trans < dimension; ++trans)
                    if (trans != d)
                        entry *= mass_entry(i, j, 1 << trans, h_);
                convection_[d](i, j) = entry;
            }
        }
        const auto &nodes = face_nodes_[d];
        face_mass_[d] = Mat::Ones(nodes.size(), nodes.size());
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
            for (int j = 0; j < static_cast<int>(nodes.size()); ++j)
                for (int trans = 0; trans < dimension; ++trans)
                    if (trans != d)
                        face_mass_[d](i, j) *= mass_entry(nodes[i], nodes[j], 1 << trans, h_);
    }

    viscosity_ = Mat::Zero(corners_, corners_);
    for (int i = 0; i < corners_; ++i)
        for (int j = i + 1; j < corners_; ++j) {
            V3 edge = V3::Zero(), reverse = V3::Zero();
            for (int d = 0; d < dimension; ++d) {
                edge[d] = convection_[d](i, j);
                reverse[d] = convection_[d](j, i);
            }
            double speed = std::max(edge.norm(), reverse.norm());
            while (!exact::cone(Four(speed, edge[0], edge[1], edge[2])) ||
                   !exact::cone(Four(speed, reverse[0], reverse[1], reverse[2])))
                speed = std::nextafter(speed, std::numeric_limits<double>::infinity());
            viscosity_(i, j) = viscosity_(j, i) = speed;
        }
    viscosity_sum_ = viscosity_.row(0).sum();
    euler_cone_step_ = lump_ / (2 * viscosity_sum_);

    flux_.resize(angular.order + 1);
    const int highest = *std::max_element(orders_.begin(), orders_.end());
    for (int degree = 1; degree <= highest; ++degree)
        for (int d = 0; d < dimension; ++d) {
            const int q = moment_count(degree);
            flux_[degree][d] = angular.A[d].topLeftCorner(q, q);
        }

    for (int degree = 0; degree <= highest; ++degree) {
        auto space = std::make_unique<DegreeSpace>();
        const int copies = num_cells * corners_;
        std::vector<int> parent(copies);
        std::iota(parent.begin(), parent.end(), 0);
        auto root = [&](int x) {
            while (parent[x] != x) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        };
        // Identify spatial DOFs only across faces carrying this angular degree on both sides.
        for (int k = 0; k < num_cells; ++k) {
            if (orders_[k] < degree)
                continue;
            for (int d = 0; d < dimension; ++d) {
                const int right = neighbor_[k][d], bit = 1 << d;
                if (orders_[right] < degree)
                    continue;
                for (int local : face_nodes_[d]) {
                    const int left_root = root(k * corners_ + local);
                    const int right_root = root(right * corners_ + (local ^ bit));
                    parent[left_root] = right_root;
                }
            }
        }

        auto &map = space->local_to_global;
        map.assign(copies, -1);
        std::vector<int> numbering(copies, -1);
        std::vector<Triplet> entries;
        for (int k = 0; k < num_cells; ++k) {
            if (orders_[k] < degree)
                continue;
            for (int i = 0; i < corners_; ++i) {
                int &global = numbering[root(k * corners_ + i)];
                if (global < 0)
                    global = space->nodes++;
                map[k * corners_ + i] = global;
            }
            for (int i = 0; i < corners_; ++i)
                for (int j = 0; j < corners_; ++j)
                    entries.emplace_back(map[k * corners_ + i], map[k * corners_ + j], mass_(i, j));
        }
        Eigen::SparseMatrix<double> mass(space->nodes, space->nodes);
        mass.setFromTriplets(entries.begin(), entries.end());
        space->factor.compute(mass);
        require(space->factor.info() == Eigen::Success, "Adaptive PN mass factorization failed");
        degrees_.push_back(std::move(space));
    }
}

AdaptivePn::~AdaptivePn() = default;

std::size_t AdaptivePn::dofs() const {
    std::size_t size = 0;
    for (int degree = 0; degree < static_cast<int>(degrees_.size()); ++degree)
        size += std::size_t(degrees_[degree]->nodes) * (2 * degree + 1);
    return size;
}

int AdaptivePn::mixed_faces() const {
    int faces = 0;
    for (int k = 0; k < cells(); ++k)
        for (int d = 0; d < dimension_; ++d)
            faces += orders_[k] != orders_[neighbor_[k][d]];
    return faces;
}

V3 AdaptivePn::node_position(int cell, int local_node) const {
    V3 x = V3::Zero();
    for (int d = 0; d < dimension_; ++d) {
        x[d] = ((cell % count_ + ((local_node >> d) & 1)) % count_) * h_;
        cell /= count_;
    }
    return x;
}

AdaptivePn::State AdaptivePn::zero_state() const {
    State result;
    for (int degree = 0; degree < static_cast<int>(degrees_.size()); ++degree)
        result.push_back(Field::Zero(degrees_[degree]->nodes, 2 * degree + 1));
    return result;
}

void AdaptivePn::check_shape(const State &state) const {
    require(state.size() == degrees_.size(), "Adaptive PN state has the wrong number of degrees");
    for (int degree = 0; degree < static_cast<int>(degrees_.size()); ++degree)
        require(state[degree].rows() == degrees_[degree]->nodes && state[degree].cols() == 2 * degree + 1,
                "Adaptive PN state block has the wrong shape");
}

Field AdaptivePn::element(const State &state, int cell) const {
    Field local(corners_, moment_count(orders_[cell]));
    for (int degree = 0; degree <= orders_[cell]; ++degree) {
        const int *map = &degrees_[degree]->local_to_global[cell * corners_];
        for (int i = 0; i < corners_; ++i)
            local.row(i).segment(degree * degree, 2 * degree + 1) = state[degree].row(map[i]);
    }
    return local;
}

void AdaptivePn::add_element(State &load, int cell, const Field &local) const {
    for (int degree = 0; degree <= orders_[cell]; ++degree) {
        const int *map = &degrees_[degree]->local_to_global[cell * corners_];
        for (int i = 0; i < corners_; ++i)
            load[degree].row(map[i]) += local.row(i).segment(degree * degree, 2 * degree + 1);
    }
}

void AdaptivePn::solve_mass(State &load) const {
    for (int degree = 0; degree < static_cast<int>(degrees_.size()); ++degree) {
        Field solution = degrees_[degree]->factor.solve(load[degree]);
        load[degree].swap(solution);
    }
}

AdaptivePn::State AdaptivePn::interpolate(const std::function<Vec(const V3 &)> &f) const {
    State state = zero_state();
    for (int k = 0; k < cells(); ++k)
        for (int i = 0; i < corners_; ++i) {
            const Vec value = f(node_position(k, i));
            require(value.size() >= moment_count(orders_[k]) && value.allFinite(), "Invalid PN initial coefficients");
            for (int degree = 0; degree <= orders_[k]; ++degree) {
                const int global = degrees_[degree]->local_to_global[k * corners_ + i];
                state[degree].row(global) = value.segment(degree * degree, 2 * degree + 1).transpose();
            }
        }
    return state;
}

AdaptivePn::State AdaptivePn::project_from(const AdaptivePn &old, const State &state) const {
    require(dimension_ == old.dimension_ && count_ == old.count_ && &angular_ == &old.angular_,
            "Projection requires the same mesh and angular basis");
    old.check_shape(state);
    if (orders_ == old.orders_)
        return state;
    State result = zero_state();
    for (int k = 0; k < cells(); ++k) {
        const Field before = old.element(state, k);
        Field load = Field::Zero(corners_, moment_count(orders_[k]));
        const int common = std::min(load.cols(), before.cols());
        load.leftCols(common).noalias() = mass_ * before.leftCols(common);
        add_element(result, k, load);
    }
    solve_mass(result);
    // All spatially unchanged degrees, including rho and J, transfer bit for bit.
    for (std::size_t degree = 0; degree < std::min(degrees_.size(), old.degrees_.size()); ++degree)
        if (degrees_[degree]->local_to_global == old.degrees_[degree]->local_to_global)
            result[degree] = state[degree];
    return result;
}

std::vector<int> AdaptivePn::select_orders(const State &state, double lower, double upper) const {
    check_shape(state);
    require(std::isfinite(upper) && lower >= 0 && lower < upper, "Invalid PN hysteresis thresholds");
    std::vector<int> next = orders_;
    for (int k = 0; k < cells(); ++k) {
        const int *map = &degrees_[1]->local_to_global[k * corners_];
        Field j(corners_, 3);
        for (int i = 0; i < corners_; ++i)
            j.row(i) = std::sqrt(4 * pi / 3) * state[1].row(map[i]);
        const double square = (j.array() * (mass_ * j).array()).sum() / (corners_ * lump_);
        require(std::isfinite(square), "Nonfinite first-moment sensor");
        const double norm = std::sqrt(std::max(0., square));
        if (norm > upper)
            next[k] = std::min(angular_.order, next[k] + 2);
        else if (norm < lower)
            next[k] = std::max(1, next[k] - 2);
    }
    return next;
}

AdaptivePn::State AdaptivePn::rate(const State &state, bool limit, AdaptivePnDiagnostics *stats) const {
    check_shape(state);
    if (stats)
        *stats = {};
    std::vector<Field> local, load;
    local.reserve(cells());
    load.reserve(cells());
    std::vector<std::array<Field, 3>> high_flux(limit ? cells() : 0);
    for (int k = 0; k < cells(); ++k) {
        local.push_back(element(state, k));
        load.push_back(Field::Zero(corners_, local.back().cols()));
        for (int d = 0; d < dimension_; ++d) {
            const Field flux = (flux_[orders_[k]][d] * local.back().transpose()).transpose();
            load.back().noalias() += convection_[d].transpose() * flux;
            if (limit) {
                high_flux[k][d].resize(corners_, 4);
                for (int i = 0; i < corners_; ++i)
                    high_flux[k][d].row(i) = project(flux.row(i).transpose()).transpose();
            }
        }
    }
    for (int left = 0; left < cells(); ++left)
        for (int d = 0; d < dimension_; ++d) {
            const int right = neighbor_[left][d];
            if (orders_[left] == orders_[right])
                continue;
            const auto &nodes = face_nodes_[d];
            const int ql = local[left].cols(), qr = local[right].cols();
            const int q = std::max(ql, qr), bit = 1 << d;
            Field x = Field::Zero(nodes.size(), q), y = x;
            for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
                x.row(i).head(ql) = local[left].row(nodes[i]);
                y.row(i).head(qr) = local[right].row(nodes[i] ^ bit);
            }
            const Field jump = y - x;
            const Field flux = .5 * ((flux_[std::max(orders_[left], orders_[right])][d] *
                                      (x + y).transpose()).transpose() - jump);
            const Field face_load = face_mass_[d] * flux;
            // The two projections cancel exactly in shared CG test DOFs. Scatter only the unmatched tail.
            for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
                if (ql > qr)
                    load[left].row(nodes[i]).tail(ql - qr) -= face_load.row(i).tail(ql - qr);
                else
                    load[right].row(nodes[i] ^ bit).tail(qr - ql) += face_load.row(i).tail(qr - ql);
            }
            if (stats)
                stats->face_dissipation += .5 * (jump.array() * (face_mass_[d] * jump).array()).sum();
        }
    State result = zero_state();
    for (int k = 0; k < cells(); ++k)
        add_element(result, k, load[k]);
    solve_mass(result);
    if (limit)
        limit_low(state, high_flux, result, stats);
    return result;
}

Field AdaptivePn::low_moments(const State &state) const {
    // Degrees 0 and 1 are assembled on the same space with identical numbering.
    Field low(state[0].rows(), 4);
    for (int i = 0; i < low.rows(); ++i) {
        Four coefficients;
        coefficients << state[0](i, 0), state[1].row(i).transpose();
        low.row(i) = project(coefficients).transpose();
    }
    return low;
}

void AdaptivePn::limit_low(const State &state, const std::vector<std::array<Field, 3>> &high_flux,
                           State &result, AdaptivePnDiagnostics *stats) const {
    const Field low = low_moments(state), target = low_moments(result);
    std::array<Field, 3> low_flux;
    for (int d = 0; d < dimension_; ++d)
        low_flux[d].resize(low.rows(), 4);
    for (int i = 0; i < low.rows(); ++i) {
        const Four u = low.row(i).transpose();
        cone_require(exact::cone(u), "Adaptive PN limiter input outside M1 cone");
        const Levermore closure(u);
        for (int d = 0; d < dimension_; ++d)
            low_flux[d].row(i) = closure.flux(V3::Unit(d)).transpose();
    }

    Field accepted = Field::Zero(low.rows(), 4), raw;
    if (stats)
        raw = accepted;
    for (int k = 0; k < cells(); ++k) {
        const int *map = &degrees_[0]->local_to_global[k * corners_];
        std::array<Field, 3> closure;
        for (int d = 0; d < dimension_; ++d) {
            closure[d] = high_flux[k][d];
            for (int i = 0; i < corners_; ++i)
                closure[d].row(i) -= low_flux[d].row(map[i]);
        }
        Field antidiffusion = Field::Zero(corners_, 4), base(corners_, 4);
        double factor = 1.;
        for (int i = 0; i < corners_; ++i) {
            const Four ui = low.row(map[i]).transpose(), qi = target.row(map[i]).transpose();
            Four weighted = Four::Zero(), correction = Four::Zero();
            for (int j = 0; j < corners_; ++j) {
                if (j == i)
                    continue;
                const Four uj = low.row(map[j]).transpose(), qj = target.row(map[j]).transpose();
                const double speed = viscosity_(i, j);
                Four difference = Four::Zero();
                V3 edge = V3::Zero();
                for (int d = 0; d < dimension_; ++d) {
                    const double cij = convection_[d](i, j), cji = convection_[d](j, i);
                    edge[d] = cij;
                    difference += cij * (low_flux[d].row(map[j]) - low_flux[d].row(map[i])).transpose();
                    // Weak divergence of the discontinuous, element-local closure.
                    correction += (cji * closure[d].row(j) - cij * closure[d].row(i)).transpose();
                }
                Four bar = .5 * (ui + uj) - difference / (2 * speed);
                if (!exact::cone(bar))
                    bar = stable_base_bar_outward(ui, uj, edge, speed, nullptr);
                weighted += speed * bar;
                correction += mass_(i, j) * (qi - qj) + speed * (ui - uj);
            }
            const Four bar = weighted / viscosity_sum_;
            const Four increment = correction / (2 * viscosity_sum_);
            if (!exact::cone(bar + increment))
                factor = std::min(factor, .999 * segment(bar, increment));
            antidiffusion.row(i) = correction.transpose();
            base.row(i) = (2 * viscosity_sum_ * (bar - ui)).transpose();
        }
        for (int i = 0; i < corners_; ++i) {
            accepted.row(map[i]) += base.row(i) + factor * antidiffusion.row(i);
            if (stats)
                raw.row(map[i]) += base.row(i) + antidiffusion.row(i);
        }
        if (stats) {
            stats->limited_elements += factor < 1.;
            stats->element_balance_defect = std::max(stats->element_balance_defect,
                                                      maxabs(antidiffusion.colwise().sum()));
        }
    }
    const double lumped = corners_ * lump_;
    if (stats)
        stats->raw_low_defect = maxabs(raw / lumped - target);
    for (int i = 0; i < low.rows(); ++i) {
        Four coefficients;
        set_low(coefficients, (accepted.row(i) / lumped).transpose());
        result[0](i, 0) = coefficients[0];
        result[1].row(i) = coefficients.tail<3>().transpose();
    }
}

AdaptivePn::State AdaptivePn::advance(const State &old, double dt, bool limit, int *limited_elements) const {
    require(dt > 0 && std::isfinite(dt), "Invalid time step");
    require(!limit || dt <= 2 * euler_cone_step_, "Adaptive PN step exceeds M1 cone CFL");
    State state = old;
    for (int stage = 0; stage < 4; ++stage) {
        AdaptivePnDiagnostics stats;
        const State slope = rate(state, limit, limited_elements ? &stats : nullptr);
        for (std::size_t degree = 0; degree < state.size(); ++degree) {
            state[degree] += (.5 * dt) * slope[degree];
            if (stage == 2)
                state[degree] = (2. / 3) * old[degree] + (1. / 3) * state[degree];
            require(state[degree].allFinite(), "Nonfinite adaptive PN stage");
        }
        if (limited_elements)
            *limited_elements += stats.limited_elements;
        if (limit) {
            const Field low = low_moments(state);
            for (int i = 0; i < low.rows(); ++i)
                cone_require(exact::cone(low.row(i).transpose()), "Adaptive PN stage outside M1 cone");
        }
    }
    return state;
}

double AdaptivePn::inner_product(const State &a, const State &b) const {
    check_shape(a);
    check_shape(b);
    double value = 0;
    for (int k = 0; k < cells(); ++k)
        value += (element(a, k).array() * (mass_ * element(b, k)).array()).sum();
    return value;
}

Four AdaptivePn::integrated_moments(const State &state) const {
    check_shape(state);
    return (corners_ * lump_) * low_moments(state).colwise().sum().transpose();
}

double AdaptivePn::min_cone_margin(const State &state) const {
    check_shape(state);
    const Field low = low_moments(state);
    double minimum = std::numeric_limits<double>::infinity();
    for (int i = 0; i < low.rows(); ++i)
        minimum = std::min(minimum, margin(low.row(i).transpose()));
    return minimum;
}

} // namespace m1pn
