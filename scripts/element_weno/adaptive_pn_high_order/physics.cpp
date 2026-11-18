#include "adaptive_pn.hpp"
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sys/resource.h>

using namespace m1pn;
using Moments = Eigen::Matrix<double, 10, 1>;

double cpu_seconds() {
    timespec stamp{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stamp);
    return static_cast<double>(stamp.tv_sec) + 1e-9 * stamp.tv_nsec;
}

struct Problem {
    std::string name;
    int dim;
    V3 direction;
    double background;
    int components;
    Problem(const std::string &case_name, int dimension)
        : name(case_name), dim(dimension), direction(V3::Zero()), background(name == "broad" ? .2 : .04),
          components(name == "counter" ? 2 : 1) {
        require(name == "broad" || name == "beam" || name == "counter", "Unknown case");
        direction.head(dim).setOnes();
        direction.normalize();
    }
    double shape(const V3 &x) const {
        double s = 0;
        for (int d = 0; d < dim; ++d) {
            const double sine = std::sin(pi * (x[d] - (.27 + .08 * d)));
            s += sine * sine;
        }
        return std::exp(-(name == "broad" ? 12. : 18.) * s);
    }
    double angle(int component, const V3 &omega) const {
        const double mu = (component == 0 ? 1. : -1.) * direction.dot(omega);
        if (name == "broad")
            return 5. * std::pow(1 + mu, 4) / (64 * pi);
        constexpr double concentration = 24.;
        return concentration * std::exp(concentration * (mu - 1.)) /
               (2 * pi * (-std::expm1(-2 * concentration)) * components);
    }
    double kinetic(const V3 &x, const V3 &omega, double t) const {
        double angular = 0;
        for (int j = 0; j < components; ++j)
            angular += angle(j, omega);
        return angular * shape(x - t * omega);
    }
};

Moments angular_test(const V3 &omega) {
    const double a = omega[0], b = omega[1], c = omega[2], s = std::sqrt(2.);
    Moments m;
    m << 1., a, b, c, a*a, b*b, c*c, s*a*b, s*a*c, s*b*c;
    return m;
}

class Reference {
public:
    Reference(const Problem &p, int count, int cells, double end, int quadrature)
        : dim(p.dim), per_axis(count), values(cells, 10), weights(1 << dim, 1 << dim),
          transform(Mat::Zero(10, 9)) {
        const auto rule = sphere_rule(quadrature);
        std::vector<Moments> tests;
        for (const auto &point : rule)
            tests.push_back(point.weight * angular_test(point.omega));
        for (int k = 0; k < cells; ++k) {
            V3 x = V3::Zero();
            int id = k;
            for (int d = 0; d < dim; ++d) {
                x[d] = static_cast<double>(id % count) / count;
                id /= count;
            }
            Moments total = Moments::Zero();
            total[0] = p.background;
            total.segment<3>(4).setConstant(p.background / 3.);
            for (std::size_t n = 0; n < rule.size(); ++n)
                total += p.kinetic(x, rule[n].omega, end) * tests[n];
            values.row(k) = total.transpose();
        }
        for (const auto &q : sphere_rule(5))
            transform.noalias() += q.weight * angular_test(q.omega) * harmonics(2, q.omega).transpose();
        for (int i = 0; i < weights.rows(); ++i)
            for (int j = 0; j < weights.cols(); ++j) {
                double mass = 1.;
                for (int d = 0; d < dim; ++d)
                    mass /= ((i ^ j) & (1 << d) ? 6. : 3.) * count;
                weights(i, j) = mass;
            }
    }

    Eigen::Vector3d errors(const AdaptivePn &scheme, const AdaptivePn::State &values_h) const {
        Moments squares = Moments::Zero();
        for (int k = 0; k < scheme.cells(); ++k) {
            const Field local = scheme.element(values_h, k);
            const int q = std::min<int>(local.cols(), 9);
            Field diff = local.leftCols(q) * transform.leftCols(q).transpose();
            for (int i = 0; i < scheme.local_nodes(); ++i) {
                const V3 x = scheme.node_position(k, i);
                int node = 0, stride = 1;
                for (int d = 0; d < dim; ++d) {
                    node += static_cast<int>(std::lround(x[d] * per_axis)) * stride;
                    stride *= per_axis;
                }
                diff.row(i) -= values.row(node);
            }
            squares += (diff.array() * (weights * diff).array()).colwise().sum().transpose().matrix();
        }
        return {std::sqrt(squares[0]), std::sqrt(squares.segment<3>(1).sum()),
                std::sqrt(squares.tail<6>().sum())};
    }

private:
    int dim, per_axis;
    Field values;
    Mat weights, transform;
};

int main(int argc, char **argv) {
    try {
        require(argc == 11 || argc == 12, "case dim count N fixed|warm|cold steps lower upper pn|m1 quadrature [end]");
        const int dim = std::stoi(argv[2]), count = std::stoi(argv[3]), order = std::stoi(argv[4]);
        const int steps = std::stoi(argv[6]), quadrature = std::stoi(argv[10]);
        const double lower = std::stod(argv[7]), upper = std::stod(argv[8]);
        const double end = argc == 12 ? std::stod(argv[11]) : .2;
        const std::string policy = argv[5], limiter = argv[9];
        require(dim >= 1 && dim <= 3 && count >= 2 && steps >= 8 && steps % 8 == 0, "Invalid grid or steps");
        require(policy == "fixed" || policy == "warm" || policy == "cold", "Invalid adaptive mode");
        const bool limit = limiter == "m1";
        const Problem p(argv[1], dim);
        int cells = 1;
        for (int d = 0; d < dim; ++d)
            cells *= count;
        const Angular angular(order);
        Vec packet = Vec::Zero(angular.q);
        for (const auto &q : sphere_rule(order + 12))
            for (int j = 0; j < p.components; ++j)
                packet += q.weight * p.angle(j, q.omega) * harmonics(order, q.omega);
        Vec background = Vec::Zero(angular.q);
        background[0] = p.background / std::sqrt(4 * pi);
        const auto initialize = [&](const V3 &x) -> Vec { return background + p.shape(x) * packet; };

        const double start = seconds();
        auto scheme = std::make_unique<AdaptivePn>(angular, dim, count, std::vector<int>(cells, policy == "cold" ? 1 : order));
        auto state = scheme->interpolate(initialize);
        const Four initial_balance = scheme->integrated_moments(state);
        double energy = .5 * scheme->inner_product(state, state);
        const double initial_energy = energy;
        double energy_gain = 0, projection_gain = 0, work_adapt = 0, work_advance = 0, sum_dofs = 0;
        double cpu_adapt = 0., cpu_advance = 0.;
        int regrids = 0, limited = 0, high_fraction_count = 0, max_active_order = 1;
        const double dt = end / steps;
        for (int k = 0; k < steps; ++k) {
            if (policy != "fixed" && k % (steps / 8) == 0) {
                const double adapt_begin = seconds(), cpu_begin = cpu_seconds();
                auto orders = scheme->select_orders(state, lower, upper);
                const bool regrid = orders != scheme->orders();
                if (regrid) {
                    auto fresh = std::make_unique<AdaptivePn>(angular, dim, count, std::move(orders));
                    state = fresh->project_from(*scheme, state);
                    scheme = std::move(fresh);
                    ++regrids;
                }
                cpu_adapt += cpu_seconds() - cpu_begin;
                work_adapt += seconds() - adapt_begin;
                if (regrid) {
                    const double next = .5 * scheme->inner_product(state, state);
                    projection_gain = std::max(projection_gain, next - energy);
                    energy = next;
                }
            }
            sum_dofs += scheme->dofs();
            high_fraction_count += std::count(scheme->orders().begin(), scheme->orders().end(), order);
            max_active_order = std::max(max_active_order, *std::max_element(scheme->orders().begin(), scheme->orders().end()));
            const double advance_begin = seconds(), cpu_begin = cpu_seconds();
            state = scheme->advance(state, dt, limit, limit ? &limited : nullptr);
            cpu_advance += cpu_seconds() - cpu_begin;
            work_advance += seconds() - advance_begin;
            const double next = .5 * scheme->inner_product(state, state);
            energy_gain = std::max(energy_gain, next - energy);
            energy = next;
        }
        struct rusage usage {};
        getrusage(RUSAGE_SELF, &usage);
        const Reference reference(p, count, cells, end, quadrature);
        const auto error = reference.errors(*scheme, state);
        std::cout << std::setprecision(12)
                  << "case,dim,count,N,policy,mode,CFL,dt,steps,lower,upper,mean_dofs,final_dofs,peak_order_seen,"
                     "avg_fraction_at_maxN,regrids,mixed_faces,rho_error,J_error,P_error,moment_drift,"
                     "max_step_energy_increase_rel,max_projection_increase_rel,final_energy_over_initial,"
                     "limited_element_stages,min_cone_margin,m1_CFL_ceiling,step_seconds,adapt_seconds,"
                     "timed_seconds,cpu_advance,cpu_adapt,max_rss_kib,total_seconds\n";
        std::cout << p.name << ',' << dim << ',' << count << ',' << order << ',' << policy << ',' << limiter << ','
                  << count * dt << ',' << dt << ',' << steps << ',' << lower << ',' << upper << ',' << sum_dofs / steps
                  << ',' << scheme->dofs() << ',' << max_active_order << ',' << double(high_fraction_count) / (steps * cells)
                  << ',' << regrids << ',' << scheme->mixed_faces() << ',' << error[0] << ',' << error[1] << ','
                  << error[2] << ',' << maxabs(scheme->integrated_moments(state) - initial_balance) << ','
                  << energy_gain / initial_energy << ',' << projection_gain / initial_energy << ',' << energy / initial_energy
                  << ',' << limited << ',' << scheme->min_cone_margin(state) << ',' << count * 2 * scheme->euler_cone_step()
                  << ',' << work_advance << ',' << work_adapt << ',' << work_advance + work_adapt << ','
                  << cpu_advance << ',' << cpu_adapt << ',' << usage.ru_maxrss << ',' << seconds() - start << '\n';
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
