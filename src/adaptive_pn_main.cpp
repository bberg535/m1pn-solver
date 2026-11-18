#include "adaptive_pn.hpp"
#include <iomanip>
#include <iostream>
#include <map>

using namespace m1pn;
namespace {

struct Problem {
    bool quadrupole;
    int dimension;
    V3 direction;

    explicit Problem(bool is_quadrupole, int dim)
        : quadrupole(is_quadrupole), dimension(dim), direction(V3::Zero()) {
        direction.head(dimension).setOnes();
        direction.normalize();
    }

    double spatial(const V3 &x) const {
        if (quadrupole)
            return .8 * std::cos(2 * pi * x[0]);
        double r2 = 0;
        for (int d = 0; d < dimension; ++d) {
            const double s = std::sin(pi * (x[d] - .27));
            r2 += s * s;
        }
        return std::exp(-12 * r2);
    }

    double angular(const V3 &omega) const {
        if (quadrupole)
            return .5 * (3 * omega[2] * omega[2] - 1) / (4 * pi);
        return 5. * std::pow(1 + direction.dot(omega), 4) / (64 * pi);
    }

    double background() const { return quadrupole ? 1. : .2; }
};

struct Run {
    std::unique_ptr<AdaptivePn> solver;
    AdaptivePn::State state;
    double mean_dofs = 0, max_energy_rise = 0, moment_drift = 0, elapsed = 0;
    int changes = 0, limited = 0;
};

Run evolve(const Angular &pn, const Problem &problem, int count, int cells, int order, bool adaptive,
           double end, bool limit, double lower, double upper) {
    Vec angular = Vec::Zero(pn.q);
    for (const auto &q : sphere_rule(pn.order + 3))
        angular += q.weight * problem.angular(q.omega) * harmonics(pn.order, q.omega);
    Vec background = Vec::Zero(pn.q);
    background[0] = problem.background() / std::sqrt(4 * pi);
    auto initial = [&](const V3 &x) -> Vec {
        Vec a = background + problem.spatial(x) * angular;
        return a;
    };

    Run run;
    const double start = seconds();
    run.solver = std::make_unique<AdaptivePn>(pn, problem.dimension, count, std::vector<int>(cells, order));
    run.state = run.solver->interpolate(initial);
    const Four balance = run.solver->integrated_moments(run.state);
    const double dt = std::min(.12 / count, run.solver->euler_cone_step());
    const int steps = static_cast<int>(std::ceil(end / dt));
    double energy = .5 * run.solver->inner_product(run.state, run.state);
    for (int k = 0; k < steps; ++k) {
        if (adaptive && k % 5 == 0) {
            auto orders = run.solver->select_orders(run.state, lower, upper);
            if (orders != run.solver->orders()) {
                auto next = std::make_unique<AdaptivePn>(pn, problem.dimension, count, std::move(orders));
                run.state = next->project_from(*run.solver, run.state);
                run.solver = std::move(next);
                ++run.changes;
            }
        }
        run.mean_dofs += static_cast<double>(run.solver->dofs()) / steps;
        run.state = run.solver->advance(run.state, end / steps, limit, limit ? &run.limited : nullptr);
        const double after = .5 * run.solver->inner_product(run.state, run.state);
        run.max_energy_rise = std::max(run.max_energy_rise, after - energy);
        energy = after;
    }
    run.moment_drift = maxabs(run.solver->integrated_moments(run.state) - balance);
    run.elapsed = seconds() - start;
    return run;
}

AdaptivePn::State exact_low(const Problem &problem, const Run &reference, int count, double end, int moments) {
    const auto rule = sphere_rule(24);
    std::map<std::array<int, 3>, Vec> cache;
    return reference.solver->interpolate([&](const V3 &x) -> Vec {
        std::array<int, 3> node{};
        for (int d = 0; d < problem.dimension; ++d)
            node[d] = static_cast<int>(std::lround(x[d] * count));
        auto [it, inserted] = cache.try_emplace(node);
        if (inserted) {
            Four u(problem.background(), 0, 0, 0);
            for (const auto &q : rule) {
                const double value = q.weight * problem.angular(q.omega) *
                                     problem.spatial(x - end * q.omega);
                u[0] += value;
                u.tail<3>() += value * q.omega;
            }
            it->second = Vec::Zero(moments);
            set_low(it->second, u);
        }
        return it->second;
    });
}

void print_run(const char *name, const Run &run, const Run &reference, const AdaptivePn::State &exact) {
    const auto &solver = *reference.solver;
    auto error = solver.project_from(*run.solver, run.state);
    error[0] -= exact[0];
    error[1] -= exact[1];
    for (std::size_t degree = 2; degree < error.size(); ++degree)
        error[degree].setZero();
    const Field current = error[1];
    error[1].setZero();
    const double rho = std::sqrt(4 * pi * solver.inner_product(error, error));
    error[0].setZero();
    error[1] = current;
    const double j = std::sqrt((4 * pi / 3) * solver.inner_product(error, error));
    std::cout << name << ',' << run.mean_dofs << ',' << run.solver->dofs() << ',' << run.solver->mixed_faces()
              << ',' << run.changes << ',' << rho << ',' << j << ',' << run.moment_drift << ','
              << run.max_energy_rise << ',' << run.solver->min_cone_margin(run.state) << ','
              << run.limited << ',' << run.elapsed << '\n';
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: m1pn_adaptive_pn [pulse|quadrupole] [dim] [cells/axis] [max odd N]"
                         " [end time] [pn|m1] [lower] [upper]\n";
            return 0;
        }
        require(argc <= 9, "Too many arguments (see --help)");
        const std::string name = argc > 1 ? argv[1] : "pulse";
        const int dim = argc > 2 ? std::stoi(argv[2]) : 2;
        const int count = argc > 3 ? std::stoi(argv[3]) : 20;
        const int order = argc > 4 ? std::stoi(argv[4]) : 3;
        const double end = argc > 5 ? std::stod(argv[5]) : .2;
        const std::string limiter = argc > 6 ? argv[6] : "pn";
        const double lower = argc > 7 ? std::stod(argv[7]) : .055;
        const double upper = argc > 8 ? std::stod(argv[8]) : .10;
        require(name == "pulse" || name == "quadrupole", "Expected pulse or quadrupole");
        require(limiter == "pn" || limiter == "m1", "Expected pn or m1");
        require(dim >= 1 && dim <= 3 && count >= 2 && count <= 256, "Invalid periodic grid");
        require(order >= 3 && order <= 15 && order % 2 == 1, "Expected odd max N from 3 through 15");
        require(std::isfinite(end) && end > 0 && end <= 10, "End time out of range");
        require(std::isfinite(upper) && lower >= 0 && lower < upper, "Invalid sensor thresholds");
        int cells = 1;
        for (int d = 0; d < dim; ++d)
            cells *= count;
        Budget{}.check(12.L * cells * (1 << dim) * (order + 1) * (order + 1) * sizeof(double),
                       "serial adaptive PN study estimate");
        const Angular pn(order);
        const Problem problem(name == "quadrupole", dim);
        const bool limit = limiter == "m1";
        const auto low = evolve(pn, problem, count, cells, 1, false, end, limit, lower, upper);
        const auto high = evolve(pn, problem, count, cells, order, false, end, limit, lower, upper);
        const auto adaptive = evolve(pn, problem, count, cells, order, true, end, limit, lower, upper);
        const auto answer = exact_low(problem, high, count, end, pn.q);
        std::cout << std::setprecision(12);
        std::cout << "mode,mean_dofs,final_dofs,final_mixed_faces,adaptations,rho_l2,J_l2,moment_drift,"
                     "max_energy_rise,min_final_cone_margin,limited_element_stages,seconds\n";
        print_run("P1", low, high, answer);
        print_run("Pmax", high, high, answer);
        print_run("adaptive", adaptive, high, answer);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
