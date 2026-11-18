#define main adaptive_physics_main
#include "physics.cpp"
#undef main
#include "element_weno.hpp"

int main(int argc, char **argv) {
    try {
        require(argc == 7, "case count N steps weno|plain end");
        const int count = std::stoi(argv[2]), n = std::stoi(argv[3]), steps = std::stoi(argv[4]);
        const double end = std::stod(argv[6]);
        const Problem problem(argv[1], 2);
        const Angular pn(n);
        Vec packet = Vec::Zero(pn.q);
        for (const auto &q : sphere_rule(n + 12))
            for (int j = 0; j < problem.components; ++j)
                packet += q.weight * problem.angle(j, q.omega) * harmonics(n, q.omega);
        Vec background = Vec::Zero(pn.q);
        background[0] = problem.background / std::sqrt(4 * pi);
        ElementWenoOptions options;
        options.degree = 1;
        options.dimension = 2;
        options.elements = count;
        options.periodic = true;
        options.stabilize = std::string(argv[5]) == "weno";
        options.limit = true;
        options.ssprk43 = true;
        ElementWeno native(pn, options);
        auto field = native.interpolate([&](const V3 &x) -> Vec { return background + problem.shape(x) * packet; });
        const double dt = end / steps;
        require(dt <= native.realizability_step(), "native M1 CFL");
        ElementWenoStats stats;
        const double begin = seconds(), cpu_begin = cpu_seconds();
        for (int j = 0; j < steps; ++j)
            field = native.advance(j * dt, dt, field, &stats);
        const double t = seconds() - begin, cpu = cpu_seconds() - cpu_begin;
        struct rusage usage {};
        getrusage(RUSAGE_SELF, &usage);
        const AdaptivePn common(pn, 2, count, std::vector<int>(count * count, n));
        const auto state = common.interpolate([&](const V3 &x) -> Vec {
            const int a = std::lround(x[0] * count), b = std::lround(x[1] * count);
            return field.row(a + b * count).transpose();
        });
        const auto err = Reference(problem, count, count * count, end, 32).errors(common, state);
        std::cout << std::setprecision(12)
                  << "case,count,N,steps,CFL,mode,rho_error,J_error,P_error,limit_count,native_CFL_m1,timed_seconds,cpu_seconds,max_rss_kib\n"
                  << problem.name << ',' << count << ',' << n << ',' << steps << ',' << dt * count << ',' << argv[5]
                  << ',' << err[0] << ',' << err[1] << ',' << err[2] << ',' << stats.limited_elements << ','
                  << native.realizability_step() * count << ',' << t << ',' << cpu << ',' << usage.ru_maxrss << '\n';
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
