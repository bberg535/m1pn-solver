#include "adaptive_pn.hpp"
#include <iomanip>
#include <iostream>
#include <random>

using namespace m1pn;

void parity(AdaptivePn::State &a) {
    for (std::size_t l = 1; l < a.size(); l += 2)
        a[l] *= -1;
}
void divide(AdaptivePn::State &a, double length) {
    for (auto &block : a)
        block /= length;
}
AdaptivePn::State seeded(const AdaptivePn &scheme, int seed) {
    auto a = scheme.interpolate([](const V3 &) { return Vec::Zero(4 * 1024); });
    std::mt19937 gen(seed);
    std::normal_distribution<double> draw;
    for (auto &block : a)
        for (Eigen::Index i = 0; i < block.size(); ++i)
            block.data()[i] = draw(gen);
    divide(a, std::sqrt(scheme.inner_product(a, a)));
    return a;
}
double spatial_symbol_max(int count) {
    double largest = 0.;
    for (int k = 0; k < count; ++k) {
        const double theta = 2 * pi * k / count;
        largest = std::max(largest, 3 * std::abs(std::sin(theta)) / (2 + std::cos(theta)));
    }
    return largest;
}

int main(int argc, char **argv) {
    try {
        require(argc >= 7, "rank count N fixed|patch|checker|graded iterations CFL...");
        const int dim = std::stoi(argv[1]), count = std::stoi(argv[2]);
        const int n = std::stoi(argv[3]), iterations = std::stoi(argv[5]);
        const std::string mask = argv[4];
        require(dim >= 1 && dim <= 3 && iterations > 0, "Invalid dim or iterations");
        int cells = 1;
        for (int d = 0; d < dim; ++d)
            cells *= count;
        const Angular pn(n);
        std::vector<int> orders(cells, n);
        for (int k = 0; k < cells; ++k) {
            int id = k;
            const int x = k % count;
            if (mask == "patch") {
                orders[k] = x < count / 2 ? n : 1;
            } else if (mask == "checker") {
                int sum = 0;
                for (int d = 0; d < dim; ++d) {
                    sum += id % count;
                    id /= count;
                }
                orders[k] = (sum % 2 == 0 ? n : 1);
            } else if (mask == "graded") {
                const double distance = std::abs((2. * x) / count - 1.);
                orders[k] = 1 + 2 * static_cast<int>(.5 * (n - 1) * distance);
            } else {
                require(mask == "fixed", "invalid order map");
            }
        }
        const AdaptivePn scheme(pn, dim, count, orders);
        const double max_c_fixed = std::sqrt(-8 + 4 * std::sqrt(10.)) /
                                   (pn.speed * std::sqrt(dim) * spatial_symbol_max(count));
        std::cout << "dim,count,N,orders,DOFs,cfl,one_step_sigma_lower,20step_factor,adjoint_defect,"
                     "exact_cfl_fixed,cone_cfl,iterations\n" << std::setprecision(13);
        const auto starter = seeded(scheme, 1979);
        const auto other = seeded(scheme, 12841);
        for (int i = 6; i < argc; ++i) {
            const double cfl = std::stod(argv[i]), dt = cfl / count;
            auto step = [&](const AdaptivePn::State &x) { return scheme.advance(x, dt); };
            auto adjoint = [&](AdaptivePn::State y) {
                parity(y);
                y = step(y);
                parity(y);
                return y;
            };
            const double defect = std::abs(scheme.inner_product(step(starter), other) -
                                           scheme.inner_product(starter, adjoint(other)));
            auto q = starter;
            for (int it = 0; it < iterations; ++it) {
                q = adjoint(step(q));
                divide(q, std::sqrt(scheme.inner_product(q, q)));
            }
            auto result = step(q);
            const double gain = std::sqrt(scheme.inner_product(result, result));
            for (int it = 1; it < 20; ++it)
                result = step(result);
            const double gain20 = std::sqrt(scheme.inner_product(result, result));
            std::cout << dim << ',' << count << ',' << n << ',' << mask << ',' << scheme.dofs() << ',' << cfl
                      << ',' << gain << ',' << gain20 << ',' << defect << ',' << max_c_fixed
                      << ',' << count * 2 * scheme.euler_cone_step() << ',' << iterations << '\n';
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
