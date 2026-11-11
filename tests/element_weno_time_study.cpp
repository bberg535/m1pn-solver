// Temporal order of SSPRK3 / SSPRK(4,3) transport with Strang half-step reactions (docs/ELEMENT_HETEROGENEOUS.md).
// Fixed periodic 2D mesh (P3, Q2, 8 x 8), smooth realizable data, smooth space-dependent sigma_a, sigma_s, q0.
// Error at T = 0.2 of all PN coefficients against the run with dt / 2^7 (self reference on the same mesh).
// Not a ctest: prints a table. Usage: m1pn_element_weno_time_study
#include "element_weno.hpp"
#include <cstdio>
#include <memory>
using namespace m1pn;
int main() {
    Angular pn(3);
    for (int reaction = 0; reaction < 2; ++reaction)
        for (int ssp43 = 0; ssp43 < 2; ++ssp43) {
            ElementWenoOptions o;
            o.degree = 2, o.dimension = 2, o.elements = 8, o.periodic = true, o.ssprk43 = ssp43;
            o.limit = true;
            if (reaction) {
                auto m = std::make_shared<ElementWenoMaterial>();
                m->sigma_a = [](const V3 &x) { return 2 + std::sin(2 * pi * x[0]) * std::cos(2 * pi * x[1]); };
                m->sigma_s = [](const V3 &x) { return 3 + std::cos(2 * pi * (x[0] + x[1])); };
                m->source = [](const V3 &x) { return 1 + .5 * std::sin(2 * pi * x[1]); };
                o.material = m;
            }
            ElementWeno s(pn, o);
            const Field a0 = s.interpolate([&](const V3 &x) {
                Vec r = Vec::Zero(pn.q);
                const double rho = 1 + .3 * std::sin(2 * pi * x[0]) * std::sin(2 * pi * x[1]);
                set_low(r, Four(rho, .3 * rho * std::cos(2 * pi * x[1]), .2 * rho * std::sin(2 * pi * x[0]), 0));
                for (int k = 4; k < pn.q; ++k) r[k] = .02 * std::sin(2 * pi * (x[0] + k * x[1]));
                return r;
            });
            const double T = .2, dt0 = std::min(s.step_bound(), T / 4);
            auto run = [&](double dt) {
                Field a = a0;
                const int n = int(std::llround(T / dt));
                for (int k = 0; k < n; ++k) a = s.advance(k * dt, dt, a);
                return a;
            };
            const int levels = 6, ref_level = 9;
            const Field ref = run(T / (std::ceil(T / dt0) * std::pow(2., ref_level)));
            std::printf("%s, %s:\n", reaction ? "with reaction (Strang)" : "transport only", ssp43 ? "SSPRK(4,3)" : "SSPRK3");
            double prev = 0;
            for (int l = 0; l < levels; ++l) {
                const double dt = T / (std::ceil(T / dt0) * std::pow(2., l));
                const double err = (run(dt) - ref).norm() / ref.norm();
                std::printf("  dt=%.4e  rel. error %.3e  EOC %s\n", dt, err,
                            l ? std::to_string(std::log2(prev / err)).substr(0, 5).c_str() : "  -  ");
                prev = err;
            }
        }
}
