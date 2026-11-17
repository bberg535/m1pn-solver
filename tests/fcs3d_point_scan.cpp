// Diagnosis (not a ctest): runs the 3D lattice uncollided time row (src/uncollided_lattice3d_core.hpp) on exactly the
// FCS quadrature points of the element-WENO solver on an E^3 grid of (0,7)^3 (degree p: p + 1 Gauss points per
// direction and element), optionally only on the elements z-slab [k0, k1). Reports points whose row takes longer than
// LIMIT seconds (watchdog), non-finite rows and, on SIGSEGV/SIGBUS/SIGFPE, the points the threads were working on.
//   m1pn_fcs3d_point_scan E [p] [k0 k1] [LIMIT]
#include "benchmark_cases.hpp"
#include "uncollided_lattice3d.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <omp.h>
#include <unistd.h>
using namespace m1pn;
namespace {
constexpr int kMaxThreads = 1024;
std::atomic<long long> current[kMaxThreads];
std::atomic<double> started[kMaxThreads];
std::vector<V3> *g_points = nullptr;
void report(int sig) {
    char buf[256];
    int n = snprintf(buf, sizeof buf, "\nSIGNAL %d; points in work:\n", sig);
    write(2, buf, n);
    for (int t = 0; t < kMaxThreads; ++t) {
        const long long k = current[t].load();
        if (k < 0)
            continue;
        const V3 &x = (*g_points)[k];
        n = snprintf(buf, sizeof buf, "  thread %d point %lld x = (%.17g, %.17g, %.17g)\n", t, k, x[0], x[1], x[2]);
        write(2, buf, n);
    }
    _exit(128 + sig);
}
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
} // namespace
int main(int argc, char **argv) {
    const int E = argc > 1 ? std::atoi(argv[1]) : 168, p = argc > 2 ? std::atoi(argv[2]) : 1;
    const int k0 = argc > 4 ? std::atoi(argv[3]) : 0, k1 = argc > 4 ? std::atoi(argv[4]) : E;
    const double limit = argc > 5 ? std::atof(argv[5]) : 60;
    const double T = 3.2, h = 7. / E;
    uncollided::Lattice3dUncollidedTable tab;
    tab.lattice = uncollided::lattice3d_cells([](int i, int j, int k) {
        const V3 c(i + .5, j + .5, k + .5);
        return cases::lattice3d_sigma_a(c) + cases::lattice3d_sigma_s(c);
    });
    const auto [g, gw] = uncollided::gauss_legendre(p + 1, 0, 1);
    const int per = g.size();
    std::vector<V3> pts;
    for (int k = k0; k < k1; ++k)
        for (int j = 0; j < E; ++j)
            for (int i = 0; i < E; ++i)
                for (int c = 0; c < per * per * per; ++c) {
                    V3 x(i * h, j * h, k * h);
                    for (int d = 0, r = c; d < 3; ++d, r /= per)
                        x[d] += h * g[r % per];
                    pts.push_back(x);
                }
    g_points = &pts;
    for (auto &c : current)
        c = -1;
    std::signal(SIGSEGV, report), std::signal(SIGBUS, report), std::signal(SIGFPE, report);
    const int M = tab.times_for(T);
    std::printf("E=%d p=%d slab [%d,%d) points %zu M=%d threads %d\n", E, p, k0, k1, pts.size(), M, omp_get_max_threads());
    std::fflush(stdout);
    std::atomic<bool> done{false};
    std::thread dog([&] {
        while (!done) {
            for (int s = 0; s < 30 && !done; ++s)
                sleep(1);
            const double t = now();
            for (int th = 0; th < kMaxThreads; ++th) {
                const long long k = current[th].load();
                if (k >= 0 && t - started[th].load() > limit) {
                    const V3 &x = pts[k];
                    std::printf("SLOW thread %d point %lld x = (%.17g, %.17g, %.17g) for %.0f s\n", th, k, x[0], x[1], x[2],
                                t - started[th].load());
                    std::fflush(stdout);
                }
            }
        }
    });
    std::atomic<long long> bad{0}, worked{0};
    std::atomic<double> slowest{0};
    const double t0 = now();
    std::vector<double> work, row;
#pragma omp parallel private(work, row)
    {
        const int th = omp_get_thread_num();
        work.resize(3 * std::size_t(M + 2)), row.resize(M + 1);
#pragma omp for schedule(dynamic, 64)
        for (long long k = 0; k < (long long)pts.size(); ++k) {
            if (tab.lattice.distance(pts[k]) >= M * tab.dt)
                continue;
            const double s = now();
            started[th] = s, current[th] = k;
            const double x[3] = {pts[k][0], pts[k][1], pts[k][2]};
            tab.lattice.with_view([&](const uncollided::Lattice3dView &v) {
                uncollided::lattice3d_row(v, x, tab.dt, M, work.data(), row.data());
                return 0;
            });
            current[th] = -1;
            const double d = now() - s;
            double old = slowest.load();
            while (d > old && !slowest.compare_exchange_weak(old, d)) {
            }
            ++worked;
            for (double r : row)
                if (!std::isfinite(r)) {
                    if (bad++ < 20)
                        std::printf("NONFINITE point %lld x = (%.17g, %.17g, %.17g)\n", k, x[0], x[1], x[2]);
                    break;
                }
        }
    }
    done = true;
    dog.join();
    std::printf("done: %lld rows, %lld non-finite, slowest row %.3f s, %.1f s total\n", worked.load(), bad.load(),
                slowest.load(), now() - t0);
    return bad ? 1 : 0;
}
