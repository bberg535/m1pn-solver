// Monte-Carlo reference for the lattice cases (docs/LATTICE_MC.md). Independent of the solver: kinetic transport with
// speed 1, isotropic scattering, cellwise constant sigma_a/sigma_s from src/benchmark_cases.hpp, isotropic source q0 = 1
// on the source cell switched on at t = 0, vacuum outside [0,7]^d (nothing re-enters), initial density --background.
//
// lattice-paul: fields depend on (x, y), directions on S^2 (slab infinite in z), like the solver and the uncollided
// analytics (src/uncollided_lattice.hpp). lattice-3d: (x, y, z) in [0,7]^3.
//
// Estimators (implicit capture: the weight decays with exp(-sigma_a s); scattering by free flight in sigma_s):
//   mass(t_k)       = int rho(t_k) dx                weight at t_k
//   absorbed(t_k)   = int_0^t_k int sigma_a rho dx dt weight lost on [0, t_k] (exact per segment)
//   rate(t_k)       = int sigma_a rho(t_k) dx         sigma_a(cell) * weight at t_k
//   outflow(t_k)    = weight that left before t_k
// and the same for the uncollided part (before the first scattering). Every history balances exactly:
// mass + absorbed + outflow = injected. Units are those of summary.json (rho = int_{S^2} psi).
//
// Reproducibility: history h of batch b draws from a counter-based stream (SplitMix64 of seed, stream, b, h); batch
// sums are combined in batch order, so the result is bitwise independent of the thread count.
#include "benchmark_cases.hpp"
#include "uncollided_lattice.hpp"
#include "uncollided_lattice3d.hpp"
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {
using m1pn::V3;
constexpr double pi = 3.14159265358979323846;
constexpr int N = 7; // cells per axis
constexpr double inf = std::numeric_limits<double>::infinity();

struct Rng {
    std::uint64_t s;
    static std::uint64_t mix(std::uint64_t z) {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    Rng(std::uint64_t seed, std::uint64_t stream, std::uint64_t batch, std::uint64_t index)
        : s(mix(mix(mix(seed + 0x9e3779b97f4a7c15ULL) ^ (stream * 0xd1b54a32d192ed03ULL)) ^
                mix(batch * 0x8cb92ba72f3d8dd7ULL + index))) {}
    std::uint64_t next() { return mix(s += 0x9e3779b97f4a7c15ULL); }
    double uniform() { return double(next() >> 11) * 0x1.0p-53; }    // [0, 1)
    double open() { return (double(next() >> 11) + 1) * 0x1.0p-53; } // (0, 1]
};

struct Geometry {
    std::string name;
    int dim = 2;
    std::vector<double> sa, ss; // per cell, i + 7 (j + 7 k)
    int source[3] = {0, 0, 0};
    double volume() const { return dim == 3 ? N * N * N : N * N; }
    int index(const int c[3]) const { return c[0] + N * (c[1] + (dim == 3 ? N * c[2] : 0)); }
};

Geometry make_geometry(const std::string &name) {
    Geometry g;
    g.name = name;
    int sources = 0;
    if (name == "lattice-paul") {
        g.dim = 2;
        g.sa.assign(N * N, 0), g.ss.assign(N * N, 0);
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                // Same point predicates as src/fast_main.cpp (lattice-paul), evaluated at the cell centre.
                const double x = i + .5, y = j + .5;
                const bool absorber = m1pn::cases::lattice_paul_absorber(x, y);
                g.sa[i + N * j] = absorber ? 10. : 0.;
                g.ss[i + N * j] = absorber ? 0. : 1.;
                if (m1pn::cases::lattice_paul_source(x, y))
                    g.source[0] = i, g.source[1] = j, ++sources;
            }
    } else if (name == "lattice-3d") {
        g.dim = 3;
        g.sa.assign(N * N * N, 0), g.ss.assign(N * N * N, 0);
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < N; ++i) {
                    const V3 x(i + .5, j + .5, k + .5);
                    g.sa[i + N * (j + N * k)] = m1pn::cases::lattice3d_sigma_a(x);
                    g.ss[i + N * (j + N * k)] = m1pn::cases::lattice3d_sigma_s(x);
                    if (m1pn::cases::lattice3d_source(x) > 0)
                        g.source[0] = i, g.source[1] = j, g.source[2] = k, ++sources;
                }
    } else
        throw std::runtime_error("unknown case " + name + " (lattice-paul, lattice-3d)");
    if (sources != 1)
        throw std::runtime_error("expected exactly one source cell");
    return g;
}

// Tallies per snapshot time.
enum Q { Mass, Absorbed, Rate, Outflow, Injected, UMass, UAbsorbed, URate, NQ };
const char *qname[NQ] = {"mass",           "absorbed", "rate",  "outflow", "injected", "uncollided_mass",
                         "uncollided_absorbed", "uncollided_rate"};

struct Counters {
    double collisions = 0, segments = 0, max_imbalance = 0;
    void add(const Counters &o) {
        collisions += o.collisions, segments += o.segments, max_imbalance = std::max(max_imbalance, o.max_imbalance);
    }
};

void direction(Rng &r, double o[3]) {
    const double mu = 2 * r.uniform() - 1, phi = 2 * pi * r.uniform(), s = std::sqrt(std::max(0., 1 - mu * mu));
    o[0] = s * std::cos(phi), o[1] = s * std::sin(phi), o[2] = mu;
}

// One history from (x, t0) with weight w0 (t0 < T); adds into acc[k * NQ + q] for all snapshot times t_k > t0.
// Conventions: a particle at time t_k on a segment (t, t_end] with t_k <= t_end counts as mass; outflow at time t counts
// for t_k > t only.
void history(const Geometry &g, const std::vector<double> &times, double x[3], double t0, double w0, Rng &r,
             double *acc, Counters &cnt) {
    const int K = int(times.size());
    const double T = times.back();
    int c[3] = {0, 0, 0};
    for (int a = 0; a < g.dim; ++a)
        c[a] = std::min(N - 1, std::max(0, int(std::floor(x[a]))));
    double o[3];
    direction(r, o);
    double w = w0, t = t0, eta = -std::log(r.open());
    bool collided = false;
    int first = 0; // first snapshot with t_k > t
    while (first < K && times[first] <= t)
        ++first;
    for (int k = first; k < K; ++k)
        acc[k * NQ + Injected] += w0;
    double lost_total = 0, out = 0;
    for (;;) {
        const int cell = g.index(c);
        const double sa = g.sa[cell], ss = g.ss[cell];
        double db = inf;
        int axis = -1;
        for (int a = 0; a < g.dim; ++a) {
            double d = inf;
            if (o[a] > 0)
                d = (c[a] + 1 - x[a]) / o[a];
            else if (o[a] < 0)
                d = (c[a] - x[a]) / o[a];
            if (d < db)
                db = d, axis = a;
        }
        db = std::max(db, 0.);
        const double ds = ss > 0 ? std::max(eta, 0.) / ss : inf, dt = T - t;
        int event; // 0 final time, 1 scattering, 2 face
        double l;
        if (dt <= ds && dt <= db)
            event = 0, l = dt;
        else if (ds <= db)
            event = 1, l = ds;
        else
            event = 2, l = db;
        cnt.segments += 1;
        const double tend = event == 0 ? T : t + l;
        for (int k = first; k < K; ++k) { // snapshots after t: absorption up to min(t_k, t_end), weight at t_k
            const double tk = times[k];
            const double lost = w * -std::expm1(-sa * (std::min(tk, tend) - t));
            acc[k * NQ + Absorbed] += lost;
            if (!collided)
                acc[k * NQ + UAbsorbed] += lost;
            if (tk <= tend) {
                const double wk = w - lost;
                acc[k * NQ + Mass] += wk, acc[k * NQ + Rate] += sa * wk;
                if (!collided)
                    acc[k * NQ + UMass] += wk, acc[k * NQ + URate] += sa * wk;
            }
        }
        const double lost = w * -std::expm1(-sa * l);
        lost_total += lost;
        w -= lost;
        for (int a = 0; a < 3; ++a)
            x[a] += l * o[a];
        t = tend;
        while (first < K && times[first] <= t)
            ++first;
        if (event == 0)
            break;
        if (event == 1) {
            direction(r, o);
            eta = -std::log(r.open());
            collided = true;
            cnt.collisions += 1;
            continue;
        }
        eta -= ss * l;
        const int step = o[axis] > 0 ? 1 : -1;
        x[axis] = double(c[axis] + (step > 0 ? 1 : 0)); // exactly on the face
        c[axis] += step;
        if (c[axis] < 0 || c[axis] >= N) { // left the domain at time t
            out = w;
            for (int k = first; k < K; ++k)
                acc[k * NQ + Outflow] += w;
            break;
        }
    }
    const double end_mass = out > 0 ? 0. : w;
    cnt.max_imbalance = std::max(cnt.max_imbalance, std::abs(end_mass + lost_total + out - w0) / w0);
}

struct Estimate {
    std::vector<double> mean, stderr_; // K * NQ, already scaled to physical units
    Counters counters;
    double histories = 0, seconds = 0;
};

// kind 0: source histories (t0 ~ U(0, T), x ~ U(source cell)), scale q0 |S| T / n per history.
// kind 1: initial background (t0 = 0, x ~ U(domain)), scale background |D| / n per history.
Estimate run(const Geometry &g, const std::vector<double> &times, int kind, double scale_total, long batches,
             long per_batch, std::uint64_t seed) {
    const int K = int(times.size());
    const double T = times.back();
    std::vector<double> sums(std::size_t(batches) * K * NQ, 0.);
    std::vector<Counters> counters(batches);
    const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long b = 0; b < batches; ++b) {
        double *acc = &sums[std::size_t(b) * K * NQ];
        Counters cnt;
        for (long h = 0; h < per_batch; ++h) {
            Rng r(seed, std::uint64_t(kind), std::uint64_t(b), std::uint64_t(h));
            double x[3] = {0, 0, 0}, t0 = 0;
            if (kind == 0) {
                t0 = T * r.uniform();
                for (int a = 0; a < g.dim; ++a)
                    x[a] = g.source[a] + r.uniform();
            } else
                for (int a = 0; a < g.dim; ++a)
                    x[a] = N * r.uniform();
            history(g, times, x, t0, 1., r, acc, cnt);
        }
        counters[b] = cnt;
    }
    Estimate e;
    e.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    e.histories = double(batches) * per_batch;
    e.mean.assign(K * NQ, 0.), e.stderr_.assign(K * NQ, 0.);
    const double scale = scale_total / per_batch; // batch estimate = scale * batch sum
    for (int q = 0; q < K * NQ; ++q) {
        double m = 0, m2 = 0;
        for (long b = 0; b < batches; ++b) { // Welford in batch order
            const double v = scale * sums[std::size_t(b) * K * NQ + q], d = v - m;
            m += d / (b + 1);
            m2 += d * (v - m);
        }
        e.mean[q] = m;
        e.stderr_[q] = std::sqrt(std::max(0., m2) / (batches - 1) / batches);
    }
    for (const auto &c : counters)
        e.counters.add(c);
    return e;
}

// Deterministic uncollided mass and rate at time t from the analytic module (validation): tensor Gauss rule with
// `panels` equal sub-panels per cell and axis, `order` points each.
std::pair<double, double> uncollided_quadrature(const Geometry &g, double t, int panels, int order) {
    const auto rule = m1pn::uncollided::gauss_legendre(order, 0, 1);
    const int cells = g.dim == 3 ? N * N * N : N * N, sub = g.dim == 3 ? panels * panels * panels : panels * panels;
    const int pts = g.dim == 3 ? order * order * order : order * order;
    std::vector<double> mass(cells, 0.), rate(cells, 0.);
    m1pn::uncollided::CellLattice2D l2;
    m1pn::uncollided::CellLattice3D l3;
    if (g.dim == 2) {
        l2 = m1pn::uncollided::lattice_paul(m1pn::cases::lattice_paul_absorber_cell);
        l2.source_i = g.source[0], l2.source_j = g.source[1];
    } else {
        l3 = m1pn::uncollided::lattice3d_cells([&](int i, int j, int k) {
            return g.sa[i + N * (j + N * k)] + g.ss[i + N * (j + N * k)];
        });
        for (int a = 0; a < 3; ++a)
            l3.source[a] = g.source[a];
    }
    const double hp = 1. / panels;
#pragma omp parallel for schedule(dynamic, 1)
    for (int cell = 0; cell < cells; ++cell) {
        const int ci = cell % N, cj = (cell / N) % N, ck = g.dim == 3 ? cell / (N * N) : 0;
        double m = 0;
        for (int s = 0; s < sub; ++s) {
            const int si = s % panels, sj = (s / panels) % panels, sk = g.dim == 3 ? s / (panels * panels) : 0;
            for (int p = 0; p < pts; ++p) {
                const int pi_ = p % order, pj = (p / order) % order, pk = g.dim == 3 ? p / (order * order) : 0;
                const double x = ci + hp * (si + rule.first[pi_]), y = cj + hp * (sj + rule.first[pj]);
                double wgt = rule.second[pi_] * rule.second[pj] * hp * hp, v;
                if (g.dim == 2)
                    v = l2.density(x, y, t);
                else {
                    const double z = ck + hp * (sk + rule.first[pk]);
                    wgt *= rule.second[pk] * hp;
                    v = l3.density(V3(x, y, z), t);
                }
                m += wgt * v;
            }
        }
        mass[cell] = m, rate[cell] = g.sa[cell] * m;
    }
    double M = 0, R = 0;
    for (int c = 0; c < cells; ++c)
        M += mass[c], R += rate[c];
    return {M, R};
}

std::vector<double> parse_list(const std::string &s) {
    std::vector<double> v;
    std::stringstream in(s);
    std::string item;
    while (std::getline(in, item, ','))
        if (!item.empty())
            v.push_back(std::stod(item));
    return v;
}

void json_array(std::ostream &os, const std::vector<double> &v, int K, int q) {
    os << "[";
    for (int k = 0; k < K; ++k)
        os << (k ? "," : "") << v[k * NQ + q];
    os << "]";
}
} // namespace

int main(int argc, char **argv) {
    try {
        std::string name = "lattice-paul", output, check_times_arg;
        double T = 3.2, background = 1e-10;
        long histories = 1000000, batches = 1000, snapshots = 32;
        std::uint64_t seed = 20261116;
        int check_panels = 0, check_order = 4;
        std::string times_arg;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc)
                    throw std::runtime_error("missing value for " + key);
                return argv[++i];
            };
            if (key == "--case")
                name = next();
            else if (key == "--time")
                T = std::stod(next());
            else if (key == "--times")
                times_arg = next();
            else if (key == "--snapshots")
                snapshots = std::stol(next());
            else if (key == "--histories")
                histories = std::stol(std::to_string(std::llround(std::stod(next()))));
            else if (key == "--batches")
                batches = std::stol(next());
            else if (key == "--seed")
                seed = std::stoull(next());
            else if (key == "--background")
                background = std::stod(next());
            else if (key == "--output")
                output = next();
            else if (key == "--check-uncollided")
                check_panels = std::stoi(next());
            else if (key == "--check-order")
                check_order = std::stoi(next());
            else if (key == "--check-times")
                check_times_arg = next();
            else if (key == "--help" || key == "-h") {
                std::cout << "m1pn_lattice_mc --case lattice-paul|lattice-3d --time T [--times t1,..,T | --snapshots K]\n"
                             "  --histories N --batches B --seed S --background RHO0 --output FILE.json\n"
                             "  --check-uncollided PANELS [--check-order G --check-times t1,..]  (analytic quadrature)\n";
                return 0;
            } else
                throw std::runtime_error("unknown option " + key);
        }
        if (!(T > 0) || batches < 2 || histories < batches || background < 0)
            throw std::runtime_error("need T > 0, batches >= 2, histories >= batches, background >= 0");
        std::vector<double> times = times_arg.empty() ? std::vector<double>() : parse_list(times_arg);
        if (times.empty())
            for (long k = 1; k <= snapshots; ++k)
                times.push_back(T * double(k) / double(snapshots));
        for (std::size_t k = 1; k < times.size(); ++k)
            if (!(times[k] > times[k - 1]))
                throw std::runtime_error("--times must increase");
        if (!(times.front() > 0) || times.back() != T)
            throw std::runtime_error("--times must lie in (0, T] and end with T");
        const Geometry g = make_geometry(name);
        const int K = int(times.size());
        const long per_batch = histories / batches;
        const Estimate src = run(g, times, 0, T /* q0 |S| T */, batches, per_batch, seed);
        Estimate bg;
        const long bg_per_batch = std::max(1L, per_batch / 64);
        if (background > 0)
            bg = run(g, times, 1, background * g.volume(), batches, bg_per_batch, seed);
        std::ostringstream js;
        js << std::setprecision(17);
        js << "{\"case\":\"" << name << "\",\"dimension\":" << g.dim << ",\"final_time\":" << T
           << ",\"background\":" << background << ",\"seed\":" << seed << ",\"batches\":" << batches
           << ",\"histories\":" << src.histories << ",\"background_histories\":" << (background > 0 ? bg.histories : 0)
           << ",\"threads\":" << omp_get_max_threads() << ",\"seconds\":" << src.seconds + bg.seconds
           << ",\"collisions_per_history\":" << src.counters.collisions / src.histories
           << ",\"segments_per_history\":" << src.counters.segments / src.histories
           << ",\"max_history_imbalance\":" << std::max(src.counters.max_imbalance, bg.counters.max_imbalance)
           << ",\"times\":[";
        for (int k = 0; k < K; ++k)
            js << (k ? "," : "") << times[k];
        js << "]";
        auto block = [&](const char *key, const Estimate &e) {
            js << ",\"" << key << "\":{";
            for (int q = 0; q < NQ; ++q) {
                js << (q ? "," : "") << "\"" << qname[q] << "\":";
                json_array(js, e.mean, K, q);
                js << ",\"" << qname[q] << "_stderr\":";
                json_array(js, e.stderr_, K, q);
            }
            js << "}";
        };
        block("source", src);
        if (background > 0)
            block("background", bg);
        Estimate total = src;
        if (background > 0)
            for (int q = 0; q < K * NQ; ++q)
                total.mean[q] += bg.mean[q], total.stderr_[q] = std::hypot(src.stderr_[q], bg.stderr_[q]);
        block("total", total);
        if (check_panels > 0) {
            std::vector<double> ct = check_times_arg.empty() ? std::vector<double>{T} : parse_list(check_times_arg);
            js << ",\"uncollided_check\":[";
            for (std::size_t i = 0; i < ct.size(); ++i) {
                int k = -1;
                for (int j = 0; j < K; ++j)
                    if (std::abs(times[j] - ct[i]) <= 1e-12 * T)
                        k = j;
                if (k < 0)
                    throw std::runtime_error("--check-times must be snapshot times");
                const auto t0 = std::chrono::steady_clock::now();
                const auto [M, R] = uncollided_quadrature(g, times[k], check_panels, check_order);
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                const double mm = src.mean[k * NQ + UMass], ms = src.stderr_[k * NQ + UMass];
                const double rm = src.mean[k * NQ + URate], rs = src.stderr_[k * NQ + URate];
                js << (i ? "," : "") << "{\"time\":" << times[k] << ",\"panels\":" << check_panels
                   << ",\"order\":" << check_order << ",\"seconds\":" << sec << ",\"mass_quadrature\":" << M
                   << ",\"mass_mc\":" << mm << ",\"mass_mc_stderr\":" << ms
                   << ",\"mass_z\":" << (ms > 0 ? (mm - M) / ms : 0.) << ",\"rate_quadrature\":" << R
                   << ",\"rate_mc\":" << rm << ",\"rate_mc_stderr\":" << rs
                   << ",\"rate_z\":" << (rs > 0 ? (rm - R) / rs : 0.) << "}";
            }
            js << "]";
        }
        js << "}\n";
        if (!output.empty()) {
            std::ofstream f(output);
            f << js.str();
            if (!f)
                throw std::runtime_error("cannot write " + output);
        }
        // Short report: final time.
        const int k = K - 1;
        std::cout << std::setprecision(8) << name << " T=" << T << " histories=" << src.histories
                  << " seconds=" << src.seconds + bg.seconds << "\n";
        for (int q : {Mass, Absorbed, Rate, Outflow, UMass, UAbsorbed, URate})
            std::cout << "  " << std::setw(20) << std::left << qname[q] << std::right << total.mean[k * NQ + q]
                      << " +- " << std::setprecision(3) << total.stderr_[k * NQ + q] << std::setprecision(8) << "\n";
        std::cout << "  balance (mass+absorbed+outflow-injected-initial): "
                  << total.mean[k * NQ + Mass] + total.mean[k * NQ + Absorbed] + total.mean[k * NQ + Outflow] -
                         src.mean[k * NQ + Injected] - (background > 0 ? bg.mean[k * NQ + Injected] : 0.)
                  << ", max per history " << std::max(src.counters.max_imbalance, bg.counters.max_imbalance) << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "m1pn_lattice_mc: " << e.what() << "\n";
        return 1;
    }
}
