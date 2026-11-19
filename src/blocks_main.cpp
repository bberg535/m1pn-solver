// m1pn_blocks: static hp blocks (docs/STATIC_HP_BLOCKS.md). The box is tiled by rectangular blocks, each an
// ElementWeno scheme with its own PN order; neighbouring blocks couple through interface sides (Rusanov term with the
// neighbour's PN trace, limiter bar state with its moments). All blocks advance in lockstep through the SSPRK(4,3)
// stages (ElementWeno::ssprk43_stage) with one global dt.
//
// Version 1: lattice-paul (2D, CPU, serial blocks with OpenMP inside), conforming interfaces (one element width h for
// all blocks), blocks from an order grid (merged into maximal rectangles of equal order).
//
// m1pn_blocks --order-grid FILE --elements-per-unit n --pn-max N --time T --output DIR
//             [--fcs analytic|off] [--cfl C] [--filter-strength F] [--threads K] [--sample-per-unit m] [--merge on|off]
//             [--h-grid FILE] [--gpu on] [--break-units on] [--nodal auto|on|off]
// --break-units: broken lines on the unit lines inside the blocks (the material lines; block faces lie on them too).
// --nodal: conforming interfaces (all blocks with one width) are coupled node by node like broken lines (no point
// loop; GPU: regular pipeline, partner term as face datum); auto = on when the widths agree.
// --gpu: every block runs the GPU time loop (ElementWenoGpu) on the one device, in lockstep through the stage halves;
// interface faces are done on the host (ElementWeno::add_boundary on the downloaded boundary rows).
// --h-grid: elements per unit per grid cell (same shape as the order grid; default --elements-per-unit everywhere).
// Blocks of different width meet at non-conforming interfaces: the neighbour trace is evaluated at the face points
// of each side, coarse faces with a finer neighbour use a composite rule split at the neighbour's element lines.
// FILE: whitespace separated integer grid of PN orders over [0,7]^2, row 0 = lowest y (as m1pn_fast --order-map);
// orders above --pn-max are clipped.
#include "angular.hpp"
#include "benchmark_cases.hpp"
#include "boundary.hpp"
#include "element_weno.hpp"
#include "m1.hpp"
#include "parallel.hpp"
#include "uncollided_lattice.hpp"
#ifdef M1PN_ELEMENT_GPU
#include "element_weno_gpu.hpp"
#endif
#include <omp.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <unordered_map>
#include <memory>
#include <sstream>

using namespace m1pn;

namespace {
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Rect {
    int i0, j0, ni, nj, order, per_unit; // grid cells [i0, i0 + ni) x [j0, j0 + nj); PN order, elements per unit
};

// Greedy maximal rectangles of equal order (row-major scan).
std::vector<Rect> rectangles(const std::vector<std::vector<int>> &grid, bool merge) {
    const int ny = int(grid.size()), nx = int(grid[0].size());
    std::vector<std::vector<char>> used(ny, std::vector<char>(nx, 0));
    std::vector<Rect> out;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            if (used[j][i])
                continue;
            const int order = grid[j][i];
            int ni = 1;
            while (merge && i + ni < nx && !used[j][i + ni] && grid[j][i + ni] == order)
                ++ni;
            int nj = 1;
            for (bool grow = merge; grow && j + nj < ny; ) {
                for (int k = 0; k < ni && grow; ++k)
                    grow = !used[j + nj][i + k] && grid[j + nj][i + k] == order;
                if (grow)
                    ++nj;
            }
            for (int b = 0; b < nj; ++b)
                for (int a = 0; a < ni; ++a)
                    used[j + b][i + a] = 1;
            out.push_back({i, j, ni, nj, order / 100000, order % 100000});
        }
    return out;
}

struct Block {
    Rect rect;
    V3 lo, hi;
    std::shared_ptr<Angular> pn; // shared by the blocks of one order
    std::unique_ptr<ElementWeno> scheme;
    std::shared_ptr<uncollided::LatticeUncollidedTable> table;
    Field a;
    ElementWenoStats stats;
    std::vector<int> position; // full PN index -> active position (-1: inactive)
#ifdef M1PN_ELEMENT_GPU
    std::unique_ptr<ElementWenoGpu> gpu;
#endif
    // Profile: wall time of the block's own work (stages, reactions; GPU: begin / end halves) and of the interface
    // trace evaluations inside it (atomic: a large block evaluates its faces in parallel).
    double t_work = 0, t_begin = 0, t_end = 0, t_react = 0;
    std::shared_ptr<std::atomic<long long>> trace_ns = std::make_shared<std::atomic<long long>>(0);
    std::shared_ptr<std::atomic<long long>> traces = std::make_shared<std::atomic<long long>>(0);
};
} // namespace

int main(int argc, char **argv) {
    try {
        std::string grid_file, h_file, output, fcs = "analytic";
        int per_unit = 0, pn_max = 21, threads = 1, sample_per_unit = 0, merge = 1, use_gpu = 0, break_units = 0;
        int split_breaks = 0; // EXPERIMENT: split Euler stage at broken-line nodes (docs/BREAK_DT.md)
        std::string nodal_mode = "auto";
        std::vector<int> cells_per_unit; // EXPERIMENT: graded mesh (as m1pn_fast --cells-per-unit), cut into the blocks
        double final_time = 3.2, cfl = 1., filter_strength = 0.;
        for (int k = 1; k + 1 < argc; k += 2) {
            const std::string key = argv[k], v = argv[k + 1];
            if (key == "--order-grid") grid_file = v;
            else if (key == "--elements-per-unit") per_unit = std::stoi(v);
            else if (key == "--pn-max") pn_max = std::stoi(v);
            else if (key == "--time") final_time = std::stod(v);
            else if (key == "--output") output = v;
            else if (key == "--fcs") fcs = v;
            else if (key == "--cfl") cfl = std::stod(v);
            else if (key == "--filter-strength") filter_strength = std::stod(v);
            else if (key == "--threads") threads = std::stoi(v);
            else if (key == "--sample-per-unit") sample_per_unit = std::stoi(v);
            else if (key == "--merge") merge = v == "on";
            else if (key == "--h-grid") h_file = v;
            else if (key == "--gpu") use_gpu = v == "on";
            else if (key == "--break-units") break_units = v == "on";
            else if (key == "--split-breaks") split_breaks = v == "on";
            else if (key == "--nodal") nodal_mode = v; // auto|on|off: conforming interfaces node by node
            else if (key == "--cells-per-unit") { // graded mesh, the same planes in x and y (experiment)
                std::stringstream list(v);
                std::string item;
                while (std::getline(list, item, ','))
                    cells_per_unit.push_back(std::stoi(item));
                require(cells_per_unit.size() == 7, "--cells-per-unit: seven entries (unit intervals of the lattice)");
            }
            else throw std::runtime_error("Unknown option " + key);
        }
        require(!grid_file.empty() && !output.empty(), "--order-grid and --output are required");
        require(per_unit > 0 || !h_file.empty() || !cells_per_unit.empty(), "--elements-per-unit, --h-grid or --cells-per-unit");
        require(cells_per_unit.empty() || h_file.empty(), "--cells-per-unit and --h-grid exclude each other");
        // Graded mesh: global planes; every block takes the planes inside its range (conforming block faces).
        std::vector<double> planes;
        if (!cells_per_unit.empty()) {
            planes.assign(1, 0.);
            for (int u = 0; u < 7; ++u)
                for (int k = 1; k <= cells_per_unit[u]; ++k)
                    planes.push_back(u + double(k) / cells_per_unit[u]);
            per_unit = *std::max_element(cells_per_unit.begin(), cells_per_unit.end()); // samples, merge key only
        }
        require(fcs == "analytic" || fcs == "off", "--fcs analytic|off");
        require(!std::filesystem::exists(output), "Output directory already exists: " + output);
        std::filesystem::create_directories(output);
        omp_set_num_threads(threads);
        const double start = now();

        // Order grid, width grid and blocks.
        auto read_grid = [](const std::string &file, const char *what) {
            std::vector<std::vector<int>> g;
            std::ifstream in(file);
            require(bool(in), std::string("Cannot read ") + what);
            for (std::string line; std::getline(in, line);) {
                std::istringstream row(line);
                std::vector<int> r;
                for (int v; row >> v;) r.push_back(v);
                if (!r.empty()) g.push_back(r);
            }
            require(!g.empty() && g.size() == g[0].size(), std::string(what) + ": square grid expected");
            return g;
        };
        std::vector<std::vector<int>> grid = read_grid(grid_file, "--order-grid");
        const int nb = int(grid.size());
        std::vector<std::vector<int>> hgrid = h_file.empty() ? std::vector<std::vector<int>>(nb, std::vector<int>(nb, per_unit))
                                                             : read_grid(h_file, "--h-grid");
        require(int(hgrid.size()) == nb, "--h-grid: same shape as the order grid");
        int finest = 0;
        for (int j = 0; j < nb; ++j)
            for (int i = 0; i < nb; ++i) {
                const int e = hgrid[j][i];
                require(e > 0 && (7 * e) % nb == 0, "--h-grid: 7 * elements per unit must be divisible by the grid size");
                finest = std::max(finest, e);
                grid[j][i] = 100000 * std::max(1, std::min(grid[j][i], pn_max)) + e; // merge key (order, width)
            }
        if (sample_per_unit <= 0)
            sample_per_unit = finest;
        const double cell = 7. / nb;
        const bool with_fcs = fcs == "analytic";

        std::vector<Block> blocks;
        for (const Rect &r : rectangles(grid, merge)) {
            Block b;
            b.rect = r;
            b.lo = V3(r.i0 * cell, r.j0 * cell, 0);
            b.hi = V3((r.i0 + r.ni) * cell, (r.j0 + r.nj) * cell, 0);
            blocks.push_back(std::move(b));
        }
        const int B = int(blocks.size());
        // Neighbour block containing x (on or next to a block face, slightly inside along the outward normal n).
        auto block_at = [&](const V3 &x) {
            for (int k = 0; k < B; ++k)
                if (x[0] >= blocks[k].lo[0] && x[0] <= blocks[k].hi[0] && x[1] >= blocks[k].lo[1] && x[1] <= blocks[k].hi[1])
                    return k;
            return -1;
        };
        // Current stage inputs of all blocks (set by the driver before each stage).
        std::vector<const Field *> current(B, nullptr);

        auto material = std::make_shared<ElementWenoMaterial>();
        material->sigma_a = [](const V3 &x) { return cases::lattice_paul_absorber(x[0], x[1]) ? 10. : 0.; };
        material->sigma_s = [](const V3 &x) { return cases::lattice_paul_absorber(x[0], x[1]) ? 0. : 1.; };
        material->source = [](const V3 &x) { return cases::lattice_paul_source(x[0], x[1]) ? 1. : 0.; };
        material->name = "lattice-paul (44)-(45)";
        if (with_fcs) {
            material->source = [](const V3 &) { return 0.; };
            material->name += ", source uncollided (FCS)";
        }
        material->subcells = 1;
        Budget budget;
        budget.bytes = std::size_t(1) << 40;
        std::map<int, std::shared_ptr<Angular>> angular_of;
        bool same_width = true;
        for (const Block &b : blocks)
            same_width &= b.rect.per_unit == blocks[0].rect.per_unit;
        same_width = same_width || !planes.empty(); // graded: one plane set, all block faces conforming
        require(nodal_mode == "auto" || nodal_mode == "on" || nodal_mode == "off", "--nodal auto|on|off");
        require(nodal_mode != "on" || same_width, "--nodal on: all blocks need the same width");
        const bool nodal = nodal_mode == "on" || (nodal_mode == "auto" && same_width);
        // Partner of an interface node (side, element, node) -> (block, node), per block, filled after construction.
        std::vector<std::unordered_map<std::uint64_t, std::pair<int, int>>> partner(B);
        auto key = [](int side, int element, int node) {
            return (std::uint64_t(std::uint32_t(element) * 6u + std::uint32_t(side)) << 32) | std::uint32_t(node);
        };
        for (int k = 0; k < B; ++k) {
            Block &b = blocks[k];
            const int order = b.rect.order;
            auto &shared = angular_of[order];
            if (!shared)
                shared = std::make_shared<Angular>(order, budget);
            b.pn = shared;
            ElementWenoOptions eo;
            eo.degree = 1;
            eo.dimension = 2;
            const int cells_per_cell = 7 * b.rect.per_unit / nb;
            const double h = 1. / b.rect.per_unit;
            int cx = b.rect.ni * cells_per_cell, cy = b.rect.nj * cells_per_cell;
            if (!planes.empty()) { // graded: the planes of the block's range (block faces lie on planes)
                for (int d = 0; d < 2; ++d) {
                    auto &P = eo.planes[d];
                    for (double x : planes)
                        if (x >= b.lo[d] - 1e-12 && x <= b.hi[d] + 1e-12)
                            P.push_back(x);
                    require(P.size() >= 2 && std::abs(P.front() - b.lo[d]) < 1e-12 && std::abs(P.back() - b.hi[d]) < 1e-12,
                            "--cells-per-unit: block faces must lie on planes");
                    P.front() = b.lo[d], P.back() = b.hi[d];
                }
                cx = int(eo.planes[0].size()) - 1, cy = int(eo.planes[1].size()) - 1;
            }
            eo.elements = cx;
            eo.elements_per_axis = {cx, cy, 0};
            eo.origin = b.lo;
            eo.length = planes.empty() ? cx * h : b.hi[0] - b.lo[0];
            eo.material = material;
            eo.periodic = false; // ElementWenoOptions defaults to a periodic mesh
            eo.weno = true;
            eo.limit = true;
            eo.weno_linear = 1e-3;
            eo.ssprk43 = true;
            eo.filter_strength = filter_strength;
            eo.reduce_angular = true;
            // eo.split_breaks is set below, once the block's broken lines are known (blocks without any ignore it)
            for (int d = 0; d < 2; ++d)
                for (int side = 0; side < 2; ++side) {
                    const double face = side ? b.hi[d] : b.lo[d];
                    eo.interface_side[2 * d + side] = face > 1e-12 && face < 7 - 1e-12;
                }
            if (break_units)
                for (int d = 0; d < 2; ++d) {
                    const int cd = d ? cy : cx;
                    for (int k2 = 1; k2 < cd; ++k2) {
                        const double xk = planes.empty() ? b.lo[d] + k2 * h : eo.planes[d][k2];
                        if (std::abs(xk - std::round(xk)) < 1e-9)
                            eo.breaks[d].push_back(k2);
                    }
                }
            eo.split_breaks = split_breaks && (!eo.breaks[0].empty() || !eo.breaks[1].empty());
            if (nodal) {
                eo.interface_partner = [&, k, key](int side, int element, int node) {
                    const auto it = partner[k].find(key(side, element, node));
                    require(it != partner[k].end(), "Interface node without partner");
                    return it->second;
                };
                // Partner node values in this block's layout (CPU: stage input; GPU: the host rows of the partner).
                auto map_row = [&, k](int pb, int pn, Vec &v) {
                    const ElementWeno &ns = *blocks[pb].scheme;
                    const ElementWeno &own = *blocks[k].scheme;
                    v = Vec::Zero(own.moments);
                    const double *row = current[pb]->data() + std::size_t(pn) * ns.moments;
                    for (int c = 0; c < ns.moments; ++c) {
                        const int full = ns.active[c];
                        if (full < own.pn.q && blocks[k].position[full] >= 0)
                            v[blocks[k].position[full]] = row[c];
                    }
                };
                eo.interface_value = map_row;
                eo.interface_low = [map_row](int pb, int pn, const V3 &normal, Four &u, Four &flux) {
                    thread_local Vec v;
                    map_row(pb, pn, v);
                    u = project(v);
                    const Levermore closure(u); // as the owner's closure step (flux per axis, then n)
                    flux = Four::Zero();
                    for (int d = 0; d < 3; ++d)
                        if (normal[d] != 0)
                            flux += normal[d] * closure.flux(V3::Unit(d));
                };
            }
            eo.interface_trace = [&, k, trace_ns = b.trace_ns, traces = b.traces](int side_id, const V3 &x, Vec &v) {
                const auto t0 = std::chrono::steady_clock::now();
                const int d = side_id / 2, side = side_id % 2;
                V3 y = x;
                y[d] += (side ? 1 : -1) * 1e-9; // just across the face
                const int nbk = block_at(y);
                require(nbk >= 0 && nbk != k, "Interface point without neighbour block");
                const ElementWeno &ns = *blocks[nbk].scheme;
                const Vec values = ns.evaluate(*current[nbk], x); // neighbour active layout
                const ElementWeno &own = *blocks[k].scheme;
                v = Vec::Zero(own.moments);
                // full index -> own active position (both use l^2 + l + m)
                for (int c = 0; c < ns.moments; ++c) {
                    const int full = ns.active[c];
                    if (full < own.pn.q && blocks[k].position[full] >= 0)
                        v[blocks[k].position[full]] = values[c];
                }
                *trace_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
                ++*traces;
            };
            eo.interface_split = [&, k](int side_id, const V3 &centre, double face_width) {
                const int d = side_id / 2, side = side_id % 2;
                V3 y = centre;
                y[d] += (side ? 1 : -1) * 1e-9;
                const int nbk = block_at(y);
                require(nbk >= 0 && nbk != k, "Interface face without neighbour block");
                const double hn = 1. / blocks[nbk].rect.per_unit;
                return std::max(1, int(std::lround(face_width / hn)));
            };
            if (with_fcs) {
                b.table = std::make_shared<uncollided::LatticeUncollidedTable>();
                b.table->lattice = uncollided::lattice_paul(cases::lattice_paul_absorber_cell);
                b.table->final_time = final_time;
                eo.uncollided_lattice = b.table;
                auto table = b.table;
                eo.uncollided_density = [table](double t, const std::vector<V3> &x, std::vector<double> &out) {
                    if (table->points.empty())
                        table->build(x, table->final_time);
                    table->density(t, x, out);
                };
                eo.uncollided_points = 2;
            }
            Inflow vacuum;
            vacuum.steady = true;
            b.scheme = std::make_unique<ElementWeno>(*b.pn, eo, vacuum);
            b.position.assign(b.pn->q, -1);
            for (int c = 0; c < b.scheme->moments; ++c)
                b.position[b.scheme->active[c]] = c;
            b.a = Field::Zero(b.scheme->nodes, b.scheme->moments);
            b.a.col(0).setConstant(1e-10 / std::sqrt(4 * pi));
        }
        if (nodal)
            for (int k = 0; k < B; ++k) {
                const ElementWeno &s = *blocks[k].scheme;
                const double hk = 1. / blocks[k].rect.per_unit;
                s.for_each_interface_node([&](int side, int element, int node) {
                    const V3 x = s.node_position(node), c = s.element_centre(element);
                    const int d = side / 2;
                    V3 probe = x + 1e-3 * (c - x); // tangentially inside the face element (the copy at a break)
                    probe[d] = x[d] + (side % 2 ? 1 : -1) * 1e-3 * hk; // just across the face
                    const int nbk = block_at(probe);
                    require(nbk >= 0 && nbk != k, "Interface node without neighbour block");
                    const int pn = blocks[nbk].scheme->node_near(x, probe);
                    require((blocks[nbk].scheme->node_position(pn) - x).norm() < 1e-9, "Non-conforming interface node");
                    partner[k][key(side, element, node)] = {nbk, pn};
                });
            }
#ifdef M1PN_ELEMENT_GPU
        if (use_gpu)
            for (Block &b : blocks) {
                b.gpu = std::make_unique<ElementWenoGpu>(*b.scheme);
                b.scheme->set_spectral_radius(b.gpu->spectral_radius()); // as m1pn_fast --gpu
            }
#else
        require(!use_gpu, "--gpu on: built without M1PN_ELEMENT_GPU");
#endif
        double dt = 1e300;
        for (const Block &b : blocks)
            dt = std::min(dt, b.scheme->step_bound());
        const double selected = cfl * dt; // steps as m1pn_fast: dt = min(selected, T - t)
        dt = selected;
        auto mass_of = [&](const Block &b, const Field &s) {
            double m = 0;
            for (int i = 0; i < b.scheme->nodes; ++i) m += b.scheme->lumped[i] * project(s.row(i).transpose())[0];
            return m;
        };
        double initial_mass = 0;
        for (const Block &b : blocks) initial_mass += mass_of(b, b.a);
        std::cerr << B << " blocks, dt " << selected << "\n";
        for (int k = 0; k < std::min(B, 3); ++k)
            std::cerr << "  block " << k << ": dt_bound " << blocks[k].scheme->dt_bound << ", realizability "
                      << blocks[k].scheme->realizability_step() << ", high order " << blocks[k].scheme->high_order_step()
                      << ", nodes " << blocks[k].scheme->nodes << ", moments " << blocks[k].scheme->moments << "\n";
        const double setup_seconds = now() - start;

        // Scheduling: blocks with at least 1/16 of the work run one after another with all threads (OpenMP inside
        // ElementWeno); the others run side by side, one thread each (the inner parallel_for is then serial).
        // Results do not depend on the schedule (blocks are independent within a stage, reductions are ordered).
        std::vector<int> large, small;
        {
            double total = 0;
            std::vector<double> cost(B);
            for (int k = 0; k < B; ++k)
                total += cost[k] = double(blocks[k].scheme->nodes) * blocks[k].scheme->moments;
            for (int k = 0; k < B; ++k)
                (threads > 1 && cost[k] < total / 16 ? small : large).push_back(k);
            std::sort(small.begin(), small.end(), [&](int a, int b) { return cost[a] > cost[b]; });
        }
        omp_set_max_active_levels(1);
        auto timed = [&](const std::function<void(int)> &work, int k) {
            const double t0 = now();
            work(k);
            blocks[k].t_work += now() - t0;
        };
        auto for_blocks = [&](const std::function<void(int)> &work) {
            for (int k : large)
                timed(work, k);
            std::exception_ptr error;
#pragma omp parallel for schedule(dynamic, 1)
            for (int m = 0; m < int(small.size()); ++m) {
                try {
                    timed(work, small[m]);
                } catch (...) {
#pragma omp critical(m1pn_blocks_error)
                    if (!error) error = std::current_exception();
                }
            }
            if (error)
                std::rethrow_exception(error);
        };
        std::cerr << large.size() << " large blocks (all threads), " << small.size() << " small blocks (side by side)\n";
        const int large_count = int(large.size()), small_count = int(small.size());
        // Lockstep SSPRK(4,3) with the reaction half steps (ElementWeno::advance_ssprk43).
        double t = 0;
        int steps = 0;
        std::vector<Field> base(B), x(B), y(B);
#ifdef M1PN_ELEMENT_GPU
        if (use_gpu) {
            for (int k = 0; k < B; ++k) {
                blocks[k].gpu->begin(blocks[k].a);
                current[k] = &blocks[k].gpu->host_state(); // boundary rows of the stage input after lockstep_begin
            }
            while (t < final_time) {
                dt = std::min(selected, final_time - t);
                auto clock = [](double &slot, const std::function<void()> &f) {
                    const double t0 = now();
                    f();
                    slot += now() - t0;
                };
                for (Block &b : blocks)
                    if (b.scheme->reacts_public())
                        clock(b.t_react, [&] { b.gpu->lockstep_react(.5 * dt, t + .25 * dt, b.stats); });
                // Context streams (experiment/big-rebuilds): the blocks of a phase run side by side from host threads;
                // all begins finish before any end reads a neighbour's boundary rows. M1PN_BLOCKS_SERIAL=1: one by one.
                static const bool serial = [] {
                    const char *e = std::getenv("M1PN_BLOCKS_SERIAL");
                    return e && std::atoi(e) != 0;
                }();
                auto each_block = [&](const std::function<void(Block &)> &work) {
                    if (serial || B == 1) {
                        for (Block &b : blocks)
                            work(b);
                        return;
                    }
                    // Nested: `outer` blocks at a time, each with threads / outer threads for its host phases.
                    const int outer = std::min(B, threads), inner = std::max(1, threads / outer);
                    static std::vector<int> by_cost; // largest blocks first (dynamic schedule)
                    if (int(by_cost.size()) != B) {
                        by_cost.resize(B);
                        for (int k = 0; k < B; ++k) by_cost[k] = k;
                        std::sort(by_cost.begin(), by_cost.end(), [&](int a, int c) {
                            return double(blocks[a].scheme->nodes) * blocks[a].scheme->moments >
                                   double(blocks[c].scheme->nodes) * blocks[c].scheme->moments;
                        });
                    }
                    omp_set_max_active_levels(2);
                    std::exception_ptr error;
#pragma omp parallel for num_threads(outer) schedule(dynamic, 1)
                    for (int m = 0; m < B; ++m) {
                        omp_set_num_threads(inner);
                        try {
                            work(blocks[by_cost[m]]);
                        } catch (...) {
#pragma omp critical(m1pn_blocks_gpu_error)
                            if (!error) error = std::current_exception();
                        }
                    }
                    omp_set_max_active_levels(1);
                    if (error)
                        std::rethrow_exception(error);
                };
                for (int s = 0; s < 4; ++s) {
                    each_block([&](Block &b) { clock(b.t_begin, [&] { b.gpu->lockstep_begin(s, t, dt); }); });
                    each_block([&](Block &b) { clock(b.t_end, [&] { b.gpu->lockstep_end(s, t, dt, b.stats); }); });
                }
                for (Block &b : blocks)
                    if (b.scheme->reacts_public())
                        clock(b.t_react, [&] { b.gpu->lockstep_react(.5 * dt, t + .75 * dt, b.stats); });
                t = (final_time - (t + dt) <= 8 * std::numeric_limits<double>::epsilon() * final_time) ? final_time : t + dt;
                ++steps;
            }
            for (Block &b : blocks)
                b.a = b.gpu->end(b.stats);
        }
#endif
        while (!use_gpu && t < final_time) {
            dt = std::min(selected, final_time - t);
            for_blocks([&](int k) {
                Block &b = blocks[k];
                base[k] = b.scheme->reacts_public() ? b.scheme->react(b.a, .5 * dt, &b.stats, t + .25 * dt) : b.a;
            });
            const double stage_t[4] = {t, t + .5 * dt, t + dt, t + .5 * dt};
            const double stage_old[4] = {0, 0, 2. / 3, 0}, stage_fresh[4] = {1, 1, 1. / 3, 1};
            const double stage_w[4] = {1. / 6, 1. / 6, 1. / 6, .5};
            for (int k = 0; k < B; ++k) x[k] = base[k];
            for (int s = 0; s < 4; ++s) {
                for (int k = 0; k < B; ++k) current[k] = &x[k];
                for_blocks([&](int k) {
                    y[k] = blocks[k].scheme->ssprk43_stage(stage_t[s], dt, x[k], base[k], stage_old[s], stage_fresh[s],
                                                           stage_w[s], &blocks[k].stats);
                });
                std::swap(x, y);
            }
            for_blocks([&](int k) {
                Block &b = blocks[k];
                b.a = b.scheme->reacts_public() ? b.scheme->react(x[k], .5 * dt, &b.stats, t + .75 * dt) : x[k];
            });
            t = (final_time - (t + dt) <= 8 * std::numeric_limits<double>::epsilon() * final_time) ? final_time : t + dt;
            ++steps;
        }
        const double solver_seconds = now() - start - setup_seconds;

        // QoIs (lattice_fcs totals as m1pn_fast) and the global balance (interface fluxes cancel up to the
        // non-conservative part of the independent interface limiting).
        double mass = 0, rate_c = 0, injected = 0, absorbed = 0, outflow = 0, mu = 0, ru = 0, au = 0;
        long cells = 0, coefficients = 0;
        for (Block &b : blocks) {
            const ElementWeno &s = *b.scheme;
            mass += mass_of(b, b.a);
            for (int i = 0; i < s.nodes; ++i)
                rate_c += s.lumped[i] * s.node_sigma_a[i] * project(b.a.row(i).transpose())[0];
            injected += b.stats.injected, absorbed += b.stats.absorbed, outflow += b.stats.boundary_outflow;
            cells += s.cells, coefficients += long(s.nodes) * s.moments;
            if (with_fcs) {
                const auto &xs = s.fcs_points();
                const auto &ws = s.fcs_weights();
                for (std::size_t k = 0; k < xs.size(); ++k) {
                    const double sa = material->sigma_a(xs[k]), r = b.table->value(k, t);
                    mu += ws[k] * r, ru += ws[k] * sa * r;
                    if (sa > 0) au += ws[k] * sa * b.table->time_integral(k, t);
                }
            }
        }
        const double balance = mass - initial_mass + outflow + absorbed - injected;
        {
            std::ofstream j(output + "/summary.json");
            j << std::setprecision(17) << "{\"driver\":\"m1pn_blocks\",\"blocks\":" << B << ",\"steps\":" << steps
              << ",\"dt\":" << dt << ",\"time\":" << t << ",\"cells\":" << cells << ",\"coefficients\":" << coefficients
              << ",\"final_density_mass\":" << mass << ",\"absorption_rate_final\":" << rate_c
              << ",\"ledger\":{\"injected\":" << injected << ",\"absorbed\":" << absorbed
              << ",\"boundary_outflow\":" << outflow << ",\"balance_residual\":" << balance << "}";
            if (with_fcs)
                j << ",\"lattice_fcs\":{\"uncollided_mass_final\":" << mu << ",\"uncollided_absorption_rate_final\":" << ru
                  << ",\"uncollided_absorbed\":" << au << ",\"total_mass_final\":" << mass + mu
                  << ",\"total_absorption_rate_final\":" << rate_c + ru << ",\"total_absorbed\":" << absorbed + au << "}";
            double work = 0, begin = 0, end = 0, react = 0, trace = 0;
            long long trace_calls = 0;
            for (const Block &b : blocks)
                work += b.t_work, begin += b.t_begin, end += b.t_end, react += b.t_react,
                    trace += 1e-9 * double(b.trace_ns->load()), trace_calls += b.traces->load();
            j << ",\"profile\":{\"block_work_sum\":" << work << ",\"gpu_begin_sum\":" << begin << ",\"gpu_end_sum\":" << end
              << ",\"gpu_react_sum\":" << react << ",\"interface_trace_sum\":" << trace
              << ",\"interface_trace_calls\":" << trace_calls << ",\"large_blocks\":" << large_count
              << ",\"small_blocks\":" << small_count << "}";
            j << ",\"setup_seconds\":" << setup_seconds << ",\"solver_seconds\":" << solver_seconds << "}\n";
        }
        {
            std::ofstream p(output + "/progress.json");
            p << std::setprecision(17) << "{\"steps\":" << steps << ",\"wall_seconds\":" << now() - start << "}\n";
        }
        {
            std::ofstream bl(output + "/blocks.csv");
            bl << "block,x0,x1,y0,y1,order,per_unit,cells,moments\n";
            for (int k = 0; k < B; ++k)
                bl << k << "," << blocks[k].lo[0] << "," << blocks[k].hi[0] << "," << blocks[k].lo[1] << ","
                   << blocks[k].hi[1] << "," << blocks[k].rect.order << "," << blocks[k].rect.per_unit << ","
                   << blocks[k].scheme->cells << ","
                   << blocks[k].scheme->moments << "\n";
        }
        // Samples on a uniform grid (u0 = total density / sqrt(4 pi) with FCS, as m1pn_fast final.csv).
        {
            const int n = 7 * sample_per_unit + 1;
            std::vector<std::array<double, 5>> rows(std::size_t(n) * n);
            parallel_for(n * n, [&](int s) {
                const V3 xs(7. * (s % n) / (n - 1), 7. * (s / n) / (n - 1), 0);
                const int k = block_at(xs);
                const Block &b = blocks[k];
                const Vec v = b.scheme->evaluate(b.a, xs);
                double u0 = v[0];
                double unc = with_fcs ? b.table->lattice.density(xs[0], xs[1], t) : 0.;
                u0 += unc / std::sqrt(4 * pi);
                rows[s] = {xs[0], xs[1], u0, unc, double(k)};
            });
            std::ofstream f(output + "/final.csv");
            f << std::setprecision(17) << "node,x,y,z,u0,rho_uncollided,block\n";
            for (std::size_t s = 0; s < rows.size(); ++s)
                f << s << "," << rows[s][0] << "," << rows[s][1] << ",0," << rows[s][2] << "," << rows[s][3] << ","
                  << int(rows[s][4]) << "\n";
        }
        std::cout << "done: " << B << " blocks, " << steps << " steps, mass " << mass << ", balance " << balance
                  << ", " << now() - start << " s\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
