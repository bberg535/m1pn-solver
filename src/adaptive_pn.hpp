#pragma once
// Dynamic tile-based adaptive PN order controller (experiment/dynamic-adaptive-pn).
//
// Partitions the single tensor-product node grid into rectangular tiles of size tb[0] x tb[1] x tb[2].
// Each tile T carries an odd active PN order N_T in [min_order, max_order] (active moments [0, tile_q[T])).
// All inactive moments c >= tile_q[T] on nodes in tile T are kept identically zero (exact Galerkin subspace
// V_{h,0}(A_l) with nested active sets A_{l+1} subseteq A_l and l <= 1 always active everywhere, so M1 cone
// realizability, conservation, WENO and broken-line split steps are untouched).
//
// Shared by CPU (ElementWeno) and GPU (ElementWenoGpu) so both paths execute the exact same sensor and
// adaptation decisions and build compact coalesced worklists for line solves and tile kernels.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace m1pn {

struct AdaptiveLineSeg {
    int line;    // 0 .. lines - 1 along axis d
    int c0;      // start component (multiple of chunk width)
    int k_start; // inclusive physical node index along axis d [0, n)
    int k_end;   // exclusive physical node index along axis d (0, n]
};

struct AdaptiveTileChunk {
    int tx, ty, tz; // tile coordinates
    int c0;         // start component (multiple of chunk width)
    int q_lim;      // active moment count tile_q[T]; active lanes satisfy c0 + lane < q_lim
};

struct AdaptivePnPlan {
    static constexpr int kChunk = 32; // warp-aligned moment chunk for GPU coalescing (and CPU SIMD)
    static constexpr int kBorderMargin = 3; // border thickness (in nodes) for wave-front dilation

    int dim = 2;
    std::array<int, 3> npa{1, 1, 1};
    std::array<int, 3> stride{1, 1, 1};
    int nodes = 0;
    int q_max = 0;
    int max_order = 1;
    int min_order = 1;
    int num_chunks = 0;

    std::array<int, 3> tb{1, 1, 1}; // tile width per axis in nodes
    std::array<int, 3> nt{1, 1, 1}; // number of tiles per axis
    int num_tiles = 0;

    double tol = 3e-4;
    double hysteresis = 0.25;
    int halo = 1; // 0: off, 1: border-triggered neighbor dilation, 2: full 3x3 dilation
    // Dynamic AMR restart (docs/DYNAMIC_AMR.md): the initial adaptation of a transferred state dilates only across
    // borders with significant energy (as later updates), not around every tile above the minimum order.
    bool restart = false;
    // Local relative tolerance (docs/DYNAMIC_AMR.md, coupling B): bounds per tile against max(rho_T, floor rho_max)
    // instead of rho_max. 0: global (the original rule, bitwise).
    double floor = 0;
    // Carried orders (coupling A): per tile the order mapped from the previous mesh. The initial adaptation then starts
    // from these orders like a regular update (+-2 with hysteresis, border-triggered dilation). Empty: none.
    std::vector<int> preset;
    // preset[T] = max over the nodes of tile T of node_order (odd, clamped).
    void set_preset_from_nodes(const std::vector<int> &node_order) {
        preset.assign(num_tiles, min_order);
        for (int i = 0; i < nodes && i < int(node_order.size()); ++i) {
            int &p = preset[node_tile(i)];
            p = std::max(p, node_order[i]);
        }
        for (int &p : preset)
            p = clamp_odd_order(p);
    }

    std::vector<int> order_q;     // [0..max_order]: number of active moments with degree <= n
    std::vector<int> deg_start;   // [0..max_order+1]: start index of degree l in [0, q_max)
    std::vector<int> comp_degree; // [0..q_max-1]: degree l of active moment c

    std::vector<int> tile_order;  // [num_tiles]: current odd active PN order
    std::vector<int> tile_q;      // [num_tiles]: current active moments = order_q[tile_order[T]]
    std::vector<int> tile_q_prev; // [num_tiles]: previous active moments
    std::vector<int> tile_nodes;  // [num_tiles]: number of nodes in tile T
    std::vector<int> tile_floor;  // [num_tiles]: per-tile min order (e.g. raised on non-vacuum inflow walls)

    int global_max_order = 1;
    int global_max_q = 4;
    std::array<int, 6> side_q_dom{}; // max tile_q over domain boundary rows per side
    std::array<int, 6> side_q_brk{}; // max tile_q over broken-line rows per side
    int bnd_max_q = 4;               // max over all boundary rows
    double active_coeff_fraction = 1.0;
    double mean_order = 1.0;
    int min_active_order = 1;

    // Compact worklists rebuilt only when tile_order changes.
    // line_segs_grad[d]: active 1D segments padded by 1 node on each side (for C_d a -> g_d).
    // line_segs_mass[d]: active 1D segments without padding (for M_d^-1 residual).
    std::array<std::vector<AdaptiveLineSeg>, 3> line_segs_grad;
    std::array<std::vector<AdaptiveLineSeg>, 3> line_segs_mass;
    std::vector<AdaptiveTileChunk> tile_chunks;

    int tile_index(int tx, int ty, int tz) const {
        return (tz * nt[1] + ty) * nt[0] + tx;
    }

    int node_tile(int i) const {
        const int ix = i % npa[0];
        const int iy = dim > 1 ? (i / npa[0]) % npa[1] : 0;
        const int iz = dim > 2 ? i / (npa[0] * npa[1]) : 0;
        return tile_index(ix / tb[0], iy / tb[1], iz / tb[2]);
    }

    int clamp_odd_order(int n) const {
        if (n >= max_order)
            return max_order;
        n = std::max(min_order, n);
        if ((n & 1) == 0)
            ++n;
        return std::min(max_order, n);
    }

    void init(int dimension, const std::array<int, 3> &nodes_per_axis, int pn_order,
              const std::vector<int> &active_indices, int tile_size, double tolerance, double hyst,
              int min_ord, int halo_mode) {
        dim = dimension;
        npa = nodes_per_axis;
        nodes = npa[0] * npa[1] * npa[2];
        stride = {1, npa[0], npa[0] * npa[1]};
        max_order = pn_order;
        q_max = int(active_indices.size());
        min_order = std::max(1, std::min(pn_order, min_ord));
        if ((min_order & 1) == 0 && min_order < max_order)
            ++min_order;
        tol = tolerance;
        hysteresis = hyst;
        halo = halo_mode;
        num_chunks = (q_max + kChunk - 1) / kChunk;

        comp_degree.resize(q_max);
        deg_start.assign(max_order + 2, q_max);
        order_q.assign(max_order + 1, 0);
        for (int c = 0; c < q_max; ++c) {
            const int l = int(std::sqrt(double(active_indices[c]) + 0.5));
            comp_degree[c] = l;
            if (deg_start[l] == q_max)
                deg_start[l] = c;
            order_q[l] = c + 1;
        }
        for (int l = max_order; l >= 0; --l)
            if (deg_start[l] == q_max)
                deg_start[l] = deg_start[l + 1];
        for (int l = 1; l <= max_order; ++l)
            if (order_q[l] < order_q[l - 1])
                order_q[l] = order_q[l - 1];
        order_q[1] = std::max(order_q[1], std::min(4, q_max));

        const int ts = std::max(4, tile_size);
        for (int d = 0; d < 3; ++d) {
            tb[d] = (d < dim) ? std::min(ts, npa[d]) : 1;
            nt[d] = (npa[d] + tb[d] - 1) / tb[d];
        }
        num_tiles = nt[0] * nt[1] * nt[2];
        tile_nodes.resize(num_tiles);
        for (int tz = 0; tz < nt[2]; ++tz) {
            const int nz = std::min(npa[2], (tz + 1) * tb[2]) - tz * tb[2];
            for (int ty = 0; ty < nt[1]; ++ty) {
                const int ny = std::min(npa[1], (ty + 1) * tb[1]) - ty * tb[1];
                for (int tx = 0; tx < nt[0]; ++tx) {
                    const int nx = std::min(npa[0], (tx + 1) * tb[0]) - tx * tb[0];
                    tile_nodes[tile_index(tx, ty, tz)] = nx * ny * nz;
                }
            }
        }
        // Start with full order before initial adapt (so initial state is untouched until adapt runs).
        tile_order.assign(num_tiles, max_order);
        tile_q.assign(num_tiles, q_max);
        tile_q_prev.assign(num_tiles, q_max);
        tile_floor.assign(num_tiles, min_order);
        side_q_dom.fill(q_max);
        side_q_brk.fill(q_max);
        bnd_max_q = q_max;
        rebuild_worklists();
    }

    // Update tile_order from per-tile max density `tile_rho` [num_tiles], per-tile per-degree max
    // L2-over-m amplitude `tile_E` [num_tiles * (max_order + 1)], and per-tile border top-shell
    // amplitude `tile_face_E` [num_tiles * 6] (face 2*d + side).
    // Returns true if any tile changed order; populates `shrunken_tiles` with tiles where tile_q decreased.
    bool update_from_energies(const std::vector<double> &tile_rho, const std::vector<double> &tile_E,
                              const std::vector<double> &tile_face_E, bool initial,
                              std::vector<int> &shrunken_tiles) {
        const int L1 = max_order + 1;
        double rho_max = 0.0;
        for (int t = 0; t < num_tiles; ++t)
            rho_max = std::max(rho_max, tile_rho[t]);
        const double scale = std::max(rho_max, 1e-30);
        std::vector<double> up(num_tiles, tol * scale);
        if (floor > 0)
            for (int t = 0; t < num_tiles; ++t)
                up[t] = tol * std::max(tile_rho[t], floor * scale);
        const bool carried = initial && int(preset.size()) == num_tiles;

        std::vector<int> next_ord(num_tiles, min_order);
        for (int t = 0; t < num_tiles; ++t) {
            const double *E = &tile_E[std::size_t(t) * L1];
            const double bound_up = up[t], bound_down = hysteresis * bound_up;
            const int cur = carried ? preset[t] : (initial ? max_order : tile_order[t]);
            if (initial && !carried) {
                double tail_sq = 0.0;
                int need = min_order;
                for (int l = max_order; l >= min_order; --l) {
                    if (std::sqrt(tail_sq) > bound_up) {
                        need = clamp_odd_order(l + 1);
                        break;
                    }
                    tail_sq += E[l] * E[l];
                }
                // Check l = 1 -> wake up to P3 if flux already exceeds 0.25 * bound_up
                if (need == 1 && max_order >= 3 && E[1] > 0.25 * bound_up)
                    need = clamp_odd_order(3);
                next_ord[t] = need;
            } else {
                // Top active shell energy for refinement check.
                double top = 0.0;
                bool refine = false;
                if (cur == 1) {
                    top = E[1];
                    refine = (max_order >= 3) && (top > 0.25 * bound_up);
                } else {
                    const double e_hi = E[cur];
                    const double e_prev = E[cur - 1];
                    top = std::hypot(e_prev, e_hi);
                    refine = (cur < max_order) && (top > bound_up);
                }
                if (refine) {
                    int bump = 2;
                    if (top > 64.0 * bound_up)
                        bump = 6;
                    else if (top > 8.0 * bound_up)
                        bump = 4;
                    next_ord[t] = clamp_odd_order(cur + bump);
                } else {
                    // Check coarsening from top down against bound_down.
                    double tail_sq = 0.0;
                    int need = min_order;
                    for (int l = cur; l >= min_order; --l) {
                        if (std::sqrt(tail_sq) > bound_down) {
                            need = clamp_odd_order(l + 1);
                            break;
                        }
                        tail_sq += E[l] * E[l];
                    }
                    if (need == 1 && cur >= 3 && E[1] > 0.25 * bound_down)
                        need = clamp_odd_order(3);
                    // Smooth coarsening: drop at most 2 orders per adapt interval.
                    next_ord[t] = clamp_odd_order(std::max(need, cur - 2));
                }
            }
            if (t < int(tile_floor.size()) && next_ord[t] < tile_floor[t])
                next_ord[t] = clamp_odd_order(tile_floor[t]);
        }

        // Spatial wave-front dilation: prevent a sharp front from entering a low-order tile before next adapt.
        if (halo > 0 && num_tiles > 1) {
            std::vector<int> dilated = next_ord;
            for (int tz = 0; tz < nt[2]; ++tz) {
                for (int ty = 0; ty < nt[1]; ++ty) {
                    for (int tx = 0; tx < nt[0]; ++tx) {
                        const int t = tile_index(tx, ty, tz);
                        const int ord = next_ord[t];
                        if (ord <= min_order)
                            continue;
                        if (halo >= 2) {
                            for (int dz = (dim > 2 ? -1 : 0); dz <= (dim > 2 ? 1 : 0); ++dz)
                                for (int dy = (dim > 1 ? -1 : 0); dy <= (dim > 1 ? 1 : 0); ++dy)
                                    for (int dx = -1; dx <= 1; ++dx) {
                                        const int nx = tx + dx, ny = ty + dy, nz = tz + dz;
                                        if (nx >= 0 && nx < nt[0] && ny >= 0 && ny < nt[1] && nz >= 0 && nz < nt[2]) {
                                            int &dst = dilated[tile_index(nx, ny, nz)];
                                            dst = std::max(dst, ord);
                                        }
                                    }
                        } else {
                            // Border-triggered dilation: propagate to face neighbor only when this tile's
                            // border strip facing that neighbor has significant angular energy.
                            const double *fe = &tile_face_E[std::size_t(t) * 6];
                            const double face_thresh = (ord == 1) ? 0.25 * up[t] : up[t];
                            bool active_face[6] = {};
                            for (int f = 0; f < 2 * dim; ++f)
                                active_face[f] = (initial && !restart && !carried) ? (ord > min_order) : (fe[f] > face_thresh);
                            const int step_d[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
                            for (int d = 0; d < dim; ++d) {
                                for (int s = 0; s < 2; ++s) {
                                    if (!active_face[2 * d + s])
                                        continue;
                                    const int dir = s ? 1 : -1;
                                    const int nx = tx + dir * step_d[d][0];
                                    const int ny = ty + dir * step_d[d][1];
                                    const int nz = tz + dir * step_d[d][2];
                                    if (nx >= 0 && nx < nt[0] && ny >= 0 && ny < nt[1] && nz >= 0 && nz < nt[2]) {
                                        int &dst = dilated[tile_index(nx, ny, nz)];
                                        dst = std::max(dst, ord);
                                    }
                                }
                            }
                            // Diagonal corner propagation in 2D if both adjacent faces are active.
                            if (dim >= 2) {
                                for (int sx = 0; sx < 2; ++sx)
                                    for (int sy = 0; sy < 2; ++sy)
                                        if (active_face[sx] && active_face[2 + sy]) {
                                            const int nx = tx + (sx ? 1 : -1), ny = ty + (sy ? 1 : -1);
                                            if (nx >= 0 && nx < nt[0] && ny >= 0 && ny < nt[1]) {
                                                int &dst = dilated[tile_index(nx, ny, tz)];
                                                dst = std::max(dst, ord);
                                            }
                                        }
                            }
                        }
                    }
                }
            }
            next_ord.swap(dilated);
        }

        shrunken_tiles.clear();
        bool changed = false;
        tile_q_prev = tile_q;
        for (int t = 0; t < num_tiles; ++t) {
            const int nq = order_q[next_ord[t]];
            if (next_ord[t] != tile_order[t])
                changed = true;
            if (nq < tile_q[t])
                shrunken_tiles.push_back(t);
            tile_order[t] = next_ord[t];
            tile_q[t] = nq;
        }
        if (carried)
            preset.clear(); // used once (first adaptation on the transferred state)
        if (changed || initial)
            rebuild_worklists();
        return changed || initial;
    }

    void rebuild_worklists() {
        long long active_coeffs = 0;
        double ord_sum = 0.0;
        global_max_order = min_order;
        min_active_order = max_order;
        global_max_q = 4;
        for (int t = 0; t < num_tiles; ++t) {
            active_coeffs += (long long)tile_nodes[t] * tile_q[t];
            ord_sum += double(tile_nodes[t]) * tile_order[t];
            global_max_order = std::max(global_max_order, tile_order[t]);
            min_active_order = std::min(min_active_order, tile_order[t]);
            global_max_q = std::max(global_max_q, tile_q[t]);
        }
        active_coeff_fraction = double(active_coeffs) / (double(nodes) * double(q_max));
        mean_order = ord_sum / double(nodes);

        // 1. Tile-chunk worklist for high moments c >= 4 (c0 = 4, 36, 68, ...): exact 4 + 32 = 36 = P7 alignment!
        // Low moments c = 0..3 run in dedicated full-mesh low4 passes; P1 tiles (tile_q == 4) emit ZERO chunks!
        tile_chunks.clear();
        for (int tz = 0; tz < nt[2]; ++tz)
            for (int ty = 0; ty < nt[1]; ++ty)
                for (int tx = 0; tx < nt[0]; ++tx) {
                    const int t = tile_index(tx, ty, tz);
                    const int q_lim = tile_q[t];
                    for (int c0 = 4; c0 < q_lim; c0 += kChunk)
                        tile_chunks.push_back({tx, ty, tz, c0, q_lim});
                }

        // 2. Per-axis active 1D line segments for high chunks c0 = 4, 36, 68, ... (sorted longest-first so warps
        // inside a GPU block have matched lengths and zero wave-tail stragglers).
        for (int d = 0; d < 3; ++d) {
            line_segs_grad[d].clear();
            line_segs_mass[d].clear();
        }
        const int hi_chunks = global_max_q > 4 ? (global_max_q - 4 + kChunk - 1) / kChunk : 0;
        for (int d = 0; d < dim; ++d) {
            const int n = npa[d], s = stride[d];
            const int lines = nodes / n;
            std::vector<std::pair<int, int>> segs_m, segs_g;
            for (int cb = 0; cb < hi_chunks; ++cb) {
                const int c0 = 4 + cb * kChunk;
                for (int line = 0; line < lines; ++line) {
                    const int start = (line / s) * (s * n) + line % s;
                    const int ix0 = start % npa[0];
                    const int iy0 = dim > 1 ? (start / npa[0]) % npa[1] : 0;
                    const int iz0 = dim > 2 ? start / (npa[0] * npa[1]) : 0;
                    const int t_fixed[3] = {ix0 / tb[0], iy0 / tb[1], iz0 / tb[2]};
                    segs_m.clear();
                    int k = 0;
                    while (k < nt[d]) {
                        int tc[3] = {t_fixed[0], t_fixed[1], t_fixed[2]};
                        tc[d] = k;
                        if (tile_q[tile_index(tc[0], tc[1], tc[2])] <= c0) {
                            ++k;
                            continue;
                        }
                        const int t_start = k;
                        while (k < nt[d]) {
                            tc[d] = k;
                            if (tile_q[tile_index(tc[0], tc[1], tc[2])] <= c0)
                                break;
                            ++k;
                        }
                        const int k_start = t_start * tb[d];
                        const int k_end = std::min(n, k * tb[d]);
                        segs_m.push_back({k_start, k_end});
                    }
                    if (segs_m.empty())
                        continue;
                    for (const auto &sg : segs_m)
                        line_segs_mass[d].push_back({line, c0, sg.first, sg.second});

                    // Gradient segments: pad by 1 node on each side to capture C_d a across tile borders.
                    segs_g.clear();
                    for (const auto &sg : segs_m) {
                        const int ks = std::max(0, sg.first - 1);
                        const int ke = std::min(n, sg.second + 1);
                        if (!segs_g.empty() && ks <= segs_g.back().second)
                            segs_g.back().second = std::max(segs_g.back().second, ke);
                        else
                            segs_g.push_back({ks, ke});
                    }
                    for (const auto &sg : segs_g)
                        line_segs_grad[d].push_back({line, c0, sg.first, sg.second});
                }
            }
            auto by_len_desc = [](const AdaptiveLineSeg &a, const AdaptiveLineSeg &b) {
                const int la = a.k_end - a.k_start, lb = b.k_end - b.k_start;
                if (la != lb) return la > lb;
                if (a.c0 != b.c0) return a.c0 < b.c0;
                return a.line < b.line;
            };
            std::sort(line_segs_mass[d].begin(), line_segs_mass[d].end(), by_len_desc);
            std::sort(line_segs_grad[d].begin(), line_segs_grad[d].end(), by_len_desc);
        }
    }
};

} // namespace m1pn
