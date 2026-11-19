#pragma once
// Experimental GPU time loop of the element WENO scheme (docs/ELEMENT_WENO_GPU.md). The state stays on the device; the
// volume work of rate() runs there, the boundary (C6 ghost / positivity projection, face loads, boundary corrections)
// runs on the host on the boundary rows only, overlapped with the device work. Same scheme as ElementWeno::advance up
// to rounding (summation order, fused multiply-adds, natural-order line factors).
#include "element_weno.hpp"
#include "element_weno_gpu_device.hpp"
namespace m1pn {
namespace gpu {
struct Context;
}
namespace uncollided {
struct GaussianPulse;
struct LatticeUncollidedTable;
struct Lattice3dUncollidedTable;
}
class ElementWenoGpu {
public:
    explicit ElementWenoGpu(const ElementWeno &);
    ~ElementWenoGpu();
    ElementWenoGpu(const ElementWenoGpu &) = delete;
    ElementWenoGpu &operator=(const ElementWenoGpu &) = delete;
    // `steps` steps of size dt from `time` (options.ssprk43 selects the integrator); returns the final state.
    Field run(double time, double dt, int steps, const Field &a, ElementWenoStats *stats);
    // Stepwise use (drivers with their own time loop): begin uploads the state, step advances it on the device
    // (Strang reaction half steps when the scheme reacts; `next` = start time of the next step, NaN if none),
    // state() downloads it, end() merges the device counters into stats and returns the state.
    void begin(const Field &a);
    void step(double time, double dt, double next, ElementWenoStats &stats);
    Field state();
    Field end(ElementWenoStats &stats);
    // Multi-block lockstep (src/blocks_main.cpp, SSPRK(4,3) only): stage s = 0..3 of the step from `time` in two
    // halves; the driver runs lockstep_begin on all blocks before any lockstep_end (interface traces of all blocks are
    // then on the host: host_state() rows of the boundary nodes, zero elsewhere). lockstep_react = the Strang half
    // step of step(). Same arithmetic as step().
    void lockstep_react(double tau, double source_time, ElementWenoStats &stats) { react(tau, source_time, stats); }
    void lockstep_begin(int s, double time, double dt);
    void lockstep_end(int s, double time, double dt, ElementWenoStats &stats);
    // Split stage at block interfaces: lockstep_end_a stops after the step-1 states, lockstep_end_b completes it
    // (lockstep_end = a then b). Without a pending second step, b does nothing.
    void lockstep_end_a(int s, double time, double dt, ElementWenoStats &stats);
    void lockstep_end_b(ElementWenoStats &stats);
    const Field &host_state() const { return a_h_; }
    // Spectral radius of high_order_step() (same power iteration, on the device).
    double spectral_radius();
    double spectral_radius(int iterations, const std::array<Field, 2> *start, std::array<Field, 2> *end);
    // Lattice FCS: (re)build the device time table for the window of the shared table object (dynamic AMR).
    void build_lattice_table();
    const AdaptivePnPlan *adaptive_plan() const { return adaptive_enabled_ ? &adaptive_plan_ : nullptr; }
    // Host-side profile.
    double t_boundary = 0, t_boundary_high = 0, t_face_data = 0, t_correction = 0, t_fallback = 0, t_wait = 0, t_sync = 0;
    long closure_fallbacks = 0, limiter_fallbacks = 0;
    int fallback_capacity_ = 1 << 16;
    long ghost_flagged = 0; // ghost pairs solved by the host (device projection flagged them)
    double t_wait_ghost = 0;
    double t_ghost = 0; // host phase B: ghost moments (part of t_boundary)
    double t_diag_device = 0; // M1PN_GPU_DIAG_SERIAL=1: device work overlapped by phase B

private:
    const ElementWeno &s_;
    gpu::Context *ctx_ = nullptr;
    int nb_ = 0;                       // boundary nodes (boundary_nodes_ order = boundary slot)
    int *d_bnodes_ = nullptr;          // device copy of boundary_nodes_
    // Boundary nodes whose full PN row the host reads (ghost pairs, block interfaces, host phase A); the nodes only on
    // broken lines need the low moments alone. nf_ == nb_ and d_fnodes_ == d_bnodes_ when every row is needed.
    int nf_ = 0;
    int *d_fnodes_ = nullptr;
    std::vector<int> full_nodes_;
    // Host buffers: node-indexed fields valid on the boundary rows only, compact boundary arrays.
    Field a_h_, low_h_, low_flux_h_, closure_h_;
    Field boundary_low_, flux_area_, boundary_bar_;
    Vec boundary_area_, boundary_inflow_;
    Field boundary_high_; // nb x q
    Field load_b_;        // nb x 4 (slot order)
    // Pinned staging: boundary rows of (x, low, low_flux, closure), H upload, load download/upload.
    double *pin_fetch_ = nullptr, *pin_high_ = nullptr, *pin_load_ = nullptr, *pin_load_up_ = nullptr;
    // Split boundary (docs/ELEMENT_WENO_GPU.md): phase A = PN face load H (needs the trace and the inflow datum only),
    // phase B = ghost moments and the low-order face terms (needed only by the boundary corrections).
    // Inflow data per (face, point) for a stage time (unsteady inflow), double-buffered: the data of the next stage are
    // computed while the host waits for the device (they depend on the time only).
    std::array<std::vector<Incoming>, 2> face_data_;
    std::array<double, 2> face_time_{{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()}};
    int face_points_ = 0, last_face_ = -1;
    bool closure_ready_ = false;
    bool device_ghost_ = false; // positivity projection of the ghost on the device
    // Ghost traces on the device, projections on the host (docs/STATIC_HP_BLOCKS.md; experiment, M1PN_GPU_GHOST_TRACES=1).
    bool trace_ghost_ = false;
    double *pin_traces_ = nullptr;
    std::array<std::size_t, 6> trace_offset_{};
    // Phase A (PN face load) on the device (gpu::boundary_high); M1PN_GPU_BOUNDARY_HIGH=0 keeps it on the host.
    bool device_high_ = false;
    double *pin_db_ = nullptr, *pin_low4_ = nullptr; // datum rows (row-major) up, l <= 1 node sums down
    // split_breaks (experiment, docs/BREAK_DT.md): l <= 1 sums of the broken-line rows (pinned, per boundary slot) and
    // the low-order broken-line sums of phase B (node-indexed), the unsplit term T and area A per node.
    bool split_ = false;
    double *pin_low4b_ = nullptr;
    Field brk_low_, brk_bar_, brk_fa_, brk_high4_, split_term_h_;
    Vec brk_area_;
    double *pin_ghost_m_ = nullptr;
    int *pin_ghost_flags_ = nullptr;
    long ghost_flagged_merged = 0;
    PositivityProjectionStats ghost_merged_; // the closure of the next stage input was computed by the last finish_stage
    // Per buffer also the datum part of the PN face load, DB = D llf_datum^T (face rows x q), and whether a side has a
    // nonzero datum (time-dependent only, so prepared ahead as well).
    std::array<Field, 2> DB_;
    std::array<std::array<bool, 6>, 2> datum_zero_{};
    bool steady_ready_ = false;
    int select_face(double time);          // buffer of `time` (prepared if missing); becomes the current buffer
    void prepare_face_data(double time);   // fill the other buffer for `time` (inflow data and DB)
    void fill_datum(int buffer);           // DB_ / datum_zero_ of a buffer from its inflow data
    Field G_, D_, H_, L_, BR_, FA_;
    Vec AR_, IN_;
    std::vector<char> face_datum_;
    std::vector<Four> outgoing_;
    void boundary_high(int buffer);
    void boundary_low(int buffer, PositivityProjectionStats *projection);
    double *u_ = nullptr, *b1_ = nullptr, *b2_ = nullptr; // device state and stage buffers
    const uncollided::GaussianPulse *pulse_ = nullptr;    // FCS: analytic uncollided pulse
    std::vector<double> table_;
    uncollided::LatticeUncollidedTable *lattice_ = nullptr; // FCS: lattice time table (device copy in the context)
    uncollided::Lattice3dUncollidedTable *lattice3_ = nullptr; // FCS: lattice-3d time table (device copy)
    gpu::Counters merged_;
    double ledger_absorbed_ = 0, ledger_injected_ = 0;
    void react(double tau, double source_time, ElementWenoStats &stats);
    void stage(double time, const double *x, double *out, double *u, double old, double fresh, double tau,
               double weight, double dt, ElementWenoStats &stats, double next_time);
    int stage_begin(double time, const double *x, double *out, double next_time);
    // defer (multi-block split stage): stop after the step-1 states (ElementWeno::split_compute_y); stage_end_finish
    // completes the stage once all blocks published theirs.
    void stage_end(int buffer, double time, const double *x, double *out, double *u, double old, double fresh,
                   double tau, double weight, double dt, ElementWenoStats &stats, bool defer = false);
    void stage_end_finish(const double *x, double *out, double *u, double old, double fresh, double tau, double weight,
                          double dt, ElementWenoStats &stats);
    std::vector<Four> split_u_h_, split_rate_h_;
    bool finish_pending_ = false;
    int lockstep_stage_ = 0;
    double lockstep_time_ = 0, lockstep_dt_ = 0;
    bool interfaces_ = false; // multi-block interface sides: host face work by ElementWeno::add_boundary
    bool nodal_interfaces_ = false; // conforming interface sides: partner term as face datum, regular pipeline
    bool nodal_db_zeroed_ = false;
    int lockstep_buffer_ = 0;
    // Dynamic tile-based adaptive PN (experiment/dynamic-adaptive-pn).
    bool adaptive_enabled_ = false;
    AdaptivePnPlan adaptive_plan_;
    std::vector<double> adaptive_rho_, adaptive_E_, adaptive_face_E_;
    std::vector<int> adaptive_shrunken_;
    long adaptive_step_ = 0;
    int adaptive_fetch_q_ = 0; // max tile_q across full_nodes_ (for truncated D2H fetch)
    void adapt_gpu(bool initial, ElementWenoStats *stats);
    void closure_fallback(const double *x);
    void limiter_fallback(ElementWenoStats &stats);
    void boundary_correction(ElementWenoStats &stats);
    void check_error();
};
} // namespace m1pn
