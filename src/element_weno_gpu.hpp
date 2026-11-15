#pragma once
// Experimental GPU time loop of the element WENO scheme (docs/ELEMENT_WENO_GPU.md). The state stays on the device; the
// volume work of rate() runs there, the boundary (C6 ghost / positivity projection, face loads, boundary corrections)
// runs on the host on the boundary rows only, overlapped with the device work. Same scheme as ElementWeno::advance up
// to rounding (summation order, fused multiply-adds, natural-order line factors).
#include "element_weno.hpp"
namespace m1pn {
namespace gpu {
struct Context;
}
class ElementWenoGpu {
public:
    explicit ElementWenoGpu(const ElementWeno &);
    ~ElementWenoGpu();
    ElementWenoGpu(const ElementWenoGpu &) = delete;
    ElementWenoGpu &operator=(const ElementWenoGpu &) = delete;
    // `steps` steps of size dt from `time` (options.ssprk43 selects the integrator); returns the final state.
    Field run(double time, double dt, int steps, const Field &a, ElementWenoStats *stats);
    // Spectral radius of high_order_step() (same power iteration, on the device).
    double spectral_radius();
    // Host-side profile.
    double t_boundary = 0, t_boundary_high = 0, t_face_data = 0, t_correction = 0, t_fallback = 0, t_wait = 0, t_sync = 0;
    long closure_fallbacks = 0, limiter_fallbacks = 0;
    long ghost_flagged = 0; // ghost pairs solved by the host (device projection flagged them)
    double t_wait_ghost = 0;

private:
    const ElementWeno &s_;
    gpu::Context *ctx_ = nullptr;
    int nb_ = 0;                       // boundary nodes (boundary_nodes_ order = boundary slot)
    int *d_bnodes_ = nullptr;          // device copy of boundary_nodes_
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
    void stage(double time, const double *x, double *out, double *u, double old, double fresh, double tau,
               double weight, double dt, ElementWenoStats &stats, double next_time);
    void closure_fallback(const double *x);
    void limiter_fallback(ElementWenoStats &stats);
    void boundary_correction(ElementWenoStats &stats);
    void check_error();
};
} // namespace m1pn
