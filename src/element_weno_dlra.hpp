#pragma once
// Dynamical low-rank approximation (DLRA) of the element-WENO M1PN scheme (docs/DLRA.md).
//
// State: a = [a_L | X S W^T]. The four M1 moments a_L (l <= 1, nodes x 4) stay full rank and go through the
// unchanged limited low-order update (cone, mass ledger, boundary corrections). The PN moments with l >= 2 are a
// rank-r factorization with X^T M X = I (M = consistent Bernstein mass), W^T W = I (PN coefficients are an
// orthonormal angular basis) and S r x r. Keeping the conserved/limited moments outside the low-rank factor is the
// conservative truncation of Einkemmer, Kusch, Schotthoefer (arXiv:2311.06399).
//
// Time step (rank-adaptive augmented BUG integrator of Ceruti, Kusch, Lubich, BIT 62 (2022), as used by Kusch and
// Stammer, ESAIM M2AN 57 (2023)):
//   1. K and L step (forward Euler, a_L frozen), sharing one operator evaluation at a^n:
//        K1 = X S + dt F(a^n) W,   L1 = W S^T + dt (X^T M F(a^n))^T,   F = PN rate of the l >= 2 moments.
//   2. Augmented bases X^ = orth_M[X, K1], W^ = orth[W, L1] (rank <= 2r), S^ = X^T M X S W^T W^.
//   3. Galerkin S step coupled to the limited M1 update with the scheme's SSPRK3 / SSPRK(4,3) stages:
//        dS^/dt = X^T R(a) W^  (R = M F, no mass solve),  a_L by the limited stage of rate().
//   4. Truncation: SVD of S^, smallest rank with sqrt(sum_{j>r1} sigma_j^2) <= theta,
//      theta = tolerance * ||a||_M (full state; or ||S^|| with relative_to_high) + absolute_tolerance.
// Reaction (filter, scattering, absorption, sources, FCS) stays the exact Strang half step: diagonal in angle and in
// space, applied to W and X followed by re-orthonormalization.
#include "element_weno.hpp"
#include <climits>
namespace m1pn {
struct DlraOptions {
    int initial_rank = 8;  // rank after compression of the initial data (padded with orthonormal completions)
    int min_rank = 4;      // lower bound after truncation (keeps the K/L steps able to pick up new directions)
    int max_rank = 0;      // upper bound after truncation (0: none)
    double tolerance = 1e-3;         // relative truncation tolerance theta-bar
    double absolute_tolerance = 0;   // added to theta
    bool relative_to_high = false;   // theta relative to ||S^|| (Kusch-Stammer on the low-rank block) instead of ||a||_M
    bool fixed_rank = false;         // truncate to exactly initial_rank (no adaptivity)
    // S step with operators on the fixed bases precomputed once per step (O(nodes r^2) per step, O(nodes r) per stage).
    // The WENO factor gamma of s_h in the Galerkin rate of the l >= 2 moments is then the one of a^n (the M1 moments
    // always use the current gamma). false: every stage re-evaluates all operators on [a_L | X^ S^] (exact gamma).
    bool frozen_gamma = true;
    // Truncation tolerance per unit time (Ceruti, Kusch, Lubich: theta = c h): theta = tolerance * norm * dt, so the
    // accumulated truncation error over [0, T] is about tolerance * T * norm. false: theta per step (Kusch-Stammer).
    bool per_unit_time = false;
};
struct DlraState {
    Field low; // nodes x 4: coefficients of the active moments 0..3 (l <= 1)
    Field X;   // nodes x r, X^T M X = I
    Mat S;     // r x r (rectangular only inside a step)
    Mat W;     // q_H x r, W^T W = I; row k is active moment 4 + k
    int rank() const { return int(S.rows()); }
    // Angular projections of W carried from the previous step (W_{n+1} = W^ Q_1): per boundary side W^T B_H (r x q,
    // B = (A_n - llf_homogeneous)^T) and H_H W (outgoing points x r). Valid only while `cached` (W unchanged since).
    std::array<Mat, 6> wtb, hw;
    bool cached = false;
};
struct DlraStats {
    long steps = 0;
    int min_rank = INT_MAX, max_rank = 0, last_rank = 0;
    double rank_sum = 0;          // sum over steps of the rank after truncation
    double discarded = 0;         // sum over steps of the discarded M-norm
    double max_relative_discard = 0;
    // Wall seconds: shared K/L evaluation, basis augmentation, Galerkin stages, truncation, reaction half steps.
    double t_kl = 0, t_basis = 0, t_galerkin = 0, t_truncate = 0, t_reaction = 0;
    // Inside operator evaluations: spatial operators on Z, angular/projection products, limiter.
    double t_spatial = 0, t_products = 0, t_limiter = 0;
};
class ElementWenoDlra {
public:
    ElementWenoDlra(const ElementWeno &, DlraOptions = {});
    const ElementWeno &scheme;
    DlraOptions options;
    int qh; // number of low-rank (l >= 2) active moments

    // Compression of a full state (nodes x moments): truncated eigen-decomposition of the M-Gram matrix.
    DlraState compress(const Field &full) const;
    // Full state nodes x moments (for output and tests; costs nodes x q).
    Field expand(const DlraState &) const;
    // Density coefficient field (nodes x 4) of the state, usable wherever only the low moments are read.
    const Field &low(const DlraState &s) const { return s.low; }
    // ||a||_M of the full state.
    double norm(const DlraState &) const;
    // One time step (reaction half steps, K/L, augmentation, Galerkin SSP stages, truncation).
    DlraState advance(double time, double dt, const DlraState &, ElementWenoStats * = nullptr,
                      DlraStats * = nullptr) const;

    // Test access. For a = [low | Y V^T]: low_rate = the limited low rows of scheme.rate(a) (coefficients) and
    // high = F(a)_H U (the PN rate of the l >= 2 moments times U, nodes x U.cols()).
    void factored_rate(double time, const Field &low, const Field &Y, const Mat &V, const Mat &U, Field &low_rate,
                       Field &high, ElementWenoStats * = nullptr) const;
    // Test access: the reaction half step of the low moments (must equal scheme.react on the first four columns).
    Field react_low(const Field &low, double tau, double source_time, ElementWenoStats * = nullptr) const;

private:
    struct Context;    // angular matrices for a basis V
    struct Evaluation; // spatial operators applied to Z = [low | Y]
    struct Fixed;      // S step: operators on the fixed bases X^, W^
    // `old` (optional): state whose W are the first columns of Wh and whose caches are valid.
    Fixed prepare(const Field &Xh, const Field &MXh, const Mat &Wh, const Vec &gamma, const DlraState *old,
                  DlraStats *) const;
    // PN traces at the outgoing hemisphere points of all C6 ghost pairs (pairs x points), one GEMM per side:
    // row k = H[side_k] Z.row(node_k)^T.
    Field ghost_traces(const Field &Z, const std::array<Mat, 6> &H) const;
    // One fast Galerkin stage: limited low rate (coefficients) and dS/dt.
    void fast_stage(double time, const Fixed &, const Field &low, const Mat &S, Field &low_rate, Mat &Sdot,
                    ElementWenoStats *, DlraStats *) const;
    // PN face loads sum_faces (face mass) Z_face per side (boundary slots x Z.cols()).
    std::array<Field, 6> face_loads(const Field &Z) const;
    Context context(const Mat &V, const DlraState *cache = nullptr) const;
    Evaluation evaluate(double time, const Field &Z, const Context &, bool boundary_low, DlraStats *,
                        ElementWenoStats *) const;
    // F(a) T for T = E_L (U == nullptr, 4 columns) or T = E_H U; pre (optional) receives the boundary rows of the
    // residual before the mass solve (boundary slots x columns).
    Field full_rate(const Evaluation &, const Context &, const Mat *U, Field *pre, DlraStats *) const;
    // X^T M F(a) E_H U (X^T R): projected high rate, rows = X.cols(), columns = U.cols() (U == nullptr: identity).
    Mat projected_rate(const Evaluation &, const Context &, const Field &X, const Field &MX, const Mat *U,
                       DlraStats *) const;
    // Limited low-order rate (coefficients, nodes x 4) of an evaluation, with the boundary mass rate in stats.
    Field low_rate(const Evaluation &, const Context &, ElementWenoStats *, DlraStats *) const;
    // M-orthonormal basis [X, Q] spanning [X, K] (X already M-orthonormal; X may have zero columns).
    Field augment(const Field &X, const Field &MX, const Field &K) const;
    // Reaction half step of the whole factored state.
    DlraState react(const DlraState &, double tau, double source_time, ElementWenoStats *) const;
    // Re-orthonormalize X in the M inner product: X <- X L^-T, returns L^T (so that X_old = X_new L^T).
    Mat reorthonormalize(Field &X) const;
    std::array<Sparse, 3> phi_;       // A_d restricted to the active moments (row vector convention: rate -= g A_d)
    std::array<Mat, 3> phi_dense_;
    std::vector<int> degree_;         // l of every active moment
    std::array<Field, 6> bt_h_, harm_h_, datum_h_; // per side: B_H^T (q x q_H), H_H (points x q_H), datum_t_H; row-major
};
} // namespace m1pn
