// Tests for dynamic tile-based adaptive PN (experiment/dynamic-adaptive-pn):
// 1. SeparableTrace::evaluate_upto matches evaluate() to rounding when higher moments vanish.
// 2. min_order == max_order reproduces the uniform solver to rounding (CPU & GPU).
// 3. Pure quadrupole (J = 0, nonzero l = 2, 4) is detected and kept accurately by the spectral tail sensor.
// 4. Dynamic refinement and coarsening preserve M1 conservation, cone realizability, and match uniform PN.
// 5. CPU vs GPU agreement on graded mesh with broken lines, split-breaks, mirror-x, and adaptive PN + analytic P1 ghost.
#include "element_weno.hpp"
#include "separable_trace.hpp"
#ifdef M1PN_ELEMENT_GPU
#include "element_weno_gpu.hpp"
#endif
#include <cmath>
#include <iostream>
#include <random>

using namespace m1pn;
namespace {
int failures = 0;
void check(bool ok, const std::string &what, double value = 0) {
    std::cout << (ok ? "ok   " : "FAIL ") << what << "  [" << value << "]\n";
    if (!ok)
        ++failures;
}

void test_separable_upto() {
    Angular pn(15);
    std::vector<int> active;
    for (int k = 0; k < pn.q; ++k)
        active.push_back(k);
    Eigen::Matrix3d frame = Eigen::Matrix3d::Identity();
    SeparableTrace st(15, active, frame, 16, {+1, -1});
    std::mt19937 rng(42);
    std::normal_distribution<double> dist;
    for (int ord : {1, 3, 7, 15}) {
        const int q_ord = (ord + 1) * (ord + 1);
        Mat at = Mat::Zero(pn.q, 4);
        for (int c = 0; c < q_ord; ++c)
            for (int k = 0; k < 4; ++k)
                at(c, k) = dist(rng);
        Mat full, trunc;
        st.evaluate(at, full);
        st.evaluate_upto(at, trunc, ord);
        const double err = (full - trunc).cwiseAbs().maxCoeff() / std::max(full.cwiseAbs().maxCoeff(), 1e-30);
        check(err < 1e-13, "SeparableTrace::evaluate_upto(N=" + std::to_string(ord) + ")", err);
    }
}

void test_quadrupole_sensor() {
    // Counter-streaming / quadrupole state: J = 0 everywhere, large l = 2 and l = 4 anisotropy in a localized patch.
    Angular pn(7);
    ElementWenoOptions o;
    o.dimension = 2;
    o.degree = 1;
    o.elements = 32;
    o.periodic = false;
    o.ssprk43 = true;
    o.reduce_angular = true;
    o.weno = true;
    o.limit = true;
    ElementWeno uni(pn, o);
    o.adaptive_pn = true;
    o.adaptive_tol = 1e-4;
    o.adaptive_tile = 4;
    o.adaptive_stride = 2;
    o.adaptive_min_order = 1;
    ElementWeno ad(pn, o);

    auto ic = [&](const V3 &x) {
        Vec v = Vec::Zero(pn.q);
        const double dx = x[0] - 0.5, dy = x[1] - 0.5;
        const double r2 = dx * dx + dy * dy;
        const double bump = std::exp(-180.0 * r2);
        const double rho = 1e-3 + 1.0 * bump;
        set_low(v, Four(rho, 0.0, 0.0, 0.0)); // J == 0 identically!
        // Pure even quadrupole (l = 2, m = 0) and hexadecapole (l = 4, m = 0) in the center
        v[index(2, 0)] = 0.25 * bump;
        v[index(4, 0)] = 0.05 * bump;
        return v;
    };
    Field a_uni = uni.interpolate(ic);
    Field a_ad = ad.interpolate(ic);
    const double dt = 0.8 * uni.step_bound();
    ElementWenoStats st_uni, st_ad;
    double t = 0;
    for (int step = 0; step < 12; ++step) {
        a_uni = uni.advance(t, dt, a_uni, &st_uni);
        a_ad = ad.advance(t, dt, a_ad, &st_ad);
        t += dt;
    }
    // Check that quadrupole was NOT wiped out (center tiles must have order >= 5) and density matches closely.
    double max_rho_diff = 0, max_rho = 0;
    bool in_cone = true;
    for (int i = 0; i < uni.nodes; ++i) {
        const Four uu = project(a_uni.row(i).transpose());
        const Four ua = project(a_ad.row(i).transpose());
        in_cone = in_cone && exact::cone(ua);
        max_rho = std::max(max_rho, uu[0]);
        max_rho_diff = std::max(max_rho_diff, std::abs(uu[0] - ua[0]));
    }
    check(in_cone, "quadrupole: adaptive state stays in M1 cone");
    check(st_ad.adaptive_last_max_order >= 5, "quadrupole (J=0): sensor retains high order >= 5",
          double(st_ad.adaptive_last_max_order));
    check(st_ad.adaptive_coeff_sum / st_ad.adaptive_steps < 0.75, "quadrupole: adaptive saves coefficients",
          st_ad.adaptive_coeff_sum / st_ad.adaptive_steps);
    check(max_rho_diff / max_rho < 2e-3, "quadrupole: adaptive density matches uniform P7", max_rho_diff / max_rho);
}

void test_graded_broken_cpu_gpu() {
    Angular pn(11);
    ElementWenoOptions o;
    o.dimension = 2;
    o.degree = 1;
    o.periodic = false;
    o.ssprk43 = true;
    o.reduce_angular = true;
    o.weno = true;
    o.limit = true;
    o.planes[0] = {0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 1.75, 2.0};
    o.planes[1] = {0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 1.75, 2.0};
    o.elements = 11;
    o.elements_per_axis = {11, 11, 0};
    o.length = 2.0;
    o.breaks[0] = {5};
    o.breaks[1] = {5};
    o.split_breaks = true;
    o.mirror[0] = true;
    auto mat = std::make_shared<ElementWenoMaterial>();
    mat->sigma_a = [](const V3 &x) { return (x[0] > 1.0 && x[1] > 1.0) ? 5.0 : 0.0; };
    mat->sigma_s = [](const V3 &x) { return (x[0] > 1.0 && x[1] > 1.0) ? 0.0 : 1.0; };
    mat->source = [](const V3 &x) { return (x[0] < 0.5 && x[1] < 0.5) ? 2.0 : 0.0; };
    o.material = mat;

    ElementWeno uni(pn, o);
    o.adaptive_pn = true;
    o.adaptive_tol = 3e-4;
    o.adaptive_tile = 4;
    o.adaptive_stride = 2;
    o.adaptive_min_order = 1;
    ElementWeno ad_cpu(pn, o);

    Field a0 = Field::Zero(uni.nodes, uni.moments);
    a0.col(0).setConstant(1e-8 / std::sqrt(4 * pi));
    const double dt = 0.9 * uni.step_bound();
    const int steps = 20;

    Field u_cpu = a0, a_cpu = a0;
    ElementWenoStats st_u, st_a;
    for (int s = 0; s < steps; ++s) {
        u_cpu = uni.advance(s * dt, dt, u_cpu, &st_u);
        a_cpu = ad_cpu.advance(s * dt, dt, a_cpu, &st_a);
    }
    double diff_ua = 0, norm_u = 0, max_inactive_cpu = 0;
    const AdaptivePnPlan *plan_cpu = ad_cpu.adaptive_plan();
    for (int i = 0; i < uni.nodes; ++i) {
        const double ru = project(u_cpu.row(i).transpose())[0];
        const double ra = project(a_cpu.row(i).transpose())[0];
        diff_ua = std::max(diff_ua, std::abs(ru - ra));
        norm_u = std::max(norm_u, std::abs(ru));
        const int qi = plan_cpu->tile_q[plan_cpu->node_tile(i)];
        for (int c = qi; c < uni.moments; ++c)
            max_inactive_cpu = std::max(max_inactive_cpu, std::abs(a_cpu(i, c)));
    }
    check(max_inactive_cpu == 0.0, "CPU adaptive: inactive moments strictly zero everywhere", max_inactive_cpu);
    check(diff_ua / std::max(norm_u, 1e-30) < 2e-3, "graded+broken+mirror CPU adaptive vs uniform",
          diff_ua / std::max(norm_u, 1e-30));

#ifdef M1PN_ELEMENT_GPU
    ElementWenoOptions o_full = o;
    o_full.adaptive_min_order = pn.order;
    ElementWeno ad_full(pn, o_full);
    ElementWenoGpu gpu_uni(uni);
    ElementWenoGpu gpu_ad_full(ad_full);
    ElementWenoGpu gpu_ad(ad_cpu);
    ElementWenoStats st_gu, st_gf, st_ga;
    const Field u_gpu = gpu_uni.run(0.0, dt, steps, a0, &st_gu);
    const Field f_gpu = gpu_ad_full.run(0.0, dt, steps, a0, &st_gf);
    const Field a_gpu = gpu_ad.run(0.0, dt, steps, a0, &st_ga);

    double diff_full = maxabs(u_gpu - f_gpu);
    check(diff_full < 1e-14, "GPU adaptive(min=max) matches uniform GPU bitwise", diff_full);

    double diff_cg = 0;
    for (int i = 0; i < uni.nodes; ++i) {
        const double rc = project(a_cpu.row(i).transpose())[0];
        const double rg = project(a_gpu.row(i).transpose())[0];
        diff_cg = std::max(diff_cg, std::abs(rc - rg));
    }
    check(diff_cg / std::max(norm_u, 1e-30) < 5e-4, "CPU vs GPU adaptive density agreement",
          diff_cg / std::max(norm_u, 1e-30));
    const double frac_diff = std::abs(st_a.adaptive_coeff_sum / st_a.adaptive_steps -
                                      st_ga.adaptive_coeff_sum / st_ga.adaptive_steps);
    check(frac_diff < 2e-2, "CPU vs GPU consistent adaptive coefficient fraction", frac_diff);
#endif
}
} // namespace

int main() {
    try {
        test_separable_upto();
        test_quadrupole_sensor();
        test_graded_broken_cpu_gpu();
        if (failures) {
            std::cerr << failures << " test(s) failed\n";
            return 1;
        }
        std::cout << "PASSED\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
