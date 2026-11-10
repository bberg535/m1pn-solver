#pragma once
// Test-only access to the private members of ElementWeno (no production change).
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <functional>
#include <memory>
#include <array>
#include <vector>
#include <map>
#include <optional>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <random>
#include <atomic>
#define private public
#include "element_weno.hpp"
#undef private
#include "base_bar.hpp"
namespace m1pn {
extern bool audit_low_order, audit_skip_element_limit, audit_skip_boundary_limit;
extern std::vector<double> audit_factor, audit_theta, audit_scale;
} // namespace m1pn
using namespace m1pn;

struct J {
    std::ostringstream s;
    explicit J(const std::string &test) { s << std::setprecision(17) << "{\"test\":\"" << test << "\""; }
    J &add(const std::string &k, double v) {
        s << ",\"" << k << "\":";
        if (std::isfinite(v)) s << v; else s << "null";
        return *this;
    }
    J &str(const std::string &k, const std::string &v) { s << ",\"" << k << "\":\"" << v << "\""; return *this; }
    void done() { s << "}"; std::cout << s.str() << std::endl; }
};
inline bool nudge(Four &v, long *ulps = nullptr) {
    if (exact::cone(v)) return false;
    long k = 0;
    for (; k < 256 && !exact::cone(v); ++k) v[0] = std::nextafter(v[0], std::numeric_limits<double>::infinity());
    if (ulps) *ulps = std::max(*ulps, k);
    if (!exact::cone(v)) throw std::runtime_error("State left cone beyond 256 ulps");
    return true;
}

// AUDIT_FIX=legacy | new | sign=0/1,ghost=0/1,c1=0/1,wenoscale=0/1,sym=0/1 (comma list; unset = production defaults)
inline void apply_env(ElementWenoOptions &o) {
    const char *env = std::getenv("AUDIT_FIX");
    if (!env) return;
    const std::string v = env;
    if (v == "legacy") { o.vacuum_sign_limit = o.outward_bar_rounding = o.fullsphere_ghost = o.scale_invariant_weno = o.symmetric_residual = false; return; }
    if (v == "new") return;
    std::stringstream ss(v);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const auto eq = tok.find('=');
        const std::string k = tok.substr(0, eq);
        const bool on = tok.substr(eq + 1) != "0";
        if (k == "sign") o.vacuum_sign_limit = on; else if (k == "ghost") o.fullsphere_ghost = on; else if (k == "c1") o.outward_bar_rounding = on;
        else if (k == "wenoscale") o.scale_invariant_weno = on; else if (k == "sym") o.symmetric_residual = on;
    }
}
