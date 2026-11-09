#include "cuda_path.hpp"
#include <Eigen/Sparse>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>
#ifdef M1PN_HAS_CUDA
#include "cuda_device.hpp"
#endif
namespace m1pn {
namespace {
int g_override = 0;
#ifdef M1PN_HAS_CUDA
bool g_ready = false;
int g_nodes = 0, g_q = 0;
const double *g_angular_values = nullptr;
const double *g_scatter_values = nullptr;
const double *g_mass_values = nullptr;
const double *g_flux_host[3] = {nullptr, nullptr, nullptr};
bool g_flux_resident = false;
void require_packed(const Field &field, const char *what) {
    require(field.innerStride() == 1 && field.outerStride() == field.cols(),
            std::string(what) + " is not a packed row-major field");
}
struct OwnedColumn {
    std::vector<int> outer, index;
    std::vector<double> value;
};
void column_pattern(const Sparse &matrix, OwnedColumn &pattern) {
    pattern.outer.clear();
    pattern.index.clear();
    pattern.value.clear();
    using Transposed = Eigen::Transpose<const Sparse>;
    const Transposed transposed = matrix.transpose();
    Eigen::internal::evaluator<Transposed> eval(transposed);
    pattern.outer.reserve(std::size_t(transposed.outerSize()) + 1);
    for (int j = 0; j < int(transposed.outerSize()); ++j) {
        pattern.outer.push_back(int(pattern.index.size()));
        for (typename Eigen::internal::evaluator<Transposed>::InnerIterator it(eval, j); it; ++it) {
            require(it.index() >= 0 && it.index() <= std::numeric_limits<int>::max(),
                    "CUDA scatter index does not fit in int");
            pattern.index.push_back(int(it.index()));
            pattern.value.push_back(it.value());
        }
    }
    pattern.outer.push_back(int(pattern.index.size()));
}
cuda_device::Csr view_csr(const Sparse &matrix) {
    require(matrix.isCompressed(), "CUDA sparse operator must be compressed");
    cuda_device::Csr view;
    view.rows = int(matrix.rows());
    view.nnz = int(matrix.nonZeros());
    view.row_ptr = matrix.outerIndexPtr();
    view.col = matrix.innerIndexPtr();
    view.value = matrix.valuePtr();
    require(view.rows >= 0 && view.nnz >= 0, "CUDA sparse operator size is invalid");
    return view;
}
void forget_flux() {
    g_flux_resident = false;
    g_flux_host[0] = g_flux_host[1] = g_flux_host[2] = nullptr;
}
#endif
} // namespace
void cuda_set_requested(bool on) { g_override = on ? 1 : -1; }
bool cuda_requested() {
    if (g_override > 0)
        return true;
    if (g_override < 0)
        return false;
    const char *value = std::getenv("M1PN_CUDA");
    return value && std::string(value) == "1";
}
bool cuda_ready() {
#ifdef M1PN_HAS_CUDA
    return g_ready;
#else
    return false;
#endif
}
void cuda_prepare(const Angular &pn, const Spatial &fe) {
#ifndef M1PN_HAS_CUDA
    (void)pn;
    (void)fe;
    throw std::runtime_error(
        "CUDA was requested but this binary was built without CUDA support");
#else
    require(pn.q > 0 && fe.nodes >= 0, "CUDA transport dimensions are invalid");
    for (int d = 0; d < 3; ++d) {
        require(pn.A[d].rows() == pn.q && pn.A[d].cols() == pn.q,
                "CUDA angular matrix is not the PN width");
        require(fe.C[d].rows() == fe.nodes && fe.C[d].cols() == fe.nodes,
                "CUDA transport matrix does not match the mesh");
    }
    require(fe.M.rows() == fe.nodes && fe.M.cols() == fe.nodes && fe.mass.size() == fe.nodes,
            "CUDA mass operator does not match the mesh");
    cuda_device::startup();
    cuda_device::Csr angular[3];
    for (int d = 0; d < 3; ++d)
        angular[d] = view_csr(pn.A[d]);
    OwnedColumn columns[3];
    cuda_device::ColumnPattern scatter[3];
    for (int d = 0; d < 3; ++d) {
        column_pattern(fe.C[d], columns[d]);
        require(int(columns[d].outer.size()) == fe.nodes + 1,
                "CUDA scatter pattern does not cover every node");
        scatter[d].outer = columns[d].outer.data();
        scatter[d].index = columns[d].index.data();
        scatter[d].value = columns[d].value.data();
        scatter[d].columns = fe.nodes;
        scatter[d].nnz = int(columns[d].index.size());
    }
    const cuda_device::Csr mass = view_csr(fe.M);
    try {
        cuda_device::upload_operators(fe.nodes, pn.q, angular, scatter, mass, fe.mass.data());
    } catch (...) {
        g_ready = false;
        forget_flux();
        throw;
    }
    g_nodes = fe.nodes;
    g_q = pn.q;
    g_angular_values = pn.A[0].valuePtr();
    g_scatter_values = fe.C[0].valuePtr();
    g_mass_values = fe.M.valuePtr();
    forget_flux();
    g_ready = true;
#endif
}
void cuda_fill_angular_products(const Angular &pn, const Field &state, std::array<Field, 3> &flux) {
#ifndef M1PN_HAS_CUDA
    (void)pn;
    (void)state;
    (void)flux;
    throw std::runtime_error("CUDA angular product is unavailable in this binary");
#else
    require(g_ready, "CUDA angular product was requested before the device operators were uploaded");
    require(pn.A[0].valuePtr() == g_angular_values,
            "CUDA angular operators do not match the prepared matrix");
    require(state.rows() == g_nodes && state.cols() == g_q && pn.q == g_q,
            "CUDA nodal state does not match the prepared mesh");
    require_packed(state, "CUDA nodal state");
    const int n = int(state.rows());
    for (int d = 0; d < 3; ++d)
        if (flux[d].rows() != n || flux[d].cols() != pn.q)
            flux[d].resize(n, pn.q);
    cuda_device::angular_products(state.data(), flux[0].data(), flux[1].data(), flux[2].data());
    for (int d = 0; d < 3; ++d)
        g_flux_host[d] = flux[d].data();
    g_flux_resident = true;
#endif
}
void cuda_add_directional_scatter(Field &destination, const std::array<Sparse, 3> &matrices,
                                  const std::array<Field, 3> &flux) {
#ifndef M1PN_HAS_CUDA
    (void)destination;
    (void)matrices;
    (void)flux;
    throw std::runtime_error("CUDA directional scatter is unavailable in this binary");
#else
    require(g_ready, "CUDA scatter was requested before the device operators were uploaded");
    require(matrices[0].valuePtr() == g_scatter_values,
            "CUDA scatter operators do not match the prepared matrix");
    require(destination.rows() == g_nodes && destination.cols() == g_q,
            "CUDA scatter destination does not match the prepared mesh");
    require_packed(destination, "CUDA scatter destination");
    for (int d = 0; d < 3; ++d) {
        require(flux[d].rows() == g_nodes && flux[d].cols() == g_q,
                "CUDA scatter flux does not match the prepared mesh");
        require_packed(flux[d], "CUDA scatter flux");
    }
    const bool resident = g_flux_resident && flux[0].data() == g_flux_host[0] &&
                          flux[1].data() == g_flux_host[1] && flux[2].data() == g_flux_host[2];
    cuda_device::directional_scatter(destination.data(), destination.data(), flux[0].data(),
                                     flux[1].data(), flux[2].data(), resident);
#endif
}
void cuda_mass_correction(const Spatial &fe, const Field &rate, Field &correction) {
#ifndef M1PN_HAS_CUDA
    (void)fe;
    (void)rate;
    (void)correction;
    throw std::runtime_error("CUDA mass correction is unavailable in this binary");
#else
    require(g_ready, "CUDA mass correction was requested before the device operators were uploaded");
    require(fe.M.valuePtr() == g_mass_values, "CUDA mass operator does not match the prepared matrix");
    require(rate.rows() == g_nodes && correction.rows() == g_nodes && rate.cols() == g_q &&
                correction.cols() == g_q,
            "CUDA mass-correction fields do not match the prepared mesh");
    require_packed(rate, "CUDA mass-correction rate");
    require_packed(correction, "CUDA mass-correction result");
    cuda_device::mass_correction(rate.data(), correction.data());
#endif
}
} // namespace m1pn
