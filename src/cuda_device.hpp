#pragma once
// Device-side entry points for the narrow transport kernels. No Eigen and no
// CUDA headers, so the host translation unit can call them either way.
namespace m1pn {
namespace cuda_device {
struct Csr {
    const int *row_ptr = nullptr;
    const int *col = nullptr;
    const double *value = nullptr;
    int rows = 0;
    int nnz = 0;
};
// Column nonzero list of one spatial matrix, in Eigen Transpose-iterator order.
struct ColumnPattern {
    const int *outer = nullptr;
    const int *index = nullptr;
    const double *value = nullptr;
    int columns = 0;
    int nnz = 0;
};
// Throws if the device cannot load an sm_86 kernel.
void startup();
void upload_operators(int nodes, int q, const Csr angular[3], const ColumnPattern scatter[3],
                      const Csr &mass, const double *lumped);
void angular_products(const double *state, double *flux0, double *flux1, double *flux2);
// fluxes_resident: the three flux fields are already the device result of the
// last angular_products call, so they are not uploaded again.
void directional_scatter(const double *rhs, double *rhs_out, const double *flux0, const double *flux1,
                         const double *flux2, bool fluxes_resident);
void mass_correction(const double *rate, double *correction);
} // namespace cuda_device
} // namespace m1pn
