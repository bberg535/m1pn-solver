#include "cuda_device.hpp"
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>
namespace m1pn {
namespace cuda_device {
namespace {
void fail(const std::string &what) { throw std::runtime_error(what); }
void check(cudaError_t error, const char *what) {
    if (error != cudaSuccess)
        fail(std::string("CUDA ") + what + " failed: " + cudaGetErrorString(error));
}
struct Buffer {
    void *data = nullptr;
    std::size_t bytes = 0;
    void release() {
        if (data)
            cudaFree(data);
        data = nullptr;
        bytes = 0;
    }
    void allocate(std::size_t count) {
        if (count == bytes)
            return;
        release();
        bytes = count;
        if (count == 0)
            return;
        cudaError_t error = cudaMalloc(&data, count);
        if (error != cudaSuccess) {
            data = nullptr;
            bytes = 0;
            fail(std::string("CUDA allocation of ") + std::to_string(count) +
                 " bytes failed: " + cudaGetErrorString(error));
        }
    }
    void upload(const void *host, std::size_t count) {
        allocate(count);
        if (count == 0)
            return;
        check(cudaMemcpy(data, host, count, cudaMemcpyHostToDevice), "upload");
    }
    void download(void *host, std::size_t count) const {
        if (count == 0)
            return;
        check(cudaMemcpy(host, data, count, cudaMemcpyDeviceToHost), "download");
    }
    template <class T> T *as() const { return static_cast<T *>(data); }
};
struct Operators {
    bool ready = false;
    int nodes = 0;
    int q = 0;
    bool fused_scatter = true;
    Buffer a_row[3], a_col[3], a_val[3];
    Buffer s_outer[3], s_index[3], s_val[3];
    Buffer m_row, m_col, m_val, lumped;
    Buffer state, flux[3], rhs, rate, correction;
    void release() {
        for (int d = 0; d < 3; ++d) {
            a_row[d].release();
            a_col[d].release();
            a_val[d].release();
            s_outer[d].release();
            s_index[d].release();
            s_val[d].release();
            flux[d].release();
        }
        m_row.release();
        m_col.release();
        m_val.release();
        lumped.release();
        state.release();
        rhs.release();
        rate.release();
        correction.release();
        ready = false;
        nodes = 0;
        q = 0;
    }
};
Operators &operators() {
    static Operators stored;
    return stored;
}
std::size_t field_bytes(const Operators &op) {
    return std::size_t(op.nodes) * std::size_t(op.q) * sizeof(double);
}
int launch_threads(int q) {
    if (q >= 128)
        return 128;
    int threads = 32;
    while (threads < q)
        threads *= 2;
    return threads;
}
__global__ void probe_kernel() {}
__global__ void angular_kernel(int q, const int *row0, const int *col0, const double *val0,
                               const int *row1, const int *col1, const double *val1, const int *row2,
                               const int *col2, const double *val2, const double *state, double *flux0,
                               double *flux1, double *flux2) {
    extern __shared__ double nodal[];
    const int node = blockIdx.x;
    const double *in = state + std::size_t(node) * q;
    for (int c = threadIdx.x; c < q; c += blockDim.x)
        nodal[c] = in[c];
    __syncthreads();
    const int *rows[3] = {row0, row1, row2};
    const int *cols[3] = {col0, col1, col2};
    const double *vals[3] = {val0, val1, val2};
    double *flux[3] = {flux0, flux1, flux2};
    for (int d = 0; d < 3; ++d) {
        double *out = flux[d] + std::size_t(node) * q;
        for (int i = threadIdx.x; i < q; i += blockDim.x) {
            double tmp = 0.;
            const int begin = rows[d][i];
            const int end = rows[d][i + 1];
            for (int k = begin; k < end; ++k)
                tmp += vals[d][k] * nodal[cols[d][k]];
            out[i] = tmp;
        }
    }
}
// One block per matrix column. Threads walk the moment width and axpy that
// column of C[d]^T into rhs, fusing the three directions the way the CPU
// column axpy does. Distinct columns share destination rows, so the adds are
// atomic; each product itself is a plain multiply.
__global__ void column_scatter_kernel(int q, const int *outer0, const int *index0, const double *value0,
                                      const int *outer1, const int *index1, const double *value1,
                                      const int *outer2, const int *index2, const double *value2,
                                      const double *flux0, const double *flux1, const double *flux2,
                                      double *rhs) {
    const int column = blockIdx.x;
    const int *outer[3] = {outer0, outer1, outer2};
    const int *index[3] = {index0, index1, index2};
    const double *value[3] = {value0, value1, value2};
    const double *flux[3] = {flux0, flux1, flux2};
    for (int d = 0; d < 3; ++d) {
        const double *src = flux[d] + std::size_t(column) * q;
        const int begin = outer[d][column];
        const int end = outer[d][column + 1];
        for (int k = begin; k < end; ++k) {
            const double scale = value[d][k];
            double *dst = rhs + std::size_t(index[d][k]) * q;
            for (int c = threadIdx.x; c < q; c += blockDim.x)
                atomicAdd(dst + c, scale * src[c]);
        }
    }
}
__global__ void column_scatter_one(int q, const int *outer, const int *index, const double *value,
                                   const double *flux, double *rhs) {
    const int column = blockIdx.x;
    const double *src = flux + std::size_t(column) * q;
    const int begin = outer[column];
    const int end = outer[column + 1];
    for (int k = begin; k < end; ++k) {
        const double scale = value[k];
        double *dst = rhs + std::size_t(index[k]) * q;
        for (int c = threadIdx.x; c < q; c += blockDim.x)
            atomicAdd(dst + c, scale * src[c]);
    }
}
__global__ void mass_kernel(int q, const int *row_ptr, const int *col, const double *value,
                            const double *lumped, const double *rate, double *correction) {
    const int row = blockIdx.x;
    const int begin = row_ptr[row];
    const int end = row_ptr[row + 1];
    for (int c = threadIdx.x; c < q; c += blockDim.x) {
        double acc = 0.;
        for (int k = begin; k < end; ++k)
            acc += value[k] * rate[std::size_t(col[k]) * q + c];
        const std::size_t at = std::size_t(row) * q + c;
        correction[at] = lumped[row] * rate[at] - acc;
    }
}
void launch_checked(const char *what) {
    check(cudaGetLastError(), what);
    check(cudaDeviceSynchronize(), what);
}
} // namespace
void startup() {
    int devices = 0;
    cudaError_t error = cudaGetDeviceCount(&devices);
    if (error != cudaSuccess || devices < 1) {
        const char *detail = error == cudaSuccess ? "no CUDA device was enumerated"
                                                   : cudaGetErrorString(error);
        fail(std::string("CUDA was requested but no usable CUDA device was found: ") + detail);
    }
    check(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    probe_kernel<<<1, 1>>>();
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::ostringstream message;
        message << "CUDA device " << prop.name << " (compute capability " << prop.major << "."
                << prop.minor
                << ") cannot run the compiled sm_86 kernels: " << cudaGetErrorString(error);
        fail(message.str());
    }
}
void upload_operators(int nodes, int q, const Csr angular[3], const ColumnPattern scatter[3],
                      const Csr &mass, const double *lumped) {
    Operators &op = operators();
    op.release();
    if (nodes < 0 || q <= 0)
        fail("CUDA transport dimensions are invalid");
    if (std::size_t(q) > std::size_t(48 * 1024 / sizeof(double)))
        fail("CUDA angular row exceeds the shared-memory capacity of this kernel");
    try {
        op.nodes = nodes;
        op.q = q;
        op.fused_scatter = true;
        for (int d = 0; d < 3; ++d) {
            if (angular[d].rows != q)
                fail("CUDA angular matrix row count does not match the PN width");
            if (scatter[d].columns != nodes)
                fail("CUDA scatter column count does not match the mesh");
            if (d > 0 && scatter[d].columns != scatter[0].columns)
                op.fused_scatter = false;
            op.a_row[d].upload(angular[d].row_ptr, std::size_t(q + 1) * sizeof(int));
            op.a_col[d].upload(angular[d].col, std::size_t(angular[d].nnz) * sizeof(int));
            op.a_val[d].upload(angular[d].value, std::size_t(angular[d].nnz) * sizeof(double));
            op.s_outer[d].upload(scatter[d].outer, std::size_t(nodes + 1) * sizeof(int));
            op.s_index[d].upload(scatter[d].index, std::size_t(scatter[d].nnz) * sizeof(int));
            op.s_val[d].upload(scatter[d].value, std::size_t(scatter[d].nnz) * sizeof(double));
        }
        if (mass.rows != nodes)
            fail("CUDA mass matrix dimensions do not match the mesh");
        op.m_row.upload(mass.row_ptr, std::size_t(nodes + 1) * sizeof(int));
        op.m_col.upload(mass.col, std::size_t(mass.nnz) * sizeof(int));
        op.m_val.upload(mass.value, std::size_t(mass.nnz) * sizeof(double));
        op.lumped.upload(lumped, std::size_t(nodes) * sizeof(double));
        const std::size_t bytes = field_bytes(op);
        op.state.allocate(bytes);
        for (int d = 0; d < 3; ++d)
            op.flux[d].allocate(bytes);
        op.rhs.allocate(bytes);
        op.rate.allocate(bytes);
        op.correction.allocate(bytes);
        op.ready = true;
    } catch (...) {
        op.release();
        throw;
    }
}
void angular_products(const double *state, double *flux0, double *flux1, double *flux2) {
    Operators &op = operators();
    if (!op.ready)
        fail("CUDA angular product called before the operators were uploaded");
    if (op.nodes == 0)
        return;
    const std::size_t bytes = field_bytes(op);
    op.state.upload(state, bytes);
    const int threads = launch_threads(op.q);
    angular_kernel<<<op.nodes, threads, std::size_t(op.q) * sizeof(double)>>>(
        op.q, op.a_row[0].as<int>(), op.a_col[0].as<int>(), op.a_val[0].as<double>(),
        op.a_row[1].as<int>(), op.a_col[1].as<int>(), op.a_val[1].as<double>(),
        op.a_row[2].as<int>(), op.a_col[2].as<int>(), op.a_val[2].as<double>(),
        op.state.as<double>(), op.flux[0].as<double>(), op.flux[1].as<double>(),
        op.flux[2].as<double>());
    launch_checked("angular product");
    op.flux[0].download(flux0, bytes);
    op.flux[1].download(flux1, bytes);
    op.flux[2].download(flux2, bytes);
}
void directional_scatter(const double *rhs, double *rhs_out, const double *flux0, const double *flux1,
                         const double *flux2, bool fluxes_resident) {
    Operators &op = operators();
    if (!op.ready)
        fail("CUDA directional scatter called before the operators were uploaded");
    if (op.nodes == 0)
        return;
    const std::size_t bytes = field_bytes(op);
    if (!fluxes_resident) {
        op.flux[0].upload(flux0, bytes);
        op.flux[1].upload(flux1, bytes);
        op.flux[2].upload(flux2, bytes);
    }
    op.rhs.upload(rhs, bytes);
    const int threads = launch_threads(op.q);
    if (op.fused_scatter) {
        column_scatter_kernel<<<op.nodes, threads>>>(
            op.q, op.s_outer[0].as<int>(), op.s_index[0].as<int>(), op.s_val[0].as<double>(),
            op.s_outer[1].as<int>(), op.s_index[1].as<int>(), op.s_val[1].as<double>(),
            op.s_outer[2].as<int>(), op.s_index[2].as<int>(), op.s_val[2].as<double>(),
            op.flux[0].as<double>(), op.flux[1].as<double>(), op.flux[2].as<double>(),
            op.rhs.as<double>());
        launch_checked("directional scatter");
    } else {
        const double *flux[3] = {op.flux[0].as<double>(), op.flux[1].as<double>(),
                                 op.flux[2].as<double>()};
        for (int d = 0; d < 3; ++d) {
            column_scatter_one<<<op.nodes, threads>>>(op.q, op.s_outer[d].as<int>(),
                                                      op.s_index[d].as<int>(), op.s_val[d].as<double>(),
                                                      flux[d], op.rhs.as<double>());
            launch_checked("directional scatter");
        }
    }
    op.rhs.download(rhs_out, bytes);
}
void mass_correction(const double *rate, double *correction) {
    Operators &op = operators();
    if (!op.ready)
        fail("CUDA mass correction called before the operators were uploaded");
    if (op.nodes == 0)
        return;
    const std::size_t bytes = field_bytes(op);
    op.rate.upload(rate, bytes);
    const int threads = launch_threads(op.q);
    mass_kernel<<<op.nodes, threads>>>(op.q, op.m_row.as<int>(), op.m_col.as<int>(),
                                       op.m_val.as<double>(), op.lumped.as<double>(),
                                       op.rate.as<double>(), op.correction.as<double>());
    launch_checked("mass correction");
    op.correction.download(correction, bytes);
}
} // namespace cuda_device
} // namespace m1pn
