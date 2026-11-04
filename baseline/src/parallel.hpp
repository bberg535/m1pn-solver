#pragma once
#include "common.hpp"
#include <atomic>
#include <exception>
#ifdef M1PN_MPI
#include <mpi.h>
#endif
namespace m1pn {
struct Runtime {
    int rank = 0, size = 1;
    bool deterministic = false;
    Runtime() {
#ifdef M1PN_MPI
        int init = 0;
        MPI_Initialized(&init);
        if (init) {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
        }
#endif
    }
    bool owns(std::size_t i) const { return int(i % size) == rank; }
    void barrier() const {
#ifdef M1PN_MPI
        if (size > 1)
            MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
    void finish_local_phase(int local_failure, const std::string &message) const {
        int failure = local_failure;
        std::string reason = message;
#ifdef M1PN_MPI
        if (size > 1) {
            MPI_Allreduce(MPI_IN_PLACE, &failure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (failure) {
                int owner = local_failure ? rank : size;
                MPI_Allreduce(MPI_IN_PLACE, &owner, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
                int length = rank == owner ? int(reason.size()) : 0;
                MPI_Bcast(&length, 1, MPI_INT, owner, MPI_COMM_WORLD);
                reason.resize(length);
                MPI_Bcast(reason.data(), length, MPI_CHAR, owner, MPI_COMM_WORLD);
            }
        }
#endif
        if (failure == 1)
            throw StepRejected(reason);
        if (failure)
            throw std::runtime_error(reason);
    }
    void sum(Field &a) const {
#ifdef M1PN_MPI
        if (size == 1)
            return;
        require(a.size() <= std::numeric_limits<int>::max(), "MPI message exceeds count limit");
        if (!deterministic) {
            MPI_Allreduce(MPI_IN_PLACE, a.data(), int(a.size()), MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            return;
        }
        Field result = Field::Zero(a.rows(), a.cols()), part(a.rows(), a.cols());
        for (int r = 0; r < size; ++r) {
            if (rank == r)
                part = a;
            MPI_Bcast(part.data(), int(a.size()), MPI_DOUBLE, r, MPI_COMM_WORLD);
            result += part;
        }
        a = std::move(result);
#else
        (void)a;
#endif
    }
    double reduce(double a, int mode) const {
#ifdef M1PN_MPI
        if (size > 1) {
            double b;
            MPI_Allreduce(&a, &b, 1, MPI_DOUBLE,
                          mode < 0   ? MPI_MIN
                          : mode > 0 ? MPI_MAX
                                     : MPI_SUM,
                          MPI_COMM_WORLD);
            return b;
        }
#else
        (void)mode;
#endif
        return a;
    }
};
template <class F> void parallel_for(int n, F f) {
    std::exception_ptr error;
    std::atomic<bool> failed{false};
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        if (failed.load())
            continue;
        try {
            f(i);
        } catch (...) {
#pragma omp critical(m1pn_exception)
            {
                if (!error)
                    error = std::current_exception();
                failed = true;
            }
        }
    }
    if (error)
        std::rethrow_exception(error);
}
} // namespace m1pn
