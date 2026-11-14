// MPI tests of the slab-decomposed element WENO (src/element_weno_mpi.hpp): the distributed line solves and the
// projected gradient against the serial global scheme, bitwise. Run with mpirun -np 2..4.
#include "element_weno_mpi.hpp"
#include <cstdio>
#include <random>
using namespace m1pn;
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int failures = 0;
    for (int degree : {1, 2}) {
        Angular pn(3);
        ElementWenoOptions o;
        o.degree = degree, o.dimension = 2, o.elements = 12, o.periodic = false;
        ElementWeno global(pn, o);
        const SlabPartition part = slab_partition(12, size, rank, 1, 3);
        ElementWeno local(pn, slab_options(o, part));
        ElementWenoMpi mpi(local, MPI_COMM_WORLD, part, 3);
        std::mt19937 rng(5);
        std::uniform_real_distribution<double> u(-1, 1);
        Field g(global.nodes, global.moments);
        for (int i = 0; i < g.rows(); ++i)
            for (int c = 0; c < g.cols(); ++c) g(i, c) = u(rng);
        Field l(local.nodes, local.moments);
        for (int i = 0; i < local.nodes; ++i) l.row(i) = g.row(mpi.global_id(i));
        auto check = [&](const char *what, const Field &G, const Field &L) {
            double d = 0;
            long equal = 0, total = 0;
            for (int i = 0; i < local.nodes; ++i)
                if (local.is_owned(i)) {
                    for (int c = 0; c < G.cols(); ++c) {
                        d = std::max(d, std::abs(G(mpi.global_id(i), c) - L(i, c)));
                        equal += G(mpi.global_id(i), c) == L(i, c), ++total;
                    }
                }
            const double dmax = mpi.max(d);
            const double neq = mpi.sum(double(total - equal));
            if (rank == 0)
                std::printf("%s p=%d: max |serial - mpi| = %.3e, unequal entries %g\n", what, degree, dmax, neq);
            failures += neq != 0;
        };
        check("solve_mass", global.solve_mass(g), local.solve_mass(l));
        for (int d = 0; d < 2; ++d)
            check(d ? "gradient y" : "gradient x", global.gradient(g, d), local.gradient(l, d));
    }
    MPI_Finalize();
    return failures && rank == 0 ? 1 : 0;
}
