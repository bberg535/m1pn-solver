// MPI tests of the slab-decomposed element WENO (src/element_weno_mpi.hpp): the distributed line solves and the
// projected gradient against the serial global scheme, bitwise. Run with mpirun -np 2..4.
#include "element_weno_mpi.hpp"
#include <cstdlib>
#include <cstdio>
#include <random>
#include <string>
using namespace m1pn;
int main(int argc, char **argv) {
    int provided = 0;
    // UCX: the line pipeline sends many messages of some 10 KB from a thread that is rarely inside MPI; the
    // rendezvous protocol (default above a few KB) then stalls (NX, N=31, 4 ranks: 49 s -> 18.5 s with 4 MB).
    setenv("UCX_RNDV_THRESH", "4mb", 0); // keeps a value set by the user
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int failures = 0;
    if (argc > 1 && std::string(argv[1]) == "bench") { // bench E pn reps: serial vs distributed mass solve timing
        const int E = std::stoi(argv[2]), N = std::stoi(argv[3]), reps = std::stoi(argv[4]);
        Angular pn(N);
        ElementWenoOptions o;
        o.degree = 1, o.dimension = 2, o.elements = E, o.periodic = false;
        const SlabPartition part = slab_partition(E, size, rank, 1, 2);
        ElementWeno local(pn, slab_options(o, part));
        ElementWenoMpi mpi(local, MPI_COMM_WORLD, part);
        Field l = Field::Random(local.nodes, local.moments);
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int k = 0; k < reps; ++k) { Field r = local.solve_mass(l); }
        const double tl = mpi.max(MPI_Wtime() - t0);
        Field x = l;
        t0 = MPI_Wtime();
        for (int k = 0; k < reps; ++k) mpi.solve_axis(x);
        const double ta = mpi.max(MPI_Wtime() - t0);
        if (rank == 0) {
            ElementWeno global(pn, o);
            Field g = Field::Random(global.nodes, global.moments);
            t0 = MPI_Wtime();
            for (int k = 0; k < reps; ++k) { Field r = global.solve_mass(g); }
            const double tg = MPI_Wtime() - t0;
            std::printf("E=%d q=%d ranks=%d: serial solve_mass %.4f s/call, distributed solve_mass %.4f s/call, axis solve only %.4f s/call (wait %.4f)\n",
                        E, global.moments, size, tg / reps, tl / reps, ta / reps, mpi.t_wait / (2 * reps));
        }
        MPI_Finalize();
        return 0;
    }
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
