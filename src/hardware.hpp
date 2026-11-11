#pragma once
// Machine-adapted defaults (docs/ELEMENT_WENO.md, "Hardware").
//
// Element WENO is limited by memory bandwidth. Measured: on the 6-core workstation (hyperthreading) 6 threads beat
// 8 and 12; on nx (2 x EPYC 9354, one NUMA node per socket) 32 threads on one socket beat 64 on both, because the
// fields are first touched by the master thread. The default is therefore one thread per physical core of the
// NUMA node of the first CPU the process may run on (Slurm/numactl affinity respected). OMP_NUM_THREADS or an
// explicit --threads always take precedence.
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#ifdef __linux__
#include <sched.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
namespace m1pn {
struct ThreadDefault {
    int threads = 1;
    std::string reason; // "OMP_NUM_THREADS", "physical cores of NUMA node k", or "OpenMP maximum"
};
inline ThreadDefault default_threads() {
    ThreadDefault out;
#ifdef _OPENMP
    out.threads = omp_get_max_threads();
    out.reason = "OpenMP maximum";
    if (std::getenv("OMP_NUM_THREADS")) {
        out.reason = "OMP_NUM_THREADS";
        return out;
    }
#ifdef __linux__
    // Allowed CPUs. With OMP_PROC_BIND/OMP_PLACES set, libgomp has already bound the master thread to its first
    // place before main(), so its affinity shows one core only; the place list still covers the whole process mask.
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (omp_get_num_places() > 0) {
        for (int place = 0; place < omp_get_num_places(); ++place) {
            const int count = omp_get_place_num_procs(place);
            std::vector<int> ids(count);
            omp_get_place_proc_ids(place, ids.data());
            for (int id : ids)
                if (id >= 0 && id < CPU_SETSIZE)
                    CPU_SET(id, &mask);
        }
    } else if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
        return out;
    auto read_int = [](const std::string &path, int &value) {
        std::ifstream f(path);
        return bool(f >> value);
    };
    auto numa_node = [](int cpu) {
        for (int node = 0; node < 64; ++node) {
            std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/node" + std::to_string(node) +
                            "/cpulist");
            if (f.good())
                return node;
        }
        return 0;
    };
    int first = -1;
    for (int cpu = 0; cpu < CPU_SETSIZE && first < 0; ++cpu)
        if (CPU_ISSET(cpu, &mask))
            first = cpu;
    if (first < 0)
        return out;
    const int node = numa_node(first);
    std::set<std::pair<int, int>> cores; // (package, core) of allowed CPUs on that NUMA node
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &mask) || numa_node(cpu) != node)
            continue;
        int package = 0, core = cpu;
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        if (!read_int(base + "core_id", core) || !read_int(base + "physical_package_id", package))
            return out;
        cores.insert({package, core});
    }
    if (!cores.empty()) {
        out.threads = int(cores.size());
        out.reason = "physical cores of NUMA node " + std::to_string(node);
    }
#endif
#endif
    return out;
}
} // namespace m1pn
