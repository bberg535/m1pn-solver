// WENO sensor gamma_e on a stored state (study tool, not a ctest): reads the density column u0 of an m1pn_fast
// final.csv on the unit square (Q_p, n x n elements, 2D) and writes gamma per element. gamma depends on the density
// coefficients only (ElementWeno::weno_gamma), so the other moments are irrelevant; P_1 keeps the setup cheap.
//   m1pn_weno_gamma_field FINAL_CSV ELEMENTS DEGREE OUT_CSV
#include "element_weno.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
using namespace m1pn;
int main(int argc, char **argv) {
    try {
        require(argc == 5, "usage: m1pn_weno_gamma_field FINAL_CSV ELEMENTS DEGREE OUT_CSV");
        const int n = std::stoi(argv[2]), p = std::stoi(argv[3]);
        Angular pn(1);
        ElementWenoOptions o;
        o.degree = p, o.dimension = 2, o.elements = n, o.periodic = false;
        ElementWeno s(pn, o, Inflow{});
        // Node index by position (rounded to the node lattice).
        const double step = 1. / (n * p);
        std::map<std::pair<long, long>, int> index;
        for (int i = 0; i < s.nodes; ++i) {
            const V3 x = s.node_position(i);
            index[{std::lround(x[0] / step), std::lround(x[1] / step)}] = i;
        }
        Field a = Field::Zero(s.nodes, s.moments);
        std::ifstream in(argv[1]);
        std::string line;
        std::getline(in, line);
        std::vector<std::string> header;
        {
            std::stringstream h(line);
            for (std::string c; std::getline(h, c, ',');)
                header.push_back(c);
        }
        // With --fcs the written u0 is the total density; the grid (and WENO) carries the collided part only:
        // u0_collided = u0 - rho_uncollided / sqrt(4 pi).
        int ix = -1, iy = -1, iu = -1, iun = -1;
        for (int c = 0; c < int(header.size()); ++c)
            ix = header[c] == "x" ? c : ix, iy = header[c] == "y" ? c : iy, iu = header[c] == "u0" ? c : iu,
            iun = header[c] == "rho_uncollided" ? c : iun;
        require(ix >= 0 && iy >= 0 && iu >= 0, "final.csv needs x, y, u0");
        long count = 0;
        while (std::getline(in, line)) {
            std::stringstream r(line);
            std::vector<double> v;
            for (std::string c; std::getline(r, c, ',');)
                v.push_back(std::stod(c));
            const auto it = index.find({std::lround(v[ix] / step), std::lround(v[iy] / step)});
            if (it == index.end())
                throw std::runtime_error("sample point (" + std::to_string(v[ix]) + ", " + std::to_string(v[iy]) +
                                         ") is not a node; nodes " + std::to_string(s.nodes) + ", per axis " +
                                         std::to_string(s.nodes_per_axis[0]) + ", h " + std::to_string(s.h));
            a(it->second, 0) = v[iu] - (iun >= 0 ? v[iun] / std::sqrt(4 * pi) : 0.);
            ++count;
        }
        require(count == s.nodes, "final.csv must hold every node (Q1 output on the node grid)");
        std::cout << (iun >= 0 ? "FCS state: collided part (u0 - rho_uncollided/sqrt(4 pi))\n" : "");
        const Vec g = s.weno_gamma(a);
        std::ofstream out(argv[4]);
        out << std::setprecision(10) << "ix,iy,x,y,gamma\n";
        for (int e = 0; e < s.cells; ++e) {
            const int ex = e % n, ey = e / n;
            out << ex << "," << ey << "," << (ex + .5) / n << "," << (ey + .5) / n << "," << g[e] << "\n";
        }
        std::cout << "min gamma " << g.minCoeff() << ", elements with 1-gamma > 1e-3: " << (g.array() < 1 - 1e-3).count()
                  << " of " << s.cells << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "m1pn_weno_gamma_field: " << e.what() << "\n";
        return 1;
    }
}
