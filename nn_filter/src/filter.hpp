#pragma once
#include "time.hpp"
#include <fstream>

namespace m1pn::nnfilter {

// The authors' 2D four-layer residual tanh/LayerNorm network (evaluation only).
struct Network {
    int order = 0, features = 0, width = 0;
    std::array<Mat, 4> weight;
    std::array<Vec, 4> bias, scale, shift;
    std::array<double, 4> epsilon;
    Vec output;
    double output_bias = 0;
    explicit Network(const std::string &path) {
        std::ifstream f(path);
        std::string magic;
        int version;
        require(bool(f >> magic >> version >> order >> features >> width), "Cannot read filter model");
        require(magic == "NNFPN2D" && version == 1 &&
                    (order == 3 || order == 5 || order == 7 || order == 9) &&
                    features == 2 * order + 4 && width > 0 && width < 1024,
                "Unsupported filter model header");
        auto read = [&](auto &a) {
            for (int i = 0; i < a.rows(); ++i)
                for (int j = 0; j < a.cols(); ++j)
                    require(bool(f >> a(i,j)) && std::isfinite(a(i,j)), "Invalid filter model tensor");
        };
        for (int i = 0; i < 4; ++i) {
            weight[i].resize(width, i == 0 ? features : width);
            bias[i].resize(width); scale[i].resize(width); shift[i].resize(width);
            read(weight[i]); read(bias[i]);
            require(bool(f >> epsilon[i]) && std::isfinite(epsilon[i]) && epsilon[i] > 0,
                    "Invalid LayerNorm epsilon");
            read(scale[i]); read(shift[i]);
        }
        output.resize(width); read(output);
        require(bool(f >> output_bias) && std::isfinite(output_bias), "Invalid output bias");
        std::string extra;
        require(!(f >> extra), "Unexpected trailing model content");
    }
    double operator()(Vec x) const {
        require(x.size() == features && x.allFinite(), "Invalid network input");
        for (int i = 0; i < 4; ++i) {
            Vec y = (weight[i] * x + bias[i]).array().tanh();
            if (i > 0) y += x;
            const double mean = y.mean();
            const double variance = (y.array() - mean).square().mean();
            x = ((y.array() - mean) / std::sqrt(variance + epsilon[i]) * scale[i].array()
                 + shift[i].array()).matrix();
        }
        const double result = output.dot(x) + output_bias;
        require(std::isfinite(result), "Nonfinite filter prediction");
        return std::max(0., result);
    }
};

inline Vec coefficients(int order) {
    require(order >= 1, "Filter requires PN order >= 1");
    Vec f((order+1)*(order+1));
    for (int l = 0; l <= order; ++l)
        f.segment(l*l, 2*l+1).setConstant(std::pow(double(l)/(order+1), 4));
    return f;
}

inline Field damp(const Field &u, const Vec &sigma, const Vec &f, double tau) {
    require(tau >= 0 && std::isfinite(tau) && u.cols() == f.size() && u.rows() == sigma.size()
                && u.allFinite() && sigma.allFinite() && (sigma.array() >= 0).all()
                && f.allFinite() && (f.array() >= 0).all() && f[0] == 0,
            "Invalid filter damping arguments");
    if (tau == 0 || sigma.maxCoeff() == 0) return u;
    Field v(u.rows(), u.cols());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < int(u.rows()); ++i) {
        v(i,0) = u(i,0);
        // All components of one harmonic degree share f[k], hence one exponential.
        double last_f = -1, decay = 1;
        for (int k = 1; k < u.cols(); ++k) {
            if (f[k] != last_f) { last_f = f[k]; decay = std::exp(-tau * sigma[i] * f[k]); }
            v(i,k) = u(i,k) * decay;
        }
    }
    return v;
}

// Spatial sample standard deviation, correction=1, as in upstream torch.std.
inline Field normalize(Field x) {
    require(x.rows() > 1 && x.allFinite(), "Invalid feature field");
    for (int j = 0; j < x.cols(); ++j) {
        const double mean = x.col(j).mean();
        x.col(j).array() -= mean;
        const double sd = std::sqrt(x.col(j).squaredNorm() / (x.rows()-1));
        x.col(j) /= sd + 1e-10;
    }
    return x;
}

// Feature contract of published 2D code, with native FE derivatives. All SH
// components contribute to state norms; the derivative floor is used only on
// the l+1 reflection-even modes present in the authors' reduced 2D basis.
inline Field features(const Angular &pn, const Spatial &fe, const Field &u,
                      const Vec &absorption, const Vec &scattering, const Field &source) {
    require(fe.dimension == 2 && u.rows() == fe.nodes && u.cols() == pn.q && u.allFinite()
                && absorption.size() == fe.nodes && scattering.size() == fe.nodes
                && absorption.allFinite() && scattering.allFinite()
                && (absorption.array() >= 0).all() && (scattering.array() >= 0).all()
                && source.rows() == fe.nodes && source.cols() == pn.q && source.allFinite(),
            "Invalid NN feature arguments (requires 2D)");
    std::array<Field,2> derivative;
    for (int d = 0; d < 2; ++d) {
        const Field flux = u * pn.A[d].transpose();
        derivative[d].resize(fe.nodes, pn.q);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < fe.nodes; ++i)
            derivative[d].row(i) = (fe.C[d].row(i) * flux) / fe.mass[i];
    }
    Field raw = Field::Zero(fe.nodes, 2 * pn.order + 4);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < fe.nodes; ++i) {
        for (int l = 0; l <= pn.order; ++l) {
            raw(i,l) = (absorption[i] + scattering[i]) * u.row(i).segment(l*l,2*l+1).norm();
            double sum = 0;
            for (int m = -l; m <= l; ++m) {
                const int k = index(l,m);
                const double square = derivative[0](i,k)*derivative[0](i,k)
                                    + derivative[1](i,k)*derivative[1](i,k);
                sum += ((l+m)%2 == 0) ? std::max(square, 1e-12) : square;
            }
            raw(i,pn.order+1+l) = std::sqrt(sum);
        }
        raw(i,2*pn.order+2) = scattering[i] * u(i,0);
        raw(i,2*pn.order+3) = source(i,0);
    }
    return normalize(std::move(raw));
}

inline void reset_ledger(Field &ledger, int rows, int cols) {
    if (ledger.rows() != rows || ledger.cols() != cols) ledger.resize(rows, cols);
    double *data = ledger.data();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) std::fill_n(data + std::size_t(i) * cols, cols, 0.);
}
inline void add_ledger(Field &total, const Field &part) {
    const int rows = int(total.rows()), cols = int(total.cols());
    double *to = total.data();
    const double *from = part.data();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i)
        for (int k = 0; k < cols; ++k) to[std::size_t(i) * cols + k] += from[std::size_t(i) * cols + k];
}
// Parallel counterpart of Transport::check_state with the same exact predicates.
inline void check_state_parallel(const Field &u) {
    const int rows = int(u.rows()), cols = int(u.cols());
    int nonfinite = 0, failing = -1;
#pragma omp parallel for schedule(static) reduction(+ : nonfinite)
    for (int i = 0; i < rows; ++i) {
        const double *row = &u.coeff(i, 0);
        for (int k = 0; k < cols; ++k) nonfinite += !std::isfinite(row[k]);
        if (!exact::cone(project(u.row(i).transpose()))) {
#pragma omp critical(m1pn_filter_failing)
            if (failing < 0 || i < failing) failing = i;
        }
    }
    require(nonfinite == 0, "Invalid/nonfinite PN field");
    if (failing >= 0)
        cone_require(false, "Stored PN projection outside M1 cone at node " + std::to_string(failing));
}

struct Filter {
    std::string mode;
    double constant;
    std::unique_ptr<Network> network;
    Vec f;
    double elapsed = 0;
    long evaluations = 0;
    Filter(int order, std::string kind, double strength, const std::string &path)
        : mode(std::move(kind)), constant(strength), f(coefficients(order)) {
        require(mode == "off" || mode == "constant" || mode == "nn", "Invalid filter mode");
        require(constant >= 0 && std::isfinite(constant), "Invalid filter strength");
        if (mode == "nn") {
            network = std::make_unique<Network>(path);
            require(network->order == order, "Filter checkpoint/PN order mismatch");
        }
    }
    Vec strength(const Angular &pn, const Spatial &fe, const Field &u,
                 const Vec &a, const Vec &s, const Field &q) const {
        if (mode != "nn") return Vec::Constant(u.rows(), mode == "off" ? 0 : constant);
        const Field x = features(pn, fe, u, a, s, q);
        Vec sigma(u.rows());
        parallel_for(int(u.rows()), [&](int i) { sigma[i] = (*network)(x.row(i).transpose()); });
        return sigma;
    }
    bool active() const { return mode != "off"; }
    // Same flow as apply(), without temporary fields. `out` may alias `u`. Each
    // node writes its damped row and adds m_i*(out-u) to `ledger` in one pass;
    // constant strength needs a single exponential per harmonic degree.
    void apply_into(Field &out, const Angular &pn, const Spatial &fe, const Field &u,
                    const Vec &a, const Vec &s, const Field &q, double tau, Field &ledger) {
        const double begin = seconds();
        const int nodes = int(u.rows()), order = pn.order;
        if (out.rows() != u.rows() || out.cols() != u.cols()) out.resize(u.rows(), u.cols());
        std::vector<double> table(order + 1, 1.);
        Vec sigma;
        if (network) {
            sigma = strength(pn,fe,u,a,s,q); ++evaluations;
            const Field mid = damp(u,sigma,f,tau/2);
            sigma = strength(pn,fe,mid,a,s,q); ++evaluations;
        } else {
            for (int l = 0; l <= order; ++l) table[l] = std::exp(-tau * constant * f[l*l]);
        }
        const bool varying = bool(network);
#pragma omp parallel
        {
            std::vector<double> local(order + 1);
#pragma omp for schedule(static)
            for (int i = 0; i < nodes; ++i) {
                const double *decay = table.data();
                if (varying) {
                    for (int l = 0; l <= order; ++l) local[l] = std::exp(-tau * sigma[i] * f[l*l]);
                    decay = local.data();
                }
                const double *from = &u.coeff(i, 0);
                double *to = &out.coeffRef(i, 0), *book = &ledger.coeffRef(i, 0);
                const double mass = fe.mass[i];
                to[0] = from[0];
                for (int l = 1; l <= order; ++l) {
                    const double d = decay[l];
                    for (int k = l*l; k < (l+1)*(l+1); ++k) {
                        const double before = from[k], after = before * d;
                        to[k] = after;
                        book[k] += mass * (after - before);
                    }
                }
            }
        }
        elapsed += seconds() - begin;
    }
    // Exponential midpoint for nonlinear sigma(u): evaluate again at a
    // half-interval predictor. Constant sigma gives the exact local flow.
    Field apply(const Angular &pn, const Spatial &fe, const Field &u,
                const Vec &a, const Vec &s, const Field &q, double tau) {
        if (mode == "off" || tau == 0) return u;
        const double begin = seconds();
        Vec sigma = strength(pn,fe,u,a,s,q); ++evaluations;
        if (network) {
            Field mid = damp(u,sigma,f,tau/2);
            sigma = strength(pn,fe,mid,a,s,q); ++evaluations;
        }
        Field v = damp(u,sigma,f,tau);
        elapsed += seconds()-begin;
        return v;
    }
};
} // namespace m1pn::nnfilter
