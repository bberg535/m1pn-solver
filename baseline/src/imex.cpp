#include "imex.hpp"
namespace m1pn {
ImplicitMassSolver::ImplicitMassSolver(const Spatial &spatial, const Sparse &positive_A)
    : spatial_(spatial), positive_A_(positive_A) {
    require(positive_A_.rows() == spatial_.nodes && positive_A_.cols() == spatial_.nodes,
            "IMEX biharmonic matrix shape mismatch");
}
double ImplicitMassSolver::norm_lumped_inverse(const Field &x) const {
    double s=0;
    for(int i=0;i<x.rows();++i) s += x.row(i).squaredNorm()/spatial_.mass[i];
    return std::sqrt(s);
}
ImplicitResult ImplicitMassSolver::solve(const Field &b,double dt,const std::string &mode,
                                          double rtol,double atol,int max_iterations) {
    require(dt>0 && std::isfinite(dt),"IMEX dt must be finite and positive");
    require(rtol>=0 && atol>=0 && std::isfinite(rtol+atol),"Invalid IMEX tolerances");
    require(max_iterations>0,"IMEX maximum iterations must be positive");
    require(b.rows()==spatial_.nodes && b.cols()>0 && b.allFinite(),"IMEX RHS shape or values invalid");
    require(mode=="direct" || mode=="defect" || mode=="pcg","Unknown IMEX mass solver mode");
    if (cached_dt_ != dt) { cached_dt_=dt; direct_ready_=defect_ready_=false; }
    if (mode=="direct" && !direct_ready_) {
        Eigen::SparseMatrix<double> C(spatial_.M), A(positive_A_);
        direct_factor_.compute(C+dt*A);
        require(direct_factor_.info()==Eigen::Success,"IMEX direct factorization failed");
        direct_ready_=true; ++factorization_count_;
    }
    if (mode!="direct" && !defect_ready_) {
        Eigen::SparseMatrix<double> A(positive_A_), P(spatial_.nodes,spatial_.nodes);
        P.reserve(spatial_.nodes);
        for(int i=0;i<spatial_.nodes;++i) P.insert(i,i)=spatial_.mass[i];
        defect_factor_.compute(P+dt*A);
        require(defect_factor_.info()==Eigen::Success,"IMEX defect preconditioner factorization failed");
        defect_ready_=true; ++factorization_count_;
    }
    Eigen::SparseMatrix<double> A(positive_A_), M(spatial_.M);
    auto apply=[&](const Field &x){ return Field(M*x + dt*(A*x)); };
    ImplicitResult out; out.rate=Field::Zero(b.rows(),b.cols());
    const double normb=norm_lumped_inverse(b), target=atol+rtol*normb;
    if (normb == 0) {
        out.rate = Field::Zero(b.rows(), b.cols());
        out.predictor = out.rate;
        out.residual = out.relative_residual = 0;
        return out;
    }
    if(mode=="direct") {
        Mat cols=b, sol(b.rows(),b.cols());
#pragma omp parallel for schedule(static)
        for(int k=0;k<b.cols();++k) sol.col(k)=direct_factor_.solve(Vec(cols.col(k)));
        out.rate=sol; out.predictor=out.rate; out.solves=1;
    } else if(mode=="defect") {
        Field previous=Field::Zero(b.rows(),b.cols());
        Eigen::SparseMatrix<double> ML(spatial_.nodes,spatial_.nodes);
        ML.reserve(spatial_.nodes);
        for (int i=0;i<spatial_.nodes;++i) ML.insert(i,i)=spatial_.mass[i];
        Eigen::SparseMatrix<double> D=ML-M;
        bool converged=false;
        for(int k=0;k<max_iterations;++k) {
            Mat rhs=b + D*previous;
            Mat cols=rhs, sol(b.rows(),b.cols());
#pragma omp parallel for schedule(static)
            for(int j=0;j<b.cols();++j) sol.col(j)=defect_factor_.solve(Vec(cols.col(j)));
            Field current=sol;
            out.residual=norm_lumped_inverse(apply(current)-b); out.solves=k+1;
            if(out.residual<=target) { out.predictor=previous; out.rate=current; converged=true; break; }
            previous=std::move(current);
        }
        if(!converged) throw std::runtime_error("IMEX defect iteration failed residual tolerance after "+std::to_string(out.solves)+" solves");
    } else {
        // PCG for the consistent system, preconditioned by the cached lumped matrix.
        bool converged=false;
        Field x=Field::Zero(b.rows(),b.cols());
        Field r=b, z(b.rows(),b.cols()), p;
        Vec rz(b.cols());
        auto precondition=[&](const Field &v){ Mat c=v,s(v.rows(),v.cols());
#pragma omp parallel for schedule(static)
            for(int j=0;j<v.cols();++j) s.col(j)=defect_factor_.solve(Vec(c.col(j)));
            return Field(s); };
        z=precondition(r); p=z;
        for(int j=0;j<b.cols();++j) rz[j]=r.col(j).dot(z.col(j));
        for(int it=0;it<max_iterations;++it) {
            Field Ap=apply(p);
            for(int j=0;j<b.cols();++j) {
                double den=p.col(j).dot(Ap.col(j));
                if (den == 0 && r.col(j).squaredNorm() == 0) {
                    p.col(j).setZero();
                    continue;
                }
                require(den>0 && std::isfinite(den),"IMEX PCG lost positive definiteness");
                double alpha=rz[j]/den; x.col(j)+=alpha*p.col(j); r.col(j)-=alpha*Ap.col(j);
            }
            out.solves=it+1; out.residual=norm_lumped_inverse(apply(x)-b);
            if(out.residual<=target) { out.rate=x; out.predictor=x; converged=true; break; }
            Field zn=precondition(r);
            for(int j=0;j<b.cols();++j) {
                double next=r.col(j).dot(zn.col(j));
                if (rz[j] == 0 && next == 0) { p.col(j).setZero(); continue; }
                require(rz[j] > 0 && next >= 0 && std::isfinite(next), "IMEX PCG invalid preconditioned residual");
                double beta=next/rz[j]; p.col(j)=zn.col(j)+beta*p.col(j); rz[j]=next;
            }
            z=std::move(zn);
        }
        if(!converged) throw std::runtime_error("IMEX PCG failed residual tolerance after "+std::to_string(out.solves)+" iterations");
    }
    out.residual=norm_lumped_inverse(apply(out.rate)-b);
    out.relative_residual=normb>0?out.residual/normb:out.residual;
    require(out.rate.allFinite() && out.residual<=target,
            "IMEX final true residual exceeds requested tolerance");
    return out;
}
} // namespace m1pn
