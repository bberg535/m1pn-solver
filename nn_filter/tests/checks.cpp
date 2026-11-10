#include "filter.hpp"
#include <iostream>
#include <iomanip>
using namespace m1pn;
using namespace m1pn::nnfilter;

int main(int argc, char **argv) {
#ifdef M1PN_MPI
    MPI_Init(&argc, &argv);
#endif
    int status = 0, checks = 0;
    try {
        exact::environment();
        auto check = [&](bool ok, const std::string &name, double error = 0) {
            std::cout << std::setprecision(17) << "{\"check\":\"" << name
                      << "\",\"error\":" << error << ",\"passed\":" << (ok?"true":"false") << "}\n";
            require(ok,name); ++checks;
        };
        for (int n : {3,5,7,9}) {
            Network network(std::string(NN_FILTER_ROOT)+"/models/model_N"+std::to_string(n)+".txt");
            std::ifstream f(std::string(NN_FILTER_ROOT)+"/models/fixture_N"+std::to_string(n)+".txt");
            int rows, cols; f >> rows >> cols;
            require(bool(f) && rows==24 && cols==network.features,"Bad torch fixture");
            double error=0;
            for (int i=0; i<rows; ++i) {
                Vec x(cols); double expected;
                for(int j=0;j<cols;++j) f >> x[j];
                f >> expected; require(bool(f),"Truncated fixture");
                error=std::max(error,std::abs(network(x)-expected));
            }
            check(error<2e-12,"pytorch_network_P"+std::to_string(n),error);
        }
        Angular pn(3);
        const Vec f=coefficients(pn.order);
        check(f[0]==0 && f[15]==std::pow(3./4,4),"paper_coefficient_N_plus_one");
        Field u(11,pn.q);
        for(int i=0;i<u.rows();++i) {
            Vec row=Vec::Zero(pn.q);
            set_low(row,Four(1.+i*.01,.1,-.05,.02));
            for(int k=4;k<pn.q;++k) row[k]=.02*std::sin((i+1)*(k+1));
            u.row(i)=row.transpose();
        }
        Vec sigma=Vec::LinSpaced(u.rows(),0.,40.);
        const double tau=.07;
        Field v=damp(u,sigma,f,tau);
        check(maxabs(v.col(0)-u.col(0))==0,"density_preserved_bitwise");
        double err=0;
        for(int i=0;i<u.rows();++i) for(int k=1;k<pn.q;++k)
            err=std::max(err,std::abs(v(i,k)-u(i,k)*std::exp(-tau*sigma[i]*f[k])));
        check(err<2e-16,"all_degree_analytic_damping",err);
        check(maxabs(damp(v,sigma,f,tau)-damp(u,sigma,f,2*tau))<2e-16,"constant_flow_semigroup");
        check(maxabs(damp(u,Vec::Zero(u.rows()),f,tau)-u)==0,"zero_strength_identity");
        check(v.squaredNorm()<=u.squaredNorm(),"angular_L2_dissipation");
        bool cone=true;
        for(int i=0;i<u.rows();++i) cone=cone && exact::cone(project(v.row(i).transpose()));
        check(cone,"M1_cone_preserved_by_filter");
        const auto rot=Eigen::AngleAxisd(.71,V3(1.,2.,3.).normalized()).toRotationMatrix();
        const Mat R=pn.rotation(rot);
        const double rotation_error=maxabs(damp(Field(u*R.transpose()),sigma,f,tau)-v*R.transpose());
        check(rotation_error<2e-13,"angular_filter_rotation_commutes",rotation_error);
        Field raw(3,2); raw << 1,5,2,5,3,5;
        const Field x=normalize(raw);
        check(std::abs(x(0,0)+1)<2e-10 && x.col(1).isZero(),"sample_std_and_constant_feature");
        MeshSpec mesh; mesh.dimension=2; mesh.elements=4;
        Spatial fe(mesh,1,{},false);
        Field state=Field::Zero(fe.nodes,pn.q), source=state;
        for(int i=0;i<fe.nodes;++i) state(i,0)=1+.2*fe.coordinates(i,0)+.1*fe.coordinates(i,1);
        Vec zero=Vec::Zero(fe.nodes);
        const Field features_out=features(pn,fe,state,zero,zero,source);
        check(features_out.allFinite() && features_out.leftCols(4).isZero()
              && features_out.rightCols(2).isZero(),"vacuum_and_zero_source_features");
        if (argc == 2) {
            // Independent PyTorch preprocessing check consumes the actual FE
            // derivatives, before norms and normalization, in reduced 2D order.
            for(int i=0;i<fe.nodes;++i) for(int l=0;l<=pn.order;++l)
                for(int m=-l;m<=l;m+=2) {
                    double x=fe.coordinates(i,0),y=fe.coordinates(i,1);
                    state(i,index(l,m))=.1*(l+1)*std::cos((m+2)*x+.3)+.02*(l+2)*x*y;
                }
            Vec a(fe.nodes),s(fe.nodes);
            for(int i=0;i<fe.nodes;++i) {
                a[i]=.1+.4*fe.coordinates(i,0); s[i]=.2+.2*fe.coordinates(i,1);
                source(i,0)=.3*std::sin(fe.coordinates(i,0)+fe.coordinates(i,1));
            }
            const Field expected=features(pn,fe,state,a,s,source);
            std::array<Field,2> derivative;
            for(int d=0;d<2;++d) {
                derivative[d]=fe.C[d]*(state*pn.A[d].transpose());
                for(int i=0;i<fe.nodes;++i)derivative[d].row(i)/=fe.mass[i];
            }
            std::ofstream probe(argv[1]);
            probe<<std::setprecision(17)<<fe.nodes<<" "<<pn.order<<"\n";
            for(int i=0;i<fe.nodes;++i) {
                for(int block=0;block<3;++block) for(int l=0;l<=pn.order;++l)
                    for(int m=-l;m<=l;m+=2) {
                        const int k=index(l,m);
                        probe<<(block==0?(a[i]+s[i])*state(i,k):derivative[block-1](i,k))<<" ";
                    }
                probe<<s[i]*state(i,0)<<" "<<source(i,0)<<" ";
                for(int j=0;j<expected.cols();++j)probe<<expected(i,j)<<" ";
                probe<<"\n";
            }
        }
        Filter off(3,"off",0,""), constant(3,"constant",20,""),
               neural(3,"nn",0,std::string(NN_FILTER_ROOT)+"/models/model_N3.txt");
        check(maxabs(off.apply(pn,fe,state,zero,zero,source,.01)-state)==0,"off_filter_identity");
        Field decay=constant.apply(pn,fe,state,zero,zero,source,.01);
        check(maxabs(decay-damp(state,Vec::Constant(fe.nodes,20),f,.01))==0,"constant_filter_exact_flow");
        const Vec strengths=neural.strength(pn,fe,state,zero,zero,source);
        check(strengths.allFinite() && strengths.minCoeff()>=0,"original_checkpoint_nonnegative");
        bool rejected=false;
        try {Filter bad(5,"nn",0,std::string(NN_FILTER_ROOT)+"/models/model_N3.txt");}
        catch(const std::exception &){rejected=true;}
        check(rejected,"checkpoint_order_guard");
        rejected=false;
        try {damp(u,Vec::Constant(u.rows(),-1),f,.1);} catch(const std::exception &){rejected=true;}
        check(rejected,"negative_strength_guard");
        std::cout << "{\"status\":\"passed\",\"checks\":" << checks << "}\n";
    } catch(const std::exception &e) {std::cerr << e.what() << "\n"; status=2;}
#ifdef M1PN_MPI
    MPI_Finalize();
#endif
    return status;
}
