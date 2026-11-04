#include "ghost_local.hpp"
#include "stabilization.hpp"
#include "angular.hpp"
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdio>
using namespace m1pn;
void ck(bool v,const char*s){require(v,s);}
void polynomial(int dim,int cells){
    Spatial f({dim,cells,false,false,false,""});GhostLocalProjection op(f,.02,1.);
    const std::vector<std::array<int,3>> monomials={{0,0,0},{1,0,0},{2,0,0},{3,0,0},{4,0,0},{1,1,0},{2,1,0},{1,1,1},{0,2,0},{0,0,2}};
    for(auto powers:monomials){
        if((dim<2 && powers[1]) || (dim<3 && powers[2]))continue;
        Field u=Field::Ones(f.nodes,1);
        for(int i=0;i<f.nodes;++i)for(int d=0;d<dim;++d)u(i,0)*=std::pow(f.coordinates(i,d),powers[d]);
        const Field z=op.apply(u);double boundary=0,interior=0;
        for(int i=0;i<f.nodes;++i)(f.boundary[i]?boundary:interior)=std::max(f.boundary[i]?boundary:interior,std::abs(z(i,0)));
        std::cout<<"poly,"<<dim<<","<<cells<<","<<powers[0]<<powers[1]<<powers[2]<<","<<boundary<<","<<interior<<"\n";
        if(powers[0]+powers[1]+powers[2]<=2)ck(maxabs(z)<2e-7,"Polynomial degree <=2 annihilation");
    }
    Field p,u=Field::Random(f.nodes,64),z=op.apply(u,&p),g=Field::Zero(f.nodes,64);
    for(int e=0;e<int(f.pairs.size());++e){g.row(f.pairs[e].i)+=p.row(e);g.row(f.pairs[e].j)-=p.row(e);}
    ck(maxabs(g-z)<2e-10*std::max(1.,maxabs(z)),"64-component pair gather");
    ck(z.colwise().sum().cwiseAbs().maxCoeff()<2e-10*std::max(1.,maxabs(z)),"64-component global balance");
    if(cells==4){
        Field eye=Mat::Identity(f.nodes,f.nodes),A=-op.apply(eye);
        ck(maxabs(A-A.transpose())<2e-10*std::max(1.,maxabs(A)),"Symmetry");
        Eigen::GeneralizedSelfAdjointEigenSolver<Mat> eig(A,Mat(f.M));
        ck(eig.info()==Eigen::Success,"Generalized spectrum");
        double largest=eig.eigenvalues().maxCoeff();int nullity=(eig.eigenvalues().array().abs()<largest*1e-10).count();
        ck(eig.eigenvalues().minCoeff()>-largest*2e-10,"Semidefinite spectrum");
        ck(largest<=op.norm_bound/f.mass_equivalence*(1+1e-10),"Consistent mass norm bound");
        int nonzeros=(A.array().abs()>maxabs(A)*1e-13).count();
        std::cout<<"spectrum,"<<dim<<","<<cells<<",nullity="<<nullity<<",nnz="<<nonzeros<<",lambda_max="<<largest<<",bound="<<op.norm_bound/f.mass_equivalence<<",incidences="<<op.incidences()<<",routing_nnz="<<op.routing_nonzeros()<<"\n";
    }
    GhostLocalProjection zero(f,0,1.);ck(maxabs(zero.apply(u,&p))==0 && maxabs(p)==0,"Gamma zero");
    Field huge=Field::Constant(f.nodes,1,std::numeric_limits<double>::max()/4);
    ck(maxabs(zero.apply(huge,&p))==0 && maxabs(p)==0,"Gamma zero avoids overflowing intermediates");
    // Report warmed serial component-identical application, including pair routes.
    op.apply(u,&p);double start=seconds();for(int repeat=0;repeat<5;++repeat)op.apply(u,&p);
    std::cout<<"cost,"<<dim<<","<<cells<<",q=64,apply_pairs_seconds="<<(seconds()-start)/5<<"\n";
}
void guards(){
    for(MeshSpec s:std::vector<MeshSpec>{{1,3,false,false,false,""},{2,4,true,false,false,""},{2,4,false,true,false,""}}){
        Spatial f(s);bool rejected=false;try{GhostLocalProjection op(f,.01,1.);}catch(const std::runtime_error&){rejected=true;}ck(rejected,"Unsupported grid must fail explicitly");
    }
    const char*file="tmp/m1pn_ghost_warped.mesh";
    {mfem::Mesh mesh=mfem::Mesh::MakeCartesian2D(4,4,mfem::Element::QUADRILATERAL,true);for(int i=0;i<mesh.GetNV();++i){double*v=mesh.GetVertex(i);v[0]+=.1*v[1]*(1-v[1]);}std::ofstream out(file);mesh.Print(out);}
    Spatial f({2,4,false,false,false,file});bool rejected=false;try{GhostLocalProjection op(f,.01,1.);}catch(const std::runtime_error&){rejected=true;}std::remove(file);ck(rejected,"Warped Q1 rejected");
    // Rectangular Cartesian Q1 has independent directional spacings.
    {auto mesh=mfem::Mesh::MakeCartesian2D(4,6,mfem::Element::QUADRILATERAL,true,1.,2.);std::ofstream out(file);out<<std::setprecision(17);mesh.Print(out);}
    Spatial rectangle({2,4,false,false,false,file});GhostLocalProjection rect(rectangle,.02,1.);
    Field quadratic(rectangle.nodes,1),pairs;
    for(int i=0;i<rectangle.nodes;++i){double x=rectangle.coordinates(i,0),y=rectangle.coordinates(i,1);quadratic(i)=x*x+3*x*y+2*y*y;}
    ck(maxabs(rect.apply(quadratic,&pairs))<2e-7,"Rectangular Cartesian quadratic exactness");
    std::remove(file);
    Spatial one({1,4,false,false,false,""});rejected=false;
    try{GhostLocalProjection overflow(one,1e308,1.);}catch(const std::runtime_error&){rejected=true;}
    ck(rejected,"Nonfinite norm bound rejected");
}
void export_matrices(const std::string& dir,int dim,int cells){
    Spatial f({dim,cells,false,false,false,""});
    const Eigen::IOFormat csv(17,0,",","\n");
    auto save=[&](const std::string& name,const Mat& m){std::ofstream out(dir+"/"+name+".csv");ck(bool(out),"Export directory must exist");out<<m.format(csv)<<'\n';};
    save("coordinates",f.coordinates);save("mass",Mat(f.M));
    Field eye=Mat::Identity(f.nodes,f.nodes);
    GhostLocalProjection ghost(f,1.,1.);save("local_projected_ghost",ghost.apply(eye));
    SpatialStabilization kv(f,"gradient-defect",1.,1.);save("gradient_defect",kv.apply(eye));
    SpatialStabilization natural(f,"biharmonic-original",1.,1.);save("natural_biharmonic",natural.apply(eye));
    Angular pn(7);save("angular",pn.A[0]);
}
int main(int argc,char**argv){try{std::cout<<std::setprecision(17);
    if(argc==5 && std::string(argv[1])=="--export"){export_matrices(argv[2],std::stoi(argv[3]),std::stoi(argv[4]));return 0;}
    guards();for(int d=1;d<=3;++d)for(int n:{4,8})polynomial(d,n);std::cout<<"diagnostics passed\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
