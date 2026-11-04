#include "transport.hpp"
#include <iostream>
#include <Eigen/Eigenvalues>
using namespace m1pn;
namespace {
void expect(bool ok,const char *what){if(!ok) throw std::runtime_error(what);}
Field initial(const Spatial&fe,const Angular&pn,double scale=1){
 Field u=Field::Zero(fe.nodes,pn.q);
 for(int i=0;i<fe.nodes;++i){Vec v=Vec::Zero(pn.q);set_low(v,Four(scale*(1+.05*fe.coordinates(i,0)),scale*.08,scale*.03,0));for(int k=4;k<pn.q;++k)v[k]=scale*1e-3*std::sin((i+1.)*(k+2.));u.row(i)=v.transpose();}
 return u;
}
void run_case(int dim,bool tet){
 MeshSpec ms;ms.dimension=dim;ms.elements=dim==3?2:3;ms.tetrahedra=tet;
 Spatial fe(ms);Angular pn(3);Numerics opt;opt.r2=true;opt.stabilization="gradient-defect";opt.kv_gamma=.05;
 Inflow in;in.exterior_m1=[](const V3&,double){return Four(2,.1,0,0);};
 Transport tr(pn,fe,opt,in);Field u=initial(fe,pn);Stage s=tr.evaluate(0,u,false);
 expect(s.rate.allFinite(),"limited rate finite");expect(s.diagnostics.target_recovery<1e-9,"R2 residual identity");
 const double dt=.25*fe.cfl_step; for(int i=0;i<fe.nodes;++i) expect(exact::cone(project(u.row(i).transpose()+dt*s.rate.row(i).transpose())),"forward Euler state remains in cone");
 Stage un=tr.evaluate(0,u,true);expect(maxabs(un.rate-un.target)<1e-9,"unlimited target recovery");
 expect(un.diagnostics.high_rate_error<1e-10,"high modes retain target rate");
 Inflow vacuum_in;Transport vacuum_tr(pn,fe,opt,vacuum_in);Field z=Field::Zero(fe.nodes,pn.q);Stage vac=vacuum_tr.evaluate(0,z,false);expect(maxabs(vac.rate)<1e-12,"exact vacuum");
 Field us=initial(fe,pn,1e-12);Stage small=tr.evaluate(0,us,false);expect(small.rate.allFinite(),"scaled state finite");
 std::cout<<"r2 transport case dim="<<dim<<" tet="<<tet<<" nodes="<<fe.nodes<<" ok\n";
}
void incoming_atom(){
 Angular pn(3);CanonicalC1 c(pn);Boundary b(pn,c,-V3::UnitX());
 for(double scale:{1.,1e-24}) {
   Inflow g;g.atoms=[scale](const V3&,double){return std::vector<Atom>{{V3(1,.1,.05).normalized(),scale}};};
   const auto data=b.data(g,V3::Zero(),0);
   expect(exact::cone(data.moments),"rounded incoming atom has exact stored cone");
   expect(data.moments[0]>=scale && data.moments[0]<=scale*(1+1e-14),"exterior rounding is relative to atom mass");
 }
}
void characteristic(){Angular pn(3);CanonicalC1 c(pn);V3 n(1,2,3);n.normalize();Boundary b(pn,c,n);
 Vec d=Vec::LinSpaced(c.r,-.01,.02),u=Vec::LinSpaced(pn.q,-.4,.7),ue=b.exterior(u,d);
 Mat A=pn.normal(n);Eigen::SelfAdjointEigenSolver<Mat> eig(A);expect(eig.info()==Eigen::Success,"normal eigensystem");
 Vec expected=b.Rminus*c.incoming(b.Rplus.transpose()*u,d)+b.Rplus*(b.Rplus.transpose()*u); expected+=u-b.Rplus*(b.Rplus.transpose()*u)-b.Rminus*(b.Rminus.transpose()*u); expect((ue-expected).norm()<2e-11,"incoming/outgoing/zero characteristic reconstruction");
 // The outgoing and zero characteristic projections are copied exactly.
 for(int k=0;k<pn.q;++k){Vec v=eig.eigenvectors().col(k);double lam=eig.eigenvalues()[k];if(lam>1e-9||std::abs(lam)<=1e-9) expect(std::abs(v.dot(ue-u))<2e-9,"characteristic outgoing/zero retained");}
}
}
int main(){try{incoming_atom();characteristic();run_case(1,false);run_case(2,false);run_case(3,false);run_case(3,true);}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}return 0;}
