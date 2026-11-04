#include "time.hpp"
#include <Eigen/Cholesky>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace m1pn;
namespace {
int checks=0;
void check(bool ok,const std::string& name,double value=0){
 std::cout<<std::setprecision(17)<<"{\"check\":\""<<name<<"\",\"value\":"<<value<<",\"passed\":"<<(ok?"true":"false")<<"}\n";
 require(ok,name); ++checks;
}
void close(double e,double tol,const std::string& name){check(std::isfinite(e)&&std::abs(e)<=tol,name,e);}
Field rhs_field(const Spatial& f,int q){Field x(f.nodes,q);for(int i=0;i<f.nodes;++i)for(int k=0;k<q;++k)x(i,k)=std::sin(.37*(i+1)*(k+2))+.2*std::cos(.19*(i+3)*(k+1));return x;}
Field state(const Spatial& f,int q){Field x=Field::Zero(f.nodes,q);for(int i=0;i<f.nodes;++i){Vec v=Vec::Zero(q);set_low(v,Four(1+.08*f.coordinates(i,0),.08,-.03,.02));for(int k=4;k<q;++k)v[k]=.001*std::sin((i+1.)*(k+2.));x.row(i)=v.transpose();}return x;}
void oracle(){
 MeshSpec ms;ms.dimension=2;ms.elements=3; Spatial fe(ms,1,{},false); Angular pn(2); Field rhs=rhs_field(fe,pn.q);
 Mat dense(fe.M); Eigen::LDLT<Mat> ldlt(dense); require(ldlt.info()==Eigen::Success,"dense mass oracle factorization");
 for(int corrections:{1,2,4}){
  Numerics n;n.r2=true;n.mass_mode="lumped_afc";n.mass_corrections=corrections; Transport tr(pn,fe,n);
  double residual=0;Field pred;Field got=tr.mass_target(rhs,residual,&pred);
  Field p=fe.mass.cwiseInverse().asDiagonal()*rhs;
  for(int k=1;k<corrections;++k)p=fe.mass.cwiseInverse().asDiagonal()*(rhs+fe.mass.asDiagonal()*p-fe.M*p);
  Field expected=fe.mass.cwiseInverse().asDiagonal()*(rhs+fe.mass.asDiagonal()*p-fe.M*p);
  close(maxabs(got-expected),2e-13,"afc_neumann_recurrence_r"+std::to_string(corrections));
  close(maxabs(pred-p),2e-13,"afc_pair_predictor_r"+std::to_string(corrections));
  check(std::isfinite(residual),"afc_residual_finite_r"+std::to_string(corrections),residual);
  close(maxabs(fe.M*got-rhs)-residual,2e-13,"afc_residual_reports_true_defect_r"+std::to_string(corrections));
  check(!fe.factorized&&fe.mass_solve_calls==0,"afc_avoids_consistent_factorization");
 }
 Numerics n;n.r2=true;n.mass_mode="lumped";Transport tr(pn,fe,n);double r=0;Field predictor;Field lumped=tr.mass_target(rhs,r,&predictor);
 close(maxabs(lumped-fe.mass.cwiseInverse().asDiagonal()*rhs),1e-14,"lumped_inverse_diagonal");
 check(maxabs(predictor)==0,"lumped_zero_predictor");
 // Corrections must improve the M-norm inverse approximation for a smooth load.
 Field smooth=Field::Zero(fe.nodes,pn.q);for(int i=0;i<fe.nodes;++i){smooth(i,0)=1+.3*fe.coordinates(i,0);smooth(i,1)=std::sin(fe.coordinates(i,0));}
 Field exact=ldlt.solve(Mat(smooth));
 auto mnorm=[&](const Field& e){return std::sqrt((e.array()*(fe.M*e).array()).sum());};
 double e0=mnorm(fe.mass.cwiseInverse().asDiagonal()*smooth-exact);
 Numerics a;a.r2=true;a.mass_mode="lumped_afc";a.mass_corrections=1;Transport t1(pn,fe,a);Field q1=t1.mass_target(smooth,r);
 a.mass_corrections=3;Transport t3(pn,fe,a);Field q3=t3.mass_target(smooth,r);
 double e1=mnorm(q1-exact),e3=mnorm(q3-exact);
 check(e1<e0&&e3<e1,"smooth_inverse_corrections_converge",e3/e0);
 for(const std::string kind:{"gradient-defect","biharmonic-original","biharmonic-corrected"}) {
  SpatialStabilization op(fe,kind,.03,pn.speed);Field x=rhs_field(fe,pn.q),z=op.apply(x);
  check((x.array()*z.array()).sum()<=2e-11,kind+"_M_gram_nonpositive",(x.array()*z.array()).sum());
 }
}
void nonuniform_metric_counterexample(){
 const std::string folder=std::string(PN_AFC_ROOT)+"/tmp/test_tmp";
 std::filesystem::create_directories(folder);
 const std::string path=folder+"/nonuniform_counterexample.mesh";
 {auto mesh=mfem::Mesh::MakeCartesian1D(2);const double x[3]={0,1,3};
  for(int i=0;i<3;++i)mesh.GetVertex(i)[0]=x[i];std::ofstream out(path);require(bool(out),"open persistent nonuniform mesh");mesh.Print(out);}
 MeshSpec ms{1,2,false,false,false,path};Spatial fe(ms,1,{},false);
 Field u(3,1);const double v[3]={1,-1.5,.5};
 for(int i=0;i<3;++i)u(i,0)=fe.coordinates(i,0)+v[i]/10.;
 SpatialStabilization grad(fe,"gradient-defect",1,1),corrected(fe,"biharmonic-corrected",1,1);
 Field z=grad.apply(u),zc=corrected.apply(u);
 close((u.array()*z.array()).sum()+49./600.,2e-13,"counterexample_gradient_exact_work");
 Angular pn(1);Numerics n;n.r2=true;n.mass_mode="lumped_afc";n.mass_corrections=1;
 Transport tr(pn,fe,n);double res=0;Field rate=tr.mass_target(z,res);
 Mat M(fe.M),D=fe.mass.asDiagonal(),Di=fe.mass.cwiseInverse().asDiagonal();
 Field Mu=M*u;
 const double wrong_work=(Mu.array()*rate.array()).sum();
 close(wrong_work-749./10800.,3e-13,"counterexample_afc_consistent_metric_work");
 std::cout<<"PASS expected counterexample: AFC1 consistent-metric work="<<wrong_work<<" > 0\n";
 Field rate_corrected=tr.mass_target(zc,res);
 const double corrected_work=(Mu.array()*rate_corrected.array()).sum();
 close(corrected_work-749./1350.,3e-13,"counterexample_corrected_biharmonic_work");
 Mat P=2*Di-Di*M*Di;Eigen::LDLT<Mat> ldlt(P);
 require(ldlt.info()==Eigen::Success,"counterexample test-only modified metric factorization");
 Field recovered=ldlt.solve(rate);
 close(maxabs(recovered-z),2e-13,"counterexample_modified_metric_recovery");
 close((u.array()*recovered.array()).sum()+49./600.,2e-13,"counterexample_modified_metric_dissipation");
}
void edge_reconstruction(){
 MeshSpec ms;ms.dimension=2;ms.elements=3;Spatial fe(ms,1,{},false);Angular pn(2);
 Numerics n;n.r2=true;n.mass_mode="lumped_afc";n.mass_corrections=2;n.stabilization="gradient-defect";n.kv_gamma=.03;
 Inflow in;in.exterior_m1=[](const V3&,double){return Four(2,.1,-.04,.02);};Transport tr(pn,fe,n,in);
 Field u=state(fe,pn.q),stab_pairs;Field rhs=tr.stabilization_operator->apply(u,&stab_pairs);
 for(int d=0;d<3;++d)rhs.noalias()+=fe.C[d].transpose()*(u*pn.A[d].transpose());
 for(const auto &face:fe.faces)for(const auto &pt:face.points){
  const auto &b=*tr.boundaries[pt.normal_id];auto data=b.data(in,pt.x,0);
  for(size_t j=0;j<face.nodes.size();++j){const int i=face.nodes[j];Vec ui=u.row(i).transpose();
   Vec ue=b.exterior(ui,data.d);
   Vec flux=.5*(pn.apply(pt.normal,ui)+pn.apply(pt.normal,ue))-.5*(ue-ui);
   rhs.row(i)-=pt.weight*pt.shape[j]*flux.transpose();
  }
 }
 double residual=0;Field predictor;tr.mass_target(rhs,residual,&predictor);
 Stage s=tr.evaluate(0,u,true);
 Field expected=Field::Zero(fe.pairs.size(),pn.q);
 for(size_t e=0;e<fe.pairs.size();++e){const auto &p=fe.pairs[e];V3 cij,cji;
  for(int d=0;d<3;++d){cij[d]=fe.C[d].coeff(p.i,p.j);cji[d]=fe.C[d].coeff(p.j,p.i);}
  Vec q=pn.apply(cji,u.row(p.j).transpose())-pn.apply(cij,u.row(p.i).transpose())+
        stab_pairs.row(e).transpose()+p.mass*(predictor.row(p.i)-predictor.row(p.j)).transpose();
  expected.row(e)=q.transpose();
 }
 close(maxabs(s.accepted_pairs-expected),3e-11,"independent_full_q_edge_flux");
 Field pair_only=tr.gather_pairs(expected),gather_from_edges=tr.gather_pairs(s.accepted_pairs);
 close(maxabs(pair_only-gather_from_edges),3e-13,"all_q_antisymmetric_edge_ledger");
 close(pair_only.colwise().sum().cwiseAbs().maxCoeff(),3e-13,"all_q_pair_conservation");
 close(maxabs(s.gather+ s.boundary_gather - gather_from_edges - s.source),3e-11,"all_q_pair_boundary_source_ledger");
 check(maxabs(s.gather-pair_only)>0,"all_q_boundary_ledger_is_present");
}
void transport_case(int dim,bool periodic,bool tet){
 MeshSpec ms;ms.dimension=dim;ms.elements=3;ms.periodic=periodic;ms.tetrahedra=tet;Spatial fe(ms,1,{},false);Angular pn(2);
 Inflow in;in.exterior_m1=[](const V3&,double){return Four(2,.1,0,0);};
 Numerics n;n.r2=true;n.mass_mode="lumped_afc";n.mass_corrections=2;n.stabilization="gradient-defect";n.kv_gamma=.03;
 Transport tr(pn,fe,n,in);Field u=state(fe,pn.q);Stage un=tr.evaluate(0,u,true);
 check(maxabs(un.rate-un.target)<2e-10,"massafc_unlimited_target_recovery");
 check(un.diagnostics.high_rate_error<2e-10,"massafc_unlimited_high_modes");
 Field z=Field::Zero(fe.nodes,pn.q);Inflow vacuum;Transport tv(pn,fe,n,vacuum);Stage vz=tv.evaluate(0,z,false);
 check(maxabs(vz.rate)==0,"massafc_exact_vacuum");
 const double dt=.2*fe.cfl_step;Field k1=tr.evaluate(0,u).rate;Field u1=u+dt*k1;
 auto cone_field=[&](const Field& v){for(int i=0;i<fe.nodes;++i)if(!exact::cone(project(v.row(i).transpose())))return false;return true;};
 check(cone_field(u1),"massafc_fe_cone_dim"+std::to_string(dim));
 Field k2=tr.evaluate(dt,u1).rate;Field u2=.75*u+.25*(u1+dt*k2);
 check(cone_field(u2),"massafc_ssprk2_cone_dim"+std::to_string(dim));
 Field k3=tr.evaluate(.5*dt,u2).rate;Field u3=(u+2*(u2+dt*k3))/3.;
 check(cone_field(u3),"massafc_ssprk3_cone_dim"+std::to_string(dim));
 Field full= u+fe.cfl_step*tr.evaluate(0,u).rate;
 check(cone_field(full),"massafc_full_cfl_fe_cone_dim"+std::to_string(dim));
 check(!fe.factorized&&fe.mass_solve_calls==0,"massafc_no_mass_solve");
 std::cout<<"massafc case dim="<<dim<<" periodic="<<periodic<<" tet="<<tet<<" nodes="<<fe.nodes<<"\n";
}
}
int main(){try{oracle();nonuniform_metric_counterexample();edge_reconstruction();transport_case(1,true,false);transport_case(2,false,false);transport_case(3,false,true);std::cout<<"mass_afc checks="<<checks<<" passed\n";}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}return 0;}
