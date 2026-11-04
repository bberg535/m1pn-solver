#include "time.hpp"
#include <iostream>
using namespace m1pn;
int main(){
 try {
  MeshSpec mesh; mesh.dimension=1; mesh.elements=8; mesh.periodic=true;
  Spatial fe(mesh,1,{},false);
  Sparse A=fe.K; Field b(fe.nodes,4);
  for(int i=0;i<b.rows();++i) for(int k=0;k<b.cols();++k)
   b(i,k)=std::sin(.31*(i+1)*(k+1))+.2*std::cos(.17*(i+2)*(k+3));
  ImplicitMassSolver solver(fe,A); const double dt=.013;
  auto direct=solver.solve(b,dt,"direct",1e-12,1e-14,300);
  auto defect=solver.solve(b,dt,"defect",1e-9,1e-13,300);
  auto pcg=solver.solve(b,dt,"pcg",1e-9,1e-13,300);
  require(maxabs(direct.rate-defect.rate)<2e-8,"Defect iteration disagrees with direct IMEX solve");
  require(maxabs(direct.rate-pcg.rate)<2e-8,"PCG disagrees with direct IMEX solve");
  double bnorm2=0;
  for(int i=0;i<b.rows();++i) bnorm2+=b.row(i).squaredNorm()/fe.mass[i];
  require(defect.residual<=1e-13+1e-9*std::sqrt(bnorm2),"Defect residual exceeds requested ML inverse norm bound");
  Eigen::SparseMatrix<double> M(fe.M), ML(fe.nodes,fe.nodes); ML.reserve(fe.nodes);
  for(int i=0;i<fe.nodes;++i) ML.insert(i,i)=fe.mass[i];
  Eigen::SparseMatrix<double> P=ML+dt*Eigen::SparseMatrix<double>(A), D=ML-M;
  // The returned predictor is r^k and rate is r^(k+1): this is the identity
  // used by the finite defect MCL target construction.
  require(maxabs(P*defect.rate-(b+D*defect.predictor))<3e-8,
          "Finite-iterate predictor/rate target identity failed");
  bool failed=false;
  try { (void)solver.solve(b,dt,"defect",0,0,1); }
  catch(const std::runtime_error &) { failed=true; }
  require(failed,"Defect iteration must visibly fail when its iteration budget is insufficient");
  const auto before=solver.factorizations(); (void)solver.solve(b,dt,"direct",1e-12,1e-14,300);
  require(solver.factorizations()==before,"IMEX direct factorization was not cached at fixed dt");
  Field with_zero_column=b; with_zero_column.col(0).setZero();
  auto pcg_zero_column=solver.solve(with_zero_column,dt,"pcg",1e-9,1e-13,300);
  auto direct_zero_column=solver.solve(with_zero_column,dt,"direct",1e-12,1e-14,300);
  require(maxabs(pcg_zero_column.rate-direct_zero_column.rate)<2e-8,
          "PCG failed with a zero angular RHS column");
  Field zero=Field::Zero(b.rows(),b.cols());
  auto zero_pcg=solver.solve(zero,dt,"pcg",0,0,1);
  require(zero_pcg.solves==0 && maxabs(zero_pcg.rate)==0,
          "Zero RHS PCG should return without an iteration");
  for(int dim : {1,2,3}) for(bool periodic : {false,true}) {
    MeshSpec ms;ms.dimension=dim;ms.elements=3;ms.periodic=periodic;
    Spatial spatial(ms);Angular angular(3);
    Inflow inflow;inflow.density=[](const V3&,double,const V3&){return 1/(4*pi);};
    inflow.exterior_m1=[](const V3&,double){return Four(1,0,0,0);};
    Numerics n;n.r2=true;n.stabilization="biharmonic-corrected";n.gamma=.004;
    n.implicit_rtol=1e-12;n.implicit_atol=1e-14;n.implicit_max_iterations=1000;
    Field initial=Field::Zero(spatial.nodes,angular.q);
    for(int i=0;i<spatial.nodes;++i) {
      Vec a=Vec::Zero(angular.q);set_low(a,Four(1+.03*std::sin(spatial.coordinates(i,0)*3),.02,0,0));
      for(int k=4;k<angular.q;++k)a[k]=.0001*std::sin((i+1.)*(k+1.));
      initial.row(i)=a.transpose();
    }
    Transport direct_tr(angular,spatial,n,inflow);TimeControl ctrl(direct_tr,ms,1);
    const double step=std::min(1e-4,.25*ctrl.step_limit);
    auto base=advance(direct_tr,ctrl,0,step,initial);
    for(const std::string mode : {"defect","pcg"}) {
      n.implicit_solver=mode;Transport candidate(angular,spatial,n,inflow);
      auto got=advance(candidate,ctrl,0,step,initial);
      require(maxabs(got.state-base.state)<1e-9,"Full IMEX field differs from consistent reference");
      require(got.ledger_residual<1e-11,"Full IMEX Q ledger");
      require(got.stages[0].target_recovery<1e-11,"Full IMEX target identity");
    }
    n.implicit_solver="defect";n.implicit_rtol=1.;n.implicit_atol=0;
    Transport finite(angular,spatial,n,inflow);
    auto stage=finite.evaluate_r2(0,initial,true,Field(),step);
    require(stage.diagnostics.target_recovery<1e-11,"Finite defect full-Q target identity");
    require(maxabs(stage.rate-stage.target)<1e-10,"Unlimited finite defect target recovery");
  }
  std::cout<<"IMEX direct/defect/PCG and finite-iterate identity passed; defect_sweeps="<<defect.solves<<" pcg_iterations="<<pcg.solves<<"\n";
  return 0;
 } catch(const std::exception &e){ std::cerr<<"FAIL: "<<e.what()<<"\n"; return 1; }
}
