#include "time.hpp"
#include <iostream>
using namespace m1pn;
int main() {
    exact::environment();
    bool rejected = false;
    try { copied_velocity_exterior(1, Four::Zero()); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "Missing velocity at vacuum must fail");
    require(copied_velocity_exterior(0, Four::Zero()).norm() == 0, "Exact exterior vacuum");
    const Four exterior = copied_velocity_exterior(2, Four::Zero(), V3(.2, 0, 0));
    require((exterior-Four(2,.4,0,0)).norm() == 0, "Explicit vacuum velocity");
    require((copied_velocity_exterior(2, Four(1,.2,0,0))-exterior).norm() == 0,
            "Copied positive-density velocity");
    Angular pn(1);
    CanonicalC1 canonical(pn);
    require(maxabs(canonical.L - 1.5*Mat::Identity(1,1)) < 2e-13, "P1 half-range normalization");
    Boundary boundary(pn,canonical,-V3::UnitX());
    double half = 0; for (const auto &p : boundary.rule) half += p.weight;
    require(std::abs(half-2*pi) < 2e-13, "Physical dOmega hemisphere is 2pi");
    Inflow data; data.density=[](const V3&, double, const V3&){return 1/(4*pi);};
    const auto incoming = boundary.data(data,V3::Zero(),0);
    require(std::abs(incoming.moments[0]-.5) < 2e-14, "Half isotropic physical density");
    require(std::abs(incoming.kinetic[0]+.25) < 2e-14, "Isotropic incoming physical flux");
    data.density=[](const V3&, double, const V3&){return -1.;};
    rejected=false;try{boundary.data(data,V3::Zero(),0);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"Negative kinetic data must fail");
    for(int dim : {1,2,3}) {
        MeshSpec ms;ms.dimension=dim;ms.elements=2; Spatial fe(ms);
        Numerics n;n.r2=true;n.stabilization="biharmonic-corrected";n.gamma=.004;
        Inflow g;g.density=[](const V3&,double,const V3&){return 1/(4*pi);};
        int callbacks=0;
        g.exterior_state=[&](const V3&,double,const V3&,const Four&){++callbacks;return Four(1,0,0,0);};
        Transport tr(pn,fe,n,g);
        const Field vacuum=Field::Zero(fe.nodes,pn.q);
        Stage s=tr.evaluate(0,vacuum);
        require(callbacks>0,"Explicit exterior callback must be consumed");
        require(s.rate.col(0).maxCoeff()>0,"Prescribed inflow enters exact interior vacuum");
        Field u=vacuum+.1*fe.cfl_step*s.rate;
        tr.check_state(u);
        require(s.diagnostics.target_recovery < 1e-11,"Inflow target identity");
        g.exterior_state=[](const V3&,double,const V3&,const Four&inside){
            return copied_velocity_exterior(1,inside);
        };
        Transport missing_velocity(pn,fe,n,g);
        rejected=false;
        try { missing_velocity.evaluate(0,vacuum); }
        catch(const std::runtime_error&) {rejected=true;}
        require(rejected,"Transport must reject missing exterior velocity at vacuum");
    }
    for(int dim : {2,3}) {
        MeshSpec ms;ms.dimension=dim;ms.elements=3;Spatial fe(ms);Angular angular(3);
        V3 gradient(1,.5,dim==3?-.3:0.);
        Inflow affine;
        affine.density=[gradient](const V3 &x,double t,const V3 &w){
            return (1+.12*gradient.dot(x)-.12*t*gradient.dot(w))/(4*pi);
        };
        Numerics n;n.r2=true;n.stabilization="biharmonic-corrected";n.gamma=.004;
        n.bh_pair_mode="factorized";n.pn_boundary_quadrature="consistent";
        Field state=Field::Zero(fe.nodes,angular.q),expected=state;
        for(int i=0;i<fe.nodes;++i) {
            Vec a=Vec::Zero(angular.q),rate=a;
            const V3 x=fe.coordinates.row(i).transpose();
            set_low(a,Four(1+.12*gradient.dot(x),0,0,0));
            set_low(rate,Four(0,-.04*gradient[0],-.04*gradient[1],-.04*gradient[2]));
            state.row(i)=a.transpose();expected.row(i)=rate.transpose();
        }
        Transport tr(angular,fe,n,affine);
        Stage unlimited=tr.evaluate(0,state,true),limited=tr.evaluate(0,state);
        require(maxabs(unlimited.rate-expected)<2e-11,"Consistent boundary affine kinetic target");
        require(maxabs(limited.rate-expected)<2e-11,"Factorized affine transfer preserves exact target");
        require(limited.diagnostics.target_recovery<2e-11,"Affine all-Q correction identity");
    }
    std::cout << "Inflow API, normalization, exact-vacuum entry and affine integration passed\n";
}
