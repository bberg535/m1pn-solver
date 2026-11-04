#include "lazy_m1.hpp"
#include "m1.hpp"
#include "transport.hpp"
#include <iostream>

using namespace m1pn;

static void compare_transport(int dim,bool periodic) {
    MeshSpec ms; ms.dimension=dim; ms.elements=periodic?3:2; ms.periodic=periodic;
    Spatial fe(ms); Angular pn(2); Field u=Field::Zero(fe.nodes,pn.q);
    for(int i=0;i<fe.nodes;++i) {
        Vec v=Vec::Zero(pn.q);
        set_low(v,Four(1.+.02*fe.coordinates(i,0),.06,.02,-.01));
        for(int k=4;k<pn.q;++k) v[k]=1e-4*std::sin((i+1.)*(k+1.));
        u.row(i)=v.transpose();
    }
    Numerics eager; eager.r2=true; eager.stabilization="gradient-defect"; eager.kv_gamma=.04;
    Numerics lazy=eager; lazy.lazy=true; lazy.geometry_cache=true;
    Inflow inflow; inflow.exterior_m1=[](const V3&,double){return Four(1.,.02,0.,0.);};
    Transport ref(pn,fe,eager,inflow), fast(pn,fe,lazy,inflow);
    const Stage a=ref.evaluate(0.,u), b=fast.evaluate(0.,u);
    if (maxabs(a.rate-b.rate)>2e-10 || maxabs(a.accepted_pairs-b.accepted_pairs)>2e-10 ||
        maxabs(a.boundary_gather-b.boundary_gather)>2e-10 ||
        b.diagnostics.target_recovery>1e-9)
        throw std::runtime_error("lazy/eager transport mismatch in dim="+std::to_string(dim));
}

static double actual_chi(const Four &u) {
    const Levermore m(u);
    if (m.vacuum) return 1./3;
    if (m.beam) return 1.;
    return (m.K/3*(m.gap+4*m.beta.squaredNorm()))/u[0];
}

int main() {
    for(int dim=1;dim<=3;++dim) { compare_transport(dim,true); compare_transport(dim,false); }
    for (int dim=1; dim<=3; ++dim) {
        for (int k=0; k<=100; ++k) {
            const double f=.99*k/100.;
            Four u(1.,f,dim>1?.11*f:0.,dim>2?-.07*f:0.);
            const auto interval=lazy_m1::chi_interval(u);
            const double chi=actual_chi(u);
            if (!(interval.lo <= chi && chi <= interval.hi)) {
                std::cerr << "chi enclosure failed, dimension=" << dim << " f=" << f << '\n';
                return 1;
            }
        }
    }
    const Four ui(1.,.2,.1,0.), uj(.8,-.1,.05,.03);
    const V3 cij(.1,-.04,.02), cji(-.06,.03,.01);
    const Four H(.03,.01,-.02,.005);
    const Levermore ci(ui), cj(uj);
    if ((lazy_m1::pressure_times(ui,actual_chi(ui),cij-cji)-ci.flux(cij-cji)).norm()>2e-14 ||
        (lazy_m1::pressure_times(uj,actual_chi(uj),cji-cij)-cj.flux(cji-cij)).norm()>2e-14) {
        std::cerr << "pressure enclosure formula disagrees with Levermore closure\n";
        return 1;
    }
    if (lazy_m1::certify_full_pair(ui,uj,cij,cji,1.,H)) {
        const Four wi=ui+(H+ci.flux(cij-cji))/2.;
        const Four wj=uj+(-H+cj.flux(cji-cij))/2.;
        if (!lazy_m1::strictly_interior(wi) || !lazy_m1::strictly_interior(wj)) {
            std::cerr << "certified pair disagrees with actual closure\n";
            return 1;
        }
    }
    const Four bad_i(1.,-.17683786951306132,-.14820309753429603,-.7128696305585814);
    const Four bad_j(1.,.1912844705300585,-.35850926795750715,.008538983566544813);
    const V3 bad_c(-.6762804800247967,.5828870269628014,-.1135228000707216);
    const Four bad_h(.2595298162454939,-.5206568335208787,.5794211125992416,-.607691657018112);
    if (lazy_m1::certify_full_pair(bad_i,bad_j,bad_c,-bad_c,1.,bad_h)) {
        std::cerr << "known nonrealizable full pair was incorrectly certified\n";
        return 1;
    }
    return 0;
}
