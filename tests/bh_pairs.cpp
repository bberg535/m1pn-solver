#include "stabilization.hpp"
#include <iostream>
using namespace m1pn;
static void check(bool ok,const char *what){if(!ok)throw std::runtime_error(what);}
static Field gather(const Spatial &fe,const Field &p){
    Field z=Field::Zero(fe.nodes,p.cols());
    for(int k=0;k<int(fe.pairs.size());++k){z.row(fe.pairs[k].i)+=p.row(k);z.row(fe.pairs[k].j)-=p.row(k);}
    return z;
}
static Field affine(const Spatial &fe){
    Field u(fe.nodes,1);
    for(int i=0;i<fe.nodes;++i)u(i,0)=.7+fe.coordinates(i,0)-.3*fe.coordinates(i,1)+.2*fe.coordinates(i,2);
    return u;
}
static void check_case(const MeshSpec &spec,const char *kind){
    Spatial fe(spec); SpatialStabilization op(fe,kind,.07,1.);
    Field u=Field::Random(fe.nodes,3), p0,p1,p2;
    Field old=op.apply(u,&p0), default_mode=op.apply(u);
    check((old-default_mode).norm()<1e-13,"default element application changed");
    check((old+op.positive_matrix()*u).norm()<3e-10,"positive_matrix does not represent -apply");
    Field canonical=op.apply(u,&p1,"canonical"), factorized=op.apply(u,&p2,"factorized");
    check((canonical-gather(fe,p1)).norm()<3e-10,"canonical ledger does not gather");
    check((factorized-gather(fe,p2)).norm()<3e-10,"factorized ledger does not gather");
    check((canonical+op.positive_matrix()*u).norm()<3e-10,"canonical decomposition changed operator");
    check((factorized+op.positive_matrix()*u).norm()<3e-10,"factorized decomposition changed operator");
    check(canonical.colwise().sum().norm()<3e-10 && factorized.colwise().sum().norm()<3e-10,
          "BH pair decomposition is not conservative");
    if(std::string(kind)=="biharmonic-corrected" && !spec.periodic){
        Field pa; check(op.apply(affine(fe),&pa,"factorized").norm()<3e-9,
                        "factorized corrected operator lost affine null mode");
    }
}
int main(){try{
    for(int d=1;d<=3;++d){
        MeshSpec open{d,2,false,false,false,""}, periodic{d,3,true,false,false,""};
        check_case(open,"biharmonic-original"); check_case(open,"biharmonic-corrected");
        check_case(periodic,"biharmonic-original"); check_case(periodic,"biharmonic-corrected");
    }
    Spatial fe(MeshSpec{1,2,false,false,false,""});
    SpatialStabilization non_bh(fe,"gradient-defect",.1,1.); Field u=Field::Ones(fe.nodes,1),p;
    bool failed=false; try{non_bh.apply(u,&p,"canonical");}catch(const std::exception&){failed=true;}
    check(failed,"unsupported decomposition did not fail explicitly");
    std::cout<<"BH pair decompositions passed\n"; return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
