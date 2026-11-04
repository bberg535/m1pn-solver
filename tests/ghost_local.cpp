#include "ghost_local.hpp"
#include <iostream>
#include <set>
#include <map>
using namespace m1pn;
namespace {
void ck(bool v,const char *s){if(!v)throw std::runtime_error(s);}
// Independent coordinate indexing and quartic ghost Laplacian.
Mat laplace(const Spatial&f){int d=f.dimension,naxis=5;std::array<int,3> n{naxis,naxis,naxis}; Mat B=Mat::Zero(f.nodes,f.nodes); double h=1./(naxis-1); std::map<std::array<int,3>,int> ids; for(int i=0;i<f.nodes;++i){std::array<int,3>x{};for(int a=0;a<d;++a)x[a]=int(std::llround(f.coordinates(i,a)/h));ids[x]=i;}
 for(int i=0;i<f.nodes;++i){std::array<int,3>x{};for(int a=0;a<d;++a)x[a]=int(std::llround(f.coordinates(i,a)/h)); for(int a=0;a<d;++a){auto add=[&](int j,double c){B(i,j)+=c/(h*h);}; if(x[a]==0){auto y=x;add(ids.at(y),3);y[a]=1;add(ids.at(y),-9);y[a]=2;add(ids.at(y),10);y[a]=3;add(ids.at(y),-5);y[a]=4;add(ids.at(y),1);
 } else if(x[a]==naxis-1){auto y=x;add(ids.at(y),3);y[a]=naxis-2;add(ids.at(y),-9);y[a]=naxis-3;add(ids.at(y),10);y[a]=naxis-4;add(ids.at(y),-5);y[a]=naxis-5;add(ids.at(y),1);}else{auto y=x;y[a]--;add(ids.at(y),1);add(i,-2);y[a]+=2;add(ids.at(y),1);}}} return B;}
Mat explicit_reference(const Spatial&f,const Mat&B,double gamma,double speed){std::vector<std::set<int>> stars(f.nodes);for(int e=0;e<f.mesh->GetNE();++e){mfem::Array<int> ds;f.space->GetElementDofs(e,ds);for(int i:ds)for(int j:ds)stars[i].insert(j);}std::vector<int> count(f.nodes);for(auto&s:stars)for(int i:s)++count[i];Mat H=Mat::Zero(f.nodes,f.nodes);for(int p=0;p<f.nodes;++p){Vec w=Vec::Zero(f.nodes);double sw=0;for(int i:stars[p]){w[i]=gamma*std::pow(f.diameter[i],3)*speed*f.mass[i]/count[i];sw+=w[i];}if(sw==0)continue;for(int i:stars[p])for(int j:stars[p])H(i,j)+=(i==j?w[i]:0)-w[i]*w[j]/sw;}return -B.transpose()*H*B;}
void run(int d){MeshSpec ms;ms.dimension=d;ms.elements=4;Spatial f(ms);GhostLocalProjection op(f,.02,1.);Mat B=laplace(f), Z=explicit_reference(f,B,.02,1.);Field u=Field::Random(f.nodes,64),pairs,z=op.apply(u,&pairs);ck(((u.array()*z.array()).sum())<=1e-9,"negative work");ck(z.colwise().sum().norm()<1e-9,"conservative all components");ck((z-Z*u).norm()<2e-10*std::max(1.,(Z*u).norm()),"explicit B transpose H B reference");Field g=Field::Zero(f.nodes,64);for(int e=0;e<(int)f.pairs.size();++e){g.row(f.pairs[e].i)+=pairs.row(e);g.row(f.pairs[e].j)-=pairs.row(e);}ck((g-z).norm()<2e-10*std::max(1.,z.norm()),"FE pair gather");Field poly(f.nodes,1);for(int i=0;i<f.nodes;++i){double x=f.coordinates(i,0),y=f.coordinates(i,1),zz=f.coordinates(i,2);poly(i)=1+2*x-3*y+zz+x*x+y*y+zz*zz+x*y+x*zz+y*zz;}ck(op.apply(poly).norm()<2e-7,"quadratic polynomial annihilation");std::cout<<"ghost local d="<<d<<" nodes="<<f.nodes<<" norm="<<z.norm()<<"\n";}
}
int main(){try{run(1);run(2);run(3);std::cout<<"ghost local tests passed\n";}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}return 0;}
