#include "stabilization.hpp"
#include <map>
#include <queue>
#include <set>
namespace m1pn {
namespace {
using EdgePath = std::vector<std::pair<int,double>>;
using RoutedEndpoint = std::tuple<int,int,EdgePath>;
std::vector<RoutedEndpoint> build_routes(const std::set<std::pair<int,int>> &endpoints,
                                         const Spatial &fe) {
    std::vector<std::vector<std::pair<int,int>>> adj(fe.nodes);
    for (int k=0;k<int(fe.pairs.size());++k) {
        const auto &p=fe.pairs[k]; adj[p.i].push_back({p.j,k}); adj[p.j].push_back({p.i,k});
    }
    for(auto &edges:adj) std::sort(edges.begin(),edges.end());
    std::map<int,std::vector<int>> targets;
    for(auto [i,j]:endpoints) targets[i].push_back(j);
    std::vector<RoutedEndpoint> result; result.reserve(endpoints.size());
    for(auto &[source,destinations]:targets) {
        std::vector<int> parent(fe.nodes,-1), parent_edge(fe.nodes,-1);
        std::set<int> pending(destinations.begin(),destinations.end());
        std::queue<int> q; parent[source]=source; q.push(source);
        while(!q.empty()&&!pending.empty()) {
            const int a=q.front(); q.pop();
            for(auto [b,k]:adj[a]) if(parent[b]<0) {
                parent[b]=a; parent_edge[b]=k; q.push(b); pending.erase(b);
            }
        }
        require(pending.empty(),"Stabilization pair endpoints are disconnected on FE graph");
        for(int target:destinations) {
            std::vector<int> nodes,edges;
            for(int b=target;b!=source;b=parent[b]) { nodes.push_back(b); edges.push_back(parent_edge[b]); }
            nodes.push_back(source); std::reverse(nodes.begin(),nodes.end()); std::reverse(edges.begin(),edges.end());
            EdgePath path;
            for(int k=0;k<int(edges.size());++k) {
                const auto &p=fe.pairs[edges[k]];
                path.push_back({edges[k],nodes[k]==p.i ? 1. : -1.});
            }
            result.emplace_back(source,target,std::move(path));
        }
    }
    return result;
}
}
SpatialStabilization::SpatialStabilization(const Spatial &fe, const std::string &kind,
                                         double gamma, double speed)
    : fe_(fe), kind_(kind), star_mass_(Vec::Zero(fe.nodes)),
      nu_(gamma * speed * fe.diameter),
      beta_(gamma * speed * fe.diameter.array().cube().matrix()) {
    require(gamma >= 0 && std::isfinite(gamma) && speed > 0, "Invalid stabilization coefficient");
    if (kind == "m1pn_ghost_local_projected_p4") {
        ghost_local_ = std::make_unique<GhostLocalProjection>(fe,gamma,speed);
        norm_bound = ghost_local_->norm_bound;
        return;
    }
    require(kind == "gradient-defect" || kind == "biharmonic-original" ||
            kind == "biharmonic-corrected", "Unknown stabilization");
    std::map<std::pair<int,int>,int> edge_index;
    for (int k=0;k<int(fe.pairs.size());++k)
        edge_index[{fe.pairs[k].i,fe.pairs[k].j}] = k;
    std::vector<std::vector<std::pair<int,int>>> stars(fe.nodes);
    for (int e=0;e<fe.mesh->GetNE();++e) {
        auto *tr=fe.mesh->GetElementTransformation(e);
        const auto *el=fe.space->GetFE(e);
        mfem::Array<int> dofs;
        fe.space->GetElementDofs(e,dofs);
        const int nd=dofs.Size();
        Element element;
        element.nodes.assign(dofs.begin(),dofs.end());
        element.omega=Vec::Zero(nd);
        element.L=Mat::Zero(nd,nd);
        const auto &rule=mfem::IntRules.Get(el->GetGeomType(),6);
        for (int k=0;k<rule.GetNPoints();++k) {
            const auto &ip=rule.IntPoint(k); tr->SetIntPoint(&ip);
            mfem::Vector shape(nd); mfem::DenseMatrix grad(nd,fe.dimension);
            el->CalcShape(ip,shape); el->CalcPhysDShape(*tr,grad);
            const double w=ip.weight*tr->Weight();
            for(int i=0;i<nd;++i) {
                element.omega[i]+=w*shape[i];
                for(int j=0;j<nd;++j)
                    for(int d=0;d<fe.dimension;++d)
                        element.L(i,j)+=w*grad(i,d)*grad(j,d);
            }
        }
        for(int a=0;a<nd;++a) {
            require(element.omega[a]>0,"Nonpositive physical incidence mass");
            const auto &ip=el->GetNodes().IntPoint(a); tr->SetIntPoint(&ip);
            mfem::DenseMatrix grad(nd,fe.dimension); el->CalcPhysDShape(*tr,grad);
            Mat g=Mat::Zero(3,nd);
            for(int j=0;j<nd;++j) for(int d=0;d<fe.dimension;++d) g(d,j)=grad(j,d);
            element.gradients.push_back(std::move(g));
            star_mass_[dofs[a]]+=element.omega[a];
            stars[dofs[a]].push_back({e,a});
        }
        for(int i=0;i<nd;++i) for(int j=i+1;j<nd;++j) {
            auto key=std::minmax(dofs[i],dofs[j]);
            element.edges.push_back({i,j,edge_index.at(key)});
        }
        elements_.push_back(std::move(element));
    }
    if (kind == "biharmonic-corrected" && !fe.faces.empty()) {
        // Same physical face quadrature as Spatial; no normal-gradient masking.
        for(int b=0;b<fe.mesh->GetNBE();++b) {
            auto *tr=fe.mesh->GetBdrFaceTransformations(b);
            require(tr != nullptr,"Missing physical boundary transformation");
            auto &element=elements_[tr->Elem1No]; const int nd=element.nodes.size();
            const auto *el=fe.space->GetFE(tr->Elem1No);
            const auto &rule=mfem::IntRules.Get(tr->GetGeometryType(),6);
            for(int k=0;k<rule.GetNPoints();++k) {
                const auto &ip=rule.IntPoint(k); tr->SetAllIntPoints(&ip);
                mfem::Vector shape(nd),normal(fe.dimension);
                mfem::DenseMatrix grad(nd,fe.dimension);
                el->CalcShape(tr->GetElement1IntPoint(),shape);
                el->CalcPhysDShape(*tr->Elem1,grad);
                if(fe.dimension==1) normal[0]=tr->GetElement1IntPoint().x<.5?-1:1;
                else mfem::CalcOrtho(tr->Jacobian(),normal);
                for(int i=0;i<nd;++i) for(int j=0;j<nd;++j)
                    for(int d=0;d<fe.dimension;++d)
                        element.L(i,j)-=ip.weight*shape[i]*normal[d]*grad(j,d);
            }
        }
    }
    if(kind!="gradient-defect") {
        std::vector<Triplet> entries;
        for(const auto &e:elements_) for(int i=0;i<int(e.nodes.size());++i)
            for(int j=0;j<int(e.nodes.size());++j)
                entries.emplace_back(e.nodes[i],e.nodes[j],e.L(i,j));
        L_.resize(fe.nodes,fe.nodes); L_.setFromTriplets(entries.begin(),entries.end());
        norm_bound=beta_.maxCoeff()*std::pow(fe.norm_bound(L_),2);
    } else {
        // ||sqrt(Wnu) D ML^-1/2||_2^2 <= ||.||_1 ||.||_inf.
        // Only a sparse, local coefficient row is needed for this bound.
        Vec colsum=Vec::Zero(fe.nodes); double rowsum_max=0;
        for(const auto &star:stars) for(auto [e,a]:star) {
            const auto &element=elements_[e]; const int node=element.nodes[a];
            for(int d=0;d<3;++d) {
                std::map<int,double> row;
                for(int j=0;j<int(element.nodes.size());++j)
                    row[element.nodes[j]]+=element.gradients[a](d,j);
                for(auto [f,b]:star) {
                    const auto &other=elements_[f];
                    for(int j=0;j<int(other.nodes.size());++j)
                        row[other.nodes[j]]-=other.omega[b]/star_mass_[node]*other.gradients[b](d,j);
                }
                double rowsum=0;
                for(auto [j,value]:row) {
                    const double x=std::abs(value)*std::sqrt(element.omega[a]*nu_[node]/fe.mass[j]);
                    rowsum+=x; colsum[j]+=x;
                }
                rowsum_max=std::max(rowsum_max,rowsum);
            }
        }
        norm_bound=rowsum_max*colsum.maxCoeff();
    }
}
void SpatialStabilization::ensure_positive_matrix() const {
    require(kind_=="biharmonic-original" || kind_=="biharmonic-corrected",
            "Positive biharmonic matrix requires a biharmonic stabilization");
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if(positive_matrix_ready_) return;
    Sparse weighted=L_;
    for(int i=0;i<weighted.outerSize();++i)
        for(Sparse::InnerIterator it(weighted,i);it;++it)
            it.valueRef()*=beta_[it.row()]/fe_.mass[it.row()];
    positive_matrix_=Sparse(L_.transpose()*weighted);
    positive_matrix_ready_=true;
}
void SpatialStabilization::ensure_routes(bool canonical) const {
    if(canonical) ensure_positive_matrix();
    require(kind_=="biharmonic-original" || kind_=="biharmonic-corrected",
            "BH pair routes require a biharmonic stabilization");
    std::lock_guard<std::mutex> lock(cache_mutex_);
    bool &ready=canonical?canonical_routes_ready_:factorized_routes_ready_;
    if(ready) return;
    std::set<std::pair<int,int>> endpoints;
    if(canonical) {
        for(int i=0;i<positive_matrix_.outerSize();++i)
            for(Sparse::InnerIterator it(positive_matrix_,i);it;++it)
                if(it.row()!=it.col()&&it.value()!=0.)
                    endpoints.insert(std::minmax(int(it.row()),int(it.col())));
    } else {
        for(int i=0;i<L_.outerSize();++i)
            for(Sparse::InnerIterator it(L_,i);it;++it)
                if(it.row()!=it.col()&&(it.value()!=0.||L_.coeff(it.col(),it.row())!=0.))
                    endpoints.insert(std::minmax(int(it.row()),int(it.col())));
    }
    auto built=build_routes(endpoints,fe_);
    auto &routes=canonical?canonical_routes_:factorized_routes_;
    routes.reserve(built.size());
    for(auto &item:built)
        routes.push_back({std::get<0>(item),std::get<1>(item),std::move(std::get<2>(item))});
    ready=true;
}
const Sparse &SpatialStabilization::positive_matrix() const {
    ensure_positive_matrix();
    return positive_matrix_;
}
Field SpatialStabilization::apply(const Field &u, Field *pairs, const std::string &pair_mode) const {
    if (pair_mode!="element" && pair_mode!="canonical" && pair_mode!="factorized")
        throw std::runtime_error("Unknown biharmonic pair mode: "+pair_mode);
    if (pair_mode!="element")
        require(!ghost_local_ && (kind_=="biharmonic-original" || kind_=="biharmonic-corrected"),
                "Canonical/factorized pair modes require biharmonic-original or biharmonic-corrected");
    if (ghost_local_) return ghost_local_->apply(u,pairs);
    require(u.rows()==fe_.nodes && u.allFinite(),"Invalid stabilization field");
    const int q=u.cols();
    Field z=Field::Zero(fe_.nodes,q);
    if(pairs) *pairs=Field::Zero(fe_.pairs.size(),q);
    if(pair_mode!="element") {
        require(pairs!=nullptr,"Non-element pair mode requires a pair ledger");
        ensure_routes(pair_mode=="canonical");
        Field weighted=L_*u;
        for(int i=0;i<fe_.nodes;++i) weighted.row(i)*=beta_[i]/fe_.mass[i];
        const auto &routes=pair_mode=="canonical" ? canonical_routes_ : factorized_routes_;
        for(const auto &route:routes) {
            Eigen::RowVectorXd flux;
            if(pair_mode=="canonical") flux=positive_matrix_.coeff(route.i,route.j)*(u.row(route.i)-u.row(route.j));
            else flux=L_.coeff(route.i,route.j)*weighted.row(route.i)-L_.coeff(route.j,route.i)*weighted.row(route.j);
            for(auto [edge,sign]:route.path) pairs->row(edge)+=sign*flux;
        }
        for(int e=0;e<int(fe_.pairs.size());++e) {
            const auto &p=fe_.pairs[e]; z.row(p.i)+=pairs->row(e); z.row(p.j)-=pairs->row(e);
        }
        return z;
    }
    Field recovered=Field::Zero(3*fe_.nodes,q),adjoint_sum=recovered;
    std::vector<std::vector<Field>> incidence;
    Field weighted;
    if(kind_=="gradient-defect") {
        for(const auto &e:elements_) {
            Field local(e.nodes.size(),q);
            for(int j=0;j<int(e.nodes.size());++j) local.row(j)=u.row(e.nodes[j]);
            std::vector<Field> b;
            for(int a=0;a<int(e.nodes.size());++a) {
                b.push_back(e.gradients[a]*local);
                recovered.middleRows(3*e.nodes[a],3)+=e.omega[a]*b.back();
            }
            incidence.push_back(std::move(b));
        }
        for(int a=0;a<fe_.nodes;++a) recovered.middleRows(3*a,3)/=star_mass_[a];
        for(int e=0;e<int(elements_.size());++e) for(int a=0;a<int(elements_[e].nodes.size());++a) {
            const auto &el=elements_[e]; const int node=el.nodes[a];
            incidence[e][a]=(el.omega[a]*nu_[node])*(incidence[e][a]-recovered.middleRows(3*node,3));
            adjoint_sum.middleRows(3*node,3)+=incidence[e][a];
        }
    } else {
        weighted=L_*u;
        for(int i=0;i<fe_.nodes;++i) weighted.row(i)*=beta_[i]/fe_.mass[i];
    }
    for(int e=0;e<int(elements_.size());++e) {
        const auto &el=elements_[e]; const int nd=el.nodes.size();
        Field r=Field::Zero(nd,q);
        if(kind_=="gradient-defect") {
            for(int a=0;a<nd;++a) {
                const int node=el.nodes[a];
                Field y=incidence[e][a]-el.omega[a]/star_mass_[node]*adjoint_sum.middleRows(3*node,3);
                r.noalias()-=el.gradients[a].transpose()*y;
            }
        } else {
            Field w(nd,q);
            for(int i=0;i<nd;++i) w.row(i)=weighted.row(el.nodes[i]);
            r.noalias()=-el.L.transpose()*w;
        }
        for(int i=0;i<nd;++i) z.row(el.nodes[i])+=r.row(i);
        if(pairs) for(auto edge:el.edges) {
            const int i=edge[0],j=edge[1],k=edge[2];
            const double sign=el.nodes[i]==fe_.pairs[k].i?1.:-1.;
            pairs->row(k)+=(sign/nd)*(r.row(i)-r.row(j));
        }
    }
    return z;
}
}
