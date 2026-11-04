#include "ghost_local.hpp"
#include <map>
#include <set>
namespace m1pn {
GhostLocalProjection::GhostLocalProjection(const Spatial &fe, double gamma, double speed)
    : fe_(fe), disabled_(gamma==0) {
    require(gamma >= 0 && std::isfinite(gamma) && speed > 0 && std::isfinite(speed),
            "Invalid ghost projection coefficient");
    const int n=fe.nodes, dim=fe.dimension;
    require(dim>=1 && dim<=3 && !fe.faces.empty(),
            "Ghost projection supports open Cartesian Q1 meshes only");
    using Key=std::array<int,3>;
    std::array<std::vector<double>,3> axes;
    std::array<double,3> spacing{1,1,1};
    Key sizes{1,1,1};
    for(int d=0;d<dim;++d) {
        auto &axis=axes[d];
        for(int i=0;i<n;++i) axis.push_back(fe.coordinates(i,d));
        std::sort(axis.begin(),axis.end());
        const double length=axis.back()-axis.front();
        require(length>0 && std::isfinite(length),"Degenerate ghost grid axis");
        const double tol=1e-11*length;
        std::vector<double> unique;
        for(double x:axis) if(unique.empty() || x-unique.back()>tol) unique.push_back(x);
        axis=std::move(unique); sizes[d]=axis.size();
        require(sizes[d]>=5,"Quartic ghost projection requires at least five nodes per axis");
        spacing[d]=length/(sizes[d]-1);
        for(int k=0;k<sizes[d];++k)
            require(std::abs(axis[k]-(axis[0]+k*spacing[d]))<=tol,
                    "Ghost projection requires uniform Cartesian axes");
    }
    std::map<Key,int> ids;
    std::vector<Key> keys(n);
    long long total=1,cells=1;
    for(int d=0;d<dim;++d) {total*=sizes[d];cells*=sizes[d]-1;}
    require(total==n && cells==fe.mesh->GetNE(),"Incomplete or periodic Cartesian Q1 topology");
    for(int i=0;i<n;++i) {
        Key key{};
        for(int d=0;d<dim;++d) {
            key[d]=std::llround((fe.coordinates(i,d)-axes[d][0])/spacing[d]);
            require(key[d]>=0 && key[d]<sizes[d] &&
                    std::abs(fe.coordinates(i,d)-axes[d][key[d]])<=1e-11*(axes[d].back()-axes[d][0]),
                    "Non-Cartesian ghost node");
        }
        require(ids.emplace(key,i).second,"Duplicate Cartesian ghost node");
        keys[i]=key;
    }
    std::vector<std::set<int>> membership(n);
    std::set<Key> seen_cells;
    for(int e=0;e<fe.mesh->GetNE();++e) {
        const auto geometry=fe.mesh->GetElement(e)->GetGeometryType();
        require(geometry==(dim==1?mfem::Geometry::SEGMENT:dim==2?mfem::Geometry::SQUARE:mfem::Geometry::CUBE),
                "Ghost projection supports tensor Q1 elements only");
        mfem::Array<int> dofs; fe.space->GetElementDofs(e,dofs);
        require(dofs.Size()==(1<<dim),"Invalid Q1 ghost element");
        Key lo=keys[dofs[0]],hi=lo;
        std::set<int> actual(dofs.begin(),dofs.end());
        for(int i:dofs) for(int d=0;d<dim;++d) {
            lo[d]=std::min(lo[d],keys[i][d]);hi[d]=std::max(hi[d],keys[i][d]);
        }
        for(int d=0;d<dim;++d) require(hi[d]==lo[d]+1,"Nonlocal or distorted Cartesian Q1 element");
        for(int mask=0;mask<(1<<dim);++mask) {
            Key corner=lo;
            for(int d=0;d<dim;++d) corner[d]+=(mask>>d)&1;
            require(actual.count(ids.at(corner)),"Incomplete Cartesian Q1 element");
        }
        require(seen_cells.insert(lo).second,"Duplicate Cartesian Q1 cell");
        for(int i:dofs) membership[i].insert(dofs.begin(),dofs.end());
    }
    std::vector<Triplet> entries;
    for(int i=0;i<n;++i) for(int d=0;d<dim;++d) {
        Key key=keys[i]; const double invh2=1/(spacing[d]*spacing[d]);
        require(std::isfinite(10*invh2),"Unrepresentable ghost Laplace coefficient");
        if(key[d]==0 || key[d]==sizes[d]-1) {
            // Central second difference after quartic extrapolation.
            const int c[5]={3,-9,10,-5,1}; const bool left=key[d]==0;
            for(int k=0;k<5;++k) {
                key[d]=left?k:sizes[d]-1-k;
                entries.emplace_back(i,ids.at(key),c[k]*invh2);
            }
        } else {
            entries.emplace_back(i,i,-2*invh2);
            --key[d];entries.emplace_back(i,ids.at(key),invh2);
            key[d]+=2;entries.emplace_back(i,ids.at(key),invh2);
        }
    }
    laplace_.resize(n,n);laplace_.setFromTriplets(entries.begin(),entries.end());
    std::vector<int> count(n,0);
    for(const auto &s:membership) for(int i:s) ++count[i];
    Vec base(n);
    for(int i=0;i<n;++i) {
        require(count[i]>0,"Isolated ghost grid node");
        base[i]=speed*std::pow(fe.diameter[i],3)*fe.mass[i]/count[i];
        require(base[i]>0 && std::isfinite(base[i]) && std::isfinite(gamma*base[i]),
                "Unrepresentable ghost incidence weight");
    }
    for(const auto &members:membership) {
        Star s;s.nodes.assign(members.begin(),members.end());s.offset=incidences_;
        s.fractions.resize(s.nodes.size());s.weights.resize(s.nodes.size());
        double sum=0;for(int i:s.nodes)sum+=base[i];
        require(sum>0 && std::isfinite(sum),"Unrepresentable ghost star mass");
        for(int k=0;k<int(s.nodes.size());++k) {
            s.fractions[k]=base[s.nodes[k]]/sum;s.weights[k]=gamma*base[s.nodes[k]];
        }
        incidences_+=s.nodes.size();stars_.push_back(std::move(s));
    }
    std::map<std::pair<int,int>,int> edge_index;
    for(int e=0;e<int(fe.pairs.size());++e) edge_index[{fe.pairs[e].i,fe.pairs[e].j}]=e;
    // R[:,r] routes -D_r^T to incidence anchor p on original FE edges.
    // Axis-monotone paths span the FE star plus the quartic ghost reach.
    std::vector<Triplet> routes;
    Vec column_sum=Vec::Zero(n);double max_row_sum=0;
    for(int p=0;p<n;++p) {
        const auto &s=stars_[p];std::map<int,double> mean;
        for(int k=0;k<int(s.nodes.size());++k)
            for(Sparse::InnerIterator it(laplace_,s.nodes[k]);it;++it)
                mean[it.col()]+=s.fractions[k]*it.value();
        for(int k=0;k<int(s.nodes.size());++k) {
            std::map<int,double> row;
            for(auto [j,c]:mean)row[j]=-c;
            for(Sparse::InnerIterator it(laplace_,s.nodes[k]);it;++it)row[it.col()]+=it.value();
            double row_sum=0;
            for(auto [j,c]:row) {
                const double x=std::abs(c)*std::sqrt(s.weights[k]/fe.mass[j]);
                row_sum+=x;column_sum[j]+=x;
                if(j==p || c==0)continue;
                Key at=keys[j];int current=j;
                for(int d=0;d<dim;++d) while(at[d]!=keys[p][d]) {
                    at[d]+=at[d]<keys[p][d]?1:-1;int next=ids.at(at);
                    auto pair=std::minmax(current,next);
                    auto edge=edge_index.find(pair);
                    require(edge!=edge_index.end(),"Missing original FE edge for ghost route");
                    routes.emplace_back(edge->second,s.offset+k,current==pair.first?-c:c);
                    current=next;
                }
            }
            max_row_sum=std::max(max_row_sum,row_sum);
        }
    }
    norm_bound=max_row_sum*column_sum.maxCoeff();
    require(std::isfinite(norm_bound),"Unrepresentable ghost operator norm bound");
    routing_.resize(fe.pairs.size(),incidences_);
    routing_.setFromTriplets(routes.begin(),routes.end());
    routing_.prune(0.);routing_.makeCompressed();
}
Field GhostLocalProjection::apply(const Field &u,Field *pairs) const {
    require(u.rows()==fe_.nodes && u.allFinite(),"Invalid ghost projection field");
    if(disabled_) {
        if(pairs)*pairs=Field::Zero(fe_.pairs.size(),u.cols());
        return Field::Zero(fe_.nodes,u.cols());
    }
    const Field b=laplace_*u;
    Field adjoint=Field::Zero(fe_.nodes,u.cols());
    Field weighted;
    if(pairs)weighted=Field::Zero(incidences_,u.cols());
    for(const auto &s:stars_) {
        Field mean=Field::Zero(1,u.cols());
        for(int k=0;k<int(s.nodes.size());++k)mean.noalias()+=s.fractions[k]*b.row(s.nodes[k]);
        Field sum=Field::Zero(1,u.cols());
        Field local(s.nodes.size(),u.cols());
        for(int k=0;k<int(s.nodes.size());++k) {
            local.row(k)=s.weights[k]*(b.row(s.nodes[k])-mean);
            sum+=local.row(k);
            if(pairs)weighted.row(s.offset+k)=local.row(k);
        }
        // Apply weighted (I-Q)^T explicitly, including roundoff in sum.
        for(int k=0;k<int(s.nodes.size());++k)
            adjoint.row(s.nodes[k])+=local.row(k)-s.fractions[k]*sum;
    }
    if(pairs)*pairs=routing_*weighted;
    return -laplace_.transpose()*adjoint;
}
}
