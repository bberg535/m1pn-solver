#include "time.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef M1PN_MPI
#include <mpi.h>
#endif
using namespace m1pn;

namespace {
struct Args {
    std::string out, scenario, variant;
    int dimension=0, elements=0, order=0, steps=0, repeats=0;
    double dt=0;
};
Args parse(int argc,char **argv){
    require(argc==10,"usage: efficiency_benchmark output_dir scenario dimension elements pn_order dt steps repeats variant");
    Args a; a.out=argv[1]; a.scenario=argv[2]; a.dimension=std::stoi(argv[3]);
    a.elements=std::stoi(argv[4]); a.order=std::stoi(argv[5]); a.dt=std::stod(argv[6]);
    a.steps=std::stoi(argv[7]); a.repeats=std::stoi(argv[8]); a.variant=argv[9];
    require(a.dimension>=1&&a.dimension<=3&&a.elements>0&&a.order>=0&&a.dt>0&&
            a.steps>0&&a.repeats>0,"invalid benchmark dimensions or counts");
    require(a.scenario=="smooth"||a.scenario=="isotropic"||a.scenario=="crossing"||
            a.scenario=="pulse"||a.scenario=="affine","unknown benchmark scenario");
    return a;
}
void write_field(const std::filesystem::path &path,const Spatial &fe,const Field &u){
    std::ofstream f(path); require(bool(f),"cannot write field: "+path.string());
    f<<std::setprecision(17)<<"node,x,y,z";
    for(int k=0;k<u.cols();++k) f<<",u"<<k;
    f<<'\n';
    for(int i=0;i<u.rows();++i){f<<i;for(int d=0;d<3;++d)f<<','<<fe.coordinates(i,d);
        for(int k=0;k<u.cols();++k)f<<','<<u(i,k);f<<'\n';}
}
Inflow make_inflow(const Args &a){
    Inflow g;
    if(a.scenario=="isotropic"){
        g.density=[](const V3&,double,const V3&){return 1/(4*pi);};
        g.exterior_m1=[](const V3&,double){return Four(1,0,0,0);};
    } else if(a.scenario=="crossing"){
        const double z=a.dimension==2?0.:.5;
        g.density=[](const V3&,double,const V3&){return .01/(4*pi);};
        g.atoms=[z](const V3 &x,double t){
            return std::vector<Atom>{{V3::UnitX(),std::exp(-50*(x-t*V3::UnitX()-V3(.3,.5,z)).squaredNorm())},
                {V3::UnitY(),std::exp(-50*(x-t*V3::UnitY()-V3(.5,.3,z)).squaredNorm())}};
        };
    } else if(a.scenario=="pulse"){
        g.density=[](const V3 &x,double t,const V3&){return x[0]<1e-12&&t<.06?1/(4*pi):0.;};
        g.exterior_m1=[](const V3 &x,double t){return Four(x[0]<1e-12&&t<.06?1.:0.,0,0,0);};
    } else if(a.scenario=="affine"){
        const int d=a.dimension;
        g.density=[d](const V3&x,double t,const V3&w){V3 r(1,d>1?.5:0.,d>2?-.3:0.);
            return (1+.12*r.dot(x)-.12*t*r.dot(w))/(4*pi);};
    }
    return g;
}
Field initial(const Args &a,const Spatial &fe,const Angular &pn){
    Field u=Field::Zero(fe.nodes,pn.q);
    for(int i=0;i<fe.nodes;++i){const V3 x=fe.coordinates.row(i).transpose();Vec h=Vec::Zero(pn.q);
        if(a.scenario=="smooth") h[0]=(1+.2*std::sin(2*pi*x[0]))/std::sqrt(4*pi);
        else if(a.scenario=="isotropic") {
            h[0]=1/std::sqrt(4*pi);
        }
        else if(a.scenario=="crossing"){
            const double z=a.dimension==2?0.:.5; h[0]=.01/std::sqrt(4*pi);
            h+=std::exp(-50*(x-V3(.3,.5,z)).squaredNorm())*harmonics(pn.order,V3::UnitX());
            h+=std::exp(-50*(x-V3(.5,.3,z)).squaredNorm())*harmonics(pn.order,V3::UnitY());
        } else if(a.scenario=="affine"){
            V3 r(1,a.dimension>1?.5:0.,a.dimension>2?-.3:0.);
            set_low(h,Four(1+.12*r.dot(x),0,0,0));
        }
        u.row(i)=h.transpose();
    }
    return u;
}
struct Counts { long stages=0,active_pairs=0,active_faces=0,lazy_certified_pairs=0,lazy_fallback_pairs=0;
    void add(const Diagnostics&d){++stages;active_pairs+=d.active_pairs;active_faces+=d.active_faces;
 #ifndef M1PN_FROZEN
        lazy_certified_pairs+=d.lazy_certified_pairs;lazy_fallback_pairs+=d.lazy_fallback_pairs;
 #else
        (void)d;
 #endif
    }
};
void add_counts(Counts &c,const Step&s){for(const auto &d:s.stages)c.add(d);}
}

int main(int argc,char **argv){
#ifdef M1PN_MPI
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
#endif
    try{
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        Args a=parse(argc,argv);
        std::filesystem::create_directories(a.out);
        MeshSpec mesh; mesh.dimension=a.dimension; mesh.elements=a.elements;
        mesh.periodic=(a.scenario=="smooth");
        Numerics num; num.r2=true; num.gamma=.004; num.stabilization="biharmonic-corrected";
        int rk=3;
#ifdef M1PN_FROZEN
        require(a.variant=="reference","frozen executable accepts only variant=reference");
        require(a.scenario!="pulse"&&a.scenario!="affine","scenario is available only in the new core");
#else
        if(a.variant=="reference") throw std::runtime_error("reference variant belongs to efficiency_reference executable");
        if(a.variant=="eager"){}
        else if(a.variant=="lazy"){num.lazy=true;}
        else if(a.variant=="cached"){num.lazy=true;num.geometry_cache=true;}
        else if(a.variant=="cached-eager"){num.geometry_cache=true;}
        else if(a.variant=="defect"||a.variant=="pcg"||a.variant=="imex-direct"){
            rk=1;num.lazy=true;num.geometry_cache=true;
            num.implicit_solver=a.variant=="defect"?"defect":a.variant=="pcg"?"pcg":"direct";
            num.implicit_rtol=1e-12;num.implicit_atol=1e-14;
        } else throw std::runtime_error("unknown benchmark variant: "+a.variant);
        require(a.scenario!="pulse"||a.dimension>=1,"invalid pulse dimension");
#endif
        Runtime runtime; require(runtime.size==1,"efficiency benchmark requires one MPI rank");
        Angular pn(a.order); Spatial fe(mesh,1,{},true);
        Transport tr(pn,fe,num,make_inflow(a),runtime);
        TimeControl control(tr,mesh,rk);
        Field u0=initial(a,fe,pn);tr.check_state(u0);control.check(a.dt);
        write_field(std::filesystem::path(a.out)/"initial.csv",fe,u0);
        {std::ofstream m(std::filesystem::path(a.out)/"nodal_mass.csv");require(bool(m),"cannot write nodal masses");
            m<<"node,mass\n"<<std::setprecision(17);for(int i=0;i<fe.nodes;++i)m<<i<<','<<fe.mass[i]<<'\n';}
        {std::ofstream f(std::filesystem::path(a.out)/"meta.json");require(bool(f),"cannot write metadata");
            f<<std::setprecision(17)<<"{\n\"scenario\":\""<<a.scenario<<"\",\"dimension\":"<<a.dimension
             <<",\"elements\":"<<a.elements<<",\"pn_order\":"<<a.order<<",\"dt\":"<<a.dt
             <<",\"steps\":"<<a.steps<<",\"repeats\":"<<a.repeats<<",\"variant\":\""<<a.variant
             <<"\",\"warmups\":1,\"periodic\":"<<(mesh.periodic?"true":"false")<<"}\n";}
        // Warm kernels once; every measured repetition starts from the same state.
        {Field warm=u0;double t=0;for(int s=0;s<a.steps;++s){Step st=advance(tr,control,t,a.dt,warm);warm=std::move(st.state);t+=a.dt;}}
        std::ofstream repout(std::filesystem::path(a.out)/"repeat_timings.csv");
        std::ofstream stepout(std::filesystem::path(a.out)/"step_timings.csv");
        require(bool(repout)&&bool(stepout),"cannot open timing output");
        repout<<"repeat,solver_seconds,stages,active_pairs,active_faces,lazy_certified_pairs,lazy_fallback_pairs\n";
        stepout<<"repeat,step,solver_seconds\n";
        std::vector<Field> checkpoints;
        for(int repeat=0;repeat<a.repeats;++repeat){
            Field u=u0; double t=0,elapsed=0; Counts counts;
            for(int s=0;s<a.steps;++s){
                const double begin=seconds(); Step result=advance(tr,control,t,a.dt,u);
                const double sec=seconds()-begin; elapsed+=sec; stepout<<repeat<<','<<(s+1)<<','<<std::setprecision(17)<<sec<<'\n';
                add_counts(counts,result);u=std::move(result.state);t+=a.dt;
                if(repeat==0)checkpoints.push_back(u);
            }
            repout<<repeat<<','<<std::setprecision(17)<<elapsed<<','<<counts.stages<<','<<counts.active_pairs<<','
                  <<counts.active_faces<<','<<counts.lazy_certified_pairs<<','<<counts.lazy_fallback_pairs<<'\n';
        }
        for(int s=0;s<a.steps;++s){std::ostringstream name;name<<"checkpoint_step_"<<std::setw(6)<<std::setfill('0')<<(s+1)<<".csv";
            write_field(std::filesystem::path(a.out)/name.str(),fe,checkpoints[s]);}
        repout.close();stepout.close();
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';
#ifdef M1PN_MPI
        MPI_Finalize();
#endif
        return 1;}
#ifdef M1PN_MPI
    MPI_Finalize();
#endif
    return 0;
}
