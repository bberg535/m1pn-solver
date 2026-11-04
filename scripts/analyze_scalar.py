#!/usr/bin/env python3
"""Independent scalar P1 FE analysis; never a substitute for production PN tests."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from scipy.linalg import eigvals, eigvalsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def assemble(lengths, periodic=False):
    n=len(lengths)+(not periodic); M=np.zeros((n,n)); C=M.copy(); K=M.copy()
    for e,h in enumerate(lengths):
        ix=np.array([e,(e+1)%n]); idx=np.ix_(ix,ix)
        M[idx]+=h/6*np.array([[2,1],[1,2]])
        C[idx]+=np.array([[-.5,.5],[-.5,.5]])
        K[idx]+=np.array([[1,-1],[-1,1]])/h
    return M,C,K

def operators(lengths, periodic=False, c=1., scale=1.):
    M,C,K=assemble(lengths,periodic); m=M.sum(axis=1); Q=np.diag(m)-M
    D=np.zeros_like(M)
    for i in range(len(m)):
        for j in range(i+1,len(m)):
            if M[i,j]!=0:
                d=scale*abs(c)*max(abs(C[i,j]),abs(C[j,i]))
                D[i,j]=D[j,i]=d; D[i,i]-=d;D[j,j]-=d
    # Weak homogeneous inflow g=0 and upwind outflow for c>0.
    R=c*C.T.copy()
    if not periodic:R[-1,-1]-=c
    L=(R+D)/m[:,None]; Z=Q@L; G=(R+Z)/m[:,None]
    return M,m,Q,R,D,L,Z,G,K

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    findings={}; rows=[]
    n=32;h=1/n;theta=2*np.pi*np.arange(n)/n
    for scale in [.5,1.,2.]:
        M,m,Q,R,D,L,Z,G,K=operators(np.full(n,h),True,scale=scale)
        err=0
        for t in theta:
            v=np.exp(1j*t*np.arange(n));symbol=-1j/h*np.sin(t)*(1+(1-np.cos(t))/3)-scale/(3*h)*(1-np.cos(t))**2
            err=max(err,np.max(np.abs(G@v-symbol*v)))
        vals=eigvals(G); zs=eigvalsh((Z+Z.T)/2)
        def amp(dt):
            z=dt*vals
            return np.max(np.abs(1+z+z*z/2+z*z*z/6))
        lo=0.;hi=10*h
        for _ in range(70):
            mid=(lo+hi)/2
            if amp(mid)<=1+1e-13:lo=mid
            else:hi=mid
        findings[f'periodic_scale_{scale}']={'fourier_max_error':float(err),'max_real_generator_eigenvalue':float(vals.real.max()),'max_symmetric_Z_eigenvalue':float(zs.max()),'linear_RK3_dt_limit':lo,'dt_over_h':lo/h}
        assert err<2e-11
        for val in vals:rows.append(dict(grid='uniform_periodic',scale=scale,real=val.real,imag=val.imag))
    # Requested affine standard counterexample with exactly compatible inflow.
    M,m,Q,R,D,L,Z,G,K=operators(np.full(8,1/8))
    x=np.linspace(0,1,9);u=1+x;b=np.zeros(9);b[0]=u[0]
    w=(R@u+D@u+b)/m; z=Q@w; rate=(R@u+b+z)/m
    findings['affine']={'w0':float(w[0]),'w1':float(w[1]),'Z0_over_m0':float(z[0]/m[0]),'target0':float(rate[0]),'exact':-1.}
    assert abs(w[0])<1e-13 and abs(w[1]+1)<1e-13 and abs(z[0]/m[0]-1/3)<1e-13 and abs(rate[0]+2/3)<1e-13
    # Open one-element homogeneous-inflow energy counterexample.
    M,m,Q,R,D,L,Z,G,K=operators(np.array([1.]))
    u=np.array([1.,1.5]);work=float(u@Z@u)
    findings['open_energy']={'state':u.tolist(),'Z':(Z@u).tolist(),'quadratic_work':work,'expected':1/12,'max_symmetric_Z_eigenvalue':float(eigvalsh((Z+Z.T)/2)[-1])}
    assert abs(work-1/12)<1e-14
    # Periodic nonuniform four-element exact counterexample, domain rescaled to 1.
    lengths=np.array([9.,1.,9.,1.])/20
    M,m,Q,R,D,L,Z,G,K=operators(lengths,True)
    u=np.array([1.,3.,-1.,-3.]);work=float(u@Z@u)
    findings['nonuniform_periodic_energy']={'lengths':lengths.tolist(),'state':u.tolist(),'quadratic_work':work,'full_lumped_energy_derivative':float(u@(R+Z)@u),'conservation':float(np.sum(Z@u)),'max_symmetric_Z_eigenvalue':float(eigvalsh((Z+Z.T)/2)[-1]),'max_real_generator_eigenvalue':float(eigvals(G).real.max())}
    assert abs(work-.8)<1e-13 and abs(np.sum(Z@u))<1e-13
    for val in eigvals(G):rows.append(dict(grid='nonuniform_periodic',scale=1,real=val.real,imag=val.imag))
    patches=[]
    for cells in [8,16,32,64,128]:
        h=1/cells; x=np.linspace(0,1,cells+1)
        M,m,Q,R,D,L,Z,G,K=operators(np.full(cells,h))
        # Boundary-corrected stiffness annihilates affine functions before Gram product.
        Kc=K.copy();Kc[0,0]-=1/h;Kc[0,1]+=1/h;Kc[-1,-2]+=1/h;Kc[-1,-1]-=1/h
        B=-.004*h**3*Kc.T@np.diag(1/m)@Kc
        for name,u,exact in [('constant',np.ones(cells+1),np.zeros(cells+1)),('affine',1+x,-np.ones(cells+1)),('quadratic',1+x*x,-2*x),('smooth',1+.1*np.sin(2*np.pi*x),-.2*np.pi*np.cos(2*np.pi*x))]:
            b=np.zeros(cells+1);b[0]=u[0];rc=R@u+b;w=(rc+D@u)/m
            wc=w.copy();wc[0]=w[1];wc[-1]=w[-2]
            for variant,zz,rate in [('B',Q@w,(rc+Q@w)/m),('C_scalar_clamp',Q@wc,(rc+Q@wc)/m),('A_Gram',B@u,np.linalg.solve(M,rc+B@u))]:
                for region,ix in [('boundary',np.array([0,cells])),('interior',np.arange(1,cells))]:
                    patches.append(dict(cells=cells,h=h,data=name,variant=variant,region=region,Z_over_m_inf=np.max(np.abs((zz/m)[ix])),derivative_error_inf=np.max(np.abs((rate-exact)[ix])),conservation=abs(np.sum(zz))))
    def write(name,rr):
        with (a.output/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    write('spectra.csv',rows);write('patches.csv',patches)
    (a.output/'analysis.json').write_text(json.dumps(findings,indent=2)+'\n')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for v in ['A_Gram','B','C_scalar_clamp']:
        for j,data in enumerate(['affine','smooth']):
            rr=[r for r in patches if r['variant']==v and r['data']==data and r['region']=='boundary']
            ax[j].loglog([r['h'] for r in rr],np.maximum([r['derivative_error_inf'] for r in rr],1e-16),'o-',label=v)
            ax[j].set(title=data,xlabel='h',ylabel='boundary derivative error');ax[j].grid();ax[j].legend()
    fig.tight_layout();fig.savefig(a.output/'boundary_errors.png',dpi=160);fig.savefig(a.output/'boundary_errors.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4))
    for grid in sorted(set(r['grid'] for r in rows)):
        rr=[r for r in rows if r['grid']==grid and r['scale']==1]
        ax.scatter([r['real'] for r in rr],[r['imag'] for r in rr],label=grid,s=12)
    ax.set(xlabel='real eigenvalue',ylabel='imaginary eigenvalue');ax.legend();ax.grid();fig.tight_layout();fig.savefig(a.output/'spectra.png',dpi=160);plt.close(fig)
    t=np.linspace(0,np.pi,500);r=(2+np.cos(t))/3
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].plot(t,(1-np.cos(t))**2/3,label='B mass correction')
    ax[0].plot(t,4/12*(1-np.cos(t))**2/r,label='Gram beta=lambda*h^3/12, consistent mass')
    ax[0].plot(t,4*.004*(1-np.cos(t))**2/r,label='Gram gamma=.004, consistent mass')
    ax[0].set(xlabel='theta',ylabel='negative real symbol * h / lambda')
    ax[1].plot(t,np.sin(t)*(1+(1-np.cos(t))/3),label='B')
    ax[1].plot(t,np.sin(t)/r,label='consistent Galerkin')
    ax[1].set(xlabel='theta',ylabel='negative imaginary symbol * h / c')
    for aa in ax:aa.legend(fontsize=7);aa.grid()
    fig.tight_layout();fig.savefig(a.output/'matched_interior_symbols.png',dpi=160);plt.close(fig)
    import hashlib, platform, sys
    (a.output/'provenance.json').write_text(json.dumps(dict(command=sys.argv,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),host=platform.node(),python=platform.python_version(),numpy=np.__version__,notes='Independent scalar assembly; A_Gram is a proxy. .5 viscosity is an underdiffusive sensitivity case, not supported production.'),indent=2)+'\n')
    print(json.dumps(findings,indent=2))
if __name__=='__main__':main()
