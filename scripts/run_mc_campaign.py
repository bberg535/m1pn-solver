#!/usr/bin/env python3
"""Matched native CLI runs. Preserve failures, full commands, and raw outputs."""
import argparse, csv, hashlib, json, os, platform, subprocess, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
VARIANTS={'A':['--stabilization','biharmonic-corrected','--biharmonic-gamma','.004'],
          'B':['--stabilization','m1pn_mass_correction'],
          'Aex':['--stabilization','m1pn_ghost_local_projected_p4','--ghost-gamma','0.00032911848351755427'],
          'C':['--stabilization','m1pn_mass_correction_boundary_consistent']}

def metric(folder,t,scenario):
    data=np.loadtxt(folder/'final.csv',delimiter=',',skiprows=1,ndmin=2);x=data[:,1:4];u=data[:,4:]
    w=np.loadtxt(folder/'nodal_mass.csv',delimiter=',',skiprows=1,usecols=1,ndmin=1)
    rho=u[:,0]*np.sqrt(4*np.pi);J=u[:,[3,1,2]]*np.sqrt(4*np.pi/3)
    result=dict(min_density=float(rho.min()),negative_nodes=int((rho<0).sum()),cone_margin_min=float(np.min(rho-np.linalg.norm(J,axis=1))),high_moment_L2=float(np.sqrt(np.sum(w[:,None]*u[:,4:]**2))),high_moment_Linf=float(np.max(abs(u[:,4:]))) if u.shape[1]>4 else 0)
    if scenario=='crossing':
        exact=.01+np.exp(-50*np.sum((x-np.array([.3+t,.5,0.]))**2,axis=1))+np.exp(-50*np.sum((x-np.array([.5,.3+t,0.]))**2,axis=1))
        e=rho-exact;den=float(w@abs(exact))
        result.update(reference='analytic kinetic density; includes PN model error',rho_L1=float(w@abs(e)),rho_L2=float(np.sqrt(w@(e*e))),rho_Linf=float(np.max(abs(e))),relative_L1=float(w@abs(e))/den,relative_L1_denominator=den)
        for region,mask in [('boundary',np.any((x[:,:2]<1e-12)|(x[:,:2]>1-1e-12),axis=1)),('boundary_zone',np.any((x[:,:2]<.1)|(x[:,:2]>.9),axis=1)),('crossing_zone',np.linalg.norm(x[:,:2]-.5,axis=1)<.2)]:
            result[region+'_Linf']=float(np.max(abs(e[mask]))) if mask.any() else None
    return result

def run(out,name,variant,dim,n,pn,t,dt,scenario='crossing',repeat=0,binary='m1pn',extra=None,timeout=1800):
    folder=out/name
    flags=['--dimension',str(dim),'--elements',str(n),'--pn-order',str(pn),'--case',scenario,'--time',str(t),'--dt',str(dt),'--threads','1','--geometry-cache','on','--mcl-mode','eager','--checkpoint-stride','400','--output',str(folder)]
    if scenario=='crossing':flags+=['--inflow','crossing-transport']
    cmd=['taskset','-c',str(min(os.sched_getaffinity(0))),str(ROOT/'tmp/build'/binary)]+flags+VARIANTS[variant]+(extra or [])
    print('RUN',name,flush=True);start=time.perf_counter()
    try:
        p=subprocess.run(cmd,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
        code=p.returncode;stdout=p.stdout;stderr=p.stderr
    except subprocess.TimeoutExpired as e:
        code=124;stdout=str(e.stdout or '');stderr=str(e.stderr or '')+'\nTIMEOUT'
    elapsed=time.perf_counter()-start;folder.mkdir(exist_ok=True,parents=True)
    row=dict(name=name,variant=variant,repeat=repeat,dimension=dim,cells=n,pn_order=pn,end_time=t,dt=dt,path=str(folder),command=cmd,returncode=code,process_seconds=elapsed,status='completed' if code==0 else 'failed')
    (folder/'stdout.log').write_text(stdout);(folder/'stderr.log').write_text(stderr)
    if code==0:
        row.update(json.loads((folder/'summary.json').read_text()));row.update(metric(folder,t,scenario))
        row['seconds_per_SSP_stage']=row['solver_seconds']/(3*row['steps']) if row['steps'] else None
    (folder/'process.json').write_text(json.dumps(row,indent=2)+'\n')
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suite',choices=['screen','performance','main','regression','boundary','cfl'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--variants',nargs='+',default=['A','B','C']);ap.add_argument('--binary',default='m1pn');a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    plan=dict(suite=a.suite,variants=a.variants,host=platform.node(),cpu=min(os.sched_getaffinity(0)),threads=1,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),binary_sha256=hashlib.sha256((ROOT/'tmp/build/m1pn').read_bytes()).hexdigest(),reference='docs/EXPERIMENT_PLAN.json')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[]
    def record(**kw):
        kw.setdefault('binary',a.binary);rows.append(run(out,**kw));(out/'trials.json').write_text(json.dumps(rows,indent=2)+'\n')
    if a.suite in ['screen','performance']:
        for n,pn in [(8,3),(16,7)]:
            for rep in range(3 if a.suite=='performance' else 1):
                for v in (a.variants if rep%2==0 else list(reversed(a.variants))):record(name=f'crossing_{n}_P{pn}_{v}_{rep}',variant=v,dim=2,n=n,pn=pn,t=.01 if a.suite=='performance' else .002,dt=.0001,repeat=rep)
    elif a.suite=='main':
        for v in a.variants:record(name=f'crossing64_P7_{v}',variant=v,dim=2,n=64,pn=7,t=.32,dt=.0001,timeout=2400)
    elif a.suite=='regression':
        for dim,n,pn,sc in [(1,16,3,'smooth'),(2,8,3,'crossing'),(3,3,3,'stress')]:
            for binary in ['m1pn_baseline','m1pn']:
                record(name=f'regression_{dim}d_{binary}',variant='A',dim=dim,n=n,pn=pn,t=.002,dt=.0001,scenario=sc,binary=binary,extra=['--periodic'] if sc=='smooth' else [])
        checks=[]
        for dim in [1,2,3]:
            aa=out/f'regression_{dim}d_m1pn_baseline';bb=out/f'regression_{dim}d_m1pn'
            checks.append(dict(dimension=dim,final_byte_equal=(aa/'final.csv').read_bytes()==(bb/'final.csv').read_bytes(),stages_byte_equal=(aa/'stages.csv').read_bytes()==(bb/'stages.csv').read_bytes(),balance_byte_equal=(aa/'boundary_integral.csv').read_bytes()==(bb/'boundary_integral.csv').read_bytes()))
        (out/'regression.json').write_text(json.dumps(checks,indent=2)+'\n');assert all(r['final_byte_equal'] and r['stages_byte_equal'] and r['balance_byte_equal'] for r in checks)
    elif a.suite=='boundary':
        for v in a.variants:
            record(name=f'vacuum_inflow_{v}',variant=v,dim=1,n=32,pn=3,t=.06,dt=.0001,scenario='vacuum',extra=['--inflow','pulse','--m1-exterior','isotropic','--exterior-density','1','--pn-boundary-quadrature','consistent'])
            record(name=f'affine_{v}',variant=v,dim=1,n=32,pn=3,t=.05,dt=.0001,scenario='affine',extra=['--inflow','affine','--pn-boundary-quadrature','consistent'])
    else:
        for v in a.variants:
            for cfl in [.2,.5,.8,1.]:record(name=f'cfl_{v}_{cfl}',variant=v,dim=2,n=16,pn=3,t=.02,dt=0,extra=['--cfl',str(cfl)])
    keys=sorted(set(k for r in rows for k,v in r.items() if not isinstance(v,(list,dict))))
    with (out/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print('DONE',len(rows),'runs;',sum(r['returncode']!=0 for r in rows),'failed',flush=True)
if __name__=='__main__':main()
