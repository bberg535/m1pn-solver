#!/usr/bin/env python3
"""Reproducible local crossing screen with independent saved-field checks."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'nx/scripts'))
from analysis import beam_exact, physical_low, grid_tensor


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    a=np.loadtxt(path,delimiter=',',skiprows=1,ndmin=2)
    if not np.isfinite(a).all(): raise ValueError(f'Nonfinite field: {path}')
    return a


def verify(out):
    cfg=json.loads((out/'config.json').read_text())
    s=json.loads((out/'summary.json').read_text())
    assert s['status']=='completed' and abs(s['time']-cfg['final_time'])<1e-14
    initial, final = read(out/'initial.csv'), read(out/'final.csv')
    mass=read(out/'nodal_mass.csv')[:,1]
    assert np.array_equal(initial[:,:4],final[:,:4]) and np.all(mass>0)
    boundary,reaction,filt=[read(out/f) for f in ('boundary_integral.csv','reaction_integral.csv','filter_integral.csv')]
    assert all(np.array_equal(final[:,:4],a[:,:4]) for a in (boundary,reaction,filt))
    residual=np.sum(mass[:,None]*(final[:,4:]-initial[:,4:])+boundary[:,4:]-reaction[:,4:]-filt[:,4:],axis=0)
    balance=float(np.max(np.abs(residual)))
    assert balance<1e-9 and s['filter_density_ledger_max']==0 and np.max(np.abs(filt[:,4]))==0
    low=physical_low(final[:,4:])
    margin=low[:,0]-np.linalg.norm(low[:,1:],axis=1)
    if cfg['coupling']=='m1pn': assert low[:,0].min()>=-1e-13 and margin.min()>=-1e-12
    reference,_=beam_exact(final[:,1:4],s['time'],cfg['beam_geometry'])
    error=low[:,0]-reference
    strength=read(out/'filter_strength.csv')[:,1]
    assert np.min(strength)>=0
    row={'grid':cfg['elements_per_axis'],'order':cfg['pn_order'],'coupling':cfg['coupling'],
         'filter':cfg['filter_mode'],'dt':cfg['dt_selected'],'steps':s['steps'],
         'density_rel_L1':float(np.sum(mass*np.abs(error))/np.sum(mass*np.abs(reference))),
         'density_rel_L2':float(np.sqrt(np.sum(mass*error**2)/np.sum(mass*reference**2))),
         'density_Linf':float(np.max(np.abs(error))),'minimum_density':float(low[:,0].min()),
         'minimum_cone_margin':float(margin.min()),'negative_density_nodes':int(np.sum(low[:,0]<0)),
         'outside_cone_nodes_tolerance_1e12':int(np.sum(margin < -1e-12)),
         'balance_max':balance,'density_filter_ledger_max':float(np.max(np.abs(filt[:,4]))),
         'strength_min':float(strength.min()),'strength_max':float(strength.max()),
         'total_seconds':s['total_seconds'],'filter_seconds':s['filter_seconds_including_retries'],
         'rejected_attempts':s['rejected_attempts'], 'run':str(out.relative_to(ROOT))}
    names=['initial.csv','final.csv','nodal_mass.csv','config.json','summary.json',
           'boundary_integral.csv','reaction_integral.csv','filter_integral.csv','filter_strength.csv','steps.csv']
    verification={'status':'passed','balance_max':balance,'sha256':{n:sha(out/n) for n in names}}
    (out/'filter_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,default=ROOT/'tmp/build/m1pn_nn_filter')
    p.add_argument('--output',type=Path,default=ROOT/'results/crossing_20261109')
    p.add_argument('--grids',type=int,nargs='+',default=[16,32])
    p.add_argument('--orders',type=int,nargs='+',default=[3,7])
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    runs=ROOT/'tmp'/args.output.name
    runs.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    constants={3:31.75,5:27.27,7:16.52,9:13.62}
    rows=[]
    def run(name,grid,order,coupling,mode,dt=None,final=.32):
        out=runs/name
        cmd=[str(args.binary),'--dimension','2','--elements',str(grid),'--pn-order',str(order),
             '--case','crossing-study','--inflow','crossing-study-transport',
             '--stabilization','m1pn_mass_correction','--coupling',coupling,'--filter',mode,
             '--filter-strength',str(constants[order]),'--time',str(final),'--cfl','.2',
             '--threads','1','--cuda','off','--output',str(out)]
        if dt is not None:cmd+=['--dt',format(dt,'.17g')]
        print(f'START {name}',flush=True)
        with (runs/(name+'.log')).open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=900)
        if result.returncode:raise RuntimeError(f'{name} failed: {runs/(name+".log")}')
        return out
    for grid in args.grids:
        for order in args.orders:
            probe=run(f'probe_n{grid}_p{order}',grid,order,'m1pn','off',final=0.)
            dt=json.loads((probe/'config.json').read_text())['dt_selected']
            for coupling in ('m1pn','pn-only'):
                controls=None
                for mode in ('off','constant','nn'):
                    name=f'{coupling}_{mode}_n{grid}_p{order}'
                    out=run(name,grid,order,coupling,mode,dt)
                    controls_now=[sha(out/f) for f in ('initial.csv','nodal_mass.csv','steps.csv')]
                    # steps.csv includes changing diagnostics; compare time/dt columns only.
                    controls_now[2]=read(out/'steps.csv')[:,:3].tobytes()
                    if controls is None:controls=controls_now
                    assert controls==controls_now,'Comparison inputs or accepted timesteps differ'
                    row=verify(out); row['time_refined']=False; rows.append(row)
                    print(json.dumps(row),flush=True)
                    (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    if 32 in args.grids and 3 in args.orders:
        dt=next(r['dt'] for r in rows if r['grid']==32 and r['order']==3 and r['coupling']=='m1pn')/2
        for mode in ('off','constant','nn'):
            out=run(f'm1pn_{mode}_n32_p3_half_dt',32,3,'m1pn',mode,dt)
            row=verify(out);row['time_refined']=True;rows.append(row)
            print(json.dumps(row),flush=True)
    (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (args.output/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest={'protocol_sha256':sha(ROOT/'protocol.json'),'binary_sha256':sha(args.binary),
              'model_manifest_sha256':sha(ROOT/'models/manifest.json'),
              'parent_manifest_sha256':sha(ROOT/'parent_manifest.json'),
              'results_sha256':sha(args.output/'results.json')}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('COMPLETED',len(rows),flush=True)


if __name__=='__main__':main()
