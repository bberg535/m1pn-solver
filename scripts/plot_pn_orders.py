#!/usr/bin/env python3
"""Plot verified PN-only standard-B runs of different PN orders/grids."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx" / "scripts"))
import analysis
import verify_pn_run

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', nargs=3, required=True, type=Path, metavar=('P3','P5','P7'))
    ap.add_argument('--output', required=True, type=Path)
    a=ap.parse_args()
    runs=[p.resolve(strict=True) for p in a.runs]
    output=a.output.resolve()
    if any(ROOT not in r.parents for r in runs) or ROOT not in output.parents:
        raise ValueError('runs and output must be inside the experiment root')
    if output.exists(): raise ValueError(f'output already exists: {output}')
    records=[]
    for expected,run in zip((3,5,7),runs):
        verification,arrays=verify_pn_run.check(run)
        config=json.loads((run/'config.json').read_text()); summary=json.loads((run/'summary.json').read_text())
        if verification.get('status')!='passed' or config.get('pn_order')!=expected:
            raise ValueError(f'{run}: passed verification for expected P{expected} required')
        if config.get('final_time')!=.32 or summary.get('steps')!=4954:
            raise ValueError(f'{run}: expected common final time .32 and 4954 steps')
        records.append((run,verification,arrays,config,summary))
    configs=[r[3] for r in records]
    for key in ('dimension','mesh_type','final_time','dt_selected','beam_geometry','case','inflow','stabilization','model','pn_target'):
        if any(c.get(key)!=configs[0].get(key) for c in configs[1:]):
            raise ValueError(f'runs do not share required configuration: {key}')
    if any(r[4].get('steps')!=records[0][4].get('steps') for r in records):
        raise ValueError('runs do not have common step count')
    orders=[3,5,7]
    if [c.get('pn_order') for c in configs]!=orders: raise ValueError('runs must be ordered P3, P5, P7')
    # Native-grid fields, references, and lumped-mass metrics.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':200,'pdf.fonttype':42})
    fields=[]; errors=[]; metrics=[]; line_rows=[]; axes_grids=[]; line_data=[]
    for order,(run,ver,arr,cfg,summ) in zip(orders,records):
        coords=arr['coordinates']; rho=arr['physical'][:,0]; mass=arr['mass']
        ref,_=analysis.beam_exact(coords,float(cfg['final_time']),cfg['beam_geometry'])
        err=rho-ref; vol=analysis.volume_of(coords,2)
        fields.append((coords,rho,ref,order,cfg['elements_per_axis']))
        errors.append((coords,err))
        metrics.append({'pn_order':order,'grid':cfg['elements_per_axis'],'nodes':len(rho),'time':cfg['final_time'],'dt':cfg['dt_selected'],'steps':summ['steps'],'density_L1_vs_kinetic':float(np.sum(mass*np.abs(err))/vol),'density_L2_vs_kinetic':float(np.sqrt(np.sum(mass*err**2)/vol)),'density_Linf_vs_kinetic':float(np.max(np.abs(err))),'minimum_density':float(np.min(rho)),'negative_density_nodes':int(np.count_nonzero(rho<0)),'maximum_global_Q_balance':ver['maximum_global_Q_balance'],'domain_volume':vol,'solver_seconds':summ.get('solver_seconds')})
        grid, tensor=analysis.grid_tensor(coords,np.column_stack((rho,ref,err)),2); yi=int(np.argmin(np.abs(grid[1]-.5)))
        if abs(float(grid[1][yi])-.5)>1e-12: raise ValueError(f'P{order} native grid lacks y=.5')
        axes_grids.append(grid); line_data.append(tensor[:,yi,:])
        for i,x in enumerate(grid[0]): line_rows.append({'pn_order':order,'grid':cfg['elements_per_axis'],'x':float(x),'y':float(grid[1][yi]),'rho':float(tensor[i,yi,0]),'kinetic_reference_on_native_grid':float(tensor[i,yi,1]),'signed_error':float(tensor[i,yi,2])})
    density_min=min(0.,*(float(np.min(x[1])) for x in fields),*(float(np.min(x[2])) for x in fields))
    density_max=max(*(float(np.max(x[1])) for x in fields),*(float(np.max(x[2])) for x in fields))
    density_norm=TwoSlopeNorm(vmin=density_min,vcenter=0.,vmax=density_max)
    errmax=max(float(np.max(np.abs(x[1]))) for x in errors); errmax=max(errmax,np.finfo(float).tiny)
    errnorm=TwoSlopeNorm(vmin=-errmax,vcenter=0.,vmax=errmax)
    output.mkdir(parents=True)
    fig,axs=plt.subplots(2,3,figsize=(13,7.8),layout='constrained')
    for j,(coords,rho,ref,order,n) in enumerate(fields):
        xx,yy=np.unique(coords[:,0]),np.unique(coords[:,1]); _,rt=analysis.grid_tensor(coords,rho[:,None],2); _,et=analysis.grid_tensor(coords,(rho-ref)[:,None],2)
        im=axs[0,j].pcolormesh(xx,yy,rt[:,:,0].T,shading='auto',cmap='RdBu_r',norm=density_norm)
        axs[0,j].set(title=f'PN-only B, P{order}, {n}² Zellen\nmin ρ={np.min(rho):.3g}',xlabel='x',ylabel='y',xlim=(0,1),ylim=(0,1),aspect='equal')
        fig.colorbar(im,ax=axs[0,j],shrink=.84,label='ρ (gemeinsame Skala, negative Werte enthalten)')
        im=axs[1,j].pcolormesh(xx,yy,et[:,:,0].T,shading='auto',cmap='RdBu_r',norm=errnorm)
        axs[1,j].set(title=f'ρ − kinetische Referenz, P{order}, {n}² Zellen',xlabel='x',ylabel='y',xlim=(0,1),ylim=(0,1),aspect='equal')
        fig.colorbar(im,ax=axs[1,j],shrink=.84,label='signierter Dichtefehler')
    fig.suptitle('PN-only B · t=0,32 · P3/P7 auf 128², P5 auf 64² Zellen\nFehler zur kinetischen Referenz enthalten PN-Modellfehler')
    for ext in ('png','pdf'): fig.savefig(output/f'pn_orders_density_errors.{ext}',bbox_inches='tight')
    plt.close(fig)
    # Analytic line reference sampled on the finest existing x-grid (P3/P7 have N=128).
    finest_idx=int(np.argmax([len(g[0]) for g in axes_grids])); finecoords=fields[finest_idx][0]; finegrid=axes_grids[finest_idx]
    refline,_=analysis.beam_exact(np.column_stack((finegrid[0],np.full(len(finegrid[0]),.5),np.zeros(len(finegrid[0])))),.32,configs[0]['beam_geometry'])
    fig,ax=plt.subplots(figsize=(9.5,5.5),layout='constrained')
    ax.plot(finegrid[0],refline,'k-',lw=2,label='Analytische kinetische Referenz (feinstes x-Gitter)')
    for (order,grid),ag,ld in zip(zip(orders,[c['elements_per_axis'] for c in configs]),axes_grids,line_data):
        ax.plot(ag[0],ld[:,0],marker='o',ms=2.5,lw=1,label=f'PN-only P{order}, {grid}² Zellen')
    ax.set(xlabel='x',ylabel='ρ(x, 0.5)',title='PN-only-Dichte am gemeinsamen Mittelschnitt y=0,5; t=0,32'); ax.grid(alpha=.25); ax.legend(fontsize=8,ncols=2)
    for ext in ('png','pdf'): fig.savefig(output/f'pn_orders_midline.{ext}',bbox_inches='tight')
    plt.close(fig)
    with (output/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    with (output/'midline.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(line_rows[0]));w.writeheader();w.writerows(line_rows)
    provenance={'comparison':'PN-only standard B, P3/P5/P7','plot_script_sha256':sha(Path(__file__)),'analysis_helper_sha256':sha(ROOT/'nx/scripts/analysis.py'),'verifier_sha256':sha(ROOT/'scripts/verify_pn_run.py'),'runs':[str(r[0]) for r in records], 'runs_metadata':[{'pn_order':r[3]['pn_order'],'grid':r[3]['elements_per_axis'],'config_sha256':sha(r[0]/'config.json'),'summary_sha256':sha(r[0]/'summary.json'),'verification_sha256':sha(r[0]/'verification.json'),'verified_input_hashes':r[1]['sha256']} for r in records], 'common_settings':{k:configs[0].get(k) for k in ('dimension','mesh_type','final_time','dt_selected','beam_geometry','case','inflow','stabilization','model','pn_target')},'common_steps':records[0][4]['steps'],'norm':'L1=sum(m_i |e_i|)/volume; L2=sqrt(sum(m_i e_i^2)/volume); Linf=max nodal absolute error; each on its own native mesh and lumped masses.','reference':'Analytic kinetic free-transport density; includes finite PN model, boundary, and spatial errors.','density_scale':{'vmin':density_min,'vmax':density_max,'zero_centered':True},'signed_error_symmetric_scale':[-errmax,errmax],'midline':'Native y=0.5 samples per run; analytic reference evaluated on finest available x grid.'}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output':str(output),'metrics':metrics},indent=2))

if __name__=='__main__':
    try: main()
    except Exception as e: print(f'plot_pn_orders.py: {e}',file=sys.stderr); raise SystemExit(2)
