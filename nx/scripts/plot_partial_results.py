#!/usr/bin/env python3
"""Snapshot plots of completed Crossing runs; never alter campaign/raw inputs."""
import argparse, collections, csv, datetime, json, os
from pathlib import Path
import numpy as np
import analysis as study_analysis

NX=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    campaign=args.campaign.resolve(strict=True)
    stamp=datetime.datetime.now().astimezone()
    output=(args.output or campaign/'partial_results'/stamp.strftime('%Y%m%d_%H%M%S')).resolve()
    output.relative_to(NX)
    output.mkdir(parents=True,exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(output/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize,TwoSlopeNorm
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':130,'savefig.dpi':190,'pdf.fonttype':42})
    tasks=json.loads((campaign/'tasks.json').read_text())
    manifest=json.loads((campaign/'manifest.json').read_text())
    assert study_analysis.sha256(campaign/'tasks.json')==manifest['tasks_sha256']
    states={};data=[];rows=[];running=[]
    for task in tasks:
        file=campaign/'status'/f'{task["id"]}.json'
        status=json.loads(file.read_text()) if file.exists() else {'status':'not_started'}
        states[task['id']]=status['status']
        if task['family']!='crossing':continue
        if status['status']=='completed' and task['kind']=='spatial':
            d=study_analysis.task_result(campaign,task,status)
            summary=json.loads((campaign/task['output_dir']/'summary.json').read_text())
            d['wall_seconds']=status['wall_seconds'];data.append(d)
            rows.append({'grid':d['grid'],'pn_order':d['order'],'time':d['time'],
                'density_L2':d['exact_error']['density_L2'],'density_Linf':d['exact_error']['density_Linf'],
                'physical_four_moment_RMS':d['exact_error']['L2'],'wall_seconds':d['wall_seconds'],
                'minimum_cone_margin':summary['minimum_cone_margin'],
                'global_Q_balance':summary['global_Q_balance'],'mass_solve_calls':summary['mass_solve_calls'],
                'final_sha256':d['field_sha256'],'run':task['output_dir']})
        elif status['status']=='running':
            progress=campaign/task['output_dir']/'progress.json'
            if progress.is_file():
                p=json.loads(progress.read_text());fraction=p['time']/task['time']
                running.append({'task':task,'progress':p,'fraction':fraction,
                    'projected_total_seconds':p['wall_seconds']/fraction if fraction>0 else None})
    if not rows:raise ValueError('No completed Crossing runs are available')
    rows.sort(key=lambda r:(r['grid'],r['pn_order']))
    with (output/'completed_crossing.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    counts=dict(collections.Counter(states.values()))
    subtitle=f"t={data[0]['time']:g} · PN-Massenkorrektur B · {len(rows)}/18 reguläre Crossing-Läufe fertig · {stamp:%d.%m.%Y %H:%M %Z}"
    def save(fig,name):
        for suffix in ['png','pdf']:fig.savefig(output/f'{name}.{suffix}',bbox_inches='tight')
        plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12.3,4.6),layout='constrained')
    for n in sorted({r['grid'] for r in rows}):
        subset=[r for r in rows if r['grid']==n]
        axes[0].semilogy([r['pn_order'] for r in subset],[r['density_L2'] for r in subset],
                         'o-',lw=1.8,label=f'{n}×{n} Zellen')
    for N in sorted({r['pn_order'] for r in rows}):
        subset=[r for r in rows if r['pn_order']==N]
        axes[1].loglog([r['grid'] for r in subset],[r['density_L2'] for r in subset],
                       'o-',lw=1.7,label=f'N={N}')
    for ax in axes:
        ax.set_ylabel('Dichtefehler L₂ zur kinetischen Referenz')
        ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=8)
    axes[0].set(xlabel='PN-Ordnung N',title='Variation der PN-Ordnung')
    axes[0].set_xticks(sorted({r['pn_order'] for r in rows}))
    axes[1].set(xlabel='Zellen je Achse',title='Gitterverfeinerung bei gleichem N')
    axes[1].set_xticks([32,64,128],labels=['32','64','128']);axes[1].minorticks_off()
    fig.suptitle('Crossing Beams: vorläufige Fehlerkurven\n'+subtitle,fontsize=11)
    save(fig,'crossing_convergence_partial')

    # Same fine grid for the two spatial density/error comparisons.
    finest=max(d['grid'] for d in data)
    eligible=sorted([d for d in data if d['grid']==finest],key=lambda d:d['order'])
    low,high=eligible[0],eligible[-1]
    highest_N=max(data,key=lambda d:(d['order'],d['grid']))
    panels=[(high,high['exact'][:,0],'Kinetische Referenz',False),
            (low,low['physical'][:,0],f'PN-MC B: {finest}², N={low["order"]}',False),
            (high,high['physical'][:,0],f'PN-MC B: {finest}², N={high["order"]}',False),
            (highest_N,highest_N['physical'][:,0],f'PN-MC B: {highest_N["grid"]}², N={highest_N["order"]}',False),
            (low,low['physical'][:,0]-low['exact'][:,0],f'Dichtefehler: {finest}², N={low["order"]}',True),
            (high,high['physical'][:,0]-high['exact'][:,0],f'Dichtefehler: {finest}², N={high["order"]}',True)]
    density_max=max(float(np.max(v)) for _,v,_,error in panels if not error)
    error_max=max(float(np.max(np.abs(v))) for _,v,_,error in panels if error)
    error_max=max(error_max,1e-15)
    density_norm=Normalize(0,density_max);error_norm=TwoSlopeNorm(0,-error_max,error_max)
    fig,axs=plt.subplots(2,3,figsize=(12.2,7.2),layout='constrained')
    for ax,(d,values,title,is_error) in zip(axs.flat,panels):
        grid_axes,tensor=study_analysis.grid_tensor(d['coordinates'],values[:,None],2)
        mesh=ax.pcolormesh(grid_axes[0],grid_axes[1],tensor[:,:,0].T,shading='auto',
            cmap='RdBu_r' if is_error else 'viridis',norm=error_norm if is_error else density_norm)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='x',ylabel='y',title=title,aspect='equal')
        fig.colorbar(mesh,ax=ax,shrink=.84,label='ρ − ρ_ref' if is_error else 'ρ')
    fig.suptitle('Crossing Beams: Dichte und signierter Dichtefehler\n'+subtitle,fontsize=11)
    save(fig,'crossing_density_partial')

    fig,ax=plt.subplots(figsize=(8,4.7),layout='constrained')
    for n in sorted({r['grid'] for r in rows}):
        subset=[r for r in rows if r['grid']==n]
        ax.plot([r['pn_order'] for r in subset],[r['wall_seconds']/3600 for r in subset],
                'o-',label=f'{n}²: abgeschlossen')
    for k,r in enumerate(running):
        if r['projected_total_seconds']:
            ax.scatter(r['task']['order'],r['projected_total_seconds']/3600,marker='D',
                s=70,facecolors='none',edgecolors='tab:red',
                label='Hochrechnung laufender Jobs' if k==0 else None)
            ax.annotate(f'{r["task"]["grid"]}² / N={r["task"]["order"]}',
                (r['task']['order'],r['projected_total_seconds']/3600),xytext=(7,6),textcoords='offset points',fontsize=8)
    ax.axhline(manifest['plan']['execution']['timeout_seconds']/3600,color='black',ls='--',
               lw=1,label='Prozess-Zeitlimit')
    ax.set(xlabel='PN-Ordnung N',ylabel='Laufzeit [Stunden]',title='Crossing: Laufzeit und vorläufige Hochrechnungen')
    ax.set_xticks(sorted({r['pn_order'] for r in rows}));ax.set_ylim(bottom=0)
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    save(fig,'crossing_runtime_partial')

    record={'created':stamp.isoformat(),'campaign':str(campaign),'counts':counts,
        'completed_crossing':rows,'running_crossing':running,
        'reference':'Analytische kinetische freie Transportmomente; keine exakte Lösung des endlichen PN/R2-Modells.',
        'norm':'Knotenmassen-gewichtetes Dichte-L2, normiert auf das Gebietsvolumen.',
        'incomplete':True,'script_sha256':study_analysis.sha256(Path(__file__)),
        'tasks_sha256':study_analysis.sha256(campaign/'tasks.json'),
        'manifest_sha256':study_analysis.sha256(campaign/'manifest.json')}
    (output/'snapshot.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'output':str(output),'counts':counts,'completed_crossing':len(rows),
                      'plots':[p.name for p in sorted(output.glob('*.png'))]},indent=2))

if __name__=='__main__':main()
