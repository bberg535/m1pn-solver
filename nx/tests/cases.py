#!/usr/bin/env python3
"""Actual CLI/physics checks; retain raw solver output inside the experiment."""
import json, math, os, pathlib, subprocess, sys, tempfile
import numpy as np
NX=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(NX.parent/'scripts'))
from verify_run import verify

binary=pathlib.Path(sys.argv[1]).resolve()
(NX/'tmp').mkdir(exist_ok=True)
out=pathlib.Path(tempfile.mkdtemp(prefix='case_checks_',dir=NX/'tmp'))
def run(label,case,dim,grid,extra=(),time=.0001):
    folder=out/label
    command=[str(binary),'--case',case,'--dimension',str(dim),'--elements',str(grid),
             '--pn-order','3','--model','r2','--time-integrator','ssprk3',
             '--stabilization','m1pn_mass_correction','--pn-viscosity-scale','1',
             '--mass-mode','lumped',
             '--kv-gamma','.05','--background','.001' if case!='lattice-paul' else '.000001',
             '--pulse-width','.1','--time',str(time),'--cfl','.2','--threads','1',
             '--output',str(folder),*extra]
    proc=subprocess.run(command,capture_output=True,text=True,timeout=60)
    (out/(label+'.log')).write_text(json.dumps(command)+'\n'+proc.stdout+proc.stderr)
    assert proc.returncode==0,(label,proc.stderr)
    check=verify(folder)
    cfg=json.loads((folder/'config.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    assert cfg['method']=='m1pn_mass_correction' and cfg['stabilization']=='m1pn_mass_correction'
    assert cfg['mass_mode']=='lumped' and cfg['mass_corrections']==0 and not cfg['mass_factorized']
    assert cfg['pn_viscosity_scale']==1 and cfg['pn_mass_correction_applications']==1
    assert cfg['uses_pn_low_order_predictor'] is True
    assert summary['mass_solve_calls']==0 and summary['mass_iterations']==0 and summary['steps']>0
    assert summary['biharmonic_seconds']==0 and summary['mass_seconds']>=0
    assert summary['pn_predictor_seconds']>0 and summary['pn_mass_correction_seconds']>=0
    return folder,cfg,summary
def field(folder,name):return np.loadtxt(folder/name,delimiter=',',skiprows=1,ndmin=2)

for case,inflow,dim in [('crossing-study','crossing-study-transport',2),('crossing-3d','crossing-3d-transport',3)]:
    folder,cfg,summary=run(case,case,dim,4,['--inflow',inflow])
    initial=field(folder,'initial.csv');xyz=initial[:,1:4]
    exact=np.zeros((len(xyz),4));exact[:,0]=.001
    for beam in cfg['beam_geometry']['beams']:
        center=np.array(beam['center']);direction=np.array(beam['direction'])
        value=np.exp(-np.sum((xyz-center)**2,axis=1)/(.02))
        exact[:,0]+=value;exact[:,1:]+=value[:,None]*direction
    actual=initial[:,[4,7,5,6]]*np.array([math.sqrt(4*math.pi)]+[math.sqrt(4*math.pi/3)]*3)
    assert np.max(np.abs(exact-actual))<3e-14,(case,np.max(np.abs(exact-actual)))
    assert cfg['beam_count']==(3 if dim==3 else 2)
    vacuum,_,_=run(case+'_vacuum',case,dim,4,['--inflow','vacuum'])
    incoming_delta=np.sum(field(folder,'boundary_integral.csv')[:,4]-field(vacuum,'boundary_integral.csv')[:,4])
    assert incoming_delta<0,(case,'transport inflow missing',incoming_delta)
    if dim==3:assert actual[:,3].max()>.1,'third spatial streaming direction missing'

folder,cfg,summary=run('lattice','lattice-paul',2,7)
mass=field(folder,'nodal_mass.csv')[:,1];material=field(folder,'material.csv')
values=[mass.sum(),mass@material[:,3],mass@material[:,1],mass@material[:,2]]
assert np.max(np.abs(np.array(values)-[49,1,110,38]))<1e-10,values
assert np.max(field(folder,'initial.csv')[:,1:3])==7
initial=field(folder,'initial.csv')[:,4]*math.sqrt(4*math.pi)
final=field(folder,'final.csv')[:,4]*math.sqrt(4*math.pi)
assert mass@(final-initial)>9e-5,'central source failed to inject particles'
assert np.all(material[:,1:]>=0),'FE material average must be nonnegative'

bad=subprocess.run([str(binary),'--case','lattice-paul','--dimension','2','--elements','8',
                    '--time','0','--output',str(out/'misaligned')],capture_output=True,text=True)
assert bad.returncode==2 and '7' in bad.stderr,'unaligned lattice grid silently accepted'
for conflict in [('--mass-mode','lumped_afc'),('--time-integrator','ssprk2')]:
    bad=subprocess.run([str(binary),'--case','smooth','--dimension','1','--elements','3',
                        '--stabilization','m1pn_mass_correction',*conflict,
                        '--time','0','--output',str(out/('invalid_'+conflict[1]))],
                       capture_output=True,text=True)
    assert bad.returncode==2,bad.stderr
for count in [100,1000]:
    _,_,summary=run('fixed_steps_'+str(count),'smooth',1,3,
                    ['--periodic','--dt',format(.32/count,'.17g')],time=.32)
    assert summary['steps']==count,(count,summary['steps'],'extra microscopic endpoint step')
    assert summary['time']==.32
print(json.dumps({'status':'passed','raw_outputs':str(out),'lattice_integrals':values,
                  'actual_cases':['crossing-study','crossing-3d','lattice-paul']}))
