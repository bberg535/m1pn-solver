#!/usr/bin/env python3
"""End-to-end native transport/filter/ledger checks on a tiny crossing case."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

binary = Path(sys.argv[1]).resolve()
original = binary.with_name('m1pn_unfiltered')
with tempfile.TemporaryDirectory(prefix='nn_filter_') as tmp:
    root = Path(tmp)
    common = ['--dimension','2','--elements','4','--pn-order','3','--case','crossing-study',
              '--inflow','crossing-study-transport','--stabilization','m1pn_mass_correction',
              '--time','.002','--dt','.001','--threads','1','--cuda','off']
    def run(name, args=(), exe=binary, ok=True):
        out=root/name
        r=subprocess.run([str(exe),*common,*args,'--output',str(out)],text=True,capture_output=True)
        if ok and r.returncode: raise AssertionError(r.stderr)
        if not ok:
            assert r.returncode != 0
            return
        cfg=json.loads((out/'config.json').read_text())
        summary=json.loads((out/'summary.json').read_text())
        assert summary['status']=='completed' and abs(summary['time']-.002)<1e-14
        assert summary['global_Q_balance']<1e-11
        assert summary['rejected_attempts']==0
        return out, cfg, summary
    baseline,_,_=run('baseline',exe=original)
    off,_,_=run('off',['--filter','off'])
    zero,_,_=run('zero',['--filter','constant','--filter-strength','0'])
    for filename in ('initial.csv','final.csv','boundary_integral.csv','reaction_integral.csv'):
        assert (baseline/filename).read_bytes()==(off/filename).read_bytes(), filename
        assert (off/filename).read_bytes()==(zero/filename).read_bytes(), filename
    for coupling in ('m1pn','pn-only'):
        for mode in ('constant','nn'):
            out,cfg,s=run(coupling+'_'+mode,['--filter',mode,'--filter-strength','31.75','--coupling',coupling])
            assert s['filter_density_ledger_max']==0
            strength=np.loadtxt(out/'filter_strength.csv',delimiter=',',skiprows=1)[:,1]
            assert np.isfinite(strength).all() and strength.min()>=0 and strength.max()>0
            initial=np.loadtxt(out/'initial.csv',delimiter=',',skiprows=1)[:,4:]
            final=np.loadtxt(out/'final.csv',delimiter=',',skiprows=1)[:,4:]
            mass=np.loadtxt(out/'nodal_mass.csv',delimiter=',',skiprows=1)[:,1]
            loads=[np.loadtxt(out/f,delimiter=',',skiprows=1)[:,4:]
                   for f in ('boundary_integral.csv','reaction_integral.csv','filter_integral.csv')]
            residual=np.sum(mass[:,None]*(final-initial)+loads[0]-loads[1]-loads[2],axis=0)
            assert np.max(np.abs(residual))<1e-11
    run('bad_order',['--filter','nn','--pn-order','2'],ok=False)
    run('bad_strength',['--filter','constant','--filter-strength','-1'],ok=False)
    run('bad_dim',['--filter','nn','--dimension','3'],ok=False)
    for mode in ('constant','nn'):
        out,cfg,s=run('material_'+mode,['--filter',mode,'--filter-strength','31.75',
            '--case','smooth','--inflow','vacuum','--sigma-a','.2','--sigma-s','.5','--source-density','.1'])
        assert s['filter_density_ledger_max']==0 and s['global_Q_balance']<1e-11
print(json.dumps({'status':'passed','off_matches_original_bitwise':True,
                  'zero_matches_off_bitwise':True,'filtered_transport_modes':4}))
