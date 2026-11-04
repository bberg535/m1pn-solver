#!/usr/bin/env python3
"""Public CLI wiring guards: prevent silently reusing a different mass solver."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
binary=Path(sys.argv[1]).resolve()
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with tempfile.TemporaryDirectory(prefix='pn_mc_cli_') as tmp:
    root=Path(tmp)
    def run(label,extra,ok):
        out=root/label
        cmd=[str(binary),'--dimension','1','--elements','4','--pn-order','3','--case','vacuum','--time','.0001','--dt','.0001','--output',str(out)]+extra
        p=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        assert (p.returncode==0)==ok,(label,p.returncode,p.stdout,p.stderr)
        if ok:
            c=json.loads((out/'config.json').read_text());s=json.loads((out/'summary.json').read_text())
            assert c['mass_mode']=='lumped' and not c['mass_factorized']
            assert c['pn_mass_correction_applications']==1
            assert s['mass_solve_calls']==0 and s['biharmonic_seconds']==0
        return out
    for st in ['m1pn_mass_correction','m1pn_mass_correction_boundary_consistent']:
        run(st,['--stabilization',st],True)
        for suffix,flags in [('consistent',['--mass-mode','consistent']),('afc',['--mass-mode','lumped_afc']),('rk2',['--time-integrator','ssprk2']),('imex',['--time-integrator','imex-euler']),('legacy',['--model','legacy']),('speed',['--pn-viscosity-scale','.9'])]:
            run(st+suffix,['--stabilization',st]+flags,False)
    run('invalid_C_small',['--stabilization','m1pn_mass_correction_boundary_consistent','--elements','2'],False)
print('PN mass-correction CLI selectors and conflicting-mode guards passed')
