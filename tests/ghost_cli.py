#!/usr/bin/env python3
import json,os,subprocess,sys,tempfile
from pathlib import Path
binary=str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix='m1pn_ghost_cli_') as temp:
    root=Path(temp)
    common=[binary,'--method','m1pn_ghost_local_projected_p4','--ghost-gamma','.001','--dimension','1','--elements','4','--time','0']
    def run(name,extra,ok):
        x=subprocess.run(common+extra+['--output',str(root/name)],capture_output=True,text=True,timeout=30)
        assert (x.returncode==0)==ok,(name,x.stdout,x.stderr)
        return x
    run('valid',[],True)
    config=json.loads((root/'valid/config.json').read_text())
    assert config['method']==config['stabilization']=='m1pn_ghost_local_projected_p4'
    run('small',['--elements','3'],False)
    run('periodic',['--periodic'],False)
    run('legacy',['--model','legacy'],False)
    step=(config['pn_step_limit']+config['low_cfl_limit'])/2
    assert step>config['pn_step_limit'] and step<config['low_cfl_limit']
    failure=run('pn_step',['--dt',str(step),'--time',str(step)],False)
    assert 'independent PN time restriction' in failure.stderr,failure.stderr
print('ghost CLI and separate PN step restriction passed')
