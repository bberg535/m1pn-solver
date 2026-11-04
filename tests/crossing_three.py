#!/usr/bin/env python3
"""Three-beam initial moments and a short ghost-stabilized transport comparison."""
import csv, json, math, pathlib, subprocess, sys, tempfile
binary = sys.argv[1]
def read_csv(path):
    with open(path, newline="") as f: return list(csv.DictReader(f))
def run(tmp, case, inflow):
    out = pathlib.Path(tmp) / case
    subprocess.run([binary,"--dimension","2","--elements","8","--pn-order","3",
      "--case",case,"--inflow",inflow,"--stabilization","m1pn_ghost_local_projected_p4",
      "--ghost-gamma","0.001","--time","0.0001","--cfl","0.02","--output",str(out)],
      check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return out
with tempfile.TemporaryDirectory(prefix="m1pn_crossing_three_") as tmp:
    three=pathlib.Path(tmp)/"crossing-three"
    base=[binary,"--dimension","2","--elements","8","--pn-order","3","--case","crossing-three","--stabilization","m1pn_ghost_local_projected_p4","--ghost-gamma","0.001","--pulse-width","0.05","--background","0.002","--time","0.0001","--cfl","0.02"]
    subprocess.run(base+["--inflow","crossing-three-transport","--checkpoint-stride","1","--output",str(three)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    vacuum=pathlib.Path(tmp)/"crossing-three-vacuum"
    subprocess.run(base+["--inflow","vacuum","--output",str(vacuum)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    two=run(tmp,"crossing","crossing-transport")
    cfg=json.loads((three/"config.json").read_text())
    assert cfg["beam_count"]==3 and cfg["background"]==0.002 and cfg["pulse_width"]==0.05
    assert abs(cfg["beam_geometry"]["width_exponent"]+200)<1e-12 and cfg["checkpoint_stride"]==1
    rows=read_csv(three/"initial.csv"); err=0.
    for row in rows:
        x,y=float(row["x"]),float(row["y"])
        a=math.exp(-((x-.32)**2+(y-.5)**2)/(2*.05**2)); b=math.exp(-((x-.5)**2+(y-.32)**2)/(2*.05**2)); c=math.exp(-((x-.68)**2+(y-.5)**2)/(2*.05**2))
        root=math.sqrt(4*math.pi); root3=math.sqrt(4*math.pi/3)
        rho=root*float(row["u0"]); jx=root3*float(row["u3"]); jy=root3*float(row["u1"])
        err=max(err,abs(rho-(.002+a+b+c)),abs(jx-(a-c)),abs(jy-b))
    assert err<2e-13,err
    summary=json.loads((three/"summary.json").read_text()); assert summary["steps"]>0
    progress=json.loads((three/"progress.json").read_text()); assert progress["status"]=="completed" and progress["steps"]==summary["steps"]
    checkpoints=sorted(three.glob('checkpoint_step_*.json'))
    assert len(checkpoints)==summary['steps'], 'Missing accepted-step checkpoints'
    previous_time=0.
    for k,path in enumerate(checkpoints,1):
        checkpoint=json.loads(path.read_text())
        assert checkpoint['step']==k and checkpoint['time']>previous_time
        assert path.with_suffix('.csv').is_file()
        previous_time=checkpoint['time']
    assert previous_time==summary['time']
    assert checkpoints[-1].with_suffix('.csv').read_bytes()==(three/'final.csv').read_bytes()
    # Sum of weak boundary density loads is the integrated net outward mass flux.
    def rho_load(path):
        total=0.; root=math.sqrt(4*math.pi)
        for r in read_csv(path/"boundary_integral.csv"): total+=root*float(r["u0"])
        return total
    delta=rho_load(three)-rho_load(vacuum)
    assert delta<0, ("incoming kinetic data must add inward boundary mass",delta)
    old_cfg=json.loads((two/"config.json").read_text())
    assert old_cfg["beam_count"]==2 and old_cfg["beam_geometry"]["width_exponent"]==-50
    print(json.dumps({"check":"crossing_three_initial_and_transport","passed":True,"moment_error":err,"extra_inflow_mass":delta,"steps":summary["steps"]}))
