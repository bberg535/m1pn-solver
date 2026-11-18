import csv
import subprocess
from campaign_paths import BASE, CPUS

base = BASE
scans = [(2,6,n,"checker", (.62,.64,.66,.68,.70,.72,.74,.85,.875,.90)) for n in (7, 15, 21)]
scans += [(2,12,15,"checker", (.5,.60,.64,.68,.72,.75,.85,.875,.90)),
          (3,4,15,"checker", (.34,.38,.42,.46,.5,.54,.58,.62,.66,.70)),
          (2,6,31,"checker", (.5,.6,.65,.7,.75,.85,.9))]
with (base/"stability_refined.csv").open("w",newline="") as out:
    writer=None
    for dim,count,n,mask,values in scans:
        cmd=["taskset", "-c", str(CPUS[0]),str(base/"stability"),str(dim),str(count),str(n),mask,"30",*map(str,values)]
        text = subprocess.check_output(cmd,text=True,timeout=120)
        rows=list(csv.DictReader(text.splitlines()))
        if writer is None:
            writer = csv.DictWriter(out,fieldnames=rows[0].keys())
            writer.writeheader()
        writer.writerows(rows)
        out.flush()
        print(dim,count,n, [(r["cfl"],r["one_step_sigma_lower"],r["20step_factor"]) for r in rows], flush=True)
