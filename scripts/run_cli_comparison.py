#!/usr/bin/env python3
"""Matched real CLI comparisons, including initialization and output cost."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from run_efficiency import ROOT, compare, density_error


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); out=args.output.resolve();out.mkdir(exist_ok=False, parents=True)
    cases = [("smooth_2d", "smooth", 2, 16, 3), ("crossing_2d", "crossing", 2, 12, 7),
             ("crossing_3d", "crossing", 3, 6, 3)]
    cpu=min(os.sched_getaffinity(0));env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    rows=[]
    (out/"plan.json").write_text(json.dumps(dict(cases=cases, repeats=5, time=.04, dt=.0001,
         cpu=cpu, checkpoint_stride=20, variants=["reference","cached"],
         binaries_sha256={n:hashlib.sha256((ROOT/"tmp/build"/n).read_bytes()).hexdigest() for n in ["m1pn_reference","m1pn"]},
         driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+"\n")
    for case,scenario,dim,elements,pn in cases:
        for repeat in range(5):
            group={}
            for variant in (["reference","cached"] if repeat%2==0 else ["cached","reference"]):
                target=out/f"{case}_{variant}_{repeat}"
                binary=ROOT/"tmp/build"/("m1pn_reference" if variant=="reference" else "m1pn")
                cmd=["taskset","-c",str(cpu),str(binary),"--dimension",str(dim),"--elements",str(elements),
                     "--pn-order",str(pn),"--case",scenario,"--stabilization","biharmonic-corrected",
                     "--biharmonic-gamma",".004","--time",".04","--dt",".0001","--threads","1",
                     "--checkpoint-stride","20","--output",str(target)]
                if scenario=="smooth":cmd += ["--periodic"]
                else:cmd += ["--inflow","crossing-transport"]
                if variant=="cached":cmd += ["--mcl-mode","lazy","--geometry-cache","on"]
                print(f"RUN {case} {variant} {repeat}",flush=True)
                begin=time.perf_counter();proc=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                elapsed=time.perf_counter()-begin;target.mkdir(exist_ok=True)
                (target/"stdout.log").write_text(proc.stdout);(target/"stderr.log").write_text(proc.stderr)
                row=dict(case=case,variant=variant,repeat=repeat,command=cmd,path=str(target),process_seconds=elapsed,returncode=proc.returncode)
                if proc.returncode:row.update(status="failed",error=proc.stderr)
                else:row.update(json.loads((target/"summary.json").read_text()),**density_error(target,scenario,.04))
                rows.append(row);group[variant]=row
                (out/"trials.json").write_text(json.dumps(rows,indent=2)+"\n")
            if all(r["status"]=="completed" for r in group.values()):
                for variant,row in group.items():row.update(compare(Path(group["reference"]["path"]),Path(row["path"])))
                assert group["reference"]["steps"]==group["cached"]["steps"]
                assert group["reference"]["rejected_attempts"]==group["cached"]["rejected_attempts"]
            (out/"trials.json").write_text(json.dumps(rows,indent=2)+"\n")
    summary=[]
    for case,*_ in cases:
        for variant in ["reference","cached"]:
            rr=[r for r in rows if r["case"]==case and r["variant"]==variant]
            if any(r["status"]!="completed" for r in rr):
                summary.append(dict(case=case,variant=variant,passed=False));continue
            item=dict(case=case,variant=variant,passed=all(r["field_gate_pass"] for r in rr),
                      total_seconds=statistics.median(r["total_seconds"] for r in rr),
                      initialization_seconds=statistics.median(r["initialization_seconds"] for r in rr),
                      process_seconds=statistics.median(r["process_seconds"] for r in rr),
                      process_min_seconds=min(r["process_seconds"] for r in rr),
                      process_max_seconds=max(r["process_seconds"] for r in rr),
                      max_physical_difference=max(r["max_physical_difference"] for r in rr),
                      max_rho_difference=max(r["max_rho_difference"] for r in rr),
                      rho_L1=rr[0]["rho_L1"],rho_L2=rr[0]["rho_L2"],rho_Linf=rr[0]["rho_Linf"],
                      rho_relative_L1=rr[0]["rho_relative_L1"])
            summary.append(item)
    for row in summary:
        reference=next(r for r in summary if r["case"]==row["case"] and r["variant"]=="reference")
        if "process_seconds" in row and "process_seconds" in reference:
            row["process_speedup"]=reference["process_seconds"]/row["process_seconds"]
            row["total_speedup"]=reference["total_seconds"]/row["total_seconds"]
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
