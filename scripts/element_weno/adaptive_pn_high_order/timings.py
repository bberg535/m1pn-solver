import csv
import os
import random
import subprocess

from campaign_paths import BASE, CPUS

base = BASE
configs = []
def repeat(group, make_options, args, reps=5):
    for repetition in range(reps):
        labels = list(args)
        if repetition % 2:
            labels.reverse()
        configs.append((group, repetition, [(label, *make_options(label)) for label in labels]))

for n, steps in ((7,16),(11,16),(15,40),(15,16),(15,8),(21,16)):
    repeat(f"beam_{n}_{steps}_pn",
           lambda mode: ("physics_timing", ("beam",2,24,n,mode,steps,.002,.004,"pn",32)),
           ("fixed","warm"), reps=5)
repeat("counter_15_16_pn", lambda mode: ("physics_timing", ("counter",2,24,15,mode,16,.002,.004,"pn",32)),
       ("fixed","warm"))
for case in ("beam","counter"):
    repeat(f"{case}_15_m1_native", lambda mode: (
        ("physics_timing",(case,2,24,15,mode,16,.002,.004,"m1",32)) if mode in ("fixed","warm")
        else ("native", (case,24,15,16,mode,.2))),
        ("fixed","warm","plain","weno"))
    repeat(f"{case}_fine_matched", lambda mode: (
        "physics_timing", (case,2,48,15 if mode != "fixed11" else 11,
                           "warm" if mode == "warm15" else "fixed",16,.00002,.00004,"pn",32)),
        ("fixed11","fixed15","warm15"), reps=3)

random.Random(86058).shuffle(configs)
env = os.environ | {"OMP_NUM_THREADS": "1"}
columns = ("group","repetition","variant","case","count","N","CFL","rho_error","J_error","P_error",
           "timed_seconds","cpu_seconds","adapt_seconds","max_rss_kib","limited","mean_dofs")
with (base/"timings.csv").open("w",newline="") as out:
    writer = csv.DictWriter(out,fieldnames=columns,extrasaction="ignore")
    writer.writeheader()
    for group,repetition,rows in configs:
        for label, exe, argv in rows:
            cmd=["taskset", "-c", str(CPUS[0]),str(base/exe),*map(str,argv)]
            finished = subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=100)
            if finished.returncode:
                raise RuntimeError(f"{group} {label} repetition {repetition}: {finished.stderr.strip()}")
            row = next(csv.DictReader(finished.stdout.splitlines()))
            if exe == "physics_timing":
                row["cpu_seconds"] = float(row["cpu_advance"]) + float(row["cpu_adapt"])
            row["limited"] = row.get("limited_element_stages",row.get("limit_count"))
            writer.writerow(row | dict(group=group,repetition=repetition,variant=label))
            out.flush()
        print(group,repetition,flush=True)
