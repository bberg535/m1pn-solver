import json,subprocess,sys,math,os,csv
exe,base=sys.argv[1],sys.argv[2]
env=dict(os.environ,OMP_NUM_THREADS="1")
# q=3 check at degree three
prev=None
for n in (16,32,64,128):
    h=1/n; dt=.25*h**(4/3)/9
    r=subprocess.run([exe,"--problem","smooth","--pn-order","3","--degree","3","--dimension","1","--elements",str(n),"--time","0.2","--dt",str(dt),"--no-limit","--weno-q","3","--output",base+"/q3"],env=env,capture_output=True,text=True)
    s=json.loads(r.stdout.strip().splitlines()[-1]); e=s["l2_density_error"]
    print(f"q=3 p=3 n={n}: {e:.3e}"+(f" EOC {math.log2(prev/e):.2f}" if prev else "")+f" 1-g={1-s['min_gamma']:.1e}",flush=True); prev=e
w=.07; bg=.001; T=.32
def p(x,y,cx,cy): return math.exp(-((x-cx)**2+(y-cy)**2)/(2*w*w))
def exact(x,y): return bg+p(x-T,y,.3,.5)+p(x,y-T,.5,.3)
for pp,n in ((1,96),(2,48),(3,32)):
    for lab,fl in (("w1e-3",["--weno-linear","1e-3"]),("w0.2",["--weno-linear","0.2"]),("noWENOsensor",["--no-weno"]),("noStab",["--no-stabilization"])):
        o=f"{base}/c_p{pp}_{lab}"
        r=subprocess.run([exe,"--problem","crossing-study","--pn-order","3","--degree",str(pp),"--dimension","2","--elements",str(n),"--time",str(T),"--cfl",".5","--output",o]+fl,env=env,capture_output=True,text=True)
        if r.returncode: print(f"p={pp} n={n} {lab}: FAILED {r.stderr.strip()[:200]}",flush=True); continue
        s=json.loads(r.stdout.strip().splitlines()[-1])
        rows=list(csv.DictReader(open(o+"/final.csv")))
        rho=[float(x["density"]) for x in rows]; ex=[exact(float(x["x"]),float(x["y"])) for x in rows]
        e2=math.sqrt(sum((a-b)**2 for a,b in zip(rho,ex))/len(rho))
        print(f"p={pp} n={n} {lab}: steps={s['steps']} wall={s['wall_seconds']:.1f}s minrho={min(rho):.2e} maxrho={max(rho):.3f} L2(vs kinetic)={e2:.3e} min_gamma={s['min_gamma']:.3f} min_lim={s['min_limiter']:.3f} lim_el={s['limited_elements']} bd_lim={s['limited_boundary_nodes']} nudged={s['nudged_states']}",flush=True)
