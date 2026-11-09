import json,subprocess,sys,math,os
exe,tmp=sys.argv[1],sys.argv[2]
env=dict(os.environ,OMP_NUM_THREADS="1")
p=3
for label,flags,scale in (("gamma=1 (no sensor), dt",["--no-weno","--no-limit"],1),("gamma=1, dt/4",["--no-weno","--no-limit"],.25),("WENO, dt/4",["--no-limit"],.25)):
    prev=None; out=[]
    for n in (16,32,64,128):
        h=1/n; dt=scale*.25*h**((p+1)/3)/(p*p)
        r=subprocess.run([exe,"--problem","smooth","--pn-order","3","--degree",str(p),"--dimension","1","--elements",str(n),"--time","0.2","--dt",str(dt),"--output",tmp]+flags,env=env,capture_output=True,text=True,timeout=250)
        s=json.loads(r.stdout.strip().splitlines()[-1]); e=s["l2_density_error"]
        out.append(f"n={n}: {e:.3e}"+(f" (EOC {math.log2(prev/e):.2f})" if prev else "")+f" 1-g={1-s['min_gamma']:.1e}")
        prev=e
    print(label,"|"," | ".join(out))
