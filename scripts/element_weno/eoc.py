import json,subprocess,sys,math,os
exe,tmp=sys.argv[1],sys.argv[2]
env=dict(os.environ,OMP_NUM_THREADS="1")
for label,flags in (("WENO+limiter",[]),("no stabilization, no limiter",["--no-stabilization","--no-limit"])):
    print("==",label)
    for p in (1,2,3,4):
        prev=None; out=[]
        for n in (8,16,32,64):
            h=1/n; dt=.25*h**((p+1)/3)/(p*p)
            r=subprocess.run([exe,"--problem","smooth","--pn-order","3","--degree",str(p),"--dimension","1","--elements",str(n),"--time","0.2","--dt",str(dt),"--output",tmp]+flags,env=env,capture_output=True,text=True,timeout=280)
            if r.returncode: print("FAIL",p,n,r.stderr.strip()); break
            s=json.loads(r.stdout.strip().splitlines()[-1]); e=s["l2_density_error"]
            out.append(f"n={n}: {e:.3e}"+(f" (EOC {math.log2(prev/e):.2f})" if prev else "")+f" g={s['min_gamma']:.3f} lim={s['limited_elements']}")
            prev=e
        print(f"p={p} dt_bound@64={s['dt_bound']:.2e} | "+" | ".join(out))
