import csv,math,subprocess,sys,os
exe,tmp=sys.argv[1:3]; env=dict(os.environ,OMP_NUM_THREADS="1"); W=.07;BG=.001
def ref(x,y,T):
    p=lambda a,b: math.exp(-(a*a+b*b)/(2*W*W)); return BG+p(x-T-.3,y-.5)+p(x-.5,y-T-.3)
def err(path,grid,T):
    a=b=0;mx=0
    for r in csv.DictReader(open(path)):
        x,y,rho=float(r['x']),float(r['y']),float(r['density'])
        h=1/grid; m=(h/2 if x<1e-12 or x>1-1e-12 else h)*(h/2 if y<1e-12 or y>1-1e-12 else h)
        e=rho-ref(x,y,T); a+=m*abs(e); b+=m*abs(ref(x,y,T)); mx=max(mx,rho)
    return a/b,mx
T=1e-9
for g in (16,32,96):
    for p in (1,2,3,4):
        n=round(g/p); o=f"{tmp}/i"
        r=subprocess.run([exe,"--problem","crossing-study","--dimension","2","--pn-order","3","--degree",str(p),"--elements",str(n),"--time",str(T),"--sample-grid",str(g),"--output",o],env=env,capture_output=True,text=True)
        if r.returncode: print(g,p,"fail",r.stderr[:100]); continue
        e,mx=err(o+"/final.csv",g,T); print(f"grid={g} p={p} n={n}: initial L1 rel error={e:.4f} peak={mx:.3f}",flush=True)
