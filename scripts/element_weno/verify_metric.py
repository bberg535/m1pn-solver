import csv,math,sys
t=sys.argv[1]; W,BG,T=.07,.001,.32
def ref(x,y):
    p=lambda a,b: math.exp(-(a*a+b*b)/(2*W*W)); return BG+p(x-T-.3,y-.5)+p(x-.5,y-T-.3)
m=[float(r["mass"]) for r in csv.DictReader(open(t+"/nodal_mass.csv"))]
a=b=c=d=0
with open(t+"/final.csv") as f:
    rd=csv.reader(f); h=next(rd); u=h.index("u0")
    for k,r in enumerate(rd):
        rho=math.sqrt(4*math.pi)*float(r[u]); x,y=float(r[h.index("x")]),float(r[h.index("y")]); e=rho-ref(x,y); R=ref(x,y)
        a+=m[k]*abs(e); b+=m[k]*abs(R); c+=m[k]*e*e; d+=m[k]*R*R
print("L1",a/b,"L2",math.sqrt(c/d))
