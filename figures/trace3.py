"""Steered vs unsteered: change in log-prob that Qwen3-32B names the boosted paragraph when asked where it attended.
Paired per question; p from a sign-flip permutation test (2,000,000 random flips, +1 correction), no distribution assumed."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/Users/user/connectome/islands/agents/resident-a/compute/attn_intro/qwen3-32b/"
OUT="/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro/"
sys.path.insert(0,OUT); from pal import PURPLE, RED, INK, DIV, norm as dnorm
plt.rcParams.update({"font.family":"Helvetica","font.size":60,"text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"axes.edgecolor":INK})
A=json.load(open(R+"A.json")); A2=json.load(open(R+"A2.json")); byid={r["id"]:r for r in A}
def spec(d,r):
    o=[m for m in range(4) if m!=r["k"] and (r["task"]!="qa" or m!=r["j"])]; return d[r["k"]]-np.mean([d[m] for m in o])
def boot(x,NB=20000):
    g=np.random.default_rng(0); s=x[g.integers(0,len(x),(NB,len(x)))].mean(1); return x.mean(),np.percentile(s,2.5),np.percentile(s,97.5)
def signflip(x,N=2_000_000,chunk=200_000):
    g=np.random.default_rng(1); obs=abs(x.mean()); hit=0
    for _ in range(N//chunk):
        s=g.choice([-1.0,1.0],size=(chunk,len(x))); hit+=int((np.abs((s*x).mean(1))>=obs-1e-15).sum())
    return (hit+1)/(N+1)
cols=[]
for b,lab in [(1.0,"attention steered\n×2.3"),(2.0,"attention steered\n×4.9")]:
    rs=[r for r in A2 if r["task"]=="arith" and r["b"]==b]; bs=[byid[r["id"]] for r in rs]
    x=np.array([spec(np.array(r["rep_lp"])-np.array(b0["rep_lp"]),r) for r,b0 in zip(rs,bs) if r["ans"].strip()==b0["ans"].strip()])
    p=signflip(x); cols.append((lab,x,p)); print(lab.replace("\n"," "),len(x),boot(x),"p",p)
fig,a=plt.subplots(figsize=(22,15)); rng=np.random.default_rng(2)
allv=np.concatenate([c[1] for c in cols]); DN=dnorm(np.percentile(allv,1),max(np.percentile(allv,99),1e-3))
for i,(lab,x,p) in enumerate(cols):
    m,lo,hi=boot(x); col=PURPLE if m>0 else RED
    vv=a.violinplot([x],positions=[i],widths=0.85,showextrema=False)
    for bb in vv["bodies"]: bb.set_facecolor(col); bb.set_alpha(0.2); bb.set_edgecolor(col); bb.set_linewidth(2.5)
    xj=i+rng.uniform(-0.3,0.3,len(x)); a.scatter(xj,x,s=600,c=DIV(DN(x)),alpha=0.5,edgecolors="none",zorder=3); a.scatter(xj,x,s=8,color="black",zorder=4)
    a.plot([i,i],[lo,hi],color=INK,lw=12,solid_capstyle="butt",zorder=5); a.plot([i-0.2,i+0.2],[m,m],color=INK,lw=6,zorder=6)
    ps="p < 10⁻⁶" if p<1e-6 else (f"p = {p:.1e}" if p<1e-3 else f"p = {p:.3f}")
    a.text(i,x.min()-0.15,f"{m:+.2f}".replace("-","−")+f"\n{ps}",ha="center",va="top",fontsize=62,fontweight="bold",color=col,linespacing=1.15)
a.axhline(0,color=INK,lw=4,zorder=2)
a.set_xticks([0,1]); a.set_xticklabels([c[0] for c in cols],fontsize=62,fontweight="bold",linespacing=1.1)
a.set_ylabel("change in log-prob of naming\nthe boosted paragraph",fontsize=56,labelpad=14)
a.set_ylim(allv.min()-1.6,allv.max()+0.3); a.set_xlim(-0.6,1.6)
for s in ["top","right"]: a.spines[s].set_visible(False)
a.spines["left"].set_linewidth(4); a.spines["bottom"].set_linewidth(4); a.tick_params(width=4,length=16,labelsize=60); a.tick_params(axis="x",length=0)
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f"{v:+.1f}".replace("-","−") if v else "0"))
fig.tight_layout(pad=0.6); fig.savefig(OUT+"trace3.png",dpi=100,facecolor="white"); fig.savefig(OUT+"trace3.jpg",dpi=100,facecolor="white")
