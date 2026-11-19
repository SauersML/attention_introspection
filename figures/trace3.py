"""Steered vs unsteered: change in log-prob that Qwen3-32B names the boosted paragraph when asked where it attended,
minus the mean change for the other paragraphs (the question paragraph excluded in QA). Arithmetic prompts, answers identical to no boost.
Paired per question; p from a sign-flip permutation test (2,000,000 random flips, +1 correction), no distribution assumed.
Drawn on the log-prob scale, labelled as % change = exp(.) - 1 (relative to the other paragraphs); black line and big label = the mean;
dots coloured by their % change on the shared scale (pal.pct_colour). Labels [P1]..[P4] in slot order and options in order (attn_intro.py)."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
OUT=os.path.dirname(os.path.abspath(__file__))+"/"; R=os.path.join(OUT,"..","results","qwen3-32b")+"/"
sys.path.insert(0,OUT); from pal import INK, pct_colour
import math
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
for b,lab in [(1.0,"attention on the\nparagraph ×2.3"),(2.0,"attention on the\nparagraph ×4.9")]:
    rs=[r for r in A2 if r["task"]=="arith" and r["b"]==b]; bs=[byid[r["id"]] for r in rs]
    x=np.array([spec(np.array(r["rep_lp"])-np.array(b0["rep_lp"]),r) for r,b0 in zip(rs,bs) if r["ans"].strip()==b0["ans"].strip()])
    p=signflip(x); cols.append((lab,x,p)); print(lab.replace("\n"," "),len(x),boot(x),"p",p)
fig,a=plt.subplots(figsize=(22,15)); rng=np.random.default_rng(2); COL="#2B5FA8"   # the answer-time boost colour of privileged.png
neg=lambda t: t.replace("-","\u2212")
allv=np.concatenate([c[1] for c in cols])
for i,(lab,x,p) in enumerate(cols):
    m,lo,hi=boot(x); print(lab.replace("\n"," "),f"{100*(math.exp(m)-1):+.1f}% [{100*(math.exp(lo)-1):+.1f}, {100*(math.exp(hi)-1):+.1f}]")
    vv=a.violinplot([x],positions=[i],widths=0.85,showextrema=False)
    for bb in vv["bodies"]: bb.set_facecolor(COL); bb.set_alpha(0.16); bb.set_edgecolor(COL); bb.set_linewidth(2.5)
    xj=i+rng.uniform(-0.3,0.3,len(x)); a.scatter(xj,x,s=600,c=pct_colour(np.exp(x)),alpha=0.35,edgecolors="none",zorder=3); a.scatter(xj,x,s=8,color="black",zorder=4)
    a.plot([i-0.22,i+0.22],[m,m],color=INK,lw=6,zorder=6,solid_capstyle="butt")
    ps="p < 0.0001" if p<1e-4 else f"p = {p:.4f}"
    a.text(i,x.min()-0.15,neg(f"{100*(math.exp(m)-1):+.0f}%")+f"\n{ps}",ha="center",va="top",fontsize=62,fontweight="bold",color=COL,linespacing=1.15)
a.axhline(0,color=INK,lw=4,zorder=2)
a.set_xticks([0,1]); a.set_xticklabels([c[0] for c in cols],fontsize=62,fontweight="bold",linespacing=1.1)
a.set_ylabel("chance the report names the boosted\nparagraph, change relative to the others",fontsize=46,labelpad=14)
a.set_ylim(allv.min()-1.6,allv.max()+0.3); a.set_xlim(-0.6,1.6)
TV=[-94,-75,0,300,1500]   # x1/16, x1/4, x1, x4, x16: evenly spaced on the log scale
a.set_yticks([math.log(1+t/100) for t in TV]); a.set_yticklabels([neg(f"{t:+d}%") if t else "0%" for t in TV])
for s in ["top","right"]: a.spines[s].set_visible(False)
a.spines["left"].set_linewidth(4); a.spines["bottom"].set_linewidth(4); a.tick_params(width=4,length=16,labelsize=54); a.tick_params(axis="x",length=0)
fig.tight_layout(pad=0.6); fig.savefig(OUT+"trace3.png",dpi=100,facecolor="white"); fig.savefig(OUT+"trace3.jpg",dpi=100,facecolor="white")
