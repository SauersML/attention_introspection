import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import colors
R="/Users/user/connectome/islands/agents/resident-a/compute/attn_intro/qwen3-32b/"
OUT="/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro/"
BLUE="#2B5FA8"; ORANGE="#E07B24"; INK="#1D1D1F"
plt.rcParams.update({"font.family":"Helvetica","font.size":60,"text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"axes.edgecolor":INK})
A=json.load(open(R+"A.json")); A2=json.load(open(R+"A2.json"))
byid={r["id"]:r for r in A}
def spec(d,r):
    others=[m for m in range(4) if m!=r["k"] and (r["task"]!="qa" or m!=r["j"])]
    return d[r["k"]]-np.mean([d[m] for m in others])
def boot(x,NB=10000,seed=0):
    rng=np.random.default_rng(seed); s=x[rng.integers(0,len(x),(NB,len(x)))].mean(1); return x.mean(),np.percentile(s,2.5),np.percentile(s,97.5)
cols=[]
for b in [1.0,2.0]:
    rs=[r for r in A2 if r["task"]=="arith" and r["b"]==b]; bs=[byid[r["id"]] for r in rs]
    same=[r["ans"].strip()==b0["ans"].strip() for r,b0 in zip(rs,bs)]
    ls=np.array([spec(np.array(r["rep_lp"])-np.array(b0["rep_lp"]),r) for r,b0,s in zip(rs,bs,same) if s])
    lr=np.array([spec(np.array(r["reader_lp"])-np.array(b0["reader_lp"]),r) for r,b0,s in zip(rs,bs,same) if s])
    cols.append((b,ls,lr)); print(b,len(ls),boot(ls),boot(lr))
fig,a=plt.subplots(figsize=(30,15))
cm=colors.LinearSegmentedColormap.from_list("h",["#BFD3EE","#5B8BD0","#2B4FA8","#4B2A8F","#2A0F4F"])
allv=np.concatenate([c[1] for c in cols]); nm=colors.Normalize(0,np.percentile(np.abs(allv),97))
rng=np.random.default_rng(2)
pos=[0,1.2,2.8,4.0]; labs=[]
for i,(b,ls,lr) in enumerate(cols):
    for j,(v,col,name) in enumerate([(ls,BLUE,"asked about\nitself"),(lr,ORANGE,"reading the\ntranscript")]):
        p=pos[2*i+j]
        vv=a.violinplot([v],positions=[p],widths=0.95,showextrema=False)
        for bb in vv["bodies"]: bb.set_facecolor(col); bb.set_alpha(0.2); bb.set_edgecolor(col); bb.set_linewidth(2.5)
        xj=p+rng.uniform(-0.33,0.33,len(v))
        c=cm(nm(np.abs(v))) if j==0 else [ORANGE]*len(v)
        a.scatter(xj,v,s=600,c=c,alpha=0.45,edgecolors="none",zorder=3); a.scatter(xj,v,s=6,color="black",zorder=4)
        m,lo,hi=boot(v); a.plot([p,p],[lo,hi],color=INK,lw=12,solid_capstyle="butt",zorder=5); a.plot([p-0.22,p+0.22],[m,m],color=INK,lw=6,zorder=6)
        a.text(p,v.min()-0.12 if m<-0.05 else v.max()+0.12,f"{m:+.2f}",ha="center",va="top" if m<-0.05 else "bottom",fontsize=72,fontweight="bold",color=col)
        labs.append((p,name,col))
a.axhline(0,color=INK,lw=4,zorder=2)
a.set_xticks([p for p,_,_ in labs]); a.set_xticklabels([n for _,n,_ in labs],fontsize=50,fontweight="bold")
for t,(_,_,c) in zip(a.get_xticklabels(),labs): t.set_color(c)
for x,txt in [(0.6,"attention forced ×2"),(3.4,"attention forced ×5")]:
    a.text(x,allv.max()+0.75,txt,ha="center",va="top",fontsize=64,fontweight="bold")
a.set_ylabel("change in log-prob of naming\nthe boosted paragraph",fontsize=60,labelpad=12)
a.set_ylim(allv.min()-0.9,allv.max()+0.8); a.set_xlim(-0.65,4.65)

for s in ["top","right"]: a.spines[s].set_visible(False)
a.spines["left"].set_linewidth(4); a.spines["bottom"].set_linewidth(4); a.tick_params(width=4,length=16,labelsize=64); a.tick_params(axis="x",length=0)
fig.tight_layout(pad=0.6); fig.savefig(OUT+"trace2.png",dpi=100,facecolor="white"); fig.savefig(OUT+"trace2.jpg",dpi=100,facecolor="white")
