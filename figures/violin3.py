import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/Users/user/connectome/islands/agents/resident-a/compute/attn_intro/"; OUT="/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro/"
BLUE="#2B5FA8"; INK="#1D1D1F"
plt.rcParams.update({"font.family":"Helvetica","font.size":56,"text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"axes.edgecolor":INK})
def band(att,L): return np.array(att)[L].mean(0)
def boot(x,NB=10000,seed=0):
    rng=np.random.default_rng(seed); s=x[rng.integers(0,len(x),(NB,len(x)))].mean(1); return x.mean(),np.percentile(s,2.5),np.percentile(s,97.5)
res={}
for m,name,cond in [("qwen3-32b","Focus\non it","focus"),("qwen3-32b","Don't\nattend to it","ignore")]:
    meta=json.load(open(R+m+"/meta.json")); B=json.load(open(R+m+"/B.json")); L=meta["LAYERS"]
    by={c:{r["id"]:r for r in B if r["task"]=="arith" and r["cond"]==c} for c in [cond,"mention"]}
    ids=sorted(by[cond]); assert sorted(by["mention"])==ids
    assert all(by[cond][i]["k"]==by["mention"][i]["k"] for i in ids)
    x=np.array([band(by["mention"][i]["att_mean"],L)[by["mention"][i]["k"]] for i in ids])
    y=np.array([band(by[cond][i]["att_mean"],L)[by[cond][i]["k"]] for i in ids])
    d=y-x; z=d/x.std(ddof=1); mm,lo,hi=boot(z)
    res[name]=dict(z=z,mean=mm,lo=lo,hi=hi,n=len(z),above=int((d>0).sum()),dz=d.mean()/d.std(ddof=1))
    print(name,"n=%d meanSD=%.3f CI[%.3f,%.3f] above=%d (%.1f%%) d_z=%.3f sd_mention=%.3g"%(len(z),mm,lo,hi,res[name]["above"],100*res[name]["above"]/len(z),res[name]["dz"],x.std(ddof=1)))
fig,a=plt.subplots(figsize=(16,15))
from matplotlib import cm, colors
allz=np.concatenate([res[n]["z"] for n in res]); vm=np.percentile(np.abs(allz),97); norm=colors.Normalize(vmin=0,vmax=vm); cmap=colors.LinearSegmentedColormap.from_list("hy",["#BFD3EE","#5B8BD0","#2B4FA8","#4B2A8F","#2A0F4F"])
rng=np.random.default_rng(1)
for p,name in enumerate(res):
    r=res[name]; z=r["z"]
    v=a.violinplot([z],positions=[p],widths=0.95,showextrema=False)
    for b in v["bodies"]: b.set_facecolor(BLUE); b.set_alpha(0.22); b.set_edgecolor(BLUE); b.set_linewidth(2.5)
    xj=p+rng.uniform(-0.34,0.34,len(z)); a.scatter(xj,z,s=650,c=cmap(norm(np.abs(z))),alpha=0.45,edgecolors="none",zorder=3); a.scatter(xj,z,s=6,color="black",alpha=1,edgecolors="none",zorder=4)
    a.plot([p,p],[r["lo"],r["hi"]],color=INK,lw=12,solid_capstyle="butt",zorder=5)
    a.plot([p-0.22,p+0.22],[r["mean"]]*2,color=INK,lw=6,zorder=6)
    a.text(p,(z.max()+0.12) if r["mean"]>0 else (z.min()-0.12),f"{r['mean']:+.2f} SD",fontsize=84,fontweight="bold",color=INK,ha="center",va=("bottom" if r["mean"]>0 else "top"),zorder=7)
a.axhline(0,color=INK,lw=4,zorder=4)
a.set_xticks([0,1]); a.set_xticklabels(list(res),fontsize=64,fontweight="bold")
a.set_xlim(-0.5,1.5); a.set_ylim(allz.min()-0.75,allz.max()+0.9)
a.set_ylabel("attention change (SDs)",fontsize=84,labelpad=24)
for s in ["top","right"]: a.spines[s].set_visible(False)
a.spines["left"].set_linewidth(2.5); a.spines["bottom"].set_linewidth(2.5); a.tick_params(width=2.5,length=10,labelsize=72)
ylo,yhi=a.get_ylim(); a.text(1.48,0.06,"no change",ha="right",va="bottom",fontsize=56,fontweight="bold")
fig.tight_layout(); fig.savefig(OUT+"violin3.png",dpi=150,facecolor="white"); fig.savefig(OUT+"violin3.jpg",dpi=150,facecolor="white")
print(a.get_ylim())
