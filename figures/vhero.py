import json, math, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/Users/user/connectome/islands/agents/resident-a/compute/attn_intro/"
OUT="/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro/"
import sys; sys.path.insert(0,"/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro"); from pal import PURPLE, RED, INK, DIV, norm as dnorm
BLUE=PURPLE; ORANGE=RED
plt.rcParams.update({"font.family":"Helvetica","font.size":30,"text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"axes.edgecolor":INK})
def load(m):
    meta=json.load(open(R+m+"/meta.json")); A=json.load(open(R+m+"/A.json")); B=json.load(open(R+m+"/B.json"))
    return meta, {r["id"]:r for r in A}, B
def band(att, L): return np.array(att)[L].mean(0)
def boot(x, NB=10000, seed=0):
    rng=np.random.default_rng(seed); x=np.asarray(x); i=rng.integers(0,len(x),(NB,len(x))); s=x[i].mean(1)
    return x.mean(), np.percentile(s,2.5), np.percentile(s,97.5)
data={}; report={}
for m,name in [("qwen3-8b","Qwen3-8B"),("qwen3-32b","Qwen3-32B")]:
    meta,base,B=load(m); L=meta["LAYERS"]
    by={c:{r["id"]:r for r in B if r["task"]=="arith" and r["cond"]==c} for c in ["focus","ignore","mention"]}
    for c in ["focus","ignore"]:
        ids=sorted(by[c]); assert sorted(by["mention"])==ids
        x=np.array([band(by["mention"][i]["att_mean"],L)[by["mention"][i]["k"]] for i in ids])
        assert all(by[c][i]["k"]==by["mention"][i]["k"] for i in ids)
        y=np.array([band(by[c][i]["att_mean"],L)[by[c][i]["k"]] for i in ids])
        mm,lo,hi=boot(np.log(y)-np.log(x))
        data[(name,c)]=(x,y,math.exp(mm),math.exp(lo),math.exp(hi)); report[(name,c)]=(len(ids),math.exp(mm),math.exp(lo),math.exp(hi))
    if m=="qwen3-32b":
        ids=sorted(by["focus"]); NL=meta["NL"]
        lr=np.array([[math.log(np.array(by["focus"][i]["att_mean"])[l][by["focus"][i]["k"]]/np.array(base[i]["att_mean"])[l][by["focus"][i]["k"]]) for l in range(NL)] for i in ids])
        lay=[boot(lr[:,l],seed=l) for l in range(NL)]
        layband=L
for k,v in report.items(): print(k, "n=%d %.3f [%.3f, %.3f]"%v)
# ---------- scatter
allv=np.concatenate([np.r_[d[0],d[1]] for d in data.values()]); lo_,hi_=allv.min()/1.4, allv.max()*1.4
print("range",allv.min(),allv.max())
fig,ax=plt.subplots(2,2,figsize=(16,15.5),sharex=True,sharey=True)
for r,name in enumerate(["Qwen3-8B","Qwen3-32B"]):
    for c,(cond,col) in enumerate([("focus",BLUE),("ignore",ORANGE)]):
        a=ax[r,c]; x,y,mm,lo,hi=data[(name,cond)]
        a.plot([lo_,hi_],[lo_,hi_],color=INK,lw=2.5,ls="--",zorder=1)
        a.scatter(x,y,s=110,color=col,alpha=0.45,edgecolors="none",zorder=2)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo_,hi_); a.set_ylim(lo_,hi_); a.set_aspect("equal")
        for s in ["top","right"]: a.spines[s].set_visible(False)
        a.spines["left"].set_linewidth(2); a.spines["bottom"].set_linewidth(2); a.tick_params(width=2,length=8,labelsize=28)
        a.text(0.04,0.96,f"{mm:.2f}×",transform=a.transAxes,fontsize=58,fontweight="bold",color=col,va="top")
        a.text(0.04,0.80,f"95% CI {lo:.2f}–{hi:.2f}",transform=a.transAxes,fontsize=32,color=col,va="top")
        a.text(0.96,0.05,name,transform=a.transAxes,fontsize=44,fontweight="bold",color=INK,ha="right",va="bottom")
        a.text(0.80,0.865,"no change",transform=a.transAxes,fontsize=30,color=INK,ha="center",va="bottom",rotation=45,rotation_mode="anchor")
        tk=[3e-5,1e-4,3e-4]; a.set_xticks(tk); a.set_yticks(tk); a.minorticks_off()
        a.set_xticklabels(["0.00003","0.0001","0.0003"]); a.set_yticklabels(["0.00003","0.0001","0.0003"])
        if r==0: a.set_title({"focus":"“Focus your attention on\nparagraph X”","ignore":"“Do not pay attention to\nparagraph X”"}[cond],fontsize=36,color=col,pad=18)
fig.supxlabel("Attention on paragraph X when paragraph X is only mentioned",fontsize=34,y=0.012)
fig.supylabel("Attention on paragraph X with the instruction",fontsize=34,x=0.012)
fig.subplots_adjust(left=0.17,right=0.97,top=0.90,bottom=0.08,wspace=0.06,hspace=0.08)
pass #,dpi=150,facecolor="white"); pass #,dpi=150,facecolor="white")

import numpy as np, matplotlib.ticker
from matplotlib import colors
res={}
for m,name,cond in [("qwen3-32b","“focus your\nattention”","focus"),("qwen3-32b","“do not pay\nattention”","ignore")]:
    meta=json.load(open(R+m+"/meta.json")); B=json.load(open(R+m+"/B.json")); L=meta["LAYERS"]
    by={c:{r["id"]:r for r in B if r["task"]=="arith" and r["cond"]==c} for c in [cond,"mention"]}
    ids=sorted(by[cond]); assert sorted(by["mention"])==ids
    assert all(by[cond][i]["k"]==by["mention"][i]["k"] for i in ids)
    x=np.array([band(by["mention"][i]["att_mean"],L)[by["mention"][i]["k"]] for i in ids])
    y=np.array([band(by[cond][i]["att_mean"],L)[by[cond][i]["k"]] for i in ids])
    d=y-x; z=100*(y/x-1); mm,lo,hi=boot(z)
    res[name]=dict(z=z,mean=mm,lo=lo,hi=hi,n=len(z),above=int((d>0).sum()),dz=d.mean()/d.std(ddof=1))
    print(name,"n=%d meanSD=%.3f CI[%.3f,%.3f] above=%d (%.1f%%) d_z=%.3f sd_mention=%.3g"%(len(z),mm,lo,hi,res[name]["above"],100*res[name]["above"]/len(z),res[name]["dz"],x.std(ddof=1)))

plt.rcParams.update({"font.size":56})
fig,a=plt.subplots(figsize=(23,8.2)); b=None
cmap=colors.LinearSegmentedColormap.from_list("hy",["#BFD3EE","#5B8BD0","#2B4FA8","#4B2A8F","#2A0F4F"])
allz=np.concatenate([res[n]["z"] for n in res]); vm=np.percentile(np.abs(allz),97); DN=dnorm(np.percentile(allz,1),vm)
rng=np.random.default_rng(1)
for p,name in enumerate(res):
    r=res[name]; z=r["z"]
    vv=a.violinplot([z],positions=[p*1.35],widths=0.7,showextrema=False)
    col=PURPLE if np.mean(z)>0 else RED
    for bb in vv["bodies"]: bb.set_facecolor(col); bb.set_alpha(0.22); bb.set_edgecolor(col); bb.set_linewidth(2.5)
    xj=p*1.35+rng.uniform(-0.25,0.25,len(z)); a.scatter(xj,z,s=650,c=DIV(DN(z)),alpha=0.45,edgecolors="none",zorder=3); a.scatter(xj,z,s=6,color="black",zorder=4)
    a.plot([p*1.35,p*1.35],[r["lo"],r["hi"]],color=INK,lw=12,solid_capstyle="butt",zorder=5); a.plot([p*1.35-0.22,p*1.35+0.22],[r["mean"]]*2,color=INK,lw=6,zorder=6)
    a.text(p*1.35,(z.max()+2) if r["mean"]>0 else (z.min()-2),f"{r['mean']:+.0f}%".replace("-","−"),fontsize=100,fontweight="bold",color=(PURPLE if r["mean"]>0 else RED),ha="center",va=("bottom" if r["mean"]>0 else "top"),zorder=7)
a.set_xticks([]); 
a.set_ylim(allz.min()-34,allz.max()+22)
for ax in (a,):
    ax.axhline(0,color=INK,lw=4,zorder=4)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f"{v:+.0f}%".replace("-","\u2212") if v else "0%"))
    for s_ in ["top","right"]: ax.spines[s_].set_visible(False)
    ax.spines["left"].set_linewidth(3); ax.spines["bottom"].set_linewidth(3); ax.tick_params(width=4,length=16,labelsize=72)
a.set_ylabel("change in\nattention",fontsize=70,labelpad=6); a.set_yticks([0,50]); a.set_xlim(-0.42,1.90)
fig.tight_layout(pad=0.6); fig.savefig(OUT+"vhero.png",dpi=110,facecolor="white"); fig.savefig(OUT+"vhero.jpg",dpi=110,facecolor="white")
