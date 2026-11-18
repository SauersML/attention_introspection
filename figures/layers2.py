import json, math, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/Users/user/connectome/islands/agents/resident-a/compute/attn_intro/"
OUT="/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro/"
BLUE="#2B5FA8"; ORANGE="#E07B24"; INK="#1D1D1F"
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
pass #fig.savefig(OUT+"scatter.png",dpi=150,facecolor="white"); pass #fig.savefig(OUT+"scatter.jpg",dpi=150,facecolor="white")
# ---------- layers v2
plt.rcParams.update({"font.size":56})
fig,a=plt.subplots(figsize=(18,13))
l=np.arange(len(lay)); m=np.exp([t[0] for t in lay]); lo=np.exp([t[1] for t in lay]); hi=np.exp([t[2] for t in lay])
a.fill_between(l,lo,hi,color=BLUE,alpha=0.25,lw=0); a.plot(l,m,color=BLUE,lw=6)
a.axhline(1,color=INK,lw=3.5); a.text(63,0.975,"no change",ha="right",va="top",fontsize=52,fontweight="bold")
a.set_yscale("log"); a.set_yticks([1,1.5,2,2.5]); a.set_yticklabels(["1×","1.5×","2×","2.5×"]); a.set_ylim(0.8,4.3); a.minorticks_off()
a.set_xlim(-1,64); a.set_xticks([0,16,32,48,63]); a.set_xlabel("layer",fontsize=72,labelpad=14)
a.set_ylabel("attention on that paragraph",fontsize=60,labelpad=18)
i=int(np.argmax(m)); a.text(i+1.6,m[i]*0.99,f"{m[i]:.1f}×",fontsize=72,color=BLUE,fontweight="bold",va="center")
a.text(1,4.2,"“focus your attention”\nvs no instruction",fontsize=48,color=BLUE,fontweight="bold",va="top")
for s_ in ["top","right"]: a.spines[s_].set_visible(False)
a.spines["left"].set_linewidth(3); a.spines["bottom"].set_linewidth(3); a.tick_params(width=3,length=12,labelsize=60)
fig.tight_layout(); fig.savefig(OUT+"layers2.png",dpi=150,facecolor="white"); fig.savefig(OUT+"layers2.jpg",dpi=150,facecolor="white")
