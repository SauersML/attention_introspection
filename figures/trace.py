import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/Users/user/connectome/islands/agents/resident-a/compute/attn_intro/qwen3-32b/results.json"
OUT="/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro/"
BLUE="#2B5FA8"; ORANGE="#E07B24"; INK="#1D1D1F"
plt.rcParams.update({"font.family":"Helvetica","font.size":56,"text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"axes.edgecolor":INK})
A=json.load(open(R))["A2"]
keys=[("arith_b1.0","2× more"),("arith_b2.0","5× more")]
fig,a=plt.subplots(figsize=(17,14))
x=np.arange(len(keys)); w=0.36
for j,(src,lab,col) in enumerate([("identical_answers_self_logp_shift","Qwen, asked\nabout itself",BLUE),("identical_answers_reader_logp_shift","Qwen, reading\nthe transcript",ORANGE)]):
    m=[A[k][src][0] for k,_ in keys]; lo=[A[k][src][1] for k,_ in keys]; hi=[A[k][src][2] for k,_ in keys]
    xs=x+(j-0.5)*w
    a.bar(xs,m,w*0.92,color=col)
    a.errorbar(xs,m,yerr=[np.array(m)-np.array(lo),np.array(hi)-np.array(m)],fmt="none",ecolor=INK,elinewidth=5,capsize=14,capthick=5)
    a.text(xs[-1]+0.02,(m[-1] if j==0 else 0)-0.02 if j==0 else -0.03,lab,color=col,fontsize=50,fontweight="bold",ha="center",va="top") if False else None
    for xi,mi,l_ in zip(xs,m,lo): a.text(xi,(l_-0.03) if mi<-0.02 else -0.03,f"{mi:+.2f}",ha="center",va="top",fontsize=52,fontweight="bold",color=col)
a.axhline(0,color=INK,lw=4)
a.set_xticks(x); a.set_xticklabels([l for _,l in keys],fontsize=64)
a.set_xlabel("attention forced onto a paragraph",fontsize=58,labelpad=16)
a.set_ylabel("change in log-prob of\nnaming that paragraph",fontsize=62,labelpad=16)
a.set_ylim(-0.62,0.3); a.set_xlim(-0.55,1.5)
a.text(1.45,0.27,"Qwen, asked about itself",color=BLUE,fontsize=50,fontweight="bold",va="top",ha="right")
a.text(1.45,0.17,"Qwen, reading the transcript",color=ORANGE,fontsize=50,fontweight="bold",va="top",ha="right")
for s in ["top","right","bottom"]: a.spines[s].set_visible(False)
a.spines["left"].set_linewidth(3); a.tick_params(width=3,length=12,labelsize=56); a.tick_params(axis="x",length=0)
fig.tight_layout(); fig.savefig(OUT+"trace.png",dpi=150,facecolor="white"); fig.savefig(OUT+"trace.jpg",dpi=150,facecolor="white")
for k,_ in keys: print(k,A[k]["identical_answers_self_logp_shift"],A[k]["identical_answers_reader_logp_shift"],A[k]["_n_identical"])
