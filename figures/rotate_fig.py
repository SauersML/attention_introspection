import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
R = json.load(open(sys.argv[1]))["att_mean"]
OUT = "/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro/"
INK = "#1D1D1F"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 80, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
sys.path.insert(0, OUT); from pal import DIV, norm
M = np.array(R["ratio"]); P = 100 * (M - 1); vm = np.abs(P).max(); nm = norm(-vm*0.6, vm)
fig = plt.figure(figsize=(18, 16.5)); a = fig.add_axes([0.21, 0.02, 0.77, 0.76])
a.imshow(P, cmap=DIV, norm=nm)
for k in range(4):
    for m in range(4):
        v = P[k, m]; dark = abs(v) > 0.5 * vm
        a.text(m, k, (f"{v:+.0f}%".replace("-","−") if round(v) else "0%"), ha="center", va="center", fontsize=84 if k == m else 74,
               fontweight="bold" if k == m else "normal", color="white" if dark else INK)
    a.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, lw=12, ec=INK, zorder=5))
a.set_xticks(range(4)); a.set_xticklabels([f"P{i}" for i in range(1, 5)])
a.set_yticks(range(4)); a.set_yticklabels([f"P{i}" for i in range(1, 5)])
a.xaxis.tick_top(); a.xaxis.set_label_position("top")
a.tick_params(length=0, pad=18, labelsize=84)
a.set_xlabel("change in attention", labelpad=28)
a.set_ylabel("“focus your attention on . . .”", labelpad=28, fontsize=70)
for s in a.spines.values(): s.set_visible(False)
fig.savefig(OUT + "rotate.png", dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.15); fig.savefig(OUT + "rotate.jpg", dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.15)
