import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
R = json.load(open(sys.argv[1]))["att_mean"]
OUT = "/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro/"
INK = "#1D1D1F"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 60, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
M = np.array(R["ratio"]); L = np.log(M)
vm = np.abs(L).max()
cmap = colors.LinearSegmentedColormap.from_list("d", ["#B03A48", "#FFFFFF", "#2B4FA8"])
fig = plt.figure(figsize=(18, 18.5)); a = fig.add_axes([0.19, 0.02, 0.77, 0.70])
a.imshow(L, cmap=cmap, vmin=-vm, vmax=vm)
for k in range(4):
    for m in range(4):
        v = M[k, m]; dark = abs(L[k, m]) > 0.55 * vm
        a.text(m, k, f"{v:.2f}×", ha="center", va="center", fontsize=66 if k == m else 58,
               fontweight="bold" if k == m else "normal", color="white" if dark else INK)
    a.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, lw=12, ec=INK, zorder=5))
a.set_xticks(range(4)); a.set_xticklabels([f"P{i}" for i in range(1, 5)])
a.set_yticks(range(4)); a.set_yticklabels([f"P{i}" for i in range(1, 5)])
a.xaxis.tick_top(); a.xaxis.set_label_position("top")
a.tick_params(length=0, pad=18)
a.set_xlabel("paragraph where attention went", labelpad=28)
a.set_ylabel("paragraph named in the instruction", labelpad=28)
for s in a.spines.values(): s.set_visible(False)
fig.text(0.5, 0.955, "Qwen3-32B is told:\n“Focus your attention on paragraph [P?].”", ha="center", va="center", fontsize=46, linespacing=1.3)
fig.text(0.5, 0.885, "cell = attention while answering ÷ attention with no instruction", ha="center", va="center", fontsize=42)
fig.savefig(OUT + "rotate.png", dpi=150, facecolor="white"); fig.savefig(OUT + "rotate.jpg", dpi=150, facecolor="white")
