"""Two 4x4 heatmaps (focus | completely ignore): % change in attention on each measured paragraph vs the mention-only line naming the same paragraph.
Data: rotate2_results.json from rotate2_analyze.py."""
import json, sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import DIV, norm, PURPLE, RED, INK
O = json.load(open(sys.argv[1]))
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
P = {c: np.array(O[c]["mean"]) for c in ["focus", "ignore"]}
vm = max(np.abs(P["focus"]).max(), np.abs(P["ignore"]).max()); nm = norm(-vm, vm)
fig = plt.figure(figsize=(30, 17))
axs = [fig.add_axes([0.075, 0.02, 0.43, 0.74]), fig.add_axes([0.555, 0.02, 0.43, 0.74])]
for a, c, title, col in zip(axs, ["focus", "ignore"], ["focus your attention", "completely ignore"], [PURPLE, RED]):
    M = P[c]; a.imshow(M, cmap=DIV, norm=nm)
    for k in range(4):
        for m in range(4):
            v = M[k, m]; dark = abs(v) > 0.45 * vm
            a.text(m, k, f"{v:+.0f}%".replace("-", "\u2212") if round(v) else "0%", ha="center", va="center", fontsize=72 if k == m else 62,
                   fontweight="bold" if k == m else "normal", color="white" if dark else INK)
        a.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, lw=10, ec=INK, zorder=5))
    a.set_xticks(range(4)); a.set_xticklabels([f"P{i}" for i in range(1, 5)]); a.set_yticks(range(4)); a.set_yticklabels([f"P{i}" for i in range(1, 5)])
    a.xaxis.tick_top(); a.xaxis.set_label_position("top"); a.tick_params(length=0, pad=14, labelsize=66)
    a.set_xlabel("measured paragraph", fontsize=66, labelpad=20)
    if c == "focus": a.set_ylabel("named paragraph", fontsize=66, labelpad=16)
    a.set_title(title, fontsize=84, fontweight="bold", color=col, pad=40)
    for s in a.spines.values(): s.set_visible(False)
out = os.path.join(HERE, "rotate2")
fig.savefig(out + ".png", dpi=110, facecolor="white", bbox_inches="tight", pad_inches=0.12); fig.savefig(out + ".jpg", dpi=110, facecolor="white", bbox_inches="tight", pad_inches=0.12)
