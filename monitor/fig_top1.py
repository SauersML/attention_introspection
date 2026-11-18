# Top-1 accuracy at naming the most-attended paragraph: self-report vs transcript reader vs text classifier, per setup.
# usage: fig_top1.py results.json OUT.png
import json, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.abspath(__file__)) + "/../figures"]
from pal import PURPLE, INK
GREY = "#8E8E93"

R = json.load(open(sys.argv[1]))
SETUPS = [("after", w) for w in ["most", "least", "rank", "pct", "rate", "free"]] + \
         [("before", w) for w in ["most", "rate", "pct"]] + [("later", w) for w in ["most", "rate", "pct"]] + \
         [("think", w) for w in ["most", "pct"]] + [("fb4", w) for w in ["most", "rate", "pct"]] + [("fb16", w) for w in ["most", "rate", "pct"]]
SETUPS = [s for s in SETUPS if R.get("|".join(s))]
GROUP = {"after": "Asked right after answering", "before": "Asked before answering", "later": "Asked one turn later", "think": "Thinking on",
         "fb4": "First shown 4 true measurements", "fb16": "First shown 16 true measurements"}
WN = {"most": "which most?", "least": "which least?", "rank": "rank them", "pct": "percent each", "rate": "rate each 0 to 10", "free": "describe freely"}

plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 34, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
fig, ax = plt.subplots(figsize=(22, 24))
fig.subplots_adjust(left=0.30, right=0.97, top=0.95, bottom=0.08)
ys, heads, y, prev = [], [], 0.0, None
for su in SETUPS:
    if su[0] != prev:
        if prev is not None: y += 0.4
        heads.append((y, GROUP[su[0]])); y += 1.0
    ys.append(y); y += 1.25; prev = su[0]
ys = np.array(ys)
WHO = [("self", PURPLE, -0.3, "self-report"), ("reader", INK, 0.0, "same model reading the transcript"), ("clf", GREY, 0.3, "classifier on the transcript text")]
for yy, su in zip(ys, SETUPS):
    r = R["|".join(su)]
    for who, col, dy, _ in WHO:
        d = r[who]["top1"]; m = 100 * d["mean"]; lo, hi = [100 * v for v in d["0.05"]]
        ax.plot([lo, hi], [yy + dy] * 2, color=col, lw=5, solid_capstyle="round")
        ax.plot(m, yy + dy, "o", ms=20, color=col, mec="white", mew=2, zorder=3)
ax.axvline(25, color=INK, lw=3, ls=(0, (6, 5)))
ax.text(24.2, ys[-1] + 0.75, "chance", fontsize=34, va="center", ha="right")
# direct labels in their own rows above the data
for k, (who, col, dy, lab) in enumerate(WHO):
    yl = -3.6 + 0.85 * k
    ax.plot([27, 33], [yl, yl], color=col, lw=5, solid_capstyle="round")
    ax.plot(30, yl, "o", ms=20, color=col, mec="white", mew=2)
    ax.text(34.5, yl, lab, color=col, fontsize=34, ha="left", va="center", fontweight="bold")
ax.set_xlim(0, 101)
ax.set_xlabel("Picks the paragraph that truly got the most attention \n(least-attended, for \u201cwhich least?\u201d), % of contexts", fontsize=32, labelpad=16)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}%"))
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_linewidth(2)
ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=34, width=2, length=10)
ax.set_yticks(ys); ax.set_yticklabels([WN[su[1]] for su in SETUPS], fontsize=34)
tr = blended_transform_factory(fig.transFigure, ax.transData)
for yy, h in heads: fig.text(0.01, yy, h, transform=tr, fontsize=34, fontweight="bold", ha="left", va="center")
ax.set_ylim(ys[-1] + 1.2, -4.2)
fig.savefig(sys.argv[2], dpi=100)
fig.savefig(sys.argv[2].replace(".png", ".pdf"))
print("saved", sys.argv[2], "setups", len(SETUPS))
