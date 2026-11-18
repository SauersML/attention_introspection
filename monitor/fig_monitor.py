# One figure: self-report minus transcript reader, per setup, with 95% CIs.
# usage: fig_monitor.py results.json OUT.png "TITLE (the finding)"
import json, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.abspath(__file__)) + "/../figures"]
from pal import PURPLE, RED, INK

R = json.load(open(sys.argv[1]))
SETUPS = [("after", w) for w in ["most", "least", "rank", "pct", "rate", "free"]] + \
         [("before", w) for w in ["most", "rate", "pct"]] + [("later", w) for w in ["most", "rate", "pct"]] + \
         [("think", w) for w in ["most", "pct"]] + [("fb4", w) for w in ["most", "rate", "pct"]] + [("fb16", w) for w in ["most", "rate", "pct"]]
GROUP = {"after": "Asked right after answering", "before": "Asked before answering", "later": "Asked one turn later", "think": "Thinking on",
         "fb4": "First shown 4 true measurements", "fb16": "First shown 16 true measurements"}
WN = {"most": "which most?", "least": "which least?", "rank": "rank them", "pct": "percent each", "rate": "rate each 0 to 10", "free": "describe freely"}

plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 30, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
fig, axes = plt.subplots(1, 2, figsize=(24, 23), sharey=True, gridspec_kw=dict(wspace=0.12, left=0.25, right=0.98, top=0.9, bottom=0.1))
ys, heads, y, prev = [], [], 0.0, None
for su in SETUPS:
    if su[0] != prev:
        if prev is not None: y += 0.5
        heads.append((y, GROUP[su[0]])); y += 1.0
    ys.append(y); y += 1.0; prev = su[0]
ys = np.array(ys)
panels = [(axes[0], "top1", 100, "Picking the right paragraph\nself minus reader (% points)"),
          (axes[1], "rho", 1, "Ranking the four paragraphs\nself minus reader (rank correlation)")]
for ax, m, scale, xl in panels:
    for yy, su in zip(ys, SETUPS):
        r = R.get("|".join(su))
        if r is None: continue
        d = r["self-reader"][m]
        mu = d["mean"] * scale; lo, hi = [v * scale for v in d["0.05"]]
        ax.barh(yy, mu, height=0.75, color=PURPLE if mu > 0 else RED)
        ax.plot([lo, hi], [yy, yy], color=INK, lw=4, solid_capstyle="butt")
        for v in (lo, hi): ax.plot([v, v], [yy - 0.22, yy + 0.22], color=INK, lw=4)
    ax.axvline(0, color=INK, lw=3)
    ax.set_xlabel(xl, fontsize=32, labelpad=16)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=30, width=2, length=10)
    lim = max(abs(np.array(ax.get_xlim()))); ax.set_xlim(-lim, lim)
    ax.text(-lim, -1.4, "reader better", color=RED, fontsize=32, ha="left", va="center", fontweight="bold")
    ax.text(lim, -1.4, "self better", color=PURPLE, fontsize=32, ha="right", va="center", fontweight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p, s=scale: (f"{v:.0f}" if s == 100 else f"{v:.1f}").replace("-", "\u2212")))
axes[0].set_yticks(ys); axes[0].set_yticklabels([WN[su[1]] for su in SETUPS], fontsize=30)
tr = blended_transform_factory(fig.transFigure, axes[0].transData)
for yy, h in heads: fig.text(0.01, yy, h, transform=tr, fontsize=31, fontweight="bold", ha="left", va="center")
axes[0].set_ylim(ys[-1] + 0.8, -2.0)
fig.suptitle(sys.argv[3], fontsize=40, fontweight="bold", x=0.5, y=0.965)
fig.savefig(sys.argv[2], dpi=100)
fig.savefig(sys.argv[2].replace(".png", ".pdf"))
print("saved", sys.argv[2])
