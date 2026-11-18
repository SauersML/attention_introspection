# One figure: self-report minus transcript reader, per setup, with 95% CIs. usage: fig_monitor.py results.json OUT.png "TITLE (the finding)"
import json, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.abspath(__file__)) + "/../figures"]
from pal import PURPLE, RED, INK

R = json.load(open(sys.argv[1]))
SETUPS = [("after", w) for w in ["most", "least", "rank", "pct", "rate", "free"]] + \
         [("before", w) for w in ["most", "rate", "pct"]] + [("later", w) for w in ["most", "rate", "pct"]] + \
         [("think", w) for w in ["most", "pct"]] + [("fb4", w) for w in ["most", "rate", "pct"]] + [("fb16", w) for w in ["most", "rate", "pct"]]
GROUP = {"after": "Right after answering", "before": "Before answering", "later": "One turn later", "think": "Thinking on",
         "fb4": "Shown 4 true measurements", "fb16": "Shown 16 true measurements"}
WN = {"most": "which most?", "least": "which least?", "rank": "rank them", "pct": "percent each", "rate": "rate 0 to 10", "free": "free text"}

plt.rcParams.update({"font.family": "Helvetica", "font.size": 30, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(23, 21), sharey=True, gridspec_kw=dict(wspace=0.08))
ys, labels, y = [], [], 0
gaps = []
prev = None
for su in SETUPS:
    if prev is not None and su[0] != prev: y += 0.8; gaps.append(y - 0.9)
    ys.append(y); labels.append(f"{GROUP[su[0]]}:  {WN[su[1]]}" if su[0] != prev else f"{GROUP[su[0]]}:  {WN[su[1]]}")
    prev = su[0]; y += 1
ys = np.array(ys)
for ax, m, scale, xl in [(axes[0], "top1", 100, "Naming the most-attended paragraph:\nself-report minus same model reading the transcript (% points)"),
                          (axes[1], "rho", 1, "Ranking the four paragraphs:\nself-report minus transcript reader (rank correlation)")]:
    for yy, su in zip(ys, SETUPS):
        d = R["|".join(su)]["self-reader"][m]
        mu = d["mean"] * scale; lo, hi = [v * scale for v in d["0.05"]]
        col = PURPLE if mu > 0 else RED
        ax.barh(yy, mu, height=0.72, color=col)
        ax.plot([lo, hi], [yy, yy], color=INK, lw=3.5, solid_capstyle="butt")
        for v in (lo, hi): ax.plot([v, v], [yy - 0.2, yy + 0.2], color=INK, lw=3.5)
    ax.axvline(0, color=INK, lw=2.5)
    ax.set_xlabel(xl, fontsize=30, labelpad=18)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=30, width=2, length=10)
    lim = max(abs(np.array(ax.get_xlim()))); ax.set_xlim(-lim, lim)
    ax.text(-lim * 0.97, -1.25, "reader better", color=RED, fontsize=32, ha="left", va="center", fontweight="bold")
    ax.text(lim * 0.97, -1.25, "self better", color=PURPLE, fontsize=32, ha="right", va="center", fontweight="bold")
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: (f"{v:.0f}" if scale == 100 else f"{v:.1f}").replace("-", "\u2212")))
axes[0].set_yticks(ys); axes[0].set_yticklabels(labels, fontsize=30)
axes[0].set_ylim(ys[-1] + 0.7, -1.9)
fig.suptitle(sys.argv[3], fontsize=40, fontweight="bold", x=0.5, y=0.985)
fig.savefig(sys.argv[2], dpi=100, bbox_inches="tight", pad_inches=0.35)
fig.savefig(sys.argv[2].replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.35)
print("saved", sys.argv[2])
