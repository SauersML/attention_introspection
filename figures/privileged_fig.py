# One figure for the privileged-channel experiment: rotation specificity vs boost dose, for the three places the boost is applied.
# usage: privileged_fig.py results.json OUT.png
import json, sys, math, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pal import PURPLE, RED, INK

R = json.load(open(sys.argv[1])); OUT = sys.argv[2]; EX = sys.argv[3] if len(sys.argv) > 3 else "83"
DOSES = [0.25, 0.5, 2.0, 5.0, 10.0]
TICK = ["×¼", "×½", "×2", "×5", "×10"]
GREY = "#F4F4F6"
pct = lambda x: 100 * (math.exp(x) - 1)
neg = lambda s: s.replace("-", "−")
plt.rcParams.update({"font.family": "Helvetica", "font.size": 42, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
ARMS = [("A", ("answer",)), ("B", ("answer", "question")), ("C", ("question",))]
vals = {a: [R["spec"][f"{a}|W1|all|{m}"] for m in DOSES] for a, _ in ARMS}
lo = min(min(pct(v["lo"]) for v in vals[a]) for a in vals); hi = max(max(pct(v["hi"]) for v in vals[a]) for a in vals)
pad = 0.06 * (hi - lo); ylo, yhi = min(lo, 0) - pad, hi + pad

def pill(ax, x, y, w, h, text, fc=GREY, tc=INK, fs=40, ec=None, lw=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.35", fc=fc, ec=ec or fc, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs)

fig = plt.figure(figsize=(24, 19))
# ---- the chat, drawn once ----
t = fig.add_axes([0.02, 0.80, 0.96, 0.18]); t.set_xlim(0, 48); t.set_ylim(0, 6); t.axis("off")
t.text(0, 5.3, "Qwen3-32B", fontsize=44, fontweight="bold", va="center")
for j in range(4):
    pill(t, 0.0 + j * 3.1, 1.0, 2.8, 2.6, f"P{j+1}", fc=PURPLE if j == 2 else GREY, tc="white" if j == 2 else INK, fs=44)
arrow = lambda x: t.annotate("", xy=(x + 1.0, 2.3), xytext=(x, 2.3), arrowprops=dict(arrowstyle="-|>", lw=4, color=INK, mutation_scale=40))
arrow(12.45); pill(t, 13.75, 1.0, 6.0, 2.6, "answer: " + EX, fs=42)
arrow(20.05); pill(t, 21.35, 1.0, 20.4, 2.6, "Which paragraph did you attend to most?", fs=37)
arrow(42.05); pill(t, 43.35, 1.0, 4.6, 2.6, "P?", fs=44)
t.text(16.75, 5.3, "answer identical in every run", ha="center", va="center", fontsize=40)

pw, gap, x0 = 0.262, 0.03, 0.15
for i, (arm, where) in enumerate(ARMS):
    left = x0 + i * (pw + gap)
    # which part of the chat gets the boosted attention onto P3
    h = fig.add_axes([left, 0.665, pw, 0.10]); h.set_xlim(0, 10); h.set_ylim(0, 3); h.axis("off")
    for k, (name, x, w) in enumerate([("answer", 0.0, 4.2), ("question", 4.6, 5.4)]):
        on = name in where
        pill(h, x, 0.4, w, 2.2, name, fc=PURPLE if on else GREY, tc="white" if on else INK, fs=40)
    if i == 0: fig.text(left - 0.012, 0.715, "boosted\nduring", ha="right", va="center", fontsize=38)
    a = fig.add_axes([left, 0.10, pw, 0.53])
    for j, v in enumerate(vals[arm]):
        y = pct(v["mean"])
        a.bar(j, y, width=0.74, color=PURPLE if y >= 0 else RED)
        a.plot([j, j], [pct(v["lo"]), pct(v["hi"])], color=INK, lw=4, solid_capstyle="butt")
    a.axhline(0, color=INK, lw=3)
    a.set_ylim(ylo, yhi); a.set_xlim(-0.6, len(DOSES) - 0.4)
    a.set_xticks(range(len(DOSES))); a.set_xticklabels(TICK, fontsize=44)
    for s in ("top", "right", "bottom"): a.spines[s].set_visible(False)
    a.tick_params(length=0, pad=12)
    if i == 0:
        a.spines["left"].set_linewidth(3)
        a.set_ylabel("change in the chance the report\nsays the boosted part", fontsize=42, labelpad=14)
        yt = [tt for tt in a.get_yticks() if ylo <= tt <= yhi]
        a.set_yticks(yt); a.set_yticklabels([neg(f"{tt:+.0f}%") if tt else "0%" for tt in yt], fontsize=42)
    else:
        a.set_yticks([]); a.spines["left"].set_visible(False)
fig.text(x0 + 1.5 * pw + gap, 0.025, "attention on the boosted part multiplied by", ha="center", fontsize=42)
fig.savefig(OUT, dpi=100, facecolor="white")
print("wrote", OUT)
