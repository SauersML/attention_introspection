"""words_told.png: the told-injection run on single-word lists, as a plain thread figure (Qwen3-32B, 240 contexts, words [W1]..[W8]
in order; results/told_inject_words/data, told_inject_words/told_inject_words.py, analysis in told_inject_words/score.py).
The researcher's intro says attention may be boosted onto one of the eight words on half the trials; Qwen answers an arithmetic
question, then a second user turn asks "Trial 1: Do you notice your attention being boosted onto one of the words? If so, which one?".
Boosted trials add log(multiplier) to the pre-softmax attention scores on the boosted word's token, layers 21-53, for every query from
"Trial 1" on (question, reply and prefill tokens). x axis: no boost, then nominal multipliers 2/5/10/20 labelled by the measured
attention-mass ratio on the boosted word (question-token queries, layers 21-53, mean over contexts and words).
Left: per prompt (dot), P("Yes") as the first reply token renormalised over Yes/No, mean over the 8 boosted words (no boost: the one
control trial). Greedy replies: "No" on all 1,920 boosted and all 240 control trials.
Right: after a prefilled "Yes, word [W", per prompt (dot) the probability of the boosted word's number (renormalised over the 8
numbers), mean over the 8 boosted words; dashed line chance 1/8. The boost is still on while this number is read.
Big labels: right, mean / 0.125 - 1 (two-sided sign-flip test over prompts of mean - 0.125, 200,000 flips).
usage: OMP_NUM_THREADS=2 nice -n 19 python3 figures/words_told_fig.py -> figures/words_told.png"""
import json, glob, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE, pct_colour
from pfmt import pstr
D = os.path.join(os.path.dirname(HERE), "results/told_inject_words/data", "")
R = sorted([json.loads(l) for f in glob.glob(D + "recs_*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["id"]); assert len(R) == 240
NW = 8; DOSES = ["2.0", "5.0", "10.0", "20.0"]
py = lambda t: 1 / (1 + np.exp(t["lp_no"] - t["lp_yes"]))
T = lambda r, k, m: r["boost"][f"{k}|{m}"]
assert all(t["text"].strip() in ("No", "No.") for r in R for t in [r["none"]] + [T(r, k, m) for k in range(NW) for m in DOSES])
NGY = sum(T(r, k, m)["text"].strip().startswith("Yes") for r in R for k in range(NW) for m in DOSES)
NCY = sum(r["none"]["text"].strip().startswith("Yes") for r in R)
Y = [np.array([py(r["none"]) for r in R])] + [np.array([np.mean([py(T(r, k, m)) for k in range(NW)]) for r in R]) for m in DOSES]
P = [np.array([np.mean([np.exp(T(r, k, m)["lp_par"][k]) for k in range(NW)]) for r in R]) for m in DOSES]
MASS = [np.mean([np.mean([T(r, k, m)["mass"][k] / r["none"]["mass"][k] for k in range(NW)]) for r in R]) for m in DOSES]
def signflip(x, NF=200000):
    x = np.asarray(x); g = np.random.default_rng(0); obs = abs(x.mean()); h = 0
    for _ in range(NF // 20000):
        s = g.choice([-1.0, 1.0], (20000, len(x))); h += int((np.abs((s * x).mean(1)) >= obs).sum())
    return (h + 1) / (NF + 1)
print("greedy Yes boosted", NGY, "of", len(R) * NW * len(DOSES), "| control", NCY, "of", len(R))
print("P(Yes) means %", [round(100 * y.mean(), 2) for y in Y]); print("mass", [round(m, 2) for m in MASS])
print("P(boosted number) means %", [round(100 * p.mean(), 1) for p in P])

plt.rcParams.update({"font.family": "Helvetica", "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig = plt.figure(figsize=(23, 13.0), dpi=100)
axL = fig.add_axes([0.085, 0.175, 0.40, 0.54]); axR = fig.add_axes([0.675, 0.175, 0.31, 0.54])
rng = np.random.default_rng(1)
def violin(a, x, v, col, w=0.36, ylo=0, yhi=1):
    kde = gaussian_kde(v, bw_method=0.35); ys = np.linspace(max(ylo, v.min()), min(yhi, v.max()), 300); d = kde(ys); d = d / d.max() * w
    a.fill_betweenx(ys, x - d, x + d, color=col, alpha=0.18, lw=0); a.plot(np.r_[x - d, (x + d)[::-1], x - d[:1]], np.r_[ys, ys[::-1], ys[:1]], color=col, lw=2.5)
    xj = x + rng.uniform(-1, 1, len(v)) * np.interp(v, ys, d) * 0.9
    return xj
XL = ["no\nboost"] + [f"\u00d7{m:.1f}" for m in MASS]
# ---- left: P(Yes)
for i, y in enumerate(Y):
    xj = violin(axL, i, y, PURPLE, ylo=0, yhi=1)
    axL.scatter(xj, y, s=150, color=PURPLE, alpha=0.35, edgecolors="none", zorder=3); axL.scatter(xj, y, s=4, color="black", zorder=4)
    axL.plot([i - 0.28, i + 0.28], [y.mean()] * 2, color=INK, lw=5, zorder=5)
YMX = max(y.max() for y in Y); axL.set_ylim(0, YMX * 1.3); axL.set_yticks([0, 0.01, 0.02, 0.03]); axL.set_yticklabels(["0%", "1%", "2%", "3%"])
relY = Y[-1].mean() / Y[0].mean() - 1; pY = pstr(signflip(Y[-1] - Y[0]), floor=1 / 200001)
axL.text(4, YMX * 1.29, f"{100 * relY:+.0f}%\n{pY}", ha="center", va="top", fontsize=40, fontweight="bold", color=PURPLE, linespacing=1.1, zorder=6)
axL.set_ylabel("probability Qwen starts with \u201cYes\u201d", fontsize=38, fontweight="bold")
# ---- right: forced pick
for i, p in enumerate(P):
    xj = violin(axR, i, p, PURPLE)
    axR.scatter(xj, p, s=150, c=pct_colour(p / 0.125), alpha=0.8, edgecolors="none", zorder=3); axR.scatter(xj, p, s=4, color="black", zorder=4)
    axR.plot([i - 0.28, i + 0.28], [p.mean()] * 2, color=INK, lw=5, zorder=5)
    rel = p.mean() / 0.125 - 1; pv = pstr(signflip(p - 0.125), floor=1 / 200001)
    axR.text(i, min(p.max(), 0.97) + 0.03, f"{100 * rel:+.0f}%", ha="center", va="bottom", fontsize=48, fontweight="bold", color=PURPLE)
axR.axhline(0.125, color=INK, lw=4, ls=(0, (4, 3)), zorder=2)
axR.text(3.6, 0.125 - 0.02, "chance 12.5%", ha="right", va="top", fontsize=38, color=INK)
axR.set_ylim(0, 1.12); axR.set_yticks([0, 0.25, 0.5, 0.75, 1]); axR.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
axR.set_ylabel("probability Qwen picks\nthe boosted word", fontsize=38, fontweight="bold", linespacing=1.05)
for a, n in ((axL, 5), (axR, 4)):
    a.set_xlim(-0.6, n - 0.4); a.set_xticks(range(n)); a.set_xticklabels(XL if n == 5 else XL[1:], fontsize=40, fontweight="bold")
    a.tick_params(axis="y", labelsize=40, width=3, length=10); a.tick_params(axis="x", length=0, pad=12)
    for t in a.get_yticklabels(): t.set_fontweight("bold")
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"): a.spines[s_].set_linewidth(3)
fig.text(0.5, 0.012, "attention on the boosted word, compared to no boost", ha="center", fontsize=40, fontweight="bold")
# ---- headers: what was asked, and the result
fig.text(0.278, 0.985, "asked \u201cDo you notice your attention being\nboosted onto one of the words?\u201d", ha="center", va="top", fontsize=40, linespacing=1.1)
fig.text(0.278, 0.84, f"says \u201cYes\u201d {NGY} of {len(R) * NW * len(DOSES):,} times", ha="center", va="top", fontsize=56, fontweight="bold", color=PURPLE)
fig.text(0.77, 0.985, "reply forced to start \u201cYes, word [W\u201d;\nthe boost is still on while Qwen picks", ha="center", va="top", fontsize=40, linespacing=1.1)
fig.text(0.77, 0.84, "picks the boosted word", ha="center", va="top", fontsize=56, fontweight="bold", color=PURPLE)
fig.add_artist(plt.Line2D([0.535, 0.535], [0.07, 0.99], transform=fig.transFigure, color=INK, lw=3))
OUT = os.path.join(HERE, "words_told.png"); fig.savefig(OUT, dpi=100, facecolor="white"); print(OUT)
