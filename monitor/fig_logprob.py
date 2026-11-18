# Figure for results/monitor/LOGPROB.md: per-prompt rank correlation (self-report vs transcript reader) and the
# held-out gain from adding the self-report's log-probabilities, against its shuffle null.
# usage: fig_logprob.py logprob.json SHARD_DIR [SHARD_DIR ...] OUT.png
import json, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import gaussian_kde
sys.path.insert(0, "/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro")
from pal import PURPLE, INK

J = json.load(open(sys.argv[1])); OUT = sys.argv[-1]; DIRS = sys.argv[2:-1]
C, H, G = {}, {}, {}
for d in DIRS:
    for c in json.load(open(os.path.join(d, "contexts.json"))): C[str(c["id"])] = c
    H.update(json.load(open(os.path.join(d, "hf1.json")))); G.update(json.load(open(os.path.join(d, "gen1.json"))))
RES, PP, NUL = J["res"], J["perprompt"], J["null"]
ROWS = [t for t in ["after", "before", "later", "think", "fb4", "fb16"] if t in RES]
LAB = {"after": "asked right after answering", "before": "asked before answering", "later": "asked one turn later",
       "think": "thinking on", "fb4": "after 4 examples of true attention", "fb16": "after 16 examples of true attention", "pooled": "all setups"}
BLUE = "#1E8BFF"; AMB = "#F2B53A"; F = 30
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK})
W, Hh = 23, 24
fig = plt.figure(figsize=(W, Hh), dpi=100)

# ---------------- top: one real prompt
EX = "18"; c = C[EX]
v = np.array(H[EX]["truth"]["norm"])[21:54].mean(0); v = 100 * v / v.sum()
ls = np.log(H[EX]["rep"]["after|most|self"]); lr = np.log(H[EX]["rep"]["after|most|reader"])
rs = PP["after"]["self"][PP["after"]["ids"].index(EX)]; rr = PP["after"]["reader"][PP["after"]["ids"].index(EX)]
TOPF = 0.357
ax = fig.add_axes([0, 1 - TOPF, 1, TOPF]); ax.set_xlim(0, W); ax.set_ylim(0, TOPF * Hh); ax.axis("off")
top = TOPF * Hh
def bubble(x, y, w, h, fc, ec=None, lw=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.35", fc=fc, ec=ec or fc, lw=lw))
def short(p, n): return " ".join(p.split()[:n]) + " …"
# user bubble
ax.text(0.3, top - 0.1, "Qwen3-32B, 300 prompts like this one", fontsize=F, va="top", fontweight="bold")
bx, by, bw, bh = 0.3, top - 4.3, 10.7, 3.5
bubble(bx, by, bw, bh, BLUE)
lines = [f"[P{k+1}] " + short(p, 4) for k, p in enumerate(c["paras"])] + [c["task"].replace("Answer briefly. Question: ", "")]
for n, s in enumerate(lines):
    ax.text(bx + 0.3, by + bh - 0.35 - 0.6 * n, s, color="white", fontsize=F, va="top", fontweight="bold" if n == 4 else "normal")
# answer bubble with robot
rob = plt.imread("/Users/user/watching/figures/assets/robot.png")
ax.add_artist(AnnotationBbox(OffsetImage(rob, zoom=0.11), (0.85, by - 0.72), frameon=False))
bubble(1.6, by - 1.25, 9.4, 1.05, "white", "#D0D0D6", 2)
ax.text(1.9, by - 0.72, G[EX]["ans_text"].strip(), fontsize=F, va="center")
# two questions
qy = by - 2.45
bubble(0.3, qy, 10.7, 1.0, "white", PURPLE, 5)
ax.text(0.6, qy + 0.5, "Which paragraph did you attend to most?", color=PURPLE, fontsize=F, va="center", fontweight="bold")
qy2 = qy - 1.75
bubble(0.3, qy2, 10.7, 1.55, "white", INK, 5)
ax.text(0.6, qy2 + 1.15, "New chat: [the conversation above, quoted]", color=INK, fontsize=F, va="center")
ax.text(0.6, qy2 + 0.45, "Which paragraph did the assistant attend to most?", color=INK, fontsize=F, va="center", fontweight="bold")
# right: the per-prompt read-out
x0 = 11.6; colw = [2.9, 2.6, 2.6]; heads = ["true attention", "self-report", "transcript reader"]; cols = [AMB, PURPLE, INK]
rowy = [top - 1.75 - 1.05 * k for k in range(4)]
xs = [x0 + 0.8, x0 + 4.7, x0 + 8.3]
for j, (h, cc) in enumerate(zip(heads, cols)):
    ax.text(xs[j] + colw[j] / 2, top - 0.35, h, ha="center", va="top", fontsize=F, fontweight="bold", color=cc if j else INK)
ax.text(xs[0] + colw[0] / 2, top - 0.9, "from the answer tokens", ha="center", va="top", fontsize=F - 4)
ax.text((xs[1] + xs[2] + colw[2]) / 2, top - 0.9, "log probability of answering this paragraph", ha="center", va="top", fontsize=F - 4)
LO = -25.0
for k in range(4):
    y = rowy[k]
    ax.text(x0 + 0.6, y, f"P{k+1}", ha="right", va="center", fontsize=F, fontweight="bold")
    ax.add_patch(plt.Rectangle((xs[0], y - 0.32), colw[0] * v[k] / 100 / 0.6, 0.64, color=AMB))
    ax.text(xs[0] + colw[0] * v[k] / 100 / 0.6 + 0.12, y, f"{v[k]:.0f}%", va="center", fontsize=F - 2)
    for j, l in [(1, ls[k]), (2, lr[k])]:
        ax.plot([xs[j], xs[j] + colw[j]], [y, y], color="#D0D0D6", lw=2, zorder=1)
        ax.scatter([xs[j] + colw[j] * (max(l, LO) - LO) / -LO], [y], s=500, color=cols[j], zorder=3)
for j in (1, 2):
    for tx, val in [(xs[j], "−25"), (xs[j] + colw[j], "0")]:
        ax.text(tx, rowy[3] - 0.45, val, ha="center", va="top", fontsize=F - 4)
ax.text(xs[0] + colw[0] / 2, rowy[3] - 1.55, "rank agreement\nwith true attention", ha="center", va="center", fontsize=F - 2)
for j, r_ in [(1, rs), (2, rr)]:
    ax.text(xs[j] + colw[j] / 2, rowy[3] - 1.55, f"{r_:.1f}", ha="center", va="center", fontsize=F + 14, fontweight="bold", color=cols[j])
ax.plot([0.3, W - 0.3], [0.05, 0.05], color=INK, lw=3)

# ---------------- bottom: violins and gains
NR = len(ROWS) + 1
axL = fig.add_axes([0.035, 0.07, 0.53, 0.535]); axR = fig.add_axes([0.62, 0.07, 0.36, 0.535])
rng = np.random.default_rng(0)
def violin(a, vals, y, col, half=0.33, fill=0.18):
    vals = np.asarray(vals, float); g = np.linspace(-1, 1, 300)
    kd = gaussian_kde(vals, bw_method=0.35)(g); kd = half * kd / kd.max()
    a.fill_between(g, y - kd, y + kd, color=col, alpha=fill, lw=0)
    a.plot(g, y + kd, color=col, lw=2, alpha=0.6); a.plot(g, y - kd, color=col, lw=2, alpha=0.6)
    jy = y + rng.uniform(-1, 1, len(vals)) * np.interp(vals, g, kd) * 0.9
    a.scatter(vals + rng.uniform(-0.03, 0.03, len(vals)), jy, s=26, color=col, alpha=0.55, lw=0)
    a.plot([vals.mean()] * 2, [y - half - 0.04, y + half + 0.04], color=col, lw=5)
pooled_self = {}; pooled_read = {}
for t in ROWS:
    for n, i in enumerate(PP[t]["ids"]):
        pooled_self.setdefault(i, []).append(PP[t]["self"][n]); pooled_read.setdefault(i, []).append(PP[t]["reader"][n])
ks = sorted(pooled_self, key=int)
series = [(t, PP[t]["self"], PP[t]["reader"]) for t in ROWS] + [("pooled", [np.mean(pooled_self[i]) for i in ks], [np.mean(pooled_read[i]) for i in ks])]
for r, (t, s, rd) in enumerate(series):
    y0 = NR - 1 - r
    violin(axL, s, y0 + 0.08, PURPLE, half=0.17); violin(axL, rd, y0 - 0.3, INK, half=0.17, fill=0.10)
    n = RES[t]["n"]
    axL.text(-1.0, y0 + 0.3, LAB[t] + ("" if n == len(C) else f"  ({n} prompts)"), fontsize=F - 2, va="bottom", fontweight="bold", zorder=6, bbox=dict(fc="white", ec="none", pad=1))
axL.axvline(0, color=INK, ls=(0, (4, 4)), lw=2)
axL.set_xlim(-1.02, 1.02); axL.set_ylim(-0.55, NR - 0.05)
axL.set_yticks([]); [axL.spines[s].set_visible(False) for s in ["top", "right", "left"]]
axL.set_xticks([-1, 0, 1]); axL.set_xticklabels(["−1", "0", "1"], fontsize=F)
axL.set_xlabel("rank agreement with true attention, one dot per prompt", fontsize=F)
REN = fig.canvas.get_renderer()
def segs(x, y, parts):
    for s, col in parts:
        t = fig.text(x, y, s, fontsize=F + 2, fontweight="bold", va="bottom", color=col)
        x = fig.transFigure.inverted().transform(t.get_window_extent(REN))[1][0]
segs(0.035, 0.625, [("The ", INK), ("self-report", PURPLE), (" tracks attention", INK)])
segs(0.035, 0.6045, [("as well as the ", INK), ("transcript reader", INK), (", no better", INK)])
segs(0.62, 0.625, [("Adding the ", INK), ("self-report", PURPLE), (" to the reader", INK)])
segs(0.62, 0.6045, [("adds no more than chance", INK)])

# right: held-out gain from adding the self-report, vs shuffle null
LIMR = max(0.05, max(abs(RES[t]["gain_rho"][k]) for t in ROWS + ["pooled"] for k in ("lo", "hi")) + 0.01)
for r, t in enumerate(ROWS + ["pooled"]):
    y0 = NR - 1 - r; R = RES[t]
    y0 = y0 - 0.11
    if t == "pooled":
        nul = np.array([sum(NUL[u]["rho"][j] * RES[u]["n"] for u in ROWS) / sum(RES[u]["n"] for u in ROWS) for j in range(len(NUL[ROWS[0]]["rho"]))])
    else:
        nul = np.array(NUL[t]["rho"])
    if r == 0:
        axR.text(nul.max() + 0.004, y0 + 0.2, "self-report shuffled\nacross prompts", ha="left", va="bottom", fontsize=F - 2, color=INK)
    pv = R["null_gain_rho"]["p"]
    axR.text(-0.97 * LIMR, y0 + 0.17, ("p < 0.001" if pv < 0.001 else f"p = {pv:.3f}" if pv < 0.01 else f"p = {pv:.2f}"), ha="left", va="bottom", fontsize=F - 2, color=INK)
    if nul is not None:
        g = np.linspace(nul.min() - 0.01, nul.max() + 0.01, 200); kd = gaussian_kde(nul)(g); kd = 0.25 * kd / kd.max()
        axR.fill_between(g, y0 - kd, y0 + kd, color="#E6E6EA", lw=0); axR.plot(g, y0 + kd, color="#9A9AA2", lw=2); axR.plot(g, y0 - kd, color="#9A9AA2", lw=2)
    else:
        lo, hi = R["null_gain_rho"]["lo"], R["null_gain_rho"]["hi"]
        axR.add_patch(plt.Rectangle((lo, y0 - 0.2), hi - lo, 0.4, color="#E6E6EA", lw=0))
    gr = R["gain_rho"]
    axR.plot([gr["lo"], gr["hi"]], [y0, y0], color=PURPLE, lw=6, solid_capstyle="round")
    axR.scatter([gr["mean"]], [y0], s=520, color=PURPLE, zorder=4)
axR.axvline(0, color=INK, ls=(0, (4, 4)), lw=2)
lim = max(0.05, max(abs(RES[t]["gain_rho"][k]) for t in ROWS + ["pooled"] for k in ("lo", "hi")) + 0.01)
axR.set_xlim(-lim, lim); axR.set_ylim(-0.55, NR - 0.05)
axR.set_yticks([]); [axR.spines[s].set_visible(False) for s in ["top", "right", "left"]]
tk = [-0.04, 0, 0.04]; axR.set_xticks(tk); axR.set_xticklabels(["−0.04", "0", "+0.04"], fontsize=F)
axR.set_xlabel("gain in rank agreement when the\nself-report is added to the reader", fontsize=F)
fig.savefig(OUT, dpi=100, facecolor="white"); print("saved", OUT)
