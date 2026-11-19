"""Per-layer % change in attention on the named paragraph vs the mention-only control (Qwen3-32B dose run, 150 arithmetic contexts).
Measure as dose_analyze.py: answer-token queries, mean over heads, per-token mean over the paragraph body; per context % change at each layer,
mean over contexts; band = 95% bootstrap over contexts (10,000 resamples). Drawn in the style of the hero's method-box curve: one line
per instruction, labelled on the plot by its instruction line (the real wording, as in the example prompt id 300), layer 28 (the hero's
layer) marked with a dashed line and its mean written beside the marker, dashed 0% line = the mention-only line."""
import json, os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import IdentityTransform
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import PURPLE, RED, INK
D = "/Users/user/connectome/islands/agents/resident-a/compute/attn_intro/qwen3-32b-dose/"
meta = json.load(open(D + "meta.json")); NL = meta["NL"]
R = json.load(open(D + "dose.json")); by = {(r["id"], r["cond"]): r for r in R}; ids = sorted({r["id"] for r in R})
RES = json.load(open(D + "dose_results.json"))
def PL(c): return np.array([[by[i, c]["att_mean"][l][by[i, c]["k"]] for l in range(NL)] for i in ids])
ctrl = PL("1_mention"); rng = np.random.default_rng(0); BI = rng.integers(0, len(ids), (10000, len(ids)))
out = {}
for c in ["3_focus", "9_completely_ignore"]:
    z = 100 * (PL(c) / ctrl - 1); m = z.mean(0); bs = z[BI].mean(1)
    assert np.allclose(m, RES["per_layer_vs_mention"][c])
    out[c] = dict(mean=m.tolist(), lo=np.percentile(bs, 2.5, 0).tolist(), hi=np.percentile(bs, 97.5, 0).tolist())
json.dump(dict(n=len(ids), **out), open(os.path.join(HERE, "layers3_data.json"), "w"))
K = by[300, "3_focus"]["k"] + 1   # the example prompt's named paragraph, as in hero_L28.py
LINE = {c: meta["conds"][c][0].format(k=K) for c in out}
MN = lambda t: t.replace("-", "\u2212")
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
WPX, HPX = 2300, 1620
fig = plt.figure(figsize=(WPX / 100, HPX / 100))
a = fig.add_axes([0.135, 0.095, 0.855, 0.895])
l = np.arange(NL); L28 = 28
LG = lambda v: np.log1p(np.asarray(v) / 100)   # log scale with % ticks, as the heroes
mf = np.array(out["3_focus"]["mean"]); mi = np.array(out["9_completely_ignore"]["mean"])
a.axvline(L28, color=INK, lw=3.5, ls=(0, (4, 3)), zorder=1)
a.axhline(0, color=INK, lw=4, ls=(0, (4, 3)), zorder=2)
for c, col in [("3_focus", PURPLE), ("9_completely_ignore", RED)]:
    o = out[c]; a.fill_between(l, LG(o["lo"]), LG(o["hi"]), color=col, alpha=0.25, lw=0, zorder=3)
    a.plot(l, LG(o["mean"]), color=col, lw=6, solid_joinstyle="round", zorder=4)
    a.scatter([L28], [LG(o["mean"][L28])], s=520, color=col, edgecolors="white", linewidths=4, zorder=6)
a.text(L28 + 1.1, LG(mf[L28]), MN(f"{mf[L28]:+.0f}%"), fontsize=80, fontweight="bold", color=PURPLE, ha="left", va="center", zorder=7)
a.text(L28 - 0.9, LG(mi[L28]) - 0.03, MN(f"{mi[L28]:+.0f}%"), fontsize=80, fontweight="bold", color=RED, ha="right", va="top", zorder=7)
a.text(L28, LG(345), "layer 28", fontsize=56, fontweight="bold", ha="center", va="bottom", zorder=7,
       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
a.text(0.2, -0.03, "mention only = 0%", fontsize=48, fontweight="bold", ha="left", va="top", zorder=7)
a.text(0.5, LG(330), "% change in how\nmuch Qwen's\nanswer attends to\nthe mentioned\nparagraph", fontsize=50, fontweight="bold", ha="left", va="top", linespacing=1.05, zorder=7)
for s_ in ["top", "right"]: a.spines[s_].set_visible(False)
a.spines["left"].set_linewidth(3.5); a.spines["bottom"].set_linewidth(3.5)
a.tick_params(width=3.5, length=12, labelsize=64, pad=10)
a.set_xlim(-0.8, 64); a.set_xticks([0, 21, 63])   # 21: first layer where the instruction moves attention
yt = [-25, 0, 50, 100, 200, 300]; a.set_yticks(LG(yt)); a.set_yticklabels([MN(f"{v:+d}%") if v else "0%" for v in yt]); a.set_ylim(LG(-54), LG(400))
for t in a.get_xticklabels() + a.get_yticklabels(): t.set_fontweight("bold")
fig.canvas.draw(); rd = fig.canvas.get_renderer()
ty = a.get_xticklabels()[0].get_window_extent(rd); tyc = (ty.y0 + ty.y1) / 2 / HPX   # "layer" on the tick-number row, as in the hero
a.text(a.transData.transform((31.6, 0))[0] / WPX, tyc, "layer", fontsize=64, fontweight="bold", ha="center", va="center", transform=fig.transFigure)

def bubble(x, y, text, col, ha="left", va="center"):
    """compact chat bubble: the instruction line in a white inner box (coloured text) inside the blue bubble"""
    t = a.text(x, y, text, fontsize=38, fontweight="bold", color=col, ha=ha, va=va, linespacing=1.1, zorder=9,
               bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.4", fc="white", ec="none"))
    fig.canvas.draw(); bb = t.get_bbox_patch().get_window_extent(rd)
    p = 18   # blue margin around the white box, px
    a.add_artist(FancyBboxPatch((bb.x0 - p, bb.y0 - p), bb.width + 2 * p, bb.height + 2 * p, transform=IdentityTransform(),
                                boxstyle="round,pad=0,rounding_size=34", fc="#1F8BFF", ec="none", zorder=8, clip_on=False))
    return t
bubble(36, LG(150), LINE["3_focus"].replace(" paragraph", "\nparagraph"), PURPLE)
bubble(34.5, LG(-45), LINE["9_completely_ignore"].replace(" Do not", "\nDo not"), RED)
fig.savefig(os.path.join(HERE, "layers.png"), dpi=100, facecolor="white"); fig.savefig(os.path.join(HERE, "layers.jpg"), dpi=100, facecolor="white")
for c in out: m = np.array(out[c]["mean"]); print(c, "max %+.1f @%d  min %+.1f @%d  band21-53 mean-of-layer-means %+.1f" % (m.max(), m.argmax(), m.min(), m.argmin(), m[21:54].mean()))
print("focus layers 0-20 range", mf[:21].min(), mf[:21].max()); print("ignore per layer", np.round(mi).astype(int).tolist())
