"""Per-layer % change in attention on the named paragraph vs the mention-only control (Qwen3-32B dose run, 150 arithmetic contexts).
Measure as dose_analyze.py: answer-token queries, mean over heads, per-token mean over the paragraph body; per context % change at each layer,
mean over contexts; band = 95% bootstrap over contexts (10,000 resamples)."""
import json, os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
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
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
fig, a = plt.subplots(figsize=(18, 14))
l = np.arange(NL)
for c, col, lab in [("3_focus", PURPLE, "focus your attention"), ("9_completely_ignore", RED, "completely ignore")]:
    o = out[c]; a.plot(l, o["mean"], color=col, lw=3.5, zorder=3); a.fill_between(l, o["lo"], o["hi"], color=col, alpha=0.45, lw=0, zorder=2)
    o["lab"] = lab
a.axhline(0, color=INK, lw=4)
for s_ in ["top", "right"]: a.spines[s_].set_visible(False)
a.spines["left"].set_linewidth(3); a.spines["bottom"].set_linewidth(3); a.tick_params(width=3, length=12, labelsize=64)
a.set_xlim(-1, 64); a.set_xticks([0, 16, 32, 48, 63]); a.set_xlabel("layer", fontsize=76, labelpad=10)
a.set_ylabel("change in attention vs just\nmentioning the paragraph name", fontsize=56, labelpad=14, linespacing=1.0)
yt = [-50, 0, 50, 100, 150, 200, 250]; a.set_yticks(yt); a.set_yticklabels([(f"{v:+d}%" if v else "0%").replace("-", "\u2212") for v in yt]); a.set_ylim(-66, 262)
mf = np.array(out["3_focus"]["mean"]); i = int(np.argmax(mf))
a.text(1, 255, "focus your\nattention", fontsize=64, color=PURPLE, fontweight="bold", va="top", linespacing=1.0)
mi = np.array(out["9_completely_ignore"]["mean"])
a.text(36, 150, "shaded: 95% CI", fontsize=48, color=INK, va="center")
a.text(22, -32, "completely ignore", fontsize=64, color=RED, fontweight="bold", va="top")
fig.tight_layout(pad=0.4); fig.savefig(os.path.join(HERE, "layers.png"), dpi=130, facecolor="white"); fig.savefig(os.path.join(HERE, "layers.jpg"), dpi=130, facecolor="white")
for c in out: m = np.array(out[c]["mean"]); print(c, "max %+.1f @%d  min %+.1f @%d  band21-53 mean-of-layer-means %+.1f" % (m.max(), m.argmax(), m.min(), m.argmin(), m[21:54].mean()))
print("focus layers 0-20 range", mf[:21].min(), mf[:21].max()); print("ignore per layer", np.round(mi).astype(int).tolist())
