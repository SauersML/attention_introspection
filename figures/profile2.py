# Change in attention vs "only mentioned", through the four paragraphs. Qwen3-32B, the 44 arithmetic questions naming P3.
# Per bin: 100*(mean over questions with the instruction / mean over the same questions only mentioned - 1); 95% bootstrap over questions.
import json, glob, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
D = "/Users/user/mpd-data/runpod/hypha-prod-attntok2/out/"
OUT = "/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro/"
sys.path.insert(0, OUT); from pal import PURPLE, RED, INK
NB = 20
files = sorted(glob.glob(D + "tokens_ctx*.json")); prof = {c: [] for c in ("mention", "focus", "ignore")}
for fn in files:
    T = json.load(open(fn)); assert T["k"] == 2
    for c in prof:
        row = []
        for p in T["conds"][c]["paras"]:
            a = np.array(p["att"]); b = (np.arange(len(a)) * NB) // len(a); row += [a[b == i].mean() for i in range(NB)]
        prof[c].append(row)
prof = {c: np.array(v) for c, v in prof.items()}; n = len(files)
idx = np.random.default_rng(0).integers(0, n, (10000, n))
def pct(c):
    m = 100 * (prof[c].mean(0) / prof["mention"].mean(0) - 1)
    bs = 100 * (prof[c][idx].mean(1) / prof["mention"][idx].mean(1) - 1)
    return m, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
S = {c: pct(c) for c in ("focus", "ignore")}
G = 5; x = np.array([p * (NB + G) + i + 0.5 for p in range(4) for i in range(NB)])
plt.rcParams.update({"font.family": "Helvetica", "font.size": 56, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
fig, a = plt.subplots(figsize=(32, 13))
a.axvspan(2 * (NB + G), 2 * (NB + G) + NB, color="#F2B53A", alpha=0.12, lw=0, zorder=0)
for p in range(1, 4): a.axvline(p * (NB + G) - G / 2, color=INK, lw=2, zorder=1)
a.axhline(0, color=INK, lw=4, zorder=2)
for c, col in (("focus", PURPLE), ("ignore", RED)):
    m, lo, hi = S[c]
    for p in range(4):
        s = slice(p * NB, (p + 1) * NB)
        a.fill_between(x[s], lo[s], hi[s], color=col, alpha=0.18, lw=0, zorder=2); a.plot(x[s], m[s], color=col, lw=7, zorder=3)
yl = min(S["ignore"][1].min(), S["focus"][1].min()); yh = max(S["focus"][2].max(), S["ignore"][2].max())
for p in range(4):
    a.text(p * (NB + G) + NB / 2, yh + 6, f"P{p+1}", ha="center", va="bottom", fontsize=84, fontweight="bold", color=INK)
a.set_ylim(yl - 8, yh + 22); a.set_xlim(-1, 4 * (NB + G) - G + 1)
a.set_xticks([p * (NB + G) + t for p in range(4) for t in (0, NB)]); a.set_xticklabels(["0%", "100%"] * 4, fontsize=40)
for i, t in enumerate(a.get_xticklabels()): t.set_ha("left" if i % 2 == 0 else "right")
a.set_xlabel("position within each paragraph", fontsize=56, labelpad=10)
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: (f"{v:+.0f}%".replace("-", "−")) if v else "0%"))
a.set_ylabel("change in attention\nvs. only mentioned", fontsize=60, labelpad=12)
for s_ in ["top", "right"]: a.spines[s_].set_visible(False)
a.spines["left"].set_linewidth(3); a.spines["bottom"].set_linewidth(3); a.tick_params(width=4, length=14, labelsize=56)
x3 = 2 * (NB + G); i = 2 * NB + int(np.argmax(S["focus"][0][2 * NB:3 * NB]))
a.text(x3 + NB + 1.5, S["focus"][0][2 * NB:3 * NB].mean(), "“focus your\nattention”", color=PURPLE, fontsize=50, fontweight="bold", ha="left", va="center", linespacing=1.0)
a.text(x3 + NB + 1.5, min(S["ignore"][0][2 * NB:3 * NB].mean(), -8), "“do not pay\nattention”", color=RED, fontsize=50, fontweight="bold", ha="left", va="top", linespacing=1.0)
fig.tight_layout(pad=0.6); fig.savefig(OUT + "profile2.png", dpi=80, facecolor="white"); fig.savefig(OUT + "profile2.jpg", dpi=80, facecolor="white")
for c in S: print(c, "P3 mean %+.1f%%" % S[c][0][2 * NB:3 * NB].mean(), "others", [round(S[c][0][p*NB:(p+1)*NB].mean(),1) for p in range(4)])
