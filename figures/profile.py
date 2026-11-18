# Attention profile through the four paragraphs, Qwen3-32B, arithmetic contexts whose named paragraph is P3.
# Per-token attention from the answer-token queries (head-mean, answer-token mean, layer band 21-53),
# each paragraph's tokens binned into 20 equal relative-position bins; mean over contexts, 95% bootstrap band.
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
D = "/Users/user/mpd-data/runpod/hypha-prod-attntok2/out/"
OUT = "/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro/"
INK = "#1D1D1F"; BLUE = "#2B5FA8"; ORANGE = "#E07B24"
NB = 20
CONDS = [("mention", "only mentioned", INK), ("focus", "“focus your attention”", BLUE), ("ignore", "“do not pay attention”", ORANGE)]
files = sorted(glob.glob(D + "tokens_ctx*.json"))
prof = {c: [] for c, _, _ in CONDS}
for fn in files:
    T = json.load(open(fn)); assert T["k"] == 2
    for c, _, _ in CONDS:
        row = []
        for p in T["conds"][c]["paras"]:
            a = np.array(p["att"]); n = len(a); b = (np.arange(n) * NB) // n
            row += [a[b == i].mean() for i in range(NB)]
        prof[c].append(row)
prof = {c: np.array(v) for c, v in prof.items()}
n = len(files); print("contexts", n)
rng = np.random.default_rng(0); idx = rng.integers(0, n, (10000, n))
stats = {}
for c in prof:
    m = prof[c].mean(0); bs = prof[c][idx].mean(1); stats[c] = (m, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0))
G = 5
x = np.array([p * (NB + G) + i + 0.5 for p in range(4) for i in range(NB)])  # 2-bin gap between paragraphs
plt.rcParams.update({"font.family": "Helvetica", "font.size": 56, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
fig, a = plt.subplots(figsize=(32, 14))
a.axvspan(2 * (NB + G), 2 * (NB + G) + NB, color="#7B3FE4", alpha=0.07, lw=0, zorder=0)
for p in range(1, 4): a.axvline(p * (NB + G) - G / 2, color=INK, lw=2, zorder=1)
for c, lab, col in CONDS:
    m, lo, hi = stats[c]
    for p in range(4):
        s = slice(p * NB, (p + 1) * NB)
        a.fill_between(x[s], lo[s] * 1e4, hi[s] * 1e4, color=col, alpha=0.13, lw=0, zorder=2)
        a.plot(x[s], m[s] * 1e4, color=col, lw=6, zorder=3)
ymax = max(stats[c][2].max() for c in prof) * 1e4; ymin = min(stats[c][1].min() for c in prof) * 1e4
a.set_yscale("log")
for p in range(4):
    a.text(p * (NB + G) + NB / 2, ymax * 1.08, f"P{p+1}", ha="center", va="bottom", fontsize=84, fontweight="bold",
           color="#7B3FE4" if p == 2 else INK)
a.set_ylim(ymin * 0.62, ymax * 1.7); a.set_xlim(-1, 4 * (NB + G) - G + 1)
a.set_xticks([p * (NB + G) + t for p in range(4) for t in (0, NB)])
a.set_xlabel("position within each paragraph", fontsize=56, labelpad=10)
a.set_xticklabels(["0%", "100%"] * 4, fontsize=40)
for i, t in enumerate(a.get_xticklabels()): t.set_ha("left" if i % 2 == 0 else "right")
a.set_yticks([0.5, 1, 2, 4]); a.minorticks_off(); a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
a.set_ylabel("attention (×10⁻⁴)", fontsize=72, labelpad=12)
for s_ in ["top", "right"]: a.spines[s_].set_visible(False)
a.spines["left"].set_linewidth(3); a.spines["bottom"].set_linewidth(3); a.tick_params(width=4, length=14, labelsize=56)
# line labels placed later, after seeing the data
x3 = 2 * (NB + G)
a.text(x3 + 13, stats["focus"][0][2 * NB + 13] * 1e4 * 1.75, "“focus your\nattention”", color=BLUE, fontsize=50, fontweight="bold", ha="center", va="center", linespacing=1.0)
a.text(x3 + 12, stats["ignore"][0][2 * NB + 12] * 1e4 * 0.55, "“do not pay\nattention”", color=ORANGE, fontsize=50, fontweight="bold", ha="center", va="center", linespacing=1.0)
a.annotate("only\nmentioned", xy=(x[3 * NB - 1] + 0.3, stats["mention"][0][3 * NB - 1] * 1e4), xytext=(3 * (NB + G) + 12, ymin * 1.05),
           color=INK, fontsize=50, fontweight="bold", ha="center", va="center", linespacing=1.0,
           arrowprops=dict(arrowstyle="-", color=INK, lw=3, shrinkA=40, shrinkB=4))
fig.tight_layout(pad=0.6); fig.savefig(OUT + "profile.png", dpi=80, facecolor="white"); fig.savefig(OUT + "profile.jpg", dpi=80, facecolor="white")
json.dump({c: dict(mean=stats[c][0].tolist(), lo=stats[c][1].tolist(), hi=stats[c][2].tolist()) for c in stats} | {"n": n, "files": [f.split("/")[-1] for f in files]},
          open(OUT + "profile_data.json", "w"))
for c in stats: print(c, "P3 mean %.3e" % stats[c][0][2 * NB:3 * NB].mean(), "first-bin per para", np.round(stats[c][0][::NB] * 1e4, 2))
