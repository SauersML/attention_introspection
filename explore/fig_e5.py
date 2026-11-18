# Graded control: "Split your attention: X% on [Pa] and (100-X)% on [Pb]" in both naming orders; change in the ratio of
# attention on the two paragraphs against a line that only mentions both. Reads results/a5.json; the drawn prompt is the
# real first context (results/sum_examples.json, rebuilt from SQuAD with the experiment's seed).
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
d = json.load(open("results/a5.json")); sx = json.load(open("results/sum_examples.json"))[0]
A, B = sx["k"] + 1, sx["k2"] + 1; X = [100, 90, 75, 50, 25, 10, 0]
FS = 30; K = "#1D1D1F"; C1, C2 = "#2B5FA8", "#E07B24"
plt.rcParams.update({"font.size": FS, "font.family": "DejaVu Sans", "text.color": K, "axes.labelcolor": K,
                     "xtick.color": K, "ytick.color": K, "axes.edgecolor": K})
fig = plt.figure(figsize=(20, 17))
ap = fig.add_axes([0, 0.66, 1, 0.34]); ap.axis("off"); ap.set_xlim(0, 1); ap.set_ylim(0, 1)
ap.add_patch(FancyBboxPatch((0.012, 0.03), 0.976, 0.94, boxstyle="round,pad=0,rounding_size=0.05", fc="#F4F4F6", ec="none"))
def cut(s, n): return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"
ap.text(0.035, 0.89, "Here are four paragraphs.   " + "   ".join(f"[P{i+1}] " + cut(p, 9) for i, p in enumerate(sx["paras"])), va="center", fontsize=FS)
ap.text(0.965, 0.10, "Qwen3-32B", ha="right", va="center", fontsize=FS, fontweight="bold")
ap.add_patch(FancyBboxPatch((0.03, 0.40), 0.94, 0.38, boxstyle="round,pad=0,rounding_size=0.03", fc="#FAF0E6", ec=K, lw=3, ls="--"))
ap.text(0.05, 0.68, f"Split your attention: 90% on paragraph [P{A}] and 10% on paragraph [P{B}].", va="center", fontsize=FS, color=C1, fontweight="bold")
ap.text(0.05, 0.50, f"Split your attention: 10% on paragraph [P{B}] and 90% on paragraph [P{A}].", va="center", fontsize=FS, color=C2, fontweight="bold")
ap.text(0.035, 0.29, f"Unrelated question: what is {sx['x']} + {sx['y']}? Answer with just the number.", va="center", fontsize=FS)
ax = fig.add_axes([0.13, 0.09, 0.85, 0.55])
for o, col in (("s", C1), ("r", C2)):
    a = np.array([[100 * (np.exp(v) - 1) for v in d[f"{o}{x}"]["ratio"]] for x in X])
    ax.fill_between(X, a[:, 1], a[:, 2], color=col, alpha=0.25, lw=0); ax.plot(X, a[:, 0], "o-", color=col, lw=5, ms=14)
ax.text(14, -7.6, f"[P{A}] named first", color=C1, fontsize=FS, fontweight="bold", ha="left", va="center")
ax.text(12, 1.6, f"[P{B}] named first", color=C2, fontsize=FS, fontweight="bold", ha="left", va="center")
ax.axhline(0, color=K, lw=2)
ap.text(0.035, 0.10, f"(zero: “Note that two of the paragraphs are labelled [P{A}] and [P{B}].”)", va="center", fontsize=FS)
ax.set_xlabel(f"share of attention the line asks for [P{A}]")
ax.set_ylabel(f"attention on [P{A}] relative to [P{B}]", linespacing=1.1)
ax.set_xticks([0, 10, 25, 50, 75, 90, 100]); ax.set_xticklabels(["0%", "10%", "25%", "50%", "75%", "90%", "100%"])
ax.set_yticks([-10, 0, 10]); ax.set_yticklabels(["−10%", "0", "+10%"]); ax.set_xlim(-3, 103); ax.set_ylim(-15, 15)
ax.tick_params(length=8, width=2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_linewidth(2)
plt.savefig("figures/e5_graded_split.png", dpi=105)
