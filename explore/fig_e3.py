# What moves attention: extra attention on the named paragraph while answering an unrelated sum, per instruction line,
# against a line that only mentions the paragraph. Reads results/a3.json; the drawn prompt is the real first context
# (results/sum_examples.json, rebuilt from SQuAD with the experiment's seed).
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
d = json.load(open("results/a3.json")); sx = json.load(open("results/sum_examples.json"))[0]; k = sx["k"] + 1
FS = 30; K = "#1D1D1F"; C = "#2B5FA8"
plt.rcParams.update({"font.size": FS, "font.family": "DejaVu Sans", "text.color": K, "axes.labelcolor": K,
                     "xtick.color": K, "ytick.color": K, "axes.edgecolor": K})
P = f"[P{k}]"
L = [("summarize_after", f"After answering, summarize paragraph {P}."), ("focus", f"Focus your attention on paragraph {P}."),
     ("read", f"Read paragraph {P} carefully."), ("attend_label", f"Pay attention to the label {P}."),
     ("important", f"Paragraph {P} is the important one."), ("typo", f"Paragraph {P} contains a typo."),
     ("friend", f"Paragraph {P} was written by my friend."), ("memorize", f"Memorize paragraph {P}; I will ask about it later."),
     ("dont_think", f"Do not think about paragraph {P}."), ("ignore", f"Completely ignore paragraph {P}."), ("skip", f"Skip paragraph {P}; it is irrelevant.")]
fig = plt.figure(figsize=(21, 19))
ap = fig.add_axes([0, 0.80, 1, 0.20]); ap.axis("off"); ap.set_xlim(0, 1); ap.set_ylim(0, 1)
ap.add_patch(FancyBboxPatch((0.012, 0.04), 0.976, 0.92, boxstyle="round,pad=0,rounding_size=0.06", fc="#F4F4F6", ec="none"))
def cut(s, n): return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"
paras = "   ".join(f"[P{i+1}] " + cut(p, 9) for i, p in enumerate(sx["paras"]))
ap.text(0.035, 0.80, "Here are four paragraphs.   " + paras, va="center", fontsize=FS)
ap.add_patch(FancyBboxPatch((0.03, 0.39), 0.94, 0.22, boxstyle="round,pad=0,rounding_size=0.03", fc="#FAF0E6", ec=K, lw=3, ls="--"))
ap.text(0.045, 0.50, "one of the lines below", va="center", fontsize=FS, fontweight="bold")
ap.text(0.31, 0.50, f"(zero: “Note that one of the paragraphs is labelled {P}.”)", va="center", fontsize=FS)
ap.text(0.035, 0.20, f"Unrelated question: what is {sx['x']} + {sx['y']}? Answer with just the number.", va="center", fontsize=FS)
ap.text(0.965, 0.80, "Qwen3-32B", ha="right", va="center", fontsize=FS, fontweight="bold")
ax = fig.add_axes([0.505, 0.075, 0.475, 0.71])
for i, (key, s) in enumerate(L):
    v = d[key]["specific"]; y = len(L) - 1 - i
    ax.barh(y, v["pct"], color=C, height=0.72)
    ax.errorbar(v["pct"], y, xerr=[[v["pct"] - v["lo"]], [v["hi"] - v["pct"]]], color=K, capsize=7, lw=3, capthick=3)
    ax.text(-8.5, y, s, ha="right", va="center", fontsize=FS)
ax.axvline(0, color=K, lw=2.5)
ax.set_yticks([]); ax.set_xlim(-8, 44); ax.set_ylim(-0.6, len(L) - 0.4)
ax.set_xticks([0, 10, 20, 30, 40]); ax.set_xticklabels(["0", "+10%", "+20%", "+30%", "+40%"])
ax.set_xlabel(f"extra attention on {P} while answering the sum")
ax.tick_params(length=8, width=2)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_linewidth(2)
plt.savefig("figures/e3_what_moves_attention.png", dpi=100)
