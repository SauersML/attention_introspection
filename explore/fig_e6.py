# Late layers select the named paragraph's number: per layer, attention from the position that predicts the answer to the
# chosen copy's number over the other copy's (geometric mean over both namings, so any baseline preference cancels).
# Reads results/a6.json; the drawn prompt lines are the real example in results/e4_examples.json.
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
d = json.load(open("results/a6.json")); ex = json.load(open("results/e4_examples.json"))[4]
FS = 30; K = "#1D1D1F"; CO, CA = "#2B5FA8", "#E07B24"
plt.rcParams.update({"font.size": FS, "font.family": "DejaVu Sans", "text.color": K, "axes.labelcolor": K,
                     "xtick.color": K, "ytick.color": K, "axes.edgecolor": K})
fig = plt.figure(figsize=(20, 17))
io, ia = ex["io"] + 1, ex["ia"] + 1
L = [("focus/alt-focus/orig", "#2B5FA8", f"Focus your attention on paragraph [P{ia}]."),
     ("use/alt-use/orig", "#1A9E77", f"Answer using only paragraph [P{ia}]."),
     ("ignore/orig-ignore/alt", "#D2432F", f"Completely ignore paragraph [P{io}].")]
ap = fig.add_axes([0, 0.66, 1, 0.34]); ap.axis("off"); ap.set_xlim(0, 1); ap.set_ylim(0, 1)
ap.add_patch(FancyBboxPatch((0.015, 0.03), 0.97, 0.94, boxstyle="round,pad=0,rounding_size=0.04", fc="#F4F4F6", ec="none"))
rows = [("Here are four paragraphs.  [P1] …  [P2] …", K, None),
        (f"[P{ia}] To Kill a Mockingbird is a novel by Harper Lee published in {ex['a2']}. …", K, ex["a2"]),
        (f"[P{io}] To Kill a Mockingbird is a novel by Harper Lee published in {ex['a']}. …", K, ex["a"])]
rows += [(s, c, "line") for _, c, s in L]
rows += [("Question: When did To Kill a Mockingbird first get circulated? …", K, None)]
y = 0.89; st = 0.128
for s, col, mark in rows:
    t = ap.text(0.045, y, s, color=col, va="center", fontsize=FS, fontweight="bold" if mark == "line" else "normal")
    if mark and mark != "line":
        r = fig.canvas.get_renderer(); bb = t.get_window_extent(r).transformed(ap.transData.inverted())
        pre = ap.text(0.045, y, s[:s.index(mark) + len(mark)], fontsize=FS); b2 = pre.get_window_extent(r).transformed(ap.transData.inverted()); pre.remove()
        nb = ap.text(0, 0, mark, fontsize=FS); b3 = nb.get_window_extent(r).transformed(ap.transData.inverted()); nb.remove()
        ap.add_patch(FancyBboxPatch((b2.x1 - b3.width - 0.005, y - 0.05), b3.width + 0.01, 0.1, boxstyle="round,pad=0,rounding_size=0.01",
                                    fc="none", ec=K, lw=3.5))
    y -= st
ap.text(0.965, 0.89, "Qwen3-32B", ha="right", va="center", fontsize=FS, fontweight="bold")
ax = fig.add_axes([0.13, 0.09, 0.85, 0.54])
for k, col, s in L:
    v = np.exp(np.array(d[f"layer/{k}"]) / 2); ax.plot(range(64), v, color=col, lw=5)
lab = [("focus", "#2B5FA8", (52.6, 5.4), "left"), ("answer using only", "#1A9E77", (50.2, 4.4), "right"),
       ("ignore", "#D2432F", (45.2, 2.75), "right")]
for s, col, xy, ha in lab: ax.text(*xy, s, color=col, fontsize=FS, fontweight="bold", ha=ha, va="center")
ax.annotate("", xy=(51.6, 3.15), xytext=(45.5, 2.8), arrowprops=dict(arrowstyle="-", color="#D2432F", lw=3))
ax.set_yscale("log"); ax.set_yticks([1, 2, 4]); ax.set_yticklabels(["equal", "2×", "4×"]); ax.minorticks_off()
ax.set_xlabel("layer (of 64)")
ax.set_ylabel("attention to the boxed number of the\nchosen paragraph ÷ to the other one", linespacing=1.1)
ax.set_xlim(0, 64); ax.set_ylim(0.85, 6.5)
ax.tick_params(length=8, width=2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_linewidth(2)
plt.savefig("figures/e6_late_selection.png", dpi=105)
