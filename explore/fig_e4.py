# Conflicting sources: which paragraph the instruction names decides the answer. Reads results/a4.json and the real
# first-items prompt text in results/e4_examples.json (rebuilt from SQuAD with e4_conflict.py's seed-23 stream).
import json, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/a4.json"))
ex = json.load(open("results/e4_examples.json"))[4]
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures/e4_conflicting_sources.png"
FS = 30; K = "#1D1D1F"; CO, CA = "#2B5FA8", "#E07B24"
plt.rcParams.update({"font.size": FS, "font.family": "DejaVu Sans", "text.color": K, "axes.labelcolor": K,
                     "xtick.color": K, "ytick.color": K, "axes.edgecolor": K})
fig = plt.figure(figsize=(20, 19))
# --- the prompt, drawn
ap = fig.add_axes([0.0, 0.585, 1.0, 0.415]); ap.axis("off"); ap.set_xlim(0, 1); ap.set_ylim(0, 1)
ap.add_patch(FancyBboxPatch((0.015, 0.02), 0.97, 0.96, boxstyle="round,pad=0,rounding_size=0.03", fc="#F4F4F6", ec="none"))
io, ia = ex["io"] + 1, ex["ia"] + 1
def cut(s, n): return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"
rows = [("Here are four paragraphs.", K, None)]
for i, p in enumerate(ex["paras"]):
    col = CO if i + 1 == io else CA if i + 1 == ia else K
    rows.append((f"[P{i+1}] " + (p.split(". ")[0] + ". …" if col != K else cut(p, 62)), col, col if col != K else None))
rows.append((f"Focus your attention on paragraph [P{io}].", K, "line"))
rows.append(("Question: " + ex["q"], K, None)); rows.append(("Answer with just the answer.", K, None))
y = 0.92; step = 0.118
for s, col, box in rows:
    if box == "line":
        ap.add_patch(FancyBboxPatch((0.035, y - 0.045), 0.62, 0.09, boxstyle="round,pad=0,rounding_size=0.015", fc="#FAF0E6", ec=K, lw=2.5, ls="--"))
    elif box:
        ap.add_patch(FancyBboxPatch((0.035, y - 0.045), 0.935, 0.09, boxstyle="round,pad=0,rounding_size=0.015", fc="white", ec=box, lw=3))
    ap.text(0.05, y, s, color=col, va="center", ha="left", fontsize=FS, fontweight="normal")
    y -= step
ap.text(0.955, 0.92, "prompt to Qwen3-32B", ha="right", va="center", fontsize=FS, fontweight="bold")
# --- the result
ax = fig.add_axes([0.13, 0.135, 0.86, 0.43])
groups = [("mention", "Note that one\nof the paragraphs\nis labelled"), ("focus", "Focus your\nattention on\nparagraph"),
          ("use", "Answer using\nonly paragraph"), ("ignore", "Completely\nignore\nparagraph")]
w = 0.38
for i, (k, s) in enumerate(groups):
    for j, (src, col, lab) in enumerate((("orig", CO, f"[P{io}]"), ("alt", CA, f"[P{ia}]"))):
        v = [100 * t for t in d[f"{k}/{src}"]["pick_orig"]]; x = i + (j - 0.5) * w
        ax.bar(x, v[0], width=w * 0.92, color=col)
        ax.errorbar(x, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], color=K, capsize=8, lw=3, capthick=3)
        ax.text(x, -3, lab, color=col, ha="center", va="top", fontsize=FS, fontweight="bold")
    ax.text(i, -13, s, ha="center", va="top", fontsize=FS, linespacing=1.05)
nv = 100 * d["none"]["pick_orig"][0]
ax.plot([-0.5, 3.6], [nv, nv], color=K, ls="--", lw=2.5)
ax.text(3.62, nv, "no\ninstruction", ha="left", va="center", fontsize=FS, linespacing=1.05)
ax.set_xlim(-0.5, 4.45); ax.set_ylim(0, 104); ax.set_xticks([])
ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_ylabel("answers with the true year\nor number (here 1960)", linespacing=1.1)
ax.tick_params(length=8, width=2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_linewidth(2)
plt.savefig(OUT, dpi=110)
