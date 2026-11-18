# Residual patching between two prompts that differ only in the paragraph digit of "Focus your attention on paragraph [Pj].":
# which positions, copied in at which layer, move the answer. Reads results/a8.json; the drawn prompt is the first item
# (results/e4_examples.json, rebuilt from SQuAD with the experiment's seed), its tokens as Qwen3-32B's tokenizer splits them.
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
d = json.load(open("results/a8.json")); ex = json.load(open("results/e4_examples.json"))[0]
Ls = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 63]
FS = 30; K = "#1D1D1F"; CD, CF, CL = "#D2432F", "#2B5FA8", "#1A9E77"
plt.rcParams.update({"font.size": FS, "font.family": "DejaVu Sans", "text.color": K, "axes.labelcolor": K,
                     "xtick.color": K, "ytick.color": K, "axes.edgecolor": K})
fig = plt.figure(figsize=(20, 17.5))
ap = fig.add_axes([0, 0.62, 1, 0.38]); ap.axis("off"); ap.set_xlim(0, 2000); ap.set_ylim(40, 690)
r = fig.canvas.get_renderer()
def w(s):
    t = ap.text(0, 0, s, fontsize=FS); bb = t.get_window_extent(r); t.remove()
    return bb.width * 2000 / (fig.get_figwidth() * fig.dpi)
io, ia = ex["io"] + 1, ex["ia"] + 1
row1 = ["…four paragraphs…", "Focus", "your", "attention", "on", "paragraph", "[", "P", str(io), "]."]
row2 = ["Question", ":", "How", "many", "HD", "channels", "…", "2", "0", "1", "3", "?", "Answer", "…", "⏎"]
pos = {}
def strip(row, y, tag):
    x = 40
    for i, s in enumerate(row):
        cw = w(s) + 30
        dig = tag == 1 and s == str(io); last = tag == 2 and i == len(row) - 1
        ap.add_patch(FancyBboxPatch((x, y - 38), cw, 76, boxstyle="round,pad=0,rounding_size=12",
                                    fc="#FAF0E6" if dig else "#F4F4F6", ec=CD if dig else K if last else "none", lw=4 if dig else 2.5))
        ap.text(x + cw / 2, y, s, ha="center", va="center", fontsize=FS, fontweight="bold" if dig else "normal")
        pos[(tag, i)] = (x, x + cw); x += cw + 14
    return x
y1, y2 = 420, 200
strip(row1, y1, 1); strip(row2, y2, 2)
# the copied-in digit from the other prompt, above
i8 = row1.index(str(io)); x0, x1 = pos[(1, i8)]
ap.add_patch(FancyBboxPatch((x0, 580), x1 - x0, 76, boxstyle="round,pad=0,rounding_size=12", fc="white", ec=CD, lw=4))
ap.text((x0 + x1) / 2, 618, str(ia), ha="center", va="center", fontsize=FS, fontweight="bold", color=CD)
ap.annotate("", xy=((x0 + x1) / 2, y1 + 42), xytext=((x0 + x1) / 2, 578), arrowprops=dict(arrowstyle="-|>", color=CD, lw=4, mutation_scale=35))
ap.text(x0 - 25, 618, f"residual stream copied in from the same\nprompt naming [P{ia}], the copy with {ex['a2']} for {ex['a']}",
        ha="right", va="center", fontsize=FS, color=K, linespacing=1.05)
# brackets under the strips: digit only (red), digit to end (blue), last token (green)
ap.plot([x0, x1], [y1 - 58, y1 - 58], color=CD, lw=7, solid_capstyle="butt")
ap.plot([x0, pos[(1, len(row1) - 1)][1]], [y1 - 76, y1 - 76], color=CF, lw=7, solid_capstyle="butt")
ap.plot([pos[(2, 0)][0], pos[(2, len(row2) - 1)][1]], [y2 - 58, y2 - 58], color=CF, lw=7, solid_capstyle="butt")
lx0, lx1 = pos[(2, len(row2) - 1)]
ap.plot([lx0, lx1], [y2 - 80, y2 - 80], color=CL, lw=7, solid_capstyle="butt")
ap.text(40, 618, "Qwen3-32B", ha="left", va="center", fontsize=FS, fontweight="bold")
# the result
ax = fig.add_axes([0.125, 0.09, 0.85, 0.5])
for nm, col, lab, xy in (("digit", CD, "only the digit", (43.6, 62)), ("after", CF, "every token from\nthe digit to the end", (49, 88)),
                         ("last", CL, "only the last\nprompt token", (40, 22))):
    a = 100 * np.array([d[f"focus/{nm}/{L}"] for L in Ls])
    ax.fill_between(Ls, a[:, 1], a[:, 2], color=col, alpha=0.25, lw=0)
    ax.plot(Ls, a[:, 0], "o-", color=col, lw=4, ms=12)
    ax.text(*xy, lab, color=col, fontsize=FS, va="center", ha="right" if nm != "after" else "left", linespacing=1.05)
ax.set_xlabel("layer whose output is copied in (of 64)")
ax.set_ylabel(f"how far the answer moves\nto {ex['a2']}, the number in [P{ia}]", linespacing=1.1)
ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"]); ax.set_ylim(-4, 106); ax.set_xlim(-1, 64)
ax.tick_params(length=8, width=2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_linewidth(2)
plt.savefig("figures/e8_where_the_choice_lives.png", dpi=105)
