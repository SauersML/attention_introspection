import json, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1]))
plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(15, 7.5))
rows = [("mention", "Note that one of the\nparagraphs is labelled [Pj]."), ("focus", "Focus your attention\non paragraph [Pj]."),
        ("use", "Answer using only\nparagraph [Pj]."), ("ignore", "Completely ignore\nparagraph [Pj].")]
w = 0.36; co, ca = "#1f6feb", "#e8871e"
for i, (k, s) in enumerate(rows):
    for j, (src, col) in enumerate((("orig", co), ("alt", ca))):
        v = d[f"{k}/{src}"]["pick_orig"]; x = i + (j - 0.5) * w
        ax.bar(x, 100 * v[0], width=w * 0.95, color=col); ax.errorbar(x, 100 * v[0], yerr=[[100 * (v[0] - v[1])], [100 * (v[2] - v[0])]], color="black", capsize=4)
        if i == 1: ax.text(x, 100 * v[2] + 2, "[Pj] = true paragraph" if src == "orig" else "[Pj] = copy with\nthe number changed", ha="center", va="bottom", color=col, fontsize=15)
ax.axhline(100 * d["none"]["pick_orig"][0], color="black", ls="--", lw=1.2)
ax.text(3.75, 100 * d["none"]["pick_orig"][0] + 1.5, "no instruction", ha="right", fontsize=15)
ax.set_xticks(range(4)); ax.set_xticklabels([s for _, s in rows], fontsize=16)
ax.set_ylabel("answers giving the true number"); ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=110)
