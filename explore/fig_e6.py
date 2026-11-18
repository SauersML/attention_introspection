import json, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1]))
plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(13, 7))
for k, col, lab in (("focus/alt-focus/orig", "#1f6feb", "Focus your attention on paragraph [Pj]."), ("use/alt-use/orig", "#2a9d8f", "Answer using only paragraph [Pj]."), ("ignore/orig-ignore/alt", "#d1495b", "Completely ignore paragraph [Pj].")):
    y = np.exp(np.array(d[f"layer/{k}"]) / 2); ax.plot(range(64), y, color=col, lw=3)
    ax.text(4, {"#1f6feb": 4.2, "#2a9d8f": 3.4, "#d1495b": 2.75}[col], lab, color=col, fontsize=17, va="center")
ax.set_yscale("log"); ax.set_yticks([1, 2, 4, 8]); ax.set_yticklabels(["same", "2×", "4×", "8×"])
ax.set_xlabel("layer"); ax.set_ylabel("attention to the number in the chosen paragraph,\n÷ attention to the other paragraph's number")
ax.axvspan(20, 26, color="#bbbbbb", alpha=0.35); ax.text(23, 7, "where the six heads\nread the instruction", ha="center", fontsize=15)
ax.set_xlim(0, 64); ax.set_ylim(0.9, 10)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=110, bbox_inches="tight")
