import json, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1]))
plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(13, 7))
for k, col, lab in (("focus/alt-focus/orig", "#1f6feb", "Focus your attention on paragraph [Pj]."), ("use/alt-use/orig", "#2a9d8f", "Answer using only paragraph [Pj]."), ("ignore/orig-ignore/alt", "#d1495b", "Completely ignore paragraph [Pj].")):
    y = np.exp(np.array(d[f"layer/{k}"])); ax.plot(range(64), y, color=col, lw=3)
    i = int(np.argmax(y)); ax.text(i + 1, y[i], lab, color=col, fontsize=16, va="center")
ax.set_yscale("log"); ax.set_yticks([1, 3, 10, 30]); ax.set_yticklabels(["same", "3×", "10×", "30×"])
ax.set_xlabel("layer"); ax.set_ylabel("attention to the number in the chosen paragraph,\nrelative to the other paragraph's number")
ax.axvspan(20, 26, color="#bbbbbb", alpha=0.35); ax.text(23, 45, "where the six heads\nread the instruction", ha="center", fontsize=15)
ax.set_xlim(0, 75); ax.set_ylim(0.8, 70)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=110, bbox_inches="tight")
