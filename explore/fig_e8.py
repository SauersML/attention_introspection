import json, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1])); Ls = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 63]
plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(13, 7))
for nm, col, lab, yl in (("digit", "#d1495b", "only the digit in “[Pj]”", 1.07), ("after", "#1f6feb", "every token from the digit to the end of the prompt", 0.62), ("last", "#2a9d8f", "only the last prompt token", 0.33)):
    m = np.array([d[f"focus/{nm}/{L}"][0] for L in Ls]); lo = np.array([d[f"focus/{nm}/{L}"][1] for L in Ls]); hi = np.array([d[f"focus/{nm}/{L}"][2] for L in Ls])
    ax.plot(Ls, 100 * m, "o-", color=col, lw=3, ms=8); ax.fill_between(Ls, 100 * lo, 100 * hi, color=col, alpha=0.2)
    ax.text(1, 100 * yl, lab, color=col, fontsize=17, va="center")
ax.set_xlabel("layer where the residual stream is copied in"); ax.set_ylabel("share of the answer switch it carries")
ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"]); ax.set_ylim(-5, 112); ax.set_xlim(-1, 64)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=110, bbox_inches="tight")
