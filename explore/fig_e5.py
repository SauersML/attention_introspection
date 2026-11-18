import json, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1])); X = [100, 90, 75, 50, 25, 10, 0]
plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(12, 7.5))
for o, col, lab in (("s", "#1f6feb", "“Split your attention: X% on [Pa] and the rest on [Pb].”"), ("r", "#e8871e", "same split, [Pb] named first")):
    m = [100 * (np.exp(d[f"{o}{x}"]["ratio"][0]) - 1) for x in X]; lo = [100 * (np.exp(d[f"{o}{x}"]["ratio"][1]) - 1) for x in X]; hi = [100 * (np.exp(d[f"{o}{x}"]["ratio"][2]) - 1) for x in X]
    ax.plot(X, m, "o-", color=col, lw=3, ms=9); ax.fill_between(X, lo, hi, color=col, alpha=0.2)
    ax.text(0, 12 if o == "s" else 9.5, lab, color=col, fontsize=16, va="center")
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("share of attention asked for [Pa]"); ax.set_ylabel("change in attention on [Pa] relative to [Pb]")
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_yticks([-10, 0, 10]); ax.set_yticklabels(["−10%", "0", "+10%"]); ax.set_xlim(-3, 103)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=110, bbox_inches="tight")
