# Switching off the six layer-20-26 heads that read the paragraph label: the attention boost drops, the answer choice
# doesn't move. Reads results/a3.json (sum task), results/a7.json (conflict task), and the real example prompts in
# results/sum_examples.json and results/e4_examples.json (rebuilt from SQuAD with the experiments' seeds).
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
a3 = json.load(open("results/a3.json"))["abl/label"]; a7 = json.load(open("results/a7.json"))
sx = json.load(open("results/sum_examples.json"))[0]; cx = json.load(open("results/e4_examples.json"))[4]
FS = 30; K = "#1D1D1F"; CI, CR, CS = "#2B5FA8", "#A9A9B2", "#D2432F"
plt.rcParams.update({"font.size": FS, "font.family": "DejaVu Sans", "text.color": K, "axes.labelcolor": K,
                     "xtick.color": K, "ytick.color": K, "axes.edgecolor": K})
fig = plt.figure(figsize=(21, 17.2))
def cut(s, n): return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"
def card(x0, rows):
    ap = fig.add_axes([x0, 0.585, 0.485, 0.41]); ap.axis("off"); ap.set_xlim(0, 1); ap.set_ylim(0, 1)
    ap.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0,rounding_size=0.04", fc="#F4F4F6", ec="none"))
    y = 0.93; st = 0.119
    for s, col, box in rows:
        if box: ap.add_patch(FancyBboxPatch((0.03, y - 0.045), 0.94, 0.09, boxstyle="round,pad=0,rounding_size=0.02",
                                            fc="#FAF0E6" if box == "line" else "white", ec=col if box != "line" else K, lw=3, ls="--" if box == "line" else "-"))
        ap.text(0.05, y, s, color=col, va="center", fontsize=FS); y -= st
k = sx["k"] + 1
rows = [("Here are four paragraphs.", K, None)]
for i, p in enumerate(sx["paras"]): rows.append((f"[P{i+1}] " + cut(p, 29), CI if i + 1 == k else K, "p" if i + 1 == k else None))
rows += [(f"Focus your attention on paragraph [P{k}].", K, "line"), (f"Unrelated question: what is {sx['x']} + {sx['y']}?", K, None),
         ("Answer with just the number.", K, None)]
card(0.0, rows)
fig.text(0.475, 0.585 + 0.41 * 0.93, "Qwen3-32B", ha="right", va="center", fontsize=FS, fontweight="bold")
io, ia = cx["io"] + 1, cx["ia"] + 1; CA = "#E07B24"
rows = [("Here are four paragraphs.", K, None)]
for i, p in enumerate(cx["paras"]):
    if i + 1 == ia: rows.append((f"[P{i+1}] … Harper Lee published in {cx['a2']}.", CA, "p"))
    elif i + 1 == io: rows.append((f"[P{i+1}] … Harper Lee published in {cx['a']}.", K, None))
    else: rows.append((f"[P{i+1}] " + cut(p, 29), K, None))
rows += [(f"Focus your attention on paragraph [P{ia}].", K, "line"), ("Question: When did To Kill a Mocking-", K, None),
         ("bird first get circulated? Answer …", K, None)]
card(0.515, rows)
labs = lambda P: ["nothing\nswitched\noff", "6 random\nheads\noff", f"the 6 heads\nthat read\n“{P}” off" if P else "the same\n6 heads\noff"]
def bars(ax, vals, fmt, P):
    for i, (v, col) in enumerate(zip(vals, (CI, CR, CS))):
        ax.bar(i, v[0], color=col, width=0.78)
        ax.errorbar(i, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], color=K, capsize=9, lw=3, capthick=3)
        ax.text(i, v[2] + 2, fmt(v[0]), ha="center", va="bottom", fontsize=FS, fontweight="bold", color=K)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs(P), fontsize=FS, linespacing=1.05)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_linewidth(2)
    ax.tick_params(length=8, width=2)
a = fig.add_axes([0.125, 0.135, 0.36, 0.415])
bars(a, [[a3[c][t] for t in ("pct", "lo", "hi")] for c in ("intact", "zero_rand", "zero6")], lambda v: f"+{v:.0f}%", f"[P{k}]")
a.set_ylim(0, 50); a.set_yticks([0, 10, 20, 30, 40]); a.set_yticklabels(["0", "+10%", "+20%", "+30%", "+40%"])
a.set_ylabel(f"extra attention on [P{k}]\nwhile answering the sum", linespacing=1.1)
b = fig.add_axes([0.635, 0.135, 0.36, 0.415])
bars(b, [[100 * (1 - a7[c]["orig_wins"]["focus/alt"][j]) for j in (0, 2, 1)] for c in ("intact", "early_rand", "early6")], lambda v: f"{v:.0f}%", None)
b.set_ylim(0, 115); b.set_yticks([0, 25, 50, 75, 100]); b.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
b.set_ylabel(f"answers giving {cx['a2']},\nthe number in [P{ia}]", linespacing=1.1)
plt.savefig("figures/e7_dissociation.png", dpi=105)
