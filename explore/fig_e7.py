import json, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
e3 = json.load(open(sys.argv[1])); e7 = json.load(open(sys.argv[2]))
plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})
fig, (a, b) = plt.subplots(1, 2, figsize=(17, 7.5))
C = {"intact": "#1f6feb", "rand": "#8ab4f8", "six": "#d1495b", "late": "#6a4c93"}
v = e3["abl/label"]
for i, (k, col, lab) in enumerate((("intact", C["intact"], "intact"), ("zero_rand", C["rand"], "6 random\nheads off"), ("zero6", C["six"], "the 6 early\nheads off"))):
    a.bar(i, v[k]["pct"], color=col, width=0.75); a.errorbar(i, v[k]["pct"], yerr=[[v[k]["pct"] - v[k]["lo"]], [v[k]["hi"] - v[k]["pct"]]], color="black", capsize=5)
a.set_xticks(range(3)); a.set_xticklabels(["intact", "6 random\nheads off", "the 6 early\nheads off"])
a.set_yticks([0, 10, 20, 30, 40]); a.set_yticklabels(["0", "+10%", "+20%", "+30%", "+40%"])
a.set_title("“Focus on [P2]” while adding two numbers:\nextra attention to paragraph [P2]", fontsize=18, loc="left")
for i, (k, col, lab) in enumerate((("intact", C["intact"], "intact"), ("early_rand", C["rand"], ""), ("early6", C["six"], ""), ("late_rand", C["rand"], ""), ("late15", C["late"], ""))):
    m = e7[k]["focus_swing"]; b.bar(i, m[0], color=col, width=0.75); b.errorbar(i, m[0], yerr=[[m[0] - m[1]], [m[2] - m[0]]], color="black", capsize=5)
b.set_xticks(range(5)); b.set_xticklabels(["intact", "6 random\nearly off", "the 6 early\nheads off", "15 random\nlate off", "15 late\nreaders off"], fontsize=15)
b.set_ylabel("how strongly the named paragraph's\nnumber wins the answer")
b.set_title("“Focus on [Pj]” when two paragraphs disagree:\nwhich number the model answers", fontsize=18, loc="left")
for ax in (a, b):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
b.set_yticks([0, 10, 20, 30]); b.set_yticklabels(["0", "10", "20", "30 nats"])
plt.tight_layout(); plt.savefig(sys.argv[3], dpi=110, bbox_inches="tight")
