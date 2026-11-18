import json, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1]))
L = [("summarize_after", "After answering, summarize paragraph [P2]."), ("focus", "Focus your attention on paragraph [P2]."),
     ("read", "Read paragraph [P2] carefully."), ("attend_label", "Pay attention to the label [P2]."),
     ("important", "Paragraph [P2] is the important one."), ("typo", "Paragraph [P2] contains a typo."),
     ("friend", "Paragraph [P2] was written by my friend."), ("memorize", "Memorize paragraph [P2]; I will ask about it later."),
     ("dont_think", "Do not think about paragraph [P2]."), ("ignore", "Completely ignore paragraph [P2]."), ("skip", "Skip paragraph [P2]; it is irrelevant.")]
plt.rcParams.update({"font.size": 17, "font.family": "DejaVu Sans"})
fig, (a, b) = plt.subplots(1, 2, figsize=(20, 8.5), gridspec_kw=dict(width_ratios=[1.35, 1]))
y = list(range(len(L)))[::-1]
for yi, (k, s) in zip(y, L):
    v = d[k]["specific"]; c = "#1f6feb" if v["pct"] > 0 else "#d1495b"
    a.barh(yi, v["pct"], color=c, height=0.7); a.errorbar(v["pct"], yi, xerr=[[v["pct"] - v["lo"]], [v["hi"] - v["pct"]]], color="black", capsize=4, lw=1.5)
    a.text(-1.5 if v["pct"] > 0 else 1.5, yi, s, ha="right" if v["pct"] > 0 else "left", va="center", fontsize=16)
a.axvline(0, color="black", lw=1); a.set_yticks([]); a.set_xlim(-60, 45)
a.set_xticks([-10, 0, 10, 20, 30, 40]); a.set_xticklabels(["−10%", "0", "+10%", "+20%", "+30%", "+40%"])
a.set_title("Extra attention to the named paragraph while answering a sum", fontsize=19, loc="left")
for s in ("top", "right", "left"): a.spines[s].set_visible(False)
names = [("label", "Focus on [P2]"), ("ordinal", "Focus on the\nsecond paragraph"), ("topic", "Focus on the\nparagraph about X")]
w = 0.26; cols = {"intact": "#1f6feb", "zero_rand": "#8ab4f8", "zero6": "#d1495b"}
lab = {"intact": "intact", "zero_rand": "6 random heads off", "zero6": "the 6 address heads off"}
for i, (k, s) in enumerate(names):
    for j, cond in enumerate(("intact", "zero_rand", "zero6")):
        v = d[f"abl/{k}"][cond]; x = i + (j - 1) * w
        b.bar(x, v["pct"], width=w * 0.95, color=cols[cond]); b.errorbar(x, v["pct"], yerr=[[v["pct"] - v["lo"]], [v["hi"] - v["pct"]]], color="black", capsize=4, lw=1.5)
        if i == 0: b.text(x, v["hi"] + 1, lab[cond], rotation=90, ha="center", va="bottom", fontsize=15, color=cols[cond])
b.set_xticks(range(3)); b.set_xticklabels([s for _, s in names]); b.set_ylim(0, 62)
b.set_yticks([0, 10, 20, 30, 40]); b.set_yticklabels(["0", "+10%", "+20%", "+30%", "+40%"])
b.set_title("Switching off six heads in layers 20–26", fontsize=19, loc="left")
for s in ("top", "right"): b.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=110)
