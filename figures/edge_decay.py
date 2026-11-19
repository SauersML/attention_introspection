# How sharply the attention instruction's effect stops at the named item's edge (Qwen3-32B, layer 28, attention from the answer tokens).
# Top: four paragraphs (results/qwen3-32b-grid-ci/G.json, 100 contexts x 4 named paragraphs). Bottom: eight single-token words, one per line
# (results/privileged_words, in-order run, 100 contexts x 8 named words). Per token: log(attention under "focus"/"ignore" / attention under
# the matching "only mentioned" prompt), same context, same token; averaged per context at each distance, then over contexts; shown as %.
# Band = 95% bootstrap CI over contexts. Distance = tokens from the named paragraph's first text token (start edge) or last text token
# (end edge, the '.\n\n' token); inside tokens go to the nearer edge. Between paragraphs, 5 tokens were never measured
# ('.\n\n', '[P', 'k', ']', first word): para_positions.json, made by figures/para_positions.py. Words sit exactly 5 tokens apart.
# Usage (repo root): python figures/edge_decay.py
import json, gzip, glob, os, subprocess, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PURPLE = "#5E2B97"; RED = "#C0392B"; INK = "#1D1D1F"; CREAM = "#FAF0E6"
rng = np.random.default_rng(0); NB = 4000

def boot(per):  # per: {context: [log ratios]} -> mean % and 95% CI over contexts
    pc = np.array([np.mean(v) for v in per.values()])
    bs = pc[rng.integers(0, len(pc), (NB, len(pc)))].mean(1)
    f = lambda x: 100 * (np.exp(x) - 1)
    return f(pc.mean()), f(np.percentile(bs, 2.5)), f(np.percentile(bs, 97.5)), len(pc)

# ---------- paragraphs
R = json.load(open(f"{ROOT}/results/qwen3-32b-grid-ci/G.json")); P = json.load(open(f"{ROOT}/results/qwen3-32b-grid-ci/para_positions.json"))
assert all(v["gaps"] == [5, 5, 5] for v in P.values())
def para(cond):
    S, E, IN, NEAR = {}, {}, {}, {}
    for ci, r in enumerate(R):
        sp = P[str(r["id"])]["spans"]
        for k in range(1, 5):
            f = r["vers"][f"{cond}{k}"]["tok_L28"]; m = r["vers"][f"mention{k}"]["tok_L28"]
            lo, hi = sp[k - 1]; st, en = lo - 1, hi  # first text token (unmeasured first word), last text token ('.\n\n', unmeasured)
            for p in range(4):
                for i, (a, b) in enumerate(zip(f[p], m[p])):
                    x = sp[p][0] + i; v = float(np.log(a / b))
                    if p == k - 1:
                        IN.setdefault(ci, []).append(v)
                        if x - st <= en - x: S.setdefault(x - st, {}).setdefault(ci, []).append(v)
                        else: E.setdefault(x - en, {}).setdefault(ci, []).append(v)
                    elif p < k - 1: S.setdefault(x - st, {}).setdefault(ci, []).append(v)
                    else: E.setdefault(x - en, {}).setdefault(ci, []).append(v)
                    d = (st - x) if p < k - 1 else (x - en) if p > k - 1 else None
                    if d is not None and d <= 9: NEAR.setdefault(ci, []).append(v)  # the 5 measured tokens nearest each edge
    return S, E, IN, NEAR
XL = 60
def curve(D, xs):
    xs = [x for x in xs if x in D and len(D[x]) >= 30]
    return np.array(xs), np.array([boot(D[x])[:3] for x in xs])
PD = {}
for c in ("focus", "ignore"):
    S, E, IN, NEAR = para(c)
    PD[c] = dict(S=curve(S, range(-XL, 200)), E=curve(E, range(-200, XL + 1)), inside=boot(IN), near=boot(NEAR))
def far(D, xs):
    out = {}
    for x in xs:
        for ci, vs in D.get(x, {}).items(): out.setdefault(ci, []).extend(vs)
    return boot(out)
for c in ("focus", "ignore"):
    S, E, _, _ = para(c)
    PD[c]["before5"] = far(S, range(-9, -4)); PD[c]["after5"] = far(E, range(5, 10))
    PD[c]["in_start5"] = far(S, range(1, 6)); PD[c]["in_end5"] = far(E, range(-5, 0))
    PD[c]["before_far"] = far(S, range(-XL, -29)); PD[c]["after_far"] = far(E, range(30, XL + 1))

# ---------- words
W = [json.loads(l) for f in sorted(glob.glob(f"{ROOT}/results/privileged_words/data/recs_*.jsonl.gz")) for l in gzip.open(f)]
WD = {}
for c in ("focus", "ignore"):
    D = {}
    for ci, r in enumerate(W):
        I = r["instr"]
        for l in range(1, 9):
            j = r["lab"].index(l); f = I[f"{c}{l}"]["L28"]; m = I[f"mention{l}"]["L28"]
            for s in range(8): D.setdefault(s - j, {}).setdefault(ci, []).append(float(np.log(f[s] / m[s])))
    xs = sorted(D); WD[c] = (np.array(xs) * 5, np.array([boot(D[x])[:3] for x in xs]))

summary = {c: {k: [round(float(t), 1) for t in PD[c][k][:3]] for k in ("inside", "near", "before5", "after5", "in_start5", "in_end5", "before_far", "after_far")} for c in PD}
for c in WD: summary[c]["words_by_lines_away"] = {int(x // 5): [round(float(t), 1) for t in y] for x, y in zip(*WD[c])}
json.dump(summary, open(f"{HERE}/edge_decay.json", "w"), indent=1); print(json.dumps(summary, indent=1))

# ---------- draw
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 33, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.edgecolor": INK, "axes.linewidth": 2.5, "xtick.major.width": 2.5, "ytick.major.width": 2.5, "xtick.major.size": 10, "ytick.major.size": 10})
fig = plt.figure(figsize=(23, 21), dpi=100)
pct = lambda v, _=None: "0%" if abs(v) < 1e-9 else f"{v:+.0f}%"
LW = 5
def band(ax, xs, Y, col):
    ax.fill_between(xs, Y[:, 1], Y[:, 2], color=col, alpha=.22, lw=0); ax.plot(xs, Y[:, 0], color=col, lw=LW, solid_capstyle="round")
def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.axhline(0, color=INK, lw=2.5, zorder=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(pct))

lo = min(PD["ignore"]["S"][1][:, 1].min(), PD["ignore"]["E"][1][:, 1].min()) - 60
hi = max(PD["focus"]["S"][1][:, 2].max(), PD["focus"]["E"][1][:, 2].max()) + 45
L, B, Wd, H, gap = 0.115, 0.555, 0.85, 0.36, 0.03
xS, xE = PD["focus"]["S"][0], PD["focus"]["E"][0]
spanS, spanE = (-XL - 1, xS.max() + 1), (xE.min() - 1, XL + 1)
fr = (spanS[1] - spanS[0]) / ((spanS[1] - spanS[0]) + (spanE[1] - spanE[0]))
a1 = fig.add_axes([L, B, (Wd - gap) * fr, H]); a2 = fig.add_axes([L + (Wd - gap) * fr + gap, B, (Wd - gap) * (1 - fr), H], sharey=a1)
for ax, key, (x0, x1) in ((a1, "S", spanS), (a2, "E", spanE)):
    style(ax); ax.set_xlim(x0, x1); ax.set_ylim(lo, hi)
    inside = (0, x1) if key == "S" else (x0, 0)
    ax.add_patch(Rectangle((inside[0] - .5 if key == "S" else inside[0], lo), inside[1] - inside[0] + .5, hi - lo, color=CREAM, zorder=-2, lw=0))
    g0 = -4.5 if key == "S" else .5
    ax.add_patch(Rectangle((g0, lo), 4, hi - lo, facecolor="white", edgecolor=INK, hatch="//", lw=0, alpha=.6, zorder=-1))
    for c, col in (("focus", PURPLE), ("ignore", RED)): band(ax, PD[c][key][0], PD[c][key][1], col)
    ax.set_xticks([-60, -30, 0, 30] if key == "S" else [-30, 0, 30, 60]); ax.set_xticklabels(["-60", "-30", "start", "+30"] if key == "S" else ["-30", "end", "+30", "+60"])
a2.spines["left"].set_visible(False); a2.tick_params(axis="y", left=False, labelleft=False)
a1.spines["right"].set_visible(False)
for ax, x in ((a1, 1), (a2, 0)):  # break marks
    ax.plot([x - .012, x + .012], [-.02, .02], transform=ax.transAxes, color=INK, lw=3, clip_on=False)
ytop = hi - 12
a1.text(-XL / 2 - 2, ytop, "previous\nparagraph", ha="center", va="top", fontsize=33)
a2.text(XL / 2 + 3, ytop, "next\nparagraph", ha="center", va="top", fontsize=33)
a1.text(spanS[1] - 1, ytop, "named paragraph", ha="right", va="top", fontsize=36, fontweight="bold")
a1.text(-2.5, hi - 4, "not measured", rotation=90, ha="center", va="top", fontsize=28)
a2.text(2.5, hi - 4, "not measured", rotation=90, ha="center", va="top", fontsize=28)
fy = PD["focus"]["S"][1]; iy = PD["ignore"]["S"][1]
a1.text(-58, float(PD["focus"]["S"][1][PD["focus"]["S"][0] < -20, 2].max()) + 12, "told to focus", color=PURPLE, fontsize=38, fontweight="bold", ha="left", va="bottom")
a1.text(14, float(iy[(xS >= 8) & (xS <= 20), 1].min()) - 6, "told to ignore", color=RED, fontsize=38, fontweight="bold", ha="left", va="top")
a1.set_ylabel("attention on each token\nvs. paragraph only mentioned")
fig.text(L + Wd / 2, B - 0.045, "tokens from the edge of the named paragraph", ha="center", va="top")
fig.text(L - 0.09, B + H + 0.025, "Paragraphs: the effect steps down at the paragraph edge", fontsize=42, fontweight="bold", va="bottom")

# words
a3 = fig.add_axes([L, 0.07, Wd, 0.31]); style(a3)
xs = WD["focus"][0]; lo3 = WD["ignore"][1][:, 1].min() - 45; hi3 = WD["focus"][1][:, 2].max() + 60
a3.set_xlim(-37, 37); a3.set_ylim(lo3, hi3)
a3.add_patch(Rectangle((-2.5, lo3), 5, hi3 - lo3, color=CREAM, zorder=-2, lw=0))
for c, col in (("focus", PURPLE), ("ignore", RED)):
    x, Y = WD[c]; band(a3, x, Y, col); a3.plot(x, Y[:, 0], "o", color=col, ms=15)
a3.set_xticks(range(-35, 36, 5)); a3.set_xticklabels([str(abs(v) // 5) if v else "0" for v in range(-35, 36, 5)])
a3.text(0, hi3 - 6, "named\nword", ha="center", va="top", fontsize=36, fontweight="bold")
a3.text(-20, hi3 - 6, "earlier words", ha="center", va="top"); a3.text(20, hi3 - 6, "later words", ha="center", va="top")
a3.text(11, 112, "told to focus", color=PURPLE, fontsize=38, fontweight="bold", ha="left", va="bottom")
a3.text(-36, WD["ignore"][1][0, 1] - 8, "told to ignore", color=RED, fontsize=38, fontweight="bold", ha="left", va="top")
a3.set_ylabel("attention on each word\nvs. word only mentioned")
a3.set_xlabel("words away from the named word (one word per line, 5 tokens apart)")
fig.text(L - 0.09, 0.07 + 0.31 + 0.025, "Single words: the effect spreads to neighboring words", fontsize=42, fontweight="bold", va="bottom")
fig.text(0.985, 0.995, "Qwen3-32B, layer 28", ha="right", va="top", fontsize=33)
fig.savefig(f"{HERE}/edge_decay.png", facecolor="white")
subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "88", "-Z", "2300", f"{HERE}/edge_decay.png", "--out", f"{HERE}/edge_decay.jpg"], check=True, capture_output=True)
