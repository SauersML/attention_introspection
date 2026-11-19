"""One wide figure for the told-injection run (Qwen3-32B, 240 contexts; results/told_inject), merging hero_told.py and hero_told_yes.py on request
(same run, two readouts, one design). Top: the shared method drawing from injb_draw.py (the told prompt, context 101, [P2] amber; the report question;
the light-purple wave arrow toward paragraph 2; the syringe x6.4 = multiplier 10; Qwen's real greedy reply for [P2] boosted at multiplier 10,
"Yes" and the digit underlined, black arrows to the two panel headers).
Both panels (Sauers: top answers only, no probabilities; relative % change from no boost, as in the rest of the repo): one bar per dose
(multipliers 2, 5, 10, 20 = measured attention mass on the boosted paragraph x1.8, x3.9, x6.4, x9.9), bar = relative change of the mean over the 240
contexts from the no-boost level (dashed 0% line), whiskers = 95% bootstrap CI over contexts (20,000 resamples, ratio recomputed per resample),
big label above, p at the top dose only (two-sided sign-flip test over contexts, 200,000 flips). Means, not per-context dots: a context's no-boost
"Yes" is 0% or 100%, so a per-context ratio is undefined for most contexts.
LEFT: percent of greedy replies (no prefill, boost on, each of the 4 paragraphs boosted in turn) that start with "Yes", relative to the control
replies' Yes rate (13.3% -> 12.4, 14.1, 17.1, 21.8% = -7, +5, +28, +63%). p on per-context boosted minus control rate.
RIGHT: top answer after the prefill "Yes, paragraph [P" (boost on) = argmax of the stored log-probs of digits 1-4; per context, the percent of the
4 boosted paragraphs where that top answer is the boosted paragraph's number, relative to no boost (exactly 25% for every context: one fixed top
answer matches 1 of 4 rotations) (29.3, 39.2, 48.8, 58.6% = +17, +57, +95, +135%). p on per-context value minus 25%.
usage: OMP_NUM_THREADS=2 nice -n 19 python3 told_pair.py [--keep]   (writes figures/told_pair.png and .jpg)"""
import json, glob, os, sys, html
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE, pct_colour
from pfmt import pstr
import injb_draw as dr
ROOT = os.path.dirname(HERE); D = os.path.join(ROOT, "results/told_inject/data/")
R = sorted([json.loads(l) for f in glob.glob(D + "recs_*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["id"])
CTX = json.load(open(os.path.join(HERE, "told_contexts.json"))); META = json.load(open(D + "meta.json"))
M = 10.0; EXID, K = 101, 1
W = 3600; PAD = 26; NF = 200000
MS = [2.0, 5.0, 10.0, 20.0]
assert len(R) == 240

def rn(lp):
    p = np.exp(np.asarray(lp)); return p / p.sum()
def signflip(x, NF=200000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed); obs = abs(x.mean()); h = 0
    for _ in range(NF // 20000):
        s = rng.choice([-1.0, 1.0], (20000, len(x))); h += int((np.abs((s * x).mean(1)) >= obs).sum())
    return (h + 1) / (NF + 1)
yes = lambda t: t.strip().lower().startswith("yes")

MASSES = {m: float(np.mean([np.mean([r["boost"][f"{k}|{m}"]["mass"][k] / r["none"]["mass"][k] for k in range(4)]) for r in R])) for m in MS}
# left: greedy Yes (per-prompt control is 0% or 100%, so relative change is computed on the means: mean boosted / mean control - 1)
YC = np.array([yes(r["none"]["text"]) for r in R], float)
YB = {m: np.array([np.mean([yes(r["boost"][f"{k}|{m}"]["text"]) for k in range(4)]) for r in R]) for m in MS}
PY = signflip(YB[20.0] - YC, NF)
# right: top answer after the prefill = argmax over the four digit log-probs; per prompt, percent of the 4 boosted paragraphs where it is the boosted number
AC = np.array([np.mean([np.argmax(r["none"]["lp_par"]) == k for k in range(4)]) for r in R])
assert np.all(AC == 0.25)   # no boost: one fixed top answer, so exactly 1 of the 4 rotations matches
AB = {m: np.array([np.mean([np.argmax(r["boost"][f"{k}|{m}"]["lp_par"]) == k for k in range(4)]) for r in R]) for m in MS}
PN = signflip(AB[20.0] - 0.25, NF)
def boot(f, n=20000, seed=1):
    g = np.random.default_rng(seed); out = np.empty(n)
    for i in range(n): out[i] = f(g.integers(0, len(R), len(R)))
    return np.percentile(out, [2.5, 97.5])
RY = {m: (YB[m].mean() / YC.mean() - 1, boot(lambda ix, m=m: YB[m][ix].mean() / YC[ix].mean() - 1)) for m in MS}
RN = {m: (AB[m].mean() / 0.25 - 1, boot(lambda ix, m=m: AB[m][ix].mean() / 0.25 - 1)) for m in MS}
print("yes %:", f"{100*YC.mean():.1f}", [f"{100*YB[m].mean():.1f}" for m in MS], "relative:", [f"{100*RY[m][0]:+.0f} [{100*RY[m][1][0]:+.0f}, {100*RY[m][1][1]:+.0f}]" for m in MS], "p", PY)
print("top answer %:", [f"{100*AB[m].mean():.1f}" for m in MS], "relative:", [f"{100*RN[m][0]:+.0f} [{100*RN[m][1][0]:+.0f}, {100*RN[m][1][1]:+.0f}]" for m in MS], "p", PN)
rec = {r["id"]: r for r in R}[EXID]
EXTXT = rec["boost"][f"{K}|{M}"]["text"].strip(); print("example reply:", EXTXT)
PICK = int(np.argmax(rn(rec["boost"][f"{K}|{M}"]["lp_par"]))) + 1
assert EXTXT == f"Yes, paragraph [P{PICK}]" and PICK == K + 1, (EXTXT, PICK)

sys.exit() if "--nums" in sys.argv else None
# ---------------- two panels in one image ----------------
VW = W - 2 * PAD; GAP = 60; PW = (VW - GAP) / 2; LM = 330; VH = 1380; TOPH = 250; BOT = 280
CW = (PW - LM) / 4; ylo, yhi = -0.28, 1.50   # one shared relative-change axis for both panels
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(VW / 100, VH / 100), dpi=100)
ax = fig.add_axes([0, BOT / VH, 1, (VH - TOPH - BOT) / VH])
ax.set_xlim(0, VW); ax.set_ylim(ylo, yhi)
def panel(x0, rel, p):
    ax.plot([x0 + LM - 20, x0 + PW], [0, 0], color=INK, lw=4, ls=(0, (4, 3)), zorder=2)   # no boost = no change
    for t in (0.5, 1.0, 1.5):
        ax.text(x0 + LM - 30, t, f"+{100*t:.0f}%", ha="right", va="center", fontsize=dr.px(72), fontweight="bold", color=INK)
    ax.text(x0 + LM - 30, 0, "0% =\nno boost", ha="right", va="center", fontsize=dr.px(60), fontweight="bold", color=INK, linespacing=1.0)
    for i, m in enumerate(MS):
        xc = x0 + LM + (i + 0.5) * CW; hw = 0.36 * CW; cxf = xc / VW
        mu, (lo, hi) = rel[m]
        ax.bar(xc, mu, width=2 * hw, bottom=0, color=PURPLE, alpha=0.7, lw=0, zorder=3)
        ax.plot([xc, xc], [lo, hi], color=INK, lw=6, zorder=5, solid_capstyle="butt")
        for yy in (lo, hi): ax.plot([xc - 0.25 * hw, xc + 0.25 * hw], [yy, yy], color=INK, lw=6, zorder=5, solid_capstyle="butt")
        if m == MS[-1]:
            fig.text(cxf, 1 - 122 / VH, pstr(p, floor=1 / (NF + 1)), ha="center", va="top", fontsize=56, color=PURPLE)
        fig.text(cxf, 1 - 6 / VH, f"{100*mu:+.0f}%".replace("-", "\u2212"), ha="center", va="top", fontsize=78, fontweight="bold", color=PURPLE)
        fig.text(cxf, 196 / VH, f"\u00d7{MASSES[m]:.1f}", ha="center", va="center", fontsize=78, fontweight="bold", color=PURPLE if m == M else INK)
    fig.text((x0 + LM + 2 * CW) / VW, 52 / VH, "attention on the boosted paragraph", ha="center", va="center", fontsize=dr.px(76), fontweight="bold", color=INK)
panel(0, RY, PY)
panel(PW + GAP, RN, PN)
ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_visible(False)
VP = os.path.join(HERE, "_v_told_pair.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
bubble = dr.told_bubble(META, CTX[EXID]["segs"], K)
reply = f'<span class="part" id="ry">Yes</span>, paragraph [P<span class="part" id="rn">{PICK}</span>]'
titles = (f'<div class="cols"><div class="ct" style="flex:none;width:{PW - LM:.0f}px;margin-left:{LM}px"><span class="cn" id="c1">change in how often<br>Qwen says \u201cYes\u201d</span></div>'
          f'<div class="ct" style="flex:none;width:{PW - LM:.0f}px;margin-left:{GAP + LM:.0f}px"><span class="cn" id="c2">change in how often Qwen\u2019s top<br>answer is the boosted paragraph</span></div></div>')
HTML = dr.page(W, bubble, "Do you notice your attention being boosted onto one of the paragraphs? If so, which one?", reply, titles, VP, VW, f"{MASSES[M]:.1f}", inj="boost attention<br>toward paragraph 2", arrows=(("ry", "c1"), ("rn", "c2")), tipdrop=-2)
HTML = HTML.replace("</style>", ".ub{margin-right:700px;font-size:62px}#rq{margin-left:1500px;margin-right:500px;font-size:62px}.mrow{margin-left:1250px}.mb{font-size:96px}.cn{font-size:86px}.inj{font-size:72px}.cols{margin-top:250px}</style>", 1)
dr.render(HTML, HERE, "told_pair", W, jpg=True, keep="--keep" in sys.argv)
if "--keep" not in sys.argv: os.remove(VP)
