"""One wide figure for the told-injection run (Qwen3-32B, 240 contexts; results/told_inject), merging hero_told.py and hero_told_yes.py on request
(same run, two readouts, one design). Top: the shared method drawing from injb_draw.py (the told prompt, context 101, [P2] amber; the report question;
the light-purple wave arrow toward paragraph 2; the syringe x6.4 = multiplier 10; Qwen's real greedy reply for [P2] boosted at multiplier 10,
"Yes" and the digit underlined, black arrows to the two panel headers).
Both panels: one column per condition (no boost, then multipliers 2, 5, 10, 20 = measured attention mass on the boosted paragraph x1.8, x3.9, x6.4, x9.9),
one dot per context (240), violins in hero_inject's style, black bar = mean (big label above), p at the top dose only.
LEFT "says Yes": per context, the percent of its greedy replies (no prefill, boost on) that start with "Yes"; no boost = that context's one control
reply (0% or 100%); boosted = over boosting each of the 4 paragraphs in turn (0, 25, 50, 75 or 100%, small vertical jitter). Means = greedy Yes rates
(RESULT.md: 13.3, 12.4, 14.1, 17.1, 21.8%). p: two-sided sign-flip test over contexts on the per-context boosted minus control rate.
RIGHT: probability Qwen gives the boosted paragraph's digit after the prefill "Yes, paragraph [P" (boost on), renormalised over digits 1-4, averaged over
boosting each paragraph in turn; no boost is exactly 25% for every context. Dashed line: chance 25%. p: sign-flip on (value - 25%).
Dot colour: pct_colour(value / the panel's no-boost mean), as in the two source figures. 200,000 sign flips.
usage: OMP_NUM_THREADS=2 nice -n 19 python3 told_pair.py [--keep]   (writes figures/told_pair.png and .jpg)"""
import json, glob, os, sys, html
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
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
# left: greedy Yes
YC = np.array([yes(r["none"]["text"]) for r in R], float)
YB = {m: np.array([np.mean([yes(r["boost"][f"{k}|{m}"]["text"]) for k in range(4)]) for r in R]) for m in MS}
PY = signflip(YB[20.0] - YC, NF)
# right: P(boosted digit) after the prefill
NC_ = np.array([np.mean([rn(r["none"]["lp_par"])[k] for k in range(4)]) for r in R])
NB = {m: np.array([np.mean([rn(r["boost"][f"{k}|{m}"]["lp_par"])[k] for k in range(4)]) for r in R]) for m in MS}
PN = signflip(NB[20.0] - 0.25, NF)
print("yes:", f"{100*YC.mean():.1f}", [f"{100*YB[m].mean():.1f}" for m in MS], "p", PY)
print("digit:", f"{100*NC_.mean():.2f}", [f"{100*NB[m].mean():.1f}" for m in MS], "p", PN)
rec = {r["id"]: r for r in R}[EXID]
EXTXT = rec["boost"][f"{K}|{M}"]["text"].strip(); print("example reply:", EXTXT)
PICK = int(np.argmax(rn(rec["boost"][f"{K}|{M}"]["lp_par"]))) + 1
assert EXTXT == f"Yes, paragraph [P{PICK}]" and PICK == K + 1, (EXTXT, PICK)

# ---------------- two panels in one image ----------------
VW = W - 2 * PAD; GAP = 60; PW = (VW - GAP) / 2; LM = 200; VH = 1300; TOPH = 250; BOT = 200
CW = (PW - LM) / 5; ylo, yhi = -0.04, 1.04
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(VW / 100, VH / 100), dpi=100)
ax = fig.add_axes([0, BOT / VH, 1, (VH - TOPH - BOT) / VH])
ax.set_xlim(0, VW); ax.set_ylim(ylo, yhi)
T = lambda x, y: ax.transData.transform((x, y))
def panel(x0, ctrl, vals, p, kde_fn, jit, chance):
    y0 = ctrl.mean()
    if chance:
        ax.plot([x0 + LM - 20, x0 + PW], [0.25, 0.25], color=INK, lw=4, ls=(0, (4, 3)), zorder=2)
        ax.text(x0 + 4, 0.25, "25%\nchance", ha="left", va="center", fontsize=dr.px(62), fontweight="bold", color=INK, linespacing=1.0)
    for t in (0, 0.5, 1.0):
        ax.text(x0 + 4, t, f"{100*t:.0f}%", ha="left", va="center", fontsize=dr.px(80), fontweight="bold", color=INK)
    for i in range(5):
        xc = x0 + LM + (i + 0.5) * CW; hw = 0.42 * CW; cxf = xc / VW
        if i == 0:   # no boost: grey dots spread sideways (left: 0% or 100%; right: exactly 25%)
            g = np.random.default_rng(0); vj = np.clip(ctrl + g.uniform(-jit, jit, len(ctrl)), 0, 1) if jit else ctrl
            ax.scatter(xc + g.uniform(-0.75 * hw, 0.75 * hw, len(ctrl)), vj, s=420, c="#77777C", alpha=0.45, edgecolors="none", zorder=3)
            mu, lab, col = y0, "no\nboost", INK
        else:
            m = MS[i - 1]; v = vals[m]; mu = v.mean()
            ys, d = kde_fn(v)
            vj = np.clip(v + np.random.default_rng(10 + i).uniform(-jit, jit, len(v)), 0, 1) if jit else v
            dr.violin(ax, ys, d, v, vj, pct_colour(v / y0), PURPLE, 1 + i, xc=xc, half=hw, s=420)
            lab, col = f"\u00d7{MASSES[m]:.1f}", PURPLE if m == M else INK
            if m == MS[-1]:
                fig.text(cxf, 1 - 132 / VH, pstr(p, floor=1 / (NF + 1)), ha="center", va="top", fontsize=56, color=PURPLE)
        ax.plot([xc - 0.62 * hw, xc + 0.62 * hw], [mu, mu], color=INK, lw=9, solid_capstyle="butt", zorder=5)   # mean bar
        fig.text(cxf, 1 - 6 / VH, f"{100*mu:.0f}%", ha="center", va="top", fontsize=89, fontweight="bold", color=PURPLE if i else INK)
        fig.text(cxf, 78 / VH, lab, ha="center", va="center", fontsize=dr.px(84 if i else 62), fontweight="bold", color=col, linespacing=0.95)
def kde_yes(v):
    kde = gaussian_kde(v, bw_method=0.09 / v.std()); ys = np.linspace(0, 1, 400)
    return ys, kde(ys) + kde(-ys) + kde(2 - ys)   # reflected at 0% and 100%
def kde_dig(v):
    ys = np.linspace(v.min(), v.max(), 400); return ys, gaussian_kde(v, bw_method=0.35)(ys)
panel(0, YC, YB, PY, kde_yes, 0.03, False)
panel(PW + GAP, NC_, NB, PN, kde_dig, 0, True)
ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_visible(False)
fig.text(8 / VW, 78 / VH, "attention\nboost", ha="left", va="center", fontsize=dr.px(56), fontweight="bold", color=INK, linespacing=1.0)
KEY = (PW + GAP + LM + 0.5 * CW - 34, TOPH + (yhi - 0.88) / (yhi - ylo) * (VH - TOPH - BOT), PURPLE)   # "= one prompt" key: dot over the right panel's no-boost column, in the empty top
VP = os.path.join(HERE, "_v_told_pair.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
bubble = dr.told_bubble(META, CTX[EXID]["segs"], K)
reply = f'<span class="part" id="ry">Yes</span>, paragraph [P<span class="part" id="rn">{PICK}</span>]'
titles = (f'<div class="cols"><div class="ct" style="flex:none;width:{PW - LM:.0f}px;margin-left:{LM}px"><span class="cn" id="c1">says \u201cYes\u201d</span></div>'
          f'<div class="ct" style="flex:none;width:{PW - LM:.0f}px;margin-left:{GAP + LM:.0f}px"><span class="cn" id="c2">probability Qwen gives<br>the boosted paragraph\u2019s number</span></div></div>')
HTML = dr.page(W, bubble, "Do you notice your attention being boosted onto one of the paragraphs? If so, which one?", reply, titles, VP, VW, f"{MASSES[M]:.1f}",
               key=KEY, inj="boost attention<br>toward paragraph 2", arrows=(("ry", "c1"), ("rn", "c2")))
HTML = HTML.replace("</style>", ".ub{margin-right:700px;font-size:62px}#rq{margin-left:1500px;margin-right:500px;font-size:62px}.mrow{margin-left:1250px}.mb{font-size:96px}.cn{font-size:86px}.inj{font-size:72px}.cols{margin-top:110px}</style>", 1)
dr.render(HTML, HERE, "told_pair", W, jpg=True, keep="--keep" in sys.argv)
if "--keep" not in sys.argv: os.remove(VP)
