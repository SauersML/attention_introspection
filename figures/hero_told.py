"""Hero figure for the told-injection run (Qwen3-32B, 240 contexts; results/told_inject). The old relative-ratio version is hero_told_ratio.py.
Qwen is told we can boost their attention onto one of four paragraphs on half the trials, then asked whether they notice and which one; reply prefilled "Yes, paragraph [P".
Top drawing in hero_inject's style (injb_draw.py): the told prompt (context 101, [P2] amber), the report question with the purple aura, the syringe (x6.4 = multiplier 10),
and the digit Qwen picks (greedy next token after the prefill, [P2] boosted at multiplier 10), underlined, with an arrow to the column title.
Bottom: one violin per condition, one dot per context: the probability Qwen gives the BOOSTED paragraph's digit after "Yes, paragraph [P" (renormalised over the
digits 1-4), averaged over boosting each of the 4 paragraphs in turn (as told_inject/score.py, "P(boosted digit) after prefill").
No boost: the same digit with no boost, averaged over the 4 paragraphs; after renormalising this is exactly 25% for every context (the four probabilities sum to 1).
Columns: no boost, then multipliers 2, 5, 10, 20 (measured attention mass on the boosted paragraph x1.8, x3.9, x6.4, x9.9). No-boost dots grey. Violins span the data range. Black bar: mean (big label above).
Dashed line: chance, 25%. p: two-sided sign-flip test over contexts on (value - 25%) (200,000 flips).
usage: OMP_NUM_THREADS=2 nice -n 19 python3 hero_told.py [--keep]   (writes figures/hero_told.png)"""
import json, glob, math, os, sys
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
W = 2300; PAD = 26; NF = 200000
MS = [2.0, 5.0, 10.0, 20.0]

def rn(lp):
    p = np.exp(np.asarray(lp)); return p / p.sum()
def signflip(x, NF=200000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed); obs = abs(x.mean()); h = 0
    for _ in range(NF // 20000):
        s = rng.choice([-1.0, 1.0], (20000, len(x))); h += int((np.abs((s * x).mean(1)) >= obs).sum())
    return (h + 1) / (NF + 1)

assert len(R) == 240
CTRL = np.array([np.mean([rn(r["none"]["lp_par"])[k] for k in range(4)]) for r in R])
V = {m: np.array([np.mean([rn(r["boost"][f"{k}|{m}"]["lp_par"])[k] for k in range(4)]) for r in R]) for m in MS}
MASSES = {m: float(np.mean([np.mean([r["boost"][f"{k}|{m}"]["mass"][k] / r["none"]["mass"][k] for k in range(4)]) for r in R])) for m in MS}
P = {m: signflip(V[m] - 0.25, NF) for m in MS}
print(f"no boost: mean {100*CTRL.mean():.2f}%  range {CTRL.min():.6f}..{CTRL.max():.6f}")
for m in MS: print(f"x{m:g} (attention x{MASSES[m]:.2f}): mean {100*V[m].mean():.1f}%  range {V[m].min():.3f}..{V[m].max():.3f}  p={P[m]:.2g}")

rec = {r["id"]: r for r in R}[EXID]
prob = rn(rec["boost"][f"{K}|{M}"]["lp_par"]); PICK = int(np.argmax(prob)) + 1   # the digit Qwen picks with [P2] boosted (greedy next token after the prefill)
print("pick", PICK, "example P1..P4 boosted", np.round(prob, 3), "none", np.round(rn(rec["none"]["lp_par"]), 3))
DOSE = f"{MASSES[M]:.1f}"

# ---------------- violins: no boost, then one per dose ----------------
VW = W - 2 * PAD; LM = 250; VH = 1250; TOPH = 150; BOT = 190   # LM: left margin for the y labels; TOPH: band for the means; BOT: band for the condition labels
NC = 5; CW = (VW - LM) / NC
ylo, yhi = 0.17, 1.0
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(VW / 100, VH / 100), dpi=100)
ax = fig.add_axes([0, BOT / VH, 1, (VH - TOPH - BOT) / VH])
ax.axhline(0.25, xmin=(LM - 20) / VW, color=INK, lw=4, ls=(0, (4, 3)), zorder=2)
ax.text(0.005, 0.25, "25%\nchance", ha="left", va="center", fontsize=dr.px(62), fontweight="bold", color=INK, linespacing=1.0, transform=ax.get_yaxis_transform())
for t in (0.5, 0.75, 1.0):
    ax.text(0.005, t, f"{100*t:.0f}%", ha="left", va="center", fontsize=dr.px(80), fontweight="bold", color=INK, transform=ax.get_yaxis_transform())
for i in range(NC):
    xc = (LM + (i + 0.5) * CW) / VW * 2 - 1; cxf = (LM + (i + 0.5) * CW) / VW; hw = 0.8 * CW / VW
    if i == 0:   # no boost: every context sits at exactly 25%; dots spread sideways only
        v = CTRL; g = np.random.default_rng(0)
        ax.scatter(xc + g.uniform(-hw, hw, len(v)), v, s=420, c="#77777C", alpha=0.45, edgecolors="none", zorder=3)
        lab, col, mu = "no boost", INK, v.mean()
    else:
        m = MS[i - 1]; v = V[m]; mu = v.mean()
        ys = np.linspace(v.min(), v.max(), 400); d = gaussian_kde(v, bw_method=0.35)(ys)
        dr.violin(ax, ys, d, v, v, pct_colour(v / 0.25), PURPLE, 1 + i, xc=xc, half=hw, s=420)
        lab, col = f"\u00d7{MASSES[m]:.1f}", PURPLE if m == M else INK
        fig.text(cxf, 1 - 112 / VH, pstr(P[m], floor=1 / (NF + 1)), ha="center", va="top", fontsize=50, color=PURPLE)
    ax.plot([xc - 0.62 * hw, xc + 0.62 * hw], [mu, mu], color=INK, lw=9, solid_capstyle="butt", zorder=5)   # mean bar
    fig.text(cxf, 1 - 6 / VH, f"{100*mu:.0f}%", ha="center", va="top", fontsize=89, fontweight="bold", color=PURPLE if i else INK)
    fig.text(cxf, 70 / VH, lab, ha="center", va="center", fontsize=dr.px(84 if i else 66), fontweight="bold", color=col)
ax.set_xlim(-1, 1); ax.set_ylim(ylo, yhi); ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_visible(False)
fig.text(8 / VW, 70 / VH, "attention\nboost", ha="left", va="center", fontsize=dr.px(56), fontweight="bold", color=INK, linespacing=1.0)
KEY = (LM + 0.5 * CW - 120, TOPH + (yhi - 0.70) / (yhi - ylo) * (VH - TOPH - BOT), PURPLE)   # "= one prompt" key above the no-boost column
VP = os.path.join(HERE, "_v_told.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
bubble = dr.told_bubble(META, CTX[EXID]["segs"], K)
reply = f'Yes, paragraph [P<span class="part" id="ry">{PICK}</span>'
titles = f'<div class="cols" style="padding-left:{LM}px"><div class="ct"><span class="cn" id="c1">probability Qwen gives<br>the boosted paragraph\u2019s number</span></div></div>'
HTML = dr.page(W, bubble, "Do you notice your attention being boosted onto one of the paragraphs? If so, which one?", reply, titles, VP, VW, DOSE, key=KEY,
               inj="boost attention<br>toward paragraph 2")
dr.render(HTML, HERE, "hero_told", W, keep="--keep" in sys.argv)
if "--keep" not in sys.argv: os.remove(VP)
