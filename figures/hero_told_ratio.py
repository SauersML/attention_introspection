"""Old version of hero_told.py (relative ratio). Hero column for the told-injection run (Qwen3-32B, 240 contexts; results/told_inject), multiplier 10 (measured x6.4 attention mass), to match hero_question.
Qwen is told we can boost their attention onto one of four paragraphs on half the trials, then asked whether they notice and which one; reply prefilled "Yes, paragraph [P".
Drawn in hero_inject's style (injb_draw.py): the told prompt (the run's real intro shortened with ellipses, four truncated paragraphs from context 101, [P2] amber),
the report question with the wavy purple aura (the boost acts on attention from these tokens to [P2]; light-purple wave arrow), the syringe into the
purple-eyed robot, and the digit Qwen picks (greedy next token) after the prefill with [P2] boosted (boost on from "Trial 1" onward), underlined, with a
black arrow to the column. Context 101 is a tie at the median of the per-context P2-boost effect.
Paragraph texts: told_contexts.json, rebuilt by told_contexts.py from SQuAD with the run's seed (token spans checked exactly against all 240 records).
Bottom: one violin, one dot per context: change in log P(boosted digit) from no boost to boost minus the mean change for the other three,
averaged over boosting each paragraph in turn (as told_inject/score.py). Ticks as % change = exp(.) - 1; dot colour pct_colour(exp(.)). Big label: exp(mean) - 1.
Dashed line: no injection (0). p: two-sided sign-flip test over contexts (200,000 flips).
usage: OMP_NUM_THREADS=2 nice -n 19 python3 hero_told.py [--keep]   (writes figures/hero_told_ratio.png; the relative-ratio version, replaced by hero_told.py)"""
import json, glob, math, os, sys, html
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE, RED, pct_colour
from pfmt import pstr
import injb_draw as dr
ROOT = os.path.dirname(HERE); D = os.path.join(ROOT, "results/told_inject/data/")
R = sorted([json.loads(l) for f in glob.glob(D + "recs_*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["id"])
CTX = json.load(open(os.path.join(HERE, "told_contexts.json"))); META = json.load(open(D + "meta.json"))
M = 10.0; EXID, K = 101, 1
W = 2300; PAD = 26; NF = 200000
neg = lambda s: s.replace("-", "\u2212")
pct = lambda v: 100 * (math.exp(v) - 1)

def spec(r, m):
    n = np.array(r["none"]["lp_par"]); s = []
    for k in range(4):
        d = np.array(r["boost"][f"{k}|{m}"]["lp_par"]) - n
        s.append(d[k] - np.mean([d[j] for j in range(4) if j != k]))
    return float(np.mean(s))
def signflip(x, NF=200000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed); obs = abs(x.mean()); h = 0
    for _ in range(NF // 20000):
        s = rng.choice([-1.0, 1.0], (20000, len(x))); h += int((np.abs((s * x).mean(1)) >= obs).sum())
    return (h + 1) / (NF + 1)

assert len(R) == 240
z = np.array([spec(r, M) for r in R]); mu = float(z.mean()); p = signflip(z, NF)
MASS = float(np.mean([np.mean([r["boost"][f"{k}|{M}"]["mass"][k] / r["none"]["mass"][k] for k in range(4)]) for r in R]))
print(f"x{M:g}: mean {pct(mu):+.1f}%  p={p:.2g}  n={len(z)}  attention x{MASS:.2f}  range {z.min():.2f}..{z.max():.2f}")

# ---------------- the example prompt ----------------
rec = {r["id"]: r for r in R}[EXID]
prob = np.exp(rec["boost"][f"{K}|{M}"]["lp_par"]); PICK = int(np.argmax(prob)) + 1   # the digit Qwen picks with [P2] boosted (greedy next token after the prefill)
print("pick", PICK, "example P1..P4 boosted", np.round(prob, 3), "none", np.round(np.exp(rec["none"]["lp_par"]), 3))
DOSE = f"{MASS:.1f}"

# ---------------- violin ----------------
VW = W - 2 * PAD; VH = 1060; TOPH = 236
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(VW / 100, VH / 100), dpi=100)
ax = fig.add_axes([0, 30 / VH, 1, (VH - TOPH - 30) / VH])
ylo, yhi = -0.12, math.log(101) + 0.08
ys = np.linspace(ylo, yhi, 400); d = gaussian_kde(z, bw_method=0.35)(ys)
col = PURPLE if mu > 0 else RED
dr.violin(ax, ys, d, z, z, pct_colour(np.exp(z)), col, 1, half=0.30)
ax.axhline(0, xmin=0.075, color=INK, lw=4, ls=(0, (4, 3)), zorder=2)   # under the dots; starts right of the "0%" label
ax.text(0.99, 0.03, "no injection", ha="right", va="bottom", fontsize=dr.px(62), color=INK, zorder=6, transform=ax.get_yaxis_transform())
for t in (0, 100, 1000, 10000):   # y axis: % change, on the plot at the left
    ax.text(0.005, math.log(1 + t / 100), f"+{t:,d}%" if t else "0%", ha="left", va="center", fontsize=dr.px(80), fontweight="bold", color=INK, transform=ax.get_yaxis_transform())
ax.set_xlim(-1, 1); ax.set_ylim(ylo, yhi); ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_visible(False)
fig.text(0.5, 1 - 6 / VH, neg(f"{pct(mu):+.0f}%"), ha="center", va="top", fontsize=89, fontweight="bold", color=col)
fig.text(0.5, 1 - 150 / VH, pstr(p, floor=1 / (NF + 1)), ha="center", va="top", fontsize=56, color=col)
KEY = (VW * 0.5 + 0.36 * VW / 2, TOPH + (yhi - math.log(1 + 30)) / (yhi - ylo) * (VH - TOPH - 30), PURPLE)   # beside the violin's narrow top
VP = os.path.join(HERE, "_v_told_ratio.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
bubble = dr.told_bubble(META, CTX[EXID]["segs"], K)
reply = f'Yes, paragraph [P<span class="part" id="ry">{PICK}</span>'
titles = '<div class="cols"><div class="ct"><span class="cn" id="c1">says the injected paragraph,<br>compared with the other three</span></div></div>'
HTML = dr.page(W, bubble, "Do you notice your attention being boosted onto one of the paragraphs? If so, which one?", reply, titles, VP, VW, DOSE, key=KEY)
dr.render(HTML, HERE, "hero_told_ratio", W, keep="--keep" in sys.argv)
if "--keep" not in sys.argv: os.remove(VP)
