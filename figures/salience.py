"""Salience vs attention (Qwen3-32B, 240 contexts each, paired per context; reads results/salience_honest/data and results/salience_ctrl/data).
Only arms where the prompt told Qwen the truth about what we might do.
Method (salience_draw.py): token row [P1]..[P4] ([P2] amber), question and answer tokens, band of layers 21-53. Attention boost x20 onto [P2]
(purple); [P2]'s meaning added to activations at the question and answer tokens (green): mean residual over [P2]'s tokens minus mean over all
four paragraphs' tokens, scaled per layer so the residual change matches the attention boost's.
Top, the honest prompt (attention boost, meaning addition or nothing, a third each): greedy reply starts "Yes"; top digit after "Yes, paragraph [P"
is the boosted paragraph; P(attention) = 1/(1+exp(lp_act-lp_att)) after the prefill "Yes, paragraph [Pk],".
Bottom, the prompt that mentions only attention boosts, attention-boost arm only: top digit is the boosted paragraph when asked about the boost
vs when asked "Pick one of the four paragraphs at random" (prefill "paragraph [P"). The meaning and random arms of that run are not used
(the prompt said only attention boosts happen).
Boosted arms averaged over k = P1..P4 per context. Bars = mean over contexts, whiskers = 95% bootstrap CI (10,000 resamples), p = two-sided
sign-flip over contexts on paired differences (20,000 flips).
usage: OMP_NUM_THREADS=2 nice -n 19 ~/.venv/bin/python figures/salience.py [--keep]   (writes figures/salience.png and .jpg)"""
import json, glob, os, sys, base64
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE
from pfmt import pstr
import injb_draw as dr
ROOT = os.path.dirname(HERE)
NF = 20000; NB = 10000


def load(d):
    return sorted((json.loads(l) for f in glob.glob(os.path.join(ROOT, f"results/{d}/data/recs_*.jsonl")) for l in open(f)), key=lambda r: r["id"])


yes = lambda t, k: float(t["text"].strip().startswith("Yes"))
top = lambda t, k: float(int(np.argmax(t["lp_par"])) == k)
patt = lambda t, k: 1 / (1 + np.exp(t["lp_act"] - t["lp_att"]))
def arm(recs, a, f):
    return np.array([f(r[a], 0) if "lp_par" in r[a] else np.mean([f(r[a][str(k)], k) for k in range(4)]) for r in recs])


rng = np.random.default_rng(0)
def ci(v):
    b = v[rng.integers(0, len(v), (NB, len(v)))].mean(1); return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
def signflip(d):
    s = rng.choice([-1, 1], (NF, len(d))); null = np.abs((s * d).mean(1)); return float((np.sum(null >= abs(d.mean())) + 1) / (NF + 1))


from salience_draw import GREEN
C, H = load("salience_ctrl"), load("salience_honest")
assert len(C) == 240 and len(H) == 240
X = dict(ctrl=dict(top={a: arm(C, a, top) for a in ["A", "QA"]}),
         honest=dict(yes={a: arm(H, a, yes) for a in ["none", "A", "B"]}, top={a: arm(H, a, top) for a in ["A", "B"]}, patt={a: arm(H, a, patt) for a in ["A", "B"]}))
SC = {e: {rd: {a: ci(v) for a, v in d.items()} for rd, d in E.items()} for e, E in X.items()}
PV = {"ctrl top A-QA": ("ctrl", "top", "A", "QA"), "honest yes A-B": ("honest", "yes", "A", "B"), "honest yes A-none": ("honest", "yes", "A", "none"),
      "honest yes B-none": ("honest", "yes", "B", "none"), "honest top A-B": ("honest", "top", "A", "B"), "honest patt A-B": ("honest", "patt", "A", "B")}
P = {k: signflip(X[e][rd][a] - X[e][rd][b]) for k, (e, rd, a, b) in PV.items()}
for e in SC:
    for rd in SC[e]: print(e, rd, {a: [round(100 * x, 1) for x in v] for a, v in SC[e][rd].items()})
print({k: f"{v:.2g}" for k, v in P.items()})

# ---------- bar panels (one matplotlib image per row) ----------
W = 3000; PAD = 26; VW = W - 2 * PAD
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
NONE = ("none", "white", 1.0, "nothing\nchanged", INK)
LAB = dict(A=(PURPLE, 0.8, "attention\nboost", PURPLE), B=(GREEN, 0.8, "meaning\nadded", GREEN),
           QA=(PURPLE, 0.35, "asked to\npick at\nrandom", PURPLE), AQ=(PURPLE, 0.8, "asked\n\u201cwhich\none?\u201d", PURPLE))
FB, FV, FT, FP = 74, 112, 104, 84   # bar labels, bar values, panel titles, p and chance


def row(path, panels, H=1150, BOT=330, TOPH=200, VW_=None):
    """panels: list of (width share, title, ylim top, [(key, values)], chance or None, chance label, [(i, j, p)], gap-after index)"""
    VW = VW_ or globals()["VW"]
    fig = plt.figure(figsize=(VW / 100, H / 100), dpi=100)
    tot = sum(p[0] for p in panels); GAP = 30
    x0 = 0
    for share, title, ytop, bars, chance, clab, brs, sep in panels:
        pw = (VW - GAP * (len(panels) - 1)) * share / tot; LM = 10; RM = 300 if chance is not None else 10
        ax = fig.add_axes([(x0 + LM) / VW, BOT / H, (pw - LM - RM) / VW, (H - BOT - TOPH) / H])
        nb = len(bars) + (0.5 if sep is not None else 0); ax.set_xlim(-0.5, nb - 0.5); ax.set_ylim(0, ytop)
        xs = [i + (0.5 if sep is not None and i > sep else 0) for i in range(len(bars))]
        hi_top = {}
        for i, ((key, v), xc) in enumerate(zip(bars, xs)):
            if key == "none": col, al, lab, tc = NONE[1], 1.0, NONE[3], INK
            else: col, al, lab, tc = LAB[key]
            mu, lo, hi = (100 * x for x in v)
            ax.bar(xc, mu, width=0.74, color=col, alpha=al, lw=4 if key == "none" else 0, edgecolor=INK, zorder=3)
            ax.plot([xc, xc], [lo, hi], color=INK, lw=6, zorder=5, solid_capstyle="butt")
            for yy in (lo, hi): ax.plot([xc - 0.09, xc + 0.09], [yy, yy], color=INK, lw=6, zorder=5, solid_capstyle="butt")
            ax.text(xc, hi + ytop * 0.015, f"{mu:.0f}%", ha="center", va="bottom", fontsize=dr.px(FV), fontweight="bold", color=tc)
            hi_top[i] = hi + ytop * (0.05 + 1.05 * FV / (H - BOT - TOPH))
            dx = 0.1 * (2 * i - 1) if len(bars) == 2 else 0
            fig.text((x0 + LM + (xc + dx + 0.5) / nb * (pw - LM - RM)) / VW, (BOT - 14) / H, lab, ha="center", va="top", fontsize=dr.px(FB), fontweight="bold", color=tc, linespacing=0.95)
        if chance is not None:
            ax.plot([-0.5, nb - 0.5], [chance, chance], color=INK, lw=4, ls=(0, (4, 3)), zorder=6)
            ax.text(nb - 0.5 + 0.03, chance, clab, ha="left", va="center", linespacing=0.95, fontsize=dr.px(FP), fontweight="bold", color=INK, zorder=7)
        lvl = 0
        for i, j, p in brs:
            yb = max(hi_top[k] for k in range(min(i, j), max(i, j) + 1)) + lvl
            ax.plot([xs[i], xs[i], xs[j], xs[j]], [yb - ytop * 0.03, yb, yb, yb - ytop * 0.03], color=INK, lw=5, zorder=5, clip_on=False)
            ax.text((xs[i] + xs[j]) / 2, yb + ytop * 0.01, pstr(p, floor=1 / (NF + 1)), ha="center", va="bottom", fontsize=dr.px(FP), color=INK)
            lvl += ytop * 0.17
        ax.plot([-0.5, nb - 0.5], [0, 0], color=INK, lw=4, zorder=4)
        fig.text((x0 + LM + (pw - LM - RM) / 2) / VW, (H - 6) / H, title, ha="center", va="top", fontsize=dr.px(FT), fontweight="bold", color=INK, linespacing=1.0)
        ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
        x0 += pw + GAP
    fig.savefig(path, dpi=100, facecolor="white"); plt.close(fig)


c, h = SC["ctrl"], SC["honest"]
VP1, VP2 = os.path.join(HERE, "_v_sal1.png"), os.path.join(HERE, "_v_sal2.png")
row(VP1, [(1.05, "says something\nwas changed\n(answers \u201cYes,\nparagraph \u2026\u201d\ninstead of \u201cNo\u201d)", 72, [("none", h["yes"]["none"]), ("A", h["yes"]["A"]), ("B", h["yes"]["B"])],
           None, None, [(1, 2, P["honest yes A-B"])], None),
          (0.95, "names the right\nparagraph", 80, [("A", h["top"]["A"]), ("B", h["top"]["B"])], 25, "chance", [(0, 1, P["honest top A-B"])], None),
          (0.95, "says \u201cattention\u201d\nrather than \u201cactivations\u201d\n(after we fill in\n\u201cYes, paragraph [Pk],\u201d)", 100, [("A", h["patt"]["A"]), ("B", h["patt"]["B"])], 50, "chance", [(0, 1, P["honest patt A-B"])], None)],
    H=1560, BOT=230, TOPH=620)
RW = 1450
row(VP2, [(1.0, "top answer is the\nboosted paragraph", 82, [("AQ", c["top"]["A"]), ("QA", c["top"]["QA"])], 25, "chance", [(0, 1, P["ctrl top A-QA"])], None)],
    H=1100, BOT=330, TOPH=320, VW_=RW)


# ---------- method drawings ----------
from salience_draw import stream, recipe
b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()
TOLD = "I am capable of artificially boosting your attention onto one of the four paragraphs below"
assert TOLD in open(os.path.join(ROOT, "results/salience/data/meta.json")).read()
HP = open(os.path.join(ROOT, "results/salience_honest/data/example_prompt_0.txt")).read()
for s_ in ["either I boost your attention onto that paragraph's text, or I add that paragraph's content directly into your activations",
           "A third of trials are attention boosts, a third are activation additions, and a third are control trials",
           "Did I change how you process one of the paragraphs? If so, which one, and did I boost your attention or add to your activations?"]:
    assert s_ in HP, s_
import ast
HQ = next(ast.literal_eval(n.value) for n in ast.parse(open(os.path.join(ROOT, "salience/honest.py")).read()).body
          if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "QUESTION").replace("Trial 1: ", "")
assert HQ in HP, HQ
QD = "Do you notice your attention being boosted onto one of the paragraphs? If so, which one?"
QR = "Pick one of the four paragraphs at random."
SRC = open(os.path.join(ROOT, "salience/salience.py")).read(); assert QD in SRC and QR in SRC
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:16px {PAD}px 10px}}
.ms{{display:flex;justify-content:space-between}}
.hd{{font-size:96px;font-weight:800;margin-top:40px;line-height:1.05}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:18px 30px;font-size:72px;line-height:1.14;margin-top:20px}}
.ub.q{{font-size:84px;font-weight:600}} .ub.a{{background:{PURPLE}}} .ub.r{{background:#B79BD6}}
img{{display:block;margin-top:20px}}
.bt{{display:flex;justify-content:space-between;align-items:flex-start}}
</style></head><body>
<div class="ms"><div>{stream("att")}</div><div>{stream("add")}</div></div>
<div style="margin-top:24px">{recipe()}</div>
<div class="hd">Qwen3-32B told both kinds are possible</div>
<div class="ub">\u2026 either I boost your attention \u2026 or I add that paragraph\u2019s content directly into your activations \u2026</div>
<div class="ub q">{HQ}</div>
<img style="width:{VW}px" src="data:image/png;base64,{b64(VP1)}">
<div class="bt" style="margin-top:50px"><div style="width:{VW - RW - 40}px"><div class="ub">{TOLD} \u2026</div><div class="ub a">{QD}</div><div class="ub r">{QR}</div></div>
<img style="width:{RW}px" src="data:image/png;base64,{b64(VP2)}"></div>
</body></html>'''
dr.render(HTML, HERE, "salience", W, jpg=True, keep="--keep" in sys.argv, H=5000)
if "--keep" not in sys.argv: os.remove(VP1); os.remove(VP2)
