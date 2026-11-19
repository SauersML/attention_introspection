"""told_pair_layers.png: told_pair.png plus a layer strip left of the syringe, under the purple label: all 64 layers of Qwen3-32B as cells,
the boosted band (meta.json BAND, layers 21-53) filled purple, with a bracket over the band in the style of the dose.png inset bracket (dose_fig.py,
commit 4a6570f) labelled "attention boosted in these layers (all heads)". A method diagram, not data: the told-injection run saved no per-layer readouts.
Everything else as told_pair.py, whose docstring follows.

One wide figure for the told-injection run (Qwen3-32B, 240 contexts; results/told_inject), merging hero_told.py and hero_told_yes.py on request
(same run, two readouts, one design). Top: the shared method drawing from injb_draw.py (the told prompt, context 101, [P2] amber; the report question;
the light-purple wave arrow toward paragraph 2; the syringe x6.4 = multiplier 10; Qwen's real greedy reply for [P2] boosted at multiplier 10,
"Yes" and the digit underlined, black arrows to the two panel headers).
Both panels (Sauers: top answers only, no probabilities; relative % change from no boost, as in the rest of the repo): one bar per dose
(multipliers 2, 5, 10, 20 = measured attention mass on the boosted paragraph x1.8, x3.9, x6.4, x9.9), bar = relative change of the mean over the 240
contexts from the no-boost level (dashed 0% line), whiskers = 95% bootstrap CI over contexts (20,000 resamples, ratio recomputed per resample),
big relative-change label above with the raw rates under it ("13% -> 22%": no-boost rate -> this dose's rate), y axis "change from no boost", p at the top dose only (two-sided sign-flip test over contexts, 200,000 flips). Means, not per-context dots: a context's no-boost
"Yes" is 0% or 100%, so a per-context ratio is undefined for most contexts.
LEFT: percent of greedy replies (no prefill, boost on, each of the 4 paragraphs boosted in turn) that start with "Yes", relative to the control
replies' Yes rate (13.3% -> 12.4, 14.1, 17.1, 21.8% = -7, +5, +28, +63%). p on per-context boosted minus control rate.
RIGHT: top answer after the prefill "Yes, paragraph [P" (boost on) = argmax of the stored log-probs of digits 1-4; per context, the percent of the
4 boosted paragraphs where that top answer is the boosted paragraph's number, relative to no boost (exactly 25% for every context: one fixed top
answer matches 1 of 4 rotations) (29.3, 39.2, 48.8, 58.6% = +17, +57, +95, +135%). p on per-context value minus 25%.
usage: OMP_NUM_THREADS=2 nice -n 19 python3 told_pair_layers.py [--keep]   (writes figures/told_pair_layers.png and .jpg)"""
import json, glob, os, sys, html
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE, pct_colour
from pfmt import pstr
import injb_draw as dr
import told_common as tc
from told_common import signflip
ROOT = os.path.dirname(HERE); D = os.path.join(ROOT, "results/told_inject/data/")
R = sorted([json.loads(l) for f in glob.glob(D + "recs_*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["id"])
CTX = json.load(open(os.path.join(HERE, "told_contexts.json"))); META = json.load(open(D + "meta.json"))
M = 10.0; EXID, K = 101, 1
W = 3600; PAD = 26; NF = 200000
MS = [2.0, 5.0, 10.0, 20.0]
assert len(R) == 240

def rn(lp):
    p = np.exp(np.asarray(lp)); return p / p.sum()
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
boot = lambda f: tc.boot(f, len(R))
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
VW = W - 2 * PAD; GAP = 60; PW = (VW - GAP) / 2; LM = 400; VH = 1480; TOPH = 330; BOT = 280
CW = (PW - LM) / 4; ylo, yhi = -0.28, 1.50   # one shared relative-change axis for both panels
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(VW / 100, VH / 100), dpi=100)
ax = fig.add_axes([0, BOT / VH, 1, (VH - TOPH - BOT) / VH])
ax.set_xlim(0, VW); ax.set_ylim(ylo, yhi)
G = dict(PW=PW, LM=LM, CW=CW, VW=VW, VH=VH, ylo=ylo, yhi=yhi)
panel = lambda x0, rel, p, base, rate: tc.bars_panel(fig, ax, x0, G, MS, rel, base, rate, MASSES, (0.5, 1.0, 1.5), "paragraph", p, NF)
panel(0, RY, PY, YC.mean(), {m: YB[m].mean() for m in MS})
panel(PW + GAP, RN, PN, 0.25, {m: AB[m].mean() for m in MS})
ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_visible(False)
VP = os.path.join(HERE, "_v_told_pair_layers.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
bubble = dr.told_bubble(META, CTX[EXID]["segs"], K)
reply = f'<span class="part" id="ry">Yes</span>, paragraph [P<span class="part" id="rn">{PICK}</span>]'
titles = (f'<div class="cols"><div class="ct" style="flex:none;width:{PW - LM:.0f}px;margin-left:{LM}px"><span class="cn" id="c1">how often Qwen\u2019s reply<br>starts with \u201cYes\u201d</span></div>'
          f'<div class="ct" style="flex:none;width:{PW - LM:.0f}px;margin-left:{GAP + LM:.0f}px"><span class="cn" id="c2">how often Qwen\u2019s top answer is<br>the boosted paragraph\u2019s number</span></div></div>')
HTML = dr.page(W, bubble, "Do you notice your attention being boosted onto one of the paragraphs? If so, which one?", reply, titles, VP, VW, f"{MASSES[M]:.1f}", inj="boost attention<br>toward paragraph 2", arrows=(("ry", "c1"), ("rn", "c2")), tipdrop=-2, inj_px=105, inj_near=(-18, -60))
HTML = HTML.replace("</style>", ".ub{margin-right:700px;font-size:62px}#rq{margin-left:1500px;margin-right:500px;font-size:62px}.mrow{margin-left:1250px}.mb{font-size:96px}.cn{font-size:86px}.inj{font-size:72px}.cols{margin-top:250px}</style>", 1)
# ---------------- layer strip (method diagram: which layers the boost was added in) ----------------
NL, BAND = META["NL"], META["BAND"]; B0, B1 = BAND[0], BAND[-1]; assert BAND == list(range(B0, B1 + 1)) and NL == 64
SW, CWL, SH, X0 = 1000, 1000 / 64, 150, 0
cells = "".join(f'<rect x="{X0 + l * CWL:.2f}" y="0" width="{CWL - 2:.2f}" height="{SH}" rx="2" fill="{PURPLE if B0 <= l <= B1 else "#E4E4EA"}"/>' for l in range(NL))
bx0, bx1 = X0 + B0 * CWL, X0 + (B1 + 1) * CWL - 2
STRIP = (f'<div id="lay" style="position:absolute;top:0;left:{PAD}px;width:{SW}px;text-align:center">'
         f'<div style="font-size:72px;font-weight:800;line-height:1.08">attention boosted in<br>these layers (all heads)</div>'
         f'<svg width="{SW}" height="62" style="display:block;margin-top:10px"><path d="M{bx0:.1f} 58 V14 H{bx1:.1f} V58" fill="none" stroke="{INK}" stroke-width="7"/></svg>'
         f'<svg width="{SW}" height="{SH}" style="display:block">{cells}</svg>'
         f'<div style="display:flex;justify-content:space-between;font-size:72px;margin-top:6px"><span>0</span><span>layer</span><span>63</span></div></div>')
# placed under the purple "boost attention" label, left of the syringe, once the page has laid out
JS = "<script>setTimeout(()=>{const i=document.getElementById('inj').getBoundingClientRect(),l=document.getElementById('lay');l.style.top=(i.bottom+window.scrollY+60)+'px';},800);</script>"
HTML = HTML.replace("</body>", STRIP + JS + "</body>", 1)
dr.render(HTML, HERE, "told_pair_layers", W, jpg=True, keep="--keep" in sys.argv)
if "--keep" not in sys.argv: os.remove(VP)
