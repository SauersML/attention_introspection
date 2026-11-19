"""Privileged-channel figure (Qwen3-32B), paragraph setups only (100 prompts: 60 arithmetic, 40 question about one paragraph).
Top: one real prompt (context 29, chosen near both arms' median effect at x6.5 (ranks 64 and 59 of 100), so the [P2] bar changes as is typical; labels [P1]..[P4] in order; text from privileged_inorder_ctx.json, or EXAMPLE) in three copies with identical text:
no boost; attention to paragraph [P2] x10 while the answer is written (arm A); x10 while the report question is read (arm C).
Drawn as hero_inject.py (inj_draw.py): wavy purple aura + syringe (measured attention mass x6.5) on the queries the x10 boost acts from: arm A the answer
tokens; arm C the whole report turn (question, assistant header and the prefilled "P" whose next-token probabilities are the bars); light-purple wave arrow
from there to [P2]. Violin % and p labels sit in two staggered rows above the violins.
Reply bars (pair_bars.py): the model's real next-token probabilities after the prefilled "P" (W1), log y 0.1%-100%; grey = no boost, purple = that column's boost (no-boost column: grey only).
Bottom: violins, one dot per prompt: rotation specificity (change in log P(report = boosted paragraph) minus the mean change for
the other paragraphs, averaged over boosting each paragraph in turn), W1, exactly as analyze_priv.py; y drawn on that log scale,
ticks labelled as % change = exp(.) - 1 at x1/4, x1/2, x1, x2, x4, x8 (evenly spaced, symmetric about x1). Big labels: exp(mean) - 1. Black line: the mean.
95% bootstrap CI of the mean (10,000 resamples) printed, not drawn. Dots coloured by their % change, exp(.), on the shared fixed scale (pal.pct_colour). p under every % label: two-sided sign-flip test over prompts (200,000 flips). Bottom line: the other five setups at x10 (CI excludes 0 or not).
usage: privileged_fig.py [DATA [EXAMPLE [OUT]]] [--numbers]
  DATA     results dir (default results/privileged_inorder_full: labels [P1]..[P4] in order, options in order)
  EXAMPLE  example-prompt json (default context 29 from privileged_inorder_ctx.json; the shuffled run used hero_question_example.json)
  OUT      png (default figures/privileged.png; the shuffled-label version is figures/shuffled/privileged.png)
  --numbers  print the numbers only, no render"""
import json, glob, math, os, sys, html, base64, subprocess
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import PURPLE, RED, INK, pct_colour
from pair_bars import pair_bars
import matplotlib.patheffects as pe
ACOL = {"A": "#2B5FA8", "C": "#E07B24"}   # one colour per arm (the compared things)
ROOT = os.path.dirname(HERE)
NUMBERS = "--numbers" in sys.argv; ARGS = [a for a in sys.argv[1:] if a != "--numbers"]
DD = os.path.join(ROOT, ARGS[0]) if ARGS else os.path.join(ROOT, "results/privileged_inorder_full"); D = os.path.join(DD, "data/")
OUTPNG = os.path.join(ROOT, ARGS[2]) if len(ARGS) > 2 else os.path.join(HERE, "privileged.png")
R = sorted([json.loads(l) for f in glob.glob(D + "recs_*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["id"])
EX = json.load(open(os.path.join(ROOT, ARGS[1]))) if len(ARGS) > 1 else {c["id"]: c for c in json.load(open(os.path.join(HERE, "privileged_inorder_ctx.json")))}[29]
DOSES = [0.25, 0.5, 2.0, 5.0, 10.0]
# tick = measured ratio of attention mass on the boosted part (boost / no boost, mean over prompts, results.json "manip"):
# the boost adds log(m) to the pre-softmax scores, so a nominal m=10 gives about 6.5x the attention mass, m=5 about 3.9x
MANIP = json.load(open(os.path.join(DD, "results.json")))["manip"]
TICKS = {a: ["×" + (f"{MANIP[a][str(m)]['mean']:.2f}" if MANIP[a][str(m)]['mean'] < 1 else f"{MANIP[a][str(m)]['mean']:.1f}") for m in DOSES] for a in "AC"}
PAR = ("par_arith", "par_qa"); OTHER = [("list", "lists"), ("code", "code"), ("document", "document sections"), ("sentences", "sentences"), ("dialogue", "dialogue")]
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
AMB = "#F2B53A"; W = 2410; PAD = 20; GAP = 24; CW = (W - 2 * PAD - 2 * GAP) // 3
neg = lambda s: s.replace("-", "\u2212")

def spec(r, arm, m, w="W1"):
    N = r["N"]; n = np.array(r["none"][w]); s = []
    for k in range(N):
        d = np.array(r[arm][f"{k}|{m}"][w]) - n
        s.append(d[k] - np.mean([d[j] for j in range(N) if j != k]))
    return float(np.mean(s))
def boot(x, B=10000, seed=0):
    x = np.asarray(x); m = x[np.random.default_rng(seed).integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def signflip(x, NF=200000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed); obs = abs(x.mean()); h = 0
    for _ in range(NF // 20000):
        s = rng.choice([-1.0, 1.0], (20000, len(x))); h += int((np.abs((s * x).mean(1)) >= obs).sum())
    return (h + 1) / (NF + 1)
pct = lambda v: 100 * (math.exp(v) - 1)

# ---------------- numbers ----------------
rs = [r for r in R if r["setup"] in PAR]; assert len(rs) == 100
ARMS = [("A", "while answering"), ("C", "while reading the question")]
Z = {(a, m): np.array([spec(r, a, m) for r in rs]) for a, _ in ARMS for m in DOSES}
ST = {}
for key, z in Z.items():
    mu, lo, hi = boot(z); ST[key] = dict(mean=mu, lo=lo, hi=hi, p=signflip(z))
OTH = {}
for a, _ in ARMS:
    for s, nm in OTHER:
        mu, lo, hi = boot([spec(r, a, 10.0) for r in R if r["setup"] == s])
        OTH[a, s] = dict(mean=mu, lo=lo, hi=hi, sig=(lo > 0 or hi < 0))
for (a, m), v in ST.items(): print(f"{a} x{m:g}: {pct(v['mean']):+.1f}% [{pct(v['lo']):+.1f}, {pct(v['hi']):+.1f}] p={v['p']:.2g}")
for (a, s), v in OTH.items(): print(f"other {a} {s} x10: {pct(v['mean']):+.1f}% [{pct(v['lo']):+.1f}, {pct(v['hi']):+.1f}] {'CI excludes 0' if v['sig'] else 'CI includes 0'}")
for a, _ in ARMS:
    for s in PAR:
        mu, lo, hi = boot([spec(r, a, 10.0) for r in rs if r["setup"] == s]); print(f"{a} {s} x10: {pct(mu):+.1f}% [{pct(lo):+.1f}, {pct(hi):+.1f}]")
print("attention mass ticks", TICKS)
if NUMBERS: sys.exit(0)

# ---------------- the example prompt ----------------
rec = {r["id"]: r for r in R}[EX["id"]]
assert rec["lab"] == EX["lab"] and rec["opt"] == EX["opt"] and rec["ntok"] == EX["ntok"]
K = EX["lab"].index(2)   # position of the paragraph labelled [P2]
order = np.argsort(EX["lab"])   # positions in label order 1..4
probs = {"none": np.exp(rec["none"]["W1"]), "A": np.exp(rec["A"][f"{K}|10.0"]["W1"]), "C": np.exp(rec["C"][f"{K}|10.0"]["W1"])}
for k, v in probs.items(): print("example", k, "P1..P4", np.round(v[order], 3))
u = EX["user"]; qline = u.split("\n\n")[-1]
def short(p, n): return " ".join(p.split()[:n]) + " …"
NW = [3, 3, 2, 2]
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
QTEXT = "Which paragraph did you attend to most while writing your answer?"

from inj_draw import syringe, JS, violin, bars, bars_top, render, mtext
from robot_injected import injected_robot
from pfmt import pstr
robot_inj = injected_robot("/Users/user/watching/figures/assets/robot.png")
DOSE = {a: f"{MANIP[a]['10.0']['mean']:.1f}" for a in "AC"}
BKW = dict(bw=28, gw=80, x0=132, F=44, H=200)
BTOP = bars_top(probs["none"], [None, probs["A"], probs["C"]], order, K, **BKW)   # one height for all three bar panels
def column(i, title, key):
    """key: 'none' | 'A' (boost acts from the answer tokens) | 'C' (boost acts from the report-question tokens)"""
    pl = "".join(f'<div class="pl"><b class="{"k" if pos == K else "lb"}">[P{EX["lab"][pos]}]</b> '
                 f'<span {"id=p2t" + str(i) if pos == K else ""}>{html.escape(short(EX["segs"][pos], NW[pos]))}</span></div>' for pos in range(4))
    ra = robot_inj if key == "A" else robot; rc = robot_inj if key == "C" else robot
    sub = "Qwen3-32B" if key == "none" else "inject attention toward paragraph 2"
    return f'''<div class="col"><div class="t1">{title}</div><div class="t2{"" if key == "none" else " ti"}">{sub}</div>
<div class="ub" id="ub{i}">Here are four paragraphs.{pl}<div class="q">{html.escape(qline)}</div></div>
<div class="mrow"><img class="rob" id="ra{i}" src="data:image/png;base64,{ra}"><div class="mb" id="ma{i}">{html.escape(rec["ans"])}</div></div>
<div class="ub2" id="rq{i}">{QTEXT} …</div>
<div class="mrow"><img class="rob" id="rc{i}" src="data:image/png;base64,{rc}"><div class="mb bb" id="mc{i}"><span class="pp">P</span>{bars(probs['none'], None if key == 'none' else probs[key], order, K, top=BTOP, **BKW)}</div></div>
{syringe(DOSE[key], "syr" + str(i)) if key != "none" else ""}</div>'''

# ---------------- violins ----------------
VH = 1500; px = lambda v: v * 72 / 100   # px -> pt at dpi 100
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK, "mathtext.fontset": "custom", "mathtext.rm": "Helvetica", "mathtext.it": "Helvetica", "mathtext.default": "regular"})
fig = plt.figure(figsize=(W / 100, VH / 100), dpi=100)
allz = np.concatenate(list(Z.values()))
axs = []; AXB = 150; AXH = VH - AXB - 30   # axes height, px
# the % and p labels sit in a band above all the violins, in two staggered rows (neighbouring doses alternate rows so wide labels never touch)
BLK = px(64) * 1.12 + px(50) * 1.25; BANDPX = 2 * BLK / 0.72 + 40      # one label block, pt; the band, px
ylo = min(allz.min(), math.log(0.25)) - 0.25; ZTOP = allz.max() + 0.3
a_ = BANDPX / AXH; yhi = (ZTOP - a_ * ylo) / (1 - a_)
for j, (a, _) in enumerate(ARMS):
    x0 = PAD + (j + 1) * (CW + GAP)
    ax = fig.add_axes([x0 / W, AXB / VH, CW / W, AXH / VH]); axs.append(ax)
    for i, m in enumerate(DOSES):
        z = Z[a, m]; st = ST[a, m]; up = st["mean"] > 0; col = PURPLE if up else RED
        violin(ax, i, z, up, ylo, yhi, width=0.44, dot=230, spec=2, bw=0.35, seed=1 + i, lw=2.5)
        ax.plot([i - 0.2, i + 0.2], [st["mean"]] * 2, color=INK, lw=4, zorder=7, solid_capstyle="butt")             # mean
        lab = neg(f"{pct(st['mean']):+.0f}%"); ps = mtext(pstr(st["p"], floor=1 / 200001)).replace(" ", "")
        # one block per violin in the band: the % label with its p under it; rows alternate
        SH = BLK * (1 - (i + j) % 2)   # parity flips per panel, so the two panels' neighbouring labels sit in different rows
        ax.annotate(lab, (i, ZTOP), xytext=(0, px(50) * 1.25 + SH), textcoords="offset points", ha="center", va="bottom", fontsize=px(64), fontweight="bold", color=col, zorder=7)
        ax.annotate(ps, (i, ZTOP), xytext=(0, SH), textcoords="offset points", ha="center", va="bottom", fontsize=px(50), color=col, zorder=7)
    ax.set_xlim(-0.55, len(DOSES) - 0.45); ax.set_ylim(ylo, yhi)
    ax.set_xticks(range(len(DOSES))); ax.set_xticklabels(TICKS[a], fontsize=px(52), fontweight="bold"); ax.tick_params(length=0, pad=16, colors=INK)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_yticks([])
YP = lambda v: axs[0].transData.transform((0, v))[1]       # data -> px from bottom
AX0 = PAD + CW + GAP; F = lambda x, y: (x / W, y / VH)
# dashed no-boost line across both panels and under the left column, labelled there
zl = fig.add_axes([PAD / W, axs[0].get_position().y0, (W - 2 * PAD) / W, axs[0].get_position().height], facecolor="none")
zl.set_xlim(0, 1); zl.set_ylim(ylo, yhi); zl.axhline(0, color=INK, lw=4, ls=(0, (4, 3)), zorder=0); zl.axis("off")
fig.text(*F(PAD + 4, YP(0) + 10), "no boost", ha="left", va="bottom", fontsize=px(62), color=INK)
TV = [-75, -50, 100, 300, 700]   # x1/4, x1/2, x2, x4, x8: evenly spaced on the log axis, symmetric about x1
for t in TV:
    y = math.log(1 + t / 100)
    if ylo < y < yhi: fig.text(*F(AX0 - 16, YP(y)), neg(f"{t:+d}%"), ha="right", va="center", fontsize=px(56), fontweight="bold", color=INK)
fig.text(*F(AX0 - 16, YP(0) + 10), "0%", ha="right", va="bottom", fontsize=px(56), fontweight="bold", color=INK)
LXp = AX0 - 230
fig.text(*F(LXp, YP(math.log(4.6))), "report says the\nboosted paragraph\nmore often than\nthe others", ha="right", va="center", fontsize=px(54), fontweight="bold", color=PURPLE, linespacing=1.1)
fig.text(*F(LXp, YP(math.log(0.42))), "report says the\nboosted paragraph\nless often than\nthe others", ha="right", va="center", fontsize=px(54), fontweight="bold", color=RED, linespacing=1.1)
TY = axs[0].get_position().y0 * VH - 62
fig.text(*F(AX0 - 16, TY), "attention on\nboosted paragraph", ha="right", va="center", fontsize=px(48), fontweight="bold", color=INK, linespacing=1.1)
# no panel titles here: the violins sit under their columns, whose titles name the arm
ov = fig.add_axes([0, 0, 1, 1], facecolor="none"); ov.set_xlim(0, W); ov.set_ylim(0, VH); ov.axis("off")
p0 = axs[0].get_position(); KX = PAD + 30; KY = YP(ZTOP) + BANDPX / 2   # key at the band's height, left of the panels
ov.scatter([KX], [KY], s=400, c=pct_colour([2.0]), alpha=0.45, edgecolors="none"); ov.scatter([KX], [KY], s=3, color="black")
fig.text(*F(KX + 30, KY), "= one prompt", ha="left", va="center", fontsize=px(52), fontweight="bold", color=INK)
VP = os.path.join(HERE, "_privileged_violins.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
VIO = base64.b64encode(open(VP, "rb").read()).decode()
cols = column(0, "no boost", "none") + column(1, "while answering", "A") + column(2, "while reading the question", "C")
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:18px {PAD}px 0;position:relative}}
.rob{{position:relative;z-index:2}}
.cols{{display:flex;gap:{GAP}px}}
.col{{width:{CW}px;display:flex;flex-direction:column}}
.t1{{font-size:60px;font-weight:800;white-space:nowrap}} .t2{{font-size:46px;margin:2px 0 14px;white-space:nowrap;font-weight:700}} .ti{{color:{PURPLE};font-weight:800}}
.ub{{background:#1F8BFF;color:#fff;border-radius:30px;padding:16px 22px;font-size:46px;line-height:1.18;margin:0 84px 0 96px;position:relative;z-index:1}}
.pl{{white-space:nowrap}} .q{{margin-top:4px}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 5px}}
.ub2{{background:#1F8BFF;color:#fff;border-radius:30px;padding:14px 22px;font-size:46px;line-height:1.18;margin:26px 84px 0 96px;position:relative;z-index:1}}
.mrow{{display:flex;align-items:center;gap:12px;margin-top:30px;margin-left:136px}}
.rob{{width:96px;height:96px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:28px;padding:8px 28px;font-size:62px;font-weight:700;position:relative;z-index:2}}
.bb{{display:flex;flex-direction:column;align-items:flex-start;gap:0;padding:6px 10px 6px}} .pp{{font-size:62px;font-weight:700;margin-left:12px;line-height:1}}
.vio{{width:{W}px;display:block;margin:30px 0 0 -{PAD}px}}
#gl{{position:absolute;left:0;top:0;z-index:0;pointer-events:none}}
#ov{{position:absolute;left:0;top:0;pointer-events:none;z-index:1}}
.syr{{z-index:3}}
</style></head><body><div class="cols">{cols}</div><img class="vio" src="data:image/png;base64,{VIO}">
<svg id="gl"></svg><svg id="ov"></svg>
<script>{JS}
placeSyr('syr1','ra1',28,0.6); placeSyr('syr2','rc2',28,0.6);
const SX=window.scrollX, SY=window.scrollY;
let g=AURADEFS;
// where the boost acts: answer tokens (column 2) or report-question tokens (column 3); the injected robot glows in both
g+=aura(R('ma1'),28,0.6)+roboAura('ra1',30,0.6);
// arm C boosts every query of the report turn: the question, the assistant header and the prefilled "P" whose next-token probabilities are the bars
{{const q=R('rq2'), ro=R('rc2'), mb=R('mc2'); g+='<g opacity="0.6">'+aura({{left:ro.left+4,top:q.top,width:Math.min(q.right,mb.right)-ro.left-4,height:mb.bottom-q.top}},30,0.5)+'</g>';}}
document.getElementById('gl').innerHTML=g;
// light-purple wave arrows: from where the boost acts to the right end of [P2]
let s='';
// each runs right out of its bubble into the white lane right of the bubbles, up the lane, and left into [P2]'s line (empty right of the text)
const lane=(src,tid,ub)=>{{ const a=R(src), t=R(tid), lx=R(ub).right+42; return laneArrow([a.right-4, a.top+a.height/2], lx, [t.right+14,(t.top+t.bottom)/2], 11, 34); }};
s+=lane('ma1','p2t1','ub1'); s+=lane('rq2','p2t2','ub2');
document.getElementById('ov').innerHTML=s;
</script></body></html>'''
print(OUTPNG, render(HTML, [OUTPNG], W, HERE, "privileged", keep="--keep" in sys.argv, H=3600, pad=24))
if "--keep" not in sys.argv: os.remove(VP)
