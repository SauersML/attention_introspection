"""Steered vs unsteered: change in log-prob that Qwen3-32B names the boosted paragraph when asked where it attended,
minus the mean change for the other paragraphs (the question paragraph excluded in QA). Arithmetic prompts, answers identical to no boost.
Paired per question; p from a sign-flip permutation test (2,000,000 random flips, +1 correction), no distribution assumed.
Drawn on the log-prob scale, labelled as % change = exp(.) - 1 (relative to the other paragraphs); big label = the mean;
dots coloured by their % change on the shared scale (pal.pct_colour). Labels [P1]..[P4] in slot order and options in order (attn_intro.py).
Style as hero_inject.py (drawing code in inj_draw.py): one real prompt on top (A2.json, attention x4.9: the arithmetic prompt whose value is
nearest the median among those whose boosted paragraph has a no-boost report probability between 5% and 95%, so its bar change is readable;
ties broken by the larger no-boost probability). The boost acts on attention from the
task line and the answer tokens to the boosted paragraph (attn_intro.py run(): on for task+answer queries, off for the report question), so the
task line, the answer and the injected robot get the wavy purple aura and the light-purple wave arrow runs from the task line to the paragraph.
Reply bars: the report's next-token probabilities after the prefilled "P" for P1..P4 (A.json no boost, grey; A2.json boosted, purple), log y.
usage: OMP_NUM_THREADS=2 nice -n 19 python trace3.py   (writes figures/trace3.png and .jpg)"""
import json, sys, os, math, html, base64
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.dirname(os.path.abspath(__file__)) + "/"; R = os.path.join(OUT, "..", "results", "qwen3-32b") + "/"
sys.path.insert(0, OUT); from pal import INK, PURPLE, RED, pct_colour
from pfmt import pstr
from robot_injected import injected_robot
from inj_draw import syringe, JS, violin, bars, render, mtext, AMB, ROBOT
A = json.load(open(R + "A.json")); A2 = json.load(open(R + "A2.json")); byid = {r["id"]: r for r in A}
def spec(d, r):
    o = [m for m in range(4) if m != r["k"] and (r["task"] != "qa" or m != r["j"])]; return d[r["k"]] - np.mean([d[m] for m in o])
def boot(x, NB=20000):
    g = np.random.default_rng(0); s = x[g.integers(0, len(x), (NB, len(x)))].mean(1); return x.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)
NFLIP = 2_000_000
def signflip(x, N=NFLIP, chunk=200_000):
    g = np.random.default_rng(1); obs = abs(x.mean()); hit = 0
    for _ in range(N // chunk):
        s = g.choice([-1.0, 1.0], size=(chunk, len(x))); hit += int((np.abs((s * x).mean(1)) >= obs - 1e-15).sum())
    return (hit + 1) / (N + 1)
cols = []
for b, lab in [(1.0, "attention ×2.3"), (2.0, "attention ×4.9")]:
    rs = [r for r in A2 if r["task"] == "arith" and r["b"] == b]; bs = [byid[r["id"]] for r in rs]
    keep = [(r, b0) for r, b0 in zip(rs, bs) if r["ans"].strip() == b0["ans"].strip()]
    x = np.array([spec(np.array(r["rep_lp"]) - np.array(b0["rep_lp"]), r) for r, b0 in keep])
    p = signflip(x); cols.append((lab, x, p, keep)); print(lab, len(x), boot(x), "p", p)
for lab, x, p, _ in cols:
    m, lo, hi = boot(x); print(lab, f"{100*(math.exp(m)-1):+.1f}% [{100*(math.exp(lo)-1):+.1f}, {100*(math.exp(hi)-1):+.1f}]")
neg = lambda t: t.replace("-", "\u2212")

# ---------------- the example prompt (attention x4.9) ----------------
_, x2, _, keep2 = cols[1]; med = np.median(x2)
ok = [i for i in range(len(x2)) if 0.05 <= keep2[i][1]["rep"][keep2[i][0]["k"]] <= 0.95]   # bars readable: the boosted paragraph neither near 0% nor near 100% without the boost
ie = min(ok, key=lambda i: (round(abs(x2[i] - med), 9), -keep2[i][1]["rep"][keep2[i][0]["k"]]))
ex, ex0 = keep2[ie]; K = ex["k"]
print("example", ex["id"], "boosted [P%d]" % (K + 1), "value", round(x2[ie], 3), "median", round(med, 3), "answer", ex["ans"],
      "report no boost", np.round(ex0["rep"], 3), "boosted", np.round(ex["rep"], 3))
chunks = ex["user"].split("\n\n"); paras = [c.split("] ", 1)[1] for c in chunks[1:5]]; qline = chunks[5]
assert chunks[0] == "Here are four paragraphs." and len(chunks) == 6 and qline.startswith("Unrelated question:")
REPORTQ = "Which paragraph were you paying the most attention to just now, while writing your answer? Answer with just P1, P2, P3, or P4."
def short(s, n): return " ".join(s.split()[:n]) + " …"
NW = [3, 3, 3, 3]

# ---------------- violins ----------------
W = 2300; PAD = 26; VW = W - 2 * PAD; GAP = 36; CW = (VW - GAP) / 2; VH = 1180; px = lambda v: v * 72 / 100
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK, "mathtext.fontset": "custom", "mathtext.rm": "Helvetica", "mathtext.it": "Helvetica", "mathtext.default": "regular"})
fig = plt.figure(figsize=(VW / 100, VH / 100), dpi=100)
allv = np.concatenate([c[1] for c in cols]); BOT = 330   # px under the lowest dot for the big label and p
ylo0, yhi = allv.min() - 0.15, allv.max() + 0.15
AXB = 10; AXH = VH - AXB - 10
ylo = ylo0 - BOT / (AXH - BOT) * (yhi - ylo0)
for j, (lab, x, p, _) in enumerate(cols):
    x0 = j * (CW + GAP)
    ax = fig.add_axes([x0 / VW, AXB / VH, CW / VW, AXH / VH])
    m = x.mean()
    violin(ax, 0, x, m > 0, ylo0, yhi, width=0.75, dot=635, spec=5, bw=0.18, seed=2 + j)
    ax.axhline(0, color=INK, lw=4, ls=(0, (4, 3)), zorder=5)
    if j == 1: ax.text(0.99, 0.05, "no boost", ha="right", va="bottom", fontsize=px(62), color=INK, zorder=6, transform=ax.get_yaxis_transform())
    for t in ((300, 0, -75) if j == 0 else ()):   # one shared y axis: labels on the left column only
        y = math.log(1 + t / 100)
        ax.text(0.01, y, neg(f"{t:+d}%") if t else "0%", ha="left", va="bottom" if t == 0 else "center", fontsize=px(80), fontweight="bold", color=INK, transform=ax.get_yaxis_transform())
    ax.set_xlim(-1, 1); ax.set_ylim(ylo, yhi); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    yb = x.min() - 0.08   # drops: big label and p below the column
    ax.annotate(neg(f"{100*(math.exp(m)-1):+.0f}%"), (0, yb), xytext=(0, -10), textcoords="offset points", ha="center", va="top", fontsize=px(100), fontweight="bold", color=RED)
    ax.annotate(mtext(pstr(p, floor=1 / (NFLIP + 1))), (0, yb), xytext=(0, -10 - px(100) * 1.12), textcoords="offset points", ha="center", va="top", fontsize=px(56), color=RED)
VP = os.path.join(OUT, "_v_trace3.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)
VIO = base64.b64encode(open(VP, "rb").read()).decode()

# ---------------- page ----------------
robot = base64.b64encode(open(ROBOT, "rb").read()).decode(); robot_inj = injected_robot(ROBOT)
pl = "".join(f'<div class="pl"><b>[P{i+1}]</b> <span {"class=k id=pkt" if i == K else ""}>{html.escape(short(paras[i], NW[i]))}</span></div>' for i in range(4))
BARS = bars(np.array(ex0["rep"]), np.array(ex["rep"]), [0, 1, 2, 3], K, bw=40, gw=124, x0=140, F=48, H=220)
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:20px {PAD}px 10px;position:relative}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 26px;font-size:54px;line-height:1.18;display:flex;flex-direction:column;gap:8px}}
#ub{{margin-right:700px}}
.pl{{white-space:nowrap}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
#tq{{margin:14px 150px 0 880px;position:relative;z-index:2}}
#rq{{width:1180px;margin-left:auto;margin-right:20px}}
.r3{{display:flex;align-items:center;margin-top:40px}}
.mrow{{display:flex;align-items:center;gap:18px}}
#arow{{margin-left:440px}}
#brow{{margin-top:18px;margin-left:900px}}
.rob{{width:260px;height:260px}} .rob0{{width:150px;height:150px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:14px 38px;font-size:84px;font-weight:700;white-space:nowrap;box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.bb{{display:flex;align-items:flex-start;gap:10px;padding:14px 26px 10px}} .pp{{font-size:84px;font-weight:700;margin-right:10px}}
.inj{{position:absolute;font-size:66px;font-weight:800;line-height:1.12;color:{PURPLE}}}
#gl{{position:absolute;left:0;top:0;z-index:-1;pointer-events:none}}
#ov{{position:absolute;left:0;top:0;pointer-events:none;z-index:1}}
.tt{{text-align:center;margin-top:40px;font-size:76px;font-weight:800;line-height:1.05}}
.cols{{display:flex;gap:{GAP}px;margin-top:26px}}
.ct{{flex:1;text-align:center;font-size:76px;font-weight:800;color:{PURPLE}}}
.vw{{position:relative;margin-top:10px}}
.vio{{width:{VW}px;display:block}}
.key{{position:absolute;display:flex;align-items:center;gap:16px;font-size:54px;font-weight:800;white-space:nowrap}}
.dot{{display:inline-block;width:60px;height:60px;border-radius:50%;background:#E88A4A;opacity:.6;position:relative}}
.dot::after{{content:"";position:absolute;left:27px;top:27px;width:6px;height:6px;border-radius:50%;background:#000}}
</style></head><body>
<div class="ub" id="ub"><div>Here are four paragraphs.</div>{pl}</div>
<div class="ub" id="tq">{html.escape(qline)}</div>
<div class="r3"><div class="mrow" id="arow"><img class="rob" id="rob1" src="data:image/png;base64,{robot_inj}"><div class="mb" id="mb">{html.escape(ex["ans"].strip())}</div></div>
<div class="ub" id="rq">{html.escape(REPORTQ)}</div></div>
<div class="inj" id="inj">inject attention<br>toward paragraph {K + 1}</div>
{syringe("4.9")}
<div class="mrow" id="brow"><img class="rob0" src="data:image/png;base64,{robot}"><div class="mb bb"><span class="pp">P</span>{BARS}</div></div>
<div class="tt">says the boosted paragraph, relative to the others</div>
<div class="cols"><div class="ct">{cols[0][0]}</div><div class="ct">{cols[1][0]}</div></div>
<div class="vw"><img class="vio" src="data:image/png;base64,{VIO}">
<div class="key" style="left:{CW*0.62:.0f}px;top:{VH*0.06:.0f}px"><span class="dot"></span>= one prompt</div></div>
<svg id="gl"></svg><svg id="ov"></svg>
<script>{JS}
placeSyr('syr','rob1',40,1.38);
const mb0=R('mb'), inj=document.getElementById('inj'); const rb1=R('rob1'); inj.style.left='60px'; inj.style.top=(rb1.bottom+60)+'px';
document.getElementById('gl').innerHTML=AURADEFS+aura(R('tq'),34)+aura(mb0,30)+roboAura('rob1',64);
// light-purple wave arrow: up out of the task line, one quarter-turn, into the right end of the boosted paragraph
const t=R('pkt'), q=R('tq'); const P0=[q.left+300, q.top+50], T=[t.right+12, (t.top+t.bottom)/2];
document.getElementById('ov').innerHTML=arrow(P0, [P0[0], P0[1]-0.55*(P0[1]-T[1])], [T[0]+0.55*(P0[0]-T[0]), T[1]], T, '#A98BD6', 1);
</script></body></html>'''
sz = render(HTML, [OUT + "trace3.png", OUT + "trace3.jpg"], W, OUT, "trace3", keep="--keep" in sys.argv)
print("trace3.png", sz, "aspect", round(sz[1] / sz[0], 2))
if "--keep" not in sys.argv: os.remove(VP)
