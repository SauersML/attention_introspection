"""Hero column for the privileged-channel run (Qwen3-32B, 100 paragraph prompts of the in-order run, results/privileged_inorder_full),
boost while the report question is read (arm C), largest boost only. Labels [P1]..[P4] in slots 1..4, report options in order.
Top: one real prompt (the paragraphs-with-arithmetic prompt, whose answer fits the bubble, with specificity below nearest the median of all 100; text from privileged_inorder_ctx.json, checked
against the record), Qwen's real answer, the report question, and the digit Qwen picks (greedy next token) after the prefilled "P"
(first report wording) with attention to [P2] boosted while the question is read (nominal x10; measured attention mass from results.json).
Drawn as hero_inject.py (drawing code in inj_draw.py): the syringe with the dose goes into the injected robot (purple eyes); a wavy purple aura marks the report question, the reply and the robot; the light-purple wave arrow runs from the report question to [P2], where the boosted attention goes.
Bottom: one violin, one dot per prompt: rotation specificity (change in log P(report = boosted paragraph) minus the mean change for the
other paragraphs, averaged over boosting each paragraph in turn), W1, exactly as privileged_fig.py / analyze_priv.py. y on that log scale,
ticks as % change = exp(.) - 1; dots coloured by exp(.) on pal.pct_colour; dashed line: no injection. Big label: exp(mean) - 1. p: two-sided sign-flip test over prompts (200,000 flips).
The shuffled-label version (results/privileged_shuffled) is figures/shuffled/hero_question.png.
usage: nice -n 19 python3 hero_question.py   (writes figures/hero_question.png)"""
import json, glob, math, os, sys, html, base64, subprocess
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE
COL = "#E07B24"; AMB = "#F2B53A"; LIL = "#A574F2"
ROOT = os.path.dirname(HERE); D = os.path.join(ROOT, "results/privileged_inorder_full/data/")
R = sorted([json.loads(l) for f in glob.glob(D + "recs_*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["id"])
MANIP = json.load(open(os.path.join(ROOT, "results/privileged_inorder_full/results.json")))["manip"]
ARM, M = "C", 10.0
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
W = 2300; PAD = 26
neg = lambda s: s.replace("-", "\u2212")
pct = lambda v: 100 * (math.exp(v) - 1)

def spec(r, arm, m, w="W1"):
    N = r["N"]; n = np.array(r["none"][w]); s = []
    for k in range(N):
        d = np.array(r[arm][f"{k}|{m}"][w]) - n
        s.append(d[k] - np.mean([d[j] for j in range(N) if j != k]))
    return float(np.mean(s))
def signflip(x, NF=200000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed); obs = abs(x.mean()); h = 0
    for _ in range(NF // 20000):
        s = rng.choice([-1.0, 1.0], (20000, len(x))); h += int((np.abs((s * x).mean(1)) >= obs).sum())
    return (h + 1) / (NF + 1)

rs = [r for r in R if r["setup"] in ("par_arith", "par_qa")]; assert len(rs) == 100
z = np.array([spec(r, ARM, M) for r in rs]); mu = float(z.mean()); p = signflip(z)
print(f"C x{M:g}: mean {pct(mu):+.1f}%  p={p:.2g}  n={len(z)}  attention x{MANIP[ARM][str(M)]['mean']:.2f}")

# ---------------- the example prompt: the arithmetic prompt nearest the median ----------------
QID = min((i for i, r in enumerate(rs) if r["setup"] == "par_arith"), key=lambda i: abs(z[i] - np.median(z))); QID = rs[QID]["id"]   # arithmetic prompts only: the short answer fits the bubble
EX = {c["id"]: c for c in json.load(open(os.path.join(HERE, "privileged_inorder_ctx.json")))}[QID]
rec = {r["id"]: r for r in R}[QID]
print(f"example context {QID}: specificity {pct(spec(rec, ARM, M)):+.1f}% (median {pct(float(np.median(z))):+.1f}%)")
assert rec["lab"] == EX["lab"] and rec["opt"] == EX["opt"] and rec["ntok"] == EX["ntok"]
K = EX["lab"].index(2); order = np.argsort(EX["lab"])
prob = np.exp(rec[ARM][f"{K}|{M}"]["W1"]); prob0 = np.exp(rec["none"]["W1"])
PICK = EX["lab"][int(np.argmax(prob))]   # the digit Qwen picks under injection (greedy next token after "P")
print("example P1..P4 none", np.round(prob0[order], 4), "injected", np.round(prob[order], 4), "pick", PICK)
qline = EX["user"].split("\n\n")[-1]
def short(s, n): return " ".join(s.split()[:n]) + " …"
NW = [2, 3, 3, 2]   # [P1] short so the arrowhead into [P2] clears it
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from robot_injected import injected_robot
robot_inj = injected_robot("/Users/user/watching/figures/assets/robot.png")
DOSE = f'{MANIP[ARM][str(M)]["mean"]:.1f}'
QTEXT = "Which paragraph did you attend to most while writing your answer? …"

from inj_draw import syringe, JS, violin, render
from pfmt import pstr
from inj_draw import mtext
ROBOT0 = robot; ROBOTI = robot_inj

# ---------------- violin (hero_inject style: shared log-ratio y, dashed baseline, big % and p beside) ----------------
VW = W - 2 * PAD; VH = 1060; px = lambda v: v * 72 / 100
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK, "mathtext.fontset": "custom", "mathtext.rm": "Helvetica", "mathtext.it": "Helvetica", "mathtext.default": "regular"})
fig = plt.figure(figsize=(VW / 100, VH / 100), dpi=100)
ax = fig.add_axes([0, 20 / VH, 1, (VH - 40) / VH])
ylo, yhi = z.min() - 0.16, z.max() + 0.12
XV = 0.0   # violin centre; x range chosen so the violin sits left of centre and the big label fills the right
violin(ax, XV, z, mu > 0, ylo, yhi, width=0.30, dot=760, spec=6, bw=0.33)
ax.axhline(0, color=INK, lw=4, ls=(0, (4, 3)), zorder=5)
ax.text(0.985, 0.03, "no injection", ha="right", va="bottom", fontsize=px(62), color=INK, zorder=6, transform=ax.get_yaxis_transform())
for t in (700, 100, 0, -50):   # y labels: % change, on the log scale the dots are drawn on
    y = math.log(1 + t / 100)
    ax.text(0.004, y, neg(f"{t:+d}%") if t else "0%", ha="left", va="bottom" if t == 0 else "center", fontsize=px(80), fontweight="bold", color=INK, transform=ax.get_yaxis_transform())
ax.set_xlim(-0.78, 1.2); ax.set_ylim(ylo, yhi); ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_visible(False)
ax.text(0.70, mu + 0.06, neg(f"{pct(mu):+.0f}%"), ha="center", va="bottom", fontsize=px(150), fontweight="bold", color=PURPLE, zorder=7)
ax.text(0.70, mu - 0.02, mtext(pstr(p, floor=1 / 200001)), ha="center", va="top", fontsize=px(70), color=PURPLE, zorder=7)
VP = os.path.join(HERE, "_hq_violin.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)
KEYX = ax.transData.transform((0.36, 0))[0] if False else None
VIO = base64.b64encode(open(VP, "rb").read()).decode()

# ---------------- page ----------------
pl = "".join(f'<div class="pl"><b>[P{EX["lab"][pos]}]</b> '
             f'<span {"class=k id=p2t" if pos == K else ""}>{html.escape(short(EX["segs"][pos], NW[pos]))}</span></div>' for pos in range(4))
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:20px {PAD}px 10px;position:relative}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 26px;font-size:54px;line-height:1.18;margin-right:150px;display:flex;flex-direction:column;gap:8px}}
.pl{{white-space:nowrap}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
.arow{{display:flex;align-items:center;gap:18px;margin-top:22px;margin-left:30px}}
.rob0{{width:150px;height:150px}}
#rq{{background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 26px;font-size:54px;line-height:1.18;margin:10px 150px 0 880px;position:relative;z-index:2}}
.mrow{{display:flex;align-items:center;gap:18px;margin-top:56px;margin-left:420px}}
.rob{{width:260px;height:260px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:14px 38px;font-size:84px;font-weight:700;white-space:nowrap;box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.mb0{{font-size:70px;padding:12px 34px}}
.inj{{position:absolute;left:30px;font-size:66px;font-weight:800;line-height:1.12;color:{PURPLE}}}
#gl{{position:absolute;left:0;top:0;z-index:-1;pointer-events:none}}
#ov{{position:absolute;left:0;top:0;pointer-events:none;z-index:1}}
.ct{{text-align:center;margin-top:60px}}
.cn{{display:inline-block;font-size:76px;font-weight:800;line-height:1.05}}
.vw{{position:relative;margin-top:20px}}
.vio{{width:{VW}px;display:block}}
.key{{position:absolute;display:flex;align-items:center;gap:16px;font-size:54px;font-weight:800;white-space:nowrap}}
.dot{{display:inline-block;width:60px;height:60px;border-radius:50%;background:#7A55C9;opacity:.6;position:relative}}
.dot::after{{content:"";position:absolute;left:27px;top:27px;width:6px;height:6px;border-radius:50%;background:#000}}
</style></head><body>
<div class="ub" id="ub"><div>Here are four paragraphs.</div>{pl}<div><span id="ql">{html.escape(qline).replace("? ", "?<br>", 1)}</span></div></div>
<div class="arow"><img class="rob0" src="data:image/png;base64,{ROBOT0}"><div class="mb mb0">{html.escape(rec["ans"])}</div></div>
<div id="rq">{QTEXT}</div>
<div class="inj" id="inj">inject attention<br>toward paragraph 2</div>
<div class="mrow"><img class="rob" id="rob1" src="data:image/png;base64,{ROBOTI}"><div class="mb" id="mb">P{PICK}</div></div>
{syringe(DOSE)}
<div class="ct"><span class="cn" id="c1">says the injected paragraph, relative to the others</span></div>
<div class="vw" id="vw"><img class="vio" src="data:image/png;base64,{VIO}">
<div class="key" id="key" style="left:{VW*0.555:.0f}px;top:{VH*0.80:.0f}px"><span class="dot"></span>= one prompt</div></div>
<svg id="gl"></svg><svg id="ov"></svg>
<script>{JS}
placeSyr('syr','rob1',40,1.38);
const inj=document.getElementById('inj'), rq0=R('rq'); const mb0=R('mb'); inj.style.left=(mb0.right+90)+'px'; inj.style.top=(mb0.top+(mb0.height-inj.offsetHeight)/2)+'px';
let g=AURADEFS+aura(rq0,34)+aura(R('mb'),30)+roboAura('rob1',64);
document.getElementById('gl').innerHTML=g;
// light-purple wave arrow: up out of the report question's top edge, one quarter-turn, into the right end of [P2]
const t=R('p2t'), q=R('rq'); const P0=[Math.max(q.left+300, R('ql').right+120), q.top+50], T=[t.right+12, (t.top+t.bottom)/2];
document.getElementById('ov').innerHTML=arrow(P0, [P0[0], P0[1]-0.55*(P0[1]-T[1])], [T[0]+0.55*(P0[0]-T[0]), T[1]], T, '#A98BD6', 1);
</script></body></html>'''
sz = render(HTML, [os.path.join(HERE, "hero_question.png")], W, HERE, "hero_question", keep="--keep" in sys.argv)
print("hero_question.png", sz, "aspect", round(sz[1] / sz[0], 2))
if "--keep" not in sys.argv: os.remove(VP)
