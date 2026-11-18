"""Privileged-channel figure (Qwen3-32B), paragraph setups only (100 prompts: 60 arithmetic, 40 question about one paragraph).
Top: one real prompt (context id 1, text rebuilt by privileged_rebuild.py -> privileged_example.json) in three copies with identical text:
no boost; attention to paragraph [P2] x10 while the answer is written (arm A); x10 while the report question is read (arm C).
Arrows: answer -> [P2] and report question -> [P2]; the thick arrow marks where the x10 boost is applied.
Reply bars: the model's real next-token probabilities after the prefilled "P" (W1), for that prompt, in that condition.
Bottom: violins, one dot per prompt: rotation specificity (change in log P(report = boosted paragraph) minus the mean change for
the other paragraphs, averaged over boosting each paragraph in turn), W1, exactly as analyze_priv.py; y drawn on that log scale,
ticks labelled as % change = exp(.) - 1 at x0.5, x1, x2, x4, x8 (evenly spaced). Big labels: exp(mean) - 1. Black box: 95% bootstrap CI
of the mean (10,000 resamples), white line the mean. Dots coloured by arm only. p over the largest boost and wherever p >= 0.0001 (every other p < 0.0001): two-sided sign-flip test over prompts (200,000 flips). Bottom line: the other five setups at x10 (CI excludes 0 or not).
usage: privileged_fig.py   (writes figures/privileged.png)"""
import json, glob, math, os, sys, html, base64, subprocess
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import PURPLE, RED, INK, DIV, norm as dnorm
import matplotlib.patheffects as pe
ACOL = {"A": "#2B5FA8", "C": "#E07B24"}   # one colour per arm (the compared things)
ROOT = os.path.dirname(HERE); D = os.path.join(ROOT, "results/privileged/data/")
R = sorted([json.loads(l) for f in glob.glob(D + "recs_*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["id"])
EX = json.load(open(os.path.join(HERE, "privileged_example.json")))
DOSES = [0.25, 0.5, 2.0, 5.0, 10.0]
# tick = measured ratio of attention mass on the boosted part (boost / no boost, mean over prompts, results.json "manip"):
# the boost adds log(m) to the pre-softmax scores, so a nominal m=10 gives about 6.5x the attention mass, m=5 about 3.9x
MANIP = json.load(open(os.path.join(ROOT, "results/privileged/results.json")))["manip"]
TICKS = {a: ["×" + (f"{MANIP[a][str(m)]['mean']:.2f}" if MANIP[a][str(m)]['mean'] < 1 else f"{MANIP[a][str(m)]['mean']:.1f}") for m in DOSES] for a in "AC"}
PAR = ("par_arith", "par_qa"); OTHER = [("list", "lists"), ("code", "code"), ("document", "document sections"), ("sentences", "sentences"), ("dialogue", "dialogue")]
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
AMB = "#F2B53A"; W = 2410; PAD = 24; GAP = 28; CW = (W - 2 * PAD - 2 * GAP) // 3
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

# ---------------- the example prompt ----------------
rec = {r["id"]: r for r in R}[EX["id"]]
assert rec["lab"] == EX["lab"] and rec["opt"] == EX["opt"] and rec["ntok"] == EX["ntok"]
K = EX["lab"].index(2)   # position of the paragraph labelled [P2]
order = np.argsort(EX["lab"])   # positions in label order 1..4
probs = {"none": np.exp(rec["none"]["W1"]), "A": np.exp(rec["A"][f"{K}|10.0"]["W1"]), "C": np.exp(rec["C"][f"{K}|10.0"]["W1"])}
for k, v in probs.items(): print("example", k, "P1..P4", np.round(v[order], 3))
u = EX["user"]; qline = u.split("\n\n")[-1]
def short(p, n): return " ".join(p.split()[:n]) + " …"
NW = [3, 3, 3, 2]
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
QTEXT = "Which paragraph did you attend to most while writing your answer?"

def bars(p):
    w, h, bw = 300, 150, 52; o = []
    for i, pos in enumerate(order):
        x = 14 + i * 72; hh = 100 * p[pos]; c = AMB if pos == K else INK
        o.append(f'<rect x="{x}" y="{112 - hh:.1f}" width="{bw}" height="{hh:.1f}" rx="5" fill="{c}"/>')
        o.append(f'<text x="{x + bw / 2}" y="148" font-size="40" font-weight="700" text-anchor="middle" fill="{INK}">{i + 1}</text>')
        if pos == K: o.append(f'<text x="{x + bw / 2}" y="{104 - hh:.1f}" font-size="40" font-weight="700" text-anchor="middle" fill="{INK}">{100 * p[pos]:.0f}%</text>')
    o.append(f'<line x1="6" x2="{w - 6}" y1="112" y2="112" stroke="{INK}" stroke-width="3"/>')
    return f'<svg width="{w}" height="{h}" viewBox="0 -10 {w} {h + 10}">{"".join(o)}</svg>'

def column(i, title, sub, key, tcol=INK):
    pl = "".join(f'<div class="pl"><b class="{"k" if pos == K else "lb"}" {"id=p2c" + str(i) if pos == K else ""}>[P{EX["lab"][pos]}]</b> '
                 f'<span {"id=p2t" + str(i) if pos == K else ""}>{html.escape(short(EX["segs"][pos], NW[pos]))}</span></div>' for pos in range(4))
    return f'''<div class="col"><div class="t1" style="color:{tcol}">{title}</div><div class="t2">{sub}</div>
<div class="ub">Here are four paragraphs.{pl}<div class="q">{html.escape(qline)}</div></div>
<div class="mrow"><img class="rob" id="rob{i}" src="data:image/png;base64,{robot}"><div class="mb">{html.escape(rec["ans"])}</div></div>
<div class="ub2" id="rq{i}">{QTEXT} …</div>
<div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb bb"><span class="pp">P</span>{bars(probs[key])}</div></div></div>'''

# ---------------- violins ----------------
VH = 1250; px = lambda v: v * 72 / 100   # px -> pt at dpi 100
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(W / 100, VH / 100), dpi=100)
allz = np.concatenate(list(Z.values())); DN = dnorm(np.percentile(allz, 1), np.percentile(np.abs(allz), 97))
ylo, yhi = allz.min() - 0.08, allz.max() + 0.5
axs = []
for j, (a, _) in enumerate(ARMS):
    x0 = PAD + (j + 1) * (CW + GAP)
    ax = fig.add_axes([x0 / W, 230 / VH, CW / W, (VH - 250) / VH]); axs.append(ax)
    rng = np.random.default_rng(1)
    for i, m in enumerate(DOSES):
        z = Z[a, m]; st = ST[a, m]; up = st["mean"] > 0; col = ACOL[a]
        kde = gaussian_kde(z, bw_method=0.35); yy = np.linspace(z.min() - 0.05, z.max() + 0.05, 200); dd = kde(yy); dd = dd / dd.max() * 0.40
        ax.fill_betweenx(yy, i - dd, i + dd, color=col, alpha=0.16, lw=0); ax.plot(np.r_[i - dd, (i + dd)[::-1]], np.r_[yy, yy[::-1]], color=col, lw=2.5, alpha=0.7)
        xj = i + rng.uniform(-1, 1, len(z)) * kde(z) / kde(yy).max() * 0.38
        ax.scatter(xj, z, s=200, color=col, alpha=0.35, edgecolors="none", zorder=3)
        ax.add_patch(matplotlib.patches.Rectangle((i - 0.17, st["lo"]), 0.34, st["hi"] - st["lo"], color=INK, zorder=6, lw=0))  # 95% CI
        ax.plot([i - 0.17, i + 0.17], [st["mean"]] * 2, color="white", lw=3, zorder=7, solid_capstyle="butt")             # mean
        lab = neg(f"{pct(st['mean']):+.0f}%")
        yl = z.max() + 0.06 if up else z.min() - 0.06
        ax.text(i, yl, lab, ha="center", va="bottom" if up else "top", fontsize=px(58), fontweight="bold", color=col, zorder=7)
        ps = "p<0.0001" if st["p"] < 1e-4 else f"p={st['p']:.2f}"
        if m == 10.0 or st["p"] >= 1e-4: ax.text(i, yl + (0.25 if up else -0.25), ps, ha="center", va="bottom" if up else "top", fontsize=px(40), color=col)
    ax.set_xlim(-0.6, len(DOSES) - 0.4); ax.set_ylim(ylo, yhi)
    ax.set_xticks(range(len(DOSES))); ax.set_xticklabels(TICKS[a], fontsize=px(50)); ax.tick_params(length=0, pad=14, colors=INK)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_yticks([])
# zero line across both panels, ticks on the first
zl = fig.add_axes([axs[0].get_position().x0, axs[0].get_position().y0, axs[1].get_position().x1 - axs[0].get_position().x0, axs[0].get_position().height], facecolor="none")
zl.set_xlim(0, 1); zl.set_ylim(ylo, yhi); zl.axhline(0, color=INK, lw=4, zorder=0); zl.axis("off")
TV = [-50, 0, 100, 300, 700]   # x0.5, x1, x2, x4, x8: evenly spaced on the log axis
AX0 = PAD + CW + GAP; F = lambda x, y: (x / W, y / VH)   # px (from left, from bottom) -> figure coords
YP = lambda v: axs[0].transData.transform((0, v))[1]       # data -> px from bottom
for t in TV:
    y = math.log(1 + t / 100)
    if ylo < y < yhi: fig.text(*F(AX0 - 14, YP(y)), neg(f"{t:+d}%") if t else "0%", ha="right", va="center", fontsize=px(46), color=INK)
fig.add_artist(matplotlib.lines.Line2D([AX0 / W] * 2, [YP(math.log(0.5)) / VH, YP(math.log(8)) / VH], color=INK, lw=3))
LXp = AX0 - 200
fig.text(*F(LXp, YP(math.log(6))), "report says the\nboosted paragraph\nmore often than\nthe others", ha="right", va="center", fontsize=px(52), fontweight="bold", color=INK, linespacing=1.1)
fig.text(*F(AX0 - 14, YP(math.log(12))), "log scale", ha="right", va="bottom", fontsize=px(44), color=INK)
fig.text(*F(LXp, YP(math.log(0.45))), "report says the\nboosted paragraph\nless often than\nthe others", ha="right", va="center", fontsize=px(52), fontweight="bold", color=INK, linespacing=1.1)
TY = axs[0].get_position().y0 * VH - 58
fig.text(*F(AX0 - 14, TY), "attention on\nboosted\nparagraph", ha="right", va="center", fontsize=px(44), color=INK, linespacing=1.1)
KY = (YP(math.log(0.45)) + TY) / 2 - 10
KY = YP(math.log(11)) - 20
for j, (a, nm) in enumerate(ARMS):
    p0 = axs[j].get_position()
    fig.text((p0.x0 + p0.x1) / 2 - (0.0 if j == 0 else 0.045), YP(math.log(11)) / VH, "boost " + nm.replace("reading the", "reading the\n"), ha="center", va="top", linespacing=1.1, fontsize=px(52), fontweight="bold", color=ACOL[a])
ov = fig.add_axes([0, 0, 1, 1], facecolor="none"); ov.set_xlim(0, W); ov.set_ylim(0, VH); ov.axis("off")
ov.scatter([PAD + 34], [KY + 26], s=330, color=INK, alpha=0.35, edgecolors="none")
fig.text(*F(PAD + 64, KY + 26), "= one prompt", ha="left", va="center", fontsize=px(46), color=INK)
VP = os.path.join(HERE, "privileged_violins.png"); fig.savefig(VP, dpi=200, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
VIO = base64.b64encode(open(VP, "rb").read()).decode()
cols = column(0, "no boost", "Qwen3-32B", "none") + column(1, "while answering", "attention to paragraph [P2] ×" + f"{MANIP['A']['10.0']['mean']:.1f}", "A", ACOL["A"]) + column(2, "while reading the question", "attention to paragraph [P2] ×" + f"{MANIP['C']['10.0']['mean']:.1f}", "C", ACOL["C"])
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:18px {PAD}px 0;position:relative}}
.cols{{display:flex;gap:{GAP}px}}
.col{{width:{CW}px;display:flex;flex-direction:column}}
.t1{{font-size:56px;font-weight:800;white-space:nowrap}} .t2{{font-size:44px;margin:2px 0 12px;white-space:nowrap}}
.ub{{background:#1F8BFF;color:#fff;border-radius:30px;padding:16px 22px;font-size:44px;line-height:1.2;margin:0 64px 0 96px}}
.pl{{white-space:nowrap}} .q{{margin-top:4px}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 5px}}
.ub2{{background:#1F8BFF;color:#fff;border-radius:30px;padding:14px 22px;font-size:44px;line-height:1.2;margin:14px 64px 0 96px}}
.mrow{{display:flex;align-items:center;gap:12px;margin-top:12px}}
.rob{{width:84px;height:84px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:28px;padding:8px 28px;font-size:60px;font-weight:700}}
.bb{{display:flex;align-items:flex-end;gap:6px;padding:6px 18px 8px}} .pp{{font-size:60px;font-weight:700;margin-bottom:34px}}
.vio{{width:{W}px;display:block;margin-left:-{PAD}px}}
#ov{{position:absolute;left:0;top:0;pointer-events:none}}
</style></head><body><div class="cols">{cols}</div><img class="vio" src="data:image/png;base64,{VIO}">
<svg id="ov"></svg>
<script>
const ov=document.getElementById('ov');ov.setAttribute('width',document.body.scrollWidth);ov.setAttribute('height',document.body.scrollHeight);
const R=e=>document.getElementById(e).getBoundingClientRect();
let s='<defs>';for(const [n,w] of [['t',4],['T',16]]){{const a=n=='t'?16:34;s+=`<marker id="m${{n}}" markerUnits="userSpaceOnUse" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="${{a}}" markerHeight="${{a}}" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>`}}s+='</defs>';
const BOOST=[null,'a','q'];
for(let i=0;i<3;i++){{
  const rob=R('rob'+i), c=R('p2c'+i), t=R('p2t'+i), q=R('rq'+i);
  const ub=c; const yP=(c.top+c.bottom)/2;
  // answer -> [P2]: from the robot's head, up the left margin, into the label
  let w=BOOST[i]=='a'?16:4, m=BOOST[i]=='a'?'T':'t';
  const xa=rob.left+rob.width/2;
  s+=`<path d="M${{xa}},${{rob.top+2}} L${{xa}},${{yP+24}} Q${{xa}},${{yP}} ${{xa+24}},${{yP}} L${{c.left-10}},${{yP}}" fill="none" stroke="{INK}" stroke-width="${{w}}" marker-end="url(#m${{m}})"/>`;
  // report question -> [P2]: from the question bubble, up the right margin, into the paragraph's end
  w=BOOST[i]=='q'?16:4; m=BOOST[i]=='q'?'T':'t';
  const xq=q.right+34, yq=(q.top+q.bottom)/2;
  s+=`<path d="M${{q.right+2}},${{yq}} L${{xq-24}},${{yq}} Q${{xq}},${{yq}} ${{xq}},${{yq-24}} L${{xq}},${{yP+24}} Q${{xq}},${{yP}} ${{xq-24}},${{yP}} L${{t.right+14}},${{yP}}" fill="none" stroke="{INK}" stroke-width="${{w}}" marker-end="url(#m${{m}})"/>`;
}}
ov.innerHTML=s;
</script></body></html>'''
hp = os.path.join(HERE, "privileged.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, "privileged_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3400",
                "--force-device-scale-factor=1", "--virtual-time-budget=3000", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, W, nz[-1] + 24)); im.save(os.path.join(HERE, "privileged.png")); print("privileged.png", im.size)
os.remove(raw); os.remove(hp); os.remove(VP)
