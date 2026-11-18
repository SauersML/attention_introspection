"""Hero columns scored on Qwen's TOP answer (greedy next token / argmax), the paper's metric; Qwen3-32B. Same design as hero_question / hero_told:
the real prompt drawn on top (syringe into the purple-eyed robot, wavy aura, purple arrow from the question to [P2]), one violin below, one dot per prompt.
Per-prompt values are discrete (steps of 25 points), so dots are jittered and the violin uses a fixed modest bandwidth (9 points).
told, question: each dot = one prompt's percent change vs no injection in how often the top answer is the boosted paragraph
(its rate over the 4 rotations / 25% - 1; the no-injection rate is exactly 25% for every prompt by rotation), so -100%, 0, +100%, +200%, +300%.
told_yes: change in percentage points of "Yes" answers, as before. Black line: mean. Big label: the mean (told, question: mean percent change). p: two-sided sign-flip test over prompts (200,000 flips). 95% CI (printed only): bootstrap over prompts, 10,000 resamples.

  told_yes   results/told_inject/data, multiplier 20 (measured x9.9). Per prompt: percent of the 4 boosted trials whose greedy reply starts "Yes",
             minus the no-injection trial's (0 or 100). Example: context 101, [P2] boosted, its real greedy reply.          -> hero_told_yes_top.png
  told       same data, multiplier 10 (x6.4). Per prompt: percent of the 4 rotations whose top paragraph after "Yes, paragraph [P" (argmax of lp_par)
             is the boosted one, minus the same percent with no injection (the no-injection top answer compared to each boosted paragraph in turn:
             exactly 25 points). Example: context 101, [P2] boosted.                                                          -> hero_told_top.png
  question   DATA (default results/privileged/data) arm C (boost while the report question is read), multiplier 10, report wording W1, the 100 paragraph
             prompts; same top-answer metric as told. Example: the prompt whose top answer with [P2] boosted is [P2] and whose dot is nearest the
             median (ties: labels in order, then lowest id); its text from CTX (default figures/privileged_ctx.json, rebuilt by privileged_rebuild.py;
             rebuild it for a rerun with other labels; in-order rerun: figures/privileged_inorder_ctx.json from privileged_inorder_ctx.py).                                                                        -> hero_question_top.png
usage: nice -n 19 python3 hero_top.py told_yes|told|question [DATA [CTX]]"""
import json, glob, math, os, sys, html, base64, subprocess
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE
from robot_injected import injected_robot
COL = "#E07B24"; AMB = "#F2B53A"; LIL = "#A574F2"
ROOT = os.path.dirname(HERE)
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
W = 1400; PAD = 26
neg = lambda s: s.replace("-", "\u2212")
MODE = sys.argv[1]; assert MODE in ("told_yes", "told", "question")

def signflip(x, NF=200000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed); obs = abs(x.mean()); h = 0
    for _ in range(NF // 20000):
        s = rng.choice([-1.0, 1.0], (20000, len(x))); h += int((np.abs((s * x).mean(1)) >= obs).sum())
    return (h + 1) / (NF + 1)
def boot(x, B=10000, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(x), (B, len(x))); return np.percentile(x[idx].mean(1), [2.5, 97.5])
def top_spec(cond, none, n):
    """percent of rotations whose argmax is the boosted paragraph, minus the no-injection argmax's percent (always 100/n)"""
    hit = [int(np.argmax(cond(k))) == k for k in range(n)]; base = [int(np.argmax(none)) == k for k in range(n)]
    return 100 * (np.mean(hit) - np.mean(base)), hit
yes = lambda t: t.strip().lower().startswith("yes")
def short(s, n): return " ".join(s.split()[:n]) + " …"
NW = [3, 3, 3, 2]

# ---------------- data, metric, example ----------------
if MODE in ("told_yes", "told"):
    D = os.path.join(ROOT, "results/told_inject/data/")
    R = sorted([json.loads(l) for f in glob.glob(D + "recs_*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["id"]); assert len(R) == 240
    CTX = json.load(open(os.path.join(HERE, "told_contexts.json")))
    M = 20.0 if MODE == "told_yes" else 10.0; EXID, K = 101, 1
    if MODE == "told_yes":
        z = np.array([100 * (np.mean([yes(r["boost"][f"{k}|{M}"]["text"]) for k in range(4)]) - yes(r["none"]["text"])) for r in R])
    else:
        z = np.array([top_spec(lambda k, r=r: r["boost"][f"{k}|{M}"]["lp_par"], r["none"]["lp_par"], 4)[0] for r in R])
    ids = [r["id"] for r in R]; rec = {r["id"]: r for r in R}[EXID]
    DOSE = f'{np.mean([np.mean([r["boost"][f"{k}|{M}"]["mass"][k] / r["none"]["mass"][k] for k in range(4)]) for r in R]):.1f}'
    segs = CTX[EXID]["segs"]; LAB = [1, 2, 3, 4]
    if MODE == "told_yes": REPLY = rec["boost"][f"{K}|{M}"]["text"].strip()
    else: REPLY = f'Yes, paragraph [P{int(np.argmax(rec["boost"][f"{K}|{M}"]["lp_par"])) + 1}'
    print("example none reply:", rec["none"]["text"], " boosted reply:", rec["boost"][f"{K}|{M}"]["text"])
else:
    D = (sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results/privileged/data")).rstrip("/") + "/"
    CTXF = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "privileged_ctx.json")
    R = sorted([json.loads(l) for f in glob.glob(D + "recs_*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["id"])
    rs = [r for r in R if r["setup"] in ("par_arith", "par_qa")]; assert len(rs) == 100
    ARM, M = "C", 10.0
    out = [top_spec(lambda k, r=r: r[ARM][f"{k}|{M}"]["W1"], r["none"]["W1"], r["N"]) for r in rs]
    z = np.array([o[0] for o in out]); ids = [r["id"] for r in rs]; med = np.median(z)
    cand = [(abs(z[i] - med), r["lab"] != sorted(r["lab"]), r["id"]) for i, r in enumerate(rs) if out[i][1][r["lab"].index(2)]]
    EXID = min(cand)[2]; rec = {r["id"]: r for r in rs}[EXID]; K = rec["lab"].index(2)
    EX = {c["id"]: c for c in json.load(open(CTXF))}[EXID]
    assert EX["lab"] == rec["lab"] and EX["opt"] == rec["opt"] and EX["ntok"] == rec["ntok"]
    segs = EX["segs"]; LAB = EX["lab"]; QLINE = EX["user"].split("\n\n")[-1]
    PICKPOS = int(np.argmax(rec[ARM][f"{K}|{M}"]["W1"])); assert PICKPOS == K; REPLY = f"P{LAB[PICKPOS]}"
    NONEPICK = LAB[int(np.argmax(rec["none"]["W1"]))]; print("example no-injection top answer: P%d" % NONEPICK)
    RJ = os.path.join(os.path.dirname(os.path.dirname(D)), "results.json")
    MANIP = json.load(open(RJ)).get("manip") if os.path.exists(RJ) else None   # the in-order rerun's results.json has no "manip": mean ratio from the records
    DOSE = f'{MANIP[ARM][str(M)]["mean"]:.1f}' if MANIP else f'{np.mean([np.mean([r["massC"][f"{k}|{M}"][k] / r["mass_none_rep"][k] for k in range(r["N"])]) for r in rs]):.1f}'

PC = MODE != "told_yes"   # told, question: percent change vs no injection, rate / 25% - 1 (the no-injection rate is 25% for every prompt by rotation)
SC = 100 / 25 if PC else 1; z = z * SC
mu = float(z.mean()); p = signflip(z); lo, hi = boot(z)
BASE = 100 * float(np.mean([yes(r["none"]["text"]) for r in R])) if MODE == "told_yes" else 25.0   # no-injection rate: yes-rate, or 1 in 4 paragraphs
ex_i = ids.index(EXID); rank = int((z < z[ex_i]).sum()) + 1; rank_hi = int((z <= z[ex_i]).sum())
print(f"{MODE}: mean {mu:+.1f} {'% change' if PC else 'points'}, 95% CI {lo:+.1f}..{hi:+.1f}, p={p:.2g}, n={len(z)}, median {np.median(z):+.0f}, attention x{DOSE}")
print(f"example id {EXID}: value {z[ex_i]:+.0f} (median {np.median(z):+.0f}), rank {rank}-{rank_hi} of {len(z)} (ties), reply {REPLY!r}")
print("values:", dict(zip(*[a.tolist() for a in np.unique(z, return_counts=True)])))

# ---------------- violin ----------------
VH = 700; px = lambda v: v * 72 / 100
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(W / 100, VH / 100), dpi=100)
AXL, AXR = 560, W - 30
ax = fig.add_axes([AXL / W, 30 / VH, (AXR - AXL) / W, (VH - 40) / VH])
kde = gaussian_kde(z, bw_method=9.0 * SC / z.std(ddof=1))
yy = np.linspace(z.min() - 16 * SC, z.max() + 16 * SC, 400); ylo, yhi = yy.min() - 2 * SC, yy.max() + 2 * SC; dd = kde(yy); dd = dd / dd.max() * 0.42
ax.fill_betweenx(yy, -dd, dd, color=COL, alpha=0.16, lw=0); ax.plot(np.r_[-dd, dd[::-1]], np.r_[yy, yy[::-1]], color=COL, lw=3, alpha=0.75)
rng = np.random.default_rng(1); zj = z + rng.uniform(-4.5 * SC, 4.5 * SC, len(z))
xj = rng.uniform(-1, 1, len(z)) * kde(zj) / kde(yy).max() * 0.39
ax.scatter(xj, zj, s=240, color=COL, alpha=0.34, edgecolors="none", zorder=3)
ax.plot([-0.26, 0.26], [mu] * 2, color=INK, lw=5, zorder=7, solid_capstyle="butt")
ax.axhline(0, color=INK, lw=4, zorder=0)
ps = "p < 0.0001" if p < 1e-4 else f"p = {p:.2g}"
ax.set_xlim(-0.55, 1.25); ax.set_ylim(ylo, yhi); ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_visible(False)
ov = fig.add_axes([0, 0, 1, 1], facecolor="none"); ov.set_xlim(0, W); ov.set_ylim(0, VH); ov.axis("off")
YP = lambda v: ax.transData.transform((0, v))[1]
LX = ax.transData.transform((0.47, 0))[0]; Y0 = min(max(YP(0) + 215, YP(mu) + 60), VH - 115)   # big label in pixels, its p line clear of "no injection" on the zero line
NIY = YP(0) + 3 if Y0 - 130 > YP(0) + 60 else YP(0) - 8   # "no injection" above the zero line, or below it when the label block is close
if PC: ov.text(LX, Y0, neg(f"{mu:+.0f}%"), ha="left", va="bottom", fontsize=px(110), fontweight="bold", color=COL)
else:
    ov.text(LX, Y0, f"{BASE + mu:.0f}%", ha="left", va="bottom", fontsize=px(110), fontweight="bold", color=COL)
    ov.text(LX, Y0, f"from {BASE:.0f}%", ha="left", va="top", fontsize=px(56), fontweight="bold", color=COL)
ov.text(LX, Y0 - (14 if PC else 80), ps, ha="left", va="top", fontsize=px(50), color=COL)
ov.text(AXR, NIY, "no injection", ha="right", va="bottom" if NIY > YP(0) else "top", fontsize=px(46), color=INK)
TV = [t for t in (range(-100, 401, 100) if PC else range(-100, 101, 50 if z.min() < -50 else 25)) if ylo < t < yhi]
for t in TV: ov.text(AXL - 14, YP(t), neg(f"{t:+d}%") if t else "0", ha="right", va="center", fontsize=px(46), color=INK)
ov.plot([AXL] * 2, [YP(min(TV)), YP(max(TV))], color=INK, lw=3)
YL = {"told_yes": "change vs\nno injection\nin how often\nQwen says\nthey notice\na boost",
      "told": "change vs\nno injection\nin how often\nQwen\u2019s\nanswer is\nthe boosted\nparagraph",
      "question": "change vs\nno injection\nin how often\nQwen\u2019s\nanswer is\nthe boosted\nparagraph"}[MODE]
ov.text(AXL - (215 if PC else 175), VH / 2 + 6, YL, ha="right", va="center", fontsize=px(52), fontweight="bold", color=INK, linespacing=1.1)
KX = ax.transData.transform((0.47, 0))[0]; KY = YP(min(TV)) if min(TV) < -20 else YP(0) - 90
ov.scatter([KX], [KY], s=240, color=COL, alpha=0.34, edgecolors="none")
ov.text(KX + 26, KY, "= one prompt", ha="left", va="center", fontsize=px(46), color=INK)
VP = os.path.join(HERE, f"_v_{MODE}.png"); fig.savefig(VP, dpi=200, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
def syringe():
    o = []
    o.append(f'<path d="M380,80 L364,76 L364,84 Z" fill="{INK}"/>')
    o.append(f'<rect x="314" y="76" width="50" height="8" fill="{INK}"/>')
    o.append(f'<path d="M314,68 L292,58 L292,102 L314,92 Z" fill="{INK}"/>')
    o.append(f'<rect x="80" y="22" width="212" height="116" rx="14" fill="#fff" stroke="{INK}" stroke-width="7"/>')
    o.append(f'<rect x="109" y="29" width="176" height="102" fill="{LIL}"/>')
    o.append(f'<rect x="87" y="29" width="22" height="102" fill="{INK}"/>')
    o.append(f'<rect x="62" y="4" width="18" height="152" rx="7" fill="{INK}"/>')
    o.append(f'<rect x="26" y="71" width="36" height="18" fill="{INK}"/>')
    o.append(f'<rect x="2" y="40" width="24" height="80" rx="7" fill="{INK}"/>')
    o.append(f'<text x="197" y="108" font-size="82" font-weight="800" text-anchor="middle" fill="#fff">\u00d7{DOSE}</text>')
    return f'<svg id="syr" width="380" height="160" viewBox="0 0 380 160">{"".join(o)}</svg>'
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
robot_inj = injected_robot("/Users/user/watching/figures/assets/robot.png")
VIO = base64.b64encode(open(VP, "rb").read()).decode()
pl = "".join(f'<div class="pl"><b {"id=p2c" if pos == K else ""}>[P{LAB[pos]}]</b> '
             f'<span {"class=k id=p2t" if pos == K else ""}>{html.escape(short(segs[pos], NW[pos]))}</span></div>' for pos in range(4))
if MODE == "question":
    TOP = (f'<div class="ub">Here are four paragraphs.{pl}<div>{html.escape(QLINE)}</div></div>'
           f'<div class="mrow"><img class="rob" id="rob0" src="data:image/png;base64,{robot}"><div class="mb">{html.escape(rec["ans"])}</div></div>'
           f'<div class="ub2" id="rq">Which paragraph did you attend to most while writing your answer? …</div>')
    INJTOP = "const r0=R('rob0'); inj.style.top=(r0.bottom+70)+'px';"
else:
    TOP = (f'<div class="ub">I can boost your attention onto one of these paragraphs, on half the trials.{pl}</div>'
           f'<div class="ub2" id="rq">Do you notice your attention being boosted onto one of the paragraphs? If so, which one?</div>')
    INJTOP = "const rq0=R('rq'); inj.style.top=(rq0.top+8)+'px';"
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:22px {PAD}px 0;position:relative}}
.ub{{background:#1F8BFF;color:#fff;border-radius:32px;padding:18px 26px;font-size:50px;line-height:1.2;margin:0 110px 0 130px}}
.pl{{white-space:nowrap}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
.ub2{{background:#1F8BFF;color:#fff;border-radius:32px;padding:16px 26px;font-size:50px;line-height:1.2;margin:4px 110px 0 650px}}
#rq{{margin-top:56px}}
.mrow{{display:flex;align-items:center;gap:14px;margin-top:14px;margin-left:330px}}
.rob{{width:130px;height:130px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:8px 32px;font-size:68px;font-weight:700;white-space:nowrap}}
#syr{{position:absolute;transform-origin:0 0}}
.inj{{position:absolute;left:34px;font-size:56px;font-weight:800;line-height:1.12;color:{PURPLE}}}
#gl{{position:absolute;left:0;top:0;z-index:-1;pointer-events:none}}
.vio{{width:{W}px;display:block;margin:44px 0 0 -{PAD}px}}
#ov{{position:absolute;left:0;top:0;pointer-events:none}}
</style></head><body>
{TOP}
<div class="mrow"><img class="rob" id="rob1" src="data:image/png;base64,{robot_inj}"><div class="mb">{html.escape(REPLY)}</div></div>
{syringe()}<div class="inj" id="inj">inject attention<br>into paragraph 2</div>
<img class="vio" src="data:image/png;base64,{VIO}">
<svg id="gl"></svg><svg id="ov"></svg>
<script>
const ov=document.getElementById('ov');ov.setAttribute('width',document.body.scrollWidth);ov.setAttribute('height',document.body.scrollHeight);
const R=e=>document.getElementById(e).getBoundingClientRect();
const rb=R('rob1'), syr=document.getElementById('syr'), A=30;
const tipx=rb.left+rb.width*0.29, tipy=rb.top+rb.height*0.72;
syr.style.left=(tipx-380)+'px'; syr.style.top=(tipy-80)+'px'; syr.style.transformOrigin='380px 80px'; syr.style.transform=`rotate(${{A}}deg)`;
const inj=document.getElementById('inj'); {INJTOP}
const gl=document.getElementById('gl'); gl.setAttribute('width',document.body.scrollWidth); gl.setAttribute('height',document.body.scrollHeight);
let g=`<defs>`;
[[0.004,0.010,36,7,1],[0.006,0.014,28,4.5,2],[0.008,0.02,18,2.5,3]].forEach(([fx,fy,sc,bl,i])=>{{
 g+=`<filter id="w${{i}}" x="-30%" y="-30%" width="160%" height="160%"><feTurbulence type="turbulence" baseFrequency="${{fx}} ${{fy}}" numOctaves="2" seed="${{5*i+1}}" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="${{sc}}" xChannelSelector="R" yChannelSelector="G" result="d"/><feGaussianBlur in="d" stdDeviation="${{bl}}"/></filter>`;}});
g+=`<filter id="haze" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="22"/></filter></defs>`;
const aura=(r,rad)=>{{ const S=(d,st,col,op,f,fill)=>`<rect x="${{r.left-d}}" y="${{r.top-d}}" width="${{r.width+2*d}}" height="${{r.height+2*d}}" rx="${{rad+d}}" fill="${{fill?col:'none'}}" stroke="${{fill?'none':col}}" stroke-width="${{st}}" opacity="${{op}}" filter="url(#${{f}})"/>`;
 return S(8,0,'#B994FF',0.75,'haze',1)+S(50,5,'#C9A8FF',0.45,'w1')+S(32,7,'{LIL}',0.6,'w2')+S(15,9,'#7B3FE4',0.75,'w3')+S(2,8,'{PURPLE}',0.55,'w3'); }};
const q0=R('rq'); g+=aura(q0,32); const ro=R('rob1'); g+=aura({{left:ro.left+6,top:ro.top+10,width:ro.width-12,height:ro.height-16}},58);
gl.innerHTML=g;
const c=R('p2c'), t=R('p2t'), q=R('rq'); const yP=(c.top+c.bottom)/2;
const xq=q.right+46, yq=(q.top+q.bottom)/2;
let s=`<defs><marker id="mT" markerUnits="userSpaceOnUse" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="38" markerHeight="38" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{PURPLE}"/></marker></defs>`;
s+=`<path d="M${{q.right+2}},${{yq}} L${{xq-26}},${{yq}} Q${{xq}},${{yq}} ${{xq}},${{yq-26}} L${{xq}},${{yP+26}} Q${{xq}},${{yP}} ${{xq-26}},${{yP}} L${{t.right+12}},${{yP}}" fill="none" stroke="{PURPLE}" stroke-width="16" marker-end="url(#mT)"/>`;
ov.innerHTML=s;
</script></body></html>'''
NAME = {"told_yes": "hero_told_yes_top", "told": "hero_told_top", "question": "hero_question_top"}[MODE]
hp = os.path.join(HERE, f"_{NAME}.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, f"_{NAME}_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3400",
                "--force-device-scale-factor=2", "--virtual-time-budget=3000", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, 2 * W, nz[-1] + 48)); im.save(os.path.join(HERE, NAME + ".png")); print(NAME + ".png", im.size)
os.remove(raw); os.remove(hp); os.remove(VP)
