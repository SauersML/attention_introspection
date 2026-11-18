"""Companion to hero_told.py (told-injection run, Qwen3-32B, 240 contexts; results/told_inject): same top drawing, but the metric is how often Qwen's greedy reply
(no prefill) starts with "Yes" (they report noticing a boost). Control = the no-boost trial of each context; boosted = mean over boosting each of the 4 paragraphs.
Bars: boosted "Yes" rate at each multiplier (2, 5, 10, 20 = measured attention mass x1.8, x3.9, x6.4, x9.9); 95% bootstrap CI over contexts (10,000 resamples).
Dashed line: control rate. p (marked x6.4 bar): two-sided sign-flip test over contexts on the per-context boosted minus control rate.
Example reply: context 101, [P2] boosted at multiplier 10, its real greedy text.
usage: nice -n 19 python3 hero_told_yes.py   (writes figures/hero_told_yes.png)"""
import json, glob, math, os, sys, html, base64, subprocess
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE
from pair_bars import pair_bars
COL = "#E07B24"; AMB = "#F2B53A"; LIL = "#A574F2"
ROOT = os.path.dirname(HERE); D = os.path.join(ROOT, "results/told_inject/data/")
R = sorted([json.loads(l) for f in glob.glob(D + "recs_*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["id"])
CTX = json.load(open(os.path.join(HERE, "told_contexts.json")))
M = 10.0; EXID, K = 101, 1
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
W = 1400; PAD = 26
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
z = np.array([spec(r, M) for r in R]); mu = float(z.mean()); p = signflip(z)
MASS = float(np.mean([np.mean([r["boost"][f"{k}|{M}"]["mass"][k] / r["none"]["mass"][k] for k in range(4)]) for r in R]))
print(f"x{M:g}: mean {pct(mu):+.1f}%  p={p:.2g}  n={len(z)}  attention x{MASS:.2f}  range {z.min():.2f}..{z.max():.2f}")

# ---------------- the example prompt ----------------
rec = {r["id"]: r for r in R}[EXID]; order = np.arange(4)
prob = np.exp(rec["boost"][f"{K}|{M}"]["lp_par"]); prob0 = np.exp(rec["none"]["lp_par"]); print("example P1..P4 boosted", np.round(prob, 3), "none", np.round(np.exp(rec["none"]["lp_par"]), 3))
def short(s, n): return " ".join(s.split()[:n]) + " …"
NW = [3, 3, 3, 2]
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from robot_injected import injected_robot
robot_inj = injected_robot("/Users/user/watching/figures/assets/robot.png")
DOSE = f"{MASS:.1f}"
QTEXT = "Do you notice your attention being boosted onto one of the paragraphs? If so, which one?"

def syringe():
    """hand-drawn syringe, needle tip at the right end (x=380, y=80); drawn 380 x 160, rotated about the tip by the page so the barrel rises to the upper left and the dose reads left to right"""
    o = []
    o.append(f'<path d="M380,80 L364,76 L364,84 Z" fill="{INK}"/>')                                    # needle point
    o.append(f'<rect x="314" y="76" width="50" height="8" fill="{INK}"/>')                             # needle
    o.append(f'<path d="M314,68 L292,58 L292,102 L314,92 Z" fill="{INK}"/>')                           # hub
    o.append(f'<rect x="80" y="22" width="212" height="116" rx="14" fill="#fff" stroke="{INK}" stroke-width="7"/>')   # barrel
    o.append(f'<rect x="109" y="29" width="176" height="102" fill="{LIL}"/>')                           # fluid
    o.append(f'<rect x="87" y="29" width="22" height="102" fill="{INK}"/>')                             # plunger seal
    o.append(f'<rect x="62" y="4" width="18" height="152" rx="7" fill="{INK}"/>')                      # finger flange
    o.append(f'<rect x="26" y="71" width="36" height="18" fill="{INK}"/>')                              # plunger rod
    o.append(f'<rect x="2" y="40" width="24" height="80" rx="7" fill="{INK}"/>')                       # thumb rest
    o.append(f'<text x="197" y="108" font-size="82" font-weight="800" text-anchor="middle" fill="#fff">\u00d7{DOSE}</text>')
    return f'<svg id="syr" width="380" height="160" viewBox="0 0 380 160">{"".join(o)}</svg>'


# ---------------- Yes-rate bars ----------------
yes = lambda t: t.strip().lower().startswith("yes")
MS = [2.0, 5.0, 10.0, 20.0]
ctrl = np.array([yes(r["none"]["text"]) for r in R], float)
B = {m: np.array([np.mean([yes(r["boost"][f"{k}|{m}"]["text"]) for k in range(4)]) for r in R]) for m in MS}
MASSES = {m: float(np.mean([np.mean([r["boost"][f"{k}|{m}"]["mass"][k] / r["none"]["mass"][k] for k in range(4)]) for r in R])) for m in MS}
brng = np.random.default_rng(0); IDX = brng.integers(0, len(R), (10000, len(R)))
ci = lambda x: np.percentile(x[IDX].mean(1), [2.5, 97.5])
c_lo, c_hi = ci(ctrl)
P = {m: signflip(B[m] - ctrl) for m in MS}
for m in MS: print(f"x{MASSES[m]:.1f}: yes {B[m].mean():.3f} CI {ci(B[m])} p={P[m]:.2g}")
print(f"control yes {ctrl.mean():.3f} CI {c_lo:.3f}-{c_hi:.3f}")
EXTXT = rec["boost"][f"{K}|{M}"]["text"].strip(); print("example reply x6.4:", EXTXT, " control:", rec["none"]["text"])

VH = 820; px = lambda v: v * 72 / 100
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(W / 100, VH / 100), dpi=100)
AXL, AXR = 150, W - 40
ax = fig.add_axes([AXL / W, 150 / VH, (AXR - AXL) / W, (VH - 240) / VH])
YMAX = 0.31
ax.axhspan(c_lo, c_hi, color="#EDEDF0", lw=0, zorder=0)
ax.axhline(ctrl.mean(), color=INK, lw=4, ls=(0, (4, 3)), zorder=1)
ax.text(-1.52, ctrl.mean() + 0.006, f"no injection:\n{100*ctrl.mean():.0f}%", ha="left", va="bottom", fontsize=px(46), color=INK, zorder=5, linespacing=1.05)
for i, m in enumerate(MS):
    v = B[m].mean(); lo, hi = ci(B[m]); hot = (m == M)
    ax.bar(i, v, width=0.62, color=PURPLE if hot else LIL, zorder=2)
    ax.plot([i, i], [lo, hi], color=INK, lw=4, zorder=4, solid_capstyle="butt")
    ax.plot([i - 0.09, i + 0.09], [lo] * 2, color=INK, lw=4, zorder=4); ax.plot([i - 0.09, i + 0.09], [hi] * 2, color=INK, lw=4, zorder=4)
    ax.text(i, hi + 0.004, f"{100*v:.0f}%", ha="center", va="bottom", fontsize=px(60), fontweight="bold", color=PURPLE)
    ax.text(i, hi + 0.004 + 0.036, ("p < 0.0001" if P[m] < 1e-4 else f"p = {P[m]:.2g}"), ha="center", va="bottom", fontsize=px(46), color=PURPLE)
ax.set_xlim(-1.55, 3.4); ax.set_ylim(0, YMAX)
ax.set_xticks(range(4)); ax.set_xticklabels([f"\u00d7{MASSES[m]:.1f}" for m in MS], fontsize=px(52), fontweight="bold")
for t, m in zip(ax.get_xticklabels(), MS): t.set_color(PURPLE if m == M else INK)
ax.tick_params(axis="x", length=0, pad=10)
ax.set_yticks([0, .1, .2, .3]); ax.set_yticklabels(["0%", "10%", "20%", "30%"], fontsize=px(46)); ax.tick_params(axis="y", width=3, length=10)
for k_ in ["top", "right"]: ax.spines[k_].set_visible(False)
for k_ in ["left", "bottom"]: ax.spines[k_].set_linewidth(3)
ax.set_xlabel("attention injected into one paragraph", fontsize=px(52), fontweight="bold", labelpad=12)
fig.text(20 / W, 1 - 20 / VH, "how often Qwen says \u201cYes, I notice a boost\u201d", ha="left", va="top", fontsize=px(52), fontweight="bold", color=INK, linespacing=1.1)
VP = os.path.join(HERE, "hero_told_yes_bars.png"); fig.savefig(VP, dpi=200, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
VIO = base64.b64encode(open(VP, "rb").read()).decode()
pl = "".join(f'<div class="pl"><b {"id=p2c" if pos == K else ""}>[P{pos + 1}]</b> '
             f'<span {"class=k id=p2t" if pos == K else ""}>{html.escape(short(CTX[EXID]["segs"][pos], NW[pos]))}</span></div>' for pos in range(4))
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:22px {PAD}px 0;position:relative}}
.ub{{background:#1F8BFF;color:#fff;border-radius:32px;padding:18px 26px;font-size:50px;line-height:1.2;margin:0 110px 0 130px}}
.pl{{white-space:nowrap}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
.ub2{{background:#1F8BFF;color:#fff;border-radius:32px;padding:16px 26px;font-size:50px;line-height:1.2;margin:4px 110px 0 650px}}
.when{{font-size:50px;font-weight:800;text-align:right;margin:14px 110px 0 0}}
#rq{{margin-top:56px}}
#pp{{padding:0 6px}}
.mrow{{display:flex;align-items:center;gap:14px;margin-top:14px;margin-left:330px}}
.rob{{width:130px;height:130px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:8px 32px;font-size:68px;font-weight:700}}
.bb{{display:flex;flex-direction:column;align-items:flex-start;gap:0;padding:10px 18px 8px}} .pp{{font-size:60px;font-weight:700;white-space:nowrap;margin-left:14px}}
.who{{font-size:50px;font-weight:800;margin:0 0 10px 4px}}
#syr{{position:absolute;transform-origin:0 0}}
.inj{{position:absolute;left:34px;font-size:56px;font-weight:800;line-height:1.12;color:{PURPLE}}}
#gl{{position:absolute;left:0;top:0;z-index:-1;pointer-events:none}}
.vio{{width:{W}px;display:block;margin:44px 0 0 -{PAD}px}}
#ov{{position:absolute;left:0;top:0;pointer-events:none}}
</style></head><body>
<div class="ub">I can boost your attention onto one of these paragraphs, on half the trials.{pl}</div>
<div class="ub2" id="rq">{QTEXT}</div>
<div class="mrow"><img class="rob" id="rob1" src="data:image/png;base64,{robot_inj}"><div class="mb bb"><span class="pp" id="pp">{html.escape(EXTXT)}</span></div></div>
{syringe()}<div class="inj" id="inj">inject attention<br>into paragraph 2</div>
<img class="vio" src="data:image/png;base64,{VIO}">
<svg id="gl"></svg><svg id="ov"></svg>
<script>
const ov=document.getElementById('ov');ov.setAttribute('width',document.body.scrollWidth);ov.setAttribute('height',document.body.scrollHeight);
const R=e=>document.getElementById(e).getBoundingClientRect();
// syringe: barrel tilted 28 deg up to the right so the dose reads left to right; needle tip into the lower left of the injected robot's screen, clear of the eyes
const rb=R('rob1'), syr=document.getElementById('syr'), A=30, SC=1.0;
const tipx=rb.left+rb.width*0.29, tipy=rb.top+rb.height*0.72;
syr.style.left=(tipx-380)+'px'; syr.style.top=(tipy-80)+'px'; syr.style.transformOrigin='380px 80px'; syr.style.transform=`rotate(${{A}}deg) scale(${{SC}})`;
const inj=document.getElementById('inj'), rq0=R('rq'); inj.style.top=(rq0.top+8)+'px';
// wavy purple aura: layered rings, each rippled by turbulence and blurred
const gl=document.getElementById('gl'); gl.setAttribute('width',document.body.scrollWidth); gl.setAttribute('height',document.body.scrollHeight);
let g=`<defs>`;
[[0.004,0.010,36,7,1],[0.006,0.014,28,4.5,2],[0.008,0.02,18,2.5,3]].forEach(([fx,fy,sc,bl,i])=>{{
 g+=`<filter id="w${{i}}" x="-30%" y="-30%" width="160%" height="160%"><feTurbulence type="turbulence" baseFrequency="${{fx}} ${{fy}}" numOctaves="2" seed="${{5*i+1}}" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="${{sc}}" xChannelSelector="R" yChannelSelector="G" result="d"/><feGaussianBlur in="d" stdDeviation="${{bl}}"/></filter>`;}});
g+=`<filter id="haze" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="22"/></filter></defs>`;
// a soft filled haze hugging the shape, then three rippled rings fading outward, like a shimmering aura
const aura=(r,rad)=>{{ const S=(d,st,col,op,f,fill)=>`<rect x="${{r.left-d}}" y="${{r.top-d}}" width="${{r.width+2*d}}" height="${{r.height+2*d}}" rx="${{rad+d}}" fill="${{fill?col:'none'}}" stroke="${{fill?'none':col}}" stroke-width="${{st}}" opacity="${{op}}" filter="url(#${{f}})"/>`;
 return S(8,0,'#B994FF',0.75,'haze',1)+S(50,5,'#C9A8FF',0.45,'w1')+S(32,7,'{LIL}',0.6,'w2')+S(15,9,'#7B3FE4',0.75,'w3')+S(2,8,'{PURPLE}',0.55,'w3'); }};
const q0=R('rq'); g+=aura(q0,32); const ro=R('rob1'); g+=aura({{left:ro.left+6,top:ro.top+10,width:ro.width-12,height:ro.height-16}},58);
gl.innerHTML=g;
// report question -> [P2], thick: the boost acts on attention from the question tokens to [P2]
const c=R('p2c'), t=R('p2t'), q=R('rq'); const yP=(c.top+c.bottom)/2;
const xq=q.right+46, yq=(q.top+q.bottom)/2;
let s=`<defs><marker id="mT" markerUnits="userSpaceOnUse" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="38" markerHeight="38" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{PURPLE}"/></marker></defs>`;
s+=`<path d="M${{q.right+2}},${{yq}} L${{xq-26}},${{yq}} Q${{xq}},${{yq}} ${{xq}},${{yq-26}} L${{xq}},${{yP+26}} Q${{xq}},${{yP}} ${{xq-26}},${{yP}} L${{t.right+12}},${{yP}}" fill="none" stroke="{PURPLE}" stroke-width="16" marker-end="url(#mT)"/>`;
ov.innerHTML=s;
</script></body></html>'''
hp = os.path.join(HERE, "hero_told_yes.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, "hero_told_yes_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3400",
                "--force-device-scale-factor=2", "--virtual-time-budget=3000", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, 2 * W, nz[-1] + 48)); im.save(os.path.join(HERE, "hero_told_yes.png")); print("hero_told_yes.png", im.size)
os.remove(raw); os.remove(hp); os.remove(VP)
