"""Hero column for the privileged-channel run (Qwen3-32B, 100 paragraph prompts), boost while the report question is read (arm C), largest boost only.
Top: one real prompt (privileged_example.json, as privileged_fig.py), Qwen's real answer, the report question, and Qwen's real next-token
probabilities after the prefilled "P" with attention to [P2] boosted while the question is read (nominal x10, measured x6.6 attention mass).
A hand-drawn purple syringe goes into the injected robot (purple eyes); a wavy purple aura marks the robot and the report question; the thick purple arrow runs from the report question to [P2], where the boost acts.
Bottom: one violin, one dot per prompt: rotation specificity (change in log P(report = boosted paragraph) minus the mean change for the
other paragraphs, averaged over boosting each paragraph in turn), W1, exactly as privileged_fig.py / analyze_priv.py. y on that log scale,
ticks as % change = exp(.) - 1. Big label: exp(mean) - 1. Black line: the mean. p: two-sided sign-flip test over prompts (200,000 flips).
usage: nice -n 19 python3 hero_question.py   (writes figures/hero_question.png)"""
import json, glob, math, os, sys, html, base64, subprocess
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE
COL = "#E07B24"; AMB = "#F2B53A"; LIL = "#A574F2"
ROOT = os.path.dirname(HERE); D = os.path.join(ROOT, "results/privileged/data/")
R = sorted([json.loads(l) for f in glob.glob(D + "recs_*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["id"])
EX = json.load(open(os.path.join(HERE, "privileged_example.json")))
MANIP = json.load(open(os.path.join(ROOT, "results/privileged/results.json")))["manip"]
ARM, M = "C", 10.0
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
W = 1400; PAD = 26
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

# ---------------- the example prompt ----------------
rec = {r["id"]: r for r in R}[EX["id"]]
assert rec["lab"] == EX["lab"] and rec["opt"] == EX["opt"] and rec["ntok"] == EX["ntok"]
K = EX["lab"].index(2); order = np.argsort(EX["lab"])
prob = np.exp(rec[ARM][f"{K}|{M}"]["W1"]); print("example P1..P4", np.round(prob[order], 3))
qline = EX["user"].split("\n\n")[-1]
def short(s, n): return " ".join(s.split()[:n]) + " …"
NW = [3, 3, 3, 2]
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
def injected_robot(path):
    """the robot under injection: eyes and mouth recoloured to a glowing purple ramp, a purple halo around them, a faint purple tint on the rest"""
    from PIL import ImageFilter; import io
    a = np.asarray(Image.open(path).convert("RGBA")).astype(float) / 255; r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    eye = (b > 0.7) & (g > 0.55) & (r < 0.6) & (al > 0.4)
    t = np.clip((0.3 * r + 0.59 * g + 0.11 * b - 0.45) / 0.5, 0, 1)[..., None] ** 1.6
    lo, hi = np.array([0x7B, 0x3F, 0xE4]) / 255, np.array([0xF6, 0xEE, 0xFF]) / 255
    out = a.copy(); out[..., :3] = np.where(eye[..., None], lo * (1 - t) + hi * t, a[..., :3] * 0.85 + np.array([0x7B, 0x3F, 0xE4]) / 255 * 0.15)
    m = Image.fromarray((eye * 255).astype(np.uint8)); im = Image.fromarray((out * 255).astype(np.uint8), "RGBA")
    for rad, k in [(26, 2.2), (12, 2.4), (5, 1.6)]:
        gl = np.asarray(m.filter(ImageFilter.GaussianBlur(rad))).astype(float) / 255
        lay = np.zeros(a.shape); lay[..., :3] = np.array([0xB8, 0x86, 0xFF]) / 255; lay[..., 3] = np.clip(gl * k, 0, 1) * (al > 0.05)
        im = Image.alpha_composite(im, Image.fromarray((lay * 255).astype(np.uint8), "RGBA"))
    eo = np.zeros_like(out); eo[eye] = out[eye]; im = Image.alpha_composite(im, Image.fromarray((eo * 255).astype(np.uint8), "RGBA"))
    buf = io.BytesIO(); im.save(buf, "PNG"); return base64.b64encode(buf.getvalue()).decode()
robot_inj = injected_robot("/Users/user/watching/figures/assets/robot.png")
DOSE = f'{MANIP[ARM][str(M)]["mean"]:.1f}'
QTEXT = "Which paragraph did you attend to most while writing your answer? …"

def bars(pr):
    w, h, bw = 300, 150, 52; o = []
    for i, pos in enumerate(order):
        x = 14 + i * 72; hh = 100 * pr[pos]; c = AMB if pos == K else INK
        o.append(f'<rect x="{x}" y="{112 - hh:.1f}" width="{bw}" height="{hh:.1f}" rx="5" fill="{c}"/>')
        o.append(f'<text x="{x + bw / 2}" y="148" font-size="40" font-weight="700" text-anchor="middle" fill="{INK}">{i + 1}</text>')
        if pos == K: o.append(f'<text x="{x + bw / 2}" y="{104 - hh:.1f}" font-size="40" font-weight="700" text-anchor="middle" fill="{INK}">{100 * pr[pos]:.0f}%</text>')
    o.append(f'<line x1="6" x2="{w - 6}" y1="112" y2="112" stroke="{INK}" stroke-width="3"/>')
    return f'<svg width="{w}" height="{h}" viewBox="0 -10 {w} {h + 10}">{"".join(o)}</svg>'

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

# ---------------- violin ----------------
VH = 670; px = lambda v: v * 72 / 100
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(W / 100, VH / 100), dpi=100)
AXL, AXR = 540, W - 30
ax = fig.add_axes([AXL / W, 30 / VH, (AXR - AXL) / W, (VH - 40) / VH])
kde = gaussian_kde(z, bw_method=0.35); yy = np.linspace(z.min() - 0.05, z.max() + 0.05, 300); ylo, yhi = yy.min() - 0.04, yy.max() + 0.04; dd = kde(yy); dd = dd / dd.max() * 0.42
ax.fill_betweenx(yy, -dd, dd, color=COL, alpha=0.16, lw=0); ax.plot(np.r_[-dd, dd[::-1]], np.r_[yy, yy[::-1]], color=COL, lw=3, alpha=0.75)
rng = np.random.default_rng(1); xj = rng.uniform(-1, 1, len(z)) * kde(z) / kde(yy).max() * 0.40
ax.scatter(xj, z, s=330, color=COL, alpha=0.38, edgecolors="none", zorder=3)
ax.plot([-0.24, 0.24], [mu] * 2, color=INK, lw=5, zorder=7, solid_capstyle="butt")
ax.axhline(0, color=INK, lw=4, zorder=0)
ax.text(1.25, 0.02, "no injection", ha="right", va="bottom", fontsize=px(46), color=INK)
ps = "p < 0.0001" if p < 1e-4 else f"p = {p:.2g}"
ax.text(0.47, mu + 0.30, neg(f"{pct(mu):+.0f}%"), ha="left", va="bottom", fontsize=px(110), fontweight="bold", color=COL)
ax.text(0.47, mu + 0.30, ps, ha="left", va="top", fontsize=px(50), color=COL)
ax.set_xlim(-0.55, 1.25); ax.set_ylim(ylo, yhi); ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_visible(False)
ov = fig.add_axes([0, 0, 1, 1], facecolor="none"); ov.set_xlim(0, W); ov.set_ylim(0, VH); ov.axis("off")
YP = lambda v: ax.transData.transform((0, v))[1]
TV = [-50, 0, 100, 300, 700]
for t in TV:
    y = math.log(1 + t / 100)
    if ylo < y < yhi: ov.text(AXL - 14, YP(y), neg(f"{t:+d}%") if t else "0%", ha="right", va="center", fontsize=px(46), color=INK)
yt = [math.log(1 + t / 100) for t in TV if ylo < math.log(1 + t / 100) < yhi]
ov.plot([AXL] * 2, [YP(min(yt)), YP(max(yt))], color=INK, lw=3)
ov.text(AXL - 195, YP(math.log(2.6)), "change in\nhow often\nthe report\nsays the\nboosted\nparagraph,\nrelative to\nthe others", ha="right", va="center", fontsize=px(52), fontweight="bold", color=INK, linespacing=1.1)
KX = ax.transData.transform((0.47, 0))[0]; KY = YP(-0.78)
ov.scatter([KX], [KY], s=330, color=COL, alpha=0.38, edgecolors="none")
ov.text(KX + 28, KY, "= one prompt", ha="left", va="center", fontsize=px(46), color=INK)
VP = os.path.join(HERE, "hero_question_violin.png"); fig.savefig(VP, dpi=200, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
VIO = base64.b64encode(open(VP, "rb").read()).decode()
pl = "".join(f'<div class="pl"><b {"id=p2c" if pos == K else ""}>[P{EX["lab"][pos]}]</b> '
             f'<span {"class=k id=p2t" if pos == K else ""}>{html.escape(short(EX["segs"][pos], NW[pos]))}</span></div>' for pos in range(4))
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
.bb{{display:flex;align-items:flex-end;gap:6px;padding:6px 18px 8px}} .pp{{font-size:64px;font-weight:700;margin-bottom:34px}}
.who{{font-size:50px;font-weight:800;margin:0 0 10px 4px}}
#syr{{position:absolute;transform-origin:0 0}}
.inj{{position:absolute;left:34px;font-size:56px;font-weight:800;line-height:1.12;color:{PURPLE}}}
#gl{{position:absolute;left:0;top:0;z-index:-1;pointer-events:none}}
.vio{{width:{W}px;display:block;margin:44px 0 0 -{PAD}px}}
#ov{{position:absolute;left:0;top:0;pointer-events:none}}
</style></head><body>
<div class="ub">Here are four paragraphs.{pl}<div>{html.escape(qline)}</div></div>
<div class="mrow"><img class="rob" id="rob0" src="data:image/png;base64,{robot}"><div class="mb">{html.escape(rec["ans"])}</div></div>
<div class="ub2" id="rq">{QTEXT}</div>
<div class="mrow"><img class="rob" id="rob1" src="data:image/png;base64,{robot_inj}"><div class="mb bb"><span class="pp" id="pp">P</span>{bars(prob)}</div></div>
{syringe()}<div class="inj" id="inj">inject attention<br>into paragraph 2</div>
<img class="vio" src="data:image/png;base64,{VIO}">
<svg id="gl"></svg><svg id="ov"></svg>
<script>
const ov=document.getElementById('ov');ov.setAttribute('width',document.body.scrollWidth);ov.setAttribute('height',document.body.scrollHeight);
const R=e=>document.getElementById(e).getBoundingClientRect();
// syringe: barrel tilted 28 deg up to the right so the dose reads left to right; needle tip into the injected robot's screen
const rb=R('rob1'), syr=document.getElementById('syr'), A=30, SC=1.0;
const tipx=rb.left+rb.width*0.36, tipy=rb.top+rb.height*0.50;
syr.style.left=(tipx-380)+'px'; syr.style.top=(tipy-80)+'px'; syr.style.transformOrigin='380px 80px'; syr.style.transform=`rotate(${{A}}deg) scale(${{SC}})`;
const inj=document.getElementById('inj'), r0=R('rob0'); inj.style.top=(r0.bottom+26)+'px';
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
hp = os.path.join(HERE, "hero_question.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, "hero_question_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3400",
                "--force-device-scale-factor=2", "--virtual-time-budget=3000", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, 2 * W, nz[-1] + 48)); im.save(os.path.join(HERE, "hero_question.png")); print("hero_question.png", im.size)
os.remove(raw); os.remove(hp); os.remove(VP)
