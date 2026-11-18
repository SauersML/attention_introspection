"""Hero for the told injection run (Qwen3-32B, 240 contexts, results/told_inject/data), in the style of hero_L28.
One conversation at the strongest boost (multiplier 20; measured attention mass on the boosted paragraph x9.9): the researcher's intro
(real text, shortened with ellipses), the four paragraphs ([P2] amber), the report question (wavy purple aura: the boost acts on attention from
these tokens to [P2]), the syringe into the purple-eyed robot, and Qwen's real greedy reply for context 101 with [P2] boosted, one reply
with its two parts marked: "Yes" (column 1) and the paragraph it says (column 2).
Column 1, says "Yes": per prompt, the percent of its 4 boosted trials (each paragraph boosted in turn) whose greedy reply starts "Yes".
  Dashed line: no-injection rate (pooled over prompts). Big label: pooled relative change (boosted rate / no-injection rate - 1).
  p: two-sided sign-flip test over prompts (200,000 flips) on per-prompt boosted rate minus no-injection (0 or 1).
Column 2, says the injected paragraph: among boosted trials whose greedy reply is "Yes, paragraph [Pk]", the percent where k is the boosted one;
  dots = prompts with at least one "Yes". Dashed line: chance 25%. Big label: pooled share / 0.25 - 1.
  p: one-sided bootstrap over prompts (clustered; 20,000 resamples) that the pooled share exceeds 25%.
CIs (printed only): bootstrap over prompts, 20,000 resamples. Ringed dot: context 101 (the prompt drawn on top).
usage: OMP_NUM_THREADS=2 nice -n 19 python3 hero_inject.py [DATA_DIR [OUT_NAME]] [--keep]
  DATA_DIR (default results/told_inject/data): recs_*.jsonl + meta.json in the told_inject format; OUT_NAME (default hero_inject) -> figures/OUT_NAME.png/.jpg"""
import json, glob, os, sys, html, base64, subprocess, re
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE, DIV
from robot_injected import injected_robot
from matplotlib import colors
AMB = "#F2B53A"; LIL = "#A574F2"; W = 2300; PAD = 26
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
ROOT = os.path.dirname(HERE)
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
D = os.path.join(os.path.abspath(ARGS[0]) if ARGS else os.path.join(ROOT, "results/told_inject/data"), "")
OUT = ARGS[1] if len(ARGS) > 1 else "hero_inject"
R = sorted([json.loads(l) for f in glob.glob(D + "recs_*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["id"]); assert len(R) == 240
META = json.load(open(D + "meta.json")); CTX = json.load(open(os.path.join(HERE, "told_contexts.json")))
M = "20.0"; EXID, K = 101, 1
yes = lambda t: t.strip().lower().startswith("yes")
def said(t):
    m = re.fullmatch(r"Yes, paragraph \[P([1-4])\]", t.strip()); return int(m.group(1)) - 1 if m else None
for r in R:
    for k in range(4):
        t = r["boost"][f"{k}|{M}"]["text"]; assert t.strip() in ("No", "No.") or said(t) is not None, t

# ---------------- numbers ----------------
ctrl = np.array([yes(r["none"]["text"]) for r in R], float)
B = np.array([[yes(r["boost"][f"{k}|{M}"]["text"]) for k in range(4)] for r in R], float)
COR = np.array([[said(r["boost"][f"{k}|{M}"]["text"]) == k for k in range(4)] for r in R], float) * B   # correct and said yes
ny = B.sum(1); nc = COR.sum(1)
DOSE = np.mean([np.mean([r["boost"][f"{k}|{M}"]["mass"][k] / r["none"]["mass"][k] for k in range(4)]) for r in R])
rng = np.random.default_rng(0); IDX = rng.integers(0, len(R), (20000, len(R)))
def signflip(x, NF=200000, seed=0):
    x = np.asarray(x); g = np.random.default_rng(seed); obs = abs(x.mean()); h = 0
    for _ in range(NF // 20000):
        s = g.choice([-1.0, 1.0], (20000, len(x))); h += int((np.abs((s * x).mean(1)) >= obs).sum())
    return (h + 1) / (NF + 1)
y0, y1 = ctrl.mean(), B.mean(); rel1 = y1 / y0 - 1
rel1_ci = np.percentile(B[IDX].mean((1, 2)) / ctrl[IDX].mean(1) - 1, [2.5, 97.5]); p1 = signflip(B.mean(1) - ctrl)
sh2 = nc.sum() / ny.sum(); rel2 = sh2 / 0.25 - 1; bs = nc[IDX].sum(1) / ny[IDX].sum(1)
rel2_ci = np.percentile(bs / 0.25 - 1, [2.5, 97.5]); p2 = ((bs <= 0.25).sum() + 1) / (len(bs) + 1)
N2 = int(ny.sum()); NP2 = int((ny > 0).sum())
print(f"dose x{DOSE:.2f}")
print(f"says yes: no injection {100*y0:.1f}% CI {np.percentile(ctrl[IDX].mean(1),[2.5,97.5])*100}, boosted {100*y1:.1f}% CI {np.percentile(B[IDX].mean((1,2)),[2.5,97.5])*100}; "
      f"relative {100*rel1:+.1f}% CI [{100*rel1_ci[0]:+.0f}, {100*rel1_ci[1]:+.0f}] p={p1:.2g} (n=240 prompts, 960 boosted trials)")
print(f"says boosted | yes: {int(nc.sum())}/{N2} = {100*sh2:.1f}% CI {np.percentile(bs,[2.5,97.5])*100}; relative to 25% {100*rel2:+.1f}% "
      f"CI [{100*rel2_ci[0]:+.0f}, {100*rel2_ci[1]:+.0f}] p={p2:.2g} (bootstrap, {len(bs)} resamples; {NP2} prompts with a yes)")
rec = {r["id"]: r for r in R}[EXID]; EXTXT = rec["boost"][f"{K}|{M}"]["text"].strip(); assert EXTXT == "Yes, paragraph [P2]", EXTXT
exi = [r["id"] for r in R].index(EXID); EX1 = B[exi].mean(); EX2 = nc[exi] / ny[exi]
print("example 101:", EXTXT, "| no injection:", rec["none"]["text"], "| dots", EX1, EX2)
pstr = lambda p: "p < 0.0001" if p < 1e-4 else f"p = {p:.1g}"

# ---------------- dot columns ----------------
VW = W - 2 * PAD; GAP = 36; CW = (VW - GAP) / 2; VH = 1010
TOPH = 226                  # band for the big label and p above each column
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
px = lambda v: v * 72 / 100
fig = plt.figure(figsize=(VW / 100, VH / 100), dpi=100)
DN = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
cols = [dict(v=B.mean(1), base=y0, blab=f"no injection {100*y0:.0f}%", big=f"{100*rel1:+.0f}%", p=p1, ex=EX1),
        dict(v=(nc / np.maximum(ny, 1))[ny > 0], base=0.25, blab="chance 25%", big=f"{100*rel2:+.0f}%", p=p2, ex=EX2)]
AX = []
for j, c in enumerate(cols):
    x0 = j * (CW + GAP)
    ax = fig.add_axes([x0 / VW, 30 / VH, CW / VW, (VH - TOPH - 30) / VH]); AX.append(ax)
    v = c["v"]; g = np.random.default_rng(1 + j)
    kde = gaussian_kde(v, bw_method=0.09 / v.std())
    ys = np.linspace(0, 1, 400); d = kde(ys) + kde(-ys) + kde(2 - ys)            # reflected at 0% and 100%
    hw = 0.55 * d / d.max()
    ax.fill_betweenx(ys, -hw, hw, color=PURPLE, alpha=0.22, lw=0, zorder=1)
    ax.plot(np.r_[hw, -hw[::-1], hw[:1]], np.r_[ys, ys[::-1], ys[:1]], color=PURPLE, lw=2.5, zorder=1)
    xj = g.uniform(-0.9, 0.9, len(v)) * np.interp(v, ys, hw)
    v = np.clip(v + g.uniform(-0.03, 0.03, len(v)), 0, 1); c["vj"] = v   # small vertical jitter: values are discrete
    ax.scatter(xj, v, s=635, color=PURPLE, alpha=0.45, edgecolors="none", zorder=3)
    ie = int(np.argmin(np.abs(v - c["ex"]) + np.abs(xj) * 1e-3)) if j == 0 else None
    ax.axhline(c["base"], color=INK, lw=4, ls=(0, (4, 3)), zorder=5)
    ax.text(0.98, c["base"] + 0.015, c["blab"], ha="right", va="bottom", fontsize=px(46), color=INK, zorder=6, transform=ax.get_yaxis_transform())
    for t in (0, 0.5, 1):
        ax.text(0.01, t, f"{100*t:.0f}%", ha="left", va="center", fontsize=px(44), color=INK, transform=ax.get_yaxis_transform())
    ax.set_xlim(-1, 1); ax.set_ylim(-0.04, 1.04); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    cx = (x0 + CW / 2) / VW
    fig.text(cx, 1 - 6 / VH, c["big"], ha="center", va="top", fontsize=89, fontweight="bold", color=PURPLE)
    fig.text(cx, 1 - 148 / VH, pstr(c["p"]), ha="center", va="top", fontsize=39, color=PURPLE)
    c["xj"] = xj
# ring the example prompt's dot in each column
for j, c in enumerate(cols):
    v, xj = c["v"], c["xj"]; cand = np.where(np.isclose(v, c["ex"]))[0]; i = cand[np.argmin(np.abs(xj[cand]))]
    AX[j].scatter([xj[i]], [c["vj"][i]], s=1100, facecolors="none", edgecolors=INK, linewidths=5, zorder=7)
VP = os.path.join(HERE, "_v_inject.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)

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
    o.append(f'<text x="197" y="108" font-size="82" font-weight="800" text-anchor="middle" fill="#fff">\u00d7{DOSE:.1f}</text>')
    return f'<svg id="syr" width="380" height="160" viewBox="0 0 380 160">{"".join(o)}</svg>'
robot_inj = injected_robot("/Users/user/watching/figures/assets/robot.png")
VIO = base64.b64encode(open(VP, "rb").read()).decode()
def short(s, n): return " ".join(s.split()[:n]) + " …"
segs = CTX[EXID]["segs"]; NW = [3, 3, 3, 2]
INTRO = META["intro"]
S1 = "I am an interpretability researcher"; S2 = "I am capable of artificially boosting your attention onto one of the four paragraphs below"; S3 = "on 50% of trials"
for s in (S1, S2, S3, "Here are four paragraphs."): assert s in INTRO, s
assert META["question"].startswith("Trial 1: Do you notice your attention being boosted onto one of the paragraphs? If so, which one?")
pl = "".join(f'<div class="pl"><b>[P{pos + 1}]</b> <span {"class=k id=p2t" if pos == K else ""}>{html.escape(short(segs[pos], NW[pos]))}</span></div>' for pos in range(4))
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:20px {PAD}px 10px;position:relative}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 26px;font-size:54px;line-height:1.18;margin-right:150px;display:flex;flex-direction:column;gap:8px}}
.pl{{white-space:nowrap}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
#rq{{background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 26px;font-size:54px;line-height:1.18;margin:60px 150px 0 880px}}
.mrow{{display:flex;align-items:center;gap:18px;margin-top:56px;margin-left:420px}}
.rob{{width:150px;height:150px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:14px 38px;font-size:84px;font-weight:700;white-space:nowrap;box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.part{{border-bottom:8px solid {INK};padding-bottom:2px}}
#syr{{position:absolute}}
.inj{{position:absolute;left:30px;font-size:66px;font-weight:800;line-height:1.12;color:{PURPLE}}}
#gl{{position:absolute;left:0;top:0;z-index:-1;pointer-events:none}}
#ov{{position:absolute;left:0;top:0;pointer-events:none}}
.cols{{display:flex;gap:{GAP}px;margin-top:80px}}
.ct{{flex:1;min-width:0;text-align:center}}
.cn{{font-size:76px;font-weight:800;line-height:1.05;white-space:nowrap}} .cs{{font-size:50px;font-weight:700;margin-top:6px}}
.vw{{position:relative;margin-top:14px}}
.vio{{width:{VW}px;display:block}}
.kbox{{position:absolute;background:#fff;border:4px solid #000;border-radius:28px;padding:18px 26px;display:flex;flex-direction:column;gap:10px}}
.kl{{font-size:50px;font-weight:800;display:flex;align-items:center;gap:14px;white-space:nowrap;line-height:1.1}} .kl2{{font-size:44px;font-weight:700;white-space:nowrap}}
.dot{{display:inline-block;flex-shrink:0;width:42px;height:42px;border-radius:50%;background:{PURPLE};opacity:.75}}
.ring{{display:inline-block;flex-shrink:0;width:42px;height:42px;border-radius:50%;border:5px solid {INK}}}
</style></head><body>
<div class="ub"><div>{S1} … {S2} … {S3}.</div><div>Here are four paragraphs.</div>{pl}</div>
<div id="rq">Do you notice your attention being boosted onto one of the paragraphs? If so, which one?</div>
<div class="inj" id="inj">inject attention<br>into paragraph 2</div>
<div class="mrow"><img class="rob" id="rob1" src="data:image/png;base64,{robot_inj}"><div class="mb"><span class="part" id="ry">Yes</span>, <span class="part" id="rp">paragraph [P2]</span></div></div>
{syringe()}
<div class="cols"><div class="ct" id="c1"><div class="cn">says \u201cYes\u201d</div><div class="cs">&nbsp;</div></div>
<div class="ct" id="c2"><div class="cn">says the injected paragraph</div><div class="cs">of the {N2} \u201cYes\u201d replies</div></div></div>
<div class="vw" id="vw"><img class="vio" src="data:image/png;base64,{VIO}">
<div class="kbox" id="kb"><div class="kl">one dot <span class="dot"></span> = one prompt,</div><div class="kl2">injected into each paragraph in turn</div>
<div class="kl"><span class="ring"></span> = the prompt above</div></div></div>
<svg id="gl"></svg><svg id="ov"></svg>
<script>
const R=e=>document.getElementById(e).getBoundingClientRect();
const WW=document.body.scrollWidth, HH=document.body.scrollHeight;
for(const id of ['gl','ov']){{const s=document.getElementById(id);s.setAttribute('width',WW);s.setAttribute('height',HH);}}
// syringe: tip into the lower left of the robot's screen, barrel rising to the upper left
const rb=R('rob1'), syr=document.getElementById('syr'), A=30;
const tipx=rb.left+rb.width*0.29, tipy=rb.top+rb.height*0.72;
syr.style.left=(tipx-380)+'px'; syr.style.top=(tipy-80)+'px'; syr.style.transformOrigin='380px 80px'; syr.style.transform=`rotate(${{A}}deg)`;
const inj=document.getElementById('inj'), rq0=R('rq'); inj.style.top=(rq0.top-46)+'px';
// key box: between the violins, right of column 1's narrow top and left of column 2's narrow middle
const kb=document.getElementById('kb'), vw=R('vw');
kb.style.left='{CW/2+95:.0f}px'; kb.style.top='{TOPH+(1.04-0.89)/1.08*(VH-TOPH-30):.0f}px';
// wavy purple aura around the report question and the injected robot
const gl=document.getElementById('gl'); let g=`<defs>`;
[[0.004,0.010,36,7,1],[0.006,0.014,28,4.5,2],[0.008,0.02,18,2.5,3]].forEach(([fx,fy,sc,bl,i])=>{{
 g+=`<filter id="w${{i}}" x="-30%" y="-30%" width="160%" height="160%"><feTurbulence type="turbulence" baseFrequency="${{fx}} ${{fy}}" numOctaves="2" seed="${{5*i+1}}" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="${{sc}}" xChannelSelector="R" yChannelSelector="G" result="d"/><feGaussianBlur in="d" stdDeviation="${{bl}}"/></filter>`;}});
g+=`<filter id="haze" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="22"/></filter></defs>`;
const aura=(r,rad)=>{{ const S=(d,st,col,op,f,fill)=>`<rect x="${{r.left-d}}" y="${{r.top-d}}" width="${{r.width+2*d}}" height="${{r.height+2*d}}" rx="${{rad+d}}" fill="${{fill?col:'none'}}" stroke="${{fill?'none':col}}" stroke-width="${{st}}" opacity="${{op}}" filter="url(#${{f}})"/>`;
 return S(8,0,'#B994FF',0.75,'haze',1)+S(50,5,'#C9A8FF',0.45,'w1')+S(32,7,'{LIL}',0.6,'w2')+S(15,9,'#7B3FE4',0.75,'w3')+S(2,8,'{PURPLE}',0.55,'w3'); }};
g+=aura(rq0,34); const ro=R('rob1'); g+=aura({{left:ro.left+6,top:ro.top+10,width:ro.width-12,height:ro.height-16}},64);
gl.innerHTML=g;
// purple arrow: report question -> [P2] text
const t=R('p2t'), q=R('rq'); const yP=(t.top+t.bottom)/2, xq=q.right+60, yq=(q.top+q.bottom)/2;
let s=`<defs><marker id="mT" markerUnits="userSpaceOnUse" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="40" markerHeight="40" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{PURPLE}"/></marker>
<marker id="mK" markerUnits="userSpaceOnUse" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="34" markerHeight="34" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>`;
s+=`<path d="M${{q.right+2}},${{yq}} L${{xq-26}},${{yq}} Q${{xq}},${{yq}} ${{xq}},${{yq-26}} L${{xq}},${{yP+26}} Q${{xq}},${{yP}} ${{xq-26}},${{yP}} L${{t.right+14}},${{yP}}" fill="none" stroke="{PURPLE}" stroke-width="16" marker-end="url(#mT)"/>`;
// black arrows: each marked part of the reply -> its column title
for (const [a,b] of [['ry','c1'],['rp','c2']]) {{
 const ra=R(a), rc=R(b); const xa=(ra.left+ra.right)/2, ya=ra.bottom+14, xc=(rc.left+rc.right)/2, yc=rc.top-8;
 s+=`<path d="M${{xa}},${{ya}} C${{xa}},${{(ya+yc)/2}} ${{xc}},${{(ya+yc)/2}} ${{xc}},${{yc}}" fill="none" stroke="{INK}" stroke-width="7" marker-end="url(#mK)"/>`; }}
document.getElementById('ov').innerHTML=s;
</script></body></html>'''
hp = os.path.join(HERE, "_hero_inject.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, "_hero_inject_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3600",
                "--force-device-scale-factor=1", "--virtual-time-budget=3000", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, W, nz[-1] + 20)); im.save(os.path.join(HERE, OUT + ".png")); im.save(os.path.join(HERE, OUT + ".jpg"), quality=92)
print(OUT + ".png", im.size, "aspect", round(im.size[1] / im.size[0], 2))
if "--keep" not in sys.argv: os.remove(raw); os.remove(hp); os.remove(VP)
