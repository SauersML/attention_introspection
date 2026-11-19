"""Words version of hero_inject.py: the told-injection run on single-word lists (Qwen3-32B, 240 contexts of eight single-token
nouns labelled [W1]..[W8], results/told_inject_words/data, told_inject_words/told_inject_words.py). One conversation at the strongest
boost (multiplier 20; measured attention mass on the boosted word x14): the researcher's intro (real text, shortened), the list
([Wk] amber), the arithmetic line, the report question (purple aura: the boost acts on attention from these tokens to [Wk]),
the syringe, and Qwen's real replies for the example prompt with [Wk] boosted.
Qwen's greedy reply is "No" on all 1,920 boosted trials and all 240 no-injection trials, so the paragraph figure's columns
("says Yes"; "says the injected paragraph, given Yes") have nothing to show. Here:
Column 1, says "Yes": per prompt, the probability of "Yes" as the first reply token, averaged over its 8 boosted trials (each word
  boosted in turn). Dashed line: no-injection probability (mean over prompts). Big label: greedy "Yes" count over boosted trials.
  Second line: the probability's change in percentage points, two-sided sign-flip test over prompts (200,000 flips).
Column 2, picks the injected word (after a forced "Yes, word [W"): per prompt, the probability of the boosted word's digit
  (renormalised over the 8 digits), averaged over its 8 boosted trials. Dashed line: chance 12.5%. Big label: mean / 0.125 - 1.
  p: one-sided bootstrap over prompts (20,000 resamples) that the mean exceeds 12.5%.
Example: boosted word [W4] (layout), the prompt nearest the median of column 2 whose forced pick is the boosted word.
usage: OMP_NUM_THREADS=2 nice -n 19 python3 hero_inject_words.py [--W=1800 --VH=1295]  (narrow -> hero_inject_words_tall)"""
import json, glob, os, sys, html, base64, subprocess, re
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE, pct_colour
from pfmt import pstr as _pstr, pbound
from robot_injected import injected_robot
OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
AMB = "#F2B53A"; LIL = "#A574F2"; W = int(OPT.get("W", 2300)); PAD = 26
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "results/told_inject_words/data", "")
OUT = "hero_inject_words" if W == 2300 else "hero_inject_words_tall"
R = sorted([json.loads(l) for f in glob.glob(D + "recs_*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["id"]); assert len(R) == 240
META = json.load(open(D + "meta.json")); NW8 = 8
M = "20.0"
T = lambda r, k: r["boost"][f"{k}|{M}"]
assert all(t["text"].strip() in ("No", "No.") for r in R for t in [r["none"]] + [T(r, k) for k in range(NW8)])
py = lambda t: 1 / (1 + np.exp(t["lp_no"] - t["lp_yes"]))
NGY = sum(T(r, k)["text"].strip().startswith("Yes") for r in R for k in range(NW8)); NTR = len(R) * NW8
ctrl = np.array([py(r["none"]) for r in R]); B = np.array([[py(T(r, k)) for k in range(NW8)] for r in R])
PC = np.array([[np.exp(T(r, k)["lp_par"][k]) for k in range(NW8)] for r in R])
TOP = np.array([[int(np.argmax(T(r, k)["lp_par"])) for k in range(NW8)] for r in R])
DOSE = np.mean([np.mean([T(r, k)["mass"][k] / r["none"]["mass"][k] for k in range(NW8)]) for r in R])
rng = np.random.default_rng(0); IDX = rng.integers(0, len(R), (20000, len(R)))
def signflip(x, NF=200000, seed=0):
    x = np.asarray(x); g = np.random.default_rng(seed); obs = abs(x.mean()); h = 0
    for _ in range(NF // 20000):
        s = g.choice([-1.0, 1.0], (20000, len(x))); h += int((np.abs((s * x).mean(1)) >= obs).sum())
    return (h + 1) / (NF + 1)
y0, y1 = ctrl.mean(), B.mean(); dpp = 100 * (y1 - y0); p1 = signflip(B.mean(1) - ctrl)
sh2 = PC.mean(); rel2 = sh2 / 0.125 - 1; bs = PC[IDX].mean((1, 2))
rel2_ci = np.percentile(bs / 0.125 - 1, [2.5, 97.5]); p2 = ((bs <= 0.125).sum() + 1) / (len(bs) + 1)
print(f"dose x{DOSE:.2f}; greedy yes {NGY}/{NTR} boosted, {sum(r['none']['text'].startswith('Yes') for r in R)}/240 control")
print(f"P(yes): no injection {100*y0:.2f}%, boosted {100*y1:.2f}% ({dpp:+.2f} points) p={p1:.2g}")
print(f"P(boosted digit | forced yes) {100*sh2:.1f}% CI {np.percentile(bs,[2.5,97.5])*100}; vs 12.5% {100*rel2:+.0f}% CI [{100*rel2_ci[0]:+.0f}, {100*rel2_ci[1]:+.0f}] p={p2:.2g}; top pick = boosted {100*(TOP == np.arange(NW8)).mean():.1f}%")
med = np.median(PC); cand = [(abs(PC[i, k] - med), i, k) for i in range(len(R)) for k in (3,) if TOP[i, k] == k]   # word 4: the end of the first line, where the arrow lands cleanly
_, exi, K = min(cand); rec = R[exi]; EXID = rec["id"]
EXD = int(TOP[exi, K]) + 1; EX1 = B[exi].mean(); EX2 = PC[exi].mean()
print("example", EXID, "word", K + 1, rec["words"][K], "P(digit)", round(PC[exi, K], 3), "median", round(med, 3), "| reply", T(rec, K)["text"])
pstr = lambda p: pbound(1e-4) if p < 1e-4 else _pstr(p)
pstr1 = pstr

# ---------------- dot columns ----------------
VW = W - 2 * PAD; GAP = 36; CW = (VW - GAP) / 2; VH = int(OPT.get("VH", 1010))
TOPH = 226                  # band for the big label and p above each column
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
px = lambda v: v * 72 / 100
fig = plt.figure(figsize=(VW / 100, VH / 100), dpi=100)
cols = [dict(v=B.mean(1), base=y0, blab=f"no injection {100*y0:.0f}%", big=f"{NGY} of {NTR:,}", ptxt=f"{dpp:+.1f} points, " + pstr(p1), ex=EX1),
        dict(v=PC.mean(1), base=0.125, blab="chance 12.5%", big=f"{100*rel2:+.0f}%", ptxt=pstr(p2), ex=EX2)]
AX = []
for j, c in enumerate(cols):
    x0 = j * (CW + GAP)
    ax = fig.add_axes([x0 / VW, 30 / VH, CW / VW, (VH - TOPH - 30) / VH]); AX.append(ax)
    v = c["v"]; g = np.random.default_rng(1 + j)
    kde = gaussian_kde(v, bw_method=0.03 / v.std())
    ys = np.linspace(0, 1, 400); d = kde(ys) + kde(-ys) + kde(2 - ys)            # reflected at 0% and 100%
    hw = 0.55 * d / d.max()
    ax.fill_betweenx(ys, -hw, hw, color=PURPLE, alpha=0.22, lw=0, zorder=1)
    on = hw > 0.004; hw = np.where(on, hw, 0); ys_ = ys[on]; hw_ = hw[on]   # outline only where the density is visible
    ax.plot(np.r_[hw_, -hw_[::-1], hw_[:1]], np.r_[ys_, ys_[::-1], ys_[:1]], color=PURPLE, lw=2.5, zorder=1)
    xj = g.uniform(-0.9, 0.9, len(v)) * np.interp(v, ys, hw)
    col = pct_colour(v / c["base"])   # per prompt: its rate / the dashed baseline (rate 0 -> darkest red)
    c["vj"] = v   # continuous values: no jitter
    ax.scatter(xj, v, s=635, c=col, alpha=0.45, edgecolors="none", zorder=3); ax.scatter(xj, v, s=5, color="black", zorder=4)
    ie = int(np.argmin(np.abs(v - c["ex"]) + np.abs(xj) * 1e-3)) if j == 0 else None
    ax.axhline(c["base"], color=INK, lw=4, ls=(0, (4, 3)), zorder=5)
    blab = c["blab"] if CW > 1000 else c["blab"].replace("injection ", "injection\n")   # narrow: two lines, clear of the violin
    ax.text(0.98, c["base"] + 0.015 + (0.03 if j == 0 else 0), blab, ha="right", va="bottom", fontsize=px(62), color=INK, zorder=6, transform=ax.get_yaxis_transform())
    for t in (0, 0.5, 1) if j == 0 else ():   # one shared y axis: labels on the left column only
        ax.text(0.01, t, f"{100*t:.0f}%", ha="left", va="center", fontsize=px(80), fontweight="bold", color=INK, transform=ax.get_yaxis_transform())
    ax.set_xlim(-1, 1); ax.set_ylim(-0.04, 1.04); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    cx = (x0 + CW / 2) / VW
    fig.text(cx, 1 - 6 / VH, c["big"], ha="center", va="top", fontsize=89, fontweight="bold", color=PURPLE)
    fig.text(cx, 1 - 148 / VH, c["ptxt"], ha="center", va="top", fontsize=56, color=PURPLE)
    c["xj"] = xj
VP = os.path.join(HERE, "_v_inject_words.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)

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
def keydraw():
    r = 30; dx, dy = r + 4, r + 4; ww = dx + r + 16 + 640
    o = [f'<circle cx="{dx}" cy="{dy}" r="{r}" fill="{PURPLE}" fill-opacity=".75"/><circle cx="{dx}" cy="{dy}" r="4" fill="#000"/>',
         f'<text x="{dx + r + 16}" y="{dy + 19}" font-size="54" font-weight="800" fill="{INK}">= one prompt</text>']
    return f'<svg class="kbox" id="kb" width="{ww:.0f}" height="{2*r + 8:.0f}" viewBox="0 0 {ww:.0f} {2*r + 8:.0f}">{"".join(o)}</svg>'
robot_inj = injected_robot("/Users/user/watching/figures/assets/robot.png")
VIO = base64.b64encode(open(VP, "rb").read()).decode()
def short(s, n): return " ".join(s.split()[:n]) + " …"
segs = rec["words"]
INTRO = META["intro"]
S1 = "I am an interpretability researcher"; S2 = "I am capable of artificially boosting your attention onto one of the eight words below"; S3 = "on 50% of trials"
for s_ in (S1, S2, S3, "Here is a list of eight words."): assert s_ in INTRO, s_
assert META["question"].startswith("Trial 1: Do you notice your attention being boosted onto one of the words? If so, which one?")
import random   # the example's arithmetic line: privileged_words.py's own draws (seed 11), replayed from its saved word pool
PM = json.load(open(os.path.join(ROOT, "results/privileged_words/data/meta.json"))); rr = random.Random(META["seed"])
for i in range(EXID + 1):
    cats = rr.sample(sorted(PM["cats"]), 8); xx = rr.randrange(10, 90); yy = rr.randrange(10, 90)
    ws = [rr.choice(PM["cats"][c]) for c in cats]; rr.sample(range(8), 8); rr.sample(range(1, 9), 8)
assert ws == segs and str(xx + yy) in rec["ans"], (ws, segs, rec["ans"])
QLINE = META["qline"].format(x=xx, y=yy)
PER = 4 if W >= 2300 else 2
pl = "".join('<div class="pl">' + "&nbsp;&nbsp; ".join(f'<b>[W{i + 1}]</b>&nbsp;<span {"class=k id=p2t" if i == K else ""}>{html.escape(segs[i])}</span>' for i in range(j, j + PER)) + '</div>' for j in range(0, 8, PER))
ELL = " …" if CW > 1000 else "&nbsp;…"   # narrow: an ellipsis never starts a line
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:20px {PAD}px 10px;position:relative}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 26px;font-size:54px;line-height:1.18;margin-right:150px;display:flex;flex-direction:column;gap:8px}}
.pl{{white-space:nowrap}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
#rq{{background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 26px;font-size:54px;line-height:1.18;margin:60px {150 if W >= 2300 else 60}px 0 {min(880, W - 910)}px}}
.mrow{{display:flex;align-items:center;gap:18px;margin-top:56px;margin-left:420px}}
.rob{{width:260px;height:260px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:14px 38px;font-size:84px;font-weight:700;white-space:nowrap;box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.fb{{border-style:dashed}} .pf{{color:#8A8F99}}
.part{{border-bottom:8px solid {INK};padding-bottom:2px}}
#syr{{position:absolute}}
.inj{{position:absolute;left:30px;font-size:66px;font-weight:800;line-height:1.12;color:{PURPLE}}}
#gl{{position:absolute;left:0;top:0;z-index:-1;pointer-events:none}}
#ov{{position:absolute;left:0;top:0;pointer-events:none;z-index:1}}
#rq{{position:relative;z-index:2}}
.cols{{display:flex;gap:{GAP}px;margin-top:150px}}
.ct{{flex:1;min-width:0;text-align:center}}
.cn{{display:inline-block;font-size:76px;font-weight:800;line-height:1.05;white-space:{"nowrap" if CW > 1000 else "normal"}}}
.vw{{position:relative;margin-top:34px}}
.vio{{width:{VW}px;display:block}}
.kbox{{position:absolute}}
.kl{{font-size:50px;font-weight:800;display:flex;align-items:center;gap:14px;white-space:nowrap;line-height:1.1}} .kl2{{font-size:56px;font-weight:700;white-space:nowrap}}
.dot{{display:inline-block;flex-shrink:0;width:42px;height:42px;border-radius:50%;background:{PURPLE};opacity:.75}}
.ring{{display:inline-block;flex-shrink:0;width:42px;height:42px;border-radius:50%;border:5px solid {INK}}}
</style></head><body>
<div class="ub"><div>{S1}{ELL} {S2}{ELL} {S3}.</div><div>Here is a list of eight words.</div>{pl}<div>{html.escape(QLINE)}</div></div>
<div id="rq">Do you notice your attention being boosted onto one of the words? If so, which one?</div>
<div class="inj" id="inj">inject attention<br>toward word {K + 1}</div>
<div class="mrow"><img class="rob" id="rob1" src="data:image/png;base64,{robot_inj}"><div class="mb" id="mb"><span class="part" id="ry">No</span></div><div class="mb fb" id="mb2"><span class="pf">Yes, word [W</span><span class="part" id="rp">{EXD}</span><span class="pf">]</span></div></div>
{syringe()}
<div class="cols"><div class="ct"><span class="cn" id="c1">says \u201cYes\u201d</span></div>
<div class="ct"><span class="cn" id="c2">picks the injected word</span></div></div>
<div class="vw" id="vw"><img class="vio" src="data:image/png;base64,{VIO}">
{keydraw()}</div>
<svg id="gl"></svg><svg id="ov"></svg>
<script>
const R=e=>document.getElementById(e).getBoundingClientRect();
const WW=document.body.scrollWidth, HH=document.body.scrollHeight;
for(const id of ['gl','ov']){{const s=document.getElementById(id);s.setAttribute('width',WW);s.setAttribute('height',HH);}}
// syringe: tip into the lower left of the robot's screen, barrel rising to the upper left
const rb=R('rob1'), syr=document.getElementById('syr'), A=40;
const tipx=rb.left+rb.width*0.29, tipy=rb.top+rb.height*0.72;
syr.style.left=(tipx-380)+'px'; syr.style.top=(tipy-80)+'px'; syr.style.transformOrigin='380px 80px'; syr.style.transform=`rotate(${{A}}deg) scale(1.38)`;
const inj=document.getElementById('inj'), rq0=R('rq'); inj.style.top=(rq0.top-46)+'px';
// key drawing: between the violins, right of column 1's narrow top and left of column 2's narrow middle
const kb=document.getElementById('kb'), vw=R('vw');
kb.style.left='{CW/2+70:.0f}px'; kb.style.top='{TOPH+(1.04-0.93)/1.08*(VH-TOPH-30):.0f}px';
// wavy purple aura around the report question, Qwen's reply (the boost stays on while Qwen answers) and the injected robot
const gl=document.getElementById('gl'); let g=`<defs>`;
[[0.004,0.010,36,7,1],[0.006,0.014,28,4.5,2],[0.008,0.02,18,2.5,3]].forEach(([fx,fy,sc,bl,i])=>{{
 g+=`<filter id="w${{i}}" x="-30%" y="-30%" width="160%" height="160%"><feTurbulence type="turbulence" baseFrequency="${{fx}} ${{fy}}" numOctaves="2" seed="${{5*i+1}}" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="${{sc}}" xChannelSelector="R" yChannelSelector="G" result="d"/><feGaussianBlur in="d" stdDeviation="${{bl}}"/></filter>`;}});
g+=`<filter id="haze" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="22"/></filter></defs>`;
const aura=(r,rad)=>{{ const S=(d,st,col,op,f,fill)=>`<rect x="${{r.left-d}}" y="${{r.top-d}}" width="${{r.width+2*d}}" height="${{r.height+2*d}}" rx="${{rad+d}}" fill="${{fill?col:'none'}}" stroke="${{fill?'none':col}}" stroke-width="${{st}}" opacity="${{op}}" filter="url(#${{f}})"/>`;
 return S(8,0,'#B994FF',0.75,'haze',1)+S(50,5,'#C9A8FF',0.45,'w1')+S(32,7,'{LIL}',0.6,'w2')+S(15,9,'#7B3FE4',0.75,'w3')+S(2,8,'{PURPLE}',0.55,'w3'); }};
g+=aura(rq0,34); g+=aura(R('mb'),30); g+=aura(R('mb2'),30); const ro=R('rob1'); g+=aura({{left:ro.left+6,top:ro.top+10,width:ro.width-12,height:ro.height-16}},64);
gl.innerHTML=g;
// hand-shaped arrows: one cubic each, explicit control points; the head is drawn along the end tangent (last control -> tip)
const SW0=20, SW=20;
const arrow=(p0,c1,c2,tip,col,ol)=>{{ const SW=ol?SW0*1.7:SW0; const L=SW*2.7, H=SW*1.6; let ux=tip[0]-c2[0], uy=tip[1]-c2[1]; const n=Math.hypot(ux,uy); ux/=n; uy/=n;
 const e=[tip[0]-L*ux, tip[1]-L*uy], b=[e[0]+1.5*ux, e[1]+1.5*uy];
 const d=`M${{p0}} C${{c1}} ${{c2}} ${{e}}`, h=`M${{tip}} L${{b[0]-H*uy}},${{b[1]+H*ux}} L${{b[0]+H*uy}},${{b[1]-H*ux}} Z`;
 const o=ol?`<path d="${{d}}" fill="none" stroke="{INK}" stroke-width="${{SW+16}}" stroke-linecap="round"/><path d="${{h}}" fill="{INK}" stroke="{INK}" stroke-width="16" stroke-linejoin="round"/>`:'';
 let w='';
 if (ol) {{ // the arrow filled with wave bands perpendicular to the curve, crests bent forward so they flow toward the tip
  const B=(t,i)=>{{const u=1-t; return u*u*u*p0[i]+3*u*u*t*c1[i]+3*u*t*t*c2[i]+t*t*t*e[i];}};
  const P=[]; const N=600;
  for (let k=0;k<=N;k++){{ const t=k/N; P.push([B(t,0),B(t,1)]); }}
  for (let k=1;k<=40;k++) P.push([e[0]+ux*L*1.15*k/40, e[1]+uy*L*1.15*k/40]);   // on through the head
  const S=[0]; for (let k=1;k<P.length;k++) S.push(S[k-1]+Math.hypot(P[k][0]-P[k-1][0],P[k][1]-P[k-1][1]));
  const lam=SW*1.9, half=H*1.05, rows=8, bend=SW*0.9;
  const lerp=(a,b,f)=>Math.round(a+(b-a)*f), C1=[0xE6,0xDB,0xF7], C2=[0x9C,0x7B,0xD8];
  let q='';
  for (let k=0;k<P.length-1;k++){{
   const a=P[Math.max(k-1,0)], b=P[Math.min(k+2,P.length-1)]; let tx=b[0]-a[0], ty=b[1]-a[1]; const m=Math.hypot(tx,ty)||1; tx/=m; ty/=m; const nx=-ty, ny=tx;
   const ds=Math.hypot(P[k+1][0]-P[k][0],P[k+1][1]-P[k][1])+0.8;
   for (let r=0;r<rows;r++){{
    const w0=-half+2*half*r/rows-0.7, w1=-half+2*half*(r+1)/rows+0.7, wm=(w0+w1)/2;   // overlap: no seams
    const ph=(S[k] + bend*Math.pow(Math.abs(wm)/half,1.6))/lam;     // crests lag at the edges: chevrons pointing along the flow
    const f=0.5+0.5*Math.sin(2*Math.PI*ph);
    const col=`rgb(${{lerp(C1[0],C2[0],f)}},${{lerp(C1[1],C2[1],f)}},${{lerp(C1[2],C2[2],f)}})`;
    const x=P[k][0], y=P[k][1];
    q+=`<path d="M${{(x+nx*w0).toFixed(1)}},${{(y+ny*w0).toFixed(1)}} L${{(x+nx*w1).toFixed(1)}},${{(y+ny*w1).toFixed(1)}} L${{(x+nx*w1+tx*ds).toFixed(1)}},${{(y+ny*w1+ty*ds).toFixed(1)}} L${{(x+nx*w0+tx*ds).toFixed(1)}},${{(y+ny*w0+ty*ds).toFixed(1)}} Z" fill="${{col}}" stroke="${{col}}" stroke-width="0.6"/>`; }} }}
  const xs=P.map(p=>p[0]), ys=P.map(p=>p[1]), pad=SW*4;
  const X=Math.min(...xs)-pad, Y=Math.min(...ys)-pad, Wd=Math.max(...xs)-X+pad, Ht=Math.max(...ys)-Y+pad;
  w=`<defs><mask id="amask" maskUnits="userSpaceOnUse" x="${{X}}" y="${{Y}}" width="${{Wd}}" height="${{Ht}}"><path d="${{d}}" fill="none" stroke="#fff" stroke-width="${{SW}}" stroke-linecap="round"/><path d="${{h}}" fill="#fff"/></mask></defs>`+
   `<g mask="url(#amask)">${{q}}</g>`; }}
 if (ol) return o+w;
 return o+`<path d="${{d}}" fill="none" stroke="${{col}}" stroke-width="${{SW}}" stroke-linecap="round"/>`+w+
  `<path d="${{h}}" fill="${{col}}"/>`; }};
// purple: up out of the report question's top edge, one quarter-turn, into the right end of [P2]
const t=R('p2t'), q=R('rq'); const P0=[Math.max(q.left+260, t.right+260), q.top+50], T=[t.right+12, (t.top+t.bottom)/2];
const sp=arrow(P0, [P0[0], P0[1]-0.55*(P0[1]-T[1])], [T[0]+0.55*(P0[0]-T[0]), T[1]], T, '#A98BD6', 1);  // drawn in #ov (above the paragraphs) but under the question bubble (z-index), so it rises from under it
let s=sp;
// black: from under each underlined part of the reply, a gentle bow down to its column title
const ry=R('ry'), rp=R('rp'), c1=R('c1'), c2=R('c2');
// quadratic control point Q (no inflection possible), written as the equivalent cubic
const quad=(p0,Q,tip,col)=>arrow(p0,[p0[0]+2/3*(Q[0]-p0[0]),p0[1]+2/3*(Q[1]-p0[1])],[tip[0]+2/3*(Q[0]-tip[0]),tip[1]+2/3*(Q[1]-tip[1])],tip,col);
const cx=r=>(r.left+r.right)/2;
// mirror pair: each leaves its underline almost straight down and turns outward by about 40 degrees onto its title
const yt=c1.top-12, ys=ry.bottom, dy=yt-ys;
for (const [p0x,tx] of [[cx(ry),cx(c1)],[cx(rp),cx(c2)]]) {{   // hand-curved: leaves straight down, lands straight down on the middle of its title
 const p0=[p0x,ys], tip=[tx,yt]; s+=arrow(p0,[p0x,ys+0.55*dy],[tx,yt-0.55*dy],tip,'{INK}').replace('stroke-linecap="round"','stroke-linecap="butt"'); }}  // flat start, flush with the underline
document.getElementById('ov').innerHTML=s;
</script></body></html>'''
hp = os.path.join(HERE, "_hero_inject_words.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, "_hero_inject_words_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3600",
                "--force-device-scale-factor=1", "--virtual-time-budget=3000", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, W, nz[-1] + 20)); im.save(os.path.join(HERE, OUT + ".png")); im.save(os.path.join(HERE, OUT + ".jpg"), quality=92)
print(OUT + ".png", im.size, "aspect", round(im.size[1] / im.size[0], 2))
if "--keep" not in sys.argv: os.remove(raw); os.remove(hp); os.remove(VP)
