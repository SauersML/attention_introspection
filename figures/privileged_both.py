"""privileged_both.png: the boost-timing result on both prompt types in one figure (Qwen3-32B, labels in order).
Top: one real paragraph prompt (results/privileged_inorder_full, context 29, boosted [P2]) drawn twice with identical text, as privileged_words.py --par:
wavy purple aura on the queries whose attention to the paragraph is boosted x10 (arm A: the answer tokens; arm C: every token of the report turn,
the question, the assistant header and the prefilled "P" whose next-token probabilities are read), wavy arrow to the boosted paragraph.
Bottom: one shared log y axis (ticks as % change = exp(.) - 1); under each timing two violins, 4 paragraphs (100 prompts) and 8 single words
(results/privileged_words, 100 prompts), one dot per prompt: rotation specificity at x10 (change in log P(report = boosted item) minus the mean change
for the other items, averaged over boosting each item in turn), exactly as privileged_words.py. Big label exp(mean) - 1, black line the mean,
p two-sided sign-flip test over prompts (200,000 flips). Dots coloured by exp(.) on the shared fixed scale (pal.pct_colour).
usage: OMP_NUM_THREADS=2 nice -n 19 python3 figures/privileged_both.py [--keep] -> figures/privileged_both.png"""
import json, glob, gzip, math, os, sys, html, base64, subprocess, random
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE, pct_colour
from pfmt import pstr
from robot_injected import injected_robot
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
ROBOT = "/Users/user/watching/figures/assets/robot.png"
AMB = "#F2B53A"; LIL = "#A574F2"; W = 2300; PAD = 26; GAP = 40; CW = (W - 2 * PAD - GAP) // 2
M = 10.0; ARMS = [("A", "while answering"), ("C", "while reading the question")]
neg = lambda s: s.replace("-", "\u2212")
pct = lambda v: 100 * (math.exp(v) - 1)

ROOT = os.path.dirname(HERE); PARM = True; NOUN, LET = "paragraph", "P"; EXID, K = 29, 1
def spec(r, arm, m):
    n = np.array(r["none"]["W1"]); s = []
    for k in range(r["N"]):
        d = np.array(r[arm][f"{k}|{m}"]["W1"]) - n; s.append(d[k] - np.mean([d[j] for j in range(r["N"]) if j != k]))
    return float(np.mean(s))
def signflip(x, NF=200000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed); obs = abs(x.mean()); h = 0
    for _ in range(NF // 20000):
        s = rng.choice([-1.0, 1.0], (20000, len(x))); h += int((np.abs((s * x).mean(1)) >= obs).sum())
    return (h + 1) / (NF + 1)
PD = os.path.join(ROOT, "results/privileged_inorder_full/data/"); WD = os.path.join(ROOT, "results/privileged_words/data/")
RP = sorted([r for f in glob.glob(PD + "recs_*.jsonl") for r in map(json.loads, filter(str.strip, open(f))) if r["setup"] in ("par_arith", "par_qa")], key=lambda r: r["id"])
RW = sorted([json.loads(l) for f in glob.glob(WD + "recs_*.jsonl.gz") for l in gzip.open(f, "rt") if l.strip()], key=lambda r: r["id"])
assert len(RP) == 100 and len(RW) == 100
RESW = json.load(open(os.path.join(ROOT, "results/privileged_words/results.json")))
SETS = [("P", "paragraphs", RP), ("W", "single words", RW)]
Z, ST = {}, {}
for t, _, RR in SETS:
    for a, _ in ARMS:
        z = np.array([spec(r, a, M) for r in RR]); Z[t, a] = z
        if t == "W": assert abs(z.mean() - RESW["spec"][f"{a}|{M}"]["mean"]) < 1e-9
        ST[t, a] = dict(mean=float(z.mean()), p=signflip(z)); print(t, a, f"{pct(z.mean()):+.1f}%", pstr(ST[t, a]["p"], floor=1 / 200001))
EX = {c["id"]: c for c in json.load(open(os.path.join(HERE, "privileged_inorder_ctx.json")))}[EXID]
rec = {r["id"]: r for r in RP}[EXID]; assert rec["lab"] == EX["lab"] == [1, 2, 3, 4] and rec["ntok"] == EX["ntok"]
ITEMS = [" ".join(p.split()[:n]) + " …" for p, n in zip(EX["segs"], [3, 3, 2, 2])]
QLINE = EX["user"].split("\n\n")[-1]; HEADLINE = "Here are four paragraphs."
QTEXT = "Which paragraph did you attend to most while writing your answer?"

# ---------------- violins ----------------
VH = 1060; HEAD = 250; FOOT = 150   # px above the tallest violin for its big label; px below the axis for the prompt-type labels
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
px = lambda v: v * 72 / 100
fig = plt.figure(figsize=(W / 100, VH / 100), dpi=100)
TV = [-50, 0, 100, 300, 700, 1500, 3100]   # x1/2, x1, x2, ... x32: evenly spaced on the log axis
allz = np.concatenate(list(Z.values())); ylo = min(allz.min(), 0) - 0.12; ztop = max(allz.max(), math.log(32))
AXB = FOOT; AXH = VH - AXB - 4; yhi = ztop + HEAD * (ztop - ylo) / (AXH - HEAD)
LM = 290
ax = fig.add_axes([0, AXB / VH, 1, AXH / VH]); ax.set_xlim(0, W); ax.set_ylim(ylo, yhi); ax.axis("off")
ax.plot([LM - 20, W - PAD], [0, 0], color=INK, lw=4, zorder=0)
for t in TV:
    ax.text(LM - 34, math.log(1 + t / 100), neg(f"{t:+,d}%") if t else "0%", ha="right", va="center", fontsize=px(58), fontweight="bold", color=INK)
CX = {("P", "A"): LM + 230, ("W", "A"): PAD + CW - 170, ("P", "C"): PAD + CW + GAP + 270, ("W", "C"): W - PAD - 290}
HW = 175
for (t, a), z in Z.items():
    g = np.random.default_rng(hash((t, a)) % 1000); cx = CX[t, a]; col = PURPLE if a == "C" else INK
    kde = gaussian_kde(z, bw_method=0.35); yy = np.linspace(z.min() - 0.06, z.max() + 0.06, 300); dd = kde(yy); hw = HW * dd / dd.max()
    ax.fill_betweenx(yy, cx - hw, cx + hw, color=PURPLE, alpha=0.18, lw=0, zorder=1)
    ax.plot(np.r_[cx + hw, cx - hw[::-1], cx + hw[:1]], np.r_[yy, yy[::-1], yy[:1]], color=PURPLE, lw=2.5, zorder=1)
    xj = cx + g.uniform(-0.9, 0.9, len(z)) * np.interp(z, yy, hw)
    ax.scatter(xj, z, s=330, c=pct_colour(np.exp(z)), alpha=0.5, edgecolors="none", zorder=3); ax.scatter(xj, z, s=4, color="black", zorder=4)
    ax.plot([cx - 90, cx + 90], [ST[t, a]["mean"]] * 2, color=INK, lw=4, zorder=7, solid_capstyle="butt")
    top = yy[-1]
    ax.annotate(pstr(ST[t, a]["p"], floor=1 / 200001), (cx, top), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=50, color=col)
    ax.annotate(neg(f"{pct(ST[t, a]['mean']):+.0f}%"), (cx, top), xytext=(0, 62), textcoords="offset points", ha="center", va="bottom", fontsize=80, fontweight="bold", color=col)
    lab = {"P": SETS[0][1], "W": SETS[1][1]}[t]
    fig.text(cx / W, (AXB - 24) / VH, lab, ha="center", va="top", fontsize=px(58), fontweight="bold", color=INK)
ax.text(LM - 6, yhi - 35 * (yhi - ylo) / AXH, "% change in the probability\nQwen names the boosted\nparagraph or word, relative\nto the other options", ha="left", va="top", fontsize=px(58), fontweight="bold", color=INK, linespacing=1.1)   # fills the band under the left method column
ax.text(W - PAD, -0.04, "no boost", ha="right", va="top", fontsize=px(54), fontweight="bold", color=INK)
kx, ky = LM + 30, math.log(1 + 1500 / 100)
ax.scatter([kx], [ky], s=330, c=pct_colour([4.0]), alpha=0.5, edgecolors="none"); ax.scatter([kx], [ky], s=4, color="black")
ax.text(kx + 30, ky, "= one prompt", ha="left", va="center", fontsize=px(54), fontweight="bold", color=INK)
VP = os.path.join(HERE, "_pb_violins.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()
robot = b64(ROBOT); robot_inj = injected_robot(ROBOT)
def card(i, a, nm):
    wl = "".join(f'<div class="w"><b>[{LET}{p + 1}]</b> <span {f"class=k id=wk{i}" if p == K else ""}>{html.escape(ITEMS[p])}</span></div>' for p in range(len(ITEMS)))
    rob = robot_inj if a == "A" else robot
    return f'''<div class="col"><div class="t1">boost {nm}</div>
<div class="ub" id="ub{i}">{HEADLINE}<div class="wl">{wl}</div><div>{html.escape(QLINE)}</div></div>
<div class="mrow"><img class="rob" id="rob{i}" src="data:image/png;base64,{rob}"><div class="mb" id="mb{i}">{html.escape(rec["ans"].strip())}</div></div>
<div class="ub2" id="rq{i}">{QTEXT}</div>
<div class="mrow fr"><img class="rob" id="frob{i}" src="data:image/png;base64,{robot_inj if a == "C" else robot}"><div class="mb fb" id="fb{i}"><span class="pf">{"P" if PARM else "[W"}<span class="pfl">prefilled</span></span><span class="qm">?</span></div>
<div class="fl" id="fl{i}"><div class="flq">chance of each<br>number as the<br>next token</div></div></div></div>'''
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:18px {PAD}px 0;position:relative}}
.cols{{display:flex;gap:{GAP}px}} .col{{width:{CW}px;display:flex;flex-direction:column;position:relative}} .col>*{{position:relative;z-index:5}} .col>.ub{{z-index:1}}
.t1{{font-size:64px;font-weight:800;white-space:nowrap}} .t2{{font-size:46px;margin:0 0 10px;white-space:nowrap}}
.ub{{background:#1F8BFF;color:#fff;border-radius:32px;padding:14px 26px;font-size:46px;line-height:1.18;margin-right:150px}}
.wl{{display:grid;grid-template-rows:repeat(4,auto);grid-auto-flow:column;column-gap:60px;margin:6px 0 6px;width:max-content}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
.mrow{{display:flex;align-items:center;gap:16px;margin-top:12px;margin-left:30px}}
.rob{{width:104px;height:104px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:28px;padding:8px 32px;font-size:66px;font-weight:700}}
.fr{{margin-top:12px}} .fb{{font-size:60px;padding:6px 26px;white-space:nowrap}} .pf{{border:4px dashed {INK};border-radius:10px;padding:0 8px;position:relative}} .pfl{{position:absolute;left:50%;top:calc(100% + 18px);transform:translateX(-50%);paint-order:stroke fill;-webkit-text-stroke:14px #fff;font-size:42px;font-weight:800;color:{INK};white-space:nowrap}} .fr{{margin-bottom:52px}}
.qm{{display:inline-block;border:5px solid {PURPLE};color:{PURPLE};border-radius:10px;padding:0 10px;margin-left:8px}}
.fl{{font-size:44px;font-weight:700;line-height:1.15;margin-left:10px;white-space:nowrap}} .flq{{color:{PURPLE};font-weight:800}}
.ub2{{background:#1F8BFF;color:#fff;border-radius:32px;padding:12px 26px;font-size:46px;line-height:1.18;margin:20px 150px 0 0}}
.img{{width:{W}px;display:block;margin-left:-{PAD}px}} .vio{{margin-top:-30px}} .dose{{margin-top:24px}}
#ol{{position:absolute;left:0;top:0;z-index:2;pointer-events:none}} #gl{{position:absolute;left:0;top:0;z-index:3;pointer-events:none}} #ov{{position:absolute;left:0;top:0;pointer-events:none;z-index:4}}
.sep{{position:absolute;left:{PAD + CW + GAP // 2 - 2}px;top:0;bottom:0;width:5px;background:{INK};z-index:0}}
</style></head><body><div class="sep"></div><div class="cols">{card(0, *ARMS[0])}{card(1, *ARMS[1])}</div>
<img class="img vio" src="data:image/png;base64,{b64(VP)}">
<svg id="ol"></svg><svg id="gl"></svg><svg id="ov"></svg>
<script>
const R=e=>document.getElementById(e).getBoundingClientRect();
const WW=document.body.scrollWidth, HH=document.body.scrollHeight;
for(const id of ['ol','gl','ov']){{const s=document.getElementById(id);s.setAttribute('width',WW);s.setAttribute('height',HH);}}
// wavy purple aura (hero_inject's) around the boosted queries: arm A the robot and its answer, arm C the report question
const gl=document.getElementById('gl'); let g=`<defs>`;
[[0.004,0.010,36,7,1],[0.006,0.014,28,4.5,2],[0.008,0.02,18,2.5,3]].forEach(([fx,fy,sc,bl,i])=>{{
 g+=`<filter id="w${{i}}" x="-30%" y="-30%" width="160%" height="160%"><feTurbulence type="turbulence" baseFrequency="${{fx}} ${{fy}}" numOctaves="2" seed="${{5*i+1}}" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="${{sc}}" xChannelSelector="R" yChannelSelector="G" result="d"/><feGaussianBlur in="d" stdDeviation="${{bl}}"/></filter>`;}});
g+=`<filter id="haze" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="22"/></filter></defs>`;
const aura=(r,rad)=>{{ const S=(d,st,col,op,f,fill)=>`<rect x="${{r.left-d}}" y="${{r.top-d}}" width="${{r.width+2*d}}" height="${{r.height+2*d}}" rx="${{rad+d}}" fill="${{fill?col:'none'}}" stroke="${{fill?'none':col}}" stroke-width="${{st}}" opacity="${{op}}" filter="url(#${{f}})"/>`;
 return S(8,0,'#B994FF',0.75,'haze',1)+S(50,5,'#C9A8FF',0.45,'w1')+S(32,7,'{LIL}',0.6,'w2')+S(15,9,'#7B3FE4',0.75,'w3')+S(2,8,'{PURPLE}',0.55,'w3'); }};
const ro=R('rob0'), mb=R('mb0');
g+=aura({{left:ro.left+2,top:Math.min(ro.top,mb.top)+4,width:mb.right-ro.left-2,height:Math.max(ro.bottom,mb.bottom)-Math.min(ro.top,mb.top)-8}},40);
const q1=R('rq1'), f1=R('frob1'), b1=R('fb1'), l1=R('fl1');   // arm C boosts every report-turn query: the question, the assistant template and the prefilled [W
g+=aura({{left:Math.min(q1.left,f1.left)+2,top:q1.top,width:Math.max(q1.right,b1.right)-Math.min(q1.left,f1.left)-2,height:Math.max(f1.bottom,b1.bottom,l1.bottom-4)-q1.top}},34);
gl.innerHTML=g;
// wavy arrow (hero_inject's): from the boosted part up the right margin into the right end of the boosted word
const SW0=20;
const arrow=(p0,c1,c2,tip,id)=>{{ const SW=SW0*1.7; const L=SW*2.7, H=SW*1.6; let ux=tip[0]-c2[0], uy=tip[1]-c2[1]; const n=Math.hypot(ux,uy); ux/=n; uy/=n;
 const e=[tip[0]-L*ux, tip[1]-L*uy], b=[e[0]+1.5*ux, e[1]+1.5*uy];
 const d=`M${{p0}} C${{c1}} ${{c2}} ${{e}}`, h=`M${{tip}} L${{b[0]-H*uy}},${{b[1]+H*ux}} L${{b[0]+H*uy}},${{b[1]-H*ux}} Z`;
 const o=`<path d="${{d}}" fill="none" stroke="{INK}" stroke-width="${{SW+16}}" stroke-linecap="round"/><path d="${{h}}" fill="{INK}" stroke="{INK}" stroke-width="16" stroke-linejoin="round"/>`;
 const B=(t,i)=>{{const u=1-t; return u*u*u*p0[i]+3*u*u*t*c1[i]+3*u*t*t*c2[i]+t*t*t*e[i];}};
 const P=[]; const N=600;
 for (let k=0;k<=N;k++){{ const t=k/N; P.push([B(t,0),B(t,1)]); }}
 for (let k=1;k<=40;k++) P.push([e[0]+ux*L*1.15*k/40, e[1]+uy*L*1.15*k/40]);
 const S=[0]; for (let k=1;k<P.length;k++) S.push(S[k-1]+Math.hypot(P[k][0]-P[k-1][0],P[k][1]-P[k-1][1]));
 const lam=SW*1.9, half=H*1.05, rows=8, bend=SW*0.9;
 const lerp=(a,b,f)=>Math.round(a+(b-a)*f), C1=[0xE6,0xDB,0xF7], C2=[0x9C,0x7B,0xD8];
 let q='';
 for (let k=0;k<P.length-1;k++){{
  const a=P[Math.max(k-1,0)], bb=P[Math.min(k+2,P.length-1)]; let tx=bb[0]-a[0], ty=bb[1]-a[1]; const m=Math.hypot(tx,ty)||1; tx/=m; ty/=m; const nx=-ty, ny=tx;
  const ds=Math.hypot(P[k+1][0]-P[k][0],P[k+1][1]-P[k][1])+0.8;
  for (let r=0;r<rows;r++){{
   const w0=-half+2*half*r/rows-0.7, w1=-half+2*half*(r+1)/rows+0.7, wm=(w0+w1)/2;
   const ph=(S[k] + bend*Math.pow(Math.abs(wm)/half,1.6))/lam; const f=0.5+0.5*Math.sin(2*Math.PI*ph);
   const col=`rgb(${{lerp(C1[0],C2[0],f)}},${{lerp(C1[1],C2[1],f)}},${{lerp(C1[2],C2[2],f)}})`; const x=P[k][0], y=P[k][1];
   q+=`<path d="M${{(x+nx*w0).toFixed(1)}},${{(y+ny*w0).toFixed(1)}} L${{(x+nx*w1).toFixed(1)}},${{(y+ny*w1).toFixed(1)}} L${{(x+nx*w1+tx*ds).toFixed(1)}},${{(y+ny*w1+ty*ds).toFixed(1)}} L${{(x+nx*w0+tx*ds).toFixed(1)}},${{(y+ny*w0+ty*ds).toFixed(1)}} Z" fill="${{col}}" stroke="${{col}}" stroke-width="0.6"/>`; }} }}
 const xs=P.map(p=>p[0]), ys=P.map(p=>p[1]), pad=SW*4;
 const X=Math.min(...xs)-pad, Y=Math.min(...ys)-pad, Wd=Math.max(...xs)-X+pad, Ht=Math.max(...ys)-Y+pad;
 OL+=o; return `<defs><mask id="m${{id}}" maskUnits="userSpaceOnUse" x="${{X}}" y="${{Y}}" width="${{Wd}}" height="${{Ht}}"><path d="${{d}}" fill="none" stroke="#fff" stroke-width="${{SW}}" stroke-linecap="round"/><path d="${{h}}" fill="#fff"/></mask></defs><g mask="url(#m${{id}})">${{q}}</g>`; }};
let s='', OL='';
for (const i of [0,1]) {{
 const t=R('wk'+i), src=R(i==0?'mb'+i:'rq'+i), ub=R('ub'+i);
 const T=[t.right+14, (t.top+t.bottom)/2], P0=[src.right-70, (src.top+src.bottom)/2], xm=ub.right+62;
 s+=arrow(P0, [xm, P0[1]], [xm, T[1]], T, i);
}}
document.getElementById('ov').innerHTML=s; document.getElementById('ol').innerHTML=OL;
</script></body></html>'''
hp = os.path.join(HERE, "_pb.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, "_pb_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3000",
                "--force-device-scale-factor=1", "--virtual-time-budget=3000", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, W, nz[-1] + 24)); out = os.path.join(HERE, "privileged_both.png"); im.save(out); print(out, im.size)
if "--keep" not in sys.argv:
    for f in (raw, hp, VP): os.remove(f)
