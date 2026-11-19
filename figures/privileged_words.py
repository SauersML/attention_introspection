"""Privileged-channel figure on single-word lists (Qwen3-32B, 100 prompts, labels [W1]..[W8] in order; results/privileged_words, the in-order run).
Top: one real prompt (context 0: its arm-C x10 effect is the second closest to the median of the 100; words, answer and report question
from the records; X + Y rebuilt from the run's seeded draw (privileged_words.py, seed 11) and checked against the recorded words, categories and answer),
drawn twice with identical text. [W6] rose (amber) is the boosted word, picked from the right column so the arrow has a clear path.
Wavy purple aura = the queries whose attention to the word is boosted x10: the answer tokens (arm A, "while answering"; the injected robot)
or the report question's tokens (arm C, "while reading the question"). Wavy arrow: from the boosted part to the word.
Middle: violins at x10, one dot per prompt: rotation specificity (change in log P(report = boosted word) minus the mean change for the other
seven words, averaged over boosting each word in turn), exactly as privileged/analyze_words.py; y on that log scale, ticks as % change = exp(.) - 1
(x1, x2, x4, ... evenly spaced). Big label: exp(mean) - 1 (asserted equal to results.json "spec"). Black line: the mean.
p: two-sided sign-flip test over prompts (200,000 flips; pfmt). Dots coloured by exp(.) on the shared fixed scale (pal.pct_colour).
Column subtitle: measured attention mass on the boosted word, boost / no boost at x10 (results.json "manip").
Bottom: dose line, x1/4 .. x10 (nominal multiplier on the attention weights; x1 = no boost = 0 by construction), mean and 95% bootstrap CI over
prompts, read from results.json "spec".
usage: OMP_NUM_THREADS=2 nice -n 19 python3 figures/privileged_words.py [--keep]  -> figures/privileged_words.png"""
import json, glob, gzip, math, os, sys, html, base64, subprocess, random
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE, pct_colour
from pfmt import pstr
from robot_injected import injected_robot
ROOT = os.path.dirname(HERE); DD = os.path.join(ROOT, "results/privileged_words"); D = os.path.join(DD, "data/")
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
ROBOT = "/Users/user/watching/figures/assets/robot.png"
AMB = "#F2B53A"; LIL = "#A574F2"; W = 2300; PAD = 26; GAP = 40; CW = (W - 2 * PAD - GAP) // 2
R = sorted([json.loads(l) for f in glob.glob(D + "recs_*.jsonl.gz") for l in gzip.open(f, "rt") if l.strip()], key=lambda r: r["id"]); assert len(R) == 100
RES = json.load(open(os.path.join(DD, "results.json"))); META = json.load(open(D + "meta.json"))
M = 10.0; ARMS = [("A", "while answering"), ("C", "while reading the question")]
EXID, K = 0, 5
neg = lambda s: s.replace("-", "\u2212")
pct = lambda v: 100 * (math.exp(v) - 1)

# ---------------- numbers ----------------
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
Z = {a: np.array([spec(r, a, M) for r in R]) for a, _ in ARMS}
ST = {}
for a, _ in ARMS:
    mu = float(Z[a].mean()); assert abs(mu - RES["spec"][f"{a}|{M}"]["mean"]) < 1e-9
    ST[a] = dict(mean=mu, p=signflip(Z[a]), att=RES["manip"][a][str(M)]["mean"])
    print(f"{a} x10: {pct(mu):+.1f}% CI [{pct(RES['spec'][f'{a}|{M}']['lo']):+.1f}, {pct(RES['spec'][f'{a}|{M}']['hi']):+.1f}] p={ST[a]['p']:.2g} "
          f"({pstr(ST[a]['p'], floor=1/200001)}); attention x{ST[a]['att']:.2f}; prompts above 0: {(Z[a] > 0).mean():.0%}")
DOSES = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
DL = {a: [(0.0, 0.0, 0.0) if m == 1.0 else tuple(RES["spec"][f"{a}|{m}"][k] for k in ("mean", "lo", "hi")) for m in DOSES] for a, _ in ARMS}
for a, _ in ARMS: print(a, "dose", [f"{pct(v[0]):+.0f}%" for v in DL[a]])

# ---------------- the example prompt ----------------
rng = random.Random(META["seed"]); POOL = META["cats"]; XY = []
for i in range(META["n"]):   # the run's draw order (privileged_words.py): cats, x, y, words, label and option permutations
    cats = rng.sample(sorted(POOL), 8); x, y = rng.randrange(10, 90), rng.randrange(10, 90); segs = [rng.choice(POOL[c]) for c in cats]
    rng.sample(range(8), 8); rng.sample(range(1, 9), 8); XY.append((x, y, segs, cats))
rec = R[EXID]; x, y, segs, cats = XY[EXID]
assert segs == rec["words"] and cats == rec["cats"] and str(x + y) == rec["ans"].strip() and rec["lab"] == list(range(1, 9))
QLINE = f"Unrelated question: what is {x} + {y}? Answer with just the number."
QTEXT = "Which word did you attend to most while writing your answer?"
print("example", rec["words"], QLINE, rec["ans"], "boosted", rec["words"][K])

# ---------------- violins ----------------
VH = 1150; TOPH = 250
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
px = lambda v: v * 72 / 100
fig = plt.figure(figsize=(W / 100, VH / 100), dpi=100)
TV = [0, 100, 300, 700, 1500, 3100]   # x1, x2, x4, x8, x16, x32: evenly spaced on the log axis
allz = np.concatenate(list(Z.values())); ylo, yhi = min(allz.min(), 0) - 0.35, max(allz.max(), math.log(32)) + 0.12
for j, (a, nm) in enumerate(ARMS):
    x0 = PAD + j * (CW + GAP)
    ax = fig.add_axes([x0 / W, 20 / VH, CW / W, (VH - TOPH - 20) / VH])
    z = Z[a]; g = np.random.default_rng(1 + j)
    kde = gaussian_kde(z, bw_method=0.35); yy = np.linspace(z.min() - 0.06, z.max() + 0.06, 300); dd = kde(yy); hw = 0.30 * dd / dd.max()
    ax.fill_betweenx(yy, -hw, hw, color=PURPLE, alpha=0.18, lw=0, zorder=1)
    ax.plot(np.r_[hw, -hw[::-1], hw[:1]], np.r_[yy, yy[::-1], yy[:1]], color=PURPLE, lw=2.5, zorder=1)
    xj = g.uniform(-0.9, 0.9, len(z)) * np.interp(z, yy, hw)
    ax.scatter(xj, z, s=520, c=pct_colour(np.exp(z)), alpha=0.5, edgecolors="none", zorder=3); ax.scatter(xj, z, s=5, color="black", zorder=4)
    ax.plot([-0.12, 0.12], [ST[a]["mean"]] * 2, color=INK, lw=4, zorder=7, solid_capstyle="butt")
    ax.plot([-0.62 if j == 0 else -1, 1], [0, 0], color=INK, lw=4, zorder=0)
    if j == 0:
        for t in TV:
            yv = math.log(1 + t / 100)
            ax.text(-0.98, yv, f"+{t:,}%".replace("+0%", "0%"), ha="left", va="center", fontsize=px(58), fontweight="bold", color=INK)
        ax.text(-0.40, math.log(10), "report names\nthe boosted\nword more\noften than\nthe others", ha="left", va="center", fontsize=px(54), fontweight="bold", color=INK, linespacing=1.1)
    ax.set_xlim(-1, 1); ax.set_ylim(ylo, yhi); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    cx = (x0 + CW / 2) / W
    fig.text(cx, 1 - 4 / VH, neg(f"{pct(ST[a]['mean']):+.0f}%"), ha="center", va="top", fontsize=89, fontweight="bold", color=PURPLE)
    fig.text(cx, 1 - 146 / VH, pstr(ST[a]["p"], floor=1 / 200001), ha="center", va="top", fontsize=56, color=PURPLE)
    if j == 1:   # key, right of the purple violin
        kx, ky = 0.26, math.log(2.2)
        ax.scatter([kx], [ky], s=520, c=pct_colour([4.0]), alpha=0.5, edgecolors="none"); ax.scatter([kx], [ky], s=5, color="black")
        ax.text(kx + 0.07, ky, "= one prompt", ha="left", va="center", fontsize=px(54), color=INK)
VP = os.path.join(HERE, "_pw_violins.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)

# ---------------- dose line ----------------
DH = 640
fig = plt.figure(figsize=(W / 100, DH / 100), dpi=100)
ax = fig.add_axes([700 / W, 90 / DH, (W - 700 - 640) / W, (DH - 110) / DH])
lx = np.log(DOSES)
for a, nm in ARMS:
    col = PURPLE if a == "C" else INK
    mu, lo, hi = (np.array([v[i] for v in DL[a]]) for i in range(3))
    ax.fill_between(lx, lo, hi, color=col, alpha=0.18, lw=0); ax.plot(lx, mu, color=col, lw=6, marker="o", ms=16, zorder=3)
    ax.text(lx[-1] + 0.14, mu[-1] + (0 if a == "C" else 0.12), f"boost {nm}".replace("reading ", "reading\n"), ha="left", va="center", fontsize=px(56), fontweight="bold", color=col, linespacing=1.1)
ax.axhline(0, color=INK, lw=2.5, zorder=0)
for t in (0, 100, 300, 700):
    ax.text(lx[0] - 0.22, math.log(1 + t / 100), neg(f"{t:+d}%") if t else "0%", ha="right", va="center", fontsize=px(54), fontweight="bold", color=INK)
ax.text(lx[0] - 1.05, math.log(3), "report names\nthe boosted\nword", ha="right", va="center", fontsize=px(54), fontweight="bold", color=INK, linespacing=1.1)
TL = ["×¼", "×½", "none", "×2", "×5", "×10"]
for xv, t in zip(lx, TL): ax.text(xv, math.log(0.7), t, ha="center", va="top", fontsize=px(58), color=INK)
ax.text(lx[0] - 0.95, math.log(0.7), "boost", ha="right", va="top", fontsize=px(58), fontweight="bold", color=INK)
ax.set_xlim(lx[0] - 0.15, lx[-1] + 0.15); ax.set_ylim(math.log(0.62), math.log(8.6)); ax.axis("off")
DP = os.path.join(HERE, "_pw_dose.png"); fig.savefig(DP, dpi=100, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()
robot = b64(ROBOT); robot_inj = injected_robot(ROBOT)
def card(i, a, nm):
    wl = "".join(f'<div class="w"><b>[W{p + 1}]</b> <span {f"class=k id=wk{i}" if p == K else ""}>{html.escape(rec["words"][p])}</span></div>' for p in range(8))
    rob = robot_inj if a == "A" else robot
    return f'''<div class="col"><div class="t1">boost {nm}</div><div class="t2">attention to the boosted word ×{ST[a]["att"]:.1f}</div>
<div class="ub" id="ub{i}">Here is a list of eight words.<div class="wl">{wl}</div><div>{html.escape(QLINE)}</div></div>
<div class="mrow"><img class="rob" id="rob{i}" src="data:image/png;base64,{rob}"><div class="mb" id="mb{i}">{html.escape(rec["ans"].strip())}</div></div>
<div class="ub2" id="rq{i}">{QTEXT} …</div>
<div class="mrow fr"><img class="rob" id="frob{i}" src="data:image/png;base64,{robot_inj if a == "C" else robot}"><div class="mb fb" id="fb{i}"><span class="pf">[W</span><span class="qm">?</span></div>
<div class="fl"><div>prefilled</div><div class="flq">chance of each<br>number read here</div></div></div></div>'''
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:18px {PAD}px 0;position:relative}}
.cols{{display:flex;gap:{GAP}px}} .col{{width:{CW}px;display:flex;flex-direction:column;position:relative;z-index:2}}
.t1{{font-size:64px;font-weight:800;white-space:nowrap}} .t2{{font-size:46px;margin:2px 0 16px;white-space:nowrap}}
.ub{{background:#1F8BFF;color:#fff;border-radius:32px;padding:18px 26px;font-size:46px;line-height:1.2;margin-right:170px}}
.wl{{display:grid;grid-template-rows:repeat(4,auto);grid-auto-flow:column;column-gap:60px;margin:6px 0 6px;width:max-content}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
.mrow{{display:flex;align-items:center;gap:16px;margin-top:40px;margin-left:30px}}
.rob{{width:120px;height:120px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:28px;padding:8px 32px;font-size:66px;font-weight:700}}
.fr{{margin-top:34px}} .fb{{font-size:60px;padding:6px 26px;white-space:nowrap}} .pf{{border:4px dashed {INK};border-radius:10px;padding:0 8px}}
.qm{{display:inline-block;border:5px solid {PURPLE};color:{PURPLE};border-radius:10px;padding:0 10px;margin-left:8px}}
.fl{{font-size:40px;font-weight:700;line-height:1.15;margin-left:10px;white-space:nowrap}} .flq{{color:{PURPLE};font-weight:800}}
.ub2{{background:#1F8BFF;color:#fff;border-radius:32px;padding:16px 26px;font-size:46px;line-height:1.2;margin:44px 170px 0 0}}
.img{{width:{W}px;display:block;margin-left:-{PAD}px}} .vio{{margin-top:70px}} .dose{{margin-top:40px}}
#gl{{position:absolute;left:0;top:0;z-index:0;pointer-events:none}} #ov{{position:absolute;left:0;top:0;pointer-events:none;z-index:3}}
</style></head><body><div class="cols">{card(0, *ARMS[0])}{card(1, *ARMS[1])}</div>
<img class="img vio" src="data:image/png;base64,{b64(VP)}"><img class="img dose" src="data:image/png;base64,{b64(DP)}">
<svg id="gl"></svg><svg id="ov"></svg>
<script>
const R=e=>document.getElementById(e).getBoundingClientRect();
const WW=document.body.scrollWidth, HH=document.body.scrollHeight;
for(const id of ['gl','ov']){{const s=document.getElementById(id);s.setAttribute('width',WW);s.setAttribute('height',HH);}}
// wavy purple aura (hero_inject's) around the boosted queries: arm A the robot and its answer, arm C the report question
const gl=document.getElementById('gl'); let g=`<defs>`;
[[0.004,0.010,36,7,1],[0.006,0.014,28,4.5,2],[0.008,0.02,18,2.5,3]].forEach(([fx,fy,sc,bl,i])=>{{
 g+=`<filter id="w${{i}}" x="-30%" y="-30%" width="160%" height="160%"><feTurbulence type="turbulence" baseFrequency="${{fx}} ${{fy}}" numOctaves="2" seed="${{5*i+1}}" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="${{sc}}" xChannelSelector="R" yChannelSelector="G" result="d"/><feGaussianBlur in="d" stdDeviation="${{bl}}"/></filter>`;}});
g+=`<filter id="haze" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="22"/></filter></defs>`;
const aura=(r,rad)=>{{ const S=(d,st,col,op,f,fill)=>`<rect x="${{r.left-d}}" y="${{r.top-d}}" width="${{r.width+2*d}}" height="${{r.height+2*d}}" rx="${{rad+d}}" fill="${{fill?col:'none'}}" stroke="${{fill?'none':col}}" stroke-width="${{st}}" opacity="${{op}}" filter="url(#${{f}})"/>`;
 return S(8,0,'#B994FF',0.75,'haze',1)+S(50,5,'#C9A8FF',0.45,'w1')+S(32,7,'{LIL}',0.6,'w2')+S(15,9,'#7B3FE4',0.75,'w3')+S(2,8,'{PURPLE}',0.55,'w3'); }};
const ro=R('rob0'), mb=R('mb0');
g+=aura({{left:ro.left+2,top:Math.min(ro.top,mb.top)+4,width:mb.right-ro.left-2,height:Math.max(ro.bottom,mb.bottom)-Math.min(ro.top,mb.top)-8}},40);
const q1=R('rq1'), f1=R('frob1'), b1=R('fb1');   // arm C boosts every report-turn query: the question, the assistant template and the prefilled [W
g+=aura({{left:Math.min(q1.left,f1.left)+2,top:q1.top,width:Math.max(q1.right,b1.right)-Math.min(q1.left,f1.left)-2,height:Math.max(f1.bottom,b1.bottom)-q1.top}},34);
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
 return o+`<defs><mask id="m${{id}}" maskUnits="userSpaceOnUse" x="${{X}}" y="${{Y}}" width="${{Wd}}" height="${{Ht}}"><path d="${{d}}" fill="none" stroke="#fff" stroke-width="${{SW}}" stroke-linecap="round"/><path d="${{h}}" fill="#fff"/></mask></defs><g mask="url(#m${{id}})">${{q}}</g>`; }};
let s='';
for (const i of [0,1]) {{
 const t=R('wk'+i), src=R(i==0?'mb'+i:'rq'+i), ub=R('ub'+i);
 const T=[t.right+14, (t.top+t.bottom)/2], P0=[src.right+6, (src.top+src.bottom)/2], xm=ub.right+70;
 s+=arrow(P0, [xm, P0[1]], [xm, T[1]], T, i);
}}
document.getElementById('ov').innerHTML=s;
</script></body></html>'''
hp = os.path.join(HERE, "_pw.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, "_pw_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3800",
                "--force-device-scale-factor=1", "--virtual-time-budget=3000", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, W, nz[-1] + 24)); out = os.path.join(HERE, "privileged_words.png"); im.save(out); print(out, im.size)
if "--keep" not in sys.argv:
    for f in (raw, hp, VP, DP): os.remove(f)
