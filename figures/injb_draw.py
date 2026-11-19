"""Drawing code shared by the told-injection family (hero_told.py, hero_told_yes.py, hero_top.py), copied from hero_inject.py so they match it:
blue prompt bubbles, the report question and the reply with a wavy purple aura, the purple-eyed robot with the syringe "xdose",
the purple label "inject attention toward paragraph 2", the light-purple wave arrow from the question to [P2], black arrows from the
underlined part of the reply to the column title, and hero-style violins. Style only: nothing here reads data or computes a number."""
import os, html, base64, subprocess
import numpy as np
from PIL import Image
from pal import INK, PURPLE
from robot_injected import injected_robot

AMB = "#F2B53A"; LIL = "#A574F2"
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
ROBOT = "/Users/user/watching/figures/assets/robot.png"
px = lambda v: v * 72 / 100   # css px -> matplotlib points at dpi 100


def short(s, n): return " ".join(s.split()[:n]) + " …"


def syringe(dose):
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
    o.append(f'<text x="197" y="108" font-size="82" font-weight="800" text-anchor="middle" fill="#fff">\u00d7{dose}</text>')
    return f'<svg id="syr" width="380" height="160" viewBox="0 0 380 160">{"".join(o)}</svg>'


def paras(segs, K, LAB, NW=(3, 3, 3, 2)):
    """the four truncated paragraphs; the boosted one (index K) shows more words when needed so it is at least as long as its neighbours
    (the wave arrow's head lands at its right end and must not cover their text)"""
    NW = list(NW); nb = max(len(short(segs[j], NW[j])) for j in (K - 1, K + 1) if 0 <= j < 4)
    while len(short(segs[K], NW[K])) < nb + 2 and NW[K] < len(segs[K].split()): NW[K] += 1
    return "".join(f'<div class="pl"><b>[P{LAB[pos]}]</b> <span {"class=k id=p2t" if pos == K else ""}>{html.escape(short(segs[pos], NW[pos]))}</span></div>' for pos in range(4))


def told_bubble(META, segs, K=1):
    """the told run's real intro, shortened with ellipses as in hero_inject, and the four paragraphs ([P K+1] amber)"""
    INTRO = META["intro"]
    S1 = "I am an interpretability researcher"; S2 = "I am capable of artificially boosting your attention onto one of the four paragraphs below"; S3 = "on 50% of trials"
    for s in (S1, S2, S3, "Here are four paragraphs."): assert s in INTRO, s
    assert META["question"].startswith("Trial 1: Do you notice your attention being boosted onto one of the paragraphs? If so, which one?")
    return f'<div>{S1} … {S2} … {S3}.</div><div>Here are four paragraphs.</div>' + paras(segs, K, [1, 2, 3, 4])


def violin(ax, ys, d, v, vj, dotcol, col, seed, xc=0.0, half=0.55, s=635):
    """hero_inject's violin: translucent fill + 2.5 px outline in col, jittered translucent dots in dotcol with black specks.
    ys, d: density grid; v: per-prompt values; vj: plotted heights (v plus any vertical jitter for discrete values)."""
    hw = half * d / d.max()
    ax.fill_betweenx(ys, xc - hw, xc + hw, color=col, alpha=0.22, lw=0, zorder=1)
    ax.plot(np.r_[xc + hw, xc - hw[::-1], xc + hw[:1]], np.r_[ys, ys[::-1], ys[:1]], color=col, lw=2.5, zorder=1)
    g = np.random.default_rng(seed)
    xj = xc + g.uniform(-0.9, 0.9, len(v)) * np.interp(vj, ys, hw)
    ax.scatter(xj, vj, s=s, c=dotcol, alpha=0.45, edgecolors="none", zorder=3); ax.scatter(xj, vj, s=5, color="black", zorder=4)
    return xj


def key_svg(col):
    r = 30; dx, dy = r + 4, r + 4; ww = dx + r + 16 + 400
    o = [f'<circle cx="{dx}" cy="{dy}" r="{r}" fill="{col}" fill-opacity=".75"/><circle cx="{dx}" cy="{dy}" r="4" fill="#000"/>',
         f'<text x="{dx + r + 16}" y="{dy + 19}" font-size="54" font-weight="800" fill="{INK}">= one prompt</text>']
    return f'<svg class="kbox" id="kb" width="{ww:.0f}" height="{2*r + 8:.0f}" viewBox="0 0 {ww:.0f} {2*r + 8:.0f}">{"".join(o)}</svg>'


def page(W, bubble, rq, reply, titles, vio_png, VW, dose, key=None, pre="", PAD=26, inj="inject attention<br>toward paragraph 2", arrows=(("ry", "c1"),), tipdrop=None):
    """bubble: inner html of the prompt bubble (one span id=p2t); rq: the report question; reply: inner html of the reply bubble
    (underlined parts as <span class=part id=...>); tipdrop: when set, each black arrow is an S-curve leaving the underline straight down and entering
    its title straight down, its tip tipdrop px below the title's line box top (on the cap height); titles: html of the column-title row (ids c1..); key: (left, top, colour) of the
    "= one prompt" key over the violin image; pre: html between the prompt bubble and the question (an element id=rob0 for the plain
    robot's answer; the text ROBOT_PLAIN is replaced by the plain robot's image); arrows: (reply part id, title id) pairs joined by black arrows."""
    robot_inj = injected_robot(ROBOT)
    robot = base64.b64encode(open(ROBOT, "rb").read()).decode()
    VIO = base64.b64encode(open(vio_png, "rb").read()).decode()
    pre = pre.replace("ROBOT_PLAIN", robot)
    kb = key_svg(key[2]) if key else ""
    kjs = f"const kb=document.getElementById('kb'); kb.style.left='{key[0]:.0f}px'; kb.style.top='{key[1]:.0f}px';" if key else ""
    aj = ",".join(f"['{a}','{b}']" for a, b in arrows)
    above = "rowA" if "rowA" in pre else "ub"
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:20px {PAD}px 10px;position:relative}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 26px;font-size:54px;line-height:1.18;margin-right:150px;display:flex;flex-direction:column;gap:8px}}
.pl{{white-space:nowrap}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
#rq{{background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 26px;font-size:54px;line-height:1.18;margin:60px 150px 0 {min(880, W - 910)}px;position:relative;z-index:2}}
.mrow{{display:flex;align-items:center;gap:18px;margin-top:56px;margin-left:420px}}
.rob{{width:260px;height:260px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:14px 38px;font-size:84px;font-weight:700;white-space:nowrap;box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.part{{border-bottom:8px solid {INK};padding-bottom:2px}}
#syr{{position:absolute}}
.inj{{position:absolute;left:30px;font-size:66px;font-weight:800;line-height:1.12;color:{PURPLE}}}
#gl{{position:absolute;left:0;top:0;z-index:-1;pointer-events:none}}
#ov{{position:absolute;left:0;top:0;pointer-events:none;z-index:1}}
.cols{{display:flex;margin-top:150px}}
.ct{{flex:1;min-width:0;text-align:center}}
.cn{{display:inline-block;font-size:76px;font-weight:800;line-height:1.08}}
.vw{{position:relative;margin-top:34px}}
.vio{{width:{VW}px;display:block}}
.kbox{{position:absolute}}
</style></head><body>
<div class="ub" id="ub">{bubble}</div>
{pre}
<div id="rq">{rq}</div>
<div class="inj" id="inj">{inj}</div>
<div class="mrow" id="mrow"><img class="rob" id="rob1" src="data:image/png;base64,{robot_inj}"><div class="mb" id="mb">{reply}</div></div>
{syringe(dose)}
{titles}
<div class="vw" id="vw"><img class="vio" src="data:image/png;base64,{VIO}">{kb}</div>
<svg id="gl"></svg><svg id="ov"></svg>
<script>
const R=e=>document.getElementById(e).getBoundingClientRect();
const syr=document.getElementById('syr'), A=40, inj=document.getElementById('inj');
const place=()=>{{ const rb=R('rob1'); const tipx=rb.left+rb.width*0.29, tipy=rb.top+rb.height*0.72;
 syr.style.left=(tipx-380)+'px'; syr.style.top=(tipy-80)+'px'; syr.style.transformOrigin='380px 80px'; syr.style.transform=`rotate(${{A}}deg) scale(1.38)`; }};
// the label sits just under what is above it; the injected robot's row moves down until the syringe clears the label
inj.style.top=(Math.max(R('rq').top-46, R('{above}').bottom+22))+'px'; place();
const gap=R('inj').bottom+14-R('syr').top; if (gap>0) {{ const mr=document.getElementById('mrow'); mr.style.marginTop=(parseFloat(getComputedStyle(mr).marginTop)+gap)+'px'; place(); }}
const WW=document.body.scrollWidth, HH=document.body.scrollHeight;
for(const id of ['gl','ov']){{const s=document.getElementById(id);s.setAttribute('width',WW);s.setAttribute('height',HH);}}
const rq0=R('rq');
{kjs}
const gl=document.getElementById('gl'); let g=`<defs>`;
[[0.004,0.010,36,7,1],[0.006,0.014,28,4.5,2],[0.008,0.02,18,2.5,3]].forEach(([fx,fy,sc,bl,i])=>{{
 g+=`<filter id="w${{i}}" x="-30%" y="-30%" width="160%" height="160%"><feTurbulence type="turbulence" baseFrequency="${{fx}} ${{fy}}" numOctaves="2" seed="${{5*i+1}}" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="${{sc}}" xChannelSelector="R" yChannelSelector="G" result="d"/><feGaussianBlur in="d" stdDeviation="${{bl}}"/></filter>`;}});
g+=`<filter id="haze" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="22"/></filter></defs>`;
const aura=(r,rad)=>{{ const S=(d,st,col,op,f,fill)=>`<rect x="${{r.left-d}}" y="${{r.top-d}}" width="${{r.width+2*d}}" height="${{r.height+2*d}}" rx="${{rad+d}}" fill="${{fill?col:'none'}}" stroke="${{fill?'none':col}}" stroke-width="${{st}}" opacity="${{op}}" filter="url(#${{f}})"/>`;
 return S(8,0,'#B994FF',0.75,'haze',1)+S(50,5,'#C9A8FF',0.45,'w1')+S(32,7,'{LIL}',0.6,'w2')+S(15,9,'#7B3FE4',0.75,'w3')+S(2,8,'{PURPLE}',0.55,'w3'); }};
g+=aura(rq0,34); g+=aura(R('mb'),30); const ro=R('rob1'); g+=aura({{left:ro.left+6,top:ro.top+10,width:ro.width-12,height:ro.height-16}},64);
gl.innerHTML=g;
const SW0=20;
const arrow=(p0,c1,c2,tip,col,ol)=>{{ const SW=ol?SW0*1.7:SW0; const L=SW*2.7, H=SW*1.6; let ux=tip[0]-c2[0], uy=tip[1]-c2[1]; const n=Math.hypot(ux,uy); ux/=n; uy/=n;
 const e=[tip[0]-L*ux, tip[1]-L*uy], b=[e[0]-1.5*ux, e[1]-1.5*uy];
 const d=`M${{p0}} C${{c1}} ${{c2}} ${{e}}`, h=`M${{tip}} L${{b[0]-H*uy}},${{b[1]+H*ux}} L${{b[0]+H*uy}},${{b[1]-H*ux}} Z`;
 const o=ol?`<path d="${{d}}" fill="none" stroke="{INK}" stroke-width="${{SW+16}}" stroke-linecap="round"/><path d="${{h}}" fill="{INK}" stroke="{INK}" stroke-width="16" stroke-linejoin="round"/>`:'';
 let w='';
 if (ol) {{
  const B=(t,i)=>{{const u=1-t; return u*u*u*p0[i]+3*u*u*t*c1[i]+3*u*t*t*c2[i]+t*t*t*e[i];}};
  const P=[]; const N=600;
  for (let k=0;k<=N;k++){{ const t=k/N; P.push([B(t,0),B(t,1)]); }}
  for (let k=1;k<=40;k++) P.push([e[0]+ux*L*1.15*k/40, e[1]+uy*L*1.15*k/40]);
  const S=[0]; for (let k=1;k<P.length;k++) S.push(S[k-1]+Math.hypot(P[k][0]-P[k-1][0],P[k][1]-P[k-1][1]));
  const lam=SW*1.9, half=H*1.05, rows=8, bend=SW*0.9;
  const lerp=(a,b,f)=>Math.round(a+(b-a)*f), C1=[0xE6,0xDB,0xF7], C2=[0x9C,0x7B,0xD8];
  let q='';
  for (let k=0;k<P.length-1;k++){{
   const a=P[Math.max(k-1,0)], b=P[Math.min(k+2,P.length-1)]; let tx=b[0]-a[0], ty=b[1]-a[1]; const m=Math.hypot(tx,ty)||1; tx/=m; ty/=m; const nx=-ty, ny=tx;
   const ds=Math.hypot(P[k+1][0]-P[k][0],P[k+1][1]-P[k][1])+0.8;
   for (let r=0;r<rows;r++){{
    const w0=-half+2*half*r/rows-0.7, w1=-half+2*half*(r+1)/rows+0.7, wm=(w0+w1)/2;
    const ph=(S[k] + bend*Math.pow(Math.abs(wm)/half,1.6))/lam;
    const f=0.5+0.5*Math.sin(2*Math.PI*ph);
    const col=`rgb(${{lerp(C1[0],C2[0],f)}},${{lerp(C1[1],C2[1],f)}},${{lerp(C1[2],C2[2],f)}})`;
    const x=P[k][0], y=P[k][1];
    q+=`<path d="M${{(x+nx*w0).toFixed(1)}},${{(y+ny*w0).toFixed(1)}} L${{(x+nx*w1).toFixed(1)}},${{(y+ny*w1).toFixed(1)}} L${{(x+nx*w1+tx*ds).toFixed(1)}},${{(y+ny*w1+ty*ds).toFixed(1)}} L${{(x+nx*w0+tx*ds).toFixed(1)}},${{(y+ny*w0+ty*ds).toFixed(1)}} Z" fill="${{col}}" stroke="${{col}}" stroke-width="0.6"/>`; }} }}
  const xs=P.map(p=>p[0]), ys=P.map(p=>p[1]), pad=SW*4;
  const X=Math.min(...xs)-pad, Y=Math.min(...ys)-pad, Wd=Math.max(...xs)-X+pad, Ht=Math.max(...ys)-Y+pad;
  w=`<defs><mask id="amask" maskUnits="userSpaceOnUse" x="${{X}}" y="${{Y}}" width="${{Wd}}" height="${{Ht}}"><path d="${{d}}" fill="none" stroke="#fff" stroke-width="${{SW}}" stroke-linecap="round"/><path d="${{h}}" fill="#fff"/></mask></defs>`+
   `<g mask="url(#amask)">${{q}}</g>`; }}
 if (ol) return o+w;
 return o+`<path d="${{d}}" fill="none" stroke="${{col}}" stroke-width="${{SW}}" stroke-linecap="round"/>`+w+`<path d="${{h}}" fill="${{col}}"/>`; }};
// light purple wave arrow: up out of the report question's top edge, one quarter-turn, into the right end of [P2]
const t=R('p2t'), q=R('rq');
// rise clear of the text of the bubble lines below [P2] (measured as text, not as the full-width line boxes)
let xr=0; for (const el of document.querySelectorAll('#ub > div')) {{ const rg=document.createRange(); rg.selectNodeContents(el); const b=rg.getBoundingClientRect(); if (b.top>t.bottom-4) xr=Math.max(xr,b.right); }}
const clear=xr+110>q.left+260;   // a long line below [P2]: rise further right and more steeply so the curve passes beyond its end
const P0=[clear?Math.min(q.right-140, xr+450):q.left+260, q.top+50], T=[t.right+12, (t.top+t.bottom)/2];
let s=arrow(P0, [P0[0], P0[1]-(clear?0.8:0.55)*(P0[1]-T[1])], [T[0]+0.55*(P0[0]-T[0]), T[1]], T, '#A98BD6', 1);
// black: from under each underlined part of the reply down to its column title
const cx=r=>(r.left+r.right)/2;
const TD={'null' if tipdrop is None else tipdrop};
for (const [pa,ti] of [{aj}]) {{ const a=R(pa), c=R(ti); const ys=a.bottom, yt=TD===null?c.top-12:c.top+TD, dy=yt-ys, p0x=cx(a), tx=cx(c), k=TD===null?0.55:0.62;
 s+=arrow([p0x,ys],[p0x,ys+k*dy],[tx,yt-k*dy],[tx,yt],'{INK}').replace('stroke-linecap="round"','stroke-linecap="butt"'); }}
document.getElementById('ov').innerHTML=s;
</script></body></html>'''


def render(HTML, here, out, W, jpg=False, keep=False):
    hp = os.path.join(here, f"_{out}.html"); open(hp, "w").write(HTML); raw = os.path.join(here, f"_{out}_raw.png")
    subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},4000",
                    "--force-device-scale-factor=1", "--virtual-time-budget=3000", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
    im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
    im = im.crop((0, 0, W, nz[-1] + 20)); im.save(os.path.join(here, out + ".png"))
    if jpg: im.save(os.path.join(here, out + ".jpg"), quality=92)
    print(out + ".png", im.size, "aspect", round(im.size[1] / im.size[0], 2))
    if not keep: os.remove(raw); os.remove(hp)
