"""Drawing helpers for the injection-family figures (trace3.py, privileged_fig.py, hero_question.py), copied from hero_inject.py
so the three match the hero: the syringe with its dose, the wavy purple aura (SVG turbulence filters), the light-purple wave arrow
(where the boost sends attention), hero-style violins (translucent fill, 2.5 px outline, pct_colour dots with black specks), report bars
(pair_bars with larger text), and the headless-Chrome render. Nothing here reads data."""
import os, math, subprocess
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde
from pal import INK, PURPLE, RED, pct_colour

AMB = "#F2B53A"; LIL = "#A574F2"; GREY = "#B8B8C0"
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
ROBOT = "/Users/user/watching/figures/assets/robot.png"


def syringe(dose, sid="syr"):
    """hero_inject's hand-drawn syringe, needle tip at (380, 80) of a 380 x 160 box; the page rotates it about the tip"""
    o = [f'<path d="M380,80 L364,76 L364,84 Z" fill="{INK}"/>', f'<rect x="314" y="76" width="50" height="8" fill="{INK}"/>',
         f'<path d="M314,68 L292,58 L292,102 L314,92 Z" fill="{INK}"/>',
         f'<rect x="80" y="22" width="212" height="116" rx="14" fill="#fff" stroke="{INK}" stroke-width="7"/>',
         f'<rect x="109" y="29" width="176" height="102" fill="{LIL}"/>', f'<rect x="87" y="29" width="22" height="102" fill="{INK}"/>',
         f'<rect x="62" y="4" width="18" height="152" rx="7" fill="{INK}"/>', f'<rect x="26" y="71" width="36" height="18" fill="{INK}"/>',
         f'<rect x="2" y="40" width="24" height="80" rx="7" fill="{INK}"/>',
         f'<text x="197" y="108" font-size="82" font-weight="800" text-anchor="middle" fill="#fff">\u00d7{dose}</text>']
    return f'<svg class="syr" id="{sid}" width="380" height="160" viewBox="0 0 380 160" style="position:absolute">{"".join(o)}</svg>'


# JS (inside an f-string-free block): R(id) rects, aura(rect, radius) -> svg string, AURADEFS, arrow(p0,c1,c2,tip,col,ol), placeSyr(id, robotId, angle, scale)
JS = r"""
const R=e=>document.getElementById(e).getBoundingClientRect();
const WW=document.body.scrollWidth, HH=document.body.scrollHeight;
for(const id of ['gl','ov']){const s=document.getElementById(id); if(s){s.setAttribute('width',WW);s.setAttribute('height',HH);}}
let AURADEFS=`<defs>`;
[[0.004,0.010,36,7,1],[0.006,0.014,28,4.5,2],[0.008,0.02,18,2.5,3]].forEach(([fx,fy,sc,bl,i])=>{
 AURADEFS+=`<filter id="w${i}" x="-30%" y="-30%" width="160%" height="160%"><feTurbulence type="turbulence" baseFrequency="${fx} ${fy}" numOctaves="2" seed="${5*i+1}" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="${sc}" xChannelSelector="R" yChannelSelector="G" result="d"/><feGaussianBlur in="d" stdDeviation="${bl}"/></filter>`;});
AURADEFS+=`<filter id="haze" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="22"/></filter></defs>`;
const aura=(r,rad,k)=>{ k=k||1; const S=(d,st,col,op,f,fill)=>`<rect x="${r.left-d*k}" y="${r.top-d*k}" width="${r.width+2*d*k}" height="${r.height+2*d*k}" rx="${rad+d*k}" fill="${fill?col:'none'}" stroke="${fill?'none':col}" stroke-width="${st*k}" opacity="${op}" filter="url(#${f})"/>`;
 return S(8,0,'#B994FF',0.75,'haze',1)+S(50,5,'#C9A8FF',0.45,'w1')+S(32,7,'#A574F2',0.6,'w2')+S(15,9,'#7B3FE4',0.75,'w3')+S(2,8,'#5E2B97',0.55,'w3'); };
const roboAura=(id,rad,k)=>{const ro=R(id); return aura({left:ro.left+ro.width*0.03,top:ro.top+ro.height*0.04,width:ro.width*0.94,height:ro.height*0.92},rad,k);};
const placeSyr=(sid,rid,A,SC)=>{const rb=R(rid), syr=document.getElementById(sid);
 const tipx=rb.left+rb.width*0.29+window.scrollX, tipy=rb.top+rb.height*0.72+window.scrollY;
 syr.style.left=(tipx-380)+'px'; syr.style.top=(tipy-80)+'px'; syr.style.transformOrigin='380px 80px'; syr.style.transform=`rotate(${A}deg) scale(${SC})`;};
let AID=0;
const arrow=(p0,c1,c2,tip,col,ol,SW0)=>{ SW0=SW0||20; const SW=ol?SW0*1.7:SW0; const L=SW*2.7, H=SW*1.6; let ux=tip[0]-c2[0], uy=tip[1]-c2[1]; const n=Math.hypot(ux,uy); ux/=n; uy/=n;
 const e=[tip[0]-L*ux, tip[1]-L*uy], b=[e[0]+1.5*ux, e[1]+1.5*uy];
 const d=`M${p0} C${c1} ${c2} ${e}`, h=`M${tip} L${b[0]-H*uy},${b[1]+H*ux} L${b[0]+H*uy},${b[1]-H*ux} Z`;
 const o=ol?`<path d="${d}" fill="none" stroke="#1D1D1F" stroke-width="${SW+16*SW0/20}" stroke-linecap="round"/><path d="${h}" fill="#1D1D1F" stroke="#1D1D1F" stroke-width="${16*SW0/20}" stroke-linejoin="round"/>`:'';
 let w='';
 if (ol) {
  const B=(t,i)=>{const u=1-t; return u*u*u*p0[i]+3*u*u*t*c1[i]+3*u*t*t*c2[i]+t*t*t*e[i];};
  const P=[]; const N=600;
  for (let k=0;k<=N;k++){ const t=k/N; P.push([B(t,0),B(t,1)]); }
  for (let k=1;k<=40;k++) P.push([e[0]+ux*L*1.15*k/40, e[1]+uy*L*1.15*k/40]);
  const S=[0]; for (let k=1;k<P.length;k++) S.push(S[k-1]+Math.hypot(P[k][0]-P[k-1][0],P[k][1]-P[k-1][1]));
  const lam=SW*1.9, half=H*1.05, rows=8, bend=SW*0.9;
  const lerp=(a,b,f)=>Math.round(a+(b-a)*f), C1=[0xE6,0xDB,0xF7], C2=[0x9C,0x7B,0xD8];
  let q='';
  for (let k=0;k<P.length-1;k++){
   const a=P[Math.max(k-1,0)], bb=P[Math.min(k+2,P.length-1)]; let tx=bb[0]-a[0], ty=bb[1]-a[1]; const m=Math.hypot(tx,ty)||1; tx/=m; ty/=m; const nx=-ty, ny=tx;
   const ds=Math.hypot(P[k+1][0]-P[k][0],P[k+1][1]-P[k][1])+0.8;
   for (let r=0;r<rows;r++){
    const w0=-half+2*half*r/rows-0.7, w1=-half+2*half*(r+1)/rows+0.7, wm=(w0+w1)/2;
    const ph=(S[k] + bend*Math.pow(Math.abs(wm)/half,1.6))/lam;
    const f=0.5+0.5*Math.sin(2*Math.PI*ph);
    const cc=`rgb(${lerp(C1[0],C2[0],f)},${lerp(C1[1],C2[1],f)},${lerp(C1[2],C2[2],f)})`;
    const x=P[k][0], y=P[k][1];
    q+=`<path d="M${(x+nx*w0).toFixed(1)},${(y+ny*w0).toFixed(1)} L${(x+nx*w1).toFixed(1)},${(y+ny*w1).toFixed(1)} L${(x+nx*w1+tx*ds).toFixed(1)},${(y+ny*w1+ty*ds).toFixed(1)} L${(x+nx*w0+tx*ds).toFixed(1)},${(y+ny*w0+ty*ds).toFixed(1)} Z" fill="${cc}" stroke="${cc}" stroke-width="0.6"/>`; } }
  const xs=P.map(p=>p[0]), ys=P.map(p=>p[1]), pad=SW*4;
  const X=Math.min(...xs)-pad, Y=Math.min(...ys)-pad, Wd=Math.max(...xs)-X+pad, Ht=Math.max(...ys)-Y+pad;
  AID+=1;
  w=`<defs><mask id="amask${AID}" maskUnits="userSpaceOnUse" x="${X}" y="${Y}" width="${Wd}" height="${Ht}"><path d="${d}" fill="none" stroke="#fff" stroke-width="${SW}" stroke-linecap="round"/><path d="${h}" fill="#fff"/></mask></defs>`+
   `<g mask="url(#amask${AID})">${q}</g>`; }
 if (ol) return o+w;
 return o+`<path d="${d}" fill="none" stroke="${col}" stroke-width="${SW}" stroke-linecap="round"/>`+w+`<path d="${h}" fill="${col}"/>`; };
// the same wave-filled arrow along any SVG path d (ending at the head's base), head pointing along unit vector u to tip
const arrowD=(d,tip,u,SW0)=>{ SW0=SW0||20; const SW=SW0*1.7, L=SW*2.7, H=SW*1.6; const ux=u[0], uy=u[1];
 const e=[tip[0]-L*ux, tip[1]-L*uy], b=[e[0]+1.5*ux, e[1]+1.5*uy];
 const h=`M${tip} L${b[0]-H*uy},${b[1]+H*ux} L${b[0]+H*uy},${b[1]-H*ux} Z`;
 const NS='http://www.w3.org/2000/svg', tmp=document.createElementNS(NS,'svg'), pe=document.createElementNS(NS,'path'); pe.setAttribute('d',d); tmp.appendChild(pe); document.body.appendChild(tmp);
 const TL=pe.getTotalLength(), P=[]; for(let k=0;k<=900;k++){const q=pe.getPointAtLength(TL*k/900); P.push([q.x,q.y]);} tmp.remove();
 for (let k=1;k<=40;k++) P.push([e[0]+ux*L*1.15*k/40, e[1]+uy*L*1.15*k/40]);
 const o=`<path d="${d}" fill="none" stroke="#1D1D1F" stroke-width="${SW+16*SW0/20}" stroke-linecap="round" stroke-linejoin="round"/><path d="${h}" fill="#1D1D1F" stroke="#1D1D1F" stroke-width="${16*SW0/20}" stroke-linejoin="round"/>`;
 const S=[0]; for (let k=1;k<P.length;k++) S.push(S[k-1]+Math.hypot(P[k][0]-P[k-1][0],P[k][1]-P[k-1][1]));
 const lam=SW*1.9, half=H*1.05, rows=8, bend=SW*0.9; const lerp=(a,bb,f)=>Math.round(a+(bb-a)*f), C1=[0xE6,0xDB,0xF7], C2=[0x9C,0x7B,0xD8]; let q='';
 for (let k=0;k<P.length-1;k++){
  const a=P[Math.max(k-1,0)], bb=P[Math.min(k+2,P.length-1)]; let tx=bb[0]-a[0], ty=bb[1]-a[1]; const m=Math.hypot(tx,ty)||1; tx/=m; ty/=m; const nx=-ty, ny=tx;
  const ds=Math.hypot(P[k+1][0]-P[k][0],P[k+1][1]-P[k][1])+0.8;
  for (let r=0;r<rows;r++){ const w0=-half+2*half*r/rows-0.7, w1=-half+2*half*(r+1)/rows+0.7, wm=(w0+w1)/2;
   const f=0.5+0.5*Math.sin(2*Math.PI*(S[k]+bend*Math.pow(Math.abs(wm)/half,1.6))/lam);
   const cc=`rgb(${lerp(C1[0],C2[0],f)},${lerp(C1[1],C2[1],f)},${lerp(C1[2],C2[2],f)})`; const x=P[k][0], y=P[k][1];
   q+=`<path d="M${(x+nx*w0).toFixed(1)},${(y+ny*w0).toFixed(1)} L${(x+nx*w1).toFixed(1)},${(y+ny*w1).toFixed(1)} L${(x+nx*w1+tx*ds).toFixed(1)},${(y+ny*w1+ty*ds).toFixed(1)} L${(x+nx*w0+tx*ds).toFixed(1)},${(y+ny*w0+ty*ds).toFixed(1)} Z" fill="${cc}" stroke="${cc}" stroke-width="0.6"/>`; } }
 AID+=1; const xs=P.map(p=>p[0]), ys=P.map(p=>p[1]), pad=SW*4, X=Math.min(...xs)-pad, Y=Math.min(...ys)-pad;
 return o+`<defs><mask id="amask${AID}" maskUnits="userSpaceOnUse" x="${X}" y="${Y}" width="${Math.max(...xs)-X+pad}" height="${Math.max(...ys)-Y+pad}"><path d="${d}" fill="none" stroke="#fff" stroke-width="${SW}" stroke-linecap="round" stroke-linejoin="round"/><path d="${h}" fill="#fff"/></mask></defs><g mask="url(#amask${AID})">${q}</g>`; };
// right out of a bubble, up a lane at x=lx with rounded corners, left into the end of a text line at tip
const laneArrow=(p0,lx,tip,SW0,r)=>{ r=r||40; const SW=(SW0||20)*1.7, L=SW*2.7, ex=tip[0]+L;
 const d=`M${p0[0]},${p0[1]} H${lx-r} Q${lx},${p0[1]} ${lx},${p0[1]-r} V${tip[1]+r} Q${lx},${tip[1]} ${lx-r},${tip[1]} H${ex}`;
 return arrowD(d,tip,[-1,0],SW0); };
const quad=(p0,Q,tip,col,SW0)=>arrow(p0,[p0[0]+2/3*(Q[0]-p0[0]),p0[1]+2/3*(Q[1]-p0[1])],[tip[0]+2/3*(Q[0]-tip[0]),tip[1]+2/3*(Q[1]-tip[1])],tip,col,0,SW0);
"""


def violin(ax, x0, z, up, ylo, yhi, width=0.42, dot=635, spec=5, bw=None, seed=1, lw=2.5):
    """hero violin at x0 on a shared y: translucent fill + outline (PURPLE rise / RED drop), jittered pct_colour dots (exp(z) = ratio) with black specks"""
    col = PURPLE if up else RED
    kde = gaussian_kde(z, bw_method=bw)
    yy = np.linspace(z.min() - 0.04 * (yhi - ylo), z.max() + 0.04 * (yhi - ylo), 400); dd = kde(yy); hw = width * dd / dd.max()
    ax.fill_betweenx(yy, x0 - hw, x0 + hw, color=col, alpha=0.22, lw=0, zorder=1)
    ax.plot(np.r_[x0 + hw, (x0 - hw)[::-1], x0 + hw[:1]], np.r_[yy, yy[::-1], yy[:1]], color=col, lw=lw, zorder=1)
    g = np.random.default_rng(seed); xj = x0 + g.uniform(-0.9, 0.9, len(z)) * np.interp(z, yy, hw)
    ax.scatter(xj, z, s=dot, c=pct_colour(np.exp(z)), alpha=0.45, edgecolors="none", zorder=3)
    ax.scatter(xj, z, s=spec, color="black", zorder=4)
    return xj


def bars(p0, p1, order, K, bw=44, gw=136, x0=150, top=0, F=48, H=230):
    """pair_bars.py drawn with hero-size text: grey = no boost, purple = boosted, log y 0.1%..100%; boosted paragraph's digit on an amber chip"""
    base = H + 40; o = []
    Y = lambda v: base - H * (math.log10(max(v, 1e-3)) + 3) / 3
    for t, lab in ((1.0, "100%"), (0.01, "1%"), (0.001, "0.1%")):
        o.append(f'<text x="{x0 - 16}" y="{Y(t) + F * 0.35:.1f}" font-size="{F}" font-weight="700" text-anchor="end" fill="{INK}">{lab}</text>')
    o.append(f'<line x1="{x0 - 6}" x2="{x0 - 6}" y1="{Y(1):.1f}" y2="{base}" stroke="{INK}" stroke-width="3"/>')
    sets = ((p0, GREY),) if p1 is None else ((p0, GREY), (p1, PURPLE))
    for i, pos in enumerate(order):
        x = x0 + 8 + i * gw
        for j, (pr, c) in enumerate(sets):
            y = Y(pr[pos]); o.append(f'<rect x="{x + j * (bw + 4)}" y="{y:.1f}" width="{bw}" height="{base - y:.1f}" rx="5" fill="{c}"/>')
        cx = x + (bw + 2 if p1 is not None else bw / 2)
        if pos == K:
            o.append(f'<rect x="{cx - F * 0.7}" y="{base + 10}" width="{F * 1.4}" height="{F * 1.3}" rx="10" fill="{AMB}"/>')
            if p1 is None:
                top = min(top, Y(p0[pos]) - F * 1.4)
                o.append(f'<text x="{cx}" y="{Y(p0[pos]) - 14:.1f}" font-size="{F}" font-weight="800" text-anchor="middle" fill="{INK}">{100 * p0[pos]:.1f}%</text>')
            else:
                yt = Y(max(p0[pos], p1[pos])) - 14; top = min(top, yt - F * 2.2)
                o.append(f'<text x="{cx}" y="{yt - F * 1.12:.1f}" font-size="{F}" font-weight="800" text-anchor="middle" fill="{INK}">{100 * p0[pos]:.1f}% \u2192</text>')
                o.append(f'<text x="{cx}" y="{yt:.1f}" font-size="{F}" font-weight="800" text-anchor="middle" fill="{PURPLE}">{100 * p1[pos]:.1f}%</text>')
        o.append(f'<text x="{cx}" y="{base + 10 + F * 1.0:.1f}" font-size="{F}" font-weight="700" text-anchor="middle" fill="{INK}">{i + 1}</text>')
    o.append(f'<line x1="{x0 - 6}" x2="{x0 + 4 * gw}" y1="{base}" y2="{base}" stroke="{INK}" stroke-width="3"/>')
    w = x0 + 4 * gw + 10 + F; h = base + 20 + F * 1.3 - top   # + F: room for a centred label over the last pair
    return f'<svg width="{w}" height="{h:.0f}" viewBox="0 {top:.0f} {w} {h:.0f}">{"".join(o)}</svg>'


def bars_top(p0, p1s, order, K, **kw):
    """the smallest SVG top over several panels, so they share one height"""
    import re
    return min(float(re.search(r'viewBox="0 (-?[\d.]+)', bars(p0, p1, order, K, **kw)).group(1)) for p1 in p1s)


def render(html_text, out_pngs, W, here, tag, scale=1, keep=False, H=3600, pad=20):
    hp = os.path.join(here, f"_{tag}.html"); open(hp, "w").write(html_text); raw = os.path.join(here, f"_{tag}_raw.png")
    subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
                    f"--force-device-scale-factor={scale}", "--virtual-time-budget=3000", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
    im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
    im = im.crop((0, 0, W * scale, nz[-1] + pad * scale))
    for o in out_pngs:
        if o.endswith(".jpg"): im.save(o, quality=92)
        else: im.save(o)
    if not keep: os.remove(raw); os.remove(hp)
    return im.size


def mtext(s):
    """pfmt text ('p < 10⁻⁵', 'p = 2×10⁻⁴') as matplotlib mathtext, so the exponent renders as a real superscript in Helvetica"""
    import re
    SUPR = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
    m = re.search(r"10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", s)
    if not m: return s
    e = m.group(1).translate(SUPR)
    return s[:m.start()] + "$10^{" + e.replace("-", "\u2212") + "}$" + s[m.end():]
