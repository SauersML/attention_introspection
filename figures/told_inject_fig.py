"""Told-injection figure (Qwen3-32B, 240 four-paragraph prompts). Top: the real told prompt of context 0 (example_prompt_0.txt), excerpted,
with paragraph 3 boosted (amber); the real question; Qwen's real greedy reply under the boost ("No") and Qwen's real probabilities for the
paragraph digit after a prefilled "Yes, paragraph [P" (context 0, boost onto P3, nominal x20). Syringe dose = mean measured attention-mass
ratio at x20 (scores.json). Bottom left: greedy "Yes" rate on boosted trials minus the same prompt's control trial, by dose, 95% bootstrap CI
over prompts. Bottom right: rotation specificity, exp(mean change in log P(boosted digit) minus mean change for the other three) as a
multiplier, CI over prompts, log axis. Doses labelled by measured attention multiplier. usage: nice -n 19 python3 told_inject_fig.py"""
import json, math, os, sys, html, base64, subprocess, io
import numpy as np
from PIL import Image, ImageFilter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
INK = "#1D1D1F"; PURPLE = "#5E2B97"; LIL = "#A574F2"; AMB = "#F2B53A"; COL = "#E07B24"
S = json.load(open(os.path.join(ROOT, "results/told_inject/scores.json")))
DD = ["2", "5", "10", "20"]
rec = json.loads(open(os.path.join(ROOT, "results/told_inject/data/recs_0.jsonl")).readline()); assert rec["id"] == 0
user = open(os.path.join(ROOT, "results/told_inject/data/example_prompt_0.txt")).read()
segs = [s.split("] ", 1)[1] for s in user.split("\n\n") if s.startswith("[P")]; assert len(segs) == 4
K = 2; T = rec["boost"][f"{K}|20.0"]; prob = np.exp(T["lp_par"]); reply = T["text"]
print("example reply", reply, np.round(prob, 3), "control", np.round(np.exp(rec["none"]["lp_par"]), 3))
mult = {d: S["dose"][d]["attn_mass_ratio"][0] for d in DD}
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
W = 1500
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from robot_injected import injected_robot
robot_inj = injected_robot("/Users/user/watching/figures/assets/robot.png")
def syringe(dose):
    o = [f'<path d="M380,80 L364,76 L364,84 Z" fill="{INK}"/>', f'<rect x="314" y="76" width="50" height="8" fill="{INK}"/>',
         f'<path d="M314,68 L292,58 L292,102 L314,92 Z" fill="{INK}"/>',
         f'<rect x="80" y="22" width="212" height="116" rx="14" fill="#fff" stroke="{INK}" stroke-width="7"/>',
         f'<rect x="109" y="29" width="176" height="102" fill="{LIL}"/>', f'<rect x="87" y="29" width="22" height="102" fill="{INK}"/>',
         f'<rect x="62" y="4" width="18" height="152" rx="7" fill="{INK}"/>', f'<rect x="26" y="71" width="36" height="18" fill="{INK}"/>',
         f'<rect x="2" y="40" width="24" height="80" rx="7" fill="{INK}"/>',
         f'<text x="197" y="108" font-size="80" font-weight="800" text-anchor="middle" fill="#fff">\u00d7{dose}</text>']
    return f'<svg id="syr" width="380" height="160" viewBox="0 0 380 160">{"".join(o)}</svg>'
def pbars(pr):
    w, h, bw = 300, 150, 52; o = []
    for i in range(4):
        x = 14 + i * 72; hh = 100 * pr[i]; c = AMB if i == K else INK
        o.append(f'<rect x="{x}" y="{112 - hh:.1f}" width="{bw}" height="{max(hh, 3):.1f}" rx="5" fill="{c}"/>')
        o.append(f'<text x="{x + bw / 2}" y="148" font-size="40" font-weight="700" text-anchor="middle" fill="{INK}">{i + 1}</text>')
        if i == K: o.append(f'<text x="{x + bw / 2}" y="{104 - hh:.1f}" font-size="40" font-weight="700" text-anchor="middle" fill="{INK}">{100 * pr[i]:.0f}%</text>')
    o.append(f'<line x1="6" x2="{w - 6}" y1="112" y2="112" stroke="{INK}" stroke-width="3"/>')
    return f'<svg width="{w}" height="{h}" viewBox="0 -10 {w} {h + 10}">{"".join(o)}</svg>'
neg = lambda s: s.replace("-", "\u2212")
def chart(vals, title, ytick, yfmt, logy=False, plab=None):
    """bars with 95% CI whiskers; vals: list of (mean, lo, hi) in plotted units"""
    w, h = 700, 560; L, B, Tp = 150, 450, 40; bw = 104; o = []
    f = (lambda v: math.log(v)) if logy else (lambda v: v)
    lo, hi = f(ytick[0]), f(ytick[-1]); Y = lambda v: B - (f(v) - lo) / (hi - lo) * (B - Tp)
    base = 1.0 if logy else 0.0
    for t in ytick:
        o.append(f'<text x="{L - 14}" y="{Y(t) + 15:.1f}" font-size="44" text-anchor="end" fill="{INK}">{neg(yfmt(t))}</text>')
    o.append(f'<line x1="{L}" x2="{L}" y1="{Y(ytick[0]):.1f}" y2="{Y(ytick[-1]):.1f}" stroke="{INK}" stroke-width="3"/>')
    for i, (m, a, b) in enumerate(vals):
        x = L + 40 + i * 132; y0, y1 = Y(base), Y(m)
        o.append(f'<rect x="{x}" y="{min(y0, y1):.1f}" width="{bw}" height="{abs(y1 - y0):.1f}" fill="{COL}"/>')
        cx = x + bw / 2
        o.append(f'<line x1="{cx}" x2="{cx}" y1="{Y(a):.1f}" y2="{Y(b):.1f}" stroke="{INK}" stroke-width="5"/>')
        for yy in (Y(a), Y(b)): o.append(f'<line x1="{cx - 18}" x2="{cx + 18}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{INK}" stroke-width="5"/>')
        o.append(f'<text x="{cx}" y="{B + 58}" font-size="44" font-weight="700" text-anchor="middle" fill="{PURPLE}">\u00d7{mult[DD[i]]:.1f}</text>')
    o.append(f'<line x1="{L}" x2="{w - 4}" y1="{Y(base):.1f}" y2="{Y(base):.1f}" stroke="{INK}" stroke-width="4"/>')
    if plab: o.append(plab(Y, L))
    return f'<div class="ch"><div class="ct">{title}</div><svg width="{w}" height="{h}">{"".join(o)}</svg></div>'
d1 = [[100 * x for x in S["dose"][d]["greedy_yes_minus_control"][:3]] for d in DD]
d2 = [[math.exp(x) for x in S["dose"][d]["specificity_logp"][:3]] for d in DD]
p1 = S["dose"]["20"]["greedy_signflip_p"]; p2 = max(S["dose"][d]["specificity_p"] for d in DD)
pt = lambda p: "p < 0.001" if p < 1e-3 else f"p = {p:.2g}"
c1 = chart(d1, "says \u201cYes\u201d more often than with no boost", [-5, 0, 5, 10, 15], lambda t: f"{t:+d}%" if t else "0%",
           plab=lambda Y, L: f'<text x="696" y="{Y(d1[3][2]) - 16:.1f}" font-size="44" text-anchor="end" fill="{INK}">{pt(p1)}</text>'
                              f'<text x="{L + 24}" y="{Y(0) + 46:.1f}" font-size="44" fill="{INK}"></text>')
c2 = chart(d2, "after \u201cYes, paragraph [P\u201d the boosted paragraph gains", [1, 2, 4, 8, 12], lambda t: f"\u00d7{t}", logy=True,
           plab=lambda Y, L: f'<text x="{L + 40}" y="{Y(10):.1f}" font-size="44" fill="{INK}">{pt(p2)} each</text>')
short = lambda s, n: html.escape(" ".join(s.split()[:n])) + " \u2026"
pl = "".join(f'<div class="pl"><b {"id=pc" if i == K else ""}>[P{i + 1}]</b> <span {"class=k id=pt" if i == K else ""}>{short(segs[i], 4)}</span></div>' for i in range(4))
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:22px 26px 0;position:relative}}
.ub{{background:#1F8BFF;color:#fff;border-radius:32px;padding:16px 26px;font-size:44px;line-height:1.2}}
#u1{{margin:0 130px 0 40px}} .pl{{white-space:nowrap}} .k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
#rq{{margin:22px 130px 0 600px}}
.rows{{display:flex;flex-direction:column;gap:18px;margin:80px 0 0 360px}}
.mrow{{display:flex;align-items:center;gap:14px}}
.rob{{width:120px;height:120px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:8px 30px;font-size:60px;font-weight:700}}
.bb{{display:flex;align-items:flex-end;gap:4px;padding:6px 14px 8px}} .pp{{font-size:46px;font-weight:700;margin-bottom:40px;white-space:nowrap}}
#syr{{position:absolute}} .inj{{position:absolute;left:30px;font-size:50px;font-weight:800;line-height:1.12;color:{PURPLE}}}
#gl{{position:absolute;left:0;top:0;z-index:-1;pointer-events:none}} #ov{{position:absolute;left:0;top:0;pointer-events:none}}
.charts{{display:flex;justify-content:space-between;margin-top:40px}} .ch{{width:700px}} .ct{{font-size:46px;font-weight:800;line-height:1.12;padding-left:20px;height:112px}}
.xl{{font-size:46px;font-weight:800;color:{PURPLE};text-align:center;margin-top:4px}}
</style></head><body>
<div class="ub" id="u1">\u2026 I am capable of artificially boosting your attention onto one of the four paragraphs below \u2026 I will boost your attention onto one of the paragraphs on 50% of trials \u2026{pl}</div>
<div class="ub" id="rq">Do you notice your attention being boosted onto one of the paragraphs? If so, which one? \u2026</div>
<div class="rows"><div class="mrow"><img class="rob" id="rob1" src="data:image/png;base64,{robot_inj}"><div class="mb">{html.escape(reply)}</div></div>
<div class="mrow"><img class="rob" id="rob2" src="data:image/png;base64,{robot_inj}"><div class="mb bb"><span class="pp">Yes, paragraph [P</span>{pbars(prob)}</div></div></div>
{syringe(f"{mult['20']:.1f}")}<div class="inj" id="inj">inject attention<br>into paragraph 3</div>
<div class="charts">{c1}{c2}</div>
<div class="xl">attention to the boosted paragraph</div>
<svg id="gl"></svg><svg id="ov"></svg>
<script>
const R=e=>document.getElementById(e).getBoundingClientRect(); const SW=document.body.scrollWidth, SH=document.body.scrollHeight;
const ov=document.getElementById('ov');ov.setAttribute('width',SW);ov.setAttribute('height',SH);
const rb=R('rob1'), syr=document.getElementById('syr');
const tipx=rb.left+rb.width*0.29, tipy=rb.top+rb.height*0.72;
syr.style.left=(tipx-380)+'px'; syr.style.top=(tipy-80)+'px'; syr.style.transformOrigin='380px 80px'; syr.style.transform='rotate(22deg)';
const inj=document.getElementById('inj'); inj.style.top=(R('rq').top+4)+'px';
const gl=document.getElementById('gl'); gl.setAttribute('width',SW); gl.setAttribute('height',SH);
let g=`<defs>`;
[[0.004,0.010,36,7,1],[0.006,0.014,28,4.5,2],[0.008,0.02,18,2.5,3]].forEach(([fx,fy,sc,bl,i])=>{{
 g+=`<filter id="w${{i}}" x="-30%" y="-30%" width="160%" height="160%"><feTurbulence type="turbulence" baseFrequency="${{fx}} ${{fy}}" numOctaves="2" seed="${{5*i+1}}" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="${{sc}}" xChannelSelector="R" yChannelSelector="G" result="d"/><feGaussianBlur in="d" stdDeviation="${{bl}}"/></filter>`;}});
g+=`<filter id="haze" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="22"/></filter></defs>`;
const aura=(r,rad)=>{{ const S=(d,st,col,op,f,fill)=>`<rect x="${{r.left-d}}" y="${{r.top-d}}" width="${{r.width+2*d}}" height="${{r.height+2*d}}" rx="${{rad+d}}" fill="${{fill?col:'none'}}" stroke="${{fill?'none':col}}" stroke-width="${{st}}" opacity="${{op}}" filter="url(#${{f}})"/>`;
 return S(8,0,'#B994FF',0.75,'haze',1)+S(40,5,'#C9A8FF',0.45,'w1')+S(26,7,'{LIL}',0.6,'w2')+S(12,9,'#7B3FE4',0.75,'w3')+S(2,8,'{PURPLE}',0.55,'w3'); }};
g+=aura(R('rq'),32); ['rob1','rob2'].forEach(id=>{{const ro=R(id); g+=aura({{left:ro.left+6,top:ro.top+10,width:ro.width-12,height:ro.height-16}},54);}});
gl.innerHTML=g;
const t=R('pt'), q=R('rq'), c=R('pc'); const yP=(c.top+c.bottom)/2, xq=q.right+46, yq=(q.top+q.bottom)/2;
let s=`<defs><marker id="mT" markerUnits="userSpaceOnUse" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="38" markerHeight="38" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{PURPLE}"/></marker></defs>`;
s+=`<path d="M${{q.right+2}},${{yq}} L${{xq-26}},${{yq}} Q${{xq}},${{yq}} ${{xq}},${{yq-26}} L${{xq}},${{yP+26}} Q${{xq}},${{yP}} ${{xq-26}},${{yP}} L${{t.right+12}},${{yP}}" fill="none" stroke="{PURPLE}" stroke-width="16" marker-end="url(#mT)"/>`;
ov.innerHTML=s;
</script></body></html>'''
hp = os.path.join(HERE, "told_inject.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, "told_inject_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},2600",
                "--force-device-scale-factor=2", "--virtual-time-budget=3000", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, 2 * W, nz[-1] + 40)); out = os.path.join(ROOT, "results/told_inject/told.png"); im.save(out); print(out, im.size, im.size[1] / im.size[0])
