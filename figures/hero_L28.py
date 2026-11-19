"""Hero figure from the dose run (Qwen3-32B, 150 arithmetic contexts).
Two columns: 'Focus your attention on paragraph [Pk].' and 'Completely ignore paragraph [Pk]. Do not look at it at all.'
Layer 28 only. Violins are drawn on a log-ratio scale (log2 of attention ratio, hidden y axis); the big labels and p stay on the mean % change.
Method box (top right): answer tokens attending to the named paragraph, and the per-layer mean % change curve (as layers3.py, vs the same control) with layer 28 marked.
Each violin: per-context % change in attention on the named paragraph vs the mention-only line (control box, bottom left), at layer 28:
attention weights averaged over heads, summed over the paragraph's tokens and divided by its token count, averaged over the query
positions occupied by the forced answer tokens (dose.py "att_mean"; the last prompt token, which predicts the first answer token,
is not in this average; it is stored separately as "att_first"). The paragraph span starts at its second token (the first one
is merged with the space after the label). Method-box head arcs are a schematic: dose.json stores only the head mean.
p: two-sided sign-flip test over contexts on the mean % change. Example bubble text: dose.json id 300 (real prompt)."""
import json, base64, subprocess, os, html, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import PURPLE, RED, INK, pct_colour
from pfmt import pstr
D = os.path.join(os.path.dirname(HERE), "results", "qwen3-32b-dose") + "/"
meta = json.load(open(D + "meta.json")); L = [28]
R = json.load(open(D + "dose.json")); by = {(r["id"], r["cond"]): r for r in R}
ids = sorted({r["id"] for r in R})
def pk(r): return np.array(r["att_mean"])[L].mean(0)[r["k"]]
M = {c: np.array([pk(by[i, c]) for i in ids]) for c in meta["conds"]}
for i in ids:
    assert len({by[i, c]["k"] for c in meta["conds"]}) == 1
RES = json.load(open(D + "dose_results.json"))

def signflip(z, NF=200000, seed=0):
    rng = np.random.default_rng(seed); obs = abs(z.mean()); hits = 0
    for _ in range(NF // 20000):
        s = rng.choice([-1.0, 1.0], (20000, len(z))); hits += int((np.abs((s * z).mean(1)) >= obs).sum())
    return (hits + 1) / (NF + 1), NF

CONDS = [("3_focus", "focus your attention", PURPLE), ("9_completely_ignore", "completely ignore", RED)]
EX = 300
ex = {c: by[EX, c] for c, _, _ in CONDS}
ANS = {c: ex[c]["ans_free"].strip() for c in ex}
K = ex["3_focus"]["k"] + 1
u = ex["3_focus"]["user"]; paras = [p.split("] ", 1)[1] for p in u.split("\n\n")[1:5]]
qline = u.split("\n\n")[5].split("\n")[1]
LINE = {c: meta["conds"][c][0].format(k=K) for c, _, _ in CONDS}
for c in LINE: assert LINE[c] in ex[c]["user"] and qline in ex[c]["user"]

def short(p, n=3): return " ".join(p.split()[:n]) + ("\u00a0…" if NARROW else " …")   # narrow: the ellipsis never wraps alone
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
OPT = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
AMB = "#F2B53A"; W = int(OPT.get("W", 2300))   # --W=1600 --VH=15: narrower, taller version (hero_L28_tall) for hero_pair
NARROW = W < 2300; VHIN = float(OPT.get("VH", 11.5))   # violin panel height, inches at 110 dpi
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
numbers = {}

def curve(c, ctrl):   # per-layer mean % change over contexts, same measure as layers3.py
    PL = lambda cc: np.array([[by[i, cc]["att_mean"][l][by[i, cc]["k"]] for l in range(meta["NL"])] for i in ids])
    return (100 * (PL(c) / PL(ctrl) - 1)).mean(0)
L3 = json.load(open(os.path.join(HERE, "layers3_data.json")))
for c, _, _ in CONDS: assert np.allclose(curve(c, "1_mention"), L3[c]["mean"])

def method_svg(w, h, cf, ci, rob):
    """(a) answer tokens attending back to the named paragraph; (b) per-layer curve with layer 28 marked."""
    F = 40; o = []
    # ---- (a) top: token row (four paragraphs, then the answer), arcs from the answer to the named paragraph
    # real per-token attention from the answer (grid run, Qwen3-32B, focus on P2, mean over heads and answer tokens,
    # layers 21-53): each paragraph shown as 8 chunks, arrow width and darkness = that chunk's attention
    GG = json.load(open("/Users/user/mpd-data/runpod/hypha-prod-attngridv2a/out/G.json"))
    NCH = 8
    def chunks(r): return [[float(np.mean(c)) for c in np.array_split(np.array(r["vers"]["focus2"]["tok"][g]), NCH)] for g in range(4)]
    def p2share(r): v = chunks(r); return np.mean(v[1]) / np.mean([np.mean(v[g]) for g in (0, 2, 3)])
    gr = sorted(GG, key=p2share)[len(GG) // 2]   # a typical question: median P2 share
    vals = chunks(gr)
    vmax = max(max(v) for v in vals)
    ty = 150; sq = 15; gp = 3; gg = 8 if NARROW else 20; x = 12 if NARROW else 24; toks = []
    for g in range(4):
        xs = []
        for t in range(NCH):
            fill = AMB if g == 1 else "#C9CCD3"
            o.append(f'<rect x="{x:.1f}" y="{ty:.1f}" width="{sq}" height="{sq + 8}" rx="3" fill="{fill}"/>'); xs.append(x + sq / 2); x += sq + gp
        toks.append(xs)
        lab_col = INK if g == 1 else "#8A8F99"
        o.append(f'<text x="{(xs[0] + xs[-1]) / 2:.1f}" y="{ty + sq + 48:.1f}" font-size="{F - 4}" font-weight="700" fill="{lab_col}" text-anchor="middle">P{g+1}</text>')
        x += gg - gp
    bx = x + (6 if NARROW else 14); bw = 118; bh = 64; by_ = ty + sq / 2 - bh / 2
    o.append(f'<rect x="{bx:.1f}" y="{by_:.1f}" width="{bw}" height="{bh}" rx="22" fill="#fff" stroke="#BFC3CB" stroke-width="3"/>')
    o.append(f'<text x="{bx + bw / 2:.1f}" y="{by_ + bh / 2 + 15:.1f}" font-size="44" font-weight="800" fill="{INK}" text-anchor="middle">{ANS["3_focus"]}</text>')
    sx, sy = bx + bw / 2, by_ - 4
    for g in range(4):
        for t, tx in enumerate(toks[g]):
            v = (vals[g][t] / vmax) ** 0.5; ey = ty - 6; cy = ty - 40 - 70 * v
            o.append(f'<path d="M{sx:.1f},{sy:.1f} C{sx - 30:.1f},{cy:.1f} {tx + 20:.1f},{cy:.1f} {tx:.1f},{ey:.1f}" fill="none" stroke="{(INK if g == 1 else "#B9BDC5")}" stroke-opacity="{(0.35 + 0.65 * v) if g == 1 else (0.15 + 0.45 * v):.2f}" stroke-width="{0.8 + 7 * v:.1f}" stroke-linecap="round"/>')
    cx = (bx + bw + w) / 2 + 4; dy = 0
    if NARROW:   # no room right of the answer: one line under the paragraph labels
        dy = 64; o.append(f'<text x="{w / 2:.1f}" y="{ty + sq + 106:.1f}" font-size="{F + 6}" fill="{INK}" text-anchor="middle">averaged over heads and answer tokens</text>')
    else:
        AL = ["averaged over", "heads and", "answer tokens"]   # shrinks only as far as needed to stay inside the box
        hv = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 100); cx = (bx + bw + w) / 2
        AF = min(F + 6, int(100 * (w - bx - bw - 36) / max(hv.getlength(t) for t in AL)))
        for j, t in enumerate(AL):
            o.append(f'<text x="{cx:.1f}" y="{ty - 20 + 52 * j:.1f}" font-size="{AF}" fill="{INK}" text-anchor="middle">{t}</text>')
    o.append(f'<line x1="24" x2="{w - 24:.0f}" y1="{ty + 92 + dy}" y2="{ty + 92 + dy}" stroke="#E4E5E9" stroke-width="3"/>')
    # ---- (b) bottom: per-layer curve
    TF = F + 14; TY = F + 30   # y-axis % larger   # axis numbers and "layer": as large as the curve label
    x0 = 270; x1 = w - 40; y0 = ty + 165 + dy; y1 = h - 110
    lo = min(cf.min(), ci.min()) - 8; hi = max(cf.max(), ci.max())
    X = lambda l: x0 + (x1 - x0) * l / (len(cf) - 1); Y = lambda v: y1 - (y1 - y0) * (v - lo) / (hi - lo)
    o.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="{INK}" stroke-width="2.5"/>')
    for v in [0, 200]:
        o.append(f'<text x="{x0 - 12}" y="{Y(v) + 24:.1f}" font-size="{TY}" font-weight="700" fill="{INK}" text-anchor="end">{("+%d%%" % v) if v else "0%"}</text>')
    for v in [0, 63]:
        o.append(f'<text x="{X(v):.1f}" y="{y1 + 72:.1f}" font-size="{TF}" font-weight="700" fill="{INK}" text-anchor="middle">{v}</text>')
    o.append(f'<text x="{X(31.5):.1f}" y="{y1 + 72:.1f}" font-size="{TF}" font-weight="700" fill="{INK}" text-anchor="middle">layer</text>')
    o.append(f'<line x1="{X(28):.1f}" x2="{X(28):.1f}" y1="{y0 - 34}" y2="{y1 + 8}" stroke="{INK}" stroke-width="3" stroke-dasharray="8 6"/>')
    for cv, col in [(cf, PURPLE), (ci, RED)]:
        pts = " ".join(f"{X(l):.1f},{Y(v):.1f}" for l, v in enumerate(cv))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="5" stroke-linejoin="round"/>')
        o.append(f'<circle cx="{X(28):.1f}" cy="{Y(cv[28]):.1f}" r="11" fill="{col}" stroke="#fff" stroke-width="3"/>')
    LF = F + 12   # shrinks only if the label would run past the box's right edge
    LF = min(LF, int((w - 16 - X(28) - 22) / (0.56 * len("violins below:"))))
    for j, t in enumerate(["violins below:", "layer 28"]):
        o.append(f'<text x="{X(28) + 22:.1f}" y="{y0 - 6 + (LF + 8) * j:.1f}" font-size="{LF}" font-weight="700" fill="{INK}" text-anchor="start">{t}</text>')
    defs = f'<defs><marker id="ah" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>'
    return f'<svg width="{w:.0f}" height="{h:.0f}" viewBox="0 0 {w:.0f} {h:.0f}" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica Neue,Helvetica,Arial,sans-serif">{defs}{"".join(o)}</svg>'

def ctrl_svg(w):
    """each dot = attention under the instruction / attention under the mention line, same question (schematic bars)"""
    F = 46; h = 240; bw = 150; base = 165; o = []
    cx = w / 2; xs = [cx - 340, cx - 80]; hs = [140, 70]
    for x, hh, col, lab in zip(xs, hs, [INK, "#7C7C82"], ["instruction", "this line"]):
        o.append(f'<rect x="{x - bw / 2:.0f}" y="{base - hh}" width="{bw}" height="{hh}" rx="6" fill="{col}"/>')
        o.append(f'<text x="{x:.0f}" y="{base + 50}" font-size="{F}" fill="{INK}" text-anchor="middle">{lab}</text>')
    o.append(f'<text x="{(xs[0] + xs[1]) / 2:.0f}" y="{base - 28}" font-size="84" font-weight="700" fill="{INK}" text-anchor="middle">\u00f7</text>')
    o.append(f'<text x="{cx + 85:.0f}" y="{base - 28}" font-size="84" font-weight="700" fill="{INK}" text-anchor="middle">=</text>')
    dx = cx + 250; dy = base - 50
    o.append(f'<circle cx="{dx:.0f}" cy="{dy}" r="44" fill="#7A55C9" fill-opacity="0.45"/><circle cx="{dx:.0f}" cy="{dy}" r="5" fill="#000"/>')
    o.append(f'<text x="{dx:.0f}" y="{base + 50}" font-size="{F}" fill="{INK}" text-anchor="middle">one dot</text>')
    return f'<svg width="{w:.0f}" height="{h}" viewBox="0 0 {w:.0f} {h}" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica Neue,Helvetica,Arial,sans-serif">{"".join(o)}</svg>'

for ctrl, ctag, out in [("1_mention", "mention", "hero_L28_tall" if NARROW else "hero_L28")]:
    res = []
    for c, name, col in CONDS:
        z = 100 * (M[c] / M[ctrl] - 1)
        p, nf = signflip(z)   # test and printed effect: mean % change, as before
        res.append(dict(c=c, z=np.log2(M[c] / M[ctrl]), mean=z.mean(), p=p, nf=nf, above=float((z > 0).mean())))
        numbers[(ctag, c)] = (round(z.mean(), 1), p, nf, round(100 * (z > 0).mean()))
    # ---------- violins
    fig, a = plt.subplots(figsize=(23 * (W - 52) / 2248, VHIN)); fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    allz = np.concatenate([r["z"] for r in res])   # figure width scales with the page, so text keeps its size
    rng = np.random.default_rng(1); X = [0, 1.35]; texts = []
    for x0, r in zip(X, res):
        z = r["z"]; up = r["mean"] > 0; colr = PURPLE if up else RED
        vv = a.violinplot([z], positions=[x0], widths=0.7, showextrema=False)
        for bb in vv["bodies"]: bb.set_facecolor(colr); bb.set_alpha(0.22); bb.set_edgecolor(colr); bb.set_linewidth(2.5)
        xj = x0 + rng.uniform(-0.25, 0.25, len(z))
        a.scatter(xj, z, s=650, c=pct_colour(2.0 ** z), alpha=0.45, edgecolors="none", zorder=3); a.scatter(xj, z, s=6, color="black", zorder=4)
        eff = f"{r['mean']:+.0f}%{chr(10) if NARROW else ' '}attention".replace("-", "\u2212")
        pt = pstr(r["p"], floor=1 / (r["nf"] + 1))
        assert r["p"] * (r["nf"] + 1) >= 1.5 or 1 / (r["nf"] + 1) < 1e-5
        r["eff"], r["pt"] = eff, pt
        if up:   # effect above the dots, p under the effect
            texts.append(a.annotate(pt, (x0, z.max()), xytext=(0, 30), textcoords="offset points", fontsize=44, color=colr, ha="center", va="bottom"))
            texts.append(a.annotate(eff, (x0, z.max()), xytext=(0, 90), textcoords="offset points", fontsize=100, fontweight="bold", color=colr, ha="center", va="bottom"))
        else:    # effect below the dots, p under the effect
            texts.append(a.annotate(eff, (x0, z.min()), xytext=(0, -26), textcoords="offset points", fontsize=100, fontweight="bold", color=colr, ha="center", va="top"))
            texts.append(a.annotate(pt, (x0, z.min()), xytext=(0, -138 - 100 * eff.count(chr(10))), textcoords="offset points", fontsize=44, color=colr, ha="center", va="top"))
    a.axhline(0, color=INK, lw=4, zorder=0.5)
    a.set_xlim(-0.66, 2.01); a.set_xticks([]); a.set_yticks([])
    for s_ in a.spines.values(): s_.set_visible(False)
    lo, hi = min(allz.min(), 0), max(allz.max(), 0); a.set_ylim(lo, hi)
    for _ in range(6):   # grow y-limits until every label fits
        fig.canvas.draw(); rd = fig.canvas.get_renderer(); inv = a.transData.inverted()
        ys = [inv.transform(t.get_window_extent(rd).get_points())[:, 1] for t in texts]
        ylo = min(lo, min(y.min() for y in ys)); yhi = max(hi, max(y.max() for y in ys))
        pad = 0.01 * (yhi - ylo); a.set_ylim(ylo - pad, yhi + pad)
    fig.canvas.draw(); FW, FH = fig.canvas.get_width_height()
    ri = res[1]["z"].max(); ytop = a.transData.transform((0, ri))[1] + np.sqrt(650) / 2 * 110 / 72   # top of the highest ignore dot, px from bottom
    bxl = a.transData.transform((0.72, 0))[0]
    BOX = dict(left=bxl / FW, right=0.995, top=0.012, bottom=1 - (ytop + 30) / FH)
    # control box: bottom-left whitespace, below the axis and the focus dots, left of the ignore violin and its labels
    rd = fig.canvas.get_renderer()
    yax = a.transData.transform((0, 0))[1]; yfd = a.transData.transform((0, res[0]["z"].min()))[1] - np.sqrt(650) / 2 * 110 / 72
    xr = min([a.transData.transform((X[1] - 0.36, 0))[0]] + [t.get_window_extent(rd).x0 for t in texts[2:]])
    CB = dict(left=8, right=xr - 40, top=FH - min(yax, yfd) + 34, bottom=FH - 6)   # fig px from top-left
    print("control box fig px", {k: round(v) for k, v in CB.items()})
    fig.savefig(os.path.join(HERE, f"v{out}_dose.png"), dpi=110, facecolor="white"); plt.close(fig)
    # ---------- chat columns + violins
    VW = W - 52; sc = VW / FW; SVGW = (VW - 36) / 2 - 6; SVGH = (BOX["bottom"] - BOX["top"]) * FH * sc
    print(out, "box px", round(SVGW), round(SVGH))
    SVG = method_svg(SVGW, SVGH, curve("3_focus", ctrl), curve("9_completely_ignore", ctrl), robot)
    VIO = base64.b64encode(open(os.path.join(HERE, f"v{out}_dose.png"), "rb").read()).decode()
    cols = []
    for c, name, col in CONDS:
        pl = "".join(f'<div class="pl"><b>[P{i+1}]</b> <span class="{"k" if i + 1 == K else ""}">{html.escape(short(p, 2 if i == 0 else 3))}</span></div>' for i, p in enumerate(paras))   # highlight the paragraph text: attention is measured on its body tokens, not the label
        cols.append(f'''<div class="col"><div class="cname" style="color:{col}">{name}</div>
  <div class="ub"><div>Here are four paragraphs.</div>{pl}
    <div class="var" style="color:{col}">{html.escape(LINE[c])}</div>
    <div>{html.escape(qline).replace(" + ", "&nbsp;+&nbsp;")}</div></div>
  <div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb">{ANS[c]}</div></div></div>''')
    WS = "normal" if NARROW else "nowrap"
    HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:20px 26px 10px}}
.cols{{display:flex;gap:36px}}
.col{{flex:1;min-width:0;display:flex;flex-direction:column}}
.cname{{font-size:76px;font-weight:800;margin:0 0 12px 4px;white-space:{WS}{";line-height:1.05" if NARROW else ""}}}
.ub{{flex:1;background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 26px;font-size:54px;line-height:1.18;display:flex;flex-direction:column;gap:8px}}
.pl{{white-space:{WS};overflow:hidden}}
.var{{background:#fff;border-radius:16px;padding:10px 16px;font-weight:700;line-height:1.18}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
.mrow{{display:flex;align-items:center;gap:16px;margin-top:12px}}
.rob{{width:110px;height:110px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:12px 34px;font-size:70px;font-weight:700;box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.vw{{position:relative;margin-top:10px}}
.vio{{width:100%;display:block}}
.cbox{{position:absolute;background:#fff;border:4px solid #000;border-radius:28px;padding:22px 26px;display:flex;flex-direction:column;justify-content:space-evenly}}
.cub{{background:#1F8BFF;border-radius:26px;padding:14px 16px}}
.cl{{display:flex;flex-direction:column;gap:2px}} .cl1.big{{font-size:61px}} .cl1{{font-size:46px;font-weight:800;display:flex;align-items:center;gap:14px;white-space:{WS};line-height:1.15}} .cl2{{font-size:44px;font-weight:700;line-height:1.15}}
.dot{{display:inline-block;flex-shrink:0;width:46px;height:46px;border-radius:50%;background:{PURPLE};opacity:.75}}
.cl1.nw{{display:block;white-space:normal}} .cl1.nw .dot{{vertical-align:middle;margin:0 6px}}
.cub .var{{color:{INK};font-size:46px;white-space:{WS}}}
.mbox{{position:absolute;background:#fff;border:4px solid #000;border-radius:28px;overflow:hidden}}
</style></head><body><div class="cols">{"".join(cols)}</div><div class="vw"><img class="vio" src="data:image/png;base64,{VIO}">
  <div class="cbox" style="left:{CB['left']*sc:.0f}px;width:{(CB['right']-CB['left'])*sc:.0f}px;top:{CB['top']*sc:.0f}px;height:{(CB['bottom']-CB['top'])*sc:.0f}px">
    <div class="cl"><div class="cl1 big{" nw" if NARROW else ""}">one dot <span class="dot"></span> = one prompt</div><div class="cl1 nw">% change in attention to the mentioned paragraph, compared to:</div></div>
    <div class="cub"><div class="var">{html.escape(meta["conds"][ctrl][0].format(k=K))}</div></div></div>
  <div class="mbox" style="left:{(VW + 36) / 2:.0f}px;width:{(VW - 36) / 2:.0f}px;top:{BOX['top']*FH*sc:.0f}px;height:{SVGH:.0f}px">{SVG}</div></div></body></html>'''
    hp = os.path.join(HERE, f"{out}_dose.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, f"{out}_dose_raw.png")
    subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3200",
                    "--force-device-scale-factor=1", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
    im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
    im = im.crop((0, 0, W, nz[-1] + 20)); im.save(os.path.join(HERE, out + ".png")); im.save(os.path.join(HERE, out + ".jpg"), quality=92); print(out, im.size)
for k, v in numbers.items(): print(k, "mean %+.1f%%  p_signflip=%.2g (of %d flips)  share>0 %d%%" % v)
