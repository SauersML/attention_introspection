"""Salience vs attention (Qwen3-32B, 240 contexts; results/salience, numbers from salience/score.py's scores.json, which this script reads).
Top: the told prompt shortened (blue bubble) and two method drawings on the same token row ([P1]..[P4] cells, [P2] amber, then the question and
answer cells) inside the band of layers 21-53. Left: attention boost x20 from the question and answer tokens onto paragraph 2 (purple arrows).
Right: paragraph 2's content direction (mean residual over paragraph 2 minus mean over all four paragraphs) added into the residual stream at the
question and answer tokens, sized so the residual change per layer matches the attention boost's (green, plus 2x).
Bottom, two panels, one bar per arm: relative change of the mean over contexts from no boost (dashed 0 line), whiskers = the scorer's 95%
bootstrap CI (10,000 resamples), big relative % above with the raw rates under it, p bracket attention boost vs same-size content addition
(two-sided sign-flip over contexts, 20,000 flips). LEFT: greedy reply starts with "Yes". RIGHT: top digit after the prefill "Yes, paragraph [P"
is the boosted paragraph.
usage: OMP_NUM_THREADS=2 nice -n 19 python3 figures/salience.py [--keep]   (writes figures/salience.png and .jpg)"""
import json, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE
from pfmt import pstr
import injb_draw as dr
ROOT = os.path.dirname(HERE)
S = json.load(open(os.path.join(ROOT, "results/salience/scores.json"))); META = json.load(open(os.path.join(ROOT, "results/salience/data/meta.json")))
assert S["n"] == 240 and META["BAND"][0] == 21 and META["BAND"][-1] == 53 and META["M"] == 20.0
GREEN = "#1A9E77"; NF = 20000
ARMS = [("A", PURPLE, 0.75, "attention\nboost"), ("B", GREEN, 0.75, "content\nadded,\nsame size"), ("B2", GREEN, 0.4, "content\nadded,\n2\u00d7")]
READS = ["greedy_yes", "top_is_k"]
W = 3000; PAD = 26; VW = W - 2 * PAD; GAP = 70; PW = (VW - GAP) / 2; LM = 330; VH = 1500; TOPH = 40; BOT = 300
CW = (PW - LM) / 3; ylo, yhi = -0.12, 2.05
rl = lambda v: f"{100 * v:+.0f}%".replace("-", "\u2212")

plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(VW / 100, VH / 100), dpi=100)
ax = fig.add_axes([0, BOT / VH, 1, (VH - TOPH - BOT) / VH]); ax.set_xlim(0, VW); ax.set_ylim(ylo, yhi)
for pi, rd in enumerate(READS):
    o = S["reads"][rd]; x0 = pi * (PW + GAP); base = o["none"][0]
    ax.plot([x0 + LM - 20, x0 + PW], [0, 0], color=INK, lw=4, ls=(0, (4, 3)), zorder=2)
    ax.text(x0 + LM - 30, 0, "0% =\nno boost", ha="right", va="center", fontsize=dr.px(60), fontweight="bold", color=INK, linespacing=1.0)
    for t in (0.5, 1.0, 1.5): ax.text(x0 + LM - 30, t, f"+{100 * t:.0f}%", ha="right", va="center", fontsize=dr.px(68), fontweight="bold", color=INK)
    ax.text(x0 + 30, 0.85, "change from no boost", rotation=90, ha="center", va="center", fontsize=dr.px(62), fontweight="bold", color=INK)
    tops = {}
    for i, (a, col, al, lab) in enumerate(ARMS):
        xc = x0 + LM + (i + 0.5) * CW; hw = 0.36 * CW
        mu, lo, hi = (v / 100 for v in o[a]["rel_vs_none"]); rate = o[a]["rate"][0]
        ax.bar(xc, mu, width=2 * hw, color=col, alpha=al, lw=0, zorder=3)
        ax.plot([xc, xc], [lo, hi], color=INK, lw=6, zorder=5, solid_capstyle="butt")
        for yy in (lo, hi): ax.plot([xc - 0.25 * hw, xc + 0.25 * hw], [yy, yy], color=INK, lw=6, zorder=5, solid_capstyle="butt")
        ax.text(xc, hi + 0.04, f"{100 * base:.0f}% \u2192 {100 * rate:.0f}%", ha="center", va="bottom", fontsize=dr.px(48), fontweight="bold", color=INK, family=["Helvetica", "Arial Unicode MS"])
        ax.text(xc, hi + 0.13, rl(mu), ha="center", va="bottom", fontsize=dr.px(84), fontweight="bold", color=col)
        tops[a] = hi + 0.13 + 0.17
        fig.text((xc) / VW, (BOT - 18) / VH, lab, ha="center", va="top", fontsize=dr.px(56), fontweight="bold", color=col, linespacing=1.0)
    # p bracket: attention boost vs same-size content addition
    yb = max(tops["A"], tops["B"]) + 0.04; xa = x0 + LM + 0.5 * CW; xb = x0 + LM + 1.5 * CW
    ax.plot([xa, xa, xb, xb], [yb - 0.04, yb, yb, yb - 0.04], color=INK, lw=4.5, zorder=5)
    ax.text((xa + xb) / 2, yb + 0.02, pstr(o["A_minus_B"]["p"], floor=1 / (NF + 1)), ha="center", va="bottom", fontsize=dr.px(60), color=INK)
ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_visible(False)
VP = os.path.join(HERE, "_v_salience.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)
print({rd: {a: S["reads"][rd][a]["rel_vs_none"] for a, *_ in ARMS} for rd in READS})


def stream(kind):
    """one method drawing: the token row ([P1]..[P4], question + answer) under the band of layers 21-53; kind 'att' or 'add'"""
    w, h = 1440, 630; o = []; cw, cg = 44, 6
    groups = [("P1", 4), ("P2", 4), ("P3", 4), ("P4", 4), ("Q", 7)]
    x = 40; cells = {}
    for g, n in groups:
        if g == "Q": x += 40
        cells[g] = [x + j * (cw + cg) for j in range(n)]; x += n * (cw + cg) + 18
    by0, by1, ty = 150, 400, 420
    o.append(f'<rect x="20" y="{by0}" width="{w - 40}" height="{by1 - by0}" rx="26" fill="#F4F4F6" stroke="{INK}" stroke-width="4"/>')
    o.append(f'<text x="44" y="{by0 + 66}" font-size="54" font-weight="800" fill="{INK}">Qwen3-32B, layers 21\u201353</text>')
    for g, xs in cells.items():
        for cx in xs:
            fill = dr.AMB if g == "P2" else ("#1F8BFF" if g == "Q" else "#9DC9FF")
            o.append(f'<rect x="{cx}" y="{ty}" width="{cw}" height="60" rx="8" fill="{fill}"/>')
            o.append(f'<line x1="{cx + cw / 2}" y1="{ty}" x2="{cx + cw / 2}" y2="{by0 + 100 if g != "Q" else by0 + 4}" stroke="{INK}" stroke-opacity=".18" stroke-width="3"/>')
        lab = "question + answer" if g == "Q" else f"[{g}]"
        o.append(f'<text x="{(xs[0] + xs[-1] + cw) / 2}" y="{ty + 120}" font-size="52" font-weight="800" text-anchor="middle" fill="{INK}">{lab}</text>')
    p2 = (cells["P2"][0] + cells["P2"][-1] + cw) / 2
    if kind == "att":
        o.append(f'<defs><marker id="ha" markerWidth="5" markerHeight="5" refX="2.5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="{PURPLE}"/></marker></defs>')
        for j, qx in enumerate(cells["Q"][::2]):
            qx += cw / 2; y0 = by1 - 40 - 22 * j
            o.append(f'<path d="M{qx},{y0} C{qx},{by0 + 75} {p2 + 60},{by0 + 75} {p2 + 8 * j - 12},{by1 - 30}" fill="none" stroke="{PURPLE}" stroke-width="12" stroke-opacity=".8" marker-end="url(#ha)"/>')
        o.append(f'<text x="{(p2 + cells["Q"][0]) / 2 + 30}" y="{by0 - 26}" font-size="64" font-weight="800" text-anchor="middle" fill="{PURPLE}">attention \u00d720</text>')
    else:
        o.append(f'<defs><marker id="hg" markerWidth="5" markerHeight="5" refX="2.5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="{GREEN}"/></marker></defs>')
        qs = [c + cw / 2 for c in cells["Q"]]; yp = by0 + 185
        o.append(f'<path d="M{p2},{ty - 6} C{p2},{by0 + 85} {qs[0] - 120},{by0 + 85} {qs[0] - 46},{yp - 10}" fill="none" stroke="{GREEN}" stroke-width="14" marker-end="url(#hg)"/>')
        for qx in qs:
            o.append(f'<circle cx="{qx}" cy="{yp}" r="21" fill="#fff" stroke="{GREEN}" stroke-width="6"/><path d="M{qx - 11},{yp} H{qx + 11} M{qx},{yp - 11} V{yp + 11}" stroke="{GREEN}" stroke-width="6"/>')
        o.append(f'<text x="{(p2 + qs[-1]) / 2 + 20}" y="{by0 - 26}" font-size="64" font-weight="800" text-anchor="middle" fill="{GREEN}">+ paragraph 2\u2019s content, same size</text>')
    return f'<svg width="{w}" height="{h - 66}" viewBox="0 64 {w} {h - 66}">{"".join(o)}</svg>'


import base64
VIO = base64.b64encode(open(VP, "rb").read()).decode()
INTRO = META["intro"]; S2 = "I am capable of artificially boosting your attention onto one of the four paragraphs below"; assert S2 in INTRO
assert META["question"].startswith("Trial 1: Do you notice your attention being boosted onto one of the paragraphs? If so, which one?")
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:20px {PAD}px 10px}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 30px;font-size:58px;line-height:1.18}}
.row{{display:flex;justify-content:space-between;margin-top:30px}}
.ct{{font-size:66px;font-weight:800;text-align:center;line-height:1.08}}
.cols{{display:flex;margin-top:10px}}
.cn{{font-size:72px;font-weight:800;line-height:1.08;text-align:center}}
img{{display:block;width:{VW}px}}
</style></head><body>
<div class="ub">{S2} \u2026 Here are four paragraphs. \u2026 Do you notice your attention being boosted onto one of the paragraphs? If so, which one?</div>
<div class="row"><div>{stream("att")}</div><div>{stream("add")}</div></div>
<div class="cols"><div class="cn" style="width:{PW - 40:.0f}px;margin-left:40px">how often Qwen\u2019s reply<br>starts with \u201cYes\u201d</div>
<div class="cn" style="width:{PW - 40:.0f}px;margin-left:{GAP + 40:.0f}px">how often Qwen\u2019s top answer is<br>the boosted paragraph\u2019s number</div></div>
<img src="data:image/png;base64,{VIO}">
</body></html>'''
dr.render(HTML, HERE, "salience", W, jpg=True, keep="--keep" in sys.argv)
if "--keep" not in sys.argv: os.remove(VP)
