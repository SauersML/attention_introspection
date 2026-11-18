# Hero: six heads in layers 20-26 read the paragraph number; switching them off halves the focus boost,
# copying them from a prompt naming another paragraph moves attention there.
import json
from figlib import *
S = json.load(open(os.path.join(HERE, "hero_stats.json")))
W, H = 2300, 2000
F = 46
b = []
# ---- chat bubble (HTML inside the SVG, so Chrome lays out the text)
bx, by, bw, bh = 30, 30, 1330, 505
AM = f'background:{AMB};color:{INK};border-radius:8px;padding:0 6px'
para = [("[P1]", "Similar alloys …"), ("[P2]", "Comparing diets among …"), ("[P3]", "China's network security …"), ("[P4]", "DC commutating electric …")]
rows = "".join(f'<div><b style="{AM if l=="[P2]" else ""}">{l}</b> {t}</div>' for l, t in para)
htm = (f'<div xmlns="http://www.w3.org/1999/xhtml" style="font:{F}px {FONT};color:#fff;line-height:66px;padding:22px 30px">'
       f'<div>Here are four paragraphs.</div>{rows}'
       f'<div style="background:#fff;border-radius:14px;color:{PURPLE};font-weight:700;padding:4px 16px;margin:10px -6px">'
       f'Focus your attention on paragraph [P<span style="{AM};padding:0 4px">2</span>].</div>'
       f'<div>Unrelated question: what is 83 + 80?</div></div>')
b.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="40" fill="{UB}"/>')
b.append(f'<foreignObject x="{bx}" y="{by}" width="{bw}" height="{bh}">{htm}</foreignObject>')
ys = {"[P2]": by + 22 + 66 * 2 + 33, "digit": by + 22 + 66 * 5 + 10 + 40}
ay = by + bh + 24
c, cw = chip(bx + 30, ay, "163", 56, "#fff", INK, "#D8D8DE", 700, 22, 84, 20); b.append(c)
acx, acy = bx + 30 + cw, ay + 42
# ---- routes: answer chip -> right of bubble -> up -> into bubble at target line
def route(xr, yt, col, w=7):
    p = f'M {acx+8} {acy} L {xr-30} {acy} Q {xr} {acy} {xr} {acy-30} L {xr} {yt+30} Q {xr} {yt} {xr-30} {yt} L {bx+bw+22} {yt}'
    return (f'<path d="{p}" fill="none" stroke="{col}" stroke-width="{w}" stroke-linecap="round"/>'
            f'<path d="M {bx+bw+6} {yt} l 26 -16 l 0 32 z" fill="{col}"/>')
b.append(route(1420, ys["digit"], PURPLE))
b.append(route(1470, ys["[P2]"], INK))
lx = 1510
b.append(text(lx, ys["digit"] - 40, "6 heads in layers 20–26", F, PURPLE, 700))
b.append(text(lx, ys["digit"] + 16, "look at the number 2", F, PURPLE, 700))
b.append(text(lx, ys["[P2]"] - 40, "then heads in later", F, INK, 700))
b.append(text(lx, ys["[P2]"] + 16, "layers look more at P2", F, INK, 700))
b.append(text(lx, ys["[P2]"] + 72, "(+45%, layers 21–53)", F, INK))
b.append(text(lx, acy + 16, "Qwen3-32B", F, INK, 700))
# ---- bar panels
def panel(x0, y0, pw, ph, title, ylab, bars, ymax, ticks, fmt, p):
    o = []
    for i, t in enumerate(title): o.append(text(x0, y0 + 50 + i * 58, t, F, INK, 700))
    top = y0 + 50 + len(title) * 58 + 70; bot = y0 + ph - 150; ax = x0 + 190
    sc = (bot - top) / ymax
    o.append(f'<line x1="{ax}" y1="{bot}" x2="{x0+pw}" y2="{bot}" stroke="{INK}" stroke-width="3"/>')
    for t in ticks:
        yy = bot - t * sc; o.append(text(ax - 16, yy + 15, fmt(t), F, INK, 400, "end"))
        o.append(f'<line x1="{ax-8}" y1="{yy}" x2="{ax}" y2="{yy}" stroke="{INK}" stroke-width="3"/>')
    o.append(f'<line x1="{ax}" y1="{top-20}" x2="{ax}" y2="{bot}" stroke="{INK}" stroke-width="3"/>')
    o.append(text(x0 + 20, (top + bot) / 2, ylab, F, INK, 400, "middle", f'transform="rotate(-90 {x0+20} {(top+bot)/2})"'))
    n = len(bars); slot = (x0 + pw - ax - 20) / n; bwid = slot * 0.62
    for i, (lab, v, lo, hi, col) in enumerate(bars):
        cx = ax + 20 + slot * (i + .5)
        o.append(f'<rect x="{cx-bwid/2}" y="{bot - v*sc}" width="{bwid}" height="{v*sc}" fill="{col}" rx="6"/>')
        o.append(f'<line x1="{cx}" y1="{bot-lo*sc}" x2="{cx}" y2="{bot-hi*sc}" stroke="{INK}" stroke-width="5"/>')
        for yy in (lo, hi): o.append(f'<line x1="{cx-18}" y1="{bot-yy*sc}" x2="{cx+18}" y2="{bot-yy*sc}" stroke="{INK}" stroke-width="5"/>')
        o.append(text(cx, bot - hi * sc - 22, fmt(v), F, INK, 700, "middle"))
        for j, l in enumerate(lab): o.append(text(cx, bot + 58 + j * 54, l, F, INK, 400, "middle"))
    i1, i2 = len(bars) - 2, len(bars) - 1
    cx1 = ax + 20 + slot * (i1 + .5); cx2 = ax + 20 + slot * (i2 + .5); yb = bot - max(b_[3] for b_ in bars) * sc - 120
    o.append(f'<path d="M {cx1} {yb+30} L {cx1} {yb} L {cx2} {yb} L {cx2} {yb+30}" fill="none" stroke="{INK}" stroke-width="4"/>')
    o.append(text((cx1 + cx2) / 2, yb - 16, p, F, INK, 400, "middle"))
    return "".join(o)
GR = "#BDBDC6"
A = S["ablate/focus"]; P = S["patch/focus"]
pt = 720
b.append(f'<line x1="30" y1="{pt}" x2="{W-30}" y2="{pt}" stroke="{INK}" stroke-width="3"/>')
b.append(panel(30, pt + 10, 1090, 1240, ["Switching the 6 heads off", "halves the focus boost"],
               "extra attention to P2", [(["all heads", "on"], *A["intact"], PURPLE), (["6 heads", "off"], *A["movers"], PURPLE),
                                         (["6 random", "heads off"], *A["random"], GR)],
               66, [0, 25, 50], lambda t: f"+{t:.0f}%", "p < 0.0001"))
pp = {k: [100 * x for x in v] for k, v in P.items() if k in ("movers", "random")}
b.append(panel(1180, pt + 10, 1090, 1240, ["Copying the 6 heads from a", "prompt naming P3 moves", "attention toward P3"],
               "how far attention moved to P3", [(["6 heads"], *pp["movers"], PURPLE), (["6 random", "heads"], *pp["random"], GR)],
               56, [0, 20, 40], lambda t: f"{t:.0f}%", "p < 0.0001"))
render("".join(b), W, H, "hero")
