# Query-side vs key-side swap: the 10 heads whose attention to the named paragraph changes most.
import json, numpy as np
from figlib import *
Q = json.load(open(os.path.join(HERE, "qk_summary.json")))
W, H = 2000, 1480; F = 46
b = []
b.append(text(30, 66, "The instruction acts through the answer tokens' queries:", F, INK, 700))
b.append(text(30, 124, "swapping in queries from a prompt naming another paragraph moves", F, INK, 700))
b.append(text(30, 182, "attention there; swapping in keys does nothing", F, INK, 700))
x0, x1, y0, y1 = 300, 1960, 330, 1290
vals = [Q[f"{c}/readers"][s] for c in ["focus", "ignore"] for s in ["query", "key"]]
ymax = max(1.1, max(v[2] for v in vals) * 1.15); ymin = min(0, min(v[1] for v in vals) * 1.15)
Y = lambda v: y1 - (y1 - y0) * (v - ymin) / (ymax - ymin)
b.append(f'<line x1="{x0}" y1="{Y(0)}" x2="{x1}" y2="{Y(0)}" stroke="{INK}" stroke-width="3"/>')
b.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{INK}" stroke-width="3"/>')
for t in [0, 0.5, 1.0]: b.append(text(x0 - 16, Y(t) + 15, f"{t*100:.0f}%", F, INK, 400, "end"))
yl = (y0 + y1) / 2
b.append(text(90, yl, "attention moved to the", F, INK, 400, "middle", f'transform="rotate(-90 90 {yl})"'))
b.append(text(150, yl, "newly named paragraph", F, INK, 400, "middle", f'transform="rotate(-90 150 {yl})"'))
b.append(f'<line x1="{x0}" y1="{Y(1)}" x2="{x1}" y2="{Y(1)}" stroke="{INK}" stroke-width="3" stroke-dasharray="10 10"/>')
gw = (x1 - x0 - 40) / 2; bw = gw * 0.32
for gi, (c, cl) in enumerate([("focus", "focus on"), ("ignore", "ignore")]):
    gx = x0 + 40 + gi * gw
    cx = []
    for si, (s, col, lab) in enumerate([("query", BLUE, "queries"), ("key", ORANGE, "keys")]):
        m, lo, hi = Q[f"{c}/readers"][s]
        x = gx + gw * (0.25 + 0.42 * si) - bw / 2; cx.append(x + bw / 2)
        top, bot = (Y(m), Y(0)) if m >= 0 else (Y(0), Y(m))
        b.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{bot-top:.1f}" fill="{col}"/>')
        b.append(f'<line x1="{x+bw/2:.1f}" y1="{Y(lo):.1f}" x2="{x+bw/2:.1f}" y2="{Y(hi):.1f}" stroke="{INK}" stroke-width="4"/>')
        for yy in (lo, hi): b.append(f'<line x1="{x+bw/2-18:.1f}" y1="{Y(yy):.1f}" x2="{x+bw/2+18:.1f}" y2="{Y(yy):.1f}" stroke="{INK}" stroke-width="4"/>')
        b.append(text(x + bw / 2, Y(max(hi, 0)) - 20, (f"{m*100:.0f}%" if round(m*100) != 0 else "0%"), F, INK, 700, "middle"))
        b.append(text(x + bw / 2, y1 + 60, lab, F, col, 700, "middle"))
    b.append(text((cx[0] + cx[1]) / 2, y1 + 130, cl, F, INK, 700, "middle"))
    p = Q[f"{c}/readers"]["p_query_vs_key"]
    yb = Y(max(Q[f"{c}/readers"][s][2] for s in ["query", "key"])) - 90
    b.append(f'<path d="M {cx[0]} {yb+24} L {cx[0]} {yb} L {cx[1]} {yb} L {cx[1]} {yb+24}" fill="none" stroke="{INK}" stroke-width="3"/>')
    b.append(text((cx[0] + cx[1]) / 2, yb - 16, "p < 0.0001" if p < 1e-4 else f"p = {p:.2g}", F, INK, 400, "middle"))
render("".join(b), W, H, "qk")
