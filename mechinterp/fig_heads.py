# Which heads carry it: per-head change in attention to the named paragraph (vs "mention"), focus vs ignore.
import json, torch, numpy as np
from figlib import *
f = torch.load(os.path.join(HERE, "heads_focus.pt"))["dT"].numpy() * 1e4
g = torch.load(os.path.join(HERE, "heads_ignore.pt"))["dT"].numpy() * 1e4
SW = json.load(open(os.path.join(HERE, "sweep_summary.json")))
r = np.corrcoef(f.ravel(), g.ravel())[0, 1]
W, H = 1600, 1640; F = 46
b = []
# ---- left: scatter
x0, x1, y0, y1 = 200, 1560, 200, 1440
xm, xM, ym, yM = -6, 24, -6, 3
X = lambda v: x0 + (x1 - x0) * (v - xm) / (xM - xm); Y = lambda v: y1 - (y1 - y0) * (v - ym) / (yM - ym)
b.append(text(30, 60, "The same heads look more at the named paragraph", F, INK, 700))
b.append(text(30, 118, "under focus and less under ignore", F, INK, 700))
b.append(f'<line x1="{X(0)}" y1="{y0}" x2="{X(0)}" y2="{y1}" stroke="{INK}" stroke-width="2" stroke-dasharray="8 8"/>')
b.append(f'<line x1="{x0}" y1="{Y(0)}" x2="{x1}" y2="{Y(0)}" stroke="{INK}" stroke-width="2" stroke-dasharray="8 8"/>')
for a, c in zip(f.ravel(), g.ravel()):
    if xm <= a <= xM and ym <= c <= yM:
        b.append(f'<circle cx="{X(a):.1f}" cy="{Y(c):.1f}" r="10" fill="{INK}" opacity="0.35"/>')
b.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="{INK}" stroke-width="3"/>')
for t in [0, 10, 20]: b.append(text(X(t), y1 + 56, f"{t:+d}" if t else "0", F, INK, 400, "middle"))
for t in [-5, 0]: b.append(text(x0 - 16, Y(t) + 15, f"{t:+d}" if t else "0", F, INK, 400, "end"))
b.append(text((x0 + x1) / 2, y1 + 116, "attention change under focus on", F, PURPLE, 700, "middle"))
b.append(text(70, (y0 + y1) / 2, "attention change under ignore", F, RED, 700, "middle", f'transform="rotate(-90 70 {(y0+y1)/2})"'))
b.append(text(X(23), Y(2.0), "one dot = one of 4,096 heads", F, INK, 400, "end"))
b.append(text(X(23), Y(1.2), f"correlation {r:.2f}", F, INK, 700, "end"))
render("".join(b), W, H, "heads")
print("r", r)
