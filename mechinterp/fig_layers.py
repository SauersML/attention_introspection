# Where the paragraph number is written: probe accuracy at the answer tokens per layer (top) and
# residual patching from a prompt naming another paragraph, at the instruction vs the answer tokens (bottom).
import json
from figlib import *
SW = json.load(open(os.path.join(HERE, "sweep_summary.json"))); PS = json.load(open(os.path.join(HERE, "patch_summary.json")))
W, H = 2300, 2160; F = 46
b = []
def axes(x0, y0, x1, y1, xmax, ymin, ymax, yt, yfmt, xlab, ylab, xt=(0, 20, 40, 60)):
    o = [f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{INK}" stroke-width="3"/>',
         f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{INK}" stroke-width="3"/>']
    X = lambda L: x0 + (x1 - x0) * L / xmax; Y = lambda v: y1 - (y1 - y0) * (v - ymin) / (ymax - ymin)
    for t in yt: o.append(text(x0 - 16, Y(t) + 15, yfmt(t), F, INK, 400, "end"))
    for t in xt: o.append(text(X(t), y1 + 56, str(t), F, INK, 400, "middle"))
    o.append(text((x0 + x1) / 2, y1 + 112, xlab, F, INK, 400, "middle"))
    o.append(text(x0 - 150, (y0 + y1) / 2, ylab, F, INK, 400, "middle", f'transform="rotate(-90 {x0-150} {(y0+y1)/2})"'))
    return o, X, Y
def line(pts, X, Y, col, w=7, dash=""):
    d = " ".join(f"{'M' if i == 0 else 'L'} {X(x):.1f} {Y(y):.1f}" for i, (x, y) in enumerate(pts))
    return f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{w}" stroke-linejoin="round" {dash}/>'
def band(lo, hi, X, Y, col):
    d = " ".join(f"{'M' if i == 0 else 'L'} {X(x):.1f} {Y(y):.1f}" for i, (x, y) in enumerate(lo + hi[::-1])) + " Z"
    return f'<path d="{d}" fill="{col}" opacity="0.22"/>'
# ---- top: probe
x0, x1 = 260, 1900
y0, y1 = 120, 800
b.append(text(30, 60, "From layer 20 the answer tokens contain the paragraph number, whatever the instruction", F, INK, 700))
o, X, Y = axes(x0, y0, x1, y1, 63, 0.2, 1.0, [0.25, 0.5, 0.75, 1.0], lambda t: f"{t*100:.0f}%", "layer", "linear probe accuracy")
b += o
Ls = SW["probe_layers"]
b.append(f'<line x1="{x0}" y1="{Y(.25)}" x2="{x1}" y2="{Y(.25)}" stroke="{INK}" stroke-width="3" stroke-dasharray="10 10"/>')
b.append(text(x1 + 20, Y(.25) + 15, "chance", F, INK))
for cond, col, yl in [("focus", PURPLE, .80), ("mention", INK, .58), ("ignore", RED, .69)]:
    v = SW["probe"][f"{cond}/answer_mean"]; b.append(line(list(zip(Ls, v)), X, Y, col))
    lab = {"focus": "focus on", "mention": "mention", "ignore": "ignore"}[cond]
    b.append(text(x1 + 20, Y(yl) + 15, lab, F, col, 700))
# ---- bottom: patching
y0, y1 = 1150, 2010
b.append(text(30, 1030, "Copying the residual stream from a prompt naming another paragraph:", F, INK, 700))
b.append(text(30, 1088, "the number moves to the answer tokens in two steps", F, INK, 700))
o, X, Y = axes(x0, y0, x1, y1, 63, 0, 1.05, [0, 0.5, 1.0], lambda t: f"{t*100:.0f}%", "layer patched", "attention moved to new paragraph", xt=(0, 20, 40))
b += o
R = PS["kswap/focus"]
for s, col, lab in [("instr", PURPLE, "instruction"), ("answer", BLUE, "answer")]:
    d = R[s]; Lz = sorted(int(k) for k in d); Lz = [L for L in Lz if L <= 51]
    m = [(L, d[str(L)][0]) for L in Lz]; lo = [(L, d[str(L)][1]) for L in Lz]; hi = [(L, d[str(L)][2]) for L in Lz]
    b.append(band(lo, hi, X, Y, col)); b.append(line(m, X, Y, col))
    yl = Y(0.97) if s == "answer" else Y(0.17)
    b.append(text(X(51) + 30, yl, lab, F, col, 700)); b.append(text(X(51) + 30, yl + 52, "tokens", F, col, 700))
render("".join(b), W, H, "layers")
