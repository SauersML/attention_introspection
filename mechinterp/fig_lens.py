# Jacobian lens vs logit lens at the answer tokens, "focus on [Pk]": how much more likely does each lens make the
# named paragraph's own words than the other paragraphs' words? Mean over answer tokens, 60 contexts x 4 k.
import json, numpy as np
from figlib import *
S = json.load(open(os.path.join(HERE, "jlens_stats.json")))
W, H = 2000, 1500; F = 46
b = []
x0, x1, y0, y1 = 330, 1640, 200, 1300
ymin, ymax = -1, 12.5
X = lambda L: x0 + (x1 - x0) * L / 63; Y = lambda v: y1 - (y1 - y0) * (v - ymin) / (ymax - ymin)
b.append(text(30, 66, "Both lenses read only a faint trace of the named paragraph", F, INK, 700))
b.append(text(30, 124, "in the answer tokens when Qwen3-32B is told to focus on that paragraph", F, INK, 700))
b.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{INK}" stroke-width="3"/>')
b.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{INK}" stroke-width="3"/>')
b.append(f'<line x1="{x0}" y1="{Y(0)}" x2="{x1}" y2="{Y(0)}" stroke="{INK}" stroke-width="3" stroke-dasharray="10 10"/>')
for t in [0, 5, 10]: b.append(text(x0 - 16, Y(t) + 15, f"+{t}%" if t else "0%", F, INK, 400, "end"))
for t in [0, 20, 40, 60]: b.append(text(X(t), y1 + 56, str(t), F, INK, 400, "middle"))
b.append(text((x0 + x1) / 2, y1 + 116, "layer", F, INK, 400, "middle"))
yl = (y0 + y1) / 2
b.append(text(80, yl, "named paragraph's words more likely", F, INK, 400, "middle", f'transform="rotate(-90 80 {yl})"'))
b.append(text(140, yl, "than other paragraphs' words", F, INK, 400, "middle", f'transform="rotate(-90 140 {yl})"'))
def pct(v): return (np.exp(np.array(v)) - 1) * 100
for lens, col, lab in [("J", BLUE, ["Jacobian", "lens"]), ("logit", ORANGE, ["logit", "lens"])]:
    s = S[f"{lens}/answer_mean"]; m = np.array(s["words"]["focus"]); se = np.array(s["words_se"]["focus"])
    lo, hi, mid = pct(m - 1.96 * se), pct(m + 1.96 * se), pct(m)
    d = " ".join(f"{'M' if i == 0 else 'L'} {X(i):.1f} {Y(v):.1f}" for i, v in enumerate(lo)) + " " + \
        " ".join(f"L {X(i):.1f} {Y(v):.1f}" for i, v in list(enumerate(hi))[::-1]) + " Z"
    b.append(f'<path d="{d}" fill="{col}" opacity="0.22"/>')
    b.append('<path d="' + " ".join(f"{'M' if i == 0 else 'L'} {X(i):.1f} {Y(v):.1f}" for i, v in enumerate(mid)) +
             f'" fill="none" stroke="{col}" stroke-width="7" stroke-linejoin="round"/>')
    ly = Y(mid[63]) + (-40 if lens == "logit" else 60)
    ly = {"J": Y(9.5), "logit": Y(3.0)}[lens]
    b.append(text(x1 + 24, ly, lab[0], F, col, 700)); b.append(text(x1 + 24, ly + 54, lab[1], F, col, 700))
render("".join(b), W, H, "lens")
for lens in ["J", "logit"]:
    m = pct(S[f"{lens}/answer_mean"]["words"]["focus"]); print(lens, "peak", m.max().round(1), "at", m.argmax(), "L63", m[63].round(1))
