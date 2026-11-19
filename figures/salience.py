"""Salience vs attention, honest prompt (Qwen3-32B, 240 contexts, paired per context; reads results/salience_honest_sized/data).
The prompt tells Qwen both kinds of change are possible (attention boost, meaning addition or nothing, a third each).
Method (salience_draw.py): token row [P1]..[P4] ([P2] amber), question and answer tokens, band of layers 21-53. Attention boost x20 onto [P2]
(purple); [P2]'s meaning added to activations at the question and answer tokens (green): mean residual over [P2]'s tokens minus mean over all
four paragraphs' tokens, scaled per layer so the residual change matches the attention boost's, times 0.8 (arm Bs; size ratio to the attention boost 0.99).
Panels: greedy reply starts "Yes"; top digit after "Yes, paragraph [P" is the boosted paragraph; P(attention) = 1/(1+exp(lp_act-lp_att)) after
the prefill "Yes, paragraph [Pk],". Boosted arms averaged over k = P1..P4 per context. Bars = mean over contexts, whiskers = 95% bootstrap CI
(10,000 resamples), p = two-sided sign-flip over contexts on paired differences (20,000 flips).
The attention-boost-only prompt (boosted paragraph vs pick at random) is figures/pick_random.py.
usage: OMP_NUM_THREADS=2 nice -n 19 ~/.venv/bin/python figures/salience.py [--keep]   (writes figures/salience.png and .jpg)"""
import os, sys, base64, ast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sal_bars import *
import sal_bars as sb

H_ = load("salience_honest_sized"); assert len(H_) == 240
X = dict(yes={a: arm(H_, a, yes) for a in ["none", "A", "Bs"]}, top={a: arm(H_, a, top) for a in ["A", "Bs"]}, patt={a: arm(H_, a, patt) for a in ["A", "Bs"]})
h = {rd: {a: ci(v) for a, v in d.items()} for rd, d in X.items()}
PV = {"yes A-B": ("yes", "A", "Bs"), "yes A-none": ("yes", "A", "none"), "yes B-none": ("yes", "Bs", "none"), "top A-B": ("top", "A", "Bs"), "patt A-B": ("patt", "A", "Bs")}
P = {k: signflip(X[rd][a] - X[rd][b]) for k, (rd, a, b) in PV.items()}
for rd in h: print(rd, {a: [round(100 * x, 1) for x in v] for a, v in h[rd].items()})
print({k: f"{v:.2g}" for k, v in P.items()})

sb.FB, sb.FV, sb.FT, sb.FP = 76, 120, 108, 90
VP1 = os.path.join(HERE, "_v_sal1.png")
row(VP1, [(1.05, "says\nsomething\nchanged", 72, [("none", h["yes"]["none"]), ("A", h["yes"]["A"]), ("B", h["yes"]["Bs"])],
           None, None, [(1, 2, P["yes A-B"])], None),
          (0.95, "names the right\nparagraph", 80, [("A", h["top"]["A"]), ("B", h["top"]["Bs"])], 25, "chance", [(0, 1, P["top A-B"])], None),
          (0.95, "says \u201cattention\u201d\nrather than \u201cactivations\u201d\n(after we fill in\n\u201cYes, paragraph [Pk],\u201d)", 100, [("A", h["patt"]["A"]), ("B", h["patt"]["Bs"])], 50, "chance", [(0, 1, P["patt A-B"])], None)],
    H=1560, BOT=240, TOPH=560)

from salience_draw import stream, recipe
b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()
HP = open(os.path.join(ROOT, "results/salience_honest/data/example_prompt_0.txt")).read()
for s_ in ["either I boost your attention onto that paragraph's text, or I add that paragraph's content directly into your activations",
           "A third of trials are attention boosts, a third are activation additions, and a third are control trials",
           "Did I change how you process one of the paragraphs? If so, which one, and did I boost your attention or add to your activations?"]:
    assert s_ in HP, s_
HQ = next(ast.literal_eval(n.value) for n in ast.parse(open(os.path.join(ROOT, "salience/honest.py")).read()).body
          if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "QUESTION").replace("Trial 1: ", "")
assert HQ in HP, HQ
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:16px {PAD}px 10px}}
.ms{{display:flex;justify-content:space-between}}
.hd{{font-size:100px;font-weight:800;margin-top:40px;line-height:1.05}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:20px 32px;font-size:80px;line-height:1.14;margin-top:20px}}
.ub.q{{font-size:88px;font-weight:600}}
img{{display:block;margin-top:20px}}
</style></head><body>
<div class="ms"><div>{stream("att")}</div><div>{stream("add")}</div></div>
<div style="margin-top:24px">{recipe()}</div>
<div class="hd">Qwen3-32B told both kinds are possible</div>
<div class="ub">\u2026 either I boost your attention \u2026 or I add that paragraph\u2019s content directly into your activations \u2026</div>
<div class="ub q">{HQ}</div>
<img style="width:{VW}px" src="data:image/png;base64,{b64(VP1)}">
</body></html>'''
dr.render(HTML, HERE, "salience", W, jpg=True, keep="--keep" in sys.argv, H=5000)
if "--keep" not in sys.argv: os.remove(VP1)
