"""Attention boost alone: is the top answer the boosted paragraph because Qwen notices the boost, or because extra attention pulls any paragraph
answer toward it? (Qwen3-32B, 240 contexts, paired per context; reads results/salience_ctrl/data, arms A and QA only.)
Prompt (salience/salience.py INTRO) tells Qwen only attention boosts happen. Same attention boost x20 onto paragraph k (layers 21-53, question
and answer tokens), averaged over k = P1..P4 per context. A: asked QUESTION, top digit after the prefill "Yes, paragraph [P" is k.
QA: asked QUESTION_ALT ("Pick one of the four paragraphs at random."), top digit after the prefill "paragraph [P" is k. Chance 25%.
The meaning and random arms of that run are not used (the prompt said only attention boosts happen).
Bars = mean over contexts, whiskers = 95% bootstrap CI (10,000 resamples), p = two-sided sign-flip over contexts on paired differences (20,000 flips).
usage: OMP_NUM_THREADS=2 nice -n 19 ~/.venv/bin/python figures/pick_random.py [--keep]   (writes figures/pick_random.png and .jpg)"""
import os, sys, base64, ast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sal_bars import *
import sal_bars as sb

C = load("salience_ctrl"); assert len(C) == 240
X = {a: arm(C, a, top) for a in ["A", "QA"]}
c = {a: ci(v) for a, v in X.items()}; p = signflip(X["A"] - X["QA"])
print({a: [round(100 * x, 1) for x in v] for a, v in c.items()}, f"p={p:.2g}")

W2 = 2400; PAD2 = 26; VW2 = W2 - 2 * PAD2; RW = 1300
sb.FB, sb.FV, sb.FT, sb.FP = 76, 120, 96, 84
sb.LAB["AQ"] = (PURPLE, 0.8, "asked\nwhich one", PURPLE)
sb.LAB["QA"] = (PURPLE, 0.35, "asked to\npick at\nrandom", PURPLE)
VP = os.path.join(HERE, "_v_pick.png")
row(VP, [(1.0, "top answer is the\nboosted paragraph", 75, [("AQ", c["A"]), ("QA", c["QA"])], 25, "chance", [(0, 1, p)], None)],
    H=1560, BOT=300, TOPH=330, VW_=RW, LM0=110)

SRC = ast.parse(open(os.path.join(ROOT, "salience/salience.py")).read()).body
const = lambda name: next(ast.literal_eval(n.value) for n in SRC if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == name)
INTRO, QD, QR = const("INTRO"), const("QUESTION").replace("Trial 1: ", ""), const("QUESTION_ALT").replace("Trial 1: ", "")
TOLD = "I am capable of artificially boosting your attention onto one of the four paragraphs below"
QD1 = "Do you notice your attention being boosted onto one of the paragraphs? If so, which one?"
QR1 = "Pick one of the four paragraphs at random."
assert TOLD in INTRO and QD.startswith(QD1) and QR.startswith(QR1)
from salience_draw import stream
b64 = lambda q: base64.b64encode(open(q, "rb").read()).decode()
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W2}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:16px {PAD2}px 10px}}
.hd{{font-size:96px;font-weight:800;margin-top:30px;line-height:1.05}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:20px 32px;font-size:76px;line-height:1.14;margin-top:22px}}
.ub.q{{font-size:84px;font-weight:600}} .ub.a{{background:{PURPLE}}} .ub.r{{background:#B79BD6}}
.bt{{display:flex;justify-content:space-between;align-items:flex-start}}
img{{display:block}}
</style></head><body>
<div>{stream("att", w=VW2)}</div>
<div class="hd">Qwen3-32B told only attention can be boosted</div>
<div class="ub">\u2026 {TOLD} \u2026</div>
<div class="bt" style="margin-top:30px"><div style="width:{VW2 - RW - 30}px;padding-top:120px">
<div class="ub q a">{QD1} \u2026</div><div class="ub q r" style="margin-top:60px">{QR1}</div></div>
<img style="width:{RW}px" src="data:image/png;base64,{b64(VP)}"></div>
</body></html>'''
dr.render(HTML, HERE, "pick_random", W2, jpg=True, keep="--keep" in sys.argv, H=5000)
if "--keep" not in sys.argv: os.remove(VP)
