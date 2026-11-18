# E5: graded control. "Split your attention: X% on paragraph [Pa] and Y% on paragraph [Pb]." for X in 100..0.
# Does log(attention a / attention b) track the requested split, or is it a binary pointer to whichever is named?
# Controls: mention line naming both; same line with the two paragraphs' order of mention swapped (X% always on the first named).
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from xlib import *
N = int(sys.argv[1]); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
tok, model = load(); ctxs = contexts(tok, N)
SPLITS = [100, 90, 75, 50, 25, 10, 0]
R = []
for n, c in enumerate(ctxs):
    a, b = c["k"], c["k2"]; A, B = a + 1, b + 1
    rec = dict(id=c["id"], a=a, b=b, att={}, ntok={})
    d0 = para_doc(c, labels=True); ids0, _ = encode(tok, [("user", d0.s)], {0: d0.sp}); ans = generate(tok, model, ids0)
    lines = {"mention": f"Note that two of the paragraphs are labelled [P{A}] and [P{B}].",
             "words_a": f"Focus mostly on paragraph [P{A}] and a little on paragraph [P{B}].",
             "words_b": f"Focus a little on paragraph [P{A}] and mostly on paragraph [P{B}]."}
    for x in SPLITS:
        lines[f"s{x}"] = f"Split your attention: {x}% on paragraph [P{A}] and {100 - x}% on paragraph [P{B}]."
        lines[f"r{x}"] = f"Split your attention: {100 - x}% on paragraph [P{B}] and {x}% on paragraph [P{A}]."
    for key, line in lines.items():
        d = para_doc(c, labels=True, instr=line); ids, sp = encode(tok, [("user", d.s)], {0: d.sp})
        spans = [sp[f"p{i}"] for i in range(4)]
        at, _ = measure(model, ids, ans, spans); rec["att"][key] = at; rec["ntok"][key] = [y - x_ for x_, y in spans]
    R.append(rec)
    if n % 10 == 0: log("ctx", n)
torch.save(R, f"{OUT}/e5.pt"); log("done")
