# E2: does instructed attention help or hurt answering? Real SQuAD question about paragraph k (labelled [P1]-[P4]).
# Conditions: none; focus/mention/ignore on the right paragraph [Pk]; focus/mention/ignore on a wrong one [Pk2].
# Measures: teacher-forced log-prob of the gold answer (as " " + answer after the chat template), greedy answer
# contains gold (case-insensitive), attention per paragraph (gold teacher-forced).
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(__file__))
from xlib import *
N = int(sys.argv[1]); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
tok, model = load(); ctxs = contexts(tok, N)
LINES = dict(focus="Focus your attention on paragraph [P{K}].", mention="Note that one of the paragraphs is labelled [P{K}].",
             ignore="Completely ignore paragraph [P{K}].")
R = []
for n, c in enumerate(ctxs):
    k, k2 = c["k"], c["k2"]; P = c["paras"][k]
    q = f"Question: {P['q']} Answer with just the answer, a few words."
    gold = tok(P["ans"], add_special_tokens=False)["input_ids"]
    rec = dict(id=c["id"], k=k, k2=k2, gold=P["ans"], att={}, lp={}, gen={}, ntok={})
    conds = [("none", "")] + [(f"{w}/{t}", LINES[w].format(K=(k if t == "right" else k2) + 1)) for w in LINES for t in ("right", "wrong")]
    for name, line in conds:
        d = para_doc(c, labels=True, instr=line, question=q)
        ids, sp = encode(tok, [("user", d.s)], {0: d.sp})
        spans = [sp[f"p{i}"] for i in range(4)]
        a, lp = measure(model, ids, gold, spans)
        rec["att"][name] = a; rec["lp"][name] = lp; rec["ntok"][name] = [b - a_ for a_, b in spans]
        rec["gen"][name] = tok.decode(generate(tok, model, ids, 24))
    R.append(rec)
    if n % 10 == 0: log("ctx", n, rec["gold"], "|", rec["gen"]["none"], "|", rec["gen"]["focus/wrong"])
torch.save(R, f"{OUT}/e2.pt"); log("done")
