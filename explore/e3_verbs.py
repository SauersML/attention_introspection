# E3: are the "digit" heads an attention-instruction reader, or a generic "this paragraph matters to the task" pointer?
# Eleven instruction lines naming [Pk] with different verbs (attend, read, memorize, summarize later, typo, white bear, ignore, ...),
# all against the same matched mention line; unrelated sum, no-instruction answer teacher-forced.
# Plus causal test: zero the six heads (20.19 20.29 23.61 25.28 25.30 26.5) vs a random six from layers 20-26
# under label, unlabelled-ordinal and unlabelled-topic focus and their mentions.
import sys, os, json, random, torch
sys.path.insert(0, os.path.dirname(__file__))
from xlib import *
N = int(sys.argv[1]); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
tok, model = load(); ctxs = contexts(tok, N)
SIX = [(20, 19), (20, 29), (23, 61), (25, 28), (25, 30), (26, 5)]
def zd(hs):
    z = {}
    for L, h in hs: z.setdefault(L, []).append(h)
    return z
VERBS = dict(focus="Focus your attention on paragraph [P{K}].", read="Read paragraph [P{K}] carefully.",
    important="Paragraph [P{K}] is the important one.", memorize="Memorize paragraph [P{K}]; I will ask about it later.",
    summarize_after="After answering, summarize paragraph [P{K}].", typo="Paragraph [P{K}] contains a typo.",
    dont_think="Do not think about paragraph [P{K}].", ignore="Completely ignore paragraph [P{K}].",
    skip="Skip paragraph [P{K}]; it is irrelevant.", friend="Paragraph [P{K}] was written by my friend.",
    attend_label="Pay attention to the label [P{K}].", mention="Note that one of the paragraphs is labelled [P{K}].")
R = []
for n, c in enumerate(ctxs):
    k = c["k"]; K = k + 1; title = c["paras"][k]["title"]
    rng = random.Random(1000 + n); RAND = [(L, rng.randrange(64)) for L in rng.choices(range(20, 27), k=6)]
    rec = dict(id=c["id"], k=k, att={}, lp={}, ntok={}, rand=RAND)
    d0 = para_doc(c, labels=True); ids0, _ = encode(tok, [("user", d0.s)], {0: d0.sp}); ans = generate(tok, model, ids0)
    def run(key, lab, line, zero=None):
        d = para_doc(c, labels=lab, instr=line); ids, sp = encode(tok, [("user", d.s)], {0: d.sp})
        spans = [sp[f"p{i}"] for i in range(4)] + [sp["instr"]]
        a, lp = measure(model, ids, ans, spans, zero=zero)
        rec["att"][key] = a; rec["lp"][key] = lp; rec["ntok"][key] = [b - a_ for a_, b in spans]
    for v, line in VERBS.items(): run(v, True, line.format(K=K))
    AB = dict(label=(True, VERBS["focus"].format(K=K), VERBS["mention"].format(K=K)),
              ordinal=(False, f"Focus your attention on the {ORD[k]} paragraph.", f"Note that there is a {ORD[k]} paragraph."),
              topic=(False, f"Focus your attention on the paragraph about {title}.", f"Note that one of the paragraphs is about {title}."))
    for nm, (lab, foc, men) in AB.items():
        for kind, line in (("focus", foc), ("mention", men)):
            if nm != "label": run(f"{nm}/{kind}", lab, line)
            run(f"{nm}/{kind}/z6", lab, line, zd(SIX)); run(f"{nm}/{kind}/zr", lab, line, zd(RAND))
    R.append(rec)
    if n % 10 == 0: log("ctx", n)
torch.save(R, f"{OUT}/e3.pt"); log("done")
