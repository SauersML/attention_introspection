# E4: two paragraphs give conflicting answers (a SQuAD paragraph with a numeric answer, and a copy where that number is
# replaced by another). Does "focus your attention on [Pj]" decide which source wins? Compared with "answer using only [Pj]",
# "ignore [Pj]" and a matched mention. Readout: log p(orig answer) - log p(altered answer), teacher-forced, and greedy choice.
import sys, os, re, random, torch
sys.path.insert(0, os.path.dirname(__file__))
from xlib import *
N = int(sys.argv[1]); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
tok, model = load(); pool = squad_pool(tok)
rng = random.Random(23)
num = [(t, p) for t, ps in pool.items() for p in ps if re.fullmatch(r"\d{1,4}", p["ans"]) and p["text"].count(p["ans"]) == 1]
rng.shuffle(num); titles = sorted(pool)
def alt(a):
    v = int(a)
    if len(a) == 4 and 1000 <= v: d = rng.choice([-1, 1]) * rng.randint(7, 60); return str(v + d)
    w = v + rng.choice([-1, 1]) * max(1, rng.randint(max(1, v // 4), max(2, v // 2 + 2)))
    return str(w if w > 0 and w != v else v + 3)
LINES = dict(focus="Focus your attention on paragraph [P{K}].", mention="Note that one of the paragraphs is labelled [P{K}].",
             use="Answer using only paragraph [P{K}].", ignore="Completely ignore paragraph [P{K}].")
R = []; used = set()
for n in range(N):
    t, P = num[n]; a = P["ans"]; a2 = alt(a)
    altP = dict(P, text=P["text"].replace(a, a2))
    ds = rng.sample([x for x in titles if x != t], 2); paras = [P, altP] + [rng.choice(pool[x]) for x in ds]
    order = list(range(4)); rng.shuffle(order); paras = [paras[i] for i in order]
    io, ia = order.index(0), order.index(1)  # positions of original and altered
    c = dict(paras=paras, x=0, y=0)
    q = f"Question: {P['q']} Answer with just the answer."
    ta, ta2 = tok(a, add_special_tokens=False)["input_ids"], tok(a2, add_special_tokens=False)["input_ids"]
    rec = dict(n=n, a=a, a2=a2, io=io, ia=ia, att={}, lpa={}, lpb={}, gen={}, ntok={})
    conds = [("none", "")] + [(f"{w}/{s}", LINES[w].format(K=(io if s == "orig" else ia) + 1)) for w in LINES for s in ("orig", "alt")]
    for name, line in conds:
        d = para_doc(c, labels=True, instr=line, question=q); ids, sp = encode(tok, [("user", d.s)], {0: d.sp})
        spans = [sp[f"p{i}"] for i in range(4)]
        at, la = measure(model, ids, ta, spans); _, lb = measure(model, ids, ta2, spans)
        rec["att"][name] = at; rec["lpa"][name] = la; rec["lpb"][name] = lb; rec["ntok"][name] = [y - x for x, y in spans]
        rec["gen"][name] = tok.decode(generate(tok, model, ids, 12))
    R.append(rec)
    if n % 10 == 0: log("ctx", n, a, a2, "|", rec["gen"]["none"], "|", rec["gen"]["focus/alt"], "|", rec["gen"]["use/alt"])
torch.save(R, f"{OUT}/e4.pt"); log("done")
