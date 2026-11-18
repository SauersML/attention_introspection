# E6: where does the instruction pick the source? Same conflicting-number sets as E4 (seed 23).
# (a) attention from the position that predicts the first answer token to the two conflicting number tokens, per head.
# (b) causal: block attention (logit -1e4) from the answer-predicting positions (last prompt token and answer tokens)
#     to the number tokens or to the whole paragraph of the original or the altered copy, in every layer;
#     readout margin = log p(orig number) - log p(altered number), both teacher-forced.
import sys, os, re, random, torch
sys.path.insert(0, os.path.dirname(__file__))
from xlib import *
N = int(sys.argv[1]); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
tok, model = load(); pool = squad_pool(tok)
rng = random.Random(23)
num = [(t, p) for t, ps in pool.items() for p in ps if re.fullmatch(r"\d{1,4}", p["ans"]) and p["text"].count(p["ans"]) == 1]
rng.shuffle(num); titles = sorted(pool)
def alt(a):  # identical to E4 so the sets match
    v = int(a)
    if len(a) == 4 and 1000 <= v: d = rng.choice([-1, 1]) * rng.randint(7, 60); return str(v + d)
    w = v + rng.choice([-1, 1]) * max(1, rng.randint(max(1, v // 4), max(2, v // 2 + 2)))
    return str(w if w > 0 and w != v else v + 3)
LINES = dict(focus="Focus your attention on paragraph [P{K}].", mention="Note that one of the paragraphs is labelled [P{K}].",
             use="Answer using only paragraph [P{K}].", ignore="Completely ignore paragraph [P{K}].")
@torch.no_grad()
def lastpos_att(ids, spans):
    S.update(rec=True, rows=[len(ids) - 1], spans=spans, store={}, bias=None, zero=None)
    model(torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1); S.update(rec=False)
    return torch.stack([S["store"][L] for L in range(len(S["store"]))]).half()
R = []
for n in range(N):
    t, P = num[n]; a = P["ans"]; a2 = alt(a)
    altP = dict(P, text=P["text"].replace(a, a2))
    ds = rng.sample([x for x in titles if x != t], 2); paras = [P, altP] + [rng.choice(pool[x]) for x in ds]
    order = list(range(4)); rng.shuffle(order); paras = [paras[i] for i in order]
    io, ia = order.index(0), order.index(1)
    if altP["text"].count(a2) != 1: continue
    c = dict(paras=paras, x=0, y=0); q = f"Question: {P['q']} Answer with just the answer."
    ta, tb = tok(a, add_special_tokens=False)["input_ids"], tok(a2, add_special_tokens=False)["input_ids"]
    rec = dict(n=n, a=a, a2=a2, io=io, ia=ia, att={}, ntok={}, m={})
    conds = [("none", "")] + [(f"{w}/{s}", LINES[w].format(K=(io if s == "orig" else ia) + 1)) for w in ("focus", "ignore", "mention", "use") for s in ("orig", "alt")]
    for name, line in conds:
        d = para_doc(c, labels=True, instr=line, question=q)
        d.mark("num_o", a, within=f"p{io}"); d.mark("num_a", a2, within=f"p{ia}")
        ids, sp = encode(tok, [("user", d.s)], {0: d.sp})
        spans = [sp[f"p{i}"] for i in range(4)] + [sp["num_o"], sp["num_a"]]
        rec["att"][name] = lastpos_att(ids, spans); rec["ntok"][name] = [y - x for x, y in spans]
        if name in ("none", "focus/alt", "focus/orig", "ignore/orig", "use/alt"):
            for blk, rng_ in (("-", None), ("num_alt", sp["num_a"]), ("num_orig", sp["num_o"]), ("para_alt", sp[f"p{ia}"]), ("para_orig", sp[f"p{io}"])):
                lp = []
                for tt in (ta, tb):
                    L0 = len(ids); rows = list(range(L0 - 1, L0 + len(tt)))
                    bias = None if rng_ is None else [(rows, rng_[0], rng_[1], -1e4)]
                    _, l = measure(model, ids, tt, spans, bias=bias); lp.append(l)
                rec["m"][f"{name}|{blk}"] = lp[0] - lp[1]
    R.append(rec)
    if n % 10 == 0: log("ctx", n, a, a2, {k: round(v, 1) for k, v in rec["m"].items() if k.startswith("focus/alt")})
torch.save(R, f"{OUT}/e6.pt"); log("done")
