# E7: is the answer choice in the conflict task carried by (i) the six early instruction-reading heads and (ii) the late
# number-reading heads that E6 found most selective (picked on E6's 120 items; evaluated here on the next 120, fresh)?
# Ablations (zero head outputs, all positions): six early vs random six from layers 20-26; 15 late vs random 15 from the same layers.
# Readout: margin log p(orig) - log p(alt); last-position attention to the two numbers (for the late selection under early ablation).
import sys, os, re, json, random, torch
sys.path.insert(0, os.path.dirname(__file__))
from xlib import *
N0, N1 = int(sys.argv[1]), int(sys.argv[2]); OUT = sys.argv[3]; os.makedirs(OUT, exist_ok=True)
tok, model = load(); pool = squad_pool(tok)
rng = random.Random(23)
num = [(t, p) for t, ps in pool.items() for p in ps if re.fullmatch(r"\d{1,4}", p["ans"]) and p["text"].count(p["ans"]) == 1]
rng.shuffle(num); titles = sorted(pool)
def alt(a):
    v = int(a)
    if len(a) == 4 and 1000 <= v: d = rng.choice([-1, 1]) * rng.randint(7, 60); return str(v + d)
    w = v + rng.choice([-1, 1]) * max(1, rng.randint(max(1, v // 4), max(2, v // 2 + 2)))
    return str(w if w > 0 and w != v else v + 3)
SIX = [(20, 19), (20, 29), (23, 61), (25, 28), (25, 30), (26, 5)]
LATE = [tuple(x) for x in json.load(open("out/e6/late_heads.json"))]
def zd(hs):
    z = {}
    for L, h in hs: z.setdefault(L, []).append(h)
    return z
LINES = dict(focus="Focus your attention on paragraph [P{K}].", ignore="Completely ignore paragraph [P{K}].")
@torch.no_grad()
def lastpos_att(ids, spans, zero):
    S.update(rec=True, rows=[len(ids) - 1], spans=spans, store={}, bias=None, zero=zero)
    model(torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1); S.update(rec=False, zero=None)
    return torch.stack([S["store"][L] for L in range(len(S["store"]))]).half()
R = []
for n in range(N1):
    t, P = num[n]; a = P["ans"]; a2 = alt(a)
    altP = dict(P, text=P["text"].replace(a, a2))
    ds = rng.sample([x for x in titles if x != t], 2); paras = [P, altP] + [rng.choice(pool[x]) for x in ds]
    order = list(range(4)); rng.shuffle(order); paras = [paras[i] for i in order]
    io, ia = order.index(0), order.index(1)
    if n < N0 or altP["text"].count(a2) != 1: continue
    r2 = random.Random(5000 + n)
    ABL = {"intact": None, "early6": zd(SIX), "early_rand": zd([(L, r2.randrange(64)) for L in r2.choices(range(20, 27), k=6)]),
           "late15": zd(LATE), "late_rand": zd([(L, r2.randrange(64)) for L in r2.choices(sorted({L for L, _ in LATE}), k=len(LATE))])}
    c = dict(paras=paras, x=0, y=0); q = f"Question: {P['q']} Answer with just the answer."
    ta, tb = tok(a, add_special_tokens=False)["input_ids"], tok(a2, add_special_tokens=False)["input_ids"]
    rec = dict(n=n, a=a, a2=a2, io=io, ia=ia, m={}, att={}, ntok={})
    conds = [("none", "")] + [(f"{w}/{s}", LINES[w].format(K=(io if s == "orig" else ia) + 1)) for w in LINES for s in ("orig", "alt")]
    for name, line in conds:
        d = para_doc(c, labels=True, instr=line, question=q)
        d.mark("num_o", a, within=f"p{io}"); d.mark("num_a", a2, within=f"p{ia}")
        ids, sp = encode(tok, [("user", d.s)], {0: d.sp})
        spans = [sp[f"p{i}"] for i in range(4)] + [sp["num_o"], sp["num_a"]]
        for ab, z in ABL.items():
            _, la = measure(model, ids, ta, spans, zero=z); _, lb = measure(model, ids, tb, spans, zero=z)
            rec["m"][f"{name}|{ab}"] = la - lb
            if ab in ("intact", "early6", "early_rand"):
                rec["att"][f"{name}|{ab}"] = lastpos_att(ids, spans, z); rec["ntok"][name] = [y - x for x, y in spans]
    R.append(rec)
    if n % 10 == 0: log("ctx", n, {k: round(v, 1) for k, v in rec["m"].items() if k.startswith("focus/alt")})
torch.save(R, f"{OUT}/e7.pt"); log("done")
