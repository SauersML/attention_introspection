# E8: where does the instruction's choice live in the residual stream? Conflict sets (fresh: items 240-299 of the seed-23 stream).
# Source run: "Focus your attention on paragraph [P_alt]."; destination run: same prompt naming [P_orig] (differs in one digit token).
# Patch the residual stream (output of decoder layer L) from source into destination at (i) the digit token, (ii) all tokens
# between the digit and the end of the question, (iii) the last prompt token only. Readout: fraction of the margin moved,
# (m_patched - m_dest) / (m_source - m_dest). Same for "Completely ignore paragraph [Pj]." (source names orig, dest names alt).
import sys, os, re, random, torch
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
LAYERS = list(range(0, 64, 6)) + [63]
layers = model.model.layers
P_ = dict(L=None, pos=None, cache=None, grab=False)
def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    L = mod.self_attn.layer_idx
    if P_["grab"]: P_["cache"][L] = h[0].detach().clone()
    elif P_["L"] == L:
        h[0, P_["pos"]] = P_["cache"][L][P_["pos"]]
    return out
for l in layers: l.register_forward_hook(hook)
@torch.no_grad()
def lp(ids, ans):
    out = model(torch.tensor([ids + ans], device="cuda"), use_cache=False, logits_to_keep=len(ans) + 1)
    l = torch.log_softmax(out.logits[0, :-1].float(), -1); return l[torch.arange(len(ans)), torch.tensor(ans, device=l.device)].sum().item()
LINES = dict(focus="Focus your attention on paragraph [P{K}].", ignore="Completely ignore paragraph [P{K}].")
R = []
N1 = min(N1, len(num)); log('items', len(num))
for n in range(N1):
    t, P = num[n]; a = P["ans"]; a2 = alt(a)
    altP = dict(P, text=P["text"].replace(a, a2))
    ds = rng.sample([x for x in titles if x != t], 2); paras = [P, altP] + [rng.choice(pool[x]) for x in ds]
    order = list(range(4)); rng.shuffle(order); paras = [paras[i] for i in order]
    io, ia = order.index(0), order.index(1)
    if n < N0 or altP["text"].count(a2) != 1: continue
    c = dict(paras=paras, x=0, y=0); q = f"Question: {P['q']} Answer with just the answer."
    ta, tb = tok(a, add_special_tokens=False)["input_ids"], tok(a2, add_special_tokens=False)["input_ids"]
    rec = dict(n=n, res={})
    for w, (src_k, dst_k) in (("focus", (ia, io)), ("ignore", (io, ia))):
        ids_s = encode(tok, [("user", para_doc(c, labels=True, instr=LINES[w].format(K=src_k + 1), question=q).s)], {})[0]
        ids_d = encode(tok, [("user", para_doc(c, labels=True, instr=LINES[w].format(K=dst_k + 1), question=q).s)], {})[0]
        assert len(ids_s) == len(ids_d)
        diff = [i for i in range(len(ids_s)) if ids_s[i] != ids_d[i]]; dg = diff[0]
        P_.update(grab=True, cache={}, L=None)
        with torch.no_grad(): model(torch.tensor([ids_s], device="cuda"), use_cache=False, logits_to_keep=1); P_["grab"] = False
        m_s = lp(ids_s, ta) - lp(ids_s, tb); m_d = lp(ids_d, ta) - lp(ids_d, tb)
        rec["res"][w] = dict(m_s=m_s, m_d=m_d, patch={})
        for nm, pos in (("digit", diff), ("after", list(range(dg, len(ids_s)))), ("last", [len(ids_s) - 1])):
            for L in LAYERS:
                P_.update(L=L, pos=pos); m = lp(ids_d, ta) - lp(ids_d, tb); P_["L"] = None
                rec["res"][w]["patch"][f"{nm}/{L}"] = m
    R.append(rec); torch.cuda.empty_cache()
    if len(R) % 10 == 0: torch.save(R, f'{OUT}/e8.pt')
    if n % 10 == 0: log("ctx", n, {k: round((v - rec['res']['focus']['m_d']) / (rec['res']['focus']['m_s'] - rec['res']['focus']['m_d'] + 1e-9), 2) for k, v in rec["res"]["focus"]["patch"].items() if k.startswith("last") or k.startswith("digit")})
torch.save(R, f"{OUT}/e8.pt"); log("done")
