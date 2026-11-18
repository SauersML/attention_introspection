# Step 3: residual-stream activation patching.
# (a) k-swap: base "cond, k=a" vs donor "cond, k=b" (identical length, only the instruction digit differs).
#     Patch the donor residual (output of layer L) at a position set into the base run; record per-layer
#     head-mean attention from answer tokens to each paragraph.
# (b) wording: base "mention, k" <- donor "focus, k" / "ignore, k", positions after the instruction line, aligned from the end.
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/out"
LAYERS = list(range(0, 64, 3)) + [63]
tok, model = load()
NL = model.config.num_hidden_layers
ctxs = contexts(tok)[:N]
BS = 8

def run(ids, rows, spans, patches=None, cache=False):
    """patches: list (one per batch element) of (L, pos, tensor) or None. Returns att [B, NL, 4] (head mean)."""
    B = len(patches) if patches else 1
    reset(); S.update(rec=True, rows=rows, spans=spans)
    if cache: S["rescache"] = list(range(len(ids)))
    if patches:
        rp = {}
        for b, pt in enumerate(patches):
            if pt is None: continue
            L, pos, t = pt; rp.setdefault(L, []).append((b, pos, t))
        S["respatch"] = rp
    with torch.no_grad():
        model(torch.tensor([ids] * B, device="cuda"), use_cache=False, logits_to_keep=1)
    att = torch.stack([S["store"][L][:, :, :4].float().mean(1) for L in range(NL)], 1)  # [B, NL, 4]
    rc = torch.stack([S["rc"][L][0] for L in range(NL)]) if cache else None  # [NL, T, d]
    return att, rc

def batched(ids, rows, spans, specs):
    out = []
    for i in range(0, len(specs), BS):
        a, _ = run(ids, rows, spans, specs[i:i + BS]); out.append(a)
    return torch.cat(out)

R = []
for n, c in enumerate(ctxs):
    ids0, _ = build(tok, c, "none", c["k"])
    ans = generate(tok, model, ids0)
    a = c["k"]; b = (a + 1 + n % 3) % 4
    for cond in ["focus", "ignore"]:
        idsA, infA = build(tok, c, cond, a); idsB, infB = build(tok, c, cond, b)
        idsA += ans; idsB += ans; T = len(idsA); L0 = T - len(ans)
        assert len(idsA) == len(idsB) and sum(x != y for x, y in zip(idsA, idsB)) == 1
        rows = list(range(L0, T)); spans = infA["paras"]
        base, _ = run(idsA, rows, spans)
        donor, rcB = run(idsB, rows, spans, cache=True)
        sets = dict(digit=[infA["digit"]], instr=list(range(*infA["instr"])), after=list(range(infA["after"], T)), answer=list(range(L0 - 1, T)))
        specs, keys = [], []
        for sname, pos in sets.items():
            for L in LAYERS:
                specs.append((L, pos, rcB[L, pos])); keys.append((sname, L))
        att = batched(idsA, rows, spans, specs)
        R.append(dict(kind="kswap", id=c["id"], cond=cond, a=a, b=b, base=base[0], donor=donor[0],
                      patched={k: att[i] for i, k in enumerate(keys)}))
    # (b) wording, aligned from the end
    idsM, infM = build(tok, c, "mention", a); idsM += ans; TM = len(idsM); L0M = TM - len(ans)
    rowsM = list(range(L0M, TM)); spans = infM["paras"]
    baseM, _ = run(idsM, rowsM, spans)
    for cond in ["focus", "ignore"]:
        idsD, infD = build(tok, c, cond, a); idsD += ans; TD = len(idsD)
        nA = TM - infM["after"] - 1; assert idsM[TM - nA:] == idsD[TD - nA:], "suffix differs"
        donor, rcD = run(idsD, list(range(TD - len(ans), TD)), infD["paras"], cache=True)
        specs, keys = [], []
        for sname, n_ in [("after", nA), ("answer", len(ans) + 1)]:
            for L in LAYERS:
                specs.append((L, list(range(TM - n_, TM)), rcD[L, TD - n_:TD])); keys.append((sname, L))
        att = batched(idsM, rowsM, spans, specs)
        R.append(dict(kind="wording", id=c["id"], cond=cond, a=a, base=baseM[0], donor=donor[0],
                      patched={k: att[i] for i, k in enumerate(keys)}))
    log("ctx", n + 1)
    if (n + 1) % 10 == 0: torch.save(dict(R=R, LAYERS=LAYERS), f"{OUT}/patch.pt")
torch.save(dict(R=R, LAYERS=LAYERS), f"{OUT}/patch.pt")
log("done")
