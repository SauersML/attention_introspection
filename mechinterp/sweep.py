# Step 1: per-head attention and residual snapshots, every context run with every k (0..3) and 4 conditions.
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/out"
os.makedirs(OUT, exist_ok=True)
tok, model = load()
NL = model.config.num_hidden_layers
ctxs = contexts(tok)[:N]
CONDS = ["none", "mention", "focus", "ignore"]
log("contexts", len(ctxs))
ATT, LOG, RES, PARA, META, JL = [], [], [], [], [], []
for n, c in enumerate(ctxs):
    ids0, inf0 = build(tok, c, "none", c["k"])
    ans = generate(tok, model, ids0)
    for cond in CONDS:
        batch = [build(tok, c, cond, k) for k in range(4)]
        L0 = len(batch[0][0]); assert all(len(b[0]) == L0 for b in batch)
        x = torch.tensor([b[0] + ans for b in batch], device="cuda")
        inf = batch[0][1]
        spans = list(inf["paras"]) + list(inf["labels"])
        if cond != "none":
            spans += [inf["instr"], (inf["digit"], inf["digit"] + 1)]
        else:
            spans += [(0, 1), (0, 1)]
        rows = list(range(L0, L0 + len(ans)))
        pos = [inf.get("digit", 0), inf["instr"][1] - 1 if cond != "none" else 0, L0 - 1]
        reset(); S.update(rec=True, rows=rows, spans=spans)
        S["rescache"] = pos + rows
        para_pos = None
        if cond == "none":
            para_pos = sum([list(range(a, b)) for a, b in inf["paras"]], []) + [b - 1 for a, b in inf["labels"]]
            S["rescache"] = pos + rows + para_pos
        jl = n < 10
        if jl:
            S["rescache"] = list(range(L0 + len(ans)))
        with torch.no_grad():
            model(x, use_cache=False, logits_to_keep=1)
        ATT.append(torch.stack([S["store"][L] for L in range(NL)]).half())   # [NL, 4(k), H, nspans]
        LOG.append(torch.stack([S["logit"][L] for L in range(NL)]).half())
        rc = torch.stack([S["rc"][L] for L in range(NL)])  # [NL, 4, npos, d]
        if jl:
            JL.append(dict(id=c["id"], cond=cond, ids=batch[c["k"]][0] + ans, info=batch[c["k"]][1],
                           res=rc[:, c["k"]].half().cpu()))
            sel = pos + rows + (para_pos or [])
            rc = rc[:, :, sel]
        r = torch.cat([rc[:, :, :3], rc[:, :, 3:3 + len(rows)].float().mean(2, keepdim=True).to(rc.dtype)], 2)  # digit, instr_end, last_prompt, answer_mean
        RES.append(r.half().cpu())
        if cond == "none":
            pr = rc[:, 0, 3 + len(rows):]
            pm, i0 = [], 0
            for a, b in inf["paras"]:
                pm.append(pr[:, i0:i0 + b - a].float().mean(1)); i0 += b - a
            lab = pr[:, i0:i0 + 4]
            PARA.append(torch.cat([torch.stack(pm, 1), lab.float()], 1).half().cpu())  # [NL, 8, d]: 4 para means, 4 label ends
        META.append(dict(id=c["id"], cond=cond, ckey=c["k"], L0=L0, nans=len(ans), ans=tok.decode(ans), spans=spans))
    log("ctx", n + 1, tok.decode(ans))
torch.save(dict(att=torch.stack(ATT), logit=torch.stack(LOG), res=torch.stack(RES), para=torch.stack(PARA), meta=META), f"{OUT}/sweep.pt")
torch.save(JL, f"{OUT}/jl_examples.pt")
log("done", S["calls"])
