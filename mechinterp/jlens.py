# Step 4: Jacobian lens (Gurnee et al. 2026; Neuronpedia's fitted Qwen3-32B lens) vs plain logit lens
# at the instruction tokens, the named paragraph's label, and the answer positions.
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from huggingface_hub import hf_hub_download
from safetensors import safe_open
OUT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/out"
NAME = "Qwen/Qwen3-32B"
tok = AutoTokenizer.from_pretrained(NAME)
p = hf_hub_download("neuronpedia/jacobian-lens", "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt")
ck = torch.load(p, map_location="cpu", weights_only=True)
J = {l: v.float().cuda() for l, v in ck["J"].items()}
log("lens layers", min(J), max(J), "n_prompts", ck["n_prompts"])
# final norm + unembedding straight from the checkpoint
from huggingface_hub import snapshot_download
sd = snapshot_download(NAME, allow_patterns=["*.json", "*.safetensors"])
idx = json.load(open(f"{sd}/model.safetensors.index.json"))["weight_map"]
def W(n):
    with safe_open(f"{sd}/{idx[n]}", "pt", device="cuda") as f: return f.get_tensor(n)
U = W("lm_head.weight").float(); g = W("model.norm.weight").float()
cfg = json.load(open(f"{sd}/config.json")); eps = cfg["rms_norm_eps"]
def lens(h, L, jac=True):  # h [n, d] -> logits [n, V]
    h = h.float().cuda()
    if jac and L in J: h = h @ J[L].T
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * g
    return h @ U.T

def top(lg, n=8): return [[tok.decode([i]) for i in row] for row in lg.topk(n, -1).indices.tolist()]
LAY = list(range(0, 64, 4)) + [62, 63]
# (a) examples: top tokens per layer at chosen positions
EX = torch.load(f"{OUT}/jl_examples.pt", weights_only=False)
ex_out = []
for e in EX:
    inf, ids = e["info"], e["ids"]
    pos = {}
    if "instr" in inf:
        for i in range(*inf["instr"]): pos[f"instr:{i}:{tok.decode([ids[i]])}"] = i
    k = None
    T = len(ids)
    for i in range(inf["after"], T): pos[f"post:{i}:{tok.decode([ids[i]])}"] = i
    rec = {}
    for name, i in pos.items():
        rec[name] = {L: dict(j=top(lens(e["res"][L, i:i + 1], L))[0], l=top(lens(e["res"][L, i:i + 1], L, False))[0]) for L in LAY}
    ex_out.append(dict(id=e["id"], cond=e["cond"], rec=rec))
json.dump(ex_out, open(f"{OUT}/jlens_examples.json", "w"))
log("examples done")
# (b) statistics over all contexts: at the last prompt token and the answer mean, how strongly does each lens
#     read out (1) the named paragraph's digit vs the other digits, (2) the named paragraph's own words vs the
#     other paragraphs' words?
d = torch.load(f"{OUT}/sweep.pt", weights_only=False)
res = d["res"]; nC = res.shape[0] // 4
res = res.view(nC, 4, 64, 4, 4, -1)
ctxs = contexts(tok)[:nC]
digs = [tok.convert_tokens_to_ids(str(i)) for i in range(1, 5)]
STOP = set(tok(" the a an of in and to is was for on by with as at from that which it its were be are or his her their this has had have not but").input_ids)
def ptoks(c):
    sets = [set(tok(p["text"]).input_ids) | set(tok(" " + p["text"]).input_ids) for p in c["paras"]]
    out = []
    for i, s in enumerate(sets):
        o = set().union(*[sets[j] for j in range(4) if j != i])
        out.append(sorted(t for t in s - o - STOP if len(tok.decode([t]).strip()) >= 4))
    return out
CONDS = ["none", "mention", "focus", "ignore"]
stat = {}
for jac in [True, False]:
    for pi, pn in [(2, "last_prompt"), (3, "answer_mean")]:
        D = torch.zeros(nC, 4, 64); Wd = torch.zeros(nC, 4, 64)
        for n, c in enumerate(ctxs):
            pt = ptoks(c)
            for ci in range(4):
                for L in range(64):
                    lg = lens(res[n, ci, L, :, pi], L, jac)  # [4 k, V]
                    lp = lg.log_softmax(-1)
                    dg = lp[:, digs]  # [k, 4]
                    D[n, ci, L] = torch.stack([dg[k, k] - (dg[k].sum() - dg[k, k]) / 3 for k in range(4)]).mean().cpu()
                    wm = torch.stack([lp[:, pt[i]].mean(-1) for i in range(4)], -1)  # [k, para]
                    Wd[n, ci, L] = torch.stack([wm[k, k] - (wm[k].sum() - wm[k, k]) / 3 for k in range(4)]).mean().cpu()
        stat[f"{'J' if jac else 'logit'}/{pn}"] = dict(digit={c: D[:, i].mean(0).tolist() for i, c in enumerate(CONDS)},
                                                   digit_se={c: (D[:, i].std(0) / nC ** .5).tolist() for i, c in enumerate(CONDS)},
                                                   words={c: Wd[:, i].mean(0).tolist() for i, c in enumerate(CONDS)},
                                                   words_se={c: (Wd[:, i].std(0) / nC ** .5).tolist() for i, c in enumerate(CONDS)})
        log("stat", jac, pn)
json.dump(stat, open(f"{OUT}/jlens_stats.json", "w"))
log("done")
