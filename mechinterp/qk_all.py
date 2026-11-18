# Shared setup for the mechinterp study of "focus on / ignore paragraph [Pk]" in Qwen3-32B.
# Prompt builder copied from dose/dose.py (same SQuAD filter, same seed-7 generator, arithmetic block).
import random, time
import torch
import transformers.models.qwen3.modeling_qwen3 as mq
from transformers import AutoTokenizer, AutoModelForCausalLM

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

# S: global state the patched attention reads.
#   rec: record per-head attention for rows (query positions) to spans
#   headcache: {L: positions} -> store per-head outputs (before o_proj) at those positions
#   headpatch: {L: [(b, heads, positions, tensor[len(heads), len(pos), d])]}
#   headzero:  {L: [(b, heads, positions)]}  -> zero those head outputs (ablation)
S = dict(rec=False, rows=None, spans=None, store={}, logit={}, calls=0, headcache=None, hc={}, headpatch=None, headzero=None,
         respatch=None, rescache=None, rc={})

def hooked(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    S["calls"] += 1
    L = module.layer_idx
    k = mq.repeat_kv(key, module.num_key_value_groups); v = mq.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, k.transpose(2, 3)).float() * scaling
    ql, kl = w.shape[-2], w.shape[-1]; dev = w.device
    if attention_mask is not None:
        w = w + attention_mask[:, :, :, :kl].float()
    else:
        qpos = torch.arange(kl - ql, kl, device=dev)
        w = w.masked_fill(torch.arange(kl, device=dev)[None, :] > qpos[:, None], float("-inf"))
    p = torch.softmax(w, dim=-1)
    if S["rec"] and ql > 1:
        rows = S["rows"]
        pr = p[:, :, rows, :]; wr = w[:, :, rows, :]
        # mean attention per span token (sum over span / span length), mean over rows -> [B, H, nspans]
        S["store"][L] = torch.stack([pr[..., lo:hi].sum(-1).mean(-1) / (hi - lo) for (lo, hi) in S["spans"]], -1).cpu()
        S["logit"][L] = torch.stack([wr[..., lo:hi].mean(-1).mean(-1) for (lo, hi) in S["spans"]], -1).cpu()
    o = torch.matmul(p.to(value.dtype), v)  # [B, H, q, d]
    if S["headcache"] is not None and L in S["headcache"]:
        S["hc"][L] = o[:, :, S["headcache"][L], :].detach().clone()
    if S["headpatch"] is not None and L in S["headpatch"]:
        o = o.clone()
        for b, heads, pos, t in S["headpatch"][L]:
            hi = torch.tensor(heads, device=dev); pi = torch.tensor(pos, device=dev)
            o[b, hi[:, None], pi[None, :], :] = t.to(o.dtype)
    if S["headzero"] is not None and L in S["headzero"]:
        o = o.clone()
        for b, heads, pos in S["headzero"][L]:
            hi = torch.tensor(heads, device=dev); pi = torch.tensor(pos, device=dev)
            o[b, hi[:, None], pi[None, :], :] = 0
    return o.transpose(1, 2).contiguous(), None
mq.eager_attention_forward = hooked

def layer_hook(L):
    def f(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if S["rescache"] is not None:
            S["rc"][L] = h[:, S["rescache"], :].detach().clone()
        if S["respatch"] is not None and L in S["respatch"]:
            h = h.clone()
            for b, pos, t in S["respatch"][L]:
                h[b, torch.tensor(pos, device=h.device), :] = t.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return f

def load(name="Qwen/Qwen3-32B"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda").eval()
    for L, layer in enumerate(model.model.layers):
        layer.register_forward_hook(layer_hook(L))
    return tok, model

def contexts(tok, n=150, seed=7):
    from datasets import load_dataset
    sq = load_dataset("rajpurkar/squad", split="train")
    by_title, seen = {}, set()
    for r in sq:
        c = r["context"]
        if c in seen: continue
        seen.add(c)
        nt = len(tok(c)["input_ids"])
        if 60 <= nt <= 110:
            by_title.setdefault(r["title"], []).append(dict(text=c, q=r["question"], ans=r["answers"]["text"][0], ntok=nt))
    titles = sorted(by_title)
    rng = random.Random(seed)
    ctxs = []
    for i in range(3 * n):
        ts = rng.sample(titles, 4)
        paras = [rng.choice(by_title[t]) for t in ts]
        j = rng.randrange(4); k = rng.choice([x for x in range(4) if x != j])
        task = ["qa", "sum", "arith"][i // n]
        ctxs.append(dict(id=i, paras=paras, j=j, k=k, task=task, x=rng.randrange(10, 90), y=rng.randrange(10, 90)))
    return [c for c in ctxs if c["task"] == "arith"]

LINES = {
    "none": "",
    "mention": "Note that one of the paragraphs is labelled [P{k}].",
    "focus": "Focus your attention on paragraph [P{k}].",
    "ignore": "Completely ignore paragraph [P{k}]. Do not look at it at all.",
}

def build(tok, c, cond, k, perm=(0, 1, 2, 3)):
    """k is 0-based (the label named). perm[i] = label index printed on the i-th paragraph. Returns ids and token spans."""
    mid = LINES[cond].format(k=k + 1)
    parts = ["Here are four paragraphs.\n\n"]; cspans, lspans = [], []; pos = len(parts[0])
    for i, p in enumerate(c["paras"]):
        lab = f"[P{perm[i]+1}] "; lspans.append((pos, pos + len(lab) - 1)); s = pos + len(lab); cspans.append((s, s + len(p["text"])))
        seg = lab + p["text"] + "\n\n"; parts.append(seg); pos += len(seg)
    istart = pos
    q = f"Unrelated question: what is {c['x']} + {c['y']}? Answer with just the number."
    t = (mid + "\n" if mid else "") + q
    user = "".join(parts) + t
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    off = prompt.index(user)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True); ids, om = enc["input_ids"], enc["offset_mapping"]
    def tspan(cs, ce, loose=False):
        cs += off; ce += off
        if loose: idx = [i for i, (a, b) in enumerate(om) if b > cs and a < ce and b > a]
        else: idx = [i for i, (a, b) in enumerate(om) if a >= cs and b <= ce and b > a]
        return (idx[0], idx[-1] + 1)
    info = dict(paras=[tspan(a, b) for a, b in cspans], labels=[tspan(a, b, True) for a, b in lspans])
    if mid:
        info["instr"] = tspan(istart, istart + len(mid))
        dc = istart + mid.index(f"[P{k+1}]") + 2  # char index of the digit
        info["digit"] = [i for i, (a, b) in enumerate(om) if a <= dc + off < b][0]
        info["after"] = info["instr"][1]  # first token after the instruction line
    else:
        info["after"] = tspan(istart, istart + len(q))[0]
    return ids, info

END = None
@torch.no_grad()
def generate(tok, model, ids, maxnew=20):
    reset()
    g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=maxnew, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    a = g[0, len(ids):].tolist()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    while a and a[-1] in (end, tok.eos_token_id): a = a[:-1]
    return a

def reset():
    S.update(rec=False, rows=None, spans=None, store={}, logit={}, headcache=None, hc={}, headpatch=None, headzero=None,
             respatch=None, rescache=None, rc={})

# ---- qk.py ----
# Query-side vs key-side swap, every head.
# Base run names [Pa], donor run names [Pb]; the two token sequences differ only in the instruction digit.
# At every layer and head, attention from the answer tokens to each paragraph is recomputed four ways:
#   base (q_a, k_a), query swap (q_b, k_a), key swap (q_a, k_b), donor (q_b, k_b).
# Paragraph keys are identical in both runs (they come before the instruction), so a key swap can only act
# through the keys of the tokens after the digit (the softmax denominator). The forward pass itself is unchanged.
import sys, os, torch
pass  # common.py is inlined above
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/out"
os.makedirs(OUT, exist_ok=True)
tok, model = load(); NL = 64
Q = dict(mode=None, cache={}, out={}, rows=None, spans=None)

def span_att(q, k, rows, spans, scaling):  # q [H, r, d], k [H, T, d] -> [H, nspans]
    w = torch.matmul(q.float(), k.float().transpose(1, 2)) * scaling
    T = k.shape[1]; r = torch.tensor(rows, device=w.device)
    w = w.masked_fill(torch.arange(T, device=w.device)[None, :] > r[:, None], float("-inf"))
    p = torch.softmax(w, -1)
    return torch.stack([p[..., lo:hi].sum(-1).mean(-1) / (hi - lo) for lo, hi in spans], -1)

orig = hooked
def wrap(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    L = module.layer_idx
    if Q["mode"] is not None and query.shape[-2] > 1:
        k = mq.repeat_kv(key, module.num_key_value_groups)[0]; q = query[0, :, Q["rows"], :]
        if Q["mode"] == "cache":
            Q["cache"][L] = (q.clone(), k.clone())
        else:
            qb, kb = Q["cache"][L]; R, S_ = Q["rows"], Q["spans"]
            Q["out"][L] = torch.stack([span_att(q, k, R, S_, scaling), span_att(qb, k, R, S_, scaling),
                                       span_att(q, kb, R, S_, scaling), span_att(qb, kb, R, S_, scaling)], 1).half().cpu()
    return orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
mq.eager_attention_forward = wrap

def fwd(ids, mode):
    reset(); Q["mode"] = mode
    with torch.no_grad(): model(torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
    Q["mode"] = None

R = []
for n, c in enumerate(contexts(tok)[:N]):
    ans = generate(tok, model, build(tok, c, "none", c["k"])[0])
    a = c["k"]; b = (a + 1 + n % 3) % 4
    for cond in ["mention", "focus", "ignore"]:
        idsA, infA = build(tok, c, cond, a); idsB, _ = build(tok, c, cond, b)
        idsA += ans; idsB += ans; T = len(idsA)
        assert len(idsB) == T and sum(x != y for x, y in zip(idsA, idsB)) == 1
        Q["rows"] = list(range(T - len(ans), T)); Q["spans"] = infA["paras"]; Q["cache"] = {}; Q["out"] = {}
        fwd(idsB, "cache"); fwd(idsA, "swap")
        R.append(dict(id=c["id"], cond=cond, a=a, b=b, att=torch.stack([Q["out"][L] for L in range(NL)])))  # [NL, H, 4 variants, 4 paras]
    log("ctx", n + 1)
    if (n + 1) % 10 == 0: torch.save(R, f"{OUT}/qk.pt")
torch.save(R, f"{OUT}/qk.pt")
log("done")
