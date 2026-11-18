# Per-token attention grid, Qwen3-32B, arithmetic task: 100 new contexts (seed 11), 13 versions each
# (none; mention/focus/ignore x named paragraph P1..P4). Context construction (rotate.py), instruction wording (every line says "paragraph"),
# eager attention, answer-token queries, head mean and layer band are those of attn_intro.py / rotate.py.
# Every version is teacher-forced on the no-instruction answer; free (greedy) answers are also saved.
import argparse, json, os, random, time
import numpy as np, torch
import transformers.models.qwen3.modeling_qwen3 as mq
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-32B")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

S = dict(record=False, rows=None, spans=None, store={}, tok={}, calls=0)
def hooked(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):  # same math as attn_intro.py, no bias
    S["calls"] += 1
    L = module.layer_idx
    k = mq.repeat_kv(key, module.num_key_value_groups)
    v = mq.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, k.transpose(2, 3)).float() * scaling
    ql, kl = w.shape[-2], w.shape[-1]
    dev = w.device
    if attention_mask is not None:
        w = w + attention_mask[:, :, :, :kl].float()
    else:
        qpos = torch.arange(kl - ql, kl, device=dev)
        w = w.masked_fill(torch.arange(kl, device=dev)[None, :] > qpos[:, None], float("-inf"))
    p = torch.softmax(w, dim=-1)
    if S["record"] and ql > 1:
        a = p[0][:, S["rows"], :].mean(0)  # head mean, [nr, kl]
        S["store"][L] = torch.stack([a[:, lo:hi].sum(-1) / (hi - lo) for (lo, hi) in S["spans"]], -1).cpu()
        S["tok"][L] = a[1:].mean(0).cpu()  # mean over answer-token queries (rows 1..), [kl]
    out = torch.matmul(p.to(value.dtype), v).transpose(1, 2).contiguous()
    return out, None
mq.eager_attention_forward = hooked

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda")
model.eval()
NL = model.config.num_hidden_layers
LAYERS = list(range(round(12 / 36 * NL), round(30 / 36 * NL) + 1))
END = tok.convert_tokens_to_ids("<|im_end|>")
log("model", args.model, "band", LAYERS[0], LAYERS[-1])

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
rng = random.Random(args.seed)
ctxs = []
for i in range(args.n):  # same per-context sampling as rotate.py; arithmetic task only
    ts = rng.sample(titles, 4)
    paras = [rng.choice(by_title[t]) for t in ts]
    j = rng.randrange(4)
    k = rng.choice([x for x in range(4) if x != j])
    ctxs.append(dict(id=i, paras=paras, j=j, k=k, task="arith", x=rng.randrange(10, 90), y=rng.randrange(10, 90)))

INSTR = dict(none="", focus="Focus your attention on paragraph [P{k}].\n",
             ignore="Do not pay attention to paragraph [P{k}].\n",
             mention="Note that one of the paragraphs is labelled [P{k}].\n")

def build(c, cond="none", kk=0):
    parts = ["Here are four paragraphs.\n\n"]
    cspans = []
    pos = len(parts[0])
    for i, p in enumerate(c["paras"]):
        lab = f"[P{i+1}] "
        s = pos + len(lab)
        cspans.append((s, s + len(p["text"])))
        seg = lab + p["text"] + "\n\n"
        parts.append(seg); pos += len(seg)
    t = INSTR[cond].format(k=kk + 1) + f"Unrelated question: what is {c['x']} + {c['y']}? Answer with just the number."
    user = "".join(parts) + t
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    off = prompt.index(user)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, om = enc["input_ids"], enc["offset_mapping"]
    def tspan(cs, ce):
        cs += off; ce += off
        idx = [i for i, (a, b) in enumerate(om) if a >= cs and b <= ce and b > a]
        return (idx[0], idx[-1] + 1)
    spans = [tspan(a, b) for a, b in cspans]
    return ids, spans, user, om, prompt

@torch.no_grad()
def generate(ids):
    g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=80, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    a = g[0, len(ids):].tolist()
    while a and a[-1] in (END, tok.eos_token_id): a = a[:-1]
    return a

@torch.no_grad()
def measure(ids, ans, spans):
    S.update(record=True, rows=list(range(len(ids) - 1, len(ids) + len(ans))), spans=spans, store={}, tok={})
    model(torch.tensor([ids + ans], device="cuda"), use_cache=False, logits_to_keep=1)
    S["record"] = False
    span_band = torch.stack([S["store"][L] for L in LAYERS])[:, 1:, :].mean(1).mean(0).numpy()  # [4], as band(att_mean)
    tokband = torch.stack([S["tok"][L] for L in LAYERS]).mean(0).numpy()
    return span_band, [tokband[lo:hi].tolist() for (lo, hi) in spans]

VERS = [("none", 0)] + [(cond, kk) for kk in range(4) for cond in ("mention", "focus", "ignore")]
recs = []
for c in ctxs:
    ids0, spans0, user0, om0, prompt0 = build(c)
    ans0 = generate(ids0)
    rec = dict(id=c["id"], x=c["x"], y=c["y"], ans_none=tok.decode(ans0, skip_special_tokens=True), ans_none_ids=ans0,
               tokens=[[prompt0[om0[t][0]:om0[t][1]] for t in range(lo, hi)] for (lo, hi) in spans0],
               paras=[p["text"] for p in c["paras"]], vers={})
    for cond, kk in VERS:
        ids, spans, user, om, prompt = build(c, cond, kk)
        toks = [[prompt[om[t][0]:om[t][1]] for t in range(lo, hi)] for (lo, hi) in spans]
        assert toks == rec["tokens"], "paragraph tokenization differs between versions"
        free = ans0 if cond == "none" else generate(ids)
        sb, tk = measure(ids, ans0, spans)
        rec["vers"][f"{cond}{kk + 1}" if cond != "none" else "none"] = dict(
            span_band=sb.tolist(), tok=tk, free=tok.decode(free, skip_special_tokens=True), free_same=free == ans0, user=user)
    recs.append(rec)
    log("ctx", c["id"], repr(rec["ans_none"]), "free same", sum(v["free_same"] for v in rec["vers"].values()), "/ 13")
    if len(recs) % 10 == 0: json.dump(recs, open(os.path.join(args.out, "G.json"), "w"))
assert S["calls"] > 0
json.dump(recs, open(os.path.join(args.out, "G.json"), "w"))
json.dump(dict(model=args.model, NL=NL, LAYERS=LAYERS, n=args.n, seed=args.seed), open(os.path.join(args.out, "meta.json"), "w"))
log("done")
