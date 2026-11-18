# Dose-response of instruction wording on native attention (Qwen3-32B, arithmetic task).
# Contexts built exactly as attn_intro.py (same SQuAD filter, same rng sequence), new seed; arithmetic block only.
# Every condition is teacher-forced on the no-instruction answer, so all conditions score the same answer tokens.
import argparse, json, os, random, time
import numpy as np, torch
import transformers.models.qwen3.modeling_qwen3 as mq
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-32B")
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True); OUT = args.out
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

S = dict(record=False, rows=None, spans=None, store={}, calls=0)
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
    if S["record"] and ql > 1:
        a = p[0][:, S["rows"], :].mean(0)
        S["store"][L] = torch.stack([a[:, lo:hi].sum(-1) / (hi - lo) for (lo, hi) in S["spans"]], -1).cpu()
    return torch.matmul(p.to(value.dtype), v).transpose(1, 2).contiguous(), None
mq.eager_attention_forward = hooked

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda").eval()
NL = model.config.num_hidden_layers
LAYERS = list(range(round(12 / 36 * NL), round(30 / 36 * NL) + 1))
log("model", args.model, "layers", NL, "band", LAYERS[0], LAYERS[-1])
END = tok.convert_tokens_to_ids("<|im_end|>")

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
for i in range(3 * args.n):  # same generator as attn_intro.py; keep the arithmetic block
    ts = rng.sample(titles, 4)
    paras = [rng.choice(by_title[t]) for t in ts]
    j = rng.randrange(4); k = rng.choice([x for x in range(4) if x != j])
    task = ["qa", "sum", "arith"][i // args.n]
    ctxs.append(dict(id=i, paras=paras, j=j, k=k, task=task, x=rng.randrange(10, 90), y=rng.randrange(10, 90)))
ctxs = [c for c in ctxs if c["task"] == "arith"]
log("arith contexts", len(ctxs))

V5 = "Focus very strongly and only on paragraph [P{k}]. Ignore everything else."
CONDS = {  # name: (line before the question (B's position), line before the paragraphs, line after the question)
    "0_none": ("", "", ""),
    "1_mention": ("Note that one of the paragraphs is labelled [P{k}].", "", ""),
    "2_glance": ("Glance at paragraph [P{k}].", "", ""),
    "3_focus": ("Focus your attention on paragraph [P{k}].", "", ""),
    "4_focus_strongly": ("Focus strongly on paragraph [P{k}].", "", ""),
    "5_very_strongly_only": (V5, "", ""),
    "6_twice": ("", V5, V5),
    "7_little_less": ("Pay a little less attention to paragraph [P{k}].", "", ""),
    "8_do_not": ("Do not pay attention to paragraph [P{k}].", "", ""),
    "9_completely_ignore": ("Completely ignore paragraph [P{k}]. Do not look at it at all.", "", ""),
}

def build(c, cond):
    mid, pre, post = (s.format(k=c["k"] + 1) for s in CONDS[cond])
    head = (pre + "\n") if pre else ""
    parts = [head + "Here are four paragraphs.\n\n"]; cspans = []; pos = len(parts[0])
    for i, p in enumerate(c["paras"]):
        lab = f"[P{i+1}] "; s = pos + len(lab); cspans.append((s, s + len(p["text"])))
        seg = lab + p["text"] + "\n\n"; parts.append(seg); pos += len(seg)
    t = (mid + "\n" if mid else "") + f"Unrelated question: what is {c['x']} + {c['y']}? Answer with just the number." + ("\n" + post if post else "")
    user = "".join(parts) + t
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    off = prompt.index(user)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True); ids, om = enc["input_ids"], enc["offset_mapping"]
    def tspan(cs, ce):
        cs += off; ce += off
        idx = [i for i, (a, b) in enumerate(om) if a >= cs and b <= ce and b > a]; return (idx[0], idx[-1] + 1)
    spans = [tspan(a, b) for a, b in cspans]
    return ids, spans, user

@torch.no_grad()
def generate(ids, maxnew=80):
    g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=maxnew, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    a = g[0, len(ids):].tolist()
    while a and a[-1] in (END, tok.eos_token_id): a = a[:-1]
    return a

@torch.no_grad()
def measure(ids, ans, spans):
    rows = list(range(len(ids) - 1, len(ids) + len(ans)))
    S.update(record=True, rows=rows, spans=spans, store={})
    model(torch.tensor([ids + ans], device="cuda"), use_cache=False, logits_to_keep=1)
    S["record"] = False
    st = torch.stack([S["store"][L] for L in range(NL)])  # [NL, nr, 4]
    return st[:, 0, :].tolist(), st[:, 1:, :].mean(1).tolist()

recs = []
for n, c in enumerate(ctxs):
    ids0, sp0, _ = build(c, "0_none")
    ans0 = generate(ids0)
    a0 = tok.decode(ans0, skip_special_tokens=True)
    for cond in CONDS:
        ids, spans, user = build(c, cond)
        assert all(ids[a:b] == ids0[a0:b0] for (a, b), (a0, b0) in zip(spans, sp0)), "paragraph tokens differ"
        first, mean = measure(ids, ans0, spans)
        free = a0 if cond == "0_none" else tok.decode(generate(ids), skip_special_tokens=True)
        recs.append(dict(id=c["id"], cond=cond, k=c["k"], j=c["j"], ans_forced=a0, ans_free=free, free_same=(free.strip() == a0.strip()),
                         gold=str(c["x"] + c["y"]), ntok=[b - a for a, b in spans], att_first=first, att_mean=mean,
                         user=user if n < 3 else None))
    if (n + 1) % 25 == 0:
        log("ctx", n + 1); json.dump(recs, open(os.path.join(OUT, "dose.json"), "w"))
assert S["calls"] > 0
json.dump(recs, open(os.path.join(OUT, "dose.json"), "w"))
json.dump(dict(model=args.model, NL=NL, LAYERS=LAYERS, n=len(ctxs), seed=args.seed, conds={k: list(v) for k, v in CONDS.items()}), open(os.path.join(OUT, "meta.json"), "w"))
for r in recs[:10]: print(r["cond"], "|", r["ans_forced"], "|", r["ans_free"])
log("done")
