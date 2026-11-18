# Per-token attention for one arithmetic context (focus vs mention), Qwen3-32B.
# Rebuilds the context exactly as attn_intro.py does (same SQuAD filter, seed 0, n 200),
# teacher-forces the recorded answer, and saves head-averaged attention from the answer-token
# queries to every prompt token, averaged over answer tokens and the layer band.
import argparse, json, os, random
import numpy as np, torch
import transformers.models.qwen3.modeling_qwen3 as mq
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-32B")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ids", required=True)  # comma list
ap.add_argument("--conds", default="focus,mention,ignore")
ap.add_argument("--ref", required=True)  # B.json records of these ids
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

S = dict(record=False, rows=None, spans=None, store={}, tok={})

def hooked(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):  # same math as attn_intro.py, no bias
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
        a = p[0][:, S["rows"], :].mean(0)  # head-avg, [nr, kl]
        per = torch.stack([a[:, lo:hi].sum(-1) / (hi - lo) for (lo, hi) in S["spans"]], -1)
        S["store"][L] = per.cpu()
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
for i in range(3 * args.n):
    ts = rng.sample(titles, 4)
    paras = [rng.choice(by_title[t]) for t in ts]
    j = rng.randrange(4)
    k = rng.choice([x for x in range(4) if x != j])
    task = ["qa", "sum", "arith"][i // args.n]
    ctxs.append(dict(id=i, paras=paras, j=j, k=k, task=task, x=rng.randrange(10, 90), y=rng.randrange(10, 90)))

REPORTQ = "Which paragraph were you paying the most attention to just now, while writing your answer? Answer with just P1, P2, P3, or P4."
INSTR = dict(none="", focus="While you answer, focus your attention on [P{k}].\n",
             ignore="While you answer, do not pay attention to [P{k}].\n",
             mention="Note that one of the paragraphs is labelled [P{k}].\n")

def build(c, cond="none"):
    parts = ["Here are four paragraphs.\n\n"]
    cspans = []
    pos = len(parts[0])
    for i, p in enumerate(c["paras"]):
        lab = f"[P{i+1}] "
        s = pos + len(lab)
        cspans.append((s, s + len(p["text"])))
        seg = lab + p["text"] + "\n\n"
        parts.append(seg); pos += len(seg)
    task_c = pos
    instr = INSTR[cond].format(k=c["k"] + 1)
    if c["task"] == "qa":
        t = instr + "Answer briefly. Question: " + c["paras"][c["j"]]["q"]
    elif c["task"] == "sum":
        t = instr + "Summarize the four paragraphs in one sentence."
    else:
        t = instr + f"Unrelated question: what is {c['x']} + {c['y']}? Answer with just the number."
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

SUFFIX = tok("<|im_end|>\n<|im_start|>user\n" + REPORTQ + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nP", add_special_tokens=False)["input_ids"]
allrefs = json.load(open(args.ref))
for cid in [int(x) for x in args.ids.split(",")]:
  refs = {r["cond"]: r for r in allrefs if r["id"] == cid}
  c = ctxs[cid]
  assert c["task"] == "arith" and c["k"] == refs["focus"]["k"]
  out = dict(id=cid, model=args.model, LAYERS=LAYERS, k=c["k"], conds={})
  for cond in args.conds.split(","):
      ref = refs[cond]
      ids, spans, user, om, prompt = build(c, cond)
      assert user == ref["user"], "rebuilt prompt differs from B.json"
      with torch.no_grad():
          g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=80, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
      gen = g[0, len(ids):].tolist()
      while gen and gen[-1] in (END, tok.eos_token_id): gen = gen[:-1]
      ans = tok(ref["ans"], add_special_tokens=False)["input_ids"]  # teacher-force the recorded answer
      seq = ids + ans + SUFFIX
      S.update(record=True, rows=list(range(len(ids) - 1, len(ids) + len(ans))), spans=spans, store={}, tok={})
      with torch.no_grad():
          model(torch.tensor([seq], device="cuda"), use_cache=False, logits_to_keep=1)
      S["record"] = False
      att_mean = torch.stack([S["store"][L] for L in range(NL)])[:, 1:, :].mean(1).numpy()  # [NL,4]
      tokband = torch.stack([S["tok"][L] for L in LAYERS]).mean(0).numpy()  # [kl]
      paras = []
      for pi, (lo, hi) in enumerate(spans):
          toks = [prompt[om[t][0]:om[t][1]] for t in range(lo, hi)]
          paras.append(dict(span=[lo, hi], tokens=toks, att=tokband[lo:hi].tolist(),
                            mean_from_tokens=float(tokband[lo:hi].mean()),
                            band_att_mean_rerun=float(att_mean[LAYERS, pi].mean()),
                            band_att_mean_Bjson=float(np.array(ref["att_mean"])[LAYERS, pi].mean())))
      out["conds"][cond] = dict(user_matches=True, recorded_ans=ref["ans"], generated_ans=tok.decode(gen, skip_special_tokens=True),
                                ans_ids=ans, generated_ids=gen, paras=paras,
                                att_mean_rerun=att_mean.tolist())
      print(cond, "gen", repr(out["conds"][cond]["generated_ans"]), "rec", repr(ref["ans"]), "ids", ans, gen, flush=True)
      for pi, p in enumerate(paras):
          print(f"  P{pi+1} tokmean {p['mean_from_tokens']:.6e} rerun {p['band_att_mean_rerun']:.6e} Bjson {p['band_att_mean_Bjson']:.6e}", flush=True)
  json.dump(out, open(os.path.join(args.out, f"tokens_ctx{cid}.json"), "w"))
  print("saved", cid, flush=True)
print("done", flush=True)
