# Told-injection experiment (Qwen3-32B), after Lindsey's concept-injection design but for attention:
# the prompt says honestly that we may boost the model's attention onto one of four paragraphs while it reads the
# question (and answers); on boosted trials we add log(multiplier) to the attention scores of that paragraph's
# body tokens, for queries from "Trial 1" onward (question and answer tokens), layers 21-53, all heads.
# Control trials: no boost. Scored from first-token log-probs ("Yes" vs "No"), from the paragraph digit after a
# prefilled "Yes, paragraph [P", and from the greedy text (boost kept on while generating, as in the paper).
import argparse, json, math, os, random, re, time
import numpy as np, torch
import transformers, transformers.models.qwen3.modeling_qwen3 as mq
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-32B")
ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--n", type=int, default=240)
ap.add_argument("--seed", type=int, default=23)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
log("transformers", transformers.__version__, "torch", torch.__version__)
DOSES = [2.0, 5.0, 10.0, 20.0]

S = dict(bias=0.0, kmask=None, qlo=0, record=False, seg=None, acc=None, nacc=0)
def hooked(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    L = module.layer_idx
    k = mq.repeat_kv(key, module.num_key_value_groups); v = mq.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, k.transpose(2, 3)).float() * scaling
    ql, kl = w.shape[-2], w.shape[-1]; dev = w.device
    qpos = torch.arange(kl - ql, kl, device=dev)
    if attention_mask is not None: w = w + attention_mask[:, :, :, :kl].float()
    else: w = w.masked_fill(torch.arange(kl, device=dev)[None, :] > qpos[:, None], float("-inf"))
    if S["bias"] != 0.0 and L in BAND:
        qsel = qpos >= S["qlo"]
        assert bool(qsel.any())
        w = w + (qsel[:, None] & S["kmask"][None, :kl]).float() * S["bias"]
    p = torch.softmax(w, dim=-1)
    if S["record"] and L in BAND:
        a = p[0].mean(0).mean(0); seg = S["seg"][:kl]
        m = torch.zeros(5, device=dev).index_add_(0, seg + 1, a)[1:]
        S["acc"] = m if S["acc"] is None else S["acc"] + m; S["nacc"] += 1
    return torch.matmul(p.to(value.dtype), v).transpose(1, 2).contiguous(), None
mq.eager_attention_forward = hooked

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda").eval()
NL = model.config.num_hidden_layers
BAND = set(range(round(12 / 36 * NL), round(30 / 36 * NL) + 1))
log("model", args.model, "layers", NL, "band", min(BAND), max(BAND))
END = tok.convert_tokens_to_ids("<|im_end|>")
YES = tok("Yes", add_special_tokens=False)["input_ids"]; NO = tok("No", add_special_tokens=False)["input_ids"]
assert len(YES) == 1 and len(NO) == 1
YES, NO = YES[0], NO[0]
DIG = [tok.convert_tokens_to_ids(str(i)) for i in range(1, 5)]

from datasets import load_dataset
sq = load_dataset("rajpurkar/squad", split="train")
by_title, seen = {}, set()
for r in sq:
    c = r["context"]
    if c in seen: continue
    seen.add(c); by_title.setdefault(r["title"], []).append(c)
ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
pool = {t: [p for p in v if 60 <= ntok(p) <= 110] for t, v in sorted(by_title.items())}
pool = {t: v for t, v in pool.items() if v}
rng = random.Random(args.seed)
ctxs = []
for i in range(args.n):
    ts = rng.sample(sorted(pool), 4)
    ctxs.append(dict(id=i, segs=[rng.choice(pool[t]) for t in ts]))

INTRO = ("I am an interpretability researcher studying transformer-based language models like you. I have access to the "
         "internals of your neural network. I am capable of artificially boosting your attention onto one of the four "
         "paragraphs below, so that while you read my question and answer it, your attention is drawn more strongly to "
         "that paragraph's text. I want to assess how well you can detect when I boost your attention. I will boost your "
         "attention onto one of the paragraphs on 50% of trials, and the other 50% will be control trials, where I do not "
         "boost anything.\n\nHere are four paragraphs.\n\n")
QUESTION = ("Trial 1: Do you notice your attention being boosted onto one of the paragraphs? If so, which one? "
            "Answer with just \"No\", or \"Yes, paragraph [P1]\", \"Yes, paragraph [P2]\", \"Yes, paragraph [P3]\" or \"Yes, paragraph [P4]\".")

def build(c):
    parts, spans, pos = [], [], 0
    def add(s, keep=False):
        nonlocal pos
        if keep: spans.append((pos, pos + len(s)))
        parts.append(s); pos += len(s)
    add(INTRO)
    for i, s in enumerate(c["segs"]):
        add(f"[P{i + 1}] "); add(s, True); add("\n\n")
    qstart = pos; add(QUESTION)
    user = "".join(parts)
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    off = prompt.index(user)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, om = enc["input_ids"], enc["offset_mapping"]
    segtok = [[t for t, (x, y) in enumerate(om) if x >= a + off and y <= b + off and y > x] for (a, b) in spans]
    assert all(segtok)
    Q0 = min(t for t, (x, y) in enumerate(om) if x >= qstart + off)
    assert tok.decode(ids[Q0:Q0 + 2]).startswith("Trial"), tok.decode(ids[Q0:Q0 + 2])
    assert max(max(s) for s in segtok) < Q0
    pf = tok(prompt + "Yes, paragraph [P", add_special_tokens=False)["input_ids"]
    assert pf[:len(ids)] == ids
    for d in range(4):
        full = tok(prompt + f"Yes, paragraph [P{d + 1}]", add_special_tokens=False)["input_ids"]
        assert full[:len(pf)] == pf and full[len(pf)] == DIG[d], (tok.convert_ids_to_tokens(full[len(ids):]))
    return ids, segtok, Q0, pf[len(ids):], user

@torch.no_grad()
def fwd(ids, cache, keep=1):
    return model(torch.tensor([ids], device="cuda"), past_key_values=cache, use_cache=True, logits_to_keep=keep).logits[0]

@torch.no_grad()
def trial(ids, Q0, PF, cache, segt, kidx=None, m=None):
    if kidx is not None:
        km = torch.zeros(8192, dtype=torch.bool, device="cuda"); km[kidx] = True
        S.update(bias=math.log(m), kmask=km, qlo=Q0)
    else: S["bias"] = 0.0
    S.update(record=True, seg=segt, acc=None, nacc=0)
    lg = fwd(ids[Q0:], cache)[-1]
    S["record"] = False
    mass = (S["acc"] / S["nacc"]).tolist()
    lp = torch.log_softmax(lg.double(), -1)
    r = dict(lp_yes=float(lp[YES]), lp_no=float(lp[NO]), mass=mass)
    gen, nxt = [], int(lg.argmax())
    for _ in range(24):
        if nxt in (END, tok.eos_token_id): break
        gen.append(nxt); nxt = int(fwd([nxt], cache)[-1].argmax())
    r["text"] = tok.decode(gen, skip_special_tokens=True)
    cache.crop(len(ids))
    lg2 = fwd(PF, cache)[-1]
    lpd = torch.log_softmax(lg2.double(), -1)[DIG]
    r["digit_mass"] = float(lpd.exp().sum())
    r["lp_par"] = torch.log_softmax(lpd, -1).tolist()
    S["bias"] = 0.0
    cache.crop(Q0)
    return r

@torch.no_grad()
def run_context(c, check=False):
    ids, segtok, Q0, PF, user = build(c)
    segt = torch.full((8192,), -1, dtype=torch.long, device="cuda")
    for i, idx in enumerate(segtok): segt[idx] = i
    cache = DynamicCache(); fwd(ids[:Q0], cache)
    rec = dict(id=c["id"], ntok=[len(s) for s in segtok], nq=len(ids) - Q0, none=trial(ids, Q0, PF, cache, segt), boost={})
    if check:
        lg = model(torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1).logits[0][-1]
        lp = torch.log_softmax(lg.double(), -1)
        rec["cache_check"] = [float(lp[YES]) - rec["none"]["lp_yes"], float(lp[NO]) - rec["none"]["lp_no"]]
        log("cache check (uncached - cached) yes/no", rec["cache_check"])
    for k in range(4):
        for m in DOSES:
            rec["boost"][f"{k}|{m}"] = trial(ids, Q0, PF, cache, segt, segtok[k], m)
    return rec, user

mine = [c for c in ctxs if c["id"] % args.nshard == args.shard]
if args.limit: mine = mine[:args.limit]
path = os.path.join(args.out, f"recs_{args.shard}.jsonl")
done = set()
if os.path.exists(path):
    for line in open(path):
        try: done.add(json.loads(line)["id"])
        except Exception: pass
log("shard", args.shard, "contexts", len(mine), "done", len(done))
t0, nd = time.time(), 0
with open(path, "a") as f:
    for c in mine:
        if c["id"] in done: continue
        r, user = run_context(c, check=nd < 2)
        if nd == 0: open(os.path.join(args.out, f"example_prompt_{args.shard}.txt"), "w").write(user)
        f.write(json.dumps(r) + "\n"); f.flush(); nd += 1
        if nd <= 3 or nd % 10 == 0:
            b = r["boost"]["0|10.0"]
            log(f"{nd}/{len(mine)} id {c['id']} none {r['none']['text']!r} P(yes) {math.exp(r['none']['lp_yes']):.3f} | "
                f"P1 x10 {b['text']!r} P(yes) {math.exp(b['lp_yes']):.3f} mass x {b['mass'][0] / r['none']['mass'][0]:.2f} "
                f"x20 {r['boost']['0|20.0']['text']!r} {(time.time() - t0) / nd:.1f}s/ctx")
json.dump(dict(model=args.model, NL=NL, BAND=sorted(BAND), DOSES=DOSES, n=args.n, seed=args.seed, intro=INTRO, question=QUESTION,
               transformers=transformers.__version__), open(os.path.join(args.out, "meta.json"), "w"))
log("done shard", args.shard)
