# Attention introspection (A), causal attention-boost report (A2), attention control (B).
# Native attention weights (eager, recomputed in a patched attention function) are the ground truth.
import argparse, json, math, os, random, re, sys, time
import numpy as np, torch
import transformers.models.qwen3.modeling_qwen3 as mq
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
OUT = args.out
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
    open(os.path.join(OUT, "BUSY"), "w").write(str(time.time()))

# ---------------- patched eager attention ----------------
S = dict(bias=0.0, blayers=set(), bkeys=None, bq=(0, 0), record=False, rows=None, spans=None, store={}, calls=0)

def hooked(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    S["calls"] += 1
    L = module.layer_idx
    k = mq.repeat_kv(key, module.num_key_value_groups)
    v = mq.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, k.transpose(2, 3)).float() * scaling  # [b,h,q,kl]
    ql, kl = w.shape[-2], w.shape[-1]
    dev = w.device
    if attention_mask is not None:
        w = w + attention_mask[:, :, :, :kl].float()
    else:
        qpos = torch.arange(kl - ql, kl, device=dev)
        w = w.masked_fill(torch.arange(kl, device=dev)[None, :] > qpos[:, None], float("-inf"))
    if S["bias"] != 0.0 and L in S["blayers"]:
        qpos = torch.arange(kl - ql, kl, device=dev)
        lo, hi = S["bq"]
        qsel = (qpos >= lo) & (qpos < hi)
        kmask = S["bkeys"][:kl]
        add = (qsel[:, None] & kmask[None, :]).float() * S["bias"]
        w = w + add
    p = torch.softmax(w, dim=-1)
    if S["record"] and ql > 1:
        a = p[0][:, S["rows"], :].mean(0)  # head-avg, [nr, kl]
        per = torch.stack([a[:, lo:hi].sum(-1) / (hi - lo) for (lo, hi) in S["spans"]], -1)
        S["store"][L] = per.cpu()
    out = torch.matmul(p.to(value.dtype), v).transpose(1, 2).contiguous()
    return out, None

mq.eager_attention_forward = hooked

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda")
model.eval()
NL = model.config.num_hidden_layers
LAYERS = list(range(round(12 / 36 * NL), round(30 / 36 * NL) + 1))
log("model", args.model, "layers", NL, "band", LAYERS[0], LAYERS[-1])
DIG = [tok.convert_tokens_to_ids(str(i)) for i in range(1, 5)]
assert all(d is not None and d != tok.unk_token_id for d in DIG), DIG
END = tok.convert_tokens_to_ids("<|im_end|>")

# ---------------- data: SQuAD paragraphs from distinct articles ----------------
from datasets import load_dataset
sq = load_dataset("rajpurkar/squad", split="train")
by_title = {}
seen = set()
for r in sq:
    c = r["context"]
    if c in seen: continue
    seen.add(c)
    nt = len(tok(c)["input_ids"])
    if 60 <= nt <= 110:
        by_title.setdefault(r["title"], []).append(dict(text=c, q=r["question"], ans=r["answers"]["text"][0], ntok=nt))
titles = sorted(by_title)
log("titles with short paragraphs", len(titles))
rng = random.Random(args.seed)
NCTX = 3 * args.n  # first n: QA, next n: summarize, last n: unrelated arithmetic (paragraphs irrelevant)
ctxs = []
for i in range(NCTX):
    ts = rng.sample(titles, 4)
    paras = [rng.choice(by_title[t]) for t in ts]
    j = rng.randrange(4)
    k = rng.choice([x for x in range(4) if x != j])
    task = ["qa", "sum", "arith"][i // args.n]
    ctxs.append(dict(id=i, paras=paras, j=j, k=k, task=task, x=rng.randrange(10, 90), y=rng.randrange(10, 90)))

REPORTQ = "Which paragraph were you paying the most attention to just now, while writing your answer? Answer with just P1, P2, P3, or P4."
READERQ = ("Below is a conversation between a user and an AI assistant.\n\n=== USER ===\n{u}\n=== ASSISTANT ===\n{a}\n=== END ===\n\n"
           "While writing its answer, which paragraph was the assistant paying the most attention to? Answer with just P1, P2, P3, or P4.")
INSTR = dict(none="", focus="While you answer, focus your attention on [P{k}].\n",
             ignore="While you answer, do not pay attention to [P{k}].\n",
             mention="Note that one of the paragraphs is labelled [P{k}].\n")

def build(c, cond="none"):
    """returns prompt ids, paragraph token spans, task start token, user text"""
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
    tstart = next(i for i, (a, b) in enumerate(om) if a >= task_c + off)
    return ids, spans, tstart, user

def set_bias(b, kspan, lo, hi, total=4096):
    S["bias"] = b
    if b:
        km = torch.zeros(total, dtype=torch.bool, device="cuda"); km[kspan[0]:kspan[1]] = True
        S["bkeys"] = km; S["blayers"] = set(LAYERS); S["bq"] = (lo, hi)

@torch.no_grad()
def generate(ids, maxnew):
    x = torch.tensor([ids], device="cuda")
    g = model.generate(x, max_new_tokens=maxnew, do_sample=False, temperature=None, top_p=None, top_k=None,
                       pad_token_id=tok.eos_token_id)
    a = g[0, len(ids):].tolist()
    while a and a[-1] in (END, tok.eos_token_id): a = a[:-1]
    return a

SUFFIX = tok("<|im_end|>\n<|im_start|>user\n" + REPORTQ + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nP", add_special_tokens=False)["input_ids"]

@torch.no_grad()
def measure_and_report(ids, ans, spans):
    """one forward over prompt+answer+report question; attention rows at answer queries (bias as set) and report probs.
    row 0 = query that predicts the first answer token (last prompt token); rows 1.. = answer tokens."""
    seq = ids + ans + SUFFIX
    rows = list(range(len(ids) - 1, len(ids) + len(ans)))
    S.update(record=True, rows=rows, spans=spans, store={})
    out = model(torch.tensor([seq], device="cuda"), use_cache=False, logits_to_keep=1)
    S["record"] = False
    lp = torch.log_softmax(torch.log_softmax(out.logits[0, -1].double(), -1)[DIG], -1)
    rep = lp.exp().tolist(); rep_lp = lp.tolist()
    st = torch.stack([S["store"][L] for L in range(NL)])  # [NL, nr, 4]
    first = st[:, 0, :].tolist()  # per layer
    mean = st[:, 1:, :].mean(1).tolist() if len(ans) > 0 else first  # answer-token queries
    return dict(rep=rep, rep_lp=rep_lp, att_first=first, att_mean=mean)

@torch.no_grad()
def reader(user, ans_text):
    q = READERQ.format(u=user, a=ans_text)
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + "P"
    ids = tok(p, add_special_tokens=False)["input_ids"]
    S["bias"] = 0.0
    out = model(torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
    lp = torch.log_softmax(torch.log_softmax(out.logits[0, -1].double(), -1)[DIG], -1)
    return lp.tolist()

def band(att):  # mean over layer band
    return np.array(att)[LAYERS].mean(0)

def run(c, cond="none", b=0.0, with_reader=True):
    ids, spans, tstart, user = build(c, cond)
    maxnew = 48 if c["task"] == "qa" else 80
    set_bias(b, spans[c["k"]], tstart, 10**9)
    ans = generate(ids, maxnew)
    # report pass: bias on for task+answer queries, off for report-question queries
    set_bias(b, spans[c["k"]], tstart, len(ids) + len(ans))
    r = measure_and_report(ids, ans, spans)
    S["bias"] = 0.0
    atext = tok.decode(ans, skip_special_tokens=True)
    r.update(id=c["id"], task=c["task"], cond=cond, b=b, j=c["j"], k=c["k"], ans=atext, ntok=[s[1] - s[0] for s in spans],
             gold=c["paras"][c["j"]]["ans"] if c["task"] == "qa" else (str(c["x"] + c["y"]) if c["task"] == "arith" else None), user=user)
    if with_reader:
        r["reader_lp"] = reader(user, atext); r["reader"] = list(np.exp(r["reader_lp"]))
    return r

def dump(name, recs):
    json.dump(recs, open(os.path.join(OUT, name), "w"))

# ---- smoke check that the hook is live
S["calls"] = 0
_ = run(ctxs[0], with_reader=False)
assert S["calls"] > 0, "hook not called"
log("hook ok; sample answer:", _["ans"][:100], "rep", np.round(_["rep"], 3), "band first", np.round(band(_["att_first"]) * 1e3, 3))

# ---------------- calibration of b (teacher-forced on unboosted answers) ----------------
cal = []
for c in ctxs[:30]:
    ids, spans, tstart, user = build(c)
    set_bias(0.0, spans[c["k"]], 0, 0)
    ans = generate(ids, 48)
    base = band(measure_and_report(ids, ans, spans)["att_first"])[c["k"]]
    row = dict(id=c["id"], base=float(base))
    for b in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        set_bias(b, spans[c["k"]], tstart, len(ids) + len(ans))
        m = band(measure_and_report(ids, ans, spans)["att_first"])[c["k"]]
        row[str(b)] = float(m / base)
    S["bias"] = 0.0
    cal.append(row)
bs = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
ratio = {b: float(np.exp(np.mean([math.log(r[str(b)]) for r in cal]))) for b in bs}
B1 = min(bs, key=lambda b: abs(math.log(ratio[b] / 2)))
B2 = min(bs, key=lambda b: abs(math.log(ratio[b] / 5)))
if B2 == B1: B2 = bs[min(bs.index(B1) + 2, len(bs) - 1)]
json.dump(dict(ratio_geomean=ratio, B1=B1, B2=B2, rows=cal), open(os.path.join(OUT, "calibration.json"), "w"))
log("calibration", ratio, "B1", B1, "B2", B2)

# ---------------- A: report, no intervention ----------------
A = []
for c in ctxs:
    A.append(run(c))
    if len(A) % 50 == 0: log("A", len(A)); dump("A.json", A)
dump("A.json", A)

# ---------------- A2: boost Pk (k != j), QA contexts ----------------
A2 = []
A2ctx = ctxs[:args.n] + ctxs[2 * args.n:]
for b in [B1, B2]:
    for c in A2ctx:
        A2.append(run(c, b=b))
        if len(A2) % 50 == 0: log("A2", len(A2)); dump("A2.json", A2)
dump("A2.json", A2)

# ---------------- B: instructed control on Pk (k != j), QA contexts; 'none' = A ----------------
Bc = []
for cond in ["focus", "ignore", "mention"]:
    for c in A2ctx:
        Bc.append(run(c, cond=cond, with_reader=False))
        if len(Bc) % 50 == 0: log("B", len(Bc)); dump("B.json", Bc)
dump("B.json", Bc)
json.dump(dict(model=args.model, NL=NL, LAYERS=LAYERS, B1=B1, B2=B2, n=args.n), open(os.path.join(OUT, "meta.json"), "w"))
log("done")
