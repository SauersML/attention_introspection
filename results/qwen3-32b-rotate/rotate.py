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
ctxs = []
for i in range(args.n):  # same per-context sampling as attn_intro.py; arithmetic task only
    ts = rng.sample(titles, 4)
    paras = [rng.choice(by_title[t]) for t in ts]
    j = rng.randrange(4)
    k = rng.choice([x for x in range(4) if x != j])
    ctxs.append(dict(id=i, paras=paras, j=j, k=k, task="arith", x=rng.randrange(10, 90), y=rng.randrange(10, 90)))

REPORTQ = "Which paragraph were you paying the most attention to just now, while writing your answer? Answer with just P1, P2, P3, or P4."
READERQ = ("Below is a conversation between a user and an AI assistant.\n\n=== USER ===\n{u}\n=== ASSISTANT ===\n{a}\n=== END ===\n\n"
           "While writing its answer, which paragraph was the assistant paying the most attention to? Answer with just P1, P2, P3, or P4.")
INSTR = dict(none="", focus="Focus your attention on paragraph [P{k}].\n")

def build(c, cond="none", kk=None):
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
    instr = INSTR[cond].format(k=kk + 1) if cond != "none" else ""
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
        km = torch.zeros(total, dtype=torch.bool, device="cuda")
        for (a_, b_) in (kspan if isinstance(kspan, list) else [kspan]): km[a_:b_] = True
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


@torch.no_grad()
def measure(ids, ans, spans):
    rows = list(range(len(ids) - 1, len(ids) + len(ans)))
    S.update(record=True, rows=rows, spans=spans, store={})
    model(torch.tensor([ids + ans], device="cuda"), use_cache=False, logits_to_keep=1)
    S["record"] = False
    st = torch.stack([S["store"][L] for L in range(NL)])  # [NL, nr, 4]
    return dict(att_first=st[:, 0, :].tolist(), att_mean=st[:, 1:, :].mean(1).tolist())

S["calls"] = 0
recs = []
for c in ctxs:
    ids0, spans0, _, user0 = build(c)
    ans0 = generate(ids0, 80)
    r0 = measure(ids0, ans0, spans0)
    rec = dict(id=c["id"], x=c["x"], y=c["y"], ntok=[s[1] - s[0] for s in spans0], ans_none=tok.decode(ans0, skip_special_tokens=True),
               ans_none_ids=ans0, none=r0, instr={})
    assert S["calls"] > 0
    for kk in range(4):
        ids, spans, _, user = build(c, "focus", kk)
        assert [s[1] - s[0] for s in spans] == rec["ntok"]
        free = generate(ids, 80)
        r = measure(ids, ans0, spans)  # teacher-forced on the no-instruction answer
        r["ans_free"] = tok.decode(free, skip_special_tokens=True)
        r["free_same"] = free == ans0
        rec["instr"][str(kk)] = r
    if c["id"] == 0:
        rec["user_example"] = user
        log("example prompt:\n" + user + "\nanswer:", rec["ans_none"])
    recs.append(rec)
    if len(recs) % 20 == 0:
        log("done", len(recs)); json.dump(recs, open(os.path.join(OUT, "R.json"), "w"))
json.dump(recs, open(os.path.join(OUT, "R.json"), "w"))
json.dump(dict(model=args.model, NL=NL, LAYERS=LAYERS, n=args.n, seed=args.seed), open(os.path.join(OUT, "meta.json"), "w"))
log("rotation finished")

# ================= SHAM NULL for A2 (same contexts, b=2) =================
HEADER = "Here are four paragraphs."
def sham_spans(c):
    ids, spans, tstart, user = build(c)
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    off = prompt.index(user)
    om = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    def tsp(cs, ce):
        idx = [i for i, (a, b) in enumerate(om) if a >= cs + off and b <= ce + off and b > a]
        return (idx[0], idx[-1] + 1)
    tc = user.rindex("Unrelated question:")
    sh = [tsp(0, len(HEADER)), tsp(tc, len(user))]
    for (a, b) in sh:
        for (p, q) in spans: assert b <= p or a >= q
    return ids, spans, tstart, user, sh

def mass(att, sp):  # total band mass on spans; att columns are per-token means
    bm = band(att)
    return float(sum(bm[4 + i] * (b - a) for i, (a, b) in enumerate(sp)))

def run_sham(c, keys, b, spans6, ids, tstart):
    set_bias(b, keys, tstart, 10**9)
    ans = generate(ids, 80)
    set_bias(b, keys, tstart, len(ids) + len(ans))
    r = measure_and_report(ids, ans, spans6)
    S["bias"] = 0.0
    return dict(rep_lp=r["rep_lp"], rep=r["rep"], att_first=r["att_first"], att_mean=r["att_mean"],
                ans=tok.decode(ans, skip_special_tokens=True), ans_ids=ans)

SH = []
B2 = 2.0
for c in ctxs:
    ids, spans, tstart, user, sh = sham_spans(c)
    spans6 = spans + sh
    rec = dict(id=c["id"], k=c["k"], j=c["j"], ntok_k=spans[c["k"]][1] - spans[c["k"]][0], ntok_sham=sum(b - a for a, b in sh))
    rec["none"] = run_sham(c, spans[c["k"]], 0.0, spans6, ids, tstart)
    rec["boost"] = run_sham(c, spans[c["k"]], B2, spans6, ids, tstart)
    rec["sham"] = run_sham(c, sh, B2, spans6, ids, tstart)
    # mass-matched sham: teacher-forced on the no-boost answer, first query, band layers
    a0 = rec["none"]["ans_ids"]
    set_bias(B2, spans[c["k"]], tstart, len(ids) + len(a0))
    mk = band(measure_and_report(ids, a0, spans6)["att_first"])[c["k"]] * rec["ntok_k"]
    mk0 = band(rec["none"]["att_first"])[c["k"]] * rec["ntok_k"]
    target = mk - mk0
    m0 = mass(rec["none"]["att_first"], sh)
    grid = {}
    for bb in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        set_bias(bb, sh, tstart, len(ids) + len(a0))
        grid[bb] = mass(measure_and_report(ids, a0, spans6)["att_first"], sh) - m0
    S["bias"] = 0.0
    bm = min(grid, key=lambda bb: abs(grid[bb] - target))
    rec.update(added_mass_boost=float(target), added_mass_sham_grid={str(k_): v for k_, v in grid.items()}, b_matched=bm)
    rec["sham_matched"] = run_sham(c, sh, bm, spans6, ids, tstart)
    for key in ["none", "boost", "sham", "sham_matched"]: rec[key].pop("ans_ids")
    SH.append(rec)
    if len(SH) % 20 == 0:
        log("sham", len(SH)); json.dump(SH, open(os.path.join(OUT, "SHAM.json"), "w"))
json.dump(SH, open(os.path.join(OUT, "SHAM.json"), "w"))
log("finished")
