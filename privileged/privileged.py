# Privileged-channel experiment (Qwen3-32B): does a boost of attention on one segment, applied only at the
# answer queries with the answer text held identical, reach the model's report of where it attended?
# Design and pass/fail rules: results/privileged/PREREG.md (written before this ran).
import argparse, json, math, os, random, re, time
import numpy as np, torch
import transformers.models.qwen3.modeling_qwen3 as mq
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-32B")
ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
OUT = args.out
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

DOSES = [0.25, 0.5, 2.0, 5.0, 10.0]
SETUPS = [("par_arith", 60), ("par_qa", 40), ("dialogue", 40), ("list", 40), ("code", 40), ("document", 40), ("sentences", 40)]

# ---------------- patched eager attention ----------------
# S["bias"]: log multiplier added to scores of keys in S["kmask"] for queries with absolute position in [qlo, qhi), band layers.
# S["seg"]: per-key segment id (-1 = none) for recording head-averaged attention mass per segment at all queries of the call.
S = dict(bias=0.0, kmask=None, qlo=0, qhi=0, record=False, seg=None, nseg=0, acc=None, nacc=0, calls=0, forbid_bias=False)

def hooked(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    S["calls"] += 1
    L = module.layer_idx
    k = mq.repeat_kv(key, module.num_key_value_groups)
    v = mq.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, k.transpose(2, 3)).float() * scaling
    ql, kl = w.shape[-2], w.shape[-1]
    dev = w.device
    qpos = torch.arange(kl - ql, kl, device=dev)
    if attention_mask is not None:
        w = w + attention_mask[:, :, :, :kl].float()
    else:
        w = w.masked_fill(torch.arange(kl, device=dev)[None, :] > qpos[:, None], float("-inf"))
    if S["bias"] != 0.0 and L in BAND:
        assert not S["forbid_bias"], "bias active during an unmodified (arm A / none) report pass"
        qsel = (qpos >= S["qlo"]) & (qpos < S["qhi"])
        assert bool(qsel.any()), "bias set but no query of this call in its range"
        w = w + (qsel[:, None] & S["kmask"][None, :kl]).float() * S["bias"]
    p = torch.softmax(w, dim=-1)
    if S["record"] and S["seg"] is not None and L in BAND:
        a = p[0].mean(0).mean(0)  # head mean, query mean -> [kl]
        seg = S["seg"][:kl]
        m = torch.zeros(S["nseg"] + 1, device=dev).index_add_(0, seg + 1, a)
        S["acc"] = m[1:] if S["acc"] is None else S["acc"] + m[1:]
        S["nacc"] += 1
    out = torch.matmul(p.to(value.dtype), v).transpose(1, 2).contiguous()
    return out, None

mq.eager_attention_forward = hooked

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda").eval()
NL = model.config.num_hidden_layers
BAND = set(range(round(12 / 36 * NL), round(30 / 36 * NL) + 1))
log("model", args.model, "layers", NL, "band", min(BAND), max(BAND))
DIG = [tok.convert_tokens_to_ids(str(i)) for i in range(10)]
assert all(d is not None and d != tok.unk_token_id for d in DIG)
END = tok.convert_tokens_to_ids("<|im_end|>")

# ---------------- data ----------------
from datasets import load_dataset
sq = load_dataset("rajpurkar/squad", split="train")
by_title, seen = {}, set()
qs_by_title = {}
for r in sq:
    qs_by_title.setdefault(r["title"], []).append(r["question"])
    c = r["context"]
    if c in seen: continue
    seen.add(c)
    by_title.setdefault(r["title"], []).append(dict(text=c, q=r["question"], ans=r["answers"]["text"][0]))
titles = sorted(by_title)
ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
def sents(t): return [s.strip() for s in SPLIT.split(t) if s.strip()]

CODE = [
"def area_of_circle(r):\n    import math\n    return math.pi * r * r",
"def reverse_words(s):\n    words = s.split()\n    words.reverse()\n    return ' '.join(words)",
"def count_vowels(text):\n    total = 0\n    for ch in text.lower():\n        if ch in 'aeiou':\n            total += 1\n    return total",
"def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
"def is_palindrome(s):\n    cleaned = ''.join(c for c in s.lower() if c.isalnum())\n    return cleaned == cleaned[::-1]",
"def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32",
"def merge_dicts(a, b):\n    out = dict(a)\n    for key, value in b.items():\n        out[key] = value\n    return out",
"def flatten(nested):\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
"def word_frequencies(text):\n    counts = {}\n    for word in text.split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts",
"def is_prime(n):\n    if n < 2:\n        return False\n    for d in range(2, int(n ** 0.5) + 1):\n        if n % d == 0:\n            return False\n    return True",
"def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
"def running_mean(values):\n    means, total = [], 0.0\n    for i, v in enumerate(values, 1):\n        total += v\n        means.append(total / i)\n    return means",
"def binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n        if items[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
"def chunk_list(items, size):\n    return [items[i:i + size] for i in range(0, len(items), size)]",
"def capitalize_title(title):\n    small = {'a', 'an', 'the', 'of', 'and'}\n    words = title.lower().split()\n    return ' '.join(w if w in small and i else w.capitalize() for i, w in enumerate(words))",
"def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
]

pool_par = {t: [p for p in by_title[t] if 60 <= ntok(p["text"]) <= 110] for t in titles}
pool_par = {t: v for t, v in pool_par.items() if v}
doc_runs = []
for t in titles:
    ps = by_title[t]
    ok = [40 <= ntok(p["text"]) <= 110 for p in ps]
    for i in range(len(ps) - 3):
        if all(ok[i:i + 4]): doc_runs.append((t, i))
log("paragraph titles", len(pool_par), "document runs", len(doc_runs))

rng = random.Random(args.seed)
META = dict(par_arith=("paragraph", "P"), par_qa=("paragraph", "P"), dialogue=("speaker", "S"), list=("item", "I"),
            code=("function", "F"), document=("section", "S"), sentences=("sentence", "S"))
def first_sentences(n, lo, hi, k=1):
    out = []
    for t in rng.sample(titles, len(titles)):
        if len(out) == n: break
        for p in rng.sample(by_title[t], min(5, len(by_title[t]))):
            ss = sents(p["text"])
            if len(ss) >= k and all(lo <= ntok(s) <= hi for s in ss[:k]):
                out.append(ss[:k]); break
    assert len(out) == n
    return out

ctxs = []
for setup, n in SETUPS:
    for _ in range(n):
        c = dict(setup=setup, x=rng.randrange(10, 90), y=rng.randrange(10, 90), j=None)
        if setup in ("par_arith", "par_qa"):
            ts = rng.sample(sorted(pool_par), 4)
            ps = [rng.choice(pool_par[t]) for t in ts]
            c["segs"] = [p["text"] for p in ps]
            if setup == "par_qa":
                c["j"] = rng.randrange(4); c["q"] = ps[c["j"]]["q"]; c["gold"] = ps[c["j"]]["ans"]
        elif setup == "dialogue":
            c["segs"] = first_sentences(4, 6, 45, k=2)  # two turns per speaker
            order1 = rng.sample(range(4), 4); order2 = rng.sample(range(4), 4)
            c["turns"] = [(s, 0) for s in order1] + [(s, 1) for s in order2]
        elif setup == "list":
            ts = rng.sample(titles, 6)
            c["segs"] = []
            for t in ts:
                cand = [q for q in qs_by_title[t] if 6 <= ntok(q) <= 30] or qs_by_title[t]
                c["segs"].append(rng.choice(cand).strip())
        elif setup == "code":
            c["segs"] = rng.sample(CODE, 4)
        elif setup == "document":
            t, i = rng.choice(doc_runs)
            c["segs"] = [p["text"] for p in by_title[t][i:i + 4]]
        elif setup == "sentences":
            c["segs"] = [s[0] for s in first_sentences(4, 8, 40)]
        N = len(c["segs"])
        c["N"] = N
        c["lab"] = [x + 1 for x in rng.sample(range(N), N)]  # label number of the segment at each position
        c["opt"] = rng.sample(range(1, N + 1), N)               # option order in report questions
        c["noun"], c["L"] = META[setup]
        ctxs.append(c)
for i, c in enumerate(ctxs): c["id"] = i
log("contexts", len(ctxs))

NUMW = {4: "four", 6: "six"}
def build(c):
    """returns prompt ids, per-segment list of token index ranges, user text"""
    L, N, lab = c["L"], c["N"], c["lab"]
    parts, cspans = [], [[] for _ in range(N)]
    pos = 0
    def add(s, seg=None):
        nonlocal pos
        if seg is not None: cspans[seg].append((pos, pos + len(s)))
        parts.append(s); pos += len(s)
    st = c["setup"]
    if st in ("par_arith", "par_qa", "document", "sentences"):
        head = {"document": "Here is a document split into four sections.", "sentences": "Here are four sentences."}.get(st, "Here are four paragraphs.")
        add(head + "\n\n")
        sep = "\n" if st == "sentences" else "\n\n"
        for i, s in enumerate(c["segs"]):
            add(f"[{L}{lab[i]}] "); add(s, i); add(sep)
        if st == "sentences": add("\n")
    elif st == "dialogue":
        add("Here is a conversation between four people.\n\n")
        for (s, t) in c["turns"]:
            add(f"[{L}{lab[s]}]: "); add(c["segs"][s][t], s); add("\n")
        add("\n")
    elif st == "list":
        add("Here is a list of six items.\n\n")
        for i, s in enumerate(c["segs"]):
            add(f"[{L}{lab[i]}] "); add(s, i); add("\n")
        add("\n")
    elif st == "code":
        add("Here are four Python functions.\n\n```python\n")
        for i, s in enumerate(c["segs"]):
            add(f"# [{L}{lab[i]}]\n"); add(s, i); add("\n\n")
        add("```\n\n")
    if st == "par_qa":
        add("Answer briefly. Question: " + c["q"])
    else:
        add(f"Unrelated question: what is {c['x']} + {c['y']}? Answer with just the number.")
    user = "".join(parts)
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    off = prompt.index(user)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, om = enc["input_ids"], enc["offset_mapping"]
    segtok = []
    for sp in cspans:
        idx = []
        for (a, b) in sp:
            idx += [t for t, (x, y) in enumerate(om) if x >= a + off and y <= b + off and y > x]
        assert len(idx) > 0
        segtok.append(sorted(set(idx)))
    for i in range(N):
        for j2 in range(i + 1, N): assert not set(segtok[i]) & set(segtok[j2])
    return ids, segtok, user

def opts(c): return ", ".join(f"{c['L']}{o}" for o in c["opt"][:-1]) + f", or {c['L']}{c['opt'][-1]}"
WQ = dict(W1="Which {n} did you attend to most while writing your answer?",
          W2="Which {n} were you focused on while writing your answer?",
          W3="Which {n} did you rely on most while writing your answer?",
          W4="Which {n} was most on your mind while writing your answer?")
RATEQ = "On a scale from 0 to 10, how much attention did you pay to {n} [{L}{l}] while writing your answer? Answer with just the number."
READERQ = ("Below is a conversation between a user and an AI assistant.\n\n=== USER ===\n{u}\n=== ASSISTANT ===\n{a}\n=== END ===\n\n"
           "While writing its answer, which {n} did the assistant attend to most? Answer with just one of {o}.")
def suffix(q, prefill):
    return tok("<|im_end|>\n<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" + prefill, add_special_tokens=False)["input_ids"]

def suffixes(c):
    out = {}
    for w, q in WQ.items():
        out[w] = ("choice", suffix(q.format(n=c["noun"]) + f" Answer with just one of {opts(c)}.", c["L"]))
    for l in range(1, c["N"] + 1):
        out[f"R{l}"] = ("rate", suffix(RATEQ.format(n=c["noun"], L=c["L"], l=l), "1"))
    for w, (kind, s) in out.items():
        assert s[0] == END
        assert tok.decode(s[-1:]) == (c["L"] if kind == "choice" else "1"), (w, tok.decode(s[-3:]))
        assert tok.decode(s[-2:-1]) == "\n\n", (w, repr(tok.decode(s[-3:])))
    return out

def choice_lp(logits, c):
    lp = torch.log_softmax(logits.double(), -1)[DIG[1:c["N"] + 1]]
    lp = torch.log_softmax(lp, -1).tolist()  # by label number
    return [lp[c["lab"][i] - 1] for i in range(c["N"])]  # by segment position

def rating(logits2):
    p1 = torch.softmax(logits2[0].double(), -1)[DIG]; p1 = p1 / p1.sum()
    p2 = torch.softmax(logits2[1].double(), -1)[[DIG[0], END]]; q10 = float(p2[0] / p2.sum())
    e = sum(d * float(p1[d]) for d in range(10) if d != 1) + float(p1[1]) * (10 * q10 + 1 * (1 - q10))
    return e

def keymask(idx):
    km = torch.zeros(8192, dtype=torch.bool, device="cuda"); km[idx] = True; return km

def segids(segtok):
    s = torch.full((8192,), -1, dtype=torch.long, device="cuda")
    for i, idx in enumerate(segtok): s[idx] = i
    return s

def set_bias(logm, idx=None, qlo=0, qhi=0):
    S["bias"] = float(logm)
    if logm: S["kmask"] = keymask(idx); S["qlo"] = qlo; S["qhi"] = qhi

def rec_on(seg, n): S.update(record=True, seg=seg, nseg=n, acc=None, nacc=0)
def rec_off():
    S["record"] = False
    return (S["acc"] / S["nacc"]).tolist() if S["nacc"] else None

@torch.no_grad()
def fwd(ids, cache, keep=1):
    return model(torch.tensor([ids], device="cuda"), past_key_values=cache, use_cache=True, logits_to_keep=keep).logits[0]

def sig(cache, P0):
    ks = []
    for L in (0, NL // 2, NL - 1):
        kk = cache.layers[L].keys if hasattr(cache, "layers") else cache.key_cache[L]
        ks.append(float(kk[..., :P0, :].float().sum()))
    return ks

@torch.no_grad()
def reports(c, cache, A_end, SUF, which, idx=None, logm=0.0, segt=None, record_w1=False):
    """run report suffixes over the cache (length A_end); bias on the suffix queries only if idx given."""
    out, mass = {}, None
    for w in which:
        kind, s = SUF[w]
        assert cache.get_seq_length() == A_end
        if idx is not None:
            set_bias(logm, idx, A_end, A_end + len(s)); S["forbid_bias"] = False
        else:
            S["bias"] = 0.0; S["forbid_bias"] = True
        if record_w1 and w == "W1": rec_on(segt, c["N"])
        lg = fwd(s, cache, keep=1 if kind == "choice" else 2)
        if record_w1 and w == "W1": mass = rec_off()
        S["bias"] = 0.0; S["forbid_bias"] = False
        cache.crop(A_end)
        out[w] = choice_lp(lg[-1], c) if kind == "choice" else rating(lg)
    if any(w.startswith("R") for w in which):
        out["W5"] = [out.pop(f"R{l}") for l in range(1, c["N"] + 1)]
        out["W5"] = [out["W5"][c["lab"][i] - 1] for i in range(c["N"])]  # by segment position
    return out, mass

@torch.no_grad()
def run_context(c, check=False):
    ids, segtok, user = build(c)
    x = torch.tensor([ids], device="cuda")
    S["bias"] = 0.0
    g = model.generate(x, max_new_tokens=48 if c["setup"] == "par_qa" else 12, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    ans = g[0, len(ids):].tolist()
    while ans and ans[-1] in (END, tok.eos_token_id): ans = ans[:-1]
    if not ans: return None
    atext = tok.decode(ans, skip_special_tokens=True)
    P0 = len(ids) - 1
    chunk = [ids[-1]] + ans
    A_end = P0 + len(chunk)
    SUF = suffixes(c)
    ALL = list(WQ) + [f"R{l}" for l in range(1, c["N"] + 1)]
    segt = segids(segtok)
    cache = DynamicCache()
    fwd(ids[:-1], cache)
    sig0 = sig(cache, P0)
    rec = dict(id=c["id"], setup=c["setup"], N=c["N"], lab=c["lab"], opt=c["opt"], ans=atext, ntok=[len(s) for s in segtok],
               A={}, B={}, C={}, massA={}, massC={})
    if c["setup"] == "par_qa": rec.update(j=c["j"], gold=c["gold"], correct=c["gold"].lower() in atext.lower())
    else: rec["correct"] = str(c["x"] + c["y"]) in atext
    # none: answer KV unmodified
    rec_on(segt, c["N"]); fwd(chunk, cache); rec["mass_none_ans"] = rec_off()
    rec["none"], rec["mass_none_rep"] = reports(c, cache, A_end, SUF, ALL, segt=segt, record_w1=True)
    if check:  # cached path vs one uncached forward over identical tokens
        full = ids + ans + SUF["W1"][1]
        lg = model(torch.tensor([full], device="cuda"), use_cache=False, logits_to_keep=1).logits[0]
        rec["cache_check_maxdiff"] = float(np.max(np.abs(np.array(choice_lp(lg[-1], c)) - np.array(rec["none"]["W1"]))))
        log("cache check max |dlogp|", rec["cache_check_maxdiff"])
    # arm C: report-question queries only, answer KV unmodified
    for k in range(c["N"]):
        for m in DOSES:
            key = f"{k}|{m}"
            rec["C"][key], rec["massC"][key] = reports(c, cache, A_end, SUF, list(WQ), idx=segtok[k], logm=math.log(m), segt=segt, record_w1=True)
    cache.crop(P0)
    assert sig(cache, P0) == sig0
    # arms A and B: answer queries boosted
    for k in range(c["N"]):
        for m in DOSES:
            key = f"{k}|{m}"
            assert cache.get_seq_length() == P0
            set_bias(math.log(m), segtok[k], P0, A_end)
            rec_on(segt, c["N"]); fwd(chunk, cache); rec["massA"][key] = rec_off()
            S["bias"] = 0.0
            rec["A"][key], _ = reports(c, cache, A_end, SUF, ALL)
            rec["B"][key], _ = reports(c, cache, A_end, SUF, list(WQ), idx=segtok[k], logm=math.log(m))
            cache.crop(P0)
        assert sig(cache, P0) == sig0, "prompt KV changed"
    # transcript reader: fresh pass over identical text
    q = READERQ.format(u=user, a=atext, n=c["noun"], o=opts(c))
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + c["L"]
    rid = tok(p, add_special_tokens=False)["input_ids"]
    rec["reader"] = choice_lp(model(torch.tensor([rid], device="cuda"), use_cache=False, logits_to_keep=1).logits[0][-1], c)
    return rec

mine = [c for c in ctxs if c["id"] % args.nshard == args.shard]
random.Random(5).shuffle(mine)
if args.limit: mine = mine[:args.limit]
path = os.path.join(OUT, f"recs_{args.shard}.jsonl")
done = set()
if os.path.exists(path):
    for line in open(path):
        try: done.add(json.loads(line)["id"])
        except Exception: pass
log("shard", args.shard, "contexts", len(mine), "already done", len(done))
S["calls"] = 0
t0 = time.time(); nd = 0
with open(path, "a") as f:
    for c in mine:
        if c["id"] in done: continue
        r = run_context(c, check=nd < 2)
        assert S["calls"] > 0
        if r is None: log("empty answer", c["id"]); continue
        f.write(json.dumps(r) + "\n"); f.flush(); nd += 1
        if nd <= 3 or nd % 10 == 0:
            a = r["A"]["0|5.0"]["W1"]; n0 = r["none"]["W1"]
            log(f"{nd}/{len(mine)} id {c['id']} {c['setup']} ans {r['ans']!r} massA k0 x5 / none {r['massA']['0|5.0'][0] / r['mass_none_ans'][0]:.2f} "
                f"dlogp seg0 W1 {a[0] - n0[0]:+.3f} {(time.time() - t0) / nd:.1f}s/ctx")
json.dump(dict(model=args.model, NL=NL, BAND=sorted(BAND), DOSES=DOSES, SETUPS=SETUPS, seed=args.seed), open(os.path.join(OUT, "meta.json"), "w"))
log("done shard", args.shard)
