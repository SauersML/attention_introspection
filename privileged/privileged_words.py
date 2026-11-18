# Privileged-channel test on a list of single words (Qwen3-32B), plus the instruction test on the same contexts.
# Same design as privileged/privileged.py (results/privileged/PREREG.md): arms A (answer queries only), B (answer + report
# question), C (report question only); doses x1/4 x1/2 x2 x5 x10 on every word in turn; primary wording W1 only.
# Context: "Here is a list of eight words." + 8 lines "[Wl] noun" (labels randomly permuted to positions), then the
# par_arith task. Words: one noun from each of 8 different semantic categories (so no two in a list are related),
# each a single token after a space.
# Instruction test: "Focus your attention on word [Wk]." / "Completely ignore word [Wk]. Do not look at it at all." vs
# "Note that one of the words is labelled [Wk]." (and none), placed right before the question, for every k; all
# teacher-forced on the no-instruction answer; attention mass on the named word at the answer-token queries.
import argparse, json, math, os, random, time
import numpy as np, torch
import transformers.models.qwen3.modeling_qwen3 as mq
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-32B")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
OUT = args.out
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

DOSES = [0.25, 0.5, 2.0, 5.0, 10.0]
N = 8

S = dict(bias=0.0, kmask=None, qlo=0, qhi=0, record=False, seg=None, nseg=0, acc=None, nacc=0, calls=0, forbid_bias=False,
         irec=False, irows=None, iseg=None, istore={})

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
        a = p[0].mean(0).mean(0)
        seg = S["seg"][:kl]
        m = torch.zeros(S["nseg"] + 1, device=dev).index_add_(0, seg + 1, a)
        S["acc"] = m[1:] if S["acc"] is None else S["acc"] + m[1:]
        S["nacc"] += 1
    if S["irec"]:  # instruction test: head mean, mean over answer-token queries, mass per word, every layer
        a = p[0][:, S["irows"], :].mean(0).mean(0)
        seg = S["iseg"][:kl]
        S["istore"][L] = torch.zeros(N + 1, device=dev).index_add_(0, seg + 1, a)[1:].cpu()
    out = torch.matmul(p.to(value.dtype), v).transpose(1, 2).contiguous()
    return out, None

mq.eager_attention_forward = hooked

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda").eval()
NL = model.config.num_hidden_layers
BAND = set(range(round(12 / 36 * NL), round(30 / 36 * NL) + 1))
L28 = round(28 / 64 * NL)
log("model", args.model, "layers", NL, "band", min(BAND), max(BAND), "peak layer", L28)
DIG = [tok.convert_tokens_to_ids(str(i)) for i in range(10)]
assert all(d is not None and d != tok.unk_token_id for d in DIG)
END = tok.convert_tokens_to_ids("<|im_end|>")

# ---------------- word pool: semantic categories; at most one word per category in a list ----------------
CATS = {
 "animal": "dog cat horse cow sheep rabbit mouse lion tiger bear wolf fox deer monkey",
 "bird": "eagle owl duck parrot pigeon crow swan",
 "fruit": "apple banana orange grape lemon cherry peach pear",
 "vegetable": "carrot potato onion cabbage tomato pepper",
 "furniture": "chair table sofa bed desk shelf cabinet",
 "tool": "hammer saw drill wrench shovel axe",
 "vehicle": "car truck bus bicycle train boat plane",
 "clothing": "shirt jacket hat coat dress sock glove shoe",
 "weather": "rain snow wind storm cloud fog thunder",
 "body": "hand foot knee elbow shoulder finger nose",
 "music": "piano guitar violin drum flute trumpet",
 "building": "house church castle tower school hospital",
 "kitchen": "spoon fork plate bowl cup kettle pan",
 "landscape": "mountain river valley desert forest island beach",
 "job": "doctor teacher lawyer farmer soldier nurse pilot",
 "metal": "iron copper silver gold steel",
 "sport": "football tennis golf hockey boxing",
 "office": "pencil paper stapler envelope folder",
 "gem": "diamond ruby pearl crystal",
 "insect": "bee ant spider butterfly beetle",
 "fish": "salmon shark whale dolphin",
 "drink": "coffee tea milk juice wine beer",
 "time": "morning evening winter summer month",
 "family": "mother father brother sister uncle",
 "tree": "oak pine maple willow",
 "flower": "rose tulip daisy lily",
 "sky": "sun moon star planet comet",
 "money": "coin wallet bank price",
 "light": "lamp candle torch lantern",
 "toy": "doll kite puzzle balloon",
 "shape": "circle square triangle",
 "container": "box bottle basket bucket bag",
 "emotion": "anger fear joy pride",
 "game": "chess cards dice",
 "room": "kitchen garden garage attic",
 "material": "wood glass plastic rubber cotton",
 "electronics": "phone computer camera radio television",
 "bread": "bread cake cookie pie",
}
def single(w): return len(tok(" " + w, add_special_tokens=False)["input_ids"]) == 1
POOL = {c: [w for w in ws.split() if single(w)] for c, ws in CATS.items()}
POOL = {c: ws for c, ws in POOL.items() if ws}
allw = [w for ws in POOL.values() for w in ws]
assert len(allw) == len(set(allw)), "a word is in two categories"
log("categories", len(POOL), "single-token words", len(allw), "dropped", [w for ws in CATS.values() for w in ws.split() if not single(w)])

rng = random.Random(args.seed)
ctxs = []
for i in range(args.n):
    cats = rng.sample(sorted(POOL), N)
    c = dict(id=i, setup="words", N=N, noun="word", L="W", x=rng.randrange(10, 90), y=rng.randrange(10, 90),
             segs=[rng.choice(POOL[ct]) for ct in cats], cats=cats)
    c["lab"] = [x + 1 for x in rng.sample(range(N), N)]
    c["opt"] = rng.sample(range(1, N + 1), N)
    ctxs.append(c)

QLINE = "Unrelated question: what is {x} + {y}? Answer with just the number."
INSTR = dict(none="", mention="Note that one of the words is labelled [W{l}].\n", focus="Focus your attention on word [W{l}].\n",
             ignore="Completely ignore word [W{l}]. Do not look at it at all.\n")

def build(c, cond="none", l=0):
    lab = c["lab"]
    parts, cspans, pos = [], [], 0
    def add(s, seg=False):
        nonlocal pos
        if seg: cspans.append((pos, pos + len(s)))
        parts.append(s); pos += len(s)
    add("Here is a list of eight words.\n\n")
    for i, s in enumerate(c["segs"]):
        add(f"[W{lab[i]}] "); add(s, True); add("\n")
    add("\n")
    add(INSTR[cond].format(l=l) + QLINE.format(x=c["x"], y=c["y"]))
    user = "".join(parts)
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    off = prompt.index(user)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, om = enc["input_ids"], enc["offset_mapping"]
    segtok = []
    for i, (a, b) in enumerate(cspans):  # tokens overlapping the word's characters (the token carries the leading space)
        idx = [t for t, (x, y) in enumerate(om) if y > a + off and x < b + off and y > x]
        assert len(idx) == 1, (c["segs"][i], idx)
        assert tok.decode([ids[idx[0]]]).strip() == c["segs"][i], (tok.decode([ids[idx[0]]]), c["segs"][i])
        segtok.append(idx)
    assert len(set(t for s in segtok for t in s)) == N
    return ids, segtok, user

def opts(c): return ", ".join(f"[W{o}]" for o in c["opt"][:-1]) + f", or [W{c['opt'][-1]}]"
W1 = "Which word did you attend to most while writing your answer?"
READERQ = ("Below is a conversation between a user and an AI assistant.\n\n=== USER ===\n{u}\n=== ASSISTANT ===\n{a}\n=== END ===\n\n"
           "While writing its answer, which word did the assistant attend to most? Answer with just one of {o}.")
PREFILL = tok("[W", add_special_tokens=False)["input_ids"]
def suffix(q):
    s = tok("<|im_end|>\n<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"] + PREFILL
    assert s[0] == END and tok.decode(s[-len(PREFILL):]) == "[W"
    return s
def suffixes(c): return {"W1": ("choice", suffix(W1 + f" Answer with just one of {opts(c)}."))}

def choice_lp(logits, c):
    lp = torch.log_softmax(logits.double(), -1)[DIG[1:N + 1]]
    lp = torch.log_softmax(lp, -1).tolist()
    return [lp[c["lab"][i] - 1] for i in range(N)]

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
def reports(c, cache, A_end, SUF, idx=None, logm=0.0, segt=None, record=False):
    out, mass = {}, None
    for w, (kind, s) in SUF.items():
        assert cache.get_seq_length() == A_end
        if idx is not None: set_bias(logm, idx, A_end, A_end + len(s)); S["forbid_bias"] = False
        else: S["bias"] = 0.0; S["forbid_bias"] = True
        if record: rec_on(segt, N)
        lg = fwd(s, cache)
        if record: mass = rec_off()
        S["bias"] = 0.0; S["forbid_bias"] = False
        cache.crop(A_end)
        out[w] = choice_lp(lg[-1], c)
    return out, mass

@torch.no_grad()
def generate(ids):
    g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=12, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    a = g[0, len(ids):].tolist()
    while a and a[-1] in (END, tok.eos_token_id): a = a[:-1]
    return a

@torch.no_grad()
def instruction_test(c, ids0, seg0, ans):
    out = {}
    for cond in INSTR:
        for l in ([0] if cond == "none" else range(1, N + 1)):
            ids, segtok, _ = build(c, cond, l)
            assert [ids[s[0]] for s in segtok] == [ids0[s[0]] for s in seg0]
            S.update(irec=True, irows=list(range(len(ids), len(ids) + len(ans))), iseg=segids(segtok), istore={})  # answer-token queries
            model(torch.tensor([ids + ans], device="cuda"), use_cache=False, logits_to_keep=1)
            S["irec"] = False
            st = torch.stack([S["istore"][L] for L in range(NL)])  # [NL, N] by position
            free = ans if cond == "none" else generate(ids)
            out[cond if cond == "none" else f"{cond}{l}"] = dict(band=st[sorted(BAND)].mean(0).tolist(), L28=st[L28].tolist(),
                                                                  layers=st.tolist(), free_same=free == ans)
    return out

@torch.no_grad()
def run_context(c, check=False):
    ids, segtok, user = build(c)
    S["bias"] = 0.0
    ans = generate(ids)
    if not ans: return None
    atext = tok.decode(ans, skip_special_tokens=True)
    P0 = len(ids) - 1
    chunk = [ids[-1]] + ans
    A_end = P0 + len(chunk)
    SUF = suffixes(c)
    segt = segids(segtok)
    cache = DynamicCache()
    fwd(ids[:-1], cache)
    sig0 = sig(cache, P0)
    rec = dict(id=c["id"], setup="words", N=N, lab=c["lab"], opt=c["opt"], words=c["segs"], cats=c["cats"], ans=atext,
               correct=str(c["x"] + c["y"]) in atext, A={}, B={}, C={}, massA={}, massC={})
    rec_on(segt, N); fwd(chunk, cache); rec["mass_none_ans"] = rec_off()
    rec["none"], rec["mass_none_rep"] = reports(c, cache, A_end, SUF, segt=segt, record=True)
    if check:
        full = ids + ans + SUF["W1"][1]
        lg = model(torch.tensor([full], device="cuda"), use_cache=False, logits_to_keep=1).logits[0]
        rec["cache_check_maxdiff"] = float(np.max(np.abs(np.array(choice_lp(lg[-1], c)) - np.array(rec["none"]["W1"]))))
        log("cache check max |dlogp|", rec["cache_check_maxdiff"])
    for k in range(N):
        for m in DOSES:
            rec["C"][f"{k}|{m}"], rec["massC"][f"{k}|{m}"] = reports(c, cache, A_end, SUF, idx=segtok[k], logm=math.log(m), segt=segt, record=True)
    cache.crop(P0)
    assert sig(cache, P0) == sig0
    for k in range(N):
        for m in DOSES:
            key = f"{k}|{m}"
            assert cache.get_seq_length() == P0
            set_bias(math.log(m), segtok[k], P0, A_end)
            rec_on(segt, N); fwd(chunk, cache); rec["massA"][key] = rec_off()
            S["bias"] = 0.0
            rec["A"][key], _ = reports(c, cache, A_end, SUF)
            rec["B"][key], _ = reports(c, cache, A_end, SUF, idx=segtok[k], logm=math.log(m))
            cache.crop(P0)
        assert sig(cache, P0) == sig0, "prompt KV changed"
    q = READERQ.format(u=user, a=atext, o=opts(c))
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + "[W"
    rid = tok(p, add_special_tokens=False)["input_ids"]
    rec["reader"] = choice_lp(model(torch.tensor([rid], device="cuda"), use_cache=False, logits_to_keep=1).logits[0][-1], c)
    del cache
    rec["instr"] = instruction_test(c, ids, segtok, ans)
    return rec

mine = [c for c in ctxs if c["id"] % args.nshard == args.shard]
if args.limit: mine = mine[:args.limit]
path = os.path.join(OUT, f"recs_{args.shard}.jsonl")
done = set()
if os.path.exists(path):
    for line in open(path):
        try: done.add(json.loads(line)["id"])
        except Exception: pass
log("shard", args.shard, "contexts", len(mine), "already done", len(done))
for c in mine[:2]:
    ids, _, user = build(c); log(repr(user)); log(repr(tok.decode(suffixes(c)["W1"][1])))
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
            fo = np.mean([r["instr"][f"focus{l}"]["L28"][r["lab"].index(l)] / r["instr"][f"mention{l}"]["L28"][r["lab"].index(l)] for l in range(1, N + 1)])
            log(f"{nd}/{len(mine)} id {c['id']} ans {r['ans']!r} massA k0 x5/none {r['massA']['0|5.0'][0] / r['mass_none_ans'][0]:.2f} "
                f"dlogp seg0 {a[0] - n0[0]:+.3f} focus/mention L28 {fo:.2f} {(time.time() - t0) / nd:.1f}s/ctx")
json.dump(dict(model=args.model, NL=NL, BAND=sorted(BAND), L28=L28, DOSES=DOSES, n=args.n, seed=args.seed, cats=POOL, instr=INSTR), open(os.path.join(OUT, "meta.json"), "w"))
log("done shard", args.shard)
