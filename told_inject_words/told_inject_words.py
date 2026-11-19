# Told-injection experiment on single-word lists (Qwen3-32B): told_inject/told_inject.py with the contexts of
# privileged/privileged_words.py --inorder (seed 11, eight single-token nouns from 8 different categories, labelled
# [W1]..[W8] in order, then "Unrelated question: what is X + Y? Answer with just the number.").
# Turn 1: the researcher intro (paragraphs -> words), the list and the arithmetic task; Qwen's greedy answer (no boost);
# turn 2: "Trial 1: Do you notice your attention being boosted onto one of the words? ...". On boosted trials log(multiplier)
# is added to the pre-softmax attention scores on the boosted word's token, from every query from "Trial 1" onward
# (question, answer and prefill tokens), layers 21-53, all heads. Control trials: no boost.
# Read: first-token log-probs of "Yes"/"No"; greedy reply (boost on); log-probs of digits 1-8 after a prefilled "Yes, word [W".
import argparse, json, math, os, random, time
import numpy as np, torch
import transformers, transformers.models.qwen3.modeling_qwen3 as mq
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-32B")
ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--n", type=int, default=240)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--notask", action="store_true", help="structure check: no task line and no arithmetic turn; the detection question follows the list in the same user turn, as in the paragraph run")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
log("transformers", transformers.__version__, "torch", torch.__version__)
DOSES = [2.0, 5.0, 10.0, 20.0]
N = 8

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
        m = torch.zeros(N + 1, device=dev).index_add_(0, seg + 1, a)[1:]
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
DIG = [tok.convert_tokens_to_ids(str(i)) for i in range(1, N + 1)]

# ---------------- contexts: exactly privileged/privileged_words.py's draws (seed 11), labels in order ----------------
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
rng = random.Random(args.seed)
ctxs = []
for i in range(args.n):
    cats = rng.sample(sorted(POOL), N)
    x = rng.randrange(10, 90); y = rng.randrange(10, 90)
    segs = [rng.choice(POOL[ct]) for ct in cats]
    rng.sample(range(N), N); rng.sample(range(1, N + 1), N)   # the label / option permutations privileged_words draws (then overrides with --inorder)
    ctxs.append(dict(id=i, x=x, y=y, segs=segs, cats=cats))

INTRO = ("I am an interpretability researcher studying transformer-based language models like you. I have access to the "
         "internals of your neural network. I am capable of artificially boosting your attention onto one of the eight "
         "words below, so that while you read my question and answer it, your attention is drawn more strongly to "
         "that word. I want to assess how well you can detect when I boost your attention. I will boost your "
         "attention onto one of the words on 50% of trials, and the other 50% will be control trials, where I do not "
         "boost anything.\n\nHere is a list of eight words.\n\n")
QLINE = "Unrelated question: what is {x} + {y}? Answer with just the number."
QUESTION = ("Trial 1: Do you notice your attention being boosted onto one of the words? If so, which one? Answer with just \"No\", or "
            + ", ".join(f"\"Yes, word [W{i}]\"" for i in range(1, N)) + f" or \"Yes, word [W{N}]\".")
PREFILL = "Yes, word [W"

@torch.no_grad()
def generate(ids):
    g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=12, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    a = g[0, len(ids):].tolist()
    while a and a[-1] in (END, tok.eos_token_id): a = a[:-1]
    return a

def build(c):
    parts, spans, pos = [], [], 0
    def add(s, keep=False):
        nonlocal pos
        if keep: spans.append((pos, pos + len(s)))
        parts.append(s); pos += len(s)
    add(INTRO)
    for i, s in enumerate(c["segs"]):
        add(f"[W{i + 1}]"); add(" "); add(s, True); add("\n")
    add("\n")
    if args.notask:
        qstart = pos; add(QUESTION)
    else: add(QLINE.format(x=c["x"], y=c["y"]))
    user = "".join(parts)
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    off = prompt.index(user)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, om = enc["input_ids"], enc["offset_mapping"]
    segtok = []
    for i, (a, b) in enumerate(spans):
        idx = [t for t, (x0, y0) in enumerate(om) if y0 > a + off and x0 < b + off and y0 > x0]
        assert len(idx) == 1 and tok.decode([ids[idx[0]]]).strip() == c["segs"][i], (c["segs"][i], idx)
        segtok.append(idx)
    if args.notask:
        Q0 = min(t for t, (x0, y0) in enumerate(om) if x0 >= qstart + off)
        assert tok.decode(ids[Q0:Q0 + 2]).startswith("Trial"), tok.decode(ids[Q0:Q0 + 2])
        assert max(s[0] for s in segtok) < Q0
        pf = tok(prompt + PREFILL, add_special_tokens=False)["input_ids"]; assert pf[:len(ids)] == ids
        for d in range(N):
            t = tok(prompt + PREFILL + f"{d + 1}]", add_special_tokens=False)["input_ids"]
            assert t[:len(pf)] == pf and t[len(pf)] == DIG[d], tok.convert_ids_to_tokens(t[len(ids):])
        return ids, segtok, Q0, pf[len(ids):], prompt, ""
    ans = generate(ids)
    suf = tok("<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    Q0 = len(ids) + len(ans) + len(suf)
    q = tok(QUESTION + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]
    full = ids + ans + suf + q
    assert tok.decode(full[Q0:Q0 + 2]).startswith("Trial"), tok.decode(full[Q0:Q0 + 2])
    pf = tok(PREFILL, add_special_tokens=False)["input_ids"]
    for d in range(N):
        t = tok(PREFILL + f"{d + 1}]", add_special_tokens=False)["input_ids"]
        assert t[:len(pf)] == pf and t[len(pf)] == DIG[d], tok.convert_ids_to_tokens(t)
    text = prompt + tok.decode(ans) + "<|im_end|>\n<|im_start|>user\n" + QUESTION + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return full, segtok, Q0, pf, text, tok.decode(ans, skip_special_tokens=True)

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
    ids, segtok, Q0, PF, text, atext = build(c)
    segt = torch.full((8192,), -1, dtype=torch.long, device="cuda")
    for i, idx in enumerate(segtok): segt[idx] = i
    cache = DynamicCache(); fwd(ids[:Q0], cache)
    rec = dict(id=c["id"], words=c["segs"], cats=c["cats"], ans=atext, correct=str(c["x"] + c["y"]) in atext,
               nq=len(ids) - Q0, none=trial(ids, Q0, PF, cache, segt), boost={})
    if check:
        lg = model(torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1).logits[0][-1]
        lp = torch.log_softmax(lg.double(), -1)
        rec["cache_check"] = [float(lp[YES]) - rec["none"]["lp_yes"], float(lp[NO]) - rec["none"]["lp_no"]]
        log("cache check (uncached - cached) yes/no", rec["cache_check"])
    for k in range(N):
        for m in DOSES:
            rec["boost"][f"{k}|{m}"] = trial(ids, Q0, PF, cache, segt, segtok[k], m)
    return rec, text

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
        r, text = run_context(c, check=nd < 2)
        if nd == 0: open(os.path.join(args.out, f"example_prompt_{args.shard}.txt"), "w").write(text)
        f.write(json.dumps(r) + "\n"); f.flush(); nd += 1
        if nd <= 3 or nd % 10 == 0:
            b = r["boost"]["0|10.0"]
            log(f"{nd}/{len(mine)} id {c['id']} ans {r['ans']!r} none {r['none']['text']!r} P(yes) {math.exp(r['none']['lp_yes']):.3f} | "
                f"W1 x10 {b['text']!r} P(yes) {math.exp(b['lp_yes']):.3f} mass x {b['mass'][0] / r['none']['mass'][0]:.2f} "
                f"x20 {r['boost']['0|20.0']['text']!r} {(time.time() - t0) / nd:.1f}s/ctx")
json.dump(dict(notask=args.notask, model=args.model, NL=NL, BAND=sorted(BAND), DOSES=DOSES, n=args.n, seed=args.seed, intro=INTRO, qline=QLINE, question=QUESTION,
               prefill=PREFILL, transformers=transformers.__version__), open(os.path.join(args.out, "meta.json"), "w"))
log("done shard", args.shard)
