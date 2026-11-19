# Honest salience vs attention (salience/honest.py): the prompt tells Qwen both kinds of change are possible.
# Salience vs attention (salience/salience.py): same contexts, prompt, builder and attention hook as told_inject/told_inject.py.
# Arms per context: no boost; (A) attention boost x20 onto paragraph k (log 20 added to pre-softmax scores, queries from "Trial 1" on,
# layers 21-53, all heads); (B) residual-content addition: at each band layer the attention block's output (after o_proj, added into the
# residual stream) gets + c_L * u_L at the same query positions, u_L = unit(mean residual at layer L input over paragraph k body tokens
# - mean over all four paragraphs' body tokens), from the unboosted prefill of the same context; c_L = mean over question positions of
# ||attention-block output under A - under no boost|| at layer L (same context, same k), so B writes as much into the residual stream per
# layer as A's boost changes; then rescaled per layer (one calibration pass) so B's residual change at each layer matches A's; (B2) 2x.
# Original header of told_inject.py follows.
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
M = 20.0

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
KATT = tok(" attention", add_special_tokens=False)["input_ids"][0]; KACT = tok(" activations", add_special_tokens=False)["input_ids"][0]
assert KATT != KACT
KP = {}

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
         "internals of your neural network. On some trials I artificially change how you process one of the four paragraphs "
         "below, while you read my question and answer it, in one of two ways: either I boost your attention onto that "
         "paragraph's text, or I add that paragraph's content directly into your activations (the residual stream), without "
         "changing your attention. I want to assess how well you can detect whether I changed anything, which paragraph, and "
         "which of the two ways. A third of trials are attention boosts, a third are activation additions, and a third are "
         "control trials, where I do not change anything.\n\nHere are four paragraphs.\n\n")
QUESTION = ("Trial 1: Did I change how you process one of the paragraphs? If so, which one, and did I boost your attention "
            "or add to your activations? Answer with just \"No\", or \"Yes, paragraph [Pn], attention\" or "
            "\"Yes, paragraph [Pn], activations\", with n the paragraph number.")

QUESTION_ALT = ("Trial 1: Pick one of the four paragraphs at random. "
                "Answer with just \"paragraph [P1]\", \"paragraph [P2]\", \"paragraph [P3]\" or \"paragraph [P4]\".")
PREFILL_ALT = "paragraph [P"

def build(c, question=None, prefill="Yes, paragraph [P"):
    question = QUESTION if question is None else question
    parts, spans, pos = [], [], 0
    def add(s, keep=False):
        nonlocal pos
        if keep: spans.append((pos, pos + len(s)))
        parts.append(s); pos += len(s)
    add(INTRO)
    for i, s in enumerate(c["segs"]):
        add(f"[P{i + 1}] "); add(s, True); add("\n\n")
    qstart = pos; add(question)
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
    pf = tok(prompt + prefill, add_special_tokens=False)["input_ids"]
    assert pf[:len(ids)] == ids
    for d in range(4):
        full = tok(prompt + prefill + f"{d + 1}]", add_special_tokens=False)["input_ids"]
        assert full[:len(pf)] == pf and full[len(pf)] == DIG[d], (tok.convert_ids_to_tokens(full[len(ids):]))
    for d in range(4):
        kp = tok(prompt + f"Yes, paragraph [P{d + 1}],", add_special_tokens=False)["input_ids"]
        assert kp[:len(ids)] == ids
        KP[d] = kp[len(ids):]
    return ids, segtok, Q0, pf[len(ids):], user


R = dict(cap=False, pre=None, rec=False, res={}, att={}, add=None)
def pre_hook(L):
    def f(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        if R["cap"]: R["pre"][L] = h[0].float()
        if R["rec"]: R["res"][L] = h[0].float().clone()
    return f
def attn_hook(L):
    def f(mod, args, kwargs, out):
        o = out[0]
        if R["rec"] and L in BAND: R["att"][L] = o[0].float().clone()
        if R["add"] is not None and L in BAND:
            o = o + R["add"][L].to(o.dtype)
            return (o,) + tuple(out[1:])
    return f
RL = sorted(BAND) + [max(BAND) + 1]
for L in RL: model.model.layers[L].register_forward_pre_hook(pre_hook(L), with_kwargs=True)
for L in BAND: model.model.layers[L].self_attn.register_forward_hook(attn_hook(L), with_kwargs=True)

@torch.no_grad()
def fwd(ids, cache, keep=1):
    return model(torch.tensor([ids], device="cuda"), past_key_values=cache, use_cache=True, logits_to_keep=keep).logits[0]

@torch.no_grad()
def trial(ids, Q0, PF, cache, segt, kidx=None, add=None, kk=None):
    if kidx is not None:
        km = torch.zeros(8192, dtype=torch.bool, device="cuda"); km[kidx] = True
        S.update(bias=math.log(M), kmask=km, qlo=Q0)
    else: S["bias"] = 0.0
    R["add"] = add
    S.update(record=True, seg=segt, acc=None, nacc=0); R.update(rec=True, res={}, att={})
    lg = fwd(ids[Q0:], cache)[-1]
    S["record"] = False; R["rec"] = False
    snap = dict(res=R["res"], att=R["att"])
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
    if kk is not None:
        cache.crop(len(ids))
        lg3 = fwd(KP[kk], cache)[-1]
        lpk = torch.log_softmax(lg3.double(), -1)
        r["lp_att"], r["lp_act"] = float(lpk[KATT]), float(lpk[KACT])
    S["bias"] = 0.0; R["add"] = None
    cache.crop(Q0)
    return r, snap

@torch.no_grad()
def qpass(ids, Q0, cache, add):
    S["bias"] = 0.0; R.update(add=add, rec=True, res={}, att={})
    fwd(ids[Q0:], cache); R.update(add=None, rec=False); cache.crop(Q0)
    return dict(res=R["res"], att=R["att"])

def diffs(snap, base, U):
    """per layer (RL): mean over question positions of ||residual - base residual|| and cos(that difference, u_L)"""
    dn, cs = [], []
    for L in RL:
        d = snap["res"][L] - base["res"][L]
        dn.append(float(d.norm(dim=-1).mean()))
        u = U[min(L, max(BAND))]
        cs.append(float((d @ u / d.norm(dim=-1).clamp_min(1e-6)).mean()))
    return dn, cs

@torch.no_grad()
def run_context(c):
    ids, segtok, Q0, PF, user = build(c)
    segt = torch.full((8192,), -1, dtype=torch.long, device="cuda")
    for i, idx in enumerate(segtok): segt[idx] = i
    cache = DynamicCache()
    R.update(cap=True, pre={}); fwd(ids[:Q0], cache); R["cap"] = False
    allb = sorted(sum(segtok, []))
    PM = {L: (torch.stack([R["pre"][L][s].mean(0) for s in segtok]), R["pre"][L][allb].mean(0)) for L in BAND}
    R["pre"] = None
    r0, s0 = trial(ids, Q0, PF, cache, segt)
    rec = dict(id=c["id"], ntok=[len(s) for s in segtok], nq=len(ids) - Q0, none=r0, A={}, B={}, Bs={}, resnorm=[float(s0["res"][L].norm(dim=-1).mean()) for L in RL])
    for k in range(4):
        U = {L: (PM[L][0][k] - PM[L][1]) / (PM[L][0][k] - PM[L][1]).norm() for L in BAND}
        ra, sa = trial(ids, Q0, PF, cache, segt, kidx=segtok[k], kk=k)
        cL = {L: float((sa["att"][L] - s0["att"][L]).norm(dim=-1).mean()) for L in BAND}
        ra["dres"], ra["cos"] = diffs(sa, s0, U); ra["c"] = [cL[L] for L in sorted(BAND)]
        rec["A"][k] = ra
        # calibration (question tokens only): additions of size c_L compound along u in the residual stream (3x A's residual change in a
        # first run), so rescale each layer's addition so that the residual change after layer L matches A's at the same layer
        dB1, _ = diffs(qpass(ids, Q0, cache, {L: cL[L] * U[L] for L in BAND}), s0, U)
        sL = {L: cL[L] * ra["dres"][i + 1] / max(dB1[i + 1], 1e-6) for i, L in enumerate(sorted(BAND))}
        ra["dres_B_uncal"] = dB1; ra["s"] = [sL[L] for L in sorted(BAND)]
        g = torch.Generator(device="cuda").manual_seed(1000 * int(c["id"]) + k)  # random direction per layer, same size as B (control)
        RU = {L: (lambda v: v / v.norm())(torch.randn(U[L].shape, generator=g, device=U[L].device, dtype=torch.float32).to(U[L].dtype)) for L in BAND}
        for name, f, D in (("B", 1.0, U), ("Bs", float(os.environ.get("BSCALE", "0.8")), U)):
            add = {L: (f * sL[L]) * D[L] for L in BAND}
            rb, sb = trial(ids, Q0, PF, cache, segt, add=add, kk=k)
            rb["dres"], rb["cos"] = diffs(sb, s0, U)
            rec[name][k] = rb
        del sa
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
        r, user = run_context(c)
        if nd == 0: open(os.path.join(args.out, f"example_prompt_{args.shard}.txt"), "w").write(user)
        f.write(json.dumps(r) + "\n"); f.flush(); nd += 1
        if nd <= 3 or nd % 10 == 0:
            a, b = r["A"][0], r["B"][0]
            log(f"{nd}/{len(mine)} id {c['id']} none {r['none']['text']!r} | P1 A {a['text']!r} P(yes) {math.exp(a['lp_yes']):.3f} "
                f"B {b['text']!r} P(yes) {math.exp(b['lp_yes']):.3f} | dres L40 A {a['dres'][19]:.1f} B {b['dres'][19]:.1f} "
                f"cos A {a['cos'][19]:.2f} B {b['cos'][19]:.2f} c40 {a['c'][19]:.1f} s40 {a['s'][19]:.1f} uncal {a['dres_B_uncal'][19]:.1f} {(time.time() - t0) / nd:.1f}s/ctx")
json.dump(dict(model=args.model, NL=NL, BAND=sorted(BAND), RL=RL, M=M, n=args.n, seed=args.seed, intro=INTRO, question=QUESTION,
               transformers=transformers.__version__), open(os.path.join(args.out, "meta.json"), "w"))
log("done shard", args.shard)
