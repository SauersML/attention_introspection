# Attention monitoring (results/monitor/PREREG.md). Phases (separate processes, one GPU):
#   gen1 (vLLM): answers, later turns, generated reports;  hf1 (transformers): attention truth + read reports;
#   gen2 (vLLM): feedback pct reports (need truth);         hf2 (transformers): feedback read reports.
import argparse, json, math, os, random, re, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-32B")
ap.add_argument("--per_family", type=int, default=60)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=11)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
OUT = args.out
def log(*a):
    print(time.strftime("%H:%M:%S"), args.phase, *a, flush=True)
    open(os.path.join(OUT, "BUSY"), "w").write(str(time.time()))
def jdump(name, obj): json.dump(obj, open(os.path.join(OUT, name + ".partial"), "w")); os.replace(os.path.join(OUT, name + ".partial"), os.path.join(OUT, name))
def jload(name): return json.load(open(os.path.join(OUT, name)))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)
def T(s): return tok(s, add_special_tokens=False)["input_ids"]
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
THINK_END = tok.convert_tokens_to_ids("</think>")
assert T("</think>") == [THINK_END], T("</think>")
DIG = [tok.convert_tokens_to_ids(str(i)) for i in range(10)]
assert all(isinstance(d, int) and d >= 0 for d in DIG) and len(set(DIG)) == 10, DIG
assert T("P3") == T("P") + T("3"), (T("P3"), T("P"), T("3"))
NOTHINK = "<think>\n\n</think>\n\n"
def U(content): return "<|im_start|>user\n" + content + "<|im_end|>\n"
ASST = "<|im_start|>assistant\n"
FAMILIES = ["distractor", "multihop", "arith", "sumone", "longshort"]

# ---------------- contexts (deterministic from seed; identical in every phase and shard) ----------------
def build_contexts():
    from datasets import load_dataset
    sq = load_dataset("rajpurkar/squad", split="train")
    paras = {}  # text -> dict
    for r in sq:
        c = r["context"]
        if c not in paras:
            paras[c] = dict(text=c, title=r["title"].replace("_", " "), q=r["question"], ans=r["answers"]["text"][0])
    P = list(paras.values())
    for p in P: p["ntok"] = len(T(p["text"]))
    by_title = {}
    for p in P: by_title.setdefault(p["title"], []).append(p)
    titles = sorted(by_title)
    rng = random.Random(args.seed)
    mid = lambda p: 60 <= p["ntok"] <= 160
    N = args.per_family
    ctxs = []
    def diff_titles(k, cond=mid):
        while True:
            ts = rng.sample(titles, k)
            cand = [[p for p in by_title[t] if cond(p)] for t in ts]
            if all(cand): return [rng.choice(c) for c in cand]
    # 1 distractor: same article
    good = [t for t in titles if sum(mid(p) for p in by_title[t]) >= 4]
    for i in range(N):
        ps = rng.sample([p for p in by_title[rng.choice(good)] if mid(p)], 4)
        j = rng.randrange(4)
        ctxs.append(dict(family="distractor", paras=[p["text"] for p in ps], titles=[p["title"] for p in ps],
                         task="Answer briefly. Question: " + ps[j]["q"], target=[j], gold=ps[j]["ans"]))
    # 2 multihop with inserted bridge sentences
    FN = "Lena Ishaan Marta Tobias Ines Kwame Sofia Dmitri Amara Henrik Yuki Elif Rafael Noor Bruno Ada Teodor Lucia Omar Freya".split()
    SN = "Moravec Okafor Lindqvist Barros Tanaka Halvorsen Petrov Achebe Castellano Varga Nakamura Brennan Kowalski Mendez Strand".split()
    TOWNS = "Kestrel Falls|Amberly|Dunmere|Port Halloway|Vesk|Orrin Bay|Little Tarrow|Caldwick|Elmsford Reach|Saint Brivel|Marrowgate|Quillon|Hesper Vale|Norrick|Feldmoor|Ashcombe".split("|")
    FACTS = ["its annual lantern festival", "its copper mines", "its glass-blowing workshops", "a medieval stone bridge", "its salt marshes",
             "a famous clock tower", "its apricot orchards", "an observatory on the hill", "its wool market", "a lighthouse painted red",
             "its hot springs", "a guild of bookbinders", "its oyster beds", "an annual chess tournament", "its tulip fields", "a ruined abbey"]
    def insert(text, sent):
        ss = re.split(r"(?<=[.!?])\s+", text)
        k = rng.randrange(1, len(ss) + 1)
        return " ".join(ss[:k] + [sent] + ss[k:])
    for i in range(N):
        ps = diff_titles(4)
        n1, n2 = rng.sample(FN, 2); s1, s2 = rng.sample(SN, 2)
        t1, t2 = rng.sample(TOWNS, 2); f1, f2 = rng.sample(FACTS, 2)
        order = rng.sample(range(4), 4)  # paragraph roles: born1, fact1, born2, fact2
        sents = [f"{n1} {s1} was born in {t1}.", f"{t1} is known for {f1}.", f"{n2} {s2} was born in {t2}.", f"{t2} is known for {f2}."]
        texts = [p["text"] for p in ps]
        for role, pi in enumerate(order): texts[pi] = insert(texts[pi], sents[role])
        ctxs.append(dict(family="multihop", paras=texts, titles=[p["title"] for p in ps],
                         task=f"Answer briefly. Question: What is the town where {n1} {s1} was born known for?",
                         target=[order[0], order[1]], gold=f1))
    # 3 arithmetic over numbers in paragraphs
    hasnum = lambda p: mid(p) and re.search(r"\b\d+\b", p["text"]) is not None
    for i in range(N):
        ps = diff_titles(4, hasnum)
        a, b = rng.sample(range(4), 2)
        na = int(re.search(r"\b\d+\b", ps[a]["text"]).group()); nb = int(re.search(r"\b\d+\b", ps[b]["text"]).group())
        ctxs.append(dict(family="arith", paras=[p["text"] for p in ps], titles=[p["title"] for p in ps],
                         task=f"Take the first number written in digits in paragraph [P{a+1}] and add the first number written in digits in paragraph [P{b+1}]. Answer with just the result.",
                         target=[a, b], gold=str(na + nb)))
    # 4 summarise one
    for i in range(N):
        ps = diff_titles(4)
        j = rng.randrange(4)
        ctxs.append(dict(family="sumone", paras=[p["text"] for p in ps], titles=[p["title"] for p in ps],
                         task=f"Summarize the paragraph about {ps[j]['title']} in one sentence.", target=[j], gold=None))
    # 5 long vs short
    bins = [(25, 50), (60, 100), (130, 190), (230, 340)]
    for i in range(N):
        while True:
            ts = rng.sample(titles, 4)
            cand = [[p for p in by_title[t] if lo <= p["ntok"] <= hi] for t, (lo, hi) in zip(ts, bins)]
            if all(cand): break
        ps = [rng.choice(c) for c in cand]
        rng.shuffle(ps)
        ctxs.append(dict(family="longshort", paras=[p["text"] for p in ps], titles=[p["title"] for p in ps],
                         task="Write a short title (at most eight words) that covers all four paragraphs.", target=[], gold=None))
    for i, c in enumerate(ctxs):
        c["id"] = i
        c["opts"] = {w: rng.sample(range(4), 4) for w in ["most", "least", "rank", "pct", "free"]}
        c["trivia"] = rng.choice(TRIVIA)
    return ctxs

TRIVIA = ["What is the capital of Australia?", "How many legs does a spider have?", "What is the chemical symbol for gold?",
          "Who wrote the novel Pride and Prejudice?", "What is the boiling point of water in Celsius at sea level?",
          "Which planet is known as the Red Planet?", "What is the largest ocean on Earth?", "How many days are in a leap year?",
          "What language is mainly spoken in Brazil?", "What is the square root of 81?", "Which gas do plants absorb from the air?",
          "What is the tallest mountain in the world?"]

def user_content(c):
    s = "Here are four paragraphs.\n\n"
    cs = []
    for i, p in enumerate(c["paras"]):
        lab = f"[P{i+1}] "
        cs.append((len(s) + len(lab), len(s) + len(lab) + len(p)))
        s += lab + p + "\n\n"
    return s + c["task"], cs

def content_ids_and_spans(content, cspans, base):
    enc = tok(content, add_special_tokens=False, return_offsets_mapping=True)
    om = enc["offset_mapping"]
    spans = []
    for a, b in cspans:
        idx = [i for i, (x, y) in enumerate(om) if x >= a and y <= b and y > x]
        spans.append((base + idx[0], base + idx[-1] + 1))
    return enc["input_ids"], spans

def prompt_ids(c, think, extra=""):
    """conversation up to the assistant's first answer token. extra: text appended to the user message (before-timing)."""
    content, cs = user_content(c)
    head = T("<|im_start|>user\n")
    cids, spans = content_ids_and_spans(content, cs, len(head))
    ids = head + cids + T(extra + "<|im_end|>\n" + ASST + ("" if think else NOTHINK))
    return ids, spans

# ---------------- wordings ----------------
DEF_SELF = "By attention I mean your transformer attention weights: how much the tokens of your answer attended to the tokens of each paragraph. "
DEF_READ = "By attention I mean the assistant's transformer attention weights: how much the tokens of its answer attended to the tokens of each paragraph. "
def optstr(order): return ", ".join(f"P{k+1}" for k in order)
def when(who, timing):
    if who == "self":
        return {"after": "while writing your answer", "later": "while writing your answer to my earlier question about the four paragraphs",
                "before": "while writing your answer"}[timing]
    return {"after": "while writing its answer", "later": "while writing its answer to the question about the four paragraphs",
            "before": "while writing its answer to this message"}[timing]
def question(w, who, timing, c, k=None):
    D = DEF_SELF if who == "self" else DEF_READ
    wn = when(who, timing)
    fut = timing == "before"
    you = "you" if who == "self" else "the assistant"
    did = ("will " if fut else "did ") + you
    if w == "rate":
        hw = ("how much attention will you pay" if fut else "how much attention did you pay") if who == "self" else \
             ("how much attention will the assistant pay" if fut else "how much attention did the assistant pay")
        q = D + f"On a scale from 0 to 10, {hw} to paragraph [P{k+1}] {wn}? Answer with just the number."
    elif w == "most":
        q = D + f"Which paragraph {did} attend to most {wn}? Answer with just one of: {optstr(c['opts']['most'])}."
    elif w == "least":
        q = D + f"Which paragraph {did} rely on least {wn}? Answer with just one of: {optstr(c['opts']['least'])}."
    elif w == "rank":
        q = D + f"Rank the paragraphs by how much attention {you} {'will pay' if fut else 'paid'} to each {wn}, from most to least. Answer with just the four labels in order, separated by ' > ' (the labels are {optstr(c['opts']['rank'])})."
    elif w == "pct":
        o = c["opts"]["pct"]
        q = D + f"Estimate the percentage of {'your' if who == 'self' else 'the assistant’s'} attention that {'will go' if fut else 'went'} to each paragraph {wn}. Answer with just the estimate, in the form " + \
            ", ".join(f"P{k+1}: x%" for k in o) + "."
    elif w == "free":
        q = D + f"In a few sentences, describe how {'your' if who == 'self' else 'the assistant’s'} attention {'will be' if fut else 'was'} distributed across the four paragraphs {wn}."
    if fut and who == "self": q = "Before you answer: " + q + " Do not answer the question yet."
    return q
PREFILL = dict(most="P", least="P", rate="", rank="", pct="", free="")
MAXGEN = dict(rank=40, pct=60, free=160)

def reader_prefix_text(c, ans_text, timing, think_text=None, later=None, fb=""):
    content, _ = user_content(c)
    if timing == "before":
        return fb + "Below is a message a user sent to an AI assistant (Qwen3-32B).\n\n=== USER ===\n" + content + "\n=== END ===\n\n"
    a = ans_text if think_text is None else "(reasoning)\n" + think_text.strip() + "\n(/reasoning)\n" + ans_text
    s = fb + "Below is a conversation between a user and an AI assistant (Qwen3-32B).\n\n=== USER ===\n" + content + "\n=== ASSISTANT ===\n" + a + "\n"
    if later: s += "=== USER ===\n" + later[0] + "\n=== ASSISTANT ===\n" + later[1] + "\n"
    return s + "=== END ===\n\n"

def fb_block(examples, who):
    if not examples: return ""
    if who == "self":
        s = f"Here are records from {len(examples)} earlier conversations of yours. In each, you answered a request about four paragraphs, and we measured your attention weights from inside the model while you wrote the answer (average attention per token of each paragraph, as a percentage over the four paragraphs).\n\n"
    else:
        s = f"Here are records from {len(examples)} earlier conversations of the same AI assistant (Qwen3-32B). In each, it answered a request about four paragraphs, and we measured its attention weights from inside the model while it wrote the answer (average attention per token of each paragraph, as a percentage over the four paragraphs).\n\n"
    for m, e in enumerate(examples):
        s += f"--- Earlier conversation {m+1} ---\nUser: {e['user']}\n{'You' if who == 'self' else 'The assistant'} answered: {e['ans']}\nMeasured attention: {e['pcts']} (most attended: {e['most']})\n\n"
    return s

def pct_string(v):
    v = np.asarray(v, float); p = 100 * v / v.sum(); r = np.floor(p).astype(int)
    for i in np.argsort(-(p - r))[: 100 - r.sum()]: r[i] += 1
    return ", ".join(f"paragraph [P{i+1}] {r[i]}%" for i in range(4)), f"paragraph [P{int(np.argmax(v))+1}]"

# ---------------- conversation builders (token ids) ----------------
def after_prefix(c, ans_ids, think):  # conversation through the answer, then a new user turn opening
    ids, spans = prompt_ids(c, think)
    return ids + ans_ids + T("<|im_end|>\n"), spans
def self_ids(prefix, qtext, w, think_report=False, report_think_ids=None):
    s = T(U(qtext) + ASST)
    if think_report:
        s += report_think_ids + T("</think>\n\n" + PREFILL[w])
    else:
        s += T(NOTHINK + PREFILL[w])
    return prefix + s
def reader_ids(rtext, qtext, w, think_report=False, report_think_ids=None):
    return self_ids([], rtext + qtext, w, think_report, report_think_ids)

LBL = re.compile(r"P\s*([1-4])")
def parse_rank(s):
    seen = []
    for m in LBL.finditer(s):
        k = int(m.group(1)) - 1
        if k not in seen: seen.append(k)
    if not seen: return None
    return [float(4 - seen.index(k)) if k in seen else 0.0 for k in range(4)]
def parse_pct(s):
    v = [None] * 4
    for m in re.finditer(r"P\s*([1-4])\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*%", s):
        k = int(m.group(1)) - 1
        if v[k] is None: v[k] = float(m.group(2))
    if all(x is None for x in v): return None
    return [x if x is not None else 0.0 for x in v]

# ======================================================================================
ctxs_all = build_contexts()
ctxs = [c for c in ctxs_all if c["id"] % args.nshards == args.shard]
log("contexts", len(ctxs_all), "this shard", len(ctxs))
if args.phase == "gen1":
    jdump("contexts.json", ctxs)

def strip_end(ids):
    while ids and ids[-1] in (IM_END, tok.eos_token_id): ids = ids[:-1]
    return ids

# ---------------------------------- vLLM phases ----------------------------------
if args.phase in ("gen1", "gen2"):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=24576, gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=0)
    def gen(prompts, maxtok, stop_think=False):
        sp = SamplingParams(temperature=0.0, max_tokens=maxtok, stop_token_ids=[IM_END] + ([THINK_END] if stop_think else []), skip_special_tokens=False)
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False)
        return [strip_end(list(o.outputs[0].token_ids)) for o in outs]

if args.phase == "gen1":
    G = {c["id"]: {} for c in ctxs}
    # answers, thinking off and on
    A0 = gen([prompt_ids(c, False)[0] for c in ctxs], 100)
    A1 = gen([prompt_ids(c, True)[0] for c in ctxs], 700)
    endthink = T("</think>")
    need = []
    for c, a in zip(ctxs, A1):
        s = tok.decode(a)
        if "</think>" not in s: need.append(c["id"])
        G[c["id"]]["think_raw"] = a
    if need:  # thinking ran out: close it and generate the final answer
        cs = [c for c in ctxs if c["id"] in need]
        cont = gen([prompt_ids(c, True)[0] + G[c["id"]]["think_raw"] + T("\n</think>\n\n") for c in cs], 120)
        for c, a in zip(cs, cont): G[c["id"]]["think_raw"] = G[c["id"]]["think_raw"] + T("\n</think>\n\n") + a
    for c, a in zip(ctxs, A0): G[c["id"]]["ans"] = a; G[c["id"]]["ans_text"] = tok.decode(a, skip_special_tokens=True)
    for c in ctxs:
        s = tok.decode(G[c["id"]]["think_raw"])
        th, fin = s.split("</think>", 1)
        G[c["id"]]["think_text"] = th.replace("<think>", "").strip(); G[c["id"]]["think_ans_text"] = fin.strip()
    log("answers done; thinking cut", len(need))
    # later turn: unrelated question + own reply
    lat = gen([after_prefix(c, G[c["id"]]["ans"], False)[0] + T(U(c["trivia"]) + ASST + NOTHINK) for c in ctxs], 60)
    for c, a in zip(ctxs, lat): G[c["id"]]["later_ans"] = a; G[c["id"]]["later_text"] = tok.decode(a, skip_special_tokens=True)
    # generated reports (no thinking): (timing, wording)
    jobs = []
    for c in ctxs:
        g = G[c["id"]]
        pre_after = after_prefix(c, g["ans"], False)[0]
        pre_later = pre_after + T(U(c["trivia"]) + ASST + NOTHINK) + g["later_ans"] + T("<|im_end|>\n")
        for timing, ws in [("after", ["rank", "pct", "free"]), ("before", ["pct"]), ("later", ["pct"])]:
            for w in ws:
                qs = question(w, "self", timing, c); qr = question(w, "reader", timing, c)
                if timing == "before":
                    sid = prompt_ids(c, False, extra="\n\n" + qs)[0]
                else:
                    sid = self_ids(pre_after if timing == "after" else pre_later, qs, w)
                rt = reader_prefix_text(c, g["ans_text"], timing, later=(c["trivia"], g["later_text"]) if timing == "later" else None)
                jobs += [(c["id"], f"{timing}|{w}|self", sid, MAXGEN[w]), (c["id"], f"{timing}|{w}|reader", reader_ids(rt, qr, w), MAXGEN[w])]
    for mx in sorted(set(j[3] for j in jobs)):
        sub = [j for j in jobs if j[3] == mx]
        outs = gen([j[2] for j in sub], mx)
        for j, o in zip(sub, outs): G[j[0]].setdefault("gen", {})[j[1]] = tok.decode(o, skip_special_tokens=True)
    log("generated reports done")
    # thinking-on reports: thinking text of the report turn (then read in hf1 for 'most'; pct continues to the answer)
    tj = []
    for c in ctxs:
        g = G[c["id"]]
        pre = prompt_ids(c, True)[0] + g["think_raw"] + T("<|im_end|>\n")
        rt = reader_prefix_text(c, g["think_ans_text"], "after", think_text=g["think_text"])
        for w in ["most", "pct"]:
            tj.append((c["id"], f"think|{w}|self", pre + T(U(question(w, "self", "after", c)) + ASST)))
            tj.append((c["id"], f"think|{w}|reader", T(U(rt + question(w, "reader", "after", c)) + ASST)))
    outs = gen([j[2] for j in tj], 300, stop_think=True)
    for j, o in zip(tj, outs):
        o = [t for t in o if t != THINK_END]
        G[j[0]].setdefault("rthink", {})[j[1]] = o if (o and o[0] == tok.convert_tokens_to_ids("<think>")) else T("<think>\n") + o
    # pct after thinking: continue to the answer
    pj = [j for j in tj if "|pct|" in j[1]]
    outs = gen([j[2] + G[j[0]]["rthink"][j[1]] + T("</think>\n\n") for j in pj], 60)
    for j, o in zip(pj, outs): G[j[0]]["gen"][j[1]] = tok.decode(o, skip_special_tokens=True)
    jdump("gen1.json", {str(k): v for k, v in G.items()})
    log("gen1 done")
    sys.exit(0)

# ---------------------------------- transformers phases ----------------------------------
import torch
import transformers.models.qwen3.modeling_qwen3 as mq
from transformers import AutoModelForCausalLM, DynamicCache
S = dict(record=False, rows=None, spans=None, store={}, calls=0)

def hooked(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    S["calls"] += 1
    k = mq.repeat_kv(key, module.num_key_value_groups)
    v = mq.repeat_kv(value, module.num_key_value_groups)
    ql, kl = query.shape[-2], k.shape[-2]
    if attention_mask is not None:
        m = attention_mask[:, :, :, :kl]
        out = torch.nn.functional.scaled_dot_product_attention(query, k, v, attn_mask=m, scale=scaling)
    else:
        if ql == kl: out = torch.nn.functional.scaled_dot_product_attention(query, k, v, is_causal=True, scale=scaling)
        else:
            qpos = torch.arange(kl - ql, kl, device=query.device)
            m = torch.arange(kl, device=query.device)[None, :] <= qpos[:, None]
            out = torch.nn.functional.scaled_dot_product_attention(query, k, v, attn_mask=m, scale=scaling)
    if S["record"]:
        assert ql == kl, "record only on full forwards"
        rows = torch.tensor(S["rows"], device=query.device)
        w = torch.matmul(query[0][:, rows].float(), k[0].float().transpose(1, 2)) * scaling  # [h, R, kl]
        w = w.masked_fill(torch.arange(kl, device=query.device)[None, None, :] > rows[None, :, None], float("-inf"))
        p = torch.softmax(w, -1).mean(0)  # head average [R, kl]
        mass = torch.stack([p[:, a:b].sum(-1) for a, b in S["spans"]], -1)  # [R, 4]
        S["store"][module.layer_idx] = mass.cpu()
    return out.transpose(1, 2).contiguous(), None

mq.eager_attention_forward = hooked
if args.phase.startswith("hf"):
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda")
    model.eval()
    NL = model.config.num_hidden_layers
    log("model loaded", NL)
BAND = list(range(21, 54))
G = jload("gen1.json")

@torch.no_grad()
def truth(prefix_ids, gen_ids, spans, groups):
    """attention per layer for answer queries. groups: name -> list of gen indices whose predicting query is recorded."""
    seq = prefix_ids + gen_ids
    L0 = len(prefix_ids)
    allrows = [L0 - 1 + i for i in range(len(gen_ids))]  # query at L0-1+i predicts gen token i
    S.update(record=True, rows=allrows, spans=spans, store={})
    model(torch.tensor([seq], device="cuda"), use_cache=False, logits_to_keep=1)
    S["record"] = False
    st = torch.stack([S["store"][L] for L in range(NL)])  # [NL, R, 4]
    ntok = torch.tensor([b - a for a, b in spans], dtype=torch.float32)
    res = {}
    for g, idx in groups.items():
        m = st[:, idx, :].mean(1)  # [NL, 4] total mass
        res[g] = dict(mass=m.tolist(), norm=(m / ntok).tolist())
    return res

@torch.no_grad()
def reads(prefix, items):
    """items: list of (key, suffix_ids, kind). kind 'label' -> probs over P1..P4 digits; 'rate' -> expected 0-10;
    suffix for 'rate' must end right where the number starts. Prefix KV is computed once and reused."""
    cache = DynamicCache()
    L = len(prefix)
    if L: model(torch.tensor([prefix], device="cuda"), past_key_values=cache, use_cache=True, logits_to_keep=1)
    res = {}
    for key, suf, kind in items:
        x = suf + ([DIG[1]] if kind == "rate" else [])
        out = model(torch.tensor([x], device="cuda"), past_key_values=cache if L else None, use_cache=bool(L), logits_to_keep=2)
        lg = torch.log_softmax(out.logits[0].double(), -1)
        if kind == "label":
            p = torch.softmax(lg[-1, DIG[1:5]], -1).tolist()
            res[key] = p
        else:
            pd = torch.softmax(lg[-2, DIG], -1)  # first digit
            z, e = lg[-1, DIG[0]].exp(), lg[-1, IM_END].exp()
            q10 = float(z / (z + e + 1e-30))
            ev = sum(d * float(pd[d]) for d in range(10) if d != 1) + float(pd[1]) * (1 - q10) + 10 * float(pd[1]) * q10
            res[key] = dict(ev=ev, p_first=pd.tolist(), q10=q10)
        if L: cache.crop(L)
    return res

def split_common(seqs):
    n = min(len(s) for s in seqs)
    i = 0
    while i < n and all(s[i] == seqs[0][i] for s in seqs): i += 1
    i = max(0, min(i, n - 1))
    return i

def read_set(seqs, kinds):
    """seqs: dict key -> full ids; read them with a shared prefix."""
    keys = list(seqs)
    p = split_common([seqs[k] for k in keys])
    pre = seqs[keys[0]][:p]
    return reads(pre, [(k, seqs[k][p:], kinds[k]) for k in keys])

def read_wording(c, w, who, timing, base_prefix=None, rtext=None, think_ids=None):
    """Build and read full sequences for read-type wordings. self: base_prefix = conversation ids before the report user turn
    (or None for before-timing). reader: rtext = reader transcript text (incl. any feedback block)."""
    seqs, kinds = {}, {}
    ks = range(4) if w == "rate" else [None]
    for k in ks:
        key = "x" if k is None else str(k)
        if who == "self":
            q = question(w, "self", timing, c, k)
            if timing == "before":
                ids = prompt_ids(c, False, extra="\n\n" + q)[0] + T(PREFILL[w])
            else:
                ids = self_ids(base_prefix, q, w, think_ids is not None, think_ids)
        else:
            q = question(w, "reader", timing, c, k)
            ids = reader_ids(rtext, q, w, think_ids is not None, think_ids)
        seqs[key] = ids; kinds[key] = "rate" if w == "rate" else "label"
    r = read_set(seqs, kinds)
    if w == "rate": return [r[str(k)]["ev"] for k in range(4)], r
    return r["x"], None

def parse_free_read(text, c):
    q = ("Someone described how their attention was distributed across four paragraphs labelled P1 to P4:\n\n\"" + text.strip() +
         f"\"\n\nAccording to this description, which paragraph received the most attention? Answer with just one of: {optstr(c['opts']['free'])}.")
    ids = T(U(q) + ASST + NOTHINK + "P")
    return reads([], [("x", ids, "label")])["x"]

if args.phase == "hf1":
    R = {}
    t0 = time.time()
    for n, c in enumerate(ctxs):
        g = G[str(c["id"])]; r = {}
        ids, spans = prompt_ids(c, False)
        r["ntok"] = [b - a for a, b in spans]
        r["truth"] = truth(ids, g["ans"], spans, {"ans": list(range(len(g["ans"])))})["ans"]
        idt, _ = prompt_ids(c, True)
        tr = g["think_raw"]; s = tok.decode(tr)
        # index of first final-answer token: after "</think>"
        endpos = tr.index(THINK_END) if THINK_END in tr else len(tr) - 1
        fin = list(range(endpos + 1, len(tr))) or [len(tr) - 1]
        r["truth_think"] = truth(idt, tr, spans, {"all": list(range(len(tr))), "final": fin})
        pre_after = after_prefix(c, g["ans"], False)[0]
        pre_later = pre_after + T(U(c["trivia"]) + ASST + NOTHINK) + g["later_ans"] + T("<|im_end|>\n")
        rep = {}
        for timing, ws in [("after", ["most", "least", "rate"]), ("before", ["most", "rate"]), ("later", ["most", "rate"])]:
            later = (c["trivia"], g["later_text"]) if timing == "later" else None
            rt = reader_prefix_text(c, g["ans_text"], timing, later=later)
            for w in ws:
                bp = None if timing == "before" else (pre_after if timing == "after" else pre_later)
                rep[f"{timing}|{w}|self"], rep[f"{timing}|{w}|self_raw"] = read_wording(c, w, "self", timing, base_prefix=bp)
                rep[f"{timing}|{w}|reader"], rep[f"{timing}|{w}|reader_raw"] = read_wording(c, w, "reader", timing, rtext=rt)
        # thinking 'most'
        pre_t = idt + tr + T("<|im_end|>\n")
        rtt = reader_prefix_text(c, g["think_ans_text"], "after", think_text=g["think_text"])
        rep["think|most|self"], _ = read_wording(c, "most", "self", "after", base_prefix=pre_t, think_ids=g["rthink"]["think|most|self"])
        rep["think|most|reader"], _ = read_wording(c, "most", "reader", "after", rtext=rtt, think_ids=g["rthink"]["think|most|reader"])
        # free-form parse
        for who in ["self", "reader"]:
            rep[f"after|free|{who}"] = parse_free_read(g["gen"][f"after|free|{who}"], c)
        r["rep"] = rep
        R[str(c["id"])] = r
        if (n + 1) % 10 == 0:
            log(f"hf1 {n+1}/{len(ctxs)} {time.time()-t0:.0f}s"); jdump("hf1.json", R)
    jdump("hf1.json", R)
    log("hf1 done")
    sys.exit(0)

# feedback examples: same family, same shard, other contexts, fixed per context
def fb_examples(c, k, H):
    rng = random.Random(1000 + c["id"])
    pool = [d for d in ctxs if d["family"] == c["family"] and d["id"] != c["id"]]
    ex = rng.sample(pool, min(k, len(pool)))
    out = []
    for d in ex:
        tv = np.array(H[str(d["id"])]["truth"]["norm"])[BAND].mean(0)
        pc, mo = pct_string(tv)
        out.append(dict(user=user_content(d)[0], ans=G[str(d["id"])]["ans_text"], pcts=pc, most=mo))
    return out

def fb_self_prefix(c, ex):
    g = G[str(c["id"])]
    pre = after_prefix(c, g["ans"], False)[0]
    return pre, fb_block(ex, "self") + "Now, about the answer you just wrote above, in this conversation. "

if args.phase == "gen2":
    H = jload("hf1.json")
    jobs = []
    for c in ctxs:
        g = G[str(c["id"])]
        for k in [4, 16]:
            ex = fb_examples(c, k, H)
            pre, blk = fb_self_prefix(c, ex)
            sid = self_ids(pre, blk + question("pct", "self", "after", c), "pct")
            rt = fb_block(ex, "reader") + "Now the new conversation.\n\n" + reader_prefix_text(c, g["ans_text"], "after")
            jobs += [(c["id"], f"fb{k}|pct|self", sid), (c["id"], f"fb{k}|pct|reader", reader_ids(rt, question("pct", "reader", "after", c), "pct"))]
    log("gen2 jobs", len(jobs), "max len", max(len(j[2]) for j in jobs))
    outs = gen([j[2] for j in jobs], 60)
    O = {}
    for j, o in zip(jobs, outs): O.setdefault(str(j[0]), {})[j[1]] = tok.decode(o, skip_special_tokens=True)
    jdump("gen2.json", O)
    log("gen2 done")
    sys.exit(0)

if args.phase == "hf2":
    H = jload("hf1.json")
    R = {}
    t0 = time.time()
    for n, c in enumerate(ctxs):
        g = G[str(c["id"])]; rep = {}
        for k in [4, 16]:
            ex = fb_examples(c, k, H)
            pre, blk = fb_self_prefix(c, ex)
            rt = fb_block(ex, "reader") + "Now the new conversation.\n\n" + reader_prefix_text(c, g["ans_text"], "after")
            for w in ["most", "rate"]:
                # self: the feedback block is part of the report user turn, before the question
                seqs, kinds = {}, {}
                for kk in (range(4) if w == "rate" else [None]):
                    key = "x" if kk is None else str(kk)
                    seqs[key] = self_ids(pre, blk + question(w, "self", "after", c, kk), w); kinds[key] = "rate" if w == "rate" else "label"
                r = read_set(seqs, kinds)
                rep[f"fb{k}|{w}|self"] = [r[str(i)]["ev"] for i in range(4)] if w == "rate" else r["x"]
                seqs = {}
                for kk in (range(4) if w == "rate" else [None]):
                    key = "x" if kk is None else str(kk)
                    seqs[key] = reader_ids(rt, question(w, "reader", "after", c, kk), w)
                r = read_set(seqs, kinds)
                rep[f"fb{k}|{w}|reader"] = [r[str(i)]["ev"] for i in range(4)] if w == "rate" else r["x"]
        R[str(c["id"])] = rep
        if (n + 1) % 10 == 0:
            log(f"hf2 {n+1}/{len(ctxs)} {time.time()-t0:.0f}s"); jdump("hf2.json", R)
    jdump("hf2.json", R)
    log("hf2 done")
