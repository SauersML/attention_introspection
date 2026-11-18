# Shared harness for the explore/ experiments (Qwen3-32B, eager attention, per-head attention to named char spans).
# Prompts are built as a user string plus named character spans; spans are mapped to tokens with offsets.
# Attention recorded: for query rows R (answer tokens), per layer, per head, mean over rows of the SUM of attention
# over each span's tokens (divide by span length for per-token attention in the analysis).
import random, time, json, base64
import torch
import transformers.models.qwen3.modeling_qwen3 as mq
from transformers import AutoTokenizer, AutoModelForCausalLM

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

S = dict(rec=False, rows=None, spans=None, store={}, bias=None, zero=None)

def hooked(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    L = module.layer_idx
    k = mq.repeat_kv(key, module.num_key_value_groups); v = mq.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, k.transpose(2, 3)).float() * scaling
    ql, kl = w.shape[-2], w.shape[-1]; dev = w.device
    if attention_mask is not None:
        w = w + attention_mask[:, :, :, :kl].float()
    else:
        qpos = torch.arange(kl - ql, kl, device=dev)
        w = w.masked_fill(torch.arange(kl, device=dev)[None, :] > qpos[:, None], float("-inf"))
    if S["bias"] is not None:  # list of (rows, lo, hi, value): add value to logits rows x [lo,hi)
        for rows, lo, hi, val in S["bias"]:
            w[:, :, rows, lo:hi] += val
    p = torch.softmax(w, dim=-1)
    if S["rec"] and ql > 1:
        pr = p[:, :, S["rows"], :]
        S["store"][L] = torch.stack([pr[..., lo:hi].sum(-1).mean(-1) for (lo, hi) in S["spans"]], -1)[0].cpu()  # [H, nspans]
    o = torch.matmul(p.to(value.dtype), v)
    if S["zero"] and L in S["zero"]:  # {layer: [heads]}: zero these heads' outputs at every position
        o[:, S["zero"][L]] = 0
    return o.transpose(1, 2).contiguous(), None
mq.eager_attention_forward = hooked

def load(name="Qwen/Qwen3-32B"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda").eval()
    return tok, model

def squad_pool(tok, lo=60, hi=110):
    from datasets import load_dataset
    sq = load_dataset("rajpurkar/squad", split="train")
    by_title, seen = {}, set()
    for r in sq:
        c = r["context"]
        if c in seen: continue
        seen.add(c)
        nt = len(tok(c)["input_ids"])
        if lo <= nt <= hi:
            by_title.setdefault(r["title"], []).append(dict(text=c, title=r["title"].replace("_", " "), q=r["question"], ans=r["answers"]["text"][0]))
    return by_title

def contexts(tok, n=100, seed=11, npara=4):
    by_title = squad_pool(tok)
    titles = sorted(by_title); rng = random.Random(seed); out = []
    for i in range(n):
        ts = rng.sample(titles, npara)
        out.append(dict(id=i, paras=[rng.choice(by_title[t]) for t in ts], k=rng.randrange(npara), k2=None,
                        x=rng.randrange(10, 90), y=rng.randrange(10, 90)))
        out[-1]["k2"] = rng.choice([j for j in range(npara) if j != out[-1]["k"]])
    return out

def encode(tok, turns, spans_by_turn, gen_prompt=True):
    """turns: list of (role, content); spans_by_turn: {turn_index: {name: (cs, ce)}} char spans inside that turn's content.
    Returns ids and {name: (tok_lo, tok_hi)}."""
    prompt = tok.apply_chat_template([{"role": r, "content": c} for r, c in turns], tokenize=False,
                                     add_generation_prompt=gen_prompt, enable_thinking=False)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True); ids, om = enc["input_ids"], enc["offset_mapping"]
    spans, cur = {}, 0
    for ti, (r, c) in enumerate(turns):
        off = prompt.index(c, cur); cur = off + len(c)
        for name, (cs, ce) in spans_by_turn.get(ti, {}).items():
            idx = [i for i, (a, b) in enumerate(om) if b > cs + off and a < ce + off and b > a]
            spans[name] = (idx[0], idx[-1] + 1)
    return ids, spans

class Doc:
    """Builds a user message piece by piece, remembering char spans of named pieces."""
    def __init__(self): self.s = ""; self.sp = {}
    def add(self, text, name=None):
        if name: self.sp[name] = (len(self.s), len(self.s) + len(text))
        self.s += text; return self
    def mark(self, name, sub, within=None):
        lo = self.sp[within][0] if within else 0
        i = self.s.index(sub, lo); self.sp[name] = (i, i + len(sub)); return self

def para_doc(c, labels=True, instr="", instr_pos="before_q", question=None):
    d = Doc(); d.add("Here are four paragraphs.\n\n")
    if instr and instr_pos == "top": d.add(instr, "instr").add("\n\n")
    for i, p in enumerate(c["paras"]):
        if labels: d.add(f"[P{i+1}] ")
        d.add(p["text"], f"p{i}").add("\n\n")
    q = question or f"Unrelated question: what is {c['x']} + {c['y']}? Answer with just the number."
    if instr and instr_pos == "before_q": d.add(instr, "instr").add("\n")
    d.add(q, "q")
    return d

@torch.no_grad()
def generate(tok, model, ids, maxnew=20):
    S.update(rec=False, bias=None, zero=None)
    g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=maxnew, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    a = g[0, len(ids):].tolist()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    while a and a[-1] in (end, tok.eos_token_id): a = a[:-1]
    return a

@torch.no_grad()
def measure(model, ids, ans, spans, bias=None, zero=None):
    """Teacher-force ans after ids; return [NL, H, nspans] attention (sum over span tokens, mean over answer rows)."""
    L0 = len(ids)
    S.update(rec=True, rows=list(range(L0, L0 + len(ans))), spans=spans, store={}, bias=bias, zero=zero)
    out = model(torch.tensor([ids + ans], device="cuda"), use_cache=False, logits_to_keep=len(ans) + 1)
    S.update(rec=False, bias=None, zero=None)
    NL = len(S["store"])
    lp = torch.log_softmax(out.logits[0, :-1].float(), -1)
    ans_lp = lp[torch.arange(len(ans)), torch.tensor(ans, device=lp.device)].sum().item()
    return torch.stack([S["store"][L] for L in range(NL)]).half(), ans_lp

ORD = ["first", "second", "third", "fourth"]
def b64(s): return base64.b64encode(s.encode()).decode()
