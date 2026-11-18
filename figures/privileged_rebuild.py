# Rebuilds the prompt text of the paragraph contexts of privileged.py (same seed, same SQuAD sampling; tokenizer only, no model).
# Writes ctx.json; checked against recs_*.jsonl: lab, opt and per-paragraph token counts match for all 100 paragraph contexts.
import json, re, random
from transformers import AutoTokenizer
class A: seed=11
args=A()
def log(*a): print(*a)
SETUPS=[["par_arith", 60], ["par_qa", 40], ["dialogue", 40], ["list", 40], ["code", 40], ["document", 40], ["sentences", 40]]
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
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


out=[]
for c in ctxs:
    if c["setup"] in ("par_arith","par_qa"):
        ids,segtok,user=build(c)
        out.append(dict(id=c["id"],setup=c["setup"],lab=c["lab"],opt=c["opt"],user=user,ntok=[len(s) for s in segtok],segs=c["segs"],q=c.get("q"),gold=c.get("gold"),j=c["j"]))
json.dump(out,open("ctx.json","w"))
print(len(out))
