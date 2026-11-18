# Rebuild the 240 told_inject contexts (paragraph texts) from SQuAD with the run's seed and check their token spans exactly against the records. Run from the repo root (tokenizer only, no model): nice -n 19 python3 figures/told_contexts.py
import json, random, glob, os
os.environ["HF_HUB_OFFLINE"]="1"
from transformers import AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
sq = load_dataset("rajpurkar/squad", split="train")
by_title, seen = {}, set()
for r in sq:
    c = r["context"]
    if c in seen: continue
    seen.add(c); by_title.setdefault(r["title"], []).append(c)
ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
pool = {t: [p for p in v if 60 <= ntok(p) <= 110] for t, v in sorted(by_title.items())}
pool = {t: v for t, v in pool.items() if v}
rng = random.Random(23); ctxs = []
for i in range(240):
    ts = rng.sample(sorted(pool), 4); ctxs.append(dict(id=i, segs=[rng.choice(pool[t]) for t in ts]))
R = {json.loads(l)["id"]: json.loads(l) for f in glob.glob("results/told_inject/data/recs_*.jsonl") for l in open(f)}
bad=sum(1 for c in ctxs if [ntok(s) for s in c["segs"]] != R[c["id"]]["ntok"]); print("ntok mismatches", bad)
#(s) for s in c["segs"]] == R[c["id"]]["ntok"] for c in ctxs)
for s in range(3): assert all(p in open(f"results/told_inject/data/example_prompt_{s}.txt").read() for p in ctxs[s]["segs"])
json.dump(ctxs, open("figures/told_contexts.json", "w"))
print("ok")
import numpy as np
dd=np.array([[ntok(s)-n for s,n in zip(c["segs"],R[c["id"]]["ntok"])] for c in ctxs]); print(np.unique(dd,return_counts=True))
INTRO = json.load(open("results/told_inject/data/meta.json"))["intro"]
src=open("told_inject/told_inject.py").read(); ns={}
exec(src[src.index("INTRO = ("):src.index("def build")], ns); exec(src[src.index("def build"):src.index("    Q0 = min")].replace("    assert tok.decode","    pass #")+"    return segtok\n", dict(ns, tok=tok), ns)
bad=0
for c in ctxs:
    st=ns["build"](c); bad += [len(s) for s in st] != R[c["id"]]["ntok"]
print("exact span mismatches", bad)
