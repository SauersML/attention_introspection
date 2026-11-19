# In-order labels vs shuffled labels, privileged-channel experiment: every number privileged.png shows, side by side, plus the
# paired difference (same prompt, same boosted part; in-order minus shuffled). Metric as analyze_priv.py / privileged_fig.py:
# rotation specificity, W1, shown as exp(.) - 1. 95% bootstrap CI over prompts (10,000 resamples); p two-sided sign-flip (200,000).
# usage: python privileged/compare_orders.py   (writes results/privileged_inorder_full/compare_shuffled.{json,md})
import json, glob, math, numpy as np
def load(d): return {r["id"]: r for f in glob.glob(f"results/{d}/data/recs_*.jsonl") for l in open(f) if l.strip() for r in [json.loads(l)]}
IN, SH = load("privileged_inorder_full"), load("privileged_shuffled"); assert sorted(IN) == sorted(SH) and len(IN) == 300
def spec(r, arm, m, w="W1"):
    N = r["N"]; n = np.array(r["none"][w]); s = []
    for k in range(N):
        d = np.array(r[arm][f"{k}|{m}"][w]) - n; s.append(d[k] - np.mean([d[j] for j in range(N) if j != k]))
    return float(np.mean(s))
def boot(x, seed=0):
    x = np.asarray(x); m = x[np.random.default_rng(seed).integers(0, len(x), (10000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def signflip(x, NF=200000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed); obs = abs(x.mean()); h = 0
    for _ in range(NF // 20000): h += int((np.abs((rng.choice([-1.0, 1.0], (20000, len(x))) * x).mean(1)) >= obs).sum())
    return (h + 1) / (NF + 1)
pct = lambda v: 100 * (math.exp(v) - 1)
fmt = lambda b: f"{pct(b[0]):+.1f}% [{pct(b[1]):+.1f}, {pct(b[2]):+.1f}]"
fp = lambda p: "<0.0001" if p < 1e-4 else f"{p:.2g}"
ROWS = []
for arm in "AC":
    for m in [0.25, 0.5, 2.0, 5.0, 10.0]: ROWS.append((arm, m, ("par_arith", "par_qa"), "paragraphs (100)"))
    for s in ["par_arith", "par_qa", "list", "code", "document", "sentences", "dialogue"]: ROWS.append((arm, 10.0, (s,), s))
    for m in [5.0, 10.0]: ROWS.append((arm, m, None, "all kinds (300)"))
out, L = [], ["| arm | ×m | prompts | in order | shuffled | in order − shuffled (paired) | p (paired) |", "|---|---|---|---|---|---|---|"]
for arm, m, setups, name in ROWS:
    ids = [i for i in sorted(IN) if setups is None or IN[i]["setup"] in setups]
    a = np.array([spec(IN[i], arm, m) for i in ids]); b = np.array([spec(SH[i], arm, m) for i in ids])
    bi, bs, bd = boot(a), boot(b), boot(a - b); pi, ps, pd = signflip(a), signflip(b), signflip(a - b)
    out.append(dict(arm=arm, m=m, prompts=name, n=len(ids), inorder=bi, inorder_p=pi, shuffled=bs, shuffled_p=ps, diff=bd, diff_p=pd))
    L.append(f"| {arm} | ×{m:g} | {name} | {fmt(bi)}, p {fp(pi)} | {fmt(bs)}, p {fp(ps)} | {fmt(bd)} | {fp(pd)} |")
json.dump(out, open("results/privileged_inorder_full/compare_shuffled.json", "w"), indent=1)
open("results/privileged_inorder_full/compare_shuffled.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
