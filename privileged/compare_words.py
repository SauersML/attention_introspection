# In-order labels vs shuffled labels, single-word privileged-channel test (privileged_words.py): rotation specificity per arm and dose,
# both runs side by side, plus the paired difference (same context, same boosted word; in-order minus shuffled). Metric as analyze_words.py:
# W1, shown as exp(.) - 1. 95% bootstrap CI over contexts (10,000 resamples); p two-sided sign-flip (200,000 flips).
# usage: python privileged/compare_words.py   (reads results/privileged_words{,_shuffled}/data/recs_*.jsonl[.gz];
#        writes results/privileged_words/compare_shuffled.{json,md})
import json, glob, gzip, math, numpy as np
def load(d):
    R = {}
    for f in glob.glob(f"results/{d}/data/recs_*.jsonl*"):
        for l in (gzip.open(f, "rt") if f.endswith(".gz") else open(f)):
            if l.strip(): r = json.loads(l); R[r["id"]] = r
    return R
IN, SH = load("privileged_words"), load("privileged_words_shuffled"); assert sorted(IN) == sorted(SH) and len(IN) == 100
assert all(IN[i]["words"] == SH[i]["words"] and IN[i]["lab"] == list(range(1, 9)) for i in IN)
def spec(r, arm, m):
    N = r["N"]; n = np.array(r["none"]["W1"]); s = []
    for k in range(N):
        d = np.array(r[arm][f"{k}|{m}"]["W1"]) - n; s.append(d[k] - np.mean([d[j] for j in range(N) if j != k]))
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
ids = sorted(IN)
same = sum(IN[i]["ans"] == SH[i]["ans"] for i in ids)
out, L = [], [f"Answers identical to the shuffled run: {same}/100.", "",
              "| arm | ×m | in order | shuffled | in order − shuffled (paired) | p (paired) |", "|---|---|---|---|---|---|"]
for arm in "ABC":
    for m in [0.25, 0.5, 2.0, 5.0, 10.0]:
        a = np.array([spec(IN[i], arm, m) for i in ids]); b = np.array([spec(SH[i], arm, m) for i in ids])
        bi, bs, bd = boot(a), boot(b), boot(a - b); pi, ps, pd = signflip(a), signflip(b), signflip(a - b)
        out.append(dict(arm=arm, m=m, n=len(ids), inorder=bi, inorder_p=pi, shuffled=bs, shuffled_p=ps, diff=bd, diff_p=pd))
        L.append(f"| {arm} | ×{m:g} | {fmt(bi)}, p {fp(pi)} | {fmt(bs)}, p {fp(ps)} | {fmt(bd)} | {fp(pd)} |")
json.dump(out, open("results/privileged_words/compare_shuffled.json", "w"), indent=1)
open("results/privileged_words/compare_shuffled.md", "w").write("# Single words: labels and options in order vs shuffled, 100 contexts, W1\n\n" + "\n".join(L) + "\n"); print("\n".join(L))
