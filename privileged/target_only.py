# Privileged test: the reported effect (analyze_priv.py) is the specificity, exp of the mean over contexts and boosted parts of
# (target label's change in log P) minus (mean change in log P of the non-target labels). Beside it, the target label's own
# change in log P, and the per-kind range in arm A. Wording W1. Multipliers are pre-softmax biases of log(m).
# usage: python privileged/target_only.py  (reads results/privileged/data/recs_*.jsonl, writes results/privileged/target_only.json)
import json, glob, math, numpy as np
R = sorted([json.loads(l) for f in glob.glob("results/privileged/data/recs_*.jsonl") for l in open(f)], key=lambda r: r["id"])
def per(r, arm, m, w="W1"):
    N = r["N"]; n = np.array(r["none"][w]); t, s = [], []
    for k in range(N):
        d = np.array(r[arm][f"{k}|{m}"][w]) - n; t.append(d[k]); s.append(d[k] - np.mean([d[j] for j in range(N) if j != k]))
    return np.mean(t), np.mean(s)
def boot(x, seed=0):
    x = np.asarray(x); m = x[np.random.default_rng(seed).integers(0, len(x), (10000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
pct = lambda v: 100 * (math.exp(v) - 1)
out = {}
for arm in "AC":
    for m in [0.25, 0.5, 2.0, 5.0, 10.0]:
        v = np.array([per(r, arm, m) for r in R])
        t, s = boot(v[:, 0]), boot(v[:, 1])
        out[f"{arm}|{m}"] = dict(target=t, spec=s)
        print(f"arm {arm} x{m}: specificity {pct(s[0]):+.1f}% [{pct(s[1]):+.1f}, {pct(s[2]):+.1f}]  target label alone {pct(t[0]):+.1f}% [{pct(t[1]):+.1f}, {pct(t[2]):+.1f}]")
kinds = sorted({r["setup"] for r in R})
for m in [5.0, 10.0]:
    ks = {s: boot([per(r, "A", m)[1] for r in R if r["setup"] == s]) for s in kinds}
    out[f"A_by_kind|{m}"] = ks
    print(f"arm A x{m} by kind: " + ", ".join(f"{s} {pct(v[0]):+.1f}%" for s, v in ks.items()))
json.dump(out, open("results/privileged/target_only.json", "w"), indent=1)
