# Specificity of focus / completely-ignore: per context and named paragraph, % change of the named paragraph minus the mean
# % change of the other three (percentage points), against the mention line and against no instruction. Reads R2.json.
import json, numpy as np
R = json.load(open("R2.json")); L = list(range(21, 54)); n = len(R)
band = lambda a: np.array(a)[L].mean(0)
A = {c: np.array([[band(r["cond"][f"{c}_{k}"]["att_mean"]) for k in range(4)] for r in R]) for c in ["mention", "focus", "ignore"]}
A0 = np.repeat(np.array([band(r["none"]["att_mean"]) for r in R])[:, None, :], 4, 1)
BI = np.random.default_rng(0).integers(0, n, (10000, n)); out = {}
for c in ["focus", "ignore"]:
    for ctrl, B in [("mention", A["mention"]), ("none", A0)]:
        Z = 100 * (A[c] / B - 1); d = np.array([[Z[i, k, k] - np.delete(Z[i, k], k).mean() for k in range(4)] for i in range(n)])
        m = d.mean(1); bs = m[BI].mean(1)
        out[f"{c}_vs_{ctrl}"] = dict(mean=float(m.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                                     by_named=d.mean(0).tolist())
        print(f"{c} vs {ctrl}: named minus other {m.mean():+.1f} pp [{np.percentile(bs,2.5):+.1f}, {np.percentile(bs,97.5):+.1f}]; by named P1..P4 "
              + ", ".join(f"{v:+.1f}" for v in d.mean(0)))
json.dump(out, open("named_minus_other.json", "w"), indent=1)
# Relative version (the README's convention: percent = relative change): per context and named paragraph, the named paragraph's
# attention ratio (instruction / control) divided by the mean ratio of the other three, minus 1; mean over contexts of its log, exp - 1.
for c in ["focus", "ignore"]:
    for ctrl, B in [("mention", A["mention"]), ("none", A0)]:
        Q = A[c] / B; lr = np.array([[np.log(Q[i, k, k] / np.delete(Q[i, k], k).mean()) for k in range(4)] for i in range(n)]).mean(1)
        bs = lr[BI].mean(1); f = lambda v: 100 * (np.exp(v) - 1)
        out[f"{c}_vs_{ctrl}"]["relative"] = dict(mean=f(lr.mean()), lo=f(np.percentile(bs, 2.5)), hi=f(np.percentile(bs, 97.5)))
        print(f"{c} vs {ctrl}: named relative to the others {f(lr.mean()):+.1f}% [{f(np.percentile(bs, 2.5)):+.1f}, {f(np.percentile(bs, 97.5)):+.1f}]")
json.dump(out, open("named_minus_other.json", "w"), indent=1)
