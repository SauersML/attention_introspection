# E5 analysis: log(att a / att b) relative to the mention line, per requested split, both naming orders.
import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from stats import *
R = torch.load(sys.argv[1]); out = {}
def lr(r, key):
    p = pertok(r["att"][key], r["ntok"][key]); m = pertok(r["att"]["mention"], r["ntok"]["mention"])
    return np.log(p[r["a"]] / m[r["a"]]) - np.log(p[r["b"]] / m[r["b"]]), np.log(p[r["a"]] / m[r["a"]]), np.log(p[r["b"]] / m[r["b"]])
for key in R[0]["att"]:
    if key == "mention": continue
    v = np.array([lr(r, key) for r in R])
    out[key] = dict(ratio=boot(v[:, 0]), a=pct(v[:, 1]), b=pct(v[:, 2]))
    print(f"{key:8s} log(a/b) {out[key]['ratio'][0]:+.3f} [{out[key]['ratio'][1]:+.3f},{out[key]['ratio'][2]:+.3f}]   a {out[key]['a']['pct']:+.1f}%  b {out[key]['b']['pct']:+.1f}%")
# monotonicity: per-prompt Spearman between requested share and log ratio, both orders
from scipy.stats import spearmanr
X = [100, 90, 75, 50, 25, 10, 0]
for o in ("s", "r"):
    rho = [spearmanr(X, [lr(r, f"{o}{x}")[0] for x in X])[0] for r in R]
    out[f"spearman/{o}"] = boot(rho); print(o, "per-prompt spearman", out[f"spearman/{o}"])
d = [lr(r, "s90")[0] - lr(r, "s10")[0] for r in R]; out["s90-s10"] = boot(d) + (signp(d),); print("90 vs 10", out["s90-s10"])
d = [lr(r, "s75")[0] - lr(r, "s25")[0] for r in R]; out["s75-s25"] = boot(d) + (signp(d),); print("75 vs 25", out["s75-s25"])
d = [lr(r, "s50")[0] - lr(r, "r50")[0] for r in R]; out["order_at_50"] = boot(d) + (signp(d),); print("50/50 first-named minus second-named order", out["order_at_50"])
json.dump(out, open(sys.argv[2], "w"), indent=1)
