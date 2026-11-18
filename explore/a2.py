# E2 analysis: accuracy, gold log-prob and attention to the answer paragraph (k) and the named wrong one (k2) per condition, vs none.
import sys, os, json, re, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from stats import *
R = torch.load(sys.argv[1]); out = {}
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
conds = list(R[0]["lp"].keys())
for c in conds:
    acc = [float(norm(r["gold"]) in norm(r["gen"][c])) for r in R]
    dlp = [r["lp"][c] - r["lp"]["none"] for r in R]
    ak = [np.log(pertok(r["att"][c], r["ntok"][c])[r["k"]] / pertok(r["att"]["none"], r["ntok"]["none"])[r["k"]]) for r in R]
    ak2 = [np.log(pertok(r["att"][c], r["ntok"][c])[r["k2"]] / pertok(r["att"]["none"], r["ntok"]["none"])[r["k2"]]) for r in R]
    out[c] = dict(acc=boot(acc), dlp=boot(dlp) + (signp(dlp),), att_k=pct(ak), att_k2=pct(ak2))
    print(f"{c:14s} acc {out[c]['acc'][0]:.3f} [{out[c]['acc'][1]:.3f},{out[c]['acc'][2]:.3f}]  dlogp {out[c]['dlp'][0]:+.3f} [{out[c]['dlp'][1]:+.3f},{out[c]['dlp'][2]:+.3f}] p={out[c]['dlp'][3]:.4f}  att_k {out[c]['att_k']['pct']:+.1f}% [{out[c]['att_k']['lo']:+.1f},{out[c]['att_k']['hi']:+.1f}]  att_k2 {out[c]['att_k2']['pct']:+.1f}% [{out[c]['att_k2']['lo']:+.1f},{out[c]['att_k2']['hi']:+.1f}]")
# paired focus vs mention, ignore vs mention (gold logprob)
for w in ("focus", "ignore"):
    for t in ("right", "wrong"):
        d = [r["lp"][f"{w}/{t}"] - r["lp"][f"mention/{t}"] for r in R]
        out[f"{w}-mention/{t}"] = boot(d) + (signp(d),); print(f"{w}-mention/{t} dlogp", [round(x, 4) for x in out[f"{w}-mention/{t}"]])
# per-prompt link: does the attention change to k predict the log-prob change? (across all instruction conditions)
xs, ys = [], []
for r in R:
    for c in conds[1:]:
        xs.append(np.log(pertok(r["att"][c], r["ntok"][c])[r["k"]] / pertok(r["att"]["none"], r["ntok"]["none"])[r["k"]])); ys.append(r["lp"][c] - r["lp"]["none"])
out["corr_attk_dlp"] = float(np.corrcoef(xs, ys)[0, 1]); print("corr att_k vs dlogp", out["corr_attk_dlp"])
json.dump(out, open(sys.argv[2], "w"), indent=1)
