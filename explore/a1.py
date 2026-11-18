import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from stats import *
R = torch.load(sys.argv[1]); out = {}
names = sorted({k.split("/")[0] for r in R for k in r["att"] if "/" in k})
for nm in names:
    tg, sp, oth = [], [], []
    for r in R:
        k = r["k"]
        f = pertok(r["att"][f"{nm}/focus"], r["ntok"][f"{nm}/focus"])[:4]
        m = pertok(r["att"][f"{nm}/mention"], r["ntok"][f"{nm}/mention"])[:4]
        lr = np.log(f / m)
        tg.append(lr[k]); o = np.mean([lr[j] for j in range(4) if j != k]); oth.append(o); sp.append(lr[k] - o)
    out[nm] = dict(target=pct(tg), others=pct(oth), specific=pct(sp),
                   top=float(np.mean([np.argmax(np.log(pertok(r["att"][f"{nm}/focus"], r["ntok"][f"{nm}/focus"])[:4] / pertok(r["att"][f"{nm}/mention"], r["ntok"][f"{nm}/mention"])[:4])) == r["k"] for r in R])))
# mention vs none (does the mention itself pull?)
for nm, base in (("label", "none_lab"), ("topic_nolab", "none_nolab"), ("ordinal_nolab", "none_nolab"), ("quote_nolab", "none_nolab")):
    lr = [np.log(pertok(r["att"][f"{nm}/mention"], r["ntok"][f"{nm}/mention"])[r["k"]] / pertok(r["att"][base], r["ntok"][base])[r["k"]]) for r in R]
    out[f"{nm}_mention_vs_none"] = pct(lr)
lpd = {nm: float(np.mean([r["lp"][f"{nm}/focus"] - r["lp"][f"{nm}/mention"] for r in R])) for nm in names}
out["answer_logprob_focus_minus_mention"] = lpd
for k, v in out.items():
    if isinstance(v, dict) and "target" in v:
        print(f"{k:15s} target {v['target']['pct']:+6.1f}% [{v['target']['lo']:+.1f},{v['target']['hi']:+.1f}]  others {v['others']['pct']:+6.1f}%  specific {v['specific']['pct']:+6.1f}% [{v['specific']['lo']:+.1f},{v['specific']['hi']:+.1f}] p={v['specific']['p']:.4f} top={v['top']:.2f}")
    elif "pct" in v: print(f"{k:28s} {v['pct']:+6.1f}% [{v['lo']:+.1f},{v['hi']:+.1f}] p={v['p']:.4f}")
print(lpd)
json.dump(out, open(sys.argv[2], "w"), indent=1)
