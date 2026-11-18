# E6 analysis: margins under attention blocks; last-position attention to the two conflicting numbers.
import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from stats import *
R = torch.load(sys.argv[1]); out = {"n": len(R)}
for k in R[0]["m"]:
    out[k] = boot([r["m"][k] for r in R]); print(f"{k:26s} margin {out[k][0]:+.2f} [{out[k][1]:+.2f},{out[k][2]:+.2f}]")
def num(r, c, which, layers=BAND):
    a = r["att"][c].float()[layers].mean(0).mean(0); i = 4 if which == "o" else 5
    return (a[i] / r["ntok"][c][i]).item()
for c in R[0]["att"]:
    o = [np.log(num(r, c, "o") / num(r, "none", "o")) for r in R]; a = [np.log(num(r, c, "a") / num(r, "none", "a")) for r in R]
    out[f"numatt/{c}"] = dict(orig=pct(o), alt=pct(a))
    print(f"{c:14s} last-pos attention to number: orig {out[f'numatt/{c}']['orig']['pct']:+.1f}% [{out[f'numatt/{c}']['orig']['lo']:+.1f},{out[f'numatt/{c}']['orig']['hi']:+.1f}]  alt {out[f'numatt/{c}']['alt']['pct']:+.1f}% [{out[f'numatt/{c}']['alt']['lo']:+.1f},{out[f'numatt/{c}']['alt']['hi']:+.1f}]")
# per layer: log(att num_alt / num_orig) under focus/alt minus focus/orig
for c1, c2 in (("focus/alt", "focus/orig"), ("ignore/orig", "ignore/alt"), ("use/alt", "use/orig")):
    prof = []
    for L in range(64):
        prof.append(np.mean([np.log(num(r, c1, "a", [L]) / num(r, c1, "o", [L])) - np.log(num(r, c2, "a", [L]) / num(r, c2, "o", [L])) for r in R]))
    out[f"layer/{c1}-{c2}"] = prof; print(c1, "-", c2, "by layer", np.round(np.array(prof), 1).tolist())
json.dump(out, open(sys.argv[2], "w"), indent=1)
