import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from stats import *
R = torch.load(sys.argv[1]); out = {"n": len(R)}
for w in ("focus", "ignore"):
    keep = [r for r in R if abs(r["res"][w]["m_s"] - r["res"][w]["m_d"]) > 2]  # need a real difference to normalise
    out[f"{w}/n"] = len(keep); print(w, "n with |m_s-m_d|>2:", len(keep), "of", len(R), " mean m_s", np.mean([r["res"][w]["m_s"] for r in R]), "m_d", np.mean([r["res"][w]["m_d"] for r in R]))
    for k in keep[0]["res"][w]["patch"]:
        f = [(r["res"][w]["patch"][k] - r["res"][w]["m_d"]) / (r["res"][w]["m_s"] - r["res"][w]["m_d"]) for r in keep]
        out[f"{w}/{k}"] = boot(f)
    for nm in ("digit", "after", "last"):
        print(" ", nm, " ".join(f"L{k.split('/')[1]}:{out[f'{w}/{k}'][0]:.2f}" for k in keep[0]["res"][w]["patch"] if k.startswith(nm)))
json.dump(out, open(sys.argv[2], "w"), indent=1)
