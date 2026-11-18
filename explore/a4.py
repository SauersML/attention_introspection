# E4 analysis: which source wins under each instruction. margin = log p(orig) - log p(alt), teacher-forced.
import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from stats import *
R = torch.load(sys.argv[1]); out = {}
conds = list(R[0]["lpa"].keys())
def m(r, c): return r["lpa"][c] - r["lpb"][c]
def pick(r, c):
    g = r["gen"][c]; a, b = r["a"] in g, r["a2"] in g
    return 1.0 if a and not b else (0.0 if b and not a else 0.5)
def att(r, c, which):
    i = r["io"] if which == "orig" else r["ia"]
    return pertok(r["att"][c], r["ntok"][c])[i]
for c in conds:
    out[c] = dict(margin=boot([m(r, c) for r in R]), pick_orig=boot([pick(r, c) for r in R]),
                  att_orig=pct([np.log(att(r, c, "orig") / att(r, "none", "orig")) for r in R]),
                  att_alt=pct([np.log(att(r, c, "alt") / att(r, "none", "alt")) for r in R]))
    o = out[c]; print(f"{c:12s} margin {o['margin'][0]:+.2f} [{o['margin'][1]:+.2f},{o['margin'][2]:+.2f}]  picks orig {o['pick_orig'][0]:.2f} [{o['pick_orig'][1]:.2f},{o['pick_orig'][2]:.2f}]  att orig {o['att_orig']['pct']:+.1f}%  att alt {o['att_alt']['pct']:+.1f}%")
# the source effect of each instruction: margin(named orig) - margin(named alt), vs the same for mention
for w in ("focus", "use", "ignore", "mention"):
    d = [m(r, f"{w}/orig") - m(r, f"{w}/alt") for r in R]; p = [pick(r, f"{w}/orig") - pick(r, f"{w}/alt") for r in R]
    out[f"swing/{w}"] = dict(margin=boot(d) + (signp(d),), pick=boot(p) + (signp(p),))
    print(f"swing {w:8s} margin {out[f'swing/{w}']['margin'][0]:+.2f} [{out[f'swing/{w}']['margin'][1]:+.2f},{out[f'swing/{w}']['margin'][2]:+.2f}] p={out[f'swing/{w}']['margin'][3]:.4f}  pick {out[f'swing/{w}']['pick'][0]:+.2f} [{out[f'swing/{w}']['pick'][1]:+.2f},{out[f'swing/{w}']['pick'][2]:+.2f}]")
for w in ("focus", "use", "ignore"):
    d = [(m(r, f"{w}/orig") - m(r, f"{w}/alt")) - (m(r, "mention/orig") - m(r, "mention/alt")) for r in R]
    out[f"swing_vs_mention/{w}"] = boot(d) + (signp(d),); print(f"{w} swing minus mention swing", [round(x, 3) for x in out[f"swing_vs_mention/{w}"]])
# order effect: does the later of the two conflicting paragraphs win by default?
d = [m(r, "none") for r in R if r["io"] > r["ia"]], [m(r, "none") for r in R if r["io"] < r["ia"]]
print("none margin when orig later", boot(d[0])[0], "earlier", boot(d[1])[0])
json.dump(out, open(sys.argv[2], "w"), indent=1)
