# E3 analysis: per verb, specific effect vs mention and six-head reading of the instruction line; ablation of six vs random six.
import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from stats import *
R = torch.load(sys.argv[1]); out = {}
SIX = [(20, 19), (20, 29), (23, 61), (25, 28), (25, 30), (26, 5)]
def pt(r, key): return pertok(r["att"][key], r["ntok"][key])[:4]
def spec(r, f, m):
    lr = np.log(pt(r, f) / pt(r, m)); k = r["k"]; return lr[k] - np.mean([lr[j] for j in range(4) if j != k]), lr[k]
def six(r, key): return float(np.mean([r["att"][key].float()[L, h, 4].item() for L, h in SIX]))
verbs = [v for v in R[0]["att"] if "/" not in v and v != "mention"]
for v in verbs:
    s = [spec(r, v, "mention") for r in R]
    rd = [six(r, v) - six(r, "mention") for r in R]
    out[v] = dict(specific=pct([x[0] for x in s]), target=pct([x[1] for x in s]), six_read=boot(rd))
    print(f"{v:16s} specific {out[v]['specific']['pct']:+6.1f}% [{out[v]['specific']['lo']:+.1f},{out[v]['specific']['hi']:+.1f}]  target {out[v]['target']['pct']:+6.1f}%  six-head read diff {out[v]['six_read'][0]:+.3f} [{out[v]['six_read'][1]:+.3f},{out[v]['six_read'][2]:+.3f}]")
# per-prompt link between six-head reading and specific effect, pooled over verbs
x = [six(r, v) - six(r, "mention") for r in R for v in verbs]; y = [spec(r, v, "mention")[0] for r in R for v in verbs]
out["r_read_spec_pooled"] = float(np.corrcoef(x, y)[0, 1]); print("pooled per-prompt r(six read, specific)", out["r_read_spec_pooled"])
for nm in ("label", "ordinal", "topic"):
    base = (lambda r, kind: "focus" if kind == "focus" else "mention") if nm == "label" else (lambda r, kind: f"{nm}/{kind}")
    s0 = [spec(r, base(r, "focus"), base(r, "mention"))[0] for r in R]
    s6 = [spec(r, f"{nm}/focus/z6", f"{nm}/mention/z6")[0] for r in R]
    sr = [spec(r, f"{nm}/focus/zr", f"{nm}/mention/zr")[0] for r in R]
    out[f"abl/{nm}"] = dict(intact=pct(s0), zero6=pct(s6), zero_rand=pct(sr), six_vs_rand=pct(np.array(s6) - np.array(sr)))
    print(f"ablation {nm:8s} intact {out[f'abl/{nm}']['intact']['pct']:+.1f}%  zero six {out[f'abl/{nm}']['zero6']['pct']:+.1f}% [{out[f'abl/{nm}']['zero6']['lo']:+.1f},{out[f'abl/{nm}']['zero6']['hi']:+.1f}]  zero random six {out[f'abl/{nm}']['zero_rand']['pct']:+.1f}% [{out[f'abl/{nm}']['zero_rand']['lo']:+.1f},{out[f'abl/{nm}']['zero_rand']['hi']:+.1f}]  six-rand p={out[f'abl/{nm}']['six_vs_rand']['p']:.4f}")
json.dump(out, open(sys.argv[2], "w"), indent=1)
