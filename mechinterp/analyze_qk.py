# Query-side vs key-side swap (qk.pt from qk_all.py). CPU only.
# For each (context, condition): base names [Pa], donor names [Pb]; four attention recomputes per layer and head
# (0 base q,k; 1 donor queries with base keys; 2 base queries with donor keys; 3 donor q,k).
# Selectivity M = mean over chosen heads of log(att to Pb) - log(att to Pa). Fraction moved = (M_v - M_0) / (M_3 - M_0).
import json, torch, numpy as np
from figlib import boot_ci, signflip_p
R = torch.load("qk.pt", weights_only=False)
SETS = json.load(open("patch_summary.json"))["SETS"]
READ = [tuple(x) for x in SETS["readers"]]
def M(att, a, b, heads=None, Ls=range(21, 54)):  # att [NL, H, 4v, 4p] -> [4v]
    x = att.float().clamp_min(1e-12).log()
    d = x[..., b] - x[..., a]  # [NL, H, 4v]
    if heads is not None: return torch.stack([d[L, h] for L, h in heads]).mean(0)
    return d[list(Ls)].mean((0, 1))
out = {}
rng = np.random.default_rng(0)
for cond in ["mention", "focus", "ignore"]:
    rs = [r for r in R if r["cond"] == cond]
    for name, heads in [("band", None), ("readers", READ)]:
        m = np.stack([M(r["att"], r["a"], r["b"], heads).numpy() for r in rs])  # [n, 4]
        den = m[:, 3] - m[:, 0]
        idx = rng.integers(0, len(den), (4000, len(den)))
        res = {}
        for v, vn in [(1, "query"), (2, "key")]:
            num = m[:, v] - m[:, 0]
            rb = num[idx].mean(1) / den[idx].mean(1)
            res[vn] = [float(num.mean() / den.mean()), float(np.quantile(rb, .025)), float(np.quantile(rb, .975))]
        res["p_query_vs_key"] = signflip_p((m[:, 1] - m[:, 2]))
        res["donor_minus_base"] = boot_ci(den)
        res["n"] = len(den)
        out[f"{cond}/{name}"] = res
    # per layer, head-mean
    per = []
    for L in range(64):
        m = np.stack([M(r["att"], r["a"], r["b"], None, [L]).numpy() for r in rs]); den = (m[:, 3] - m[:, 0]).mean()
        per.append([float(den), float((m[:, 1] - m[:, 0]).mean()), float((m[:, 2] - m[:, 0]).mean())])
    out[f"{cond}/per_layer"] = per
    print(cond, {k: v for k, v in out.items() if k.startswith(cond) and not k.endswith("per_layer")})
json.dump(out, open("qk_summary.json", "w"), default=float)
