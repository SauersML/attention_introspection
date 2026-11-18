# Arm C specificity, in-order labels vs the shuffled-label run (results/privileged), paragraph setups, W1.
# usage: analyze_inorder.py OUT.json INORDER_RECS... -- SHUFFLED_RECS...
import json, sys, numpy as np
a = sys.argv[2:]; i = a.index("--")
load = lambda fs: {r["id"]: r for f in fs for r in map(json.loads, open(f)) if r["setup"] in ("par_arith", "par_qa")}
IN, SH = load(a[:i]), load(a[i + 1:])
rng = np.random.default_rng(0)
def spec(r, m):
    n0 = np.array(r["none"]["W1"]); s = []
    for k in range(4):
        d = np.array(r["C"][f"{k}|{m}"]["W1"]) - n0
        s.append(d[k] - np.delete(d, k).mean())
    return float(np.mean(s))
def stats(x):
    x = np.asarray(x); n = len(x)
    b = x[rng.integers(0, n, (10000, n))].mean(1)
    flips = rng.choice([-1.0, 1.0], (200000, n))
    null = (flips * x).mean(1)
    p = float((np.sum(np.abs(null) >= abs(x.mean())) + 1) / (200000 + 1))
    mu = float(x.mean())
    return dict(n=n, mean=mu, lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), pct=float(np.exp(mu) - 1), p_signflip=p)
def mass(R, m):
    return float(np.exp(np.mean([np.log(r["massC"][f"{k}|{m}"][k] / r["mass_none_rep"][k]) for r in R for k in range(4)])))
res = {}
ids = sorted(set(IN) & set(SH))
res["n_in"], res["n_sh"], res["n_common"] = len(IN), len(SH), len(ids)
res["ntok_match"] = sum(IN[j]["ntok"] == SH[j]["ntok"] for j in ids)
res["same_answer"] = sum(IN[j]["ans"] == SH[j]["ans"] for j in ids)
res["correct_in"] = {s: float(np.mean([r["correct"] for r in IN.values() if r["setup"] == s])) for s in ("par_arith", "par_qa")}
res["cache_check"] = [r["cache_check_maxdiff"] for r in IN.values() if "cache_check_maxdiff" in r]
for m in (5.0, 10.0):
    for sel in (("par_arith", "par_qa"), ("par_arith",), ("par_qa",)):
        nm = "+".join(sel)
        ii = [j for j in ids if IN[j]["setup"] in sel]
        res[f"in|{nm}|{m}"] = stats([spec(IN[j], m) for j in ii])
        res[f"sh|{nm}|{m}"] = stats([spec(SH[j], m) for j in ii])
        res[f"in-sh|{nm}|{m}"] = stats([spec(IN[j], m) - spec(SH[j], m) for j in ii])
    res[f"mass_in|{m}"] = mass(list(IN.values()), m)
    res[f"mass_sh|{m}"] = mass([SH[j] for j in ids], m)
# no-boost report distribution: mean P(slot) in-order vs shuffled
res["none_P_by_slot_in"] = np.exp(np.array([IN[j]["none"]["W1"] for j in ids])).mean(0).tolist()
res["none_P_by_slot_sh"] = np.exp(np.array([SH[j]["none"]["W1"] for j in ids])).mean(0).tolist()
json.dump(res, open(sys.argv[1], "w"), indent=1)
for k, v in res.items(): print(k, v)
