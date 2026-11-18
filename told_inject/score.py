# Scores the told-injection run: python3 told_inject/score.py results/told_inject/data
import json, sys, re, math, glob, os
import numpy as np
D = [2.0, 5.0, 10.0, 20.0]
R = sorted([json.loads(l) for f in sorted(glob.glob(os.path.join(sys.argv[1], "recs_*.jsonl"))) for l in open(f)], key=lambda r: r["id"])
rng = np.random.default_rng(0)
def ci(x, B=10000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(x))]
def signflip(x, B=20000):  # two-sided sign-flip p for mean != 0 over contexts
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    s = rng.choice([-1, 1], (B, len(x))); null = np.abs((s * x).mean(1))
    return float((1 + (null >= abs(x.mean())).sum()) / (B + 1))
PAT = re.compile(r"^\s*(No|Yes, paragraph \[P([1-4])\])\.?\s*$")
def parse(t):
    m = PAT.match(t)
    if not m: return ("other", None)
    return ("no", None) if m.group(1) == "No" else ("yes", int(m.group(2)) - 1)
py = lambda t: 1 / (1 + math.exp(t["lp_no"] - t["lp_yes"]))
N = R and [r["none"] for r in R]
out = dict(n=len(R))
out["control"] = dict(p_yes=ci([py(t) for t in N]), greedy_yes=ci([parse(t["text"])[0] == "yes" for t in N]),
                      p_par=[ci([math.exp(t["lp_par"][j]) for t in N])[0] for j in range(4)],
                      greedy_texts={x: sum(t["text"] == x for t in N) for x in set(t["text"] for t in N)})
odd = [dict(id=r["id"], cond="none", text=r["none"]["text"]) for r in R if parse(r["none"]["text"])[0] == "other"]
out["dose"] = {}
for m in D:
    T = lambda r, k: r["boost"][f"{k}|{m}"]
    dpy = [np.mean([py(T(r, k)) for k in range(4)]) - py(r["none"]) for r in R]
    pyb = [np.mean([py(T(r, k)) for k in range(4)]) for r in R]
    gy = [np.mean([parse(T(r, k)["text"])[0] == "yes" for k in range(4)]) for r in R]
    dgy = [g - (parse(r["none"]["text"])[0] == "yes") for g, r in zip(gy, R)]
    pc = [np.mean([math.exp(T(r, k)["lp_par"][k]) for k in range(4)]) for r in R]
    spec = []
    for r in R:
        s = []
        for k in range(4):
            d = np.array(T(r, k)["lp_par"]) - np.array(r["none"]["lp_par"])
            s.append(d[k] - np.delete(d, k).mean())
        spec.append(np.mean(s))
    gc = [parse(T(r, k)["text"])[1] == k for r in R for k in range(4) if parse(T(r, k)["text"])[0] == "yes"]
    mr = [np.mean([T(r, k)["mass"][k] / r["none"]["mass"][k] for k in range(4)]) for r in R]
    ms_b = [np.mean([T(r, k)["mass"][k] for k in range(4)]) for r in R]
    ms_n = [np.mean([r["none"]["mass"][k] for k in range(4)]) for r in R]
    for r in R:
        for k in range(4):
            if parse(T(r, k)["text"])[0] == "other": odd.append(dict(id=r["id"], cond=f"P{k+1} x{m:g}", text=T(r, k)["text"]))
    hitfa = [np.mean([(parse(T(r, k)["text"]) == ("yes", k)) - (parse(r["none"]["text"]) == ("yes", k)) for k in range(4)]) for r in R]
    bypos = [ci([(parse(T(r, k)["text"])[0] == "yes") - (parse(r["none"]["text"])[0] == "yes") for r in R]) for k in range(4)]
    out["dose"][f"{m:g}"] = dict(yes_naming_boosted_minus_control=ci(hitfa), yes_naming_boosted_p=signflip(hitfa), greedy_yes_minus_control_by_position=bypos,p_yes_boost=ci(pyb), p_yes_minus_control=ci(dpy), p_yes_signflip_p=signflip(dpy),
        greedy_yes_boost=ci(gy), greedy_yes_minus_control=ci(dgy), greedy_signflip_p=signflip(dgy),
        p_correct_given_yes=ci(pc), greedy_correct_given_yes=[float(np.mean(gc)) if gc else None, len(gc)],
        specificity_logp=ci(spec), specificity_p=signflip(spec), specificity_x=float(math.exp(np.mean(spec))),
        attn_mass_ratio=ci(mr), attn_mass_boost=ci(ms_b)[0], attn_mass_none=ci(ms_n)[0],
        digit_mass=float(np.mean([T(r, k)["digit_mass"] for r in R for k in range(4)])))
out["odd_texts"] = odd; out["n_odd"] = len(odd); out["n_trials"] = len(R) * 17
out["cache_check"] = [r["cache_check"] for r in R if "cache_check" in r]
json.dump(out, open(os.path.join(sys.argv[1], "..", "scores.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "odd_texts"}, indent=0)[:6000]); print("odd", odd[:30])
