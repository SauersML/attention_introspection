# Scores the word-list told-injection run: python3 told_inject_words/score.py results/told_inject_words/data
# Same measures as told_inject/score.py (8 words instead of 4 paragraphs), plus a shuffle null for "names the boosted word, given Yes":
# within each context and dose, the boosted-word labels of its 8 boosted trials are permuted (10,000 times), and the share
# of Yes replies that name the (permuted) boosted word is recomputed; this keeps each context's naming habits and position bias.
import json, sys, re, math, glob, os
import numpy as np
D = [2.0, 5.0, 10.0, 20.0]; N = 8
R = sorted([json.loads(l) for f in sorted(glob.glob(os.path.join(sys.argv[1], "recs_*.jsonl"))) for l in open(f)], key=lambda r: r["id"])
rng = np.random.default_rng(0)
def ci(x, B=10000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(x))]
def signflip(x, B=20000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    s = rng.choice([-1, 1], (B, len(x))); null = np.abs((s * x).mean(1))
    return float((1 + (null >= abs(x.mean())).sum()) / (B + 1))
PAT = re.compile(r"^\s*(No|Yes, word \[W([1-8])\])\.?\s*$")
def parse(t):
    m = PAT.match(t)
    if not m: return ("other", None)
    return ("no", None) if m.group(1) == "No" else ("yes", int(m.group(2)) - 1)
py = lambda t: 1 / (1 + math.exp(t["lp_no"] - t["lp_yes"]))
C = [r["none"] for r in R]
out = dict(n=len(R), arithmetic_correct=int(sum(r["correct"] for r in R)))
out["control"] = dict(p_yes=ci([py(t) for t in C]), greedy_yes=ci([parse(t["text"])[0] == "yes" for t in C]),
                      p_par=[ci([math.exp(t["lp_par"][j]) for t in C])[0] for j in range(N)],
                      greedy_texts={x: sum(t["text"] == x for t in C) for x in set(t["text"] for t in C)})
odd = [dict(id=r["id"], cond="none", text=r["none"]["text"]) for r in R if parse(r["none"]["text"])[0] == "other"]
out["dose"] = {}
for m in D:
    T = lambda r, k: r["boost"][f"{k}|{m}"]
    P = [[parse(T(r, k)["text"]) for k in range(N)] for r in R]
    dpy = [np.mean([py(T(r, k)) for k in range(N)]) - py(r["none"]) for r in R]
    pyb = [np.mean([py(T(r, k)) for k in range(N)]) for r in R]
    gy = [np.mean([p[k][0] == "yes" for k in range(N)]) for p in P]
    dgy = [g - (parse(r["none"]["text"])[0] == "yes") for g, r in zip(gy, R)]
    pc = [np.mean([math.exp(T(r, k)["lp_par"][k]) for k in range(N)]) for r in R]
    spec = []
    for r in R:
        s = []
        for k in range(N):
            d = np.array(T(r, k)["lp_par"]) - np.array(r["none"]["lp_par"])
            s.append(d[k] - np.delete(d, k).mean())
        spec.append(np.mean(s))
    gc = [p[k][1] == k for p in P for k in range(N) if p[k][0] == "yes"]
    # shuffle null: permute which word counts as boosted among each context's 8 trials at this dose
    said = np.array([[p[k][1] if p[k][0] == "yes" else -1 for k in range(N)] for p in P])   # [ctx, boosted k] -> said word or -1
    ny = int((said >= 0).sum()); obs = float((said == np.arange(N)).sum() / max(ny, 1))
    g = np.random.default_rng(1); null = []
    for _ in range(10000):
        perm = np.argsort(g.random(said.shape), 1)
        null.append((np.take_along_axis(said, perm, 1) == np.arange(N)).sum() / max(ny, 1))
    null = np.array(null)
    # same shuffle for the prefilled read: P(the permuted "boosted" word's digit), averaged over trials
    PD = np.array([[np.exp(T(r, k)["lp_par"]) for k in range(N)] for r in R])   # [ctx, boosted k, digit]
    obs_pd = float(PD[:, np.arange(N), np.arange(N)].mean()); nullpd = []
    for _ in range(2000):
        perm = np.argsort(g.random((len(R), N)), 1)
        nullpd.append(np.take_along_axis(PD, perm[:, :, None], 2)[:, np.arange(N), 0].mean() if False else PD[np.arange(len(R))[:, None], np.arange(N)[None, :], perm].mean())
    nullpd = np.array(nullpd)
    mr = [np.mean([T(r, k)["mass"][k] / r["none"]["mass"][k] for k in range(N)]) for r in R]
    ms_b = [np.mean([T(r, k)["mass"][k] for k in range(N)]) for r in R]
    ms_n = [np.mean([r["none"]["mass"][k] for k in range(N)]) for r in R]
    for r, p in zip(R, P):
        for k in range(N):
            if p[k][0] == "other": odd.append(dict(id=r["id"], cond=f"W{k+1} x{m:g}", text=T(r, k)["text"]))
    hitfa = [np.mean([(parse(T(r, k)["text"]) == ("yes", k)) - (parse(r["none"]["text"]) == ("yes", k)) for k in range(N)]) for r in R]
    bypos = [ci([(parse(T(r, k)["text"])[0] == "yes") - (parse(r["none"]["text"])[0] == "yes") for r in R]) for k in range(N)]
    out["dose"][f"{m:g}"] = dict(yes_naming_boosted_minus_control=ci(hitfa), yes_naming_boosted_p=signflip(hitfa), greedy_yes_minus_control_by_position=bypos,
        p_yes_boost=ci(pyb), p_yes_minus_control=ci(dpy), p_yes_signflip_p=signflip(dpy),
        greedy_yes_boost=ci(gy), greedy_yes_minus_control=ci(dgy), greedy_signflip_p=signflip(dgy),
        p_correct_given_yes=ci(pc), greedy_correct_given_yes=[float(np.mean(gc)) if gc else None, len(gc)],
        shuffle_null_correct_given_yes=[float(null.mean()), float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5)), float((1 + (null >= obs).sum()) / (len(null) + 1))],
        shuffle_null_p_correct=[float(nullpd.mean()), float(np.percentile(nullpd, 2.5)), float(np.percentile(nullpd, 97.5)), float((1 + (nullpd >= obs_pd).sum()) / (len(nullpd) + 1))],
        specificity_logp=ci(spec), specificity_p=signflip(spec), specificity_x=float(math.exp(np.mean(spec))),
        attn_mass_ratio=ci(mr), attn_mass_boost=ci(ms_b)[0], attn_mass_none=ci(ms_n)[0],
        digit_mass=float(np.mean([T(r, k)["digit_mass"] for r in R for k in range(N)])),
        min_digit_mass=float(np.min([T(r, k)["digit_mass"] for r in R for k in range(N)])))
out["odd_texts"] = odd; out["n_odd"] = len(odd); out["n_trials"] = len(R) * (1 + N * len(D))
out["cache_check"] = [r["cache_check"] for r in R if "cache_check" in r]
json.dump(out, open(os.path.join(sys.argv[1], "..", "scores.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "odd_texts"}, indent=0)[:9000]); print("odd", odd[:30])
