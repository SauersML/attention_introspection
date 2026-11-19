"""Score salience runs: per arm (A attention boost x20, B matched residual-content addition, B2 = 2x B) vs no boost, paired per context.
Reads: greedy reply starts "Yes"; P(Yes) renormalised over Yes/No; after prefill "Yes, paragraph [P": top digit = boosted k, and
P(k's digit) renormalised over 1-4. Per context, boosted arms are averaged over k = P1..P4. Relative change = (arm mean - no-boost mean)
/ no-boost mean; 95% bootstrap CIs over contexts (10,000); p: two-sided sign-flip over contexts (20,000) on paired per-context differences.
usage: python3 salience/score.py"""
import json, glob, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(ROOT, "results/salience/data")
recs = sorted((json.loads(l) for f in glob.glob(f"{D}/recs_*.jsonl") for l in open(f)), key=lambda r: r["id"])
n = len(recs); rng = np.random.default_rng(0)
yes = lambda t: float(t["text"].strip().startswith("Yes"))
py = lambda t: 1 / (1 + np.exp(t["lp_no"] - t["lp_yes"]))
top = lambda t, k: float(int(np.argmax(t["lp_par"])) == k)
pk = lambda t, k: float(np.exp(t["lp_par"][k]))
READS = dict(greedy_yes=lambda t, k: yes(t), p_yes=lambda t, k: py(t), top_is_k=top, p_k=pk)
ARMS = [a for a in ["A", "B", "B2", "R"] if a in recs[0]]
X = {}
for rd, f in READS.items():
    X[rd] = {"none": np.array([np.mean([f(r["none"], k) for k in range(4)]) for r in recs])}
    for a in ARMS: X[rd][a] = np.array([np.mean([f(r[a][str(k)], k) for k in range(4)]) for r in recs])
B = rng.integers(0, n, (10000, n))
def ci(v): b = v[B].mean(1); return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
def rel(a, base):
    m = lambda idx: (a[idx].mean() - base[idx].mean()) / base[idx].mean()
    b = (a[B].mean(1) - base[B].mean(1)) / base[B].mean(1)
    return [float(100 * m(slice(None))), float(100 * np.percentile(b, 2.5)), float(100 * np.percentile(b, 97.5))]
def signflip(d):
    s = rng.choice([-1, 1], (20000, len(d))); null = np.abs((s * d).mean(1)); return float((np.sum(null >= abs(d.mean())) + 1) / 20001)
out = dict(n=n, reads={})
for rd in READS:
    o = dict(none=ci(X[rd]["none"]))
    for a in ARMS:
        o[a] = dict(rate=ci(X[rd][a]), rel_vs_none=rel(X[rd][a], X[rd]["none"]), p_vs_none=signflip(X[rd][a] - X[rd]["none"]))
    o["A_minus_B"] = dict(diff=ci(X[rd]["A"] - X[rd]["B"]), p=signflip(X[rd]["A"] - X[rd]["B"]))
    o["A_minus_B2"] = dict(diff=ci(X[rd]["A"] - X[rd]["B2"]), p=signflip(X[rd]["A"] - X[rd]["B2"]))
    out["reads"][rd] = o
RL = json.load(open(f"{D}/meta.json"))["RL"]
def lay(a, key): return np.mean([[r[a][str(k)][key] for k in range(4)] for r in recs], axis=(0, 1)).tolist()
out["layers"] = RL
out["dres"] = {a: lay(a, "dres") for a in ARMS}; out["cos"] = {a: lay(a, "cos") for a in ARMS}
out["c"] = np.mean([[r["A"][str(k)]["c"] for k in range(4)] for r in recs], axis=(0, 1)).tolist()
out["resnorm"] = np.mean([r["resnorm"] for r in recs], 0).tolist()
out["mass_ratio_A"] = float(np.mean([r["A"][str(k)]["mass"][k] / r["none"]["mass"][k] for r in recs for k in range(4)]))
out["mass_ratio_B"] = float(np.mean([r["B"][str(k)]["mass"][k] / r["none"]["mass"][k] for r in recs for k in range(4)]))
from collections import Counter
out["texts"] = {a: Counter(r[a][str(k)]["text"] for r in recs for k in range(4)).most_common(8) for a in ARMS}
out["texts"]["none"] = Counter(r["none"]["text"] for r in recs).most_common(8)
out["digit_mass_min"] = float(min(min(r[a][str(k)]["digit_mass"] for a in ARMS for k in range(4)) for r in recs))
json.dump(out, open(os.path.join(ROOT, "results/salience/scores.json"), "w"), indent=1)
for rd in READS:
    o = out["reads"][rd]; print(rd, "none %.3f" % o["none"][0], *["%s %.3f rel %+.0f%% [%+.0f,%+.0f] p %.3g" % (a, o[a]["rate"][0], *o[a]["rel_vs_none"], o[a]["p_vs_none"]) for a in ARMS], "A-B p %.3g A-B2 p %.3g" % (o["A_minus_B"]["p"], o["A_minus_B2"]["p"]))
print("dres L21,30,40,53,54:", {a: [round(out["dres"][a][i], 1) for i in (0, 9, 19, 32, 33)] for a in ARMS})
print("cos:", {a: [round(out["cos"][a][i], 2) for i in (0, 9, 19, 32, 33)] for a in ARMS})
print("c:", [round(out["c"][i], 1) for i in (0, 9, 19, 32)], "resnorm", [round(out["resnorm"][i]) for i in (0, 19, 33)], "mass A/B", out["mass_ratio_A"], out["mass_ratio_B"])
print(out["texts"]); print("digit mass min", out["digit_mass_min"])
