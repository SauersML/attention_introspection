# Arm C (boost only while the report question is read) with three boost targets: the word's token, its label "[Wk]", or both,
# in both label orders. Same specificity as analyze_words.py (change in log P(report = boosted item) minus the mean change for
# the other seven, averaged over the rotation), plus the attention mass achieved on the boosted item's word and label.
# usage: analyze_words_label.py OUTDIR INORDER_DIR SHUFFLED_DIR   (each DIR has word/ label/ both/ with recs_0.jsonl)
import json, sys, math, os
import numpy as np
OUT, DIRS = sys.argv[1], dict(inorder=sys.argv[2], shuffled=sys.argv[3])
DOSES = [0.25, 0.5, 2.0, 5.0, 10.0]
rng = np.random.default_rng(0); B = 10000
def boot(x):
    x = np.asarray(x, float); n = len(x); m = x[rng.integers(0, n, (B, n))].mean(1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=n,
                p=float(min(1, 2 * min((m <= 0).mean(), (m >= 0).mean()))))
def load(d):
    R = [json.loads(l) for f in sorted(os.listdir(d)) if f.startswith("recs_") and f.endswith(".jsonl") for l in open(f"{d}/{f}") if l.strip()]
    return sorted(R, key=lambda r: r["id"])
def delta(r, k, m): return np.array(r["C"][f"{k}|{m}"]["W1"]) - np.array(r["none"]["W1"])
def spec(r, m): N = r["N"]; return float(np.mean([delta(r, k, m)[k] - np.delete(delta(r, k, m), k).mean() for k in range(N)]))
def mass(r, m, which):  # geometric mean over k of (mass on item k's word or label under the boost) / (no boost)
    N = r["N"]; o = 0 if which == "word" else N
    return math.exp(np.mean([math.log(r["massC"][f"{k}|{m}"][o + k] / r["mass_none_rep"][o + k]) for k in range(N)]))
pct = lambda x: 100 * (math.exp(x) - 1)
res, recs = {}, {}
for o, d in DIRS.items():
    for t in ("word", "label", "both"):
        R = load(f"{d}/{t}"); recs[o, t] = R
        for m in DOSES:
            b = boot([spec(r, m) for r in R])
            mw = boot([math.log(mass(r, m, "word")) for r in R]); ml = boot([math.log(mass(r, m, "label")) for r in R])
            res[f"{o}|{t}|{m}"] = dict(spec=b, pct=pct(b["mean"]), pct_lo=pct(b["lo"]), pct_hi=pct(b["hi"]),
                                       mass_word=math.exp(mw["mean"]), mass_label=math.exp(ml["mean"]))
# baseline mass share (no boost) on words and on labels at the report-question queries
for o in DIRS:
    R = recs[o, "word"]
    res[f"{o}|none_mass"] = dict(words=float(np.mean([sum(r["mass_none_rep"][:8]) for r in R])), labels=float(np.mean([sum(r["mass_none_rep"][8:]) for r in R])))
# check against the committed runs (word target): same contexts, same spec
old = dict(inorder="results/privileged_words/results.json", shuffled="results/privileged_words_shuffled/results.json")
for o, f in old.items():
    oj = json.load(open(f)); res[f"{o}|check"] = {str(m): dict(old=oj["spec"][f"C|{m}"]["mean"], new=res[f"{o}|word|{m}"]["spec"]["mean"]) for m in DOSES}
# paired differences: label − word, and in order − shuffled for each target
for m in (5.0, 10.0):
    for o in DIRS:
        a, w = recs[o, "label"], recs[o, "word"]; assert [r["id"] for r in a] == [r["id"] for r in w]
        res[f"{o}|label_minus_word|{m}"] = boot([spec(x, m) - spec(y, m) for x, y in zip(a, w)])
    for t in ("word", "label", "both"):
        res[f"inorder_minus_shuffled|{t}|{m}"] = boot([spec(x, m) - spec(y, m) for x, y in zip(recs["inorder", t], recs["shuffled", t])])
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
L = ["| order | boosted | ×5 | ×10 | mass on word / label at ×10 (ratio to no boost) |", "|---|---|---|---|---|"]
for o in DIRS:
    for t in ("word", "label", "both"):
        a, b = res[f"{o}|{t}|5.0"], res[f"{o}|{t}|10.0"]
        L.append(f"| {o} | {t} | {a['pct']:+.0f}% [{a['pct_lo']:+.0f}, {a['pct_hi']:+.0f}] | {b['pct']:+.0f}% [{b['pct_lo']:+.0f}, {b['pct_hi']:+.0f}] | {b['mass_word']:.2f}× / {b['mass_label']:.2f}× |")
L += ["", "All doses (specificity %):", "", "| order | boosted | " + " | ".join(f"×{m:g}" for m in DOSES) + " |", "|---|---|" + "---|" * len(DOSES)]
for o in DIRS:
    for t in ("word", "label", "both"):
        L.append(f"| {o} | {t} | " + " | ".join(f"{res[f'{o}|{t}|{m}']['pct']:+.0f}%" for m in DOSES) + " |")
L += ["", "Check, word target vs committed runs (specificity, log units):", ""]
for o in DIRS: L.append(f"- {o}: " + ", ".join(f"×{m}: {v['new']:+.3f} (committed {v['old']:+.3f})" for m, v in res[f'{o}|check'].items()))
L += ["", "Paired differences (log units, 95% CI):", ""]
for k, v in res.items():
    if "minus" in k: L.append(f"- {k}: {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}], p {v['p']:.2g}")
L += ["", "No-boost attention mass at the report-question queries (band, head and query mean): " + "; ".join(f"{o}: words {res[f'{o}|none_mass']['words']:.4f}, labels {res[f'{o}|none_mass']['labels']:.4f}" for o in DIRS)]
open(f"{OUT}/tables.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
