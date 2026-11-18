# Label-token sham (sham_label.py): Pk-specific shift in the report's log-probability (change for Pk minus mean change for the
# other three) for the body boost and for the same bias on the label tokens "[Pk]" only; plus attention-mass ratios on Pk's body
# and on its label (answer queries, layers 21-53). 95% bootstrap CI over contexts. Writes sham_label_results.json.
import json, numpy as np
R = json.load(open("SHAMLABEL.json")); L = json.load(open("meta.json"))["LAYERS"] if "LAYERS" in json.load(open("meta.json")) else list(range(21, 54))
rng = np.random.default_rng(1)
def boot(x):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (10000, len(x)))].mean(1); return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def spec(r, c):
    d = np.array(r[c]["rep_lp"]) - np.array(r["none"]["rep_lp"]); k = r["k"]; return d[k] - np.mean([d[m] for m in range(4) if m != k])
band = lambda a: np.array(a)[L].mean(0)
out = dict(n=len(R), ntok_label=sorted({r["ntok_label"] for r in R}), label_tokens_example=R[0]["label_tokens"])
for c in ["boost", "label", "label4"]:
    same = np.array([r[c]["ans"].strip() == r["none"]["ans"].strip() for r in R]); sh = np.array([spec(r, c) for r in R])
    body = np.exp(np.mean([np.log(band(r[c]["att_mean"])[r["k"]] / band(r["none"]["att_mean"])[r["k"]]) for r in R]))
    lab = np.exp(np.mean([np.log(band(r[c]["att_mean"])[4] / band(r["none"]["att_mean"])[4]) for r in R]))
    out[c] = dict(shift=boot(sh), shift_identical=boot(sh[same]), n_identical=int(same.sum()), body_mass_ratio=float(body), label_mass_ratio=float(lab),
                  picks_k=[float(np.mean([np.argmax(r["none"]["rep"]) == r["k"] for r in R])), float(np.mean([np.argmax(r[c]["rep"]) == r["k"] for r in R]))])
    print(c, json.dumps(out[c]))
for c in ["label", "label4"]: out[f"boost_minus_{c}"] = boot([spec(r, "boost") - spec(r, c) for r in R]); print(f"boost - {c}", out[f"boost_minus_{c}"])
json.dump(out, open("sham_label_results.json", "w"), indent=1)
