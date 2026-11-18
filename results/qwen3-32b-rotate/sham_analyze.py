import json, os, sys
import numpy as np
D = sys.argv[1]
meta = json.load(open(os.path.join(D, "meta.json"))); LAY = meta["LAYERS"]
R = json.load(open(os.path.join(D, "SHAM.json")))
rng = np.random.default_rng(1); NB = 10000
def band(att): return np.array(att)[LAY].mean(0)
def boot(x):
    x = np.asarray(x, float); n = len(x)
    if n == 0: return [float("nan")] * 3
    m = x[rng.integers(0, n, (NB, n))].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def spec(r, cond):
    d = np.array(r[cond]["rep_lp"]) - np.array(r["none"]["rep_lp"]); k = r["k"]
    return d[k] - np.mean([d[m] for m in range(4) if m != k])
def tot(r, cond, which, cols):
    bm = band(r[cond][which])
    return sum(bm[c] * n for c, n in cols)
out = dict(n=len(R), ntok_k=boot([r["ntok_k"] for r in R]), ntok_sham=boot([r["ntok_sham"] for r in R]),
           b_matched_counts={str(b): sum(r["b_matched"] == b for r in R) for b in sorted(set(r["b_matched"] for r in R))})
for cond in ["boost", "sham", "sham_matched"]:
    same = np.array([r[cond]["ans"].strip() == r["none"]["ans"].strip() for r in R])
    sh = np.array([spec(r, cond) for r in R])
    o = dict(shift=boot(sh), shift_identical=boot(sh[same]), n_identical=int(same.sum()),
             # all-report disturbance: mean |Δ log P| over the four options
             mean_abs_dlogp=boot([np.abs(np.array(r[cond]["rep_lp"]) - np.array(r["none"]["rep_lp"])).mean() for r in R]),
             picks_k=[float(np.mean([np.argmax(r["none"]["rep"]) == r["k"] for r in R])), float(np.mean([np.argmax(r[cond]["rep"]) == r["k"] for r in R]))])
    for which in ["att_first", "att_mean"]:
        cols_k = lambda r: [(r["k"], r["ntok_k"])]
        # total mass on boosted Pk and on sham tokens (cols 4,5 per-token means × lengths; lengths reconstructed from ntok_sham split unknown -> use mass via ratio)
        o[f"Pk_mass_ratio_{which}"] = float(np.exp(np.mean([np.log(band(r[cond][which])[r["k"]] / band(r["none"][which])[r["k"]]) for r in R])))
        o[f"sham_tokens_mass_ratio_{which}"] = float(np.exp(np.mean([np.log((band(r[cond][which])[4:6]).sum() / (band(r["none"][which])[4:6]).sum()) for r in R])))
    out[cond] = o
d_bs = np.array([spec(r, "boost") - spec(r, "sham") for r in R]); out["boost_minus_sham"] = boot(d_bs)
d_bm = np.array([spec(r, "boost") - spec(r, "sham_matched") for r in R]); out["boost_minus_sham_matched"] = boot(d_bm)
out["added_mass_boost_first"] = boot([r["added_mass_boost"] for r in R])
out["added_mass_sham_b2_first"] = boot([r["added_mass_sham_grid"]["2.0"] for r in R])
out["added_mass_sham_matched_first"] = boot([r["added_mass_sham_grid"][str(r["b_matched"])] for r in R])
json.dump(out, open(os.path.join(D, "sham_results.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
