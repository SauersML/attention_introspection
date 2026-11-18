# Length-matched mention control (Qwen3-32B, the same 150 arithmetic contexts as qwen3-32b-dose, seed 7).
# "Completely ignore paragraph [Pk]. Do not look at it at all." is 16 tokens; the mention line is 12. Here a 16-token mention,
# "Note that one of the four paragraphs above is the one labelled [Pk].", is compared with the 12-token one and with completely-ignore.
# Measure as dose_analyze.py: per-token attention on Pk, mean over heads, answer-token query positions ("att_mean"), layers 21-53
# (and layer 28); % change per context; 95% bootstrap CI over contexts. Code: ../qwen3-32b-dose/dose.py with
# DOSE_CONDS=0_none,1_mention,9_completely_ignore,1b_mention16 (mention16_run.sh). Writes mention16_results.json.
import json, numpy as np
R = json.load(open("dose.json")); by = {(r["id"], r["cond"]): r for r in R}; ids = sorted({r["id"] for r in R})
BI = np.random.default_rng(0).integers(0, len(ids), (10000, len(ids))); out = {"n": len(ids)}
def pk(r, L, key="att_mean"): return np.array(r[key])[L].mean(0)[r["k"]]
for nm, L in [("band21_53", list(range(21, 54))), ("layer28", [28])]:
    M = {c: np.array([pk(by[i, c], L) for i in ids]) for c in ["0_none", "1_mention", "1b_mention16", "9_completely_ignore"]}
    for a, b in [("1b_mention16", "1_mention"), ("9_completely_ignore", "1_mention"), ("9_completely_ignore", "1b_mention16"), ("1b_mention16", "0_none"), ("1_mention", "0_none")]:
        z = 100 * (M[a] / M[b] - 1); bs = z[BI].mean(1)
        out[f"{nm}|{a}_vs_{b}"] = dict(mean=float(z.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), share_pos=float((z > 0).mean()))
        print(f"{nm} {a} vs {b}: {z.mean():+.1f}% [{np.percentile(bs,2.5):+.1f}, {np.percentile(bs,97.5):+.1f}], >0 in {100*(z>0).mean():.0f}%")
out["free_same"] = {c: float(np.mean([by[i, c]["free_same"] for i in ids])) for c in ["1_mention", "1b_mention16", "9_completely_ignore"]}
print(out["free_same"])
json.dump(out, open("mention16_results.json", "w"), indent=1)
