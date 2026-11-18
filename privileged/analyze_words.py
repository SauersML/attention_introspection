# Tables for the single-word privileged-channel test and the instruction test. usage: analyze_words.py OUTDIR RECS.jsonl...
import json, sys, math
import numpy as np
from scipy.stats import spearmanr
OUT = sys.argv[1]
R = []
for f in sys.argv[2:]: R += [json.loads(l) for l in open(f) if l.strip()]
R.sort(key=lambda r: r["id"])
DOSES = [0.25, 0.5, 2.0, 5.0, 10.0]; ALLD = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
rng = np.random.default_rng(0); B = 10000
def boot(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    idx = rng.integers(0, n, (B, n)); m = x[idx].mean(1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=n, p_two_sided=float(min(1, 2 * min((m <= 0).mean(), (m >= 0).mean()))))
def gboot(x):
    b = boot(np.log(x)); return {k: (math.exp(v) if k in ("mean", "lo", "hi") else v) for k, v in b.items()}
def delta(r, arm, k, m): return np.array(r[arm][f"{k}|{m}"]["W1"]) - np.array(r["none"]["W1"])
def spec(r, arm, m):
    if m == 1.0: return 0.0
    N = r["N"]; return float(np.mean([delta(r, arm, k, m)[k] - np.mean([delta(r, arm, k, m)[j] for j in range(N) if j != k]) for k in range(N)]))
pct = lambda x: 100 * (math.exp(x) - 1)
res = dict(n=len(R), accuracy=float(np.mean([r["correct"] for r in R])))
cc = [r["cache_check_maxdiff"] for r in R if "cache_check_maxdiff" in r]; res["cache_check_maxdiff"] = max(cc) if cc else None
def mass_ratio(arm, m):
    mk, bk = ("massA", "mass_none_ans") if arm == "A" else ("massC", "mass_none_rep")
    return gboot([math.exp(np.mean([math.log(r[mk][f"{k}|{m}"][k] / r[bk][k]) for k in range(r["N"])])) for r in R])
res["manip"] = {a: {str(m): mass_ratio(a, m) for m in DOSES} for a in "AC"}
res["spec"] = {f"{a}|{m}": boot([spec(r, a, m) for r in R]) for a in "ABC" for m in DOSES}
res["B_minus_A"] = {str(m): boot([spec(r, "B", m) - spec(r, "A", m) for r in R]) for m in DOSES}
def rho(a):
    out = []
    for r in R:
        y = [spec(r, a, m) for m in ALLD]
        out.append(np.nan if np.allclose(y, y[0]) else spearmanr(np.log(ALLD), y)[0])
    return boot(out)
res["dose_rho"] = {a: rho(a) for a in "ABC"}
res["dose_means"] = {a: [0.0 if m == 1.0 else res["spec"][f"{a}|{m}"]["mean"] for m in ALLD] for a in "ABC"}
res["disturb_A5"] = boot([np.mean([np.mean(np.abs(delta(r, "A", k, 5.0))) for k in range(r["N"])]) for r in R])
# baseline: how peaked is the unboosted report; does it favour list position?
P0 = np.array([np.exp(r["none"]["W1"]) for r in R])
res["none_report_by_position"] = P0.mean(0).tolist()
res["none_top_prob_mean"] = float(P0.max(1).mean())
# instruction test: per context, mean over named word k of log(mass_cond / mass_ref) at the named word's position
def instr(cond, ref, key):
    out, frac = [], []
    for r in R:
        lr = []
        for l in range(1, r["N"] + 1):
            pos = r["lab"].index(l)
            a = r["instr"][f"{cond}{l}"][key][pos]
            b = r["instr"]["none" if ref == "none" else f"{ref}{l}"][key][pos]
            lr.append(math.log(a / b))
        out.append(np.mean(lr)); frac += [x > 0 for x in lr]
    b = gboot(np.exp(out)); b["pct"] = 100 * (b["mean"] - 1); b["pct_lo"] = 100 * (b["lo"] - 1); b["pct_hi"] = 100 * (b["hi"] - 1)
    b["frac_named_up"] = float(np.mean(frac)); return b
res["instr"] = {f"{c}|{ref}|{key}": instr(c, ref, key) for c in ("focus", "ignore", "mention") for ref in ("mention", "none") for key in ("band", "L28") if c != ref}
res["instr_free_same"] = {c: float(np.mean([v["free_same"] for r in R for kk, v in r["instr"].items() if kk.startswith(c)])) for c in ("mention", "focus", "ignore")}
# rotation for instructions: is the named word the one whose attention rises most (focus) / falls most (ignore), vs mention of the same label
def rot(cond, key):
    hit = []
    for r in R:
        for l in range(1, r["N"] + 1):
            pos = r["lab"].index(l)
            d = np.log(np.array(r["instr"][f"{cond}{l}"][key])) - np.log(np.array(r["instr"][f"mention{l}"][key]))
            hit.append(int(np.argmax(d) if cond == "focus" else np.argmin(d)) == pos)
    return float(np.mean(hit))
res["instr_rotation_hit"] = {f"{c}|{key}": rot(c, key) for c in ("focus", "ignore") for key in ("band", "L28")}
# instruction specificity (added after the first read): log ratio cond/mention at the named word minus the mean log ratio
# at the other seven words, per context (mean over named word k); and the other words' own change
def instr_spec(cond, key):
    s, o = [], []
    for r in R:
        v = []
        for l in range(1, r["N"] + 1):
            pos = r["lab"].index(l)
            d = np.log(np.array(r["instr"][f"{cond}{l}"][key])) - np.log(np.array(r["instr"][f"mention{l}"][key]))
            oth = np.delete(d, pos).mean(); v.append((d[pos] - oth, oth))
        v = np.mean(v, 0); s.append(v[0]); o.append(v[1])
    b = gboot(np.exp(s)); bo = gboot(np.exp(o))
    return dict(named_minus_others_pct=100 * (b["mean"] - 1), lo=100 * (b["lo"] - 1), hi=100 * (b["hi"] - 1),
                frac_ctx_right_way=float(np.mean([(x > 0) if cond == "focus" else (x < 0) for x in s])),
                others_pct=100 * (bo["mean"] - 1), others_lo=100 * (bo["lo"] - 1), others_hi=100 * (bo["hi"] - 1))
res["instr_specificity_vs_mention"] = {f"{c}|{key}": instr_spec(c, key) for c in ("focus", "ignore") for key in ("band", "L28")}
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
def f(b, d=3): return f"{b['mean']:+.{d}f} [{b['lo']:+.{d}f}, {b['hi']:+.{d}f}]"
L = [f"n = {len(R)} contexts; arithmetic accuracy {res['accuracy']:.0%}; cached-path check max |Δ log p| {res['cache_check_maxdiff']}", "",
     "Manipulation (attention mass on boosted word, ratio to none):", ""]
L += ["| dose | A (answer queries) | C (question queries) |", "|---|---|---|"] + [f"| ×{m:g} | {res['manip']['A'][str(m)]['mean']:.2f}× | {res['manip']['C'][str(m)]['mean']:.2f}× |" for m in DOSES]
L += ["", "| dose | A answer only | B both | C question only | B − A |", "|---|---|---|---|---|"]
for m in DOSES:
    L.append(f"| ×{m:g} | " + " | ".join(f"{f(res['spec'][f'{a}|{m}'])} ({pct(res['spec'][f'{a}|{m}']['mean']):+.0f}%)" for a in "ABC") + f" | {f(res['B_minus_A'][str(m)])} |")
L += ["", "Spearman: " + ", ".join(f"{a} {f(res['dose_rho'][a], 2)}" for a in "ABC"), ""]
L += ["| instruction | vs | band 21–53 | layer 28 | named word up (band) |", "|---|---|---|---|---|"]
for k, b in res["instr"].items():
    if not k.endswith("band"): continue
    c, ref, _ = k.split("|"); b28 = res["instr"][f"{c}|{ref}|L28"]
    L.append(f"| {c} | {ref} | {b['pct']:+.1f}% [{b['pct_lo']:+.1f}, {b['pct_hi']:+.1f}] | {b28['pct']:+.1f}% [{b28['pct_lo']:+.1f}, {b28['pct_hi']:+.1f}] | {b['frac_named_up']:.0%} |")
L += ["", "| instruction (vs mention) | named word minus the other words, band | layer 28 | other words' own change, band | layer 28 |", "|---|---|---|---|---|"]
for c in ("focus", "ignore"):
    b, b28 = res["instr_specificity_vs_mention"][f"{c}|band"], res["instr_specificity_vs_mention"][f"{c}|L28"]
    L.append(f"| {c} | {b['named_minus_others_pct']:+.1f}% [{b['lo']:+.1f}, {b['hi']:+.1f}] | {b28['named_minus_others_pct']:+.1f}% [{b28['lo']:+.1f}, {b28['hi']:+.1f}] | {b['others_pct']:+.1f}% [{b['others_lo']:+.1f}, {b['others_hi']:+.1f}] | {b28['others_pct']:+.1f}% [{b28['others_lo']:+.1f}, {b28['others_hi']:+.1f}] |")
L += ["", f"rotation hit: {res['instr_rotation_hit']}; free answer unchanged: {res['instr_free_same']}"]
open(f"{OUT}/tables.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
