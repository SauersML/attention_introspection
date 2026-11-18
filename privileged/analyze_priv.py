# Tables and prereg rules for the privileged-channel experiment. usage: analyze_priv.py OUTDIR RECS.jsonl...
import json, sys, math, glob
import numpy as np
from scipy.stats import spearmanr

OUT = sys.argv[1]
R = []
for f in sys.argv[2:]:
    R += [json.loads(l) for l in open(f) if l.strip()]
R.sort(key=lambda r: r["id"])
DOSES = [0.25, 0.5, 2.0, 5.0, 10.0]
ALLD = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
SETUPS = ["par_arith", "par_qa", "dialogue", "list", "code", "document", "sentences"]
WS = ["W1", "W2", "W3", "W4", "W5"]
rng = np.random.default_rng(0)
B = 10000

def boot(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n == 0: return dict(mean=None, lo=None, hi=None, n=0)
    idx = rng.integers(0, n, (B, n))
    m = x[idx].mean(1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=n)

def gboot(x):  # geometric mean of ratios
    b = boot(np.log(x))
    return {k: (math.exp(v) if k in ("mean", "lo", "hi") and v is not None else v) for k, v in b.items()}

def delta(r, arm, k, m, w):
    return np.array(r[arm][f"{k}|{m}"][w]) - np.array(r["none"][w])

def spec(r, arm, m, w):
    if m == 1.0: return 0.0
    N = r["N"]; s = []
    for k in range(N):
        d = delta(r, arm, k, m, w)
        s.append(d[k] - np.mean([d[j] for j in range(N) if j != k]))
    return float(np.mean(s))

def sel(setup=None): return [r for r in R if setup is None or r["setup"] == setup]
def pct(x): return None if x is None else 100 * (math.exp(x) - 1)

res = dict(n=len(R), n_by_setup={s: len(sel(s)) for s in SETUPS})
# cached-path check
cc = [r["cache_check_maxdiff"] for r in R if "cache_check_maxdiff" in r]
res["cache_check_maxdiff"] = max(cc) if cc else None
res["accuracy"] = {s: float(np.mean([r["correct"] for r in sel(s)])) for s in SETUPS if sel(s)}

# manipulation checks
def mass_ratio(rs, arm, m):
    out = []
    for r in rs:
        mk = "massA" if arm == "A" else "massC"; base = r["mass_none_ans"] if arm == "A" else r["mass_none_rep"]
        out.append(math.exp(np.mean([math.log(r[mk][f"{k}|{m}"][k] / base[k]) for k in range(r["N"])])))
    return gboot(out)
res["manip"] = {arm: {str(m): mass_ratio(R, arm, m) for m in DOSES} for arm in ["A", "C"]}
res["manip_by_setup_A5"] = {s: mass_ratio(sel(s), "A", 5.0) for s in SETUPS if sel(s)}

# specificity tables
res["spec"] = {}
for arm in ["A", "B", "C"]:
    for w in (WS if arm == "A" else WS[:4]):
        for s in [None] + SETUPS:
            rs = sel(s)
            if not rs: continue
            for m in DOSES:
                res["spec"][f"{arm}|{w}|{s or 'all'}|{m}"] = boot([spec(r, arm, m, w) for r in rs])
# B - A (paired)
res["B_minus_A"] = {str(m): boot([spec(r, "B", m, "W1") - spec(r, "A", m, "W1") for r in R]) for m in DOSES}

# dose: per-context Spearman of log m vs specificity
def dose_rho(rs, arm, w):
    out = []
    for r in rs:
        y = [spec(r, arm, m, w) for m in ALLD]
        if np.allclose(y, y[0]): out.append(np.nan); continue
        out.append(spearmanr(np.log(ALLD), y)[0])
    return boot(out)
res["dose_rho"] = {f"{arm}|{s or 'all'}": dose_rho(sel(s), arm, "W1") for arm in ["A", "B", "C"] for s in [None] + SETUPS if sel(s)}
res["dose_means"] = {arm: [0.0 if m == 1.0 else res["spec"][f"{arm}|W1|all|{m}"]["mean"] for m in ALLD] for arm in ["A", "B", "C"]}

# 4x4 matrices (m = 5, W1, arm A) for 4-segment setups, mean delta logp
res["matrix"] = {}
for s in SETUPS:
    rs = [r for r in sel(s) if r["N"] == 4]
    if not rs: continue
    for arm in ["A", "C"]:
        M = np.mean([[delta(r, arm, k, 5.0, "W1") for k in range(4)] for r in rs], 0)
        res["matrix"][f"{arm}|{s}"] = M.tolist()
# disturbance size (mean |delta|) at m=5
res["disturb_A5"] = boot([np.mean([np.mean(np.abs(delta(r, "A", k, 5.0, "W1"))) for k in range(r["N"])]) for r in R])

# ---------------- rules ----------------
def excl(b): return b["lo"] > 0 or b["hi"] < 0
def sign(b): return 1 if b["mean"] > 0 else -1
rules = {}
rules["R0"] = res["manip"]["A"]["5.0"]["mean"] >= 2
r1 = res["spec"]["A|W1|all|5.0"]
rules["R1"] = excl(r1); s1 = sign(r1)
rules["R1_sign"] = "+" if s1 > 0 else "-"
rr = res["dose_rho"]["A|all"]
rules["R2"] = excl(rr) and sign(rr) == s1
dm = res["dose_means"]["A"]
rules["R2_monotone_means"] = bool(all(np.diff(dm) * s1 > 0))
wpass = [w for w in WS if excl(res["spec"][f"A|{w}|all|5.0"]) and sign(res["spec"][f"A|{w}|all|5.0"]) == s1]
rules["R3"] = len(wpass) >= 4; rules["R3_wordings"] = wpass
spass = [s for s in SETUPS if excl(res["spec"][f"A|W1|{s}|5.0"]) and sign(res["spec"][f"A|W1|{s}|5.0"]) == s1]
sopp = [s for s in SETUPS if excl(res["spec"][f"A|W1|{s}|5.0"]) and sign(res["spec"][f"A|W1|{s}|5.0"]) != s1]
rules["R4"] = len(spass) >= 5; rules["R4_setups"] = spass; rules["setups_opposite_sign"] = sopp
if not rules["R1"]: v = "No specific channel"
elif rules["R0"] and s1 > 0 and rules["R2"] and rules["R3"] and rules["R4"]: v = "Privileged introspective channel"
elif rules["R0"] and s1 < 0 and rules["R2"]: v = "Specific privileged trace, not introspective"
elif not rules["R2"]: v = "Specific trace, but not shown to be dose-dependent"
else: v = "R1 passes; see rules"
rules["verdict"] = v + (" (setups disagree in sign)" if sopp else "")
if not rules["R0"]: rules["verdict"] += " [R0 failed: uninformative]"
res["rules"] = rules
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)

# ---------------- tables ----------------
def f(b, p=False, d=2):
    if b["mean"] is None: return "–"
    if p: return f"{pct(b['mean']):+.1f}% [{pct(b['lo']):+.1f}, {pct(b['hi']):+.1f}]"
    return f"{b['mean']:+.{d}f} [{b['lo']:+.{d}f}, {b['hi']:+.{d}f}]"
L = [f"# Privileged-channel tables (n = {len(R)} contexts: " + ", ".join(f"{s} {res['n_by_setup'][s]}" for s in SETUPS) + ")", ""]
L += [f"Cached-path check, max |Δ log p| vs uncached forward: {res['cache_check_maxdiff']}", ""]
L += ["## Manipulation check: attention mass on the boosted segment, ratio to none (geometric mean)", "",
      "| dose | arm A (answer queries) | arm C (report-question queries) |", "|---|---|---|"]
for m in DOSES:
    a = res["manip"]["A"][str(m)]; c = res["manip"]["C"][str(m)]
    L.append(f"| ×{m:g} | {a['mean']:.2f}× [{a['lo']:.2f}, {a['hi']:.2f}] | {c['mean']:.2f}× [{c['lo']:.2f}, {c['hi']:.2f}] |")
L += ["", "## Rotation specificity (diagonal − mean off-diagonal, Δ log p; % = change in P(boosted) relative to the others), W1, all contexts", "",
      "| dose | A answer only | B answer + question | C question only | B − A |", "|---|---|---|---|---|"]
for m in DOSES:
    L.append(f"| ×{m:g} | " + " | ".join(f(res['spec'][f'{a}|W1|all|{m}']) + f" ({pct(res['spec'][f'{a}|W1|all|{m}']['mean']):+.0f}%)" for a in "ABC") + f" | {f(res['B_minus_A'][str(m)])} |")
L += ["", "Dose (per-context Spearman of log multiplier vs specificity, W1): " + ", ".join(f"{a} {f(res['dose_rho'][f'{a}|all'])}" for a in "ABC"), ""]
L += ["## Arm A, by wording (m = 5; W5 in rating points 0–10)", "", "| wording | all | " + " | ".join(SETUPS) + " |", "|---|" + "---|" * (len(SETUPS) + 1)]
for w in WS:
    L.append(f"| {w} | {f(res['spec'][f'A|{w}|all|5.0'])} | " + " | ".join(f(res['spec'][f'A|{w}|{s}|5.0']) if f'A|{w}|{s}|5.0' in res['spec'] else '–' for s in SETUPS) + " |")
L += ["", "## Arm A by setup and dose (W1)", "", "| setup | " + " | ".join(f"×{m:g}" for m in DOSES) + " | Spearman |", "|---|" + "---|" * (len(DOSES) + 1)]
for s in ["all"] + SETUPS:
    if f"A|W1|{s}|5.0" not in res["spec"]: continue
    L.append(f"| {s} | " + " | ".join(f(res['spec'][f'A|W1|{s}|{m}']) for m in DOSES) + f" | {f(res['dose_rho'][f'A|{s}'])} |")
for arm in ["B", "C"]:
    L += ["", f"## Arm {arm} by setup and dose (W1)", "", "| setup | " + " | ".join(f"×{m:g}" for m in DOSES) + " | Spearman |", "|---|" + "---|" * (len(DOSES) + 1)]
    for s in ["all"] + SETUPS:
        if f"{arm}|W1|{s}|5.0" not in res["spec"]: continue
        L.append(f"| {s} | " + " | ".join(f(res['spec'][f'{arm}|W1|{s}|{m}']) for m in DOSES) + f" | {f(res['dose_rho'][f'{arm}|{s}'])} |")
    L += ["", f"Arm {arm} by wording (m = 5, all): " + ", ".join(f"{w} {f(res['spec'][f'{arm}|{w}|all|5.0'])}" for w in WS[:4])]
L += ["", "## 4×4 matrices, arm A, m = 5, W1: mean Δ log p (rows = boosted segment position, columns = reported segment position)", ""]
for key, M in res["matrix"].items():
    if not key.startswith("A|"): continue
    L += [f"{key[2:]}:", "", "| boosted \\ reported | 1 | 2 | 3 | 4 |", "|---|---|---|---|---|"]
    L += [f"| {i+1} | " + " | ".join(f"{x:+.2f}" for x in row) + " |" for i, row in enumerate(M)] + [""]
L += [f"Overall report disturbance at m = 5 (mean |Δ log p|): {f(res['disturb_A5'])}", ""]
L += ["## Rules", "", "```", json.dumps(rules, indent=1), "```"]
open(f"{OUT}/tables.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
