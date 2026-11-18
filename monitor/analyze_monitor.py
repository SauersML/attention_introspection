# Tables, pass rule and verdict for the attention-monitoring experiment (results/monitor/PREREG.md).
# usage: analyze_monitor.py OUTDIR SHARD_DIR [SHARD_DIR ...]
import json, os, re, sys, math
import numpy as np

OUTD = sys.argv[1]; DIRS = sys.argv[2:]
os.makedirs(OUTD, exist_ok=True)
BAND = list(range(21, 54))
NB = 10000
rng = np.random.default_rng(0)

C, G, H1, G2, H2 = {}, {}, {}, {}, {}
for d in DIRS:
    for c in json.load(open(os.path.join(d, "contexts.json"))): C[str(c["id"])] = c
    G.update(json.load(open(os.path.join(d, "gen1.json"))))
    H1.update(json.load(open(os.path.join(d, "hf1.json"))))
    if os.path.exists(os.path.join(d, "gen2.json")): G2.update(json.load(open(os.path.join(d, "gen2.json"))))
    if os.path.exists(os.path.join(d, "hf2.json")): H2.update(json.load(open(os.path.join(d, "hf2.json"))))
IDS = sorted(set(C) & set(H1), key=int)
FAMS = ["distractor", "multihop", "arith", "sumone", "longshort"]
print("contexts", len(IDS), "with hf2", len(set(IDS) & set(H2)))

LBL = re.compile(r"P\s*([1-4])")
def parse_rank(s):
    seen = []
    for m in LBL.finditer(s):
        k = int(m.group(1)) - 1
        if k not in seen: seen.append(k)
    if not seen: return None
    return [float(4 - seen.index(k)) if k in seen else 0.0 for k in range(4)]
def parse_pct(s):
    v = [None] * 4
    for m in re.finditer(r"P\s*([1-4])\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*%", s):
        k = int(m.group(1)) - 1
        if v[k] is None: v[k] = float(m.group(2))
    if all(x is None for x in v): return None
    return [x if x is not None else 0.0 for x in v]

SETUPS = [("after", w) for w in ["most", "least", "rank", "pct", "rate", "free"]] + \
         [("before", w) for w in ["most", "rate", "pct"]] + [("later", w) for w in ["most", "rate", "pct"]] + \
         [("think", w) for w in ["most", "pct"]] + [("fb4", w) for w in ["most", "rate", "pct"]] + [("fb16", w) for w in ["most", "rate", "pct"]]
NAMES = {"after": "right after answering", "before": "before answering", "later": "one turn later", "think": "thinking on",
         "fb4": "after 4 feedback examples", "fb16": "after 16 feedback examples"}
WN = {"most": "which most", "least": "which least", "rank": "rank", "pct": "percent", "rate": "rate 0 to 10", "free": "free text"}

def score(i, setup, who):
    t, w = setup
    key = f"{t}|{w}|{who}"
    if w in ("rank", "pct") or (t.startswith("fb") and w == "pct"):
        src = G2.get(i, {}) if t.startswith("fb") else G[i]["gen"]
        s = src.get(key)
        if s is None: return None
        v = parse_rank(s) if w == "rank" else parse_pct(s)
        return None if v is None else np.array(v, float)
    src = H2.get(i, {}) if t.startswith("fb") else H1[i]["rep"]
    v = src.get(key)
    if v is None: return None
    v = np.array(v, float)
    return -v if w == "least" else v

def truth_vec(i, setup, kind="norm", layers="band"):
    r = H1[i]["truth_think"]["all"] if setup[0] == "think" else H1[i]["truth"]
    a = np.array(r[kind])
    return a[BAND].mean(0) if layers == "band" else a.mean(0)

def spearman(a, b):
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    # average ranks for ties in a
    from scipy.stats import spearmanr
    if np.allclose(a, a[0]) or np.allclose(b, b[0]): return 0.0
    return float(spearmanr(a, b).correlation)

STOP = set("the a an of and or to in on for with by at from is was were are be as that this which who it its into their his her he she they them than then also had has have not but".split())
def words(s): return set(w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2)
def feats(i, setup):
    c = C[i]; g = G[i]
    t = setup[0]
    ans = "" if t == "before" else (g["think_ans_text"] if t == "think" else g["ans_text"])
    rea = g["think_text"] if t == "think" else ""
    q = words(c["task"]); a = words(ans); r = words(rea)
    F = []
    for k, p in enumerate(c["paras"]):
        pw = words(p)
        f = [1.0 if k == m else 0.0 for m in range(4)]
        f += [math.log(H1[i]["ntok"][k]), len(q & pw) / (len(q) + 1), len(re.findall(r"\b\d+\b", p)),
              1.0 if f"[P{k+1}]" in c["task"] else 0.0]
        if t != "before": f += [len(a & pw) / (len(a) + 1)]
        if t == "think": f += [len(r & pw) / (len(r) + 1)]
        F.append(f)
    return np.array(F)

def fit_clogit(X, y, l2=1e-2, it=400):
    w = np.zeros(X.shape[2])
    for _ in range(it):
        s = X @ w; s -= s.max(1, keepdims=True); p = np.exp(s); p /= p.sum(1, keepdims=True)
        g = -(X[np.arange(len(y)), y] - (p[..., None] * X).sum(1)).mean(0) + l2 * w
        w -= 0.5 * g
    return w
def fit_ridge(X, Y, l2=1.0):
    Xf = X.reshape(-1, X.shape[2]); Xf = np.c_[Xf, np.ones(len(Xf))]; Yf = Y.reshape(-1)
    return np.linalg.solve(Xf.T @ Xf + l2 * np.eye(Xf.shape[1]), Xf.T @ Yf)

def classifier(ids, setup, T, least):
    """5-fold CV within each family. Returns top-1 predictions and ranking scores per context."""
    pred, sc = {}, {}
    crng = np.random.default_rng(1)
    for fam in FAMS:
        fi = [i for i in ids if C[i]["family"] == fam]
        if len(fi) < 5: continue
        X = np.stack([feats(i, setup) for i in fi])
        y = np.array([int(np.argmin(T[i]) if least else np.argmax(T[i])) for i in fi])
        Y = np.stack([np.log(T[i] / T[i].sum()) for i in fi])
        perm = crng.permutation(len(fi))
        for f in range(5):
            te = perm[f::5]; tr = np.setdiff1d(perm, te)
            mu = X[tr].reshape(-1, X.shape[2]).mean(0); sd = X[tr].reshape(-1, X.shape[2]).std(0) + 1e-6
            Xs = (X - mu) / sd
            w = fit_clogit(Xs[tr], y[tr]); b = fit_ridge(Xs[tr], Y[tr])
            for j in te:
                pred[fi[j]] = int(np.argmax(Xs[j] @ w))
                s = np.c_[Xs[j], np.ones(4)] @ b
                sc[fi[j]] = -s if least else s
    return pred, sc

def boot_ci(x, alphas=(0.05, 0.0025)):
    x = np.asarray(x, float); n = len(x)
    m = x[rng.integers(0, n, (NB, n))].mean(1)
    out = {"mean": float(x.mean())}
    for a in alphas: out[a] = (float(np.percentile(m, 100 * a / 2)), float(np.percentile(m, 100 * (1 - a / 2))))
    return out

def evaluate(setup, kind="norm", layers="band", fam=None):
    t, w = setup
    least = w == "least"
    ids = [i for i in IDS if (not t.startswith("fb") or i in H2) and (fam is None or C[i]["family"] == fam)]
    T = {i: truth_vec(i, setup, kind, layers) for i in ids}
    cp, cs = classifier([i for i in IDS if (not t.startswith("fb") or i in H2)], setup, {i: truth_vec(i, setup, kind, layers) for i in IDS}, least)
    rows = {"self": ([], []), "reader": ([], []), "clf": ([], [])}
    for i in ids:
        tgt = int(np.argmin(T[i]) if least else np.argmax(T[i]))
        tr_rank = -T[i] if least else T[i]
        for who in ["self", "reader"]:
            s = score(i, setup, who)
            if s is None: rows[who][0].append(0.0); rows[who][1].append(0.0); continue
            top = int(np.argmin(s) if least else np.argmax(s))  # least: s = -p, so argmin(s) = argmax p
            rows[who][0].append(float(top == tgt)); rows[who][1].append(spearman(s, T[i]))
        rows["clf"][0].append(float(cp[i] == tgt)); rows["clf"][1].append(spearman(cs[i], T[i]))
    res = {"n": len(ids)}
    for who in rows:
        res[who] = dict(top1=boot_ci(rows[who][0]), rho=boot_ci(rows[who][1]))
    for b in ["reader", "clf"]:
        res[f"self-{b}"] = dict(top1=boot_ci(np.array(rows["self"][0]) - rows[b][0]), rho=boot_ci(np.array(rows["self"][1]) - rows[b][1]))
    return res

def P(x, pct=True):
    m = x["mean"]; lo, hi = x[0.05]
    return f"{100*m:.1f}% [{100*lo:.1f}, {100*hi:.1f}]" if pct else f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
def Pd(x, pct=True):
    m = x["mean"]; lo, hi = x[0.05]; l2, h2 = x[0.0025]
    if pct: return f"{100*m:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] ({100*l2:+.1f}, {100*h2:+.1f})"
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({l2:+.3f}, {h2:+.3f})"

ALL = {}
md = ["# Attention monitoring, Qwen3-32B: tables\n", f"Contexts: {len(IDS)}. Truth (primary): length-normalised attention, answer queries, mean over heads and layers 21–53. "
      "Top-1: accuracy at naming the most-attended paragraph (least-attended for 'which least'). Rank: Spearman correlation of the four scores with the four truth values, mean over contexts. "
      "Brackets: 95% bootstrap CI; parentheses: Bonferroni 99.75% CI (20 setups) used by the pass rule.\n"]
md.append("## Primary truth\n")
md.append("| setup | wording | n | self top-1 | reader top-1 | classifier top-1 | self − reader top-1 | self − classifier top-1 | self rank | reader rank | classifier rank | self − reader rank | self − classifier rank | pass |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
npass = 0
for su in SETUPS:
    r = evaluate(su); ALL["|".join(su)] = r
    ok = all(r[f"self-{b}"][m][0.0025][0] > 0 for b in ["reader", "clf"] for m in ["top1", "rho"])
    npass += ok
    md.append(f"| {NAMES[su[0]]} | {WN[su[1]]} | {r['n']} | {P(r['self']['top1'])} | {P(r['reader']['top1'])} | {P(r['clf']['top1'])} | {Pd(r['self-reader']['top1'])} | {Pd(r['self-clf']['top1'])} | "
              f"{P(r['self']['rho'], False)} | {P(r['reader']['rho'], False)} | {P(r['clf']['rho'], False)} | {Pd(r['self-reader']['rho'], False)} | {Pd(r['self-clf']['rho'], False)} | {'PASS' if ok else 'no'} |")
md.append(f"\nSetups passing: {npass} of {len(SETUPS)}. Verdict by the preregistered rule: {'PASS' if npass else 'FAIL'}.\n")

for kind, layers, title in [("mass", "band", "Secondary truth: total attention mass, layers 21–53"), ("norm", "all", "Secondary truth: length-normalised, all 64 layers"),
                            ("mass", "all", "Secondary truth: total mass, all 64 layers")]:
    md.append(f"## {title}\n")
    md.append("| setup | wording | self top-1 | reader top-1 | classifier top-1 | self − reader top-1 | self − classifier top-1 | self rank | self − reader rank | self − classifier rank |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for su in SETUPS:
        r = evaluate(su, kind, layers); ALL[f"{kind}-{layers}|" + "|".join(su)] = r
        md.append(f"| {NAMES[su[0]]} | {WN[su[1]]} | {P(r['self']['top1'])} | {P(r['reader']['top1'])} | {P(r['clf']['top1'])} | {Pd(r['self-reader']['top1'])} | {Pd(r['self-clf']['top1'])} | "
                  f"{P(r['self']['rho'], False)} | {Pd(r['self-reader']['rho'], False)} | {Pd(r['self-clf']['rho'], False)} |")
    md.append("")

md.append("## Per family (primary truth): self − reader, top-1 and rank, 95% CI\n")
md.append("| setup | wording | " + " | ".join(FAMS) + " |")
md.append("|---|---|" + "---|" * len(FAMS))
for su in SETUPS:
    cells = []
    for fam in FAMS:
        r = evaluate(su, fam=fam); ALL[f"fam-{fam}|" + "|".join(su)] = r
        a = r["self-reader"]["top1"]; b = r["self-reader"]["rho"]
        cells.append(f"{100*a['mean']:+.0f} [{100*a[0.05][0]:+.0f}, {100*a[0.05][1]:+.0f}] / {b['mean']:+.2f} [{b[0.05][0]:+.2f}, {b[0.05][1]:+.2f}]")
    md.append(f"| {NAMES[su[0]]} | {WN[su[1]]} | " + " | ".join(cells) + " |")

# how lopsided is the truth, and how often it is the task paragraph
md.append("\n## Truth: how lopsided, by family (primary truth)\n")
md.append("| family | n | top paragraph's share of the four (mean) | top / second (median) | top is a task-named or answer paragraph | same top paragraph, thinking on vs off |")
md.append("|---|---|---|---|---|---|")
for fam in FAMS:
    fi = [i for i in IDS if C[i]["family"] == fam]
    if not fi: continue
    sh, ratio, tg, same = [], [], [], []
    for i in fi:
        v = truth_vec(i, ("after", "most")); s = np.sort(v)[::-1]
        sh.append(s[0] / v.sum()); ratio.append(s[0] / s[1])
        if C[i]["target"]: tg.append(float(np.argmax(v) in C[i]["target"]))
        same.append(float(np.argmax(v) == np.argmax(truth_vec(i, ("think", "most")))))
    md.append(f"| {fam} | {len(fi)} | {100*np.mean(sh):.0f}% | {np.median(ratio):.2f} | {('%.0f%%' % (100*np.mean(tg))) if tg else '–'} | {100*np.mean(same):.0f}% |")

# parse failures
md.append("\n## Parse failures (generated wordings)\n")
for su in SETUPS:
    if su[1] in ("rank", "pct"):
        ids = [i for i in IDS if (not su[0].startswith("fb") or i in H2)]
        f = {who: sum(score(i, su, who) is None for i in ids) for who in ["self", "reader"]}
        md.append(f"- {NAMES[su[0]]}, {WN[su[1]]}: self {f['self']}/{len(ids)}, reader {f['reader']}/{len(ids)}")

open(os.path.join(OUTD, "tables.md"), "w").write("\n".join(md) + "\n")
json.dump(ALL, open(os.path.join(OUTD, "results.json"), "w"))
print("\n".join(md[:30]))
print("PASSING SETUPS", npass)
