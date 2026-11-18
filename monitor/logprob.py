# Does the model's own report of where its attention went track that attention, and does the report carry anything a
# reader of the same transcript (the same model, Qwen3-32B, given the conversation as quoted text) does not?
# Writes results/monitor/logprob_results.json and prints the tables in results/monitor/LOGPROB.md.
# usage: logprob.py OUTJSON SHARD_DIR [SHARD_DIR ...]   (shards: hypha-prod-mon{0,1,2}/out with hf1.json, hf2.json, contexts.json)
import json, os, sys
import numpy as np
from scipy.stats import rankdata

OUT = sys.argv[1]; DIRS = sys.argv[2:]
BAND = list(range(21, 54)); NB = 10000; NBG = 200; NPERM = 200; NFOLD = 10; NSPLIT = 10
SETUPS = ["after|most", "after|least", "after|rate", "before|most", "before|rate", "later|most", "later|rate", "think|most", "after|free",
          "fb4|most", "fb16|most", "fb4|rate", "fb16|rate"]  # fb: asked right after, once 4 or 16 true attention measurements of other prompts were shown
H, C = {}, {}
for d in DIRS:
    H.update(json.load(open(os.path.join(d, "hf1.json"))))
    for c in json.load(open(os.path.join(d, "contexts.json"))): C[str(c["id"])] = c
    for i, r in json.load(open(os.path.join(d, "hf2.json"))).items(): H[i]["rep"].update(r)  # feedback reads
IDS = sorted(H, key=int)
print("prompts", len(IDS))

def truth(i, s):
    r = H[i]["truth_think"]["final"] if s.startswith("think") else H[i]["truth"]
    return np.array(r["mass"], float)[BAND].mean(0)
def score(i, s, who):
    v = np.array(H[i]["rep"][f"{s}|{who}"], float)
    w = s.split("|")[1]
    if w == "rate": return v
    lp = np.log(np.maximum(v, 1e-300))
    return -lp if w == "least" else lp
def spear(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if np.ptp(ra) == 0 or np.ptp(rb) == 0: return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])
def zr(v):
    r = rankdata(v); sd = r.std()
    return (r - r.mean()) / sd if sd > 0 else r * 0.0
rng = np.random.default_rng(0)
def boot(x, nb=NB):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (nb, len(x)))].mean(1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)))

# 10 folds of whole prompts; the 10-fold split is redrawn NSPLIT times and per-prompt results averaged over splits,
# because a single split moves the gain by about +-0.01 (the size of the effect)
SPLITS = [np.array([n % NFOLD for n in np.random.default_rng(sd).permutation(len(IDS))]) for sd in range(NSPLIT)]
POS = np.eye(4)
def fit_cv(X, Y, FOLD):
    """X: prompts x 4 x features, Y: prompts x 4; OLS with intercept, 10 folds grouped by prompt; returns predictions"""
    P = np.zeros_like(Y)
    for f in range(NFOLD):
        tr = FOLD != f
        A = np.c_[X[tr].reshape(-1, X.shape[2]), np.ones(tr.sum() * 4)]
        w = np.linalg.lstsq(A, Y[tr].reshape(-1), rcond=None)[0]
        P[~tr] = np.c_[X[~tr].reshape(-1, X.shape[2]), np.ones((~tr).sum() * 4)].reshape((~tr).sum(), 4, -1) @ w
    return P
def gain_per_prompt(R, S, LN, Y):
    base = np.concatenate([R[..., None], np.broadcast_to(POS[:, 1:], (len(Y), 4, 3)), LN[..., None]], 2)
    full = np.concatenate([base, S[..., None]], 2)
    rb, rf = np.zeros(len(Y)), np.zeros(len(Y))
    for FOLD in SPLITS:
        Pb, Pf = fit_cv(base, Y, FOLD), fit_cv(full, Y, FOLD)
        rb += np.array([spear(Pb[n], Y[n]) for n in range(len(Y))]) / NSPLIT; rf += np.array([spear(Pf[n], Y[n]) for n in range(len(Y))]) / NSPLIT
    return rf - rb, rb, rf

RES, PER = {}, {}
for s in SETUPS:
    T = np.stack([truth(i, s) for i in IDS])
    Sr = np.stack([score(i, s, "self") for i in IDS]); Rr = np.stack([score(i, s, "reader") for i in IDS])
    rs = np.array([spear(Sr[n], T[n]) for n in range(len(IDS))]); rr = np.array([spear(Rr[n], T[n]) for n in range(len(IDS))])
    Y = np.stack([zr(t) for t in T]); S = np.stack([zr(v) for v in Sr]); R = np.stack([zr(v) for v in Rr])
    LN = np.log(np.stack([H[i]["ntok"] for i in IDS]).astype(float)); LN = LN - LN.mean(1, keepdims=True)  # log length, within prompt
    g, rb, rf = gain_per_prompt(R, S, LN, Y)
    gb = boot(g, NBG)
    prng = np.random.default_rng(4); nul = []
    for _ in range(NPERM):
        Sp = S.copy()
        for k in range(4): Sp[:, k] = S[prng.permutation(len(IDS)), k]
        nul.append(float(gain_per_prompt(R, Sp, LN, Y)[0].mean()))
    nul = np.array(nul)
    RES[s] = dict(n=len(IDS), self=boot(rs), reader=boot(rr), self_minus_reader=boot(rs - rr), base=float(rb.mean()), full=float(rf.mean()),
                  gain=gb, null_mean=float(nul.mean()), null_95=float(np.percentile(nul, 95)), null_lo=float(np.percentile(nul, 2.5)),
                  null_hi=float(np.percentile(nul, 97.5)), p=float(np.mean(nul >= g.mean())), null_n=NPERM,
                  null=nul.tolist())
    PER[s] = dict(self=rs.tolist(), reader=rr.tolist(), gain=g.tolist())
    q = RES[s]
    print(f"{s:12s} self {q['self']['mean']:.3f} [{q['self']['lo']:.3f},{q['self']['hi']:.3f}] reader {q['reader']['mean']:.3f} "
          f"[{q['reader']['lo']:.3f},{q['reader']['hi']:.3f}] base {q['base']:.3f} gain {gb['mean']:+.3f} [{gb['lo']:+.3f},{gb['hi']:+.3f}] "
          f"null95 {q['null_95']:.3f} p {q['p']:.3f}", flush=True)
json.dump(dict(res=RES, perprompt=PER, ids=IDS), open(OUT, "w"))

# markdown tables for results/monitor/LOGPROB.md
NAME = {"after|most": "most attended, asked right after", "before|most": "most attended, asked before answering", "later|most": "most attended, asked one turn later",
        "think|most": "most attended, with thinking", "fb4|most": "most attended, after 4 true examples", "fb16|most": "most attended, after 16 true examples",
        "after|rate": "0–10 rating, asked right after", "before|rate": "0–10 rating, asked before answering", "later|rate": "0–10 rating, asked one turn later",
        "fb4|rate": "0–10 rating, after 4 true examples", "fb16|rate": "0–10 rating, after 16 true examples",
        "after|least": "least attended, asked right after", "after|free": "free answer, asked right after"}
ORDER = [k for k in NAME if k in RES]
ci = lambda d: f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
cs = lambda d: f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
pp = lambda p: f"< {1 / NPERM:.3f}" if p == 0 else f"{p:.3f}"
print("\n| way of asking | self-report | same model reading the transcript | self − reader |\n|---|---|---|---|")
for k in ORDER: q = RES[k]; print(f"| {NAME[k]} | {ci(q['self'])} | {ci(q['reader'])} | {cs(q['self_minus_reader'])} |")
print("\n| way of asking | base (reader + position + length) | + self-report | gain [95% CI] | shuffle null 95th pct | p |\n|---|---|---|---|---|---|")
for k in ORDER: q = RES[k]; print(f"| {NAME[k]} | {q['base']:.3f} | {q['full']:.3f} | {cs(q['gain'])} | {q['null_95']:+.3f} | {pp(q['p'])} |")
