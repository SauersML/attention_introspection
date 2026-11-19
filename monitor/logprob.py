# Does the model's own report of where its attention went track that attention, and does the report carry anything a
# reader of the same transcript (the same model, Qwen3-32B, given the conversation as quoted text) does not?
# Writes OUTJSON (results/monitor_{inorder,shuffled}/logprob_results.json) and prints the tables in results/monitor_shuffled/LOGPROB.md.
# usage: logprob.py OUTJSON SHARD_DIR [SHARD_DIR ...]   (shards: hypha-prod-mon{0,1,2}/out with hf1.json, hf2.json, contexts.json)
import json, os, sys
import numpy as np
from scipy.stats import rankdata

OUT = sys.argv[1]; DIRS = sys.argv[2:]
BAND = list(range(21, 54)); NB = 10000; NBG = 2000; NPERM = 200; NFOLD = 10; NSPLIT = 10
SETUPS = ["after|most", "after|least", "after|rate", "before|most", "before|rate", "later|most", "later|rate", "think|most", "after|free",
          "fb4|most", "fb16|most", "fb4|rate", "fb16|rate"]  # fb: asked right after, once 4 or 16 true attention measurements of other prompts were shown
H, C = {}, {}
for d in DIRS:
    H.update(json.load(open(os.path.join(d, "hf1.json"))))
    for c in json.load(open(os.path.join(d, "contexts.json"))): C[str(c["id"])] = c
    for i, r in json.load(open(os.path.join(d, "hf2.json"))).items(): H[i]["rep"].update(r)  # feedback reads
IDS = sorted(H, key=int)
print("prompts", len(IDS))

# truth: "norm" = attention per token of the paragraph (mass / token count), the preregistered primary truth
# (PREREG.md); "mass" = total attention mass, secondary. Both: answer queries, mean over heads, layers 21-53.
# The feedback examples (fb4, fb16) showed the model its length-normalised ("norm") attention.
TRUTHS = ["norm", "mass"]
def truth(i, s, tr):
    r = H[i]["truth_think"]["final"] if s.startswith("think") else H[i]["truth"]
    return np.array(r[tr], float)[BAND].mean(0)
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
    """R: prompts x 4 x k reader features (k reader wordings), S: prompts x 4 added feature"""
    if R.ndim == 2: R = R[..., None]
    base = np.concatenate([R, np.broadcast_to(POS[:, 1:], (len(Y), 4, 3)), LN[..., None]], 2)
    full = np.concatenate([base, S[..., None]], 2)
    rb, rf = np.zeros(len(Y)), np.zeros(len(Y))
    for FOLD in SPLITS:
        Pb, Pf = fit_cv(base, Y, FOLD), fit_cv(full, Y, FOLD)
        rb += np.array([spear(Pb[n], Y[n]) for n in range(len(Y))]) / NSPLIT; rf += np.array([spear(Pf[n], Y[n]) for n in range(len(Y))]) / NSPLIT
    return rf - rb, rb, rf

WORDS = {t: [w for w in ["most", "rate", "least", "free"] if f"{t}|{w}" in SETUPS] for t in ["after", "before", "later", "think", "fb4", "fb16"]}
LN = np.log(np.stack([H[i]["ntok"] for i in IDS]).astype(float)); LN = LN - LN.mean(1, keepdims=True)  # log length, within prompt
RD = {s: np.stack([zr(score(i, s, "reader")) for i in IDS]) for s in SETUPS}
SF = {s: np.stack([zr(score(i, s, "self")) for i in IDS]) for s in SETUPS}
RES, PER = {tr: {} for tr in TRUTHS}, {tr: {} for tr in TRUTHS}
for tr in TRUTHS:
    for s in SETUPS:
        t = s.split("|")[0]; others = [f"{t}|{v}" for v in WORDS[t] if f"{t}|{v}" != s]
        T = np.stack([truth(i, s, tr) for i in IDS])
        Sr = np.stack([score(i, s, "self") for i in IDS]); Rr = np.stack([score(i, s, "reader") for i in IDS])
        rs = np.array([spear(Sr[n], T[n]) for n in range(len(IDS))]); rr = np.array([spear(Rr[n], T[n]) for n in range(len(IDS))])
        Y = np.stack([zr(x) for x in T]); S, R = SF[s], RD[s]
        # (a) the control: base = the reader asked EVERY wording at this timing (+ position + length); does the self-report add anything?
        Rall = np.stack([R] + [RD[o] for o in others], 2)
        g_all, rb_all, rf_all = gain_per_prompt(Rall, S, LN, Y)
        # (b) alternative null: what a second reader wording adds over the single same-wording reader, beside what the self-report adds
        g1, rb1, rf1 = gain_per_prompt(R, S, LN, Y)
        g2 = {o: gain_per_prompt(R, RD[o], LN, Y)[0] for o in others}
        # (c) marginal control only (not a test of private access): self ranks shuffled across prompts within each slot
        prng = np.random.default_rng(4); nul = []
        for _ in range(NPERM):
            Sp = S.copy()
            for k in range(4): Sp[:, k] = S[prng.permutation(len(IDS)), k]
            nul.append(float(gain_per_prompt(R, Sp, LN, Y)[0].mean()))
        nul = np.array(nul); b = int(np.sum(nul >= g1.mean()))
        RES[tr][s] = dict(n=len(IDS), self=boot(rs), reader=boot(rr), self_minus_reader=boot(rs - rr),
                          base_all=float(rb_all.mean()), full_all=float(rf_all.mean()), gain_all=boot(g_all, NBG), base_wordings=[s] + others,
                          base=float(rb1.mean()), full=float(rf1.mean()), gain=boot(g1, NBG),
                          second_reader={o: boot(v, NBG) for o, v in g2.items()},
                          self_minus_second_reader={o: boot(g1 - v, NBG) for o, v in g2.items()},
                          shuffle_exceed=b, shuffle_n=NPERM, shuffle_p=(b + 1) / (NPERM + 1), shuffle_95=float(np.percentile(nul, 95)))
        PER[tr][s] = dict(self=rs.tolist(), reader=rr.tolist(), gain=g1.tolist(), gain_all=g_all.tolist())
        q = RES[tr][s]
        print(f"{tr} {s:12s} self {q['self']['mean']:.3f} reader {q['reader']['mean']:.3f} gain_all {q['gain_all']['mean']:+.4f} "
              f"[{q['gain_all']['lo']:+.4f},{q['gain_all']['hi']:+.4f}] gain1 {q['gain']['mean']:+.4f} 2nd-reader "
              + " ".join(f"{o}:{v['mean']:+.4f}" for o, v in q['second_reader'].items()) + f" shuffle {b}/{NPERM}", flush=True)
json.dump(dict(res=RES, perprompt=PER, ids=IDS, note="truth norm = preregistered primary (per-token attention); mass = secondary, post hoc"), open(OUT, "w"))

# markdown tables for results/monitor_shuffled/LOGPROB.md
NAME = {"after|most": "most attended, asked right after", "before|most": "most attended, asked before answering", "later|most": "most attended, asked one turn later",
        "think|most": "most attended, with thinking", "fb4|most": "most attended, after 4 true examples", "fb16|most": "most attended, after 16 true examples",
        "after|rate": "0–10 rating, asked right after", "before|rate": "0–10 rating, asked before answering", "later|rate": "0–10 rating, asked one turn later",
        "fb4|rate": "0–10 rating, after 4 true examples", "fb16|rate": "0–10 rating, after 16 true examples",
        "after|least": "least attended, asked right after", "after|free": "free answer, asked right after"}
SHORT = {"most": "most", "rate": "rating", "least": "least", "free": "free"}
ci = lambda d: f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
cs = lambda d: f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
for tr in TRUTHS:
    R_ = RES[tr]; ORDER = [k for k in NAME if k in R_]
    print(f"\n### Truth: {tr}\n\n| way of asking | self-report | same model reading the transcript | self − reader |\n|---|---|---|---|")
    for k in ORDER: q = R_[k]; print(f"| {NAME[k]} | {ci(q['self'])} | {ci(q['reader'])} | {cs(q['self_minus_reader'])} |")
    print("\n| way of asking | base: reader, every wording at this timing (+ position, length) | + self-report | self gain [95% CI] | base: reader, same wording | self gain over it | gain from a second reader wording instead |\n|---|---|---|---|---|---|---|")
    for k in ORDER:
        q = R_[k]; sec = ", ".join(f"{SHORT[o.split('|')[1]]} {v['mean']:+.3f}" for o, v in q['second_reader'].items()) or "none asked"
        print(f"| {NAME[k]} | {q['base_all']:.3f} | {q['full_all']:.3f} | {cs(q['gain_all'])} | {q['base']:.3f} | {cs(q['gain'])} | {sec} |")
    print("\n| way of asking | shuffled self-reports at least as large (of 200) | p = (b+1)/201 |\n|---|---|---|")
    for k in ORDER: q = R_[k]; print(f"| {NAME[k]} | {q['shuffle_exceed']} | {q['shuffle_p']:.3f} |")
