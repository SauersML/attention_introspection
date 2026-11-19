# Is there signal in the self-report's log-probabilities, whatever its calibration or bias? (results/monitor_shuffled/LOGPROB.md)
# Reads the saved label probabilities of the "which paragraph did you attend to most?" read (hf1.json, hf2.json):
# probabilities of the four label digits after the prefilled "P", renormalised over the four, so log P differs from the
# raw log-probability by one constant per prompt; every measure below is invariant to that constant.
# usage: logprob_signal.py OUTDIR SHARD_DIR [SHARD_DIR ...]
import json, os, re, sys, math
import numpy as np
from scipy.stats import rankdata

OUTD = sys.argv[1]; DIRS = sys.argv[2:]
os.makedirs(OUTD, exist_ok=True)
BAND = list(range(21, 54)); NB = 10000; NPERM = int(os.environ.get("NPERM", 200)); NFOLD = 10; LPMIN = float(os.environ.get("LPMIN", -30.0))
TRUTH = os.environ.get("TRUTH", "norm")  # norm (length-normalised, the test's primary) or mass (total attention mass)
OUTCOME = os.environ.get("OUTCOME", "share")  # share, or log (log of the share) as a sensitivity check
rng = np.random.default_rng(0)

C, G, H1, H2 = {}, {}, {}, {}
for d in DIRS:
    for c in json.load(open(os.path.join(d, "contexts.json"))): C[str(c["id"])] = c
    G.update(json.load(open(os.path.join(d, "gen1.json"))))
    H1.update(json.load(open(os.path.join(d, "hf1.json"))))
    if os.path.exists(os.path.join(d, "hf2.json")): H2.update(json.load(open(os.path.join(d, "hf2.json"))))
IDS = sorted(set(C) & set(H1), key=int)
FAMS = ["distractor", "multihop", "arith", "sumone", "longshort"]
SETUPS = ["after", "before", "later", "think", "fb4", "fb16"]
NAMES = {"after": "right after answering", "before": "before answering", "later": "one turn later", "think": "thinking on",
         "fb4": "after 4 feedback examples", "fb16": "after 16 feedback examples", "after-least": "right after, 'which least?' (minus log P)"}
print("contexts", len(IDS), "with feedback reads", len(set(IDS) & set(H2)))

def lp(v): return np.maximum(np.log(np.maximum(np.array(v, float), 1e-300)), LPMIN)
def share(i, t, kind=None):
    kind = kind or TRUTH
    r = H1[i]["truth_think"]["all"] if t == "think" else H1[i]["truth"]
    v = np.array(r[kind])[BAND].mean(0); return v / v.sum()
def get(i, t, who):
    if t.startswith("fb"):
        v = H2.get(i, {}).get(f"{t}|most|{who}")
        return None if v is None else lp(v)
    if t == "after-least": return -lp(H1[i]["rep"][f"after|least|{who}"])
    return lp(H1[i]["rep"][f"{t}|most|{who}"])
def spear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12: return 0.0
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])
def boot(x):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (NB, len(x)))].mean(1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)))

# transcript classifier scores (same features and 5-fold CV within family as analyze_monitor.py; ridge on log share)
STOP = set("the a an of and or to in on for with by at from is was were are be as that this which who it its into their his her he she they them than then also had has have not but".split())
def words(s): return set(w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2)
def feats(i, t):
    c = C[i]; g = G[i]
    ans = "" if t == "before" else (g["think_ans_text"] if t == "think" else g["ans_text"])
    rea = g["think_text"] if t == "think" else ""
    q = words(c["task"]); a = words(ans); r = words(rea); F = []
    for k, p in enumerate(c["paras"]):
        pw = words(p)
        f = [1.0 if k == m else 0.0 for m in range(4)]
        f += [math.log(H1[i]["ntok"][k]), len(q & pw) / (len(q) + 1), len(re.findall(r"\b\d+\b", p)), 1.0 if f"[P{k+1}]" in c["task"] else 0.0]
        if t != "before": f += [len(a & pw) / (len(a) + 1)]
        if t == "think": f += [len(r & pw) / (len(r) + 1)]
        F.append(f)
    return np.array(F)
CLF = {}
def classifier(t):
    """scores for every prompt; the feedback and 'least' rows use the 'after' transcript, so they share its scores"""
    tt = "after" if t in ("after-least",) or t.startswith("fb") else t
    if tt in CLF: return CLF[tt]
    sc = {}; crng = np.random.default_rng(1); ids = IDS
    for fam in FAMS:
        fi = [i for i in ids if C[i]["family"] == fam]
        X = np.stack([feats(i, tt) for i in fi]); Y = np.stack([np.log(share(i, tt)) for i in fi])
        perm = crng.permutation(len(fi))
        for f in range(5):
            te = perm[f::5]; tr = np.setdiff1d(perm, te)
            mu = X[tr].reshape(-1, X.shape[2]).mean(0); sd = X[tr].reshape(-1, X.shape[2]).std(0) + 1e-6
            Xs = (X - mu) / sd; Xf = np.c_[Xs[tr].reshape(-1, X.shape[2]), np.ones(len(tr) * 4)]
            b = np.linalg.solve(Xf.T @ Xf + np.eye(Xf.shape[1]), Xf.T @ Y[tr].reshape(-1))
            for j in te: sc[fi[j]] = np.c_[Xs[j], np.ones(4)] @ b
    CLF[tt] = sc; return sc

# grouped-CV incremental test. Rows: prompt x paragraph. Prompt fixed effects = within-prompt demeaning of outcome and features.
FOLD = {i: n % NFOLD for n, i in enumerate(np.random.default_rng(2).permutation(IDS))}
def design(ids, t, selfv, text=False):
    Xb, Xs, Y = [], [], []
    for n, i in enumerate(ids):
        tt = "after" if t == "after-least" or t.startswith("fb") else t
        pos = np.eye(4)[:, 1:]; ln = np.log(np.array(H1[i]["ntok"], float))[:, None]
        cols = [get(i, t, "reader")[:, None], pos, ln]
        if text:  # word-overlap features of the transcript classifier, per family
            F = feats(i, tt)[:, 5:]; fam = np.eye(len(FAMS))[FAMS.index(C[i]["family"])]
            cols.append(np.kron(F, fam[None, :]))
        xb = np.hstack(cols); Xb.append(xb - xb.mean(0)); Xs.append(selfv[n] - selfv[n].mean())
        y = share(i, tt); y = np.log(y) if OUTCOME == "log" else y; Y.append(y - y.mean())
    return np.stack(Xb), np.stack(Xs)[..., None], np.stack(Y)
def cv_pred(X, Y, folds):
    P = np.zeros_like(Y)
    for f in range(NFOLD):
        tr, te = folds != f, folds == f
        A = X[tr].reshape(-1, X.shape[2]); sd = A.std(0) + 1e-9
        w = np.linalg.solve((A / sd).T @ (A / sd) + 1e-3 * np.eye(A.shape[1]), (A / sd).T @ Y[tr].reshape(-1)) / sd
        P[te] = X[te] @ w
    return P
def incr(ids, t, selfv, text=False):
    Xb, Xs, Y = design(ids, t, selfv, text); folds = np.array([FOLD[i] for i in ids])
    Pb = cv_pred(Xb, Y, folds); Pf = cv_pred(np.concatenate([Xb, Xs], 2), Y, folds)
    rb = np.array([spear(Pb[n], Y[n]) for n in range(len(ids))]); rf = np.array([spear(Pf[n], Y[n]) for n in range(len(ids))])
    return dict(rb=rb, rf=rf, sseb=((Y - Pb) ** 2).sum(1), ssef=((Y - Pf) ** 2).sum(1), sst=(Y ** 2).sum(1))
def r2(s, idx): return 1 - s[0][idx].sum() / s[1][idx].sum()
def boot_r2gain(sseb, ssef, sst):
    n = len(sst); idx = rng.integers(0, n, (NB, n))
    g = (sseb[idx].sum(1) - ssef[idx].sum(1)) / sst[idx].sum(1)
    return dict(mean=float((sseb.sum() - ssef.sum()) / sst.sum()), lo=float(np.percentile(g, 2.5)), hi=float(np.percentile(g, 97.5)))
def perm_self(ids, t, prng):
    """permute self log-prob values across prompts, separately for each paragraph slot, within each family"""
    S = np.stack([get(i, t, "self") for i in ids]); out = S.copy()
    fam = np.array([C[i]["family"] for i in ids])
    for f in FAMS:
        m = np.where(fam == f)[0]
        for k in range(4): out[m, k] = S[prng.permutation(m), k]
    return out

RES = {}; PERPROMPT = {}
ROWS = SETUPS + ["after-least"]
for t in ROWS:
    ids = [i for i in IDS if get(i, t, "self") is not None and get(i, t, "reader") is not None]
    if len(ids) < 30: print("skip", t, len(ids)); continue
    tt = "after" if t == "after-least" or t.startswith("fb") else t
    Y = {i: share(i, tt) for i in ids}
    rs = np.array([spear(get(i, t, "self"), Y[i]) for i in ids]); rr = np.array([spear(get(i, t, "reader"), Y[i]) for i in ids])
    cl = classifier(t)
    rsr = np.array([spear(get(i, t, "self"), get(i, t, "reader")) for i in ids])
    rc = np.array([spear(cl[i], Y[i]) for i in ids])
    # prior-only null for (1): self vectors shuffled across prompts within family
    prng = np.random.default_rng(3)
    S = np.stack([get(i, t, "self") for i in ids]); fam = np.array([C[i]["family"] for i in ids])
    nul1 = []
    for _ in range(NPERM):
        Sp = S.copy()
        for f in FAMS:
            m = np.where(fam == f)[0]; Sp[m] = S[prng.permutation(m)]
        nul1.append(np.mean([spear(Sp[n], Y[i]) for n, i in enumerate(ids)]))
    selfv = np.stack([get(i, t, "self") for i in ids])
    inc = incr(ids, t, selfv); inc_text = incr(ids, t, selfv, text=True)
    nul_rho, nul_r2 = [], []
    prng = np.random.default_rng(4)
    for _ in range(NPERM):
        q = incr(ids, t, perm_self(ids, t, prng))
        nul_rho.append(float((q["rf"] - q["rb"]).mean())); nul_r2.append(float((q["sseb"].sum() - q["ssef"].sum()) / q["sst"].sum()))
    g_rho = inc["rf"] - inc["rb"]; g_r2 = (inc["sseb"].sum() - inc["ssef"].sum()) / inc["sst"].sum()
    RES[t] = dict(n=len(ids), self=boot(rs), reader=boot(rr), clf=boot(rc), self_minus_reader=boot(rs - rr), self_vs_reader=boot(rsr),
                  null_self_prior=dict(mean=float(np.mean(nul1)), hi=float(np.percentile(nul1, 97.5))),
                  base_rho=boot(inc["rb"]), full_rho=boot(inc["rf"]), gain_rho=boot(g_rho),
                  base_r2=float(r2((inc["sseb"], inc["sst"]), slice(None))), full_r2=float(r2((inc["ssef"], inc["sst"]), slice(None))),
                  gain_r2=boot_r2gain(inc["sseb"], inc["ssef"], inc["sst"]),
                  null_gain_rho=dict(mean=float(np.mean(nul_rho)), lo=float(np.percentile(nul_rho, 2.5)), hi=float(np.percentile(nul_rho, 97.5)),
                                     p=float((1 + np.sum(np.array(nul_rho) >= g_rho.mean())) / (1 + NPERM))),
                  null_gain_r2=dict(mean=float(np.mean(nul_r2)), lo=float(np.percentile(nul_r2, 2.5)), hi=float(np.percentile(nul_r2, 97.5)),
                                    p=float((1 + np.sum(np.array(nul_r2) >= g_r2)) / (1 + NPERM))),
                  gain_rho_text=boot(inc_text["rf"] - inc_text["rb"]), gain_r2_text=boot_r2gain(inc_text["sseb"], inc_text["ssef"], inc_text["sst"]),
                  null_rho_list=nul_rho, null_r2_list=nul_r2,
                  fam={f: dict(n=int(sum(C[i]["family"] == f for i in ids)),
                               self=float(np.mean([rs[n] for n, i in enumerate(ids) if C[i]["family"] == f])),
                               reader=float(np.mean([rr[n] for n, i in enumerate(ids) if C[i]["family"] == f])),
                               gain_rho=float(np.mean([g_rho[n] for n, i in enumerate(ids) if C[i]["family"] == f]))) for f in FAMS})
    PERPROMPT[t] = dict(ids=ids, self=rs.tolist(), reader=rr.tolist(), clf=rc.tolist(), gain_rho=g_rho.tolist(),
                        sseb=inc["sseb"].tolist(), ssef=inc["ssef"].tolist(), sst=inc["sst"].tolist())
    print(t, len(ids), "self %.3f reader %.3f clf %.3f gain_rho %.3f [%.3f,%.3f] null %.3f..%.3f p=%.3f gain_r2 %.4f [%.4f,%.4f] null hi %.4f" % (
        rs.mean(), rr.mean(), rc.mean(), g_rho.mean(), RES[t]["gain_rho"]["lo"], RES[t]["gain_rho"]["hi"],
        RES[t]["null_gain_rho"]["lo"], RES[t]["null_gain_rho"]["hi"], RES[t]["null_gain_rho"]["p"], g_r2, RES[t]["gain_r2"]["lo"], RES[t]["gain_r2"]["hi"], RES[t]["null_gain_r2"]["hi"]), flush=True)

# pooled over setups: per prompt, mean over the setups that prompt has (main six setups, "which most?")
main = [t for t in SETUPS if t in PERPROMPT]
pp = {}
for t in main:
    for n, i in enumerate(PERPROMPT[t]["ids"]):
        d = pp.setdefault(i, dict(self=[], reader=[], gain=[], sseb=0.0, ssef=0.0, sst=0.0))
        d["self"].append(PERPROMPT[t]["self"][n]); d["reader"].append(PERPROMPT[t]["reader"][n]); d["gain"].append(PERPROMPT[t]["gain_rho"][n])
        d["sseb"] += PERPROMPT[t]["sseb"][n]; d["ssef"] += PERPROMPT[t]["ssef"][n]; d["sst"] += PERPROMPT[t]["sst"][n]
ks = sorted(pp, key=int)
ps = np.array([np.mean(pp[i]["self"]) for i in ks]); pr = np.array([np.mean(pp[i]["reader"]) for i in ks]); pg = np.array([np.mean(pp[i]["gain"]) for i in ks])
sb = np.array([pp[i]["sseb"] for i in ks]); sf = np.array([pp[i]["ssef"] for i in ks]); st = np.array([pp[i]["sst"] for i in ks])
# pooled null: the same permutation index j in every setup, averaged over setups weighted by prompts
nrho = np.array([sum(RES[t]["null_rho_list"][j] * RES[t]["n"] for t in main) / sum(RES[t]["n"] for t in main) for j in range(NPERM)])
nr2 = np.array([np.mean([RES[t]["null_r2_list"][j] for t in main]) for j in range(NPERM)])
g_all = sum(np.sum(PERPROMPT[t]["gain_rho"]) for t in main) / sum(RES[t]["n"] for t in main)
RES["pooled"] = dict(setups=main, n=len(ks), self=boot(ps), reader=boot(pr), self_minus_reader=boot(ps - pr), gain_rho=boot(pg),
                     gain_r2=boot_r2gain(sb, sf, st),
                     null_gain_rho=dict(mean=float(nrho.mean()), lo=float(np.percentile(nrho, 2.5)), hi=float(np.percentile(nrho, 97.5)),
                                        p=float((1 + np.sum(nrho >= g_all)) / (1 + NPERM))),
                     null_gain_r2=dict(mean=float(nr2.mean()), lo=float(np.percentile(nr2, 2.5)), hi=float(np.percentile(nr2, 97.5))))
RES["pooled"]["settings"] = dict(TRUTH=TRUTH, OUTCOME=OUTCOME, LPMIN=LPMIN, NPERM=NPERM, NFOLD=NFOLD)
print("pooled", RES["pooled"]["self"], RES["pooled"]["reader"], RES["pooled"]["gain_rho"], RES["pooled"]["null_gain_rho"], RES["pooled"]["gain_r2"])
json.dump(dict(res={k: {kk: vv for kk, vv in v.items() if not kk.endswith("_list")} for k, v in RES.items()},
               null={t: dict(rho=RES[t]["null_rho_list"], r2=RES[t]["null_r2_list"]) for t in ROWS if t in RES},
               perprompt=PERPROMPT, names=NAMES), open(os.path.join(OUTD, "logprob.json"), "w"))
