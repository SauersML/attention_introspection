import json, os, re, sys, math
import numpy as np

D = sys.argv[1]
meta = json.load(open(os.path.join(D, "meta.json")))
LAY = meta["LAYERS"]; NL = meta["NL"]
A = json.load(open(os.path.join(D, "A.json")))
A2 = json.load(open(os.path.join(D, "A2.json")))
B = json.load(open(os.path.join(D, "B.json")))
rng = np.random.default_rng(0)
NB = 10000

def band(att, layers=LAY): return np.array(att)[layers].mean(0)
def boot(x, stat=np.mean):
    x = np.asarray(x, float); n = len(x)
    idx = rng.integers(0, n, (NB, n))
    s = np.array([stat(x[i]) for i in idx]) if stat is not np.mean else x[idx].mean(1)
    return float(stat(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
def fmt(t, pct=True):
    m, lo, hi = t
    return f"{100*m:.1f}% [{100*lo:.1f}, {100*hi:.1f}]" if pct else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"

STOP = set("the a an of and or to in on for with by at from is was were are be as that this which who it its into their his her he she they them than then also had has have not but".split())
def words(s): return set(w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2)
def paras_of(user):
    ps = re.split(r"\[P[1-4]\] ", user)[1:]
    ps[-1] = ps[-1].split("\n\n")[0]
    return [p.strip() for p in ps]
def taskq(user):
    return user.split("\n\n")[-1]

R = {}
md = []

# ===================== A =====================
def feats(r):
    ps = paras_of(r["user"]); q = words(taskq(r["user"])); a = words(r["ans"])
    F = []
    for i, p in enumerate(ps):
        pw = words(p)
        f = [1.0 if i == t else 0.0 for t in range(4)]
        f += [len(q & pw) / (len(q) + 1), len(a & pw) / (len(a) + 1), math.log(r["ntok"][i])]
        F.append(f)
    return np.array(F)

def fit_clogit(X, y, l2=1e-2, it=500):
    # conditional logit: X [n,4,f], y [n]
    w = np.zeros(X.shape[2])
    for _ in range(it):
        s = X @ w; s -= s.max(1, keepdims=True); p = np.exp(s); p /= p.sum(1, keepdims=True)
        g = -(X[np.arange(len(y)), y] - (p[..., None] * X).sum(1)).mean(0) + l2 * w
        w -= 0.5 * g
    return w
def cv_pred(X, y, k=5):
    n = len(y); perm = rng.permutation(n); pred = np.zeros(n, int)
    for f in range(k):
        te = perm[f::k]; tr = np.setdiff1d(perm, te)
        mu = X[tr].reshape(-1, X.shape[2]).mean(0); sd = X[tr].reshape(-1, X.shape[2]).std(0) + 1e-6
        w = fit_clogit((X[tr] - mu) / sd, y[tr])
        pred[te] = ((X[te] - mu) / sd @ w).argmax(1)
    return pred

def a_table(recs, key, label):
    truth = np.array([band(r[key]).argmax() for r in recs])
    selfp = np.array([np.argmax(r["rep"]) for r in recs])
    readp = np.array([np.argmax(r["reader"]) for r in recs])
    rows = {}
    rows["self-report"] = (selfp == truth)
    rows["transcript-only reader (same model)"] = (readp == truth)
    X = np.stack([feats(r) for r in recs])
    rows["transcript-feature classifier (5-fold CV)"] = (cv_pred(X, truth) == truth)
    Xr = np.concatenate([X, np.log(np.array([r["reader"] for r in recs]))[..., None]], 2)
    rows["transcript features + reader (5-fold CV)"] = (cv_pred(Xr, truth) == truth)
    Xs = np.concatenate([Xr, np.log(np.array([r["rep"] for r in recs]))[..., None]], 2)
    rows["transcript features + reader + self-report (5-fold CV)"] = (cv_pred(Xs, truth) == truth)
    if recs[0]["task"] == "qa":
        rows["task-relevant paragraph (by construction)"] = (np.array([r["j"] for r in recs]) == truth)
    maj = np.bincount(truth, minlength=4).argmax()
    rows[f"majority position (P{maj+1}, in-sample)"] = (truth == maj)
    out = {k: boot(v.astype(float)) for k, v in rows.items()}
    out["chance"] = (0.25, 0.25, 0.25)
    d = rows["self-report"].astype(float) - rows["transcript-only reader (same model)"].astype(float)
    out["self minus reader (paired)"] = boot(d)
    dis = selfp != readp
    out["_n"] = len(recs); out["_disagree_n"] = int(dis.sum())
    out["_disagree_self_right"] = int(((selfp == truth) & dis).sum()); out["_disagree_reader_right"] = int(((readp == truth) & dis).sum())
    out["_truth_dist"] = np.bincount(truth, minlength=4).tolist()
    out["_self_dist"] = np.bincount(selfp, minlength=4).tolist(); out["_reader_dist"] = np.bincount(readp, minlength=4).tolist()
    # per layer agreement
    pl = {}
    for L in range(NL):
        tL = np.array([np.array(r[key])[L].argmax() for r in recs])
        pl[L] = dict(self=float((selfp == tL).mean()), reader=float((readp == tL).mean()))
    out["_per_layer"] = pl
    md.append(f"\n### A: {label} (n={len(recs)}; truth = argmax per-token attention, {key}, layers {LAY[0]}–{LAY[-1]})\n")
    md.append("| predictor | agreement with most-attended paragraph [95% CI] |\n|---|---|")
    for k, v in out.items():
        if not k.startswith("_"): md.append(f"| {k} | {fmt(v)} |")
    md.append(f"\nWhere self and reader disagree (n={out['_disagree_n']}): self right {out['_disagree_self_right']}, reader right {out['_disagree_reader_right']}. "
              f"Truth distribution P1–P4 {out['_truth_dist']}; self {out['_self_dist']}; reader {out['_reader_dist']}.")
    return out

base = [r for r in A]
qa = [r for r in base if r["task"] == "qa"]; sm = [r for r in base if r["task"] == "sum"]; ar = [r for r in base if r["task"] == "arith"]
R["A"] = {}
for key in ["att_first", "att_mean"]:
    R["A"][key] = dict(qa=a_table(qa, key, "QA about one paragraph"), sum=a_table(sm, key, "summarize all four"),
                       arith=a_table(ar, key, "unrelated arithmetic (paragraphs irrelevant)"), all=a_table(base, key, "all three tasks"))

# ===================== A2 =====================
byid = {r["id"]: r for r in base}
def spec(d, r):
    others = [m for m in range(4) if m != r["k"] and (r["task"] != "qa" or m != r["j"])]
    return d[r["k"]] - np.mean([d[m] for m in others])
R["A2"] = {}
for task, b in sorted(set((r["task"], r["b"]) for r in A2)):
    rs = [r for r in A2 if r["b"] == b and r["task"] == task]
    o = {}
    k = np.array([r["k"] for r in rs]); bs = [byid[r["id"]] for r in rs]
    for key in ["att_first", "att_mean"]:
        lr = np.array([math.log(band(r[key])[r["k"]] / band(b0[key])[r["k"]]) for r, b0 in zip(rs, bs)])
        m, lo, hi = boot(lr); o[f"mass_ratio_{key}"] = (math.exp(m), math.exp(lo), math.exp(hi))
        o[f"k_is_argmax_boost_{key}"] = boot(np.array([band(r[key]).argmax() == r["k"] for r in rs], float))
        o[f"k_is_argmax_base_{key}"] = boot(np.array([band(r[key]).argmax() == r["k"] for r in bs], float))
    sb = np.array([np.argmax(r["rep"]) == r["k"] for r in rs], float); s0 = np.array([np.argmax(r["rep"]) == r["k"] for r in bs], float)
    rb = np.array([np.argmax(r["reader"]) == r["k"] for r in rs], float); r0 = np.array([np.argmax(r["reader"]) == r["k"] for r in bs], float)
    o["self_picks_k_boost"] = boot(sb); o["self_picks_k_base"] = boot(s0); o["self_delta"] = boot(sb - s0)
    o["reader_picks_k_boost"] = boot(rb); o["reader_picks_k_base"] = boot(r0); o["reader_delta"] = boot(rb - r0)
    o["self_minus_reader_delta"] = boot((sb - s0) - (rb - r0))
    ps = np.array([r["rep"][r["k"]] - b0["rep"][r["k"]] for r, b0 in zip(rs, bs)]); pr = np.array([r["reader"][r["k"]] - b0["reader"][r["k"]] for r, b0 in zip(rs, bs)])
    o["self_prob_k_delta"] = boot(ps); o["reader_prob_k_delta"] = boot(pr); o["self_minus_reader_prob_delta"] = boot(ps - pr)
    ls = np.array([spec(np.array(r["rep_lp"]) - np.array(b0["rep_lp"]), r) for r, b0 in zip(rs, bs)])
    lr_ = np.array([spec(np.array(r["reader_lp"]) - np.array(b0["reader_lp"]), r) for r, b0 in zip(rs, bs)])
    o["self_logp_k_specific_shift"] = boot(ls); o["reader_logp_k_specific_shift"] = boot(lr_); o["self_minus_reader_logp_shift"] = boot(ls - lr_)
    same = np.array([r["ans"].strip() == b0["ans"].strip() for r, b0 in zip(rs, bs)])
    o["answer_identical"] = boot(same.astype(float)); o["_n"] = len(rs); o["_n_identical"] = int(same.sum())
    if same.sum() >= 10:
        o["identical_answers_self_delta"] = boot((sb - s0)[same]); o["identical_answers_self_prob_k_delta"] = boot(ps[same])
        o["identical_answers_reader_delta"] = boot((rb - r0)[same])
        o["identical_answers_self_logp_shift"] = boot(ls[same]); o["identical_answers_reader_logp_shift"] = boot(lr_[same])
    ov = lambda r: len(words(r["ans"]) & words(paras_of(r["user"])[r["k"]])) / (len(words(r["ans"])) + 1)
    o["answer_overlap_with_Pk_delta"] = boot(np.array([ov(r) - ov(b0) for r, b0 in zip(rs, bs)]))
    corr = lambda r: float(r["gold"].lower() in r["ans"].lower()) if r["gold"] else float("nan")
    o["qa_correct_boost"] = boot(np.array([corr(r) for r in rs])); o["qa_correct_base"] = boot(np.array([corr(r) for r in bs]))
    R["A2"][f"{task}_b{b}"] = o
    md.append(f"\n### A2 ({task}): boost b={b} on a random non-target paragraph Pk (n={len(rs)}; layers {LAY[0]}–{LAY[-1]}, all heads, task+answer queries)\n")
    md.append("| quantity | value [95% CI] |\n|---|---|")
    md.append(f"| attention mass ratio on Pk, boost/base (first answer query) | {fmt(o['mass_ratio_att_first'], False)} |")
    md.append(f"| attention mass ratio on Pk, boost/base (mean over answer queries) | {fmt(o['mass_ratio_att_mean'], False)} |")
    md.append(f"| Pk is most-attended (first query): base → boost | {fmt(o['k_is_argmax_base_att_first'])} → {fmt(o['k_is_argmax_boost_att_first'])} |")
    md.append(f"| self-report picks Pk: base → boost | {fmt(o['self_picks_k_base'])} → {fmt(o['self_picks_k_boost'])} |")
    md.append(f"| transcript reader picks Pk: base → boost | {fmt(o['reader_picks_k_base'])} → {fmt(o['reader_picks_k_boost'])} |")
    md.append(f"| Δ self − Δ reader (difference-in-differences, picks) | {fmt(o['self_minus_reader_delta'])} |")
    md.append(f"| Δ P(Pk) self − Δ P(Pk) reader (probabilities) | {fmt(o['self_minus_reader_prob_delta'], False)} |")
    md.append(f"| Pk-specific shift in log P(report), self (Δlog P(Pk) − mean Δlog P(other non-target)) | {fmt(o['self_logp_k_specific_shift'], False)} |")
    md.append(f"| same, transcript reader | {fmt(o['reader_logp_k_specific_shift'], False)} |")
    md.append(f"| self − reader (log-prob shift) | {fmt(o['self_minus_reader_logp_shift'], False)} |")
    md.append(f"| answer text identical to unboosted | {fmt(o['answer_identical'])} ({o['_n_identical']}/{len(rs)}) |")
    if "identical_answers_self_delta" in o:
        md.append(f"| identical-answer subset: Δ self picks Pk | {fmt(o['identical_answers_self_delta'])} |")
        md.append(f"| identical-answer subset: Δ reader picks Pk (text identical, so 0 expected) | {fmt(o['identical_answers_reader_delta'])} |")
        md.append(f"| identical-answer subset: Δ self P(Pk) | {fmt(o['identical_answers_self_prob_k_delta'], False)} |")
        md.append(f"| identical-answer subset: Pk-specific log-prob shift, self (reader = 0 by construction) | {fmt(o['identical_answers_self_logp_shift'], False)} |")
    md.append(f"| text leak: Δ answer word-overlap with Pk | {fmt(o['answer_overlap_with_Pk_delta'], False)} |")
    md.append(f"| QA correct: base → boost | {fmt(o['qa_correct_base'])} → {fmt(o['qa_correct_boost'])} |")

# ===================== B =====================
R["B"] = {}
corr = lambda r: float(r["gold"].lower() in r["ans"].lower())
for task in ["qa", "arith"]:
    tq = [r for r in base if r["task"] == task]
    if not any(r["task"] == task for r in B): continue
    RB = R["B"][task] = {}
    md.append(f"\n### B ({task}): instructed attention to Pk (k ≠ question paragraph), n={len(tq)} contexts paired with no instruction\n")
    md.append("| condition | mass ratio on Pk vs none, first answer query [95% CI] | mass ratio on Pk vs none, mean over answer queries | mass ratio on question paragraph Pj (mean queries) | answer identical to none | correct |\n|---|---|---|---|---|---|")
    md.append(f"| none | 1 | 1 | 1 | 100% | {fmt(boot(np.array([corr(r) for r in tq])))} |")
    for cond in ["focus", "ignore", "mention"]:
        rs = [r for r in B if r["cond"] == cond and r["task"] == task]; bs = [byid[r["id"]] for r in rs]
        o = {}
        for key in ["att_first", "att_mean"]:
            lr = np.array([math.log(band(r[key])[r["k"]] / band(b0[key])[r["k"]]) for r, b0 in zip(rs, bs)])
            m, lo, hi = boot(lr); o[f"ratio_k_{key}"] = (math.exp(m), math.exp(lo), math.exp(hi)); o[f"_logratios_{key}"] = lr.tolist()
            lj = np.array([math.log(band(r[key])[r["j"]] / band(b0[key])[r["j"]]) for r, b0 in zip(rs, bs)])
            m, lo, hi = boot(lj); o[f"ratio_j_{key}"] = (math.exp(m), math.exp(lo), math.exp(hi))
            o[f"per_layer_logratio_k_{key}"] = [float(np.mean([math.log(np.array(r[key])[L][r["k"]] / np.array(b0[key])[L][r["k"]]) for r, b0 in zip(rs, bs)])) for L in range(NL)]
        same = np.array([r["ans"].strip() == b0["ans"].strip() for r, b0 in zip(rs, bs)], float)
        o["answer_identical"] = boot(same); o["correct"] = boot(np.array([corr(r) for r in rs]))
        RB[cond] = o
        md.append(f"| {cond} | {fmt(o['ratio_k_att_first'], False)} | {fmt(o['ratio_k_att_mean'], False)} | {fmt(o['ratio_j_att_mean'], False)} | {fmt(o['answer_identical'])} | {fmt(o['correct'])} |")
    for key in ["att_first", "att_mean"]:
        fx = {c: np.array(RB[c]["_logratios_" + key]) for c in ["focus", "ignore", "mention"]}
        for a, b in [("focus", "ignore"), ("ignore", "mention"), ("focus", "mention")]:
            m, lo, hi = boot(fx[a] - fx[b]); RB[f"{a}_vs_{b}_{key}"] = (math.exp(m), math.exp(lo), math.exp(hi))
            md.append(f"\n{a} vs {b} ({key}, paired): mass ratio {math.exp(m):.3f} [{math.exp(lo):.3f}, {math.exp(hi):.3f}]")
    md.append("\nPer-layer mean log mass ratio on Pk vs none (mean over answer queries), layer: focus / ignore / mention")
    md.append(" ".join(f"L{L}:{RB['focus']['per_layer_logratio_k_att_mean'][L]:+.2f}/{RB['ignore']['per_layer_logratio_k_att_mean'][L]:+.2f}/{RB['mention']['per_layer_logratio_k_att_mean'][L]:+.2f}" for L in range(NL)))

json.dump(R, open(os.path.join(D, "results.json"), "w"), indent=1)
open(os.path.join(D, "tables.md"), "w").write(f"# {meta['model']} (layers {NL}, band {LAY[0]}–{LAY[-1]}, B1={meta['B1']}, B2={meta['B2']})\n" + "\n".join(md) + "\n")
print("\n".join(md))
