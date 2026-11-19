# Scores the two-turn told follow-up: python3 told_followup/score.py results/told_followup/data
import json, sys, re, glob, os, math
import numpy as np
D = [10.0, 20.0]
R = sorted([json.loads(l) for f in sorted(glob.glob(os.path.join(sys.argv[1], "recs_*.jsonl"))) for l in open(f)], key=lambda r: r["id"])
rng = np.random.default_rng(0); B = 10000
yes = lambda t: t["text"].strip().startswith("Yes")
def label(t):
    m = re.search(r"\[?P([1-4])\]?", t.get("text2", "")); return int(m.group(1)) - 1 if m else None
def signflip(x, n=20000):
    x = np.asarray(x, float); s = rng.choice([-1, 1], (n, len(x)))
    return float((1 + (np.abs((s * x).mean(1)) >= abs(x.mean())).sum()) / (n + 1))
idx = rng.integers(0, len(R), (B, len(R)))
none_y = np.array([yes(r["none"]) for r in R], float)
out = dict(n_prompts=len(R), none_yes_rate=float(none_y.mean()), none_yes_n=int(none_y.sum()))
ny = [label(r["none"]) for r in R if yes(r["none"])]
out["none_turn2_labels"] = {f"P{k+1}": ny.count(k) for k in range(4)} | {"unparsed": ny.count(None)}
out["dose"] = {}
for m in D:
    T = lambda r, k: r["boost"][f"{k}|{m}"]
    by = np.array([np.mean([yes(T(r, k)) for k in range(4)]) for r in R])
    d = by - none_y
    rel = by[idx].mean(1) / none_y[idx].mean(1) - 1
    # turn-2: per prompt, (#yes trials naming the boosted paragraph, #yes trials)
    hit = np.array([sum(label(T(r, k)) == k for k in range(4) if yes(T(r, k))) for r in R], float)
    ny_ = np.array([sum(yes(T(r, k)) for k in range(4)) for r in R], float)
    share = hit.sum() / ny_.sum() if ny_.sum() else float("nan")
    bs = hit[idx].sum(1) / np.maximum(ny_[idx].sum(1), 1)
    pb = [math.exp(T(r, k)["lp_par"][k]) for r in R for k in range(4) if yes(T(r, k))]
    labs = [label(T(r, k)) for r in R for k in range(4) if yes(T(r, k))]
    # position-matched chance: share of control turn-2 labels at each boosted position, weighted by where the yes trials are
    pos = [k for r in R for k in range(4) if yes(T(r, k))]
    mr = [np.mean([T(r, k)["mass"][k] / r["none"]["mass"][k] for k in range(4)]) for r in R]
    out["dose"][f"{m:g}"] = dict(
        yes_rate=float(by.mean()), yes_n=int(ny_.sum()), n_trials=4 * len(R),
        yes_minus_none=float(d.mean()), yes_minus_none_ci=[float(np.percentile(by[idx].mean(1) - none_y[idx].mean(1), q)) for q in (2.5, 97.5)],
        rel_change=float(by.mean() / none_y.mean() - 1), rel_change_ci=[float(np.percentile(rel, q)) for q in (2.5, 97.5)],
        signflip_p=signflip(d),
        yes_rate_by_position=[float(np.mean([yes(T(r, k)) for r in R])) for k in range(4)],
        turn2_correct_share=float(share), turn2_correct_ci=[float(np.percentile(bs, q)) for q in (2.5, 97.5)], turn2_n=int(ny_.sum()),
        turn2_correct_n=int(hit.sum()), turn2_unparsed=int(sum(l is None for l in labs)),
        turn2_label_counts={f"P{j+1}": labs.count(j) for j in range(4)},
        boosted_position_counts={f"P{j+1}": pos.count(j) for j in range(4)},
        turn2_p_boosted_label_mean=float(np.mean(pb)) if pb else None,
        attn_mass_ratio=float(np.mean(mr)))
out["turn2_texts"] = sorted(set(t.get("text2") for r in R for t in [r["none"], *r["boost"].values()] if "text2" in t))
out["turn1_texts"] = {x: sum(t["text"] == x for r in R for t in [r["none"], *r["boost"].values()]) for x in set(t["text"] for r in R for t in [r["none"], *r["boost"].values()])}
json.dump(out, open(os.path.join(sys.argv[1], "..", "scores.json"), "w"), indent=1)
print(json.dumps(out, indent=1)[:5000])
