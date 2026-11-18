# Analysis of rotate2.py: per instruction (focus, ignore), 4x4 of % change vs the mention-only line for the same named paragraph.
# Measure: answer-token queries (att_mean), layers 21-53 mean, per-token mean over the paragraph body, head mean.
import json, os, sys, numpy as np
D = sys.argv[1]
meta = json.load(open(os.path.join(D, "meta.json"))); L = meta["LAYERS"]
R = json.load(open(os.path.join(D, "R2.json"))); n = len(R)
rng = np.random.default_rng(0); NB = 10000; BI = rng.integers(0, n, (NB, n))
def band(att): return np.array(att)[L].mean(0)
A = {c: np.array([[band(r["cond"][f"{c}_{k}"]["att_mean"]) for k in range(4)] for r in R]) for c in ["mention", "focus", "ignore"]}  # [n,4 named,4 measured]
A0 = np.array([band(r["none"]["att_mean"]) for r in R])
out = dict(n=n, layers=[L[0], L[-1]])
for c, sign in [("focus", 1), ("ignore", -1)]:
    for ctrl, nm in [("mention", ""), ("none", "_vs_none")]:
        B = A["mention"] if ctrl == "mention" else np.repeat(A0[:, None, :], 4, 1)
        Z = 100 * (A[c] / B - 1)                       # [n,4,4]
        bs = Z[BI].mean(1)
        win = (np.argmax(sign * Z, 2) == np.arange(4)[None]).astype(float)   # [n,4]: named paragraph moved most in the instructed direction
        bw = win[BI].mean((1, 2))
        o = dict(mean=Z.mean(0).tolist(), lo=np.percentile(bs, 2.5, 0).tolist(), hi=np.percentile(bs, 97.5, 0).tolist(),
                 share_named_most=float(win.mean()), share_ci=[float(np.percentile(bw, 2.5)), float(np.percentile(bw, 97.5))],
                 share_by_named=win.mean(0).tolist(), contexts_4of4=int((win.sum(1) == 4).sum()), hist=np.bincount(win.sum(1).astype(int), minlength=5).tolist(),
                 diag_share_right_sign=[float((sign * Z[:, k, k] > 0).mean()) for k in range(4)])
        out[c + nm] = o
same = {c: float(np.mean([[r["cond"][f"{c}_{k}"]["free_same"] for k in range(4)] for r in R])) for c in ["mention", "focus", "ignore"]}
out["free_answer_same_as_none"] = same
out["correct_none"] = float(np.mean([r["ans_none"].strip() == r["gold"] for r in R]))
json.dump(out, open(os.path.join(D, "rotate2_results.json"), "w"), indent=1)
f = lambda v: f"{v:+.1f}".replace("-", "−")
md = [f"# Rotation with focus and completely-ignore (Qwen3-32B, {n} arithmetic contexts, layers {L[0]}–{L[-1]}, answer-token queries, teacher-forced on the no-instruction answer)\n"]
for key in ["focus", "ignore", "focus_vs_none", "ignore_vs_none"]:
    o = out[key]; md.append(f"## {key}: % change (rows named, columns measured), mean over contexts [95% bootstrap CI]\n\n| named \\ measured | P1 | P2 | P3 | P4 |\n|---|---|---|---|---|")
    for k in range(4): md.append(f"| P{k+1} | " + " | ".join(("**" if k == m else "") + f"{f(o['mean'][k][m])} [{f(o['lo'][k][m])}, {f(o['hi'][k][m])}]" + ("**" if k == m else "") for m in range(4)) + " |")
    md.append(f"\nShare of (context, named paragraph) pairs where the named paragraph moved most in the instructed direction: {100*o['share_named_most']:.1f}% [{100*o['share_ci'][0]:.1f}, {100*o['share_ci'][1]:.1f}] (chance 25%); by named P1..P4 " + ", ".join(f"{100*v:.0f}%" for v in o["share_by_named"]) + f"; contexts with 0..4 of 4: {o['hist']}; diagonal in instructed direction: " + ", ".join(f"{100*v:.0f}%" for v in o["diag_share_right_sign"]) + "\n")
md.append(f"Free (greedy) answers identical to the no-instruction answer: " + ", ".join(f"{c} {100*v:.1f}%" for c, v in same.items()) + f"; no-instruction answer correct {100*out['correct_none']:.0f}%.")
open(os.path.join(D, "rotate2_tables.md"), "w").write("\n".join(md)); print("\n".join(md))
