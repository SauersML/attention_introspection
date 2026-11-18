# Instruction before the paragraphs (b_*) vs after (a_*, rerun here, and the dose run's 3_focus / 9_completely_ignore), 150 dose contexts.
import json, os, sys, numpy as np
D = sys.argv[1]; DOSE = sys.argv[2]
meta = json.load(open(os.path.join(D, "meta.json"))); L = meta["LAYERS"]
R = json.load(open(os.path.join(D, "dose.json"))); by = {(r["id"], r["cond"]): r for r in R}; ids = sorted({r["id"] for r in R})
Rd = json.load(open(os.path.join(DOSE, "dose.json"))); bd = {(r["id"], r["cond"]): r for r in Rd}
def pk(r): return np.array(r["att_mean"])[L].mean(0)[r["k"]]
M = {c: np.array([pk(by[i, c]) for i in ids]) for c in meta["conds"]}
Md = {c: np.array([pk(bd[i, c]) for i in ids]) for c in ["0_none", "1_mention", "3_focus", "9_completely_ignore"]}
assert all(by[i, "0_none"]["k"] == bd[i, "0_none"]["k"] and by[i, "0_none"]["ans_forced"] == bd[i, "0_none"]["ans_forced"] for i in ids)
rep = {"none": float(np.max(np.abs(M["0_none"] / Md["0_none"] - 1))), "a_focus": float(np.max(np.abs(M["a_focus"] / Md["3_focus"] - 1))),
       "a_ignore": float(np.max(np.abs(M["a_ignore"] / Md["9_completely_ignore"] - 1))), "a_mention": float(np.max(np.abs(M["a_mention"] / Md["1_mention"] - 1)))}
n = len(ids); rng = np.random.default_rng(0); BI = rng.integers(0, n, (10000, n))
def signflip(z, NF=200000):
    r = np.random.default_rng(0); obs = abs(z.mean()); h = 0
    for _ in range(NF // 20000): s = r.choice([-1.0, 1.0], (20000, len(z))); h += int((np.abs((s * z).mean(1)) >= obs).sum())
    return (h + 1) / (NF + 1)
def row(z): s = z[BI].mean(1); return dict(mean=float(z.mean()), lo=float(np.percentile(s, 2.5)), hi=float(np.percentile(s, 97.5)), above=float((z > 0).mean()), p=signflip(z))
out = dict(n=n, layers=[L[0], L[-1]], replication_max_rel_diff_vs_dose_run=rep)
P = lambda a, b: 100 * (a / b - 1)
for c in ["focus", "ignore"]:
    out[c] = dict(before_vs_before_mention=row(P(M["b_" + c], M["b_mention"])), before_vs_none=row(P(M["b_" + c], M["0_none"])),
                  after_vs_after_mention=row(P(M["a_" + c], M["a_mention"])), after_vs_none=row(P(M["a_" + c], M["0_none"])),
                  before_vs_after_same_instruction=row(P(M["b_" + c], M["a_" + c])),
                  diff_of_effects_vs_own_mention_pts=row(P(M["b_" + c], M["b_mention"]) - P(M["a_" + c], M["a_mention"])))
out["before_mention_vs_none"] = row(P(M["b_mention"], M["0_none"])); out["after_mention_vs_none"] = row(P(M["a_mention"], M["0_none"]))
out["free_answer_same_as_none"] = {c: float(np.mean([by[i, c]["free_same"] for i in ids])) for c in ["b_mention", "b_focus", "b_ignore"]}
json.dump(out, open(os.path.join(D, "before_results.json"), "w"), indent=1)
f = lambda d: f"{d['mean']:+.1f}% [{d['lo']:+.1f}, {d['hi']:+.1f}], {100*d['above']:.0f}% of contexts > 0, sign-flip p {d['p']:.1g}"
md = [f"# Instruction once before the paragraphs vs after them (Qwen3-32B, {n} dose contexts, seed 7, layers {L[0]}–{L[-1]}, answer-token queries, teacher-forced on the no-instruction answer)\n",
      f"After-position versions rerun on the same pod; max relative difference from the dose run: " + ", ".join(f"{k} {v:.1e}" for k, v in rep.items()) + "\n"]
for c in ["focus", "ignore"]:
    md.append(f"## {c}")
    for k, v in out[c].items(): md.append(f"- {k}: {f(v)}")
md.append(f"\n- before mention vs none: {f(out['before_mention_vs_none'])}\n- after mention vs none: {f(out['after_mention_vs_none'])}")
md.append("- free answers identical to no-instruction: " + ", ".join(f"{k} {100*v:.1f}%" for k, v in out["free_answer_same_as_none"].items()))
open(os.path.join(D, "before_tables.md"), "w").write("\n".join(md)); print("\n".join(md))
