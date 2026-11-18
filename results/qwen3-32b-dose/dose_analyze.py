import json, os, sys
import numpy as np
D = sys.argv[1]
meta = json.load(open(os.path.join(D, "meta.json"))); L = meta["LAYERS"]; NL = meta["NL"]
R = json.load(open(os.path.join(D, "dose.json")))
C = list(meta["conds"])
ids = sorted({r["id"] for r in R})
by = {(r["id"], r["cond"]): r for r in R}
def pk(r, which="att_mean", layers=L): return np.array(r[which])[layers].mean(0)[r["k"]]
M = {c: np.array([pk(by[i, c]) for i in ids]) for c in C}                 # band, mean over answer queries
F = {c: np.array([pk(by[i, c], "att_first") for i in ids]) for c in C}    # band, first answer query
PL = {c: np.array([[by[i, c]["att_mean"][l][by[i, c]["k"]] for l in range(NL)] for i in ids]) for c in C}  # [n, NL]
n = len(ids); rng = np.random.default_rng(0); NB = 10000; BI = rng.integers(0, n, (NB, n))
def pct(a, b): return 100 * (a / b - 1)
def ci(z): s = z[BI].mean(1); return [float(z.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
def row(a, b, X=M):
    z = pct(X[a], X[b]); return dict(mean=ci(z), above=float((z > 0).mean()), z=z.tolist())
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])
out = dict(n=n, layers=[L[0], L[-1]])
out["vs_mention"] = {c: row(c, "1_mention") for c in C if c != "1_mention"}
out["vs_none"] = {c: row(c, "0_none") for c in C if c != "0_none"}
out["vs_mention_first"] = {c: row(c, "1_mention", F) for c in C if c != "1_mention"}
UP = ["2_glance", "3_focus", "4_focus_strongly", "5_very_strongly_only", "6_twice"]
DN = ["7_little_less", "8_do_not", "9_completely_ignore"]
def ladder(names, sign):
    Z = np.stack([pct(M[c], M["1_mention"]) for c in names], 1)  # [n, k]
    means = Z.mean(0); order = np.arange(len(names))
    rho = spearman(order, means)
    bo = np.array([spearman(order, Z[b].mean(0)) for b in BI[:2000]])
    pc = np.array([spearman(order, Z[i]) for i in range(n)])  # per-context
    steps = {f"{a}->{b}": row(b, a) for a, b in zip(names[:-1], names[1:])}
    return dict(names=names, means=means.tolist(), spearman_of_means=rho, spearman_boot_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                share_boot_rho_eq_expected=float((bo == sign).mean()),
                per_context_spearman=ci(pc), steps=steps)
out["up"] = ladder(UP, 1.0); out["down"] = ladder(DN, -1.0)
out["down_with_mention"] = dict(step_mention_to_little_less=row("7_little_less", "1_mention"))
out["per_layer_vs_mention"] = {c: [float(pct(PL[c][:, l], PL["1_mention"][:, l]).mean()) for l in range(NL)] for c in C if c != "1_mention"}
out["free_answer_differs"] = {c: float(np.mean([not by[i, c]["free_same"] for i in ids])) for c in C}
out["free_correct"] = {c: float(np.mean([by[i, c]["ans_free"].strip() == by[i, c]["gold"] for i in ids])) for c in C}
out["forced_correct"] = float(np.mean([by[i, "0_none"]["ans_forced"].strip() == by[i, "0_none"]["gold"] for i in ids]))
json.dump(out, open(os.path.join(D, "dose_results.json"), "w"))
f = lambda t: f"{t[0]:+.1f}% [{t[1]:+.1f}, {t[2]:+.1f}]"
md = [f"n={n} contexts, band layers {L[0]}-{L[-1]}, mean over answer-token queries, teacher-forced on the no-instruction answer\n",
      "| condition | vs mention | share > 0 | vs none | share > 0 | vs mention, first query | free answer differs from none |", "|---|---|---|---|---|---|---|"]
for c in C:
    a = out["vs_mention"].get(c); b = out["vs_none"].get(c); fq = out["vs_mention_first"].get(c)
    md.append(f"| {c} | {f(a['mean']) if a else '—'} | {100*a['above']:.0f}% |" if a else f"| {c} | — | — |")
    md[-1] += (f" {f(b['mean'])} | {100*b['above']:.0f}% |" if b else " — | — |") + (f" {f(fq['mean'])} |" if fq else " — |") + f" {100*out['free_answer_differs'][c]:.1f}% |"
for nm in ["up", "down"]:
    d = out[nm]
    md.append(f"\n{nm}-ladder: means vs mention " + ", ".join(f"{c} {m:+.1f}%" for c, m in zip(d["names"], d["means"])))
    md.append(f"Spearman(order, mean) = {d['spearman_of_means']:+.2f}, bootstrap 95% CI [{d['spearman_boot_ci'][0]:+.2f}, {d['spearman_boot_ci'][1]:+.2f}], per-context Spearman mean {f(d['per_context_spearman'])[:-0]}")
    for s, v in d["steps"].items(): md.append(f"step {s}: {f(v['mean'])}, share > 0 {100*v['above']:.0f}%")
v = out["down_with_mention"]["step_mention_to_little_less"]; md.append(f"step 1_mention->7_little_less: {f(v['mean'])}")
md.append("\nper layer, % change vs mention (mean over contexts):\n| layer | " + " | ".join(c for c in C if c != "1_mention") + " |\n|---|" + "---|" * (len(C) - 1))
for l in range(0, NL, 4): md.append(f"| {l} | " + " | ".join(f"{out['per_layer_vs_mention'][c][l]:+.0f}" for c in C if c != "1_mention") + " |")
md.append(f"\nforced answer correct {100*out['forced_correct']:.1f}%; free answers correct: " + ", ".join(f"{c} {100*v:.1f}%" for c, v in out["free_correct"].items()))
open(os.path.join(D, "dose_tables.md"), "w").write("\n".join(md)); print("\n".join(md))
