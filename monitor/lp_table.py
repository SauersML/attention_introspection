# Markdown tables for results/monitor_shuffled/LOGPROB.md from logprob_signal.py outputs.
# usage: lp_table.py MAIN/logprob.json [LABEL=OTHER/logprob.json ...]
import json, sys
J = json.load(open(sys.argv[1]))["res"]
ROWS = ["after", "before", "later", "think", "fb4", "fb16", "after-least", "pooled"]
NAME = {"after": "right after answering", "before": "before answering", "later": "one turn later", "think": "thinking on",
        "fb4": "4 feedback examples", "fb16": "16 feedback examples", "after-least": "right after, 'least' (minus log P)", "pooled": "pooled (first six)"}
def ci(d, f="%.3f"): return (f + " [" + f + ", " + f + "]") % (d["mean"], d["lo"], d["hi"])
def p(d): return "%.4f" % d["p"] if "p" in d else ""
print("### (1, 2) Within-prompt rank correlation with true attention (mean over prompts, 95% bootstrap CI)\n")
print("| setup | n | self-report | transcript reader | self − reader | text classifier | self vs reader log-probs | shuffled-self prior |")
print("|---|---|---|---|---|---|---|---|")
for t in ROWS:
    if t not in J: continue
    r = J[t]
    print(f"| {NAME[t]} | {r['n']} | {ci(r['self'])} | {ci(r['reader'])} | {ci(r['self_minus_reader'])} | "
          f"{ci(r['clf']) if 'clf' in r else ''} | {('%.2f' % r['self_vs_reader']['mean']) if 'self_vs_reader' in r else ''} | "
          f"{('%.3f' % r['null_self_prior']['mean']) if 'null_self_prior' in r else ''} |")
print("\n### (3) Held-out gain from adding the self-report's log-probs (10-fold CV grouped by prompt)\n")
print("Base model: reader log-probs + paragraph position + log length, prompt fixed effects. Null: self log-probs permuted across prompts within each paragraph slot (and family).\n")
print("| setup | base rank corr | gain in rank corr | null 95% range | p | base R² | gain in R² | null 97.5% | p (R²) | gain in rank corr, base + transcript word overlap | gain in R², same |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for t in ROWS:
    if t not in J: continue
    r = J[t]; nr, n2 = r["null_gain_rho"], r["null_gain_r2"]
    br = ci(r["base_rho"]) if "base_rho" in r else ""
    b2 = ("%.3f" % r["base_r2"]) if "base_r2" in r else ""
    print(f"| {NAME[t]} | {br} | {ci(r['gain_rho'])} | {nr['lo']:.3f} to {nr['hi']:.3f} | {p(nr)} | {b2} | {ci(r['gain_r2'], '%.4f')} | "
          f"{n2['hi']:.4f} | {p(n2)} | {ci(r['gain_rho_text']) if 'gain_rho_text' in r else ''} | {ci(r['gain_r2_text'], '%.4f') if 'gain_r2_text' in r else ''} |")
print("\n### Per family (mean within-prompt rank correlation self / reader, and gain in rank corr)\n")
fams = ["distractor", "multihop", "arith", "sumone", "longshort"]
print("| setup | " + " | ".join(fams) + " |"); print("|---" * (len(fams) + 1) + "|")
for t in ROWS:
    if t not in J or "fam" not in J[t]: continue
    cells = []
    for f in fams:
        d = J[t]["fam"][f]
        cells.append("" if d["n"] == 0 else "%.2f / %.2f, %+.3f (n=%d)" % (d["self"], d["reader"], d["gain_rho"], d["n"]))
    print(f"| {NAME[t]} | " + " | ".join(cells) + " |")
for a in sys.argv[2:]:
    lab, path = a.split("=", 1); K = json.load(open(path))["res"]
    print(f"\n### Sensitivity: {lab}\n")
    print("| setup | self | reader | gain in rank corr | p | gain in R² | p (R²) |"); print("|---|---|---|---|---|---|---|")
    for t in ROWS:
        if t not in K: continue
        r = K[t]
        print(f"| {NAME[t]} | {r['self']['mean']:.3f} | {r['reader']['mean']:.3f} | {ci(r['gain_rho'])} | {p(r['null_gain_rho'])} | {ci(r['gain_r2'], '%.4f')} | {p(r['null_gain_r2'])} |")
