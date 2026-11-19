# In-order report options vs shuffled options, attention-monitoring experiment: the numbers the README and
# monitor_logprob.png show, both runs side by side, plus the paired difference (same prompt; in-order minus shuffled).
# Per-prompt values come from each run's logprob_results.json (monitor/logprob.py), primary truth (attention per token,
# answer queries, layers 21-53). Top-1 (does the report's most likely paragraph match the most-attended one) is
# recomputed from the raw shards for the "which most?" wordings. 95% bootstrap CI over prompts (10,000 resamples);
# p two-sided sign-flip (200,000).
# usage: python monitor/compare_orders.py   (writes results/monitor_inorder/compare_shuffled.{json,md})
import json, os, numpy as np
RUNS = {"inorder": "results/monitor_inorder", "shuffled": "results/monitor_shuffled"}
J = {k: json.load(open(f"{d}/logprob_results.json")) for k, d in RUNS.items()}
assert J["inorder"]["ids"] == J["shuffled"]["ids"] and len(J["inorder"]["ids"]) == 300
IDS = J["inorder"]["ids"]
def load_hf(d):
    H = {}
    for s in sorted(os.listdir(f"{d}/data")):
        H.update(json.load(open(f"{d}/data/{s}/hf1.json")))
        for i, r in json.load(open(f"{d}/data/{s}/hf2.json")).items(): H[i]["rep"].update(r)
    return H
HF = {k: load_hf(d) for k, d in RUNS.items()}
def top1(H, s, who):
    out = []
    for i in IDS:
        t = np.array((H[i]["truth_think"]["final"] if s.startswith("think") else H[i]["truth"])["norm"], float)[21:54].mean(0)
        out.append(float(np.argmax(H[i]["rep"][f"{s}|{who}"]) == np.argmax(t)))
    return np.array(out)
def boot(x, seed=0):
    x = np.asarray(x, float); m = x[np.random.default_rng(seed).integers(0, len(x), (10000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def signflip(x, NF=200000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); obs = abs(x.mean()); h = 0
    if obs == 0: return 1.0
    for _ in range(NF // 20000): h += int((np.abs((rng.choice([-1.0, 1.0], (20000, len(x))) * x).mean(1)) >= obs - 1e-12).sum())
    return (h + 1) / (NF + 1)
NAME = {"after|most": "which most, right after", "after|least": "which least, right after", "after|rate": "0–10 rating, right after",
        "after|free": "free answer, right after", "before|most": "which most, before answering", "before|rate": "0–10 rating, before answering",
        "later|most": "which most, one turn later", "later|rate": "0–10 rating, one turn later", "think|most": "which most, with thinking",
        "fb4|most": "which most, after 4 true examples", "fb16|most": "which most, after 16 true examples",
        "fb4|rate": "0–10 rating, after 4 true examples", "fb16|rate": "0–10 rating, after 16 true examples"}
fmt = lambda b, d=3: f"{b[0]:+.{d}f} [{b[1]:+.{d}f}, {b[2]:+.{d}f}]"
fp = lambda p: "<0.0001" if p < 1e-4 else f"{p:.2g}"
out, L = [], []
def row(measure, s, a, b, d=3, scale=1.0):
    a, b = np.asarray(a) * scale, np.asarray(b) * scale
    bi, bs, bd = boot(a), boot(b), boot(a - b); pd = signflip(a - b)
    out.append(dict(measure=measure, setup=s, n=len(a), inorder=bi, shuffled=bs, diff=bd, diff_p=pd))
    L.append(f"| {NAME[s]} | {fmt(bi, d)} | {fmt(bs, d)} | {fmt(bd, d)} | {fp(pd)} |")
HEAD = ["| way of asking | in order | shuffled | in order − shuffled (paired) | p (paired) |", "|---|---|---|---|---|"]
P = {k: J[k]["perprompt"]["norm"] for k in RUNS}
PM = {k: J[k]["perprompt"]["mass"] for k in RUNS}
# preregistered tables (analyze_monitor.py), primary truth, both runs side by side (not paired: those are in the tables below)
def prereg(d):
    rows, n = [], None
    for ln in open(f"{d}/prereg_tables.md"):
        if ln.startswith("## Secondary"): break
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) == 14 and c[2] == "300": rows.append(c)
        if ln.startswith("Setups passing"): n = ln.strip()
    return rows, n
PR = {k: prereg(d) for k, d in RUNS.items()}
L += ["### Preregistered pass rule (primary truth; `prereg_tables.md` of each run)", "",
      f"- in order: {PR['inorder'][1]}", f"- shuffled: {PR['shuffled'][1]}", "",
      "| setup | wording | self top-1 (in order / shuffled) | reader top-1 | classifier top-1 | self rank corr. | reader rank corr. | self − reader rank corr. | pass |",
      "|---|---|---|---|---|---|---|---|---|"]
for a, b in zip(*[PR[k][0] for k in RUNS]):
    assert a[:2] == b[:2]
    g = lambda i: f"{a[i].split(' [')[0]} / {b[i].split(' [')[0]}"
    L.append(f"| {a[0]} | {a[1]} | {g(3)} | {g(4)} | {g(5)} | {g(8)} | {g(9)} | {a[11].split(' (')[0]} / {b[11].split(' (')[0]} | {a[13]} / {b[13]} |")
for measure, key, title in [("self", "self", "Self-report: rank correlation with true attention"),
                            ("reader", "reader", "Same model reading the transcript: rank correlation with true attention"),
                            ("self_minus_reader", None, "Self-report minus reader (rank correlation)"),
                            ("gain_all", "gain_all", "Rank correlation the self-report adds to a reader asked every way at that time (held out)")]:
    L += ["", f"### {title}", ""] + HEAD
    for s in NAME:
        if key: a, b = P["inorder"][s][key], P["shuffled"][s][key]
        else: a = np.subtract(P["inorder"][s]["self"], P["inorder"][s]["reader"]); b = np.subtract(P["shuffled"][s]["self"], P["shuffled"][s]["reader"])
        row(measure, s, a, b)
L += ["", "### Secondary truth (total attention mass, post hoc): rank correlation the self-report adds to a reader asked every way (held out)", ""] + HEAD
for s in NAME: row("gain_all_mass", s, PM["inorder"][s]["gain_all"], PM["shuffled"][s]["gain_all"])
for who in ["self", "reader"]:
    L += ["", f"### Top-1, {'self-report' if who == 'self' else 'reader'}: most likely paragraph is the most attended (percentage points)", ""] + HEAD
    for s in [k for k in NAME if k.endswith("|most")]:
        row(f"top1_{who}", s, top1(HF["inorder"], s, who), top1(HF["shuffled"], s, who), d=1, scale=100)
json.dump(out, open("results/monitor_inorder/compare_shuffled.json", "w"), indent=1)
open("results/monitor_inorder/compare_shuffled.md", "w").write(
    "# Monitor experiment: report options in order vs shuffled, primary truth, 300 prompts\n\n"
    "95% bootstrap CI over prompts; paired difference on the same prompt; p two-sided sign-flip. From `monitor/compare_orders.py`.\n"
    + "\n".join(L) + "\n")
print("\n".join(L))
